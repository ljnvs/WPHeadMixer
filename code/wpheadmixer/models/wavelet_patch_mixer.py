import torch
import torch.nn as nn

from kan import KAN
from models.decomposition import Decomposition
from utils.RevIN import RevIN


def _count_parameters(module):
    return sum(parameter.numel() for parameter in module.parameters())


def _kan_layer_param_count(in_features, out_features, grid_size, spline_order,
                           enable_standalone_scale_spline=True):
    total = out_features * in_features
    total += out_features * in_features * (grid_size + spline_order)
    if enable_standalone_scale_spline:
        total += out_features * in_features
    return total


def _solve_width(param_budget, coefficient, bias_term):
    if coefficient <= 0:
        return 1
    usable_budget = max(param_budget - bias_term, coefficient)
    return max(1, int(round(usable_budget / coefficient)))


class LinearForecastHead(nn.Module):
    def __init__(self, input_dim, output_dim, param_budget=None):
        super().__init__()
        direct_budget = input_dim * output_dim + output_dim
        if param_budget is None or param_budget >= direct_budget:
            self.layers = nn.Sequential(nn.Linear(input_dim, output_dim))
        else:
            rank = _solve_width(param_budget, input_dim + output_dim + 1, output_dim)
            self.layers = nn.Sequential(
                nn.Linear(input_dim, rank),
                nn.Linear(rank, output_dim),
            )

    def forward(self, x):
        return self.layers(x)

    def regularization_loss(self, *args, **kwargs):
        device = next(self.parameters()).device
        return torch.zeros((), device=device)


class MLPForecastHead(nn.Module):
    def __init__(self, input_dim, output_dim, dropout, hidden_dim=None, param_budget=None):
        super().__init__()
        if hidden_dim is None:
            target_budget = param_budget if param_budget is not None else input_dim * output_dim + output_dim
            hidden_dim = _solve_width(target_budget, input_dim + output_dim + 1, output_dim)
        self.hidden_dim = max(1, hidden_dim)
        self.layers = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.layers(x)

    def regularization_loss(self, *args, **kwargs):
        device = next(self.parameters()).device
        return torch.zeros((), device=device)


class KANForecastHead(nn.Module):
    def __init__(self, input_dim, output_dim, dropout, grid_size=5, spline_order=3,
                 param_budget=None, adapter_dim=None):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.input_adapter = None

        if adapter_dim is None:
            if param_budget is None:
                adapter_dim = input_dim
            else:
                coefficient = input_dim + 1 + output_dim * (grid_size + spline_order + 2)
                adapter_dim = _solve_width(param_budget, coefficient, 0)

        self.adapter_dim = max(1, min(input_dim, adapter_dim))
        if self.adapter_dim != input_dim:
            self.input_adapter = nn.Linear(input_dim, self.adapter_dim)

        self.head = KAN([
            self.adapter_dim,
            output_dim,
        ], grid_size=grid_size, spline_order=spline_order)

    def forward(self, x):
        x = self.dropout(x)
        if self.input_adapter is not None:
            x = self.input_adapter(x)
        return self.head(x)

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        return self.head.regularization_loss(regularize_activation, regularize_entropy)


class HybridLinearKANForecastHead(nn.Module):
    def __init__(self, input_dim, output_dim, dropout, grid_size=5, spline_order=3,
                 param_budget=None, linear_budget_ratio=0.5, adapter_dim=None):
        super().__init__()
        if param_budget is None:
            linear_budget = input_dim * output_dim + output_dim
            kan_budget = input_dim * output_dim + output_dim
        else:
            linear_budget = max(1, int(round(param_budget * linear_budget_ratio)))
            kan_budget = max(1, param_budget - linear_budget)

        self.linear_head = LinearForecastHead(input_dim, output_dim, param_budget=linear_budget)
        self.kan_head = KANForecastHead(
            input_dim,
            output_dim,
            dropout=dropout,
            grid_size=grid_size,
            spline_order=spline_order,
            param_budget=kan_budget,
            adapter_dim=adapter_dim,
        )
        self.residual_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        linear_out = self.linear_head(x)
        residual_out = self.kan_head(x)
        return linear_out + self.residual_scale * residual_out

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        return self.kan_head.regularization_loss(regularize_activation, regularize_entropy)


def build_forecast_head(head_type, input_dim, output_dim, dropout, match_head_params=False,
                         head_param_budget=None, mlp_hidden_dim=None, kan_grid_size=5,
                         kan_spline_order=3, hybrid_linear_ratio=0.5):
    budget = head_param_budget
    if match_head_params and budget is None:
        budget = input_dim * output_dim + output_dim

    if head_type == 'linear':
        head = LinearForecastHead(input_dim, output_dim, param_budget=budget)
    elif head_type == 'mlp':
        head = MLPForecastHead(
            input_dim,
            output_dim,
            dropout=dropout,
            hidden_dim=mlp_hidden_dim,
            param_budget=budget,
        )
    elif head_type == 'kan':
        head = KANForecastHead(
            input_dim,
            output_dim,
            dropout=dropout,
            grid_size=kan_grid_size,
            spline_order=kan_spline_order,
            param_budget=budget,
        )
    elif head_type == 'hybrid':
        head = HybridLinearKANForecastHead(
            input_dim,
            output_dim,
            dropout=dropout,
            grid_size=kan_grid_size,
            spline_order=kan_spline_order,
            param_budget=budget,
            linear_budget_ratio=hybrid_linear_ratio,
        )
    else:
        raise ValueError(f'Unsupported head_type: {head_type}')

    return head


class WPMixerCore(nn.Module):
    def __init__(self,
                 input_length=[],
                 pred_length=[],
                 wavelet_name=[],
                 level=[],
                 batch_size=[],
                 channel=[],
                 d_model=[],
                 dropout=[],
                 embedding_dropout=[],
                 tfactor=[],
                 dfactor=[],
                 device=[],
                 patch_len=[],
                 patch_stride=[],
                 no_decomposition=[],
                 use_amp=[],
                 head_type='linear',
                 match_head_params=False,
                 head_param_budget=None,
                 mlp_hidden_dim=None,
                 kan_grid_size=5,
                 kan_spline_order=3,
                 hybrid_linear_ratio=0.5):

        super(WPMixerCore, self).__init__()
        self.input_length = input_length
        self.pred_length = pred_length
        self.wavelet_name = wavelet_name
        self.level = level
        self.batch_size = batch_size
        self.channel = channel
        self.d_model = d_model
        self.dropout = dropout
        self.embedding_dropout = embedding_dropout
        self.device = device
        self.no_decomposition = no_decomposition
        self.tfactor = tfactor
        self.dfactor = dfactor
        self.use_amp = use_amp
        self.head_type = head_type

        self.Decomposition_model = Decomposition(
            input_length=self.input_length,
            pred_length=self.pred_length,
            wavelet_name=self.wavelet_name,
            level=self.level,
            batch_size=self.batch_size,
            channel=self.channel,
            d_model=self.d_model,
            tfactor=self.tfactor,
            dfactor=self.dfactor,
            device=self.device,
            no_decomposition=self.no_decomposition,
            use_amp=self.use_amp,
        )

        self.input_w_dim = self.Decomposition_model.input_w_dim
        self.pred_w_dim = self.Decomposition_model.pred_w_dim
        self.patch_len = patch_len
        self.patch_stride = patch_stride

        self.resolutionBranch = nn.ModuleList([
            ResolutionBranch(
                input_seq=self.input_w_dim[i],
                pred_seq=self.pred_w_dim[i],
                batch_size=self.batch_size,
                channel=self.channel,
                d_model=self.d_model,
                dropout=self.dropout,
                embedding_dropout=self.embedding_dropout,
                tfactor=self.tfactor,
                dfactor=self.dfactor,
                patch_len=self.patch_len,
                patch_stride=self.patch_stride,
                head_type=head_type,
                match_head_params=match_head_params,
                head_param_budget=head_param_budget,
                mlp_hidden_dim=mlp_hidden_dim,
                kan_grid_size=kan_grid_size,
                kan_spline_order=kan_spline_order,
                hybrid_linear_ratio=hybrid_linear_ratio,
            )
            for i in range(len(self.input_w_dim))
        ])

        self.revin = RevIN(self.channel, eps=1e-5, affine=True, subtract_last=False)

    def forward(self, xL, return_intermediates=False):
        x = self.revin(xL, 'norm')
        x = x.transpose(1, 2)

        xA, xD = self.Decomposition_model.transform(x)

        branch_details = []
        if return_intermediates:
            yA, branch_detail = self.resolutionBranch[0](xA, return_intermediates=True)
            branch_details.append(branch_detail)
        else:
            yA = self.resolutionBranch[0](xA)

        yD = []
        for i in range(len(xD)):
            if return_intermediates:
                yD_i, branch_detail = self.resolutionBranch[i + 1](xD[i], return_intermediates=True)
                branch_details.append(branch_detail)
            else:
                yD_i = self.resolutionBranch[i + 1](xD[i])
            yD.append(yD_i)

        y = self.Decomposition_model.inv_transform(yA, yD)
        y = y.transpose(1, 2)
        y = y[:, -self.pred_length:, :]
        xT = self.revin(y, 'denorm')

        if not return_intermediates:
            return xT

        return xT, {
            'approximation_input': xA,
            'detail_inputs': xD,
            'approximation_output': yA,
            'detail_outputs': yD,
            'branch_details': branch_details,
        }

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        device = next(self.parameters()).device
        loss = torch.zeros((), device=device)
        for branch in self.resolutionBranch:
            loss = loss + branch.regularization_loss(
                regularize_activation=regularize_activation,
                regularize_entropy=regularize_entropy,
            )
        return loss

    def head_parameter_summary(self):
        return [branch.head_parameter_summary() for branch in self.resolutionBranch]


class ResolutionBranch(nn.Module):
    def __init__(self,
                 input_seq=[],
                 pred_seq=[],
                 batch_size=[],
                 channel=[],
                 d_model=[],
                 dropout=[],
                 embedding_dropout=[],
                 tfactor=[],
                 dfactor=[],
                 patch_len=[],
                 patch_stride=[],
                 head_type='linear',
                 match_head_params=False,
                 head_param_budget=None,
                 mlp_hidden_dim=None,
                 kan_grid_size=5,
                 kan_spline_order=3,
                 hybrid_linear_ratio=0.5):
        super(ResolutionBranch, self).__init__()
        self.input_seq = input_seq
        self.pred_seq = pred_seq
        self.batch_size = batch_size
        self.channel = channel
        self.d_model = d_model
        self.dropout = dropout
        self.embedding_dropout = embedding_dropout
        self.tfactor = tfactor
        self.dfactor = dfactor
        self.patch_len = patch_len
        self.patch_stride = patch_stride
        self.patch_num = int((self.input_seq - self.patch_len) / self.patch_stride + 2)
        self.head_type = head_type

        self.patch_norm = nn.BatchNorm2d(self.channel)
        self.patch_embedding_layer = nn.Linear(self.patch_len, self.d_model)
        self.mixer1 = Mixer(
            input_seq=self.patch_num,
            out_seq=self.patch_num,
            batch_size=self.batch_size,
            channel=self.channel,
            d_model=self.d_model,
            dropout=self.dropout,
            tfactor=self.tfactor,
            dfactor=self.dfactor,
        )
        self.mixer2 = Mixer(
            input_seq=self.patch_num,
            out_seq=self.patch_num,
            batch_size=self.batch_size,
            channel=self.channel,
            d_model=self.d_model,
            dropout=self.dropout,
            tfactor=self.tfactor,
            dfactor=self.dfactor,
        )
        self.norm = nn.BatchNorm2d(self.channel)
        self.dropoutLayer = nn.Dropout(self.embedding_dropout)
        self.flatten = nn.Flatten(start_dim=-2, end_dim=-1)

        head_input_dim = self.patch_num * self.d_model
        self.head = build_forecast_head(
            head_type=head_type,
            input_dim=head_input_dim,
            output_dim=self.pred_seq,
            dropout=self.embedding_dropout,
            match_head_params=match_head_params,
            head_param_budget=head_param_budget,
            mlp_hidden_dim=mlp_hidden_dim,
            kan_grid_size=kan_grid_size,
            kan_spline_order=kan_spline_order,
            hybrid_linear_ratio=hybrid_linear_ratio,
        )

        self.revin = RevIN(self.channel)

    def forward(self, x, return_intermediates=False):
        x = x.transpose(1, 2)
        x = self.revin(x, 'norm')
        x = x.transpose(1, 2)

        x_patch = self.do_patching(x)
        x_patch = self.patch_norm(x_patch)
        x_emb = self.dropoutLayer(self.patch_embedding_layer(x_patch))

        out = self.mixer1(x_emb)
        residual = out
        out = residual + self.mixer2(out)
        out = self.norm(out)

        batch_size, channel, patch_num, dim = out.shape
        out_flat = out.reshape(-1, patch_num * dim)
        out = self.head(out_flat)
        out = out.reshape(batch_size, channel, -1)

        out = out.transpose(1, 2)
        out = self.revin(out, 'denorm')
        out = out.transpose(1, 2)

        if not return_intermediates:
            return out

        return out, {
            'head_type': self.head_type,
            'patch_shape': tuple(x_patch.shape),
            'embedded_shape': tuple(x_emb.shape),
            'mixed_shape': (batch_size, channel, patch_num, dim),
            'flattened_shape': tuple(out_flat.shape),
            'prediction_shape': tuple(out.shape),
        }

    def do_patching(self, x):
        x_end = x[:, :, -1:]
        x_padding = x_end.repeat(1, 1, self.patch_stride)
        x_new = torch.cat((x, x_padding), dim=-1)
        x_patch = x_new.unfold(dimension=-1, size=self.patch_len, step=self.patch_stride)
        return x_patch

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        return self.head.regularization_loss(regularize_activation, regularize_entropy)

    def head_parameter_summary(self):
        return {
            'head_type': self.head_type,
            'input_seq': self.input_seq,
            'pred_seq': self.pred_seq,
            'patch_num': self.patch_num,
            'head_parameters': _count_parameters(self.head),
        }


class Mixer(nn.Module):
    def __init__(self,
                 input_seq=[],
                 out_seq=[],
                 batch_size=[],
                 channel=[],
                 d_model=[],
                 dropout=[],
                 tfactor=[],
                 dfactor=[]):
        super(Mixer, self).__init__()
        self.input_seq = input_seq
        self.pred_seq = out_seq
        self.batch_size = batch_size
        self.channel = channel
        self.d_model = d_model
        self.dropout = dropout
        self.tfactor = tfactor
        self.dfactor = dfactor

        self.tMixer = TokenMixer(
            input_seq=self.input_seq,
            batch_size=self.batch_size,
            channel=self.channel,
            pred_seq=self.pred_seq,
            dropout=self.dropout,
            factor=self.tfactor,
            d_model=self.d_model,
        )
        self.dropoutLayer = nn.Dropout(self.dropout)
        self.norm1 = nn.BatchNorm2d(self.channel)
        self.norm2 = nn.BatchNorm2d(self.channel)

        self.embeddingMixer = nn.Sequential(
            nn.Linear(self.d_model, self.d_model * self.dfactor),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model * self.dfactor, self.d_model),
        )

    def forward(self, x):
        x = self.norm1(x)
        x = x.permute(0, 3, 1, 2)
        x = self.dropoutLayer(self.tMixer(x))
        x = x.permute(0, 2, 3, 1)
        x = self.norm2(x)
        x = x + self.dropoutLayer(self.embeddingMixer(x))
        return x


class TokenMixer(nn.Module):
    def __init__(self, input_seq=[], batch_size=[], channel=[], pred_seq=[], dropout=[], factor=[], d_model=[]):
        super(TokenMixer, self).__init__()
        self.input_seq = input_seq
        self.batch_size = batch_size
        self.channel = channel
        self.pred_seq = pred_seq
        self.dropout = dropout
        self.factor = factor
        self.d_model = d_model

        self.dropoutLayer = nn.Dropout(self.dropout)
        self.layers = nn.Sequential(
            nn.Linear(self.input_seq, self.pred_seq * self.factor),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.pred_seq * self.factor, self.pred_seq),
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.layers(x)
        x = x.transpose(1, 2)
        return x
