import torch.nn as nn

from models.wavelet_patch_mixer import WPMixerCore


class WPMixerWrapperShortTermForecast(nn.Module):
    def __init__(self,
                 c_in=[],
                 c_out=[],
                 seq_len=[],
                 out_len=[],
                 d_model=[],
                 dropout=[],
                 embedding_dropout=[],
                 device=[],
                 batch_size=[],
                 tfactor=[],
                 dfactor=[],
                 wavelet=[],
                 level=[],
                 patch_len=[],
                 stride=[],
                 no_decomposition=[],
                 use_amp=[],
                 head_type='linear',
                 match_head_params=False,
                 head_param_budget=None,
                 mlp_hidden_dim=None,
                 kan_grid_size=5,
                 kan_spline_order=3,
                 hybrid_linear_ratio=0.5):
        super(WPMixerWrapperShortTermForecast, self).__init__()
        self.model = WPMixer(
            c_in=c_in,
            c_out=c_out,
            seq_len=seq_len,
            out_len=out_len,
            d_model=d_model,
            dropout=dropout,
            embedding_dropout=embedding_dropout,
            device=device,
            batch_size=batch_size,
            tfactor=tfactor,
            dfactor=dfactor,
            wavelet=wavelet,
            level=level,
            patch_len=patch_len,
            stride=stride,
            no_decomposition=no_decomposition,
            use_amp=use_amp,
            head_type=head_type,
            match_head_params=match_head_params,
            head_param_budget=head_param_budget,
            mlp_hidden_dim=mlp_hidden_dim,
            kan_grid_size=kan_grid_size,
            kan_spline_order=kan_spline_order,
            hybrid_linear_ratio=hybrid_linear_ratio,
        )

    def forward(self, x, _unknown1, _unknown2, _unknown3):
        return self.model(x)


class WPMixer(nn.Module):
    def __init__(self,
                 c_in=[],
                 c_out=[],
                 seq_len=[],
                 out_len=[],
                 d_model=[],
                 dropout=[],
                 embedding_dropout=[],
                 device=[],
                 batch_size=[],
                 tfactor=[],
                 dfactor=[],
                 wavelet=[],
                 level=[],
                 patch_len=[],
                 stride=[],
                 no_decomposition=[],
                 use_amp=[],
                 head_type='linear',
                 match_head_params=False,
                 head_param_budget=None,
                 mlp_hidden_dim=None,
                 kan_grid_size=5,
                 kan_spline_order=3,
                 hybrid_linear_ratio=0.5):

        super(WPMixer, self).__init__()
        self.pred_len = out_len
        self.channel_in = c_in
        self.channel_out = c_out
        self.patch_len = patch_len
        self.stride = stride
        self.seq_len = seq_len
        self.d_model = d_model
        self.dropout = dropout
        self.embedding_dropout = embedding_dropout
        self.batch_size = batch_size
        self.tfactor = tfactor
        self.dfactor = dfactor
        self.wavelet = wavelet
        self.level = level
        self.actual_seq_len = seq_len
        self.no_decomposition = no_decomposition
        self.use_amp = use_amp
        self.device = device
        self.head_type = head_type

        self.wpmixerCore = WPMixerCore(
            input_length=self.actual_seq_len,
            pred_length=self.pred_len,
            wavelet_name=self.wavelet,
            level=self.level,
            batch_size=self.batch_size,
            channel=self.channel_in,
            d_model=self.d_model,
            dropout=self.dropout,
            embedding_dropout=self.embedding_dropout,
            tfactor=self.tfactor,
            dfactor=self.dfactor,
            device=self.device,
            patch_len=self.patch_len,
            patch_stride=self.stride,
            no_decomposition=self.no_decomposition,
            use_amp=self.use_amp,
            head_type=self.head_type,
            match_head_params=match_head_params,
            head_param_budget=head_param_budget,
            mlp_hidden_dim=mlp_hidden_dim,
            kan_grid_size=kan_grid_size,
            kan_spline_order=kan_spline_order,
            hybrid_linear_ratio=hybrid_linear_ratio,
        )

    def _select_output_channels(self, pred):
        return pred[:, :, -self.channel_out:]

    def forward(self, x):
        pred = self.wpmixerCore(x)
        return self._select_output_channels(pred)

    def forward_with_intermediates(self, x):
        pred, intermediates = self.wpmixerCore(x, return_intermediates=True)
        return self._select_output_channels(pred), intermediates

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        return self.wpmixerCore.regularization_loss(regularize_activation, regularize_entropy)

    def head_parameter_summary(self):
        return self.wpmixerCore.head_parameter_summary()
