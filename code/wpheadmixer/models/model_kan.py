import torch.nn as nn

from models.model import WPMixer


class WPMixerWrapperShortTermForecast(nn.Module):
    def __init__(self, *args, **kwargs):
        super(WPMixerWrapperShortTermForecast, self).__init__()
        kwargs.setdefault('head_type', 'kan')
        self.model = WPMixer(*args, **kwargs)

    def forward(self, x, _unknown1, _unknown2, _unknown3):
        return self.model(x)


class WPMixerKAN(WPMixer):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault('head_type', 'kan')
        super(WPMixerKAN, self).__init__(*args, **kwargs)
