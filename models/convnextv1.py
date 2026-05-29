import torch
import torch.nn as nn
from .utils import LayerNorm, DropPath, trunc_normal_


class Block(nn.Module):
    """ConvNeXt V1 Block: GRN 대신 Layer Scale 사용 (V2와의 핵심 차이)."""
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(
            layer_scale_init_value * torch.ones(dim), requires_grad=True
        ) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)
        return shortcut + self.drop_path(x)


class ConvNeXtV1(nn.Module):
    def __init__(self, in_chans=3, num_classes=1000,
                 depths=[3, 3, 9, 3], dims=[96, 192, 384, 768],
                 drop_path_rate=0., layer_scale_init_value=1e-6,
                 head_init_scale=1.):
        super().__init__()
        self.downsample_layers = nn.ModuleList()

        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            self.downsample_layers.append(nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            ))

        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            self.stages.append(nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j],
                        layer_scale_init_value=layer_scale_init_value)
                  for j in range(depths[i])]
            ))
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)
        self.head = nn.Linear(dims[-1], num_classes)

        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return self.norm(x.mean([-2, -1]))

    def forward(self, x):
        return self.head(self.forward_features(x))

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())


def convnext_tiny(**kwargs):
    return ConvNeXtV1(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], **kwargs)

def convnext_small(**kwargs):
    return ConvNeXtV1(depths=[3, 3, 27, 3], dims=[96, 192, 384, 768], **kwargs)

def convnext_base(**kwargs):
    return ConvNeXtV1(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], **kwargs)

def convnext_large(**kwargs):
    return ConvNeXtV1(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536], **kwargs)

def convnext_huge(**kwargs):
    return ConvNeXtV1(depths=[3, 3, 27, 3], dims=[352, 704, 1408, 2816], **kwargs)


MODEL_CONFIGS = {
    'convnext_tiny':  convnext_tiny,
    'convnext_small': convnext_small,
    'convnext_base':  convnext_base,
    'convnext_large': convnext_large,
    'convnext_huge':  convnext_huge,
}
