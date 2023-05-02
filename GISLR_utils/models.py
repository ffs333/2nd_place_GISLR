import torch
from torch import nn
import numpy as np
import timm
from efficientnet_pytorch import EfficientNet
from torchvision.ops import SqueezeExcitation


def get_model(_cfg):
    if _cfg.model == 'img_v0':
        if _cfg.deep_supervision:
            model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=250,
                                                 in_channels=3, dropout_rate=_cfg.drop_rate)
            model = Hypercolumn_Wrapper(model, drop_rate=0.1, num_blocks_hc=4)
            # model = ModelV2DeepSuper(drop_rate=_cfg.drop_rate)
            print(f'Using HYPERCOLUMN model')
        else:
            model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=250,
                                                 in_channels=3, dropout_rate=_cfg.drop_rate)

    elif _cfg.model == 'img_v0_b1':
        _cfg.deep_supervision = False
        model = EfficientNet.from_pretrained('efficientnet-b1', num_classes=250,
                                             in_channels=3, dropout_rate=_cfg.drop_rate)

    elif _cfg.model == 'effb0_timm':
        if _cfg.deep_supervision:
            model = ModelV0DeepSuper(backbone='tf_efficientnet_b0', pretrained=True,
                                     in_channels=3, drop_rate=_cfg.drop_rate
                                     )
        else:
            model = timm.create_model('tf_efficientnet_b0', num_classes=250, drop_rate=_cfg.drop_rate)
    elif _cfg.model == 'v2_b1':
        model = timm.create_model('tf_efficientnetv2_b1', num_classes=250, drop_rate=_cfg.drop_rate)
    elif _cfg.model == 'timm':
        model = timm.create_model(_cfg.encoder, num_classes=250, drop_rate=_cfg.drop_rate)

    else:
        raise ValueError('Error in "get_Model" function:',
                         f'Wrong model name. Choose one from ["img_v0", "TREF"]')
    model.eval()
    return model


class ModelV0DeepSuper(nn.Module):
    def __init__(self, backbone: str, pretrained=False, in_channels=3, drop_rate=0.0):
        super().__init__()

        NORM_EPS = 1e-3
        self.drop_rate = drop_rate

        self.encoder = timm.create_model(backbone,
                                         pretrained=pretrained,
                                         in_chans=in_channels,
                                         num_classes=250,
                                         drop_rate=drop_rate
                                         )
        self.sup_inds = [3, 4, 5, 6]

        self.sv1 = nn.Sequential(nn.BatchNorm2d(self.encoder.blocks[self.sup_inds[0]][-1].conv_pwl.out_channels,
                                                eps=NORM_EPS),
                                 nn.AdaptiveAvgPool2d((1, 1)),
                                 nn.Flatten(),
                                 nn.Dropout(p=drop_rate),
                                 nn.Linear(self.encoder.blocks[self.sup_inds[0]][-1].conv_pwl.out_channels, 250))

        self.sv2 = nn.Sequential(nn.BatchNorm2d(self.encoder.blocks[self.sup_inds[1]][-1].conv_pwl.out_channels,
                                                eps=NORM_EPS),
                                 nn.AdaptiveAvgPool2d((1, 1)),
                                 nn.Flatten(),
                                 nn.Dropout(p=drop_rate),
                                 nn.Linear(self.encoder.blocks[self.sup_inds[1]][-1].conv_pwl.out_channels, 250))

        self.sv3 = nn.Sequential(nn.BatchNorm2d(self.encoder.blocks[self.sup_inds[2]][-1].conv_pwl.out_channels,
                                                eps=NORM_EPS),
                                 nn.AdaptiveAvgPool2d((1, 1)),
                                 nn.Flatten(),
                                 nn.Dropout(p=drop_rate),
                                 nn.Linear(self.encoder.blocks[self.sup_inds[2]][-1].conv_pwl.out_channels, 250))

        self.sv4 = nn.Sequential(nn.BatchNorm2d(self.encoder.blocks[self.sup_inds[3]][-1].conv_pwl.out_channels,
                                                eps=NORM_EPS),
                                 nn.AdaptiveAvgPool2d((1, 1)),
                                 nn.Flatten(),
                                 nn.Dropout(p=drop_rate),
                                 nn.Linear(self.encoder.blocks[self.sup_inds[3]][-1].conv_pwl.out_channels, 250))

    def forward(self, img):
        x = self.encoder.conv_stem(img)
        x = self.encoder.bn1(x)

        features = {}
        for idx in range(len(self.encoder.blocks)):
            x = self.encoder.blocks[idx](x)
            if idx in self.sup_inds:
                features[idx] = x

        x = self.encoder.conv_head(x)
        x = self.encoder.bn2(x)

        x = self.encoder.forward_head(x)
        sv1 = self.sv1(features[self.sup_inds[0]])
        sv2 = self.sv2(features[self.sup_inds[1]])
        sv3 = self.sv3(features[self.sup_inds[2]])
        sv4 = self.sv4(features[self.sup_inds[3]])
        return x, [sv1, sv2, sv3, sv4]


class ModelV2DeepSuper(nn.Module):
    def __init__(self, drop_rate=0.0):
        super().__init__()

        NORM_EPS = 1e-3
        self.drop_rate = drop_rate

        self.encoder = EfficientNet.from_pretrained('efficientnet-b0', num_classes=250,
                                                    in_channels=3, dropout_rate=drop_rate)

        self.sup_inds = [7, 10, 14, 15]

        self.sv1 = nn.Sequential(nn.BatchNorm2d(80,
                                                eps=NORM_EPS),
                                 nn.AdaptiveAvgPool2d((1, 1)),
                                 nn.Flatten(),
                                 nn.Dropout(p=drop_rate),
                                 nn.Linear(80, 250))

        self.sv2 = nn.Sequential(nn.BatchNorm2d(112,
                                                eps=NORM_EPS),
                                 nn.AdaptiveAvgPool2d((1, 1)),
                                 nn.Flatten(),
                                 nn.Dropout(p=drop_rate),
                                 nn.Linear(112, 250))

        self.sv3 = nn.Sequential(nn.BatchNorm2d(192,
                                                eps=NORM_EPS),
                                 nn.AdaptiveAvgPool2d((1, 1)),
                                 nn.Flatten(),
                                 nn.Dropout(p=drop_rate),
                                 nn.Linear(192, 250))

        self.sv4 = nn.Sequential(nn.BatchNorm2d(320,
                                                eps=NORM_EPS),
                                 nn.AdaptiveAvgPool2d((1, 1)),
                                 nn.Flatten(),
                                 nn.Dropout(p=drop_rate),
                                 nn.Linear(320, 250))

    def forward(self, img):

        x = self.encoder._swish(self.encoder._bn0(self.encoder._conv_stem(img)))
        features = {}
        for idx, block in enumerate(self.encoder._blocks):
            drop_connect_rate = self.encoder._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self.encoder._blocks)  # scale drop connect_rate
            x = block(x, drop_connect_rate=drop_connect_rate)
            if idx in self.sup_inds:
                features[idx] = x

        # Head
        x = self.encoder._swish(self.encoder._bn1(self.encoder._conv_head(x)))
        x = self.encoder._avg_pooling(x)
        x = x.flatten(start_dim=1)
        x = self.encoder._dropout(x)
        x = self.encoder._fc(x)

        sv1 = self.sv1(features[self.sup_inds[0]])
        sv2 = self.sv2(features[self.sup_inds[1]])
        sv3 = self.sv3(features[self.sup_inds[2]])
        sv4 = self.sv4(features[self.sup_inds[3]])
        return x, [sv1, sv2, sv3, sv4]


class ImgModelMeta(nn.Module):
    def __init__(self, drop_rate=0.0):
        super().__init__()

        self.encoder = EfficientNet.from_pretrained('efficientnet-b0', num_classes=250, dropout_rate=drop_rate)
        enc_out_size = self.encoder._fc.in_features
        self.encoder._fc = nn.Identity()
        self.logits = nn.Sequential(
            nn.Linear(enc_out_size + 493, 768),
            nn.ReLU(),
            nn.Dropout(drop_rate),
            nn.Linear(768, 250))

    def forward(self, img, meta):
        x = self.encoder(img)
        x = torch.cat([x, meta], dim=1)
        x = self.logits(x)
        return x


class ImgModelV0(nn.Module):
    def __init__(self, backbone: str, pretrained=True, in_channels=3, drop_rate=0.0):
        super().__init__()

        self.drop_rate = drop_rate

        self.encoder = timm.create_model(backbone,
                                         pretrained=pretrained,
                                         in_chans=in_channels,
                                         num_classes=250,
                                         drop_rate=drop_rate
                                         )

    def forward(self, img):
        x = self.encoder(img)
        return x


class Supervision_Wrapper(nn.Module):
    def __init__(
            self,
            base_model,
            num_blocks_sv=5,
            norp_eps=1e-5,
            drop_rate=0.0,
    ):
        super().__init__()
        self.super_vision = nn.ModuleDict()
        self.base_model = base_model
        self.sup_inds = []
        self.num_blocks_sv = num_blocks_sv
        need_blocks, need_sizes = self.get_block_idxs()
        for n, (i, num_features) in enumerate(zip(need_blocks, need_sizes)):
            # register
            self.base_model._blocks[i].register_forward_hook(self.get_features(f'feats_{n}'))
            # supervision layer
            self.sup_inds.append(i)
            self.super_vision[f'feats_{n}'] = nn.Sequential(nn.BatchNorm2d(num_features, eps=norp_eps),
                                                            nn.AdaptiveAvgPool2d((1, 1)),
                                                            nn.Flatten(),
                                                            nn.Dropout(drop_rate),
                                                            nn.Linear(num_features, 250))

    def get_block_idxs(self):
        sizes = []
        for n, i in enumerate(range(len(self.base_model._blocks))):
            # get last layer size
            for layer in self.base_model._blocks[i].children():
                try:
                    num_features = layer.num_features
                except:
                    pass
            sizes.append(num_features)

        sizes = np.array(sizes)
        idx_sort = np.argsort(sizes)
        sorted_sizes = sizes[idx_sort]
        vals, idx_start, count = np.unique(sorted_sizes, return_counts=True, return_index=True)
        res = np.split(idx_sort, idx_start[1:])
        vals = vals[count > 1]
        need_blocks = [max(r) for r in res]

        need_blocks = need_blocks[-self.num_blocks_sv - 1:-1]
        need_sizes = sizes[need_blocks]
        return need_blocks, need_sizes

    def get_features(self, name):
        def hook(model, input, output):
            self.features[name] = output

        return hook

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.features = {}
        x = self.base_model(x)
        svs = []
        for n in range(len(self.super_vision)):
            svs.append(self.super_vision[f'feats_{n}'](self.features[f'feats_{n}']))
        return x, svs


class Hypercolumn_Wrapper(nn.Module):
    def __init__(
            self,
            base_model,
            num_blocks_hc=5,
            norp_eps=1e-5,
            drop_rate=0.0,
            column_features=64,
            head_hc_output_features=256,
            hc_size=(16, 8)

    ):
        super().__init__()
        self.hypercolumns = nn.ModuleDict()
        self.base_model = base_model
        self.num_blocks_hc = num_blocks_hc
        self.column_features = column_features
        self.head_hc_input_features = int(self.column_features * self.num_blocks_hc)
        self.head_hc_output_features = head_hc_output_features
        self.hc_size = hc_size
        self.sup_inds = [0, ]  # increese_weights
        self.use_all_features = True
        self.drop_last = False

        need_blocks, need_sizes = self.get_block_idxs()
        total_features = 0
        for n, (i, num_features) in enumerate(zip(need_blocks, need_sizes)):
            # register
            self.base_model._blocks[i].register_forward_hook(self.get_features(f'feats_{n}'))
            # supervision layer
            if self.use_all_features:
                self.hypercolumns[f'feats_{n}'] = nn.Sequential(nn.Identity(),
                                                                )
                total_features += num_features
            else:
                self.hypercolumns[f'feats_{n}'] = nn.Sequential(nn.BatchNorm2d(num_features, eps=norp_eps),
                                                                nn.Conv2d(num_features, self.column_features,
                                                                          kernel_size=(3, 3),
                                                                          padding=1),
                                                                )

        if self.use_all_features:
            self.head_hc_input_features = total_features

        self.hyper_head = nn.Sequential(nn.BatchNorm2d(self.head_hc_input_features, eps=norp_eps),
                                        SqueezeExcitation(self.head_hc_input_features, 64),
                                        nn.Conv2d(self.head_hc_input_features,
                                                  self.head_hc_output_features,
                                                  kernel_size=(3, 3),
                                                  padding=1),
                                        nn.AdaptiveAvgPool2d((1, 1)),
                                        nn.Flatten(),
                                        nn.Dropout(drop_rate),
                                        nn.Linear(self.head_hc_output_features, 250)
                                        )

    def get_block_idxs(self):
        sizes = []
        for n, i in enumerate(range(len(self.base_model._blocks))):
            # get last layer size
            for layer in self.base_model._blocks[i].children():
                try:
                    num_features = layer.num_features
                except:
                    pass
            sizes.append(num_features)

        sizes = np.array(sizes)
        idx_sort = np.argsort(sizes)
        sorted_sizes = sizes[idx_sort]
        vals, idx_start, count = np.unique(sorted_sizes, return_counts=True, return_index=True)
        res = np.split(idx_sort, idx_start[1:])
        vals = vals[count > 1]
        need_blocks = [max(r) for r in res]
        if self.drop_last:
            need_blocks = need_blocks[-self.num_blocks_hc - 1:-1]
        else:
            need_blocks = need_blocks[-self.num_blocks_hc:]
        need_sizes = sizes[need_blocks]
        return need_blocks, need_sizes

    def get_features(self, name):
        def hook(model, input, output):
            self.features[name] = output

        return hook

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.features = {}
        x = self.base_model(x)
        layers = []
        for n in range(len(self.hypercolumns)):
            layer = self.hypercolumns[f'feats_{n}'](self.features[f'feats_{n}'])
            layer = nn.functional.interpolate(
                layer, self.hc_size,
                mode="bilinear",
                align_corners=True)

            layers.append(layer)

        hypercols = torch.cat(layers, dim=1)
        hp = self.hyper_head(hypercols)
        return x, [hp, ]
