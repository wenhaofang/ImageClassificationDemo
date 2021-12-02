# VGG

import torch
import torch.nn as nn

import torchvision.models as models

class VGG(nn.Module):
    def __init__(self, config):
        super(VGG, self).__init__()

        self.feature_extractor = self.get_vgg_layers(config)

        self.avg_pool = nn.AdaptiveAvgPool2d(config.avg_pool['output_size'])

        self.images_classifier = nn.Sequential(
            nn.Linear(
                config.images_classifier['input_dim'],
                config.images_classifier['mid_dim_1']
            ),
            nn.ReLU(inplace = True),
            nn.Dropout(0.5),
            nn.Linear(
                config.images_classifier['mid_dim_1'],
                config.images_classifier['mid_dim_2']
            ),
            nn.ReLU(inplace = True),
            nn.Dropout(0.5),
            nn.Linear(
                config.images_classifier['mid_dim_2'],
                config.images_classifier['output_dim']
            )
        )

    def get_vgg_layers(self, config):
        layers = []

        in_channels = config.image_channels

        if config.vgg_net_arch == 11:
            layer_infos = config.feature_extractor['vgg11']
        elif config.vgg_net_arch == 13:
            layer_infos = config.feature_extractor['vgg13']
        elif config.vgg_net_arch == 16:
            layer_infos = config.feature_extractor['vgg16']
        elif config.vgg_net_arch == 19:
            layer_infos = config.feature_extractor['vgg19']

        batch_norm = config.feature_extractor['batch_norm']

        for info in layer_infos:
            assert info == 'M' or isinstance(info, int)

            if info == 'M':
                layers += [nn.MaxPool2d(
                    kernel_size = config.feature_extractor['pool']['kernel_size'],
                    stride = config.feature_extractor['pool']['stride'],
                    padding = config.feature_extractor['pool']['padding'],
                    dilation = config.feature_extractor['pool']['dilation']
                )]

            if isinstance(info, int):
                layers += [nn.Conv2d(
                    in_channels = in_channels,
                    out_channels = info,
                    kernel_size = config.feature_extractor['conv']['kernel_size'],
                    stride = config.feature_extractor['conv']['stride'],
                    padding = config.feature_extractor['conv']['padding'],
                    dilation = config.feature_extractor['conv']['dilation']
                )]
                if batch_norm:
                    layers += [nn.BatchNorm2d(info)]
                    layers += [nn.ReLU(inplace = True)]
                else:
                    layers += [nn.ReLU(inplace = True)]

                in_channels = info

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.avg_pool(x)
        x = x.view(x.shape[0], -1)
        x = self.images_classifier(x)
        return x

def get_module(config):
    if config.use_pretrain:
        if config.vgg_net_arch == 11 and config.feature_extractor['batch_norm'] == True:
            module = models.vgg11_bn(pretrained = True)
        if config.vgg_net_arch == 11 and config.feature_extractor['batch_norm'] == False:
            module = models.vgg11(pretrained = True)
        if config.vgg_net_arch == 13 and config.feature_extractor['batch_norm'] == True:
            module = models.vgg13_bn(pretrained = True)
        if config.vgg_net_arch == 13 and config.feature_extractor['batch_norm'] == False:
            module = models.vgg13(pretrained = True)
        if config.vgg_net_arch == 16 and config.feature_extractor['batch_norm'] == True:
            module = models.vgg16_bn(pretrained = True)
        if config.vgg_net_arch == 16 and config.feature_extractor['batch_norm'] == False:
            module = models.vgg16(pretrained = True)
        if config.vgg_net_arch == 19 and config.feature_extractor['batch_norm'] == True:
            module = models.vgg19_bn(pretrained = True)
        if config.vgg_net_arch == 19 and config.feature_extractor['batch_norm'] == False:
            module = models.vgg19(pretrained = True)

        module.classifier[-1] = nn.Linear(module.classifier[-1].in_features, config.num_classes)
        return module
    else:
        return VGG(config)

if __name__ == '__main__':
    from utils.parser import get_config

    config = get_config()

    module = get_module(config)

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    para_num = count_parameters(module)
    print(f'The module has {para_num} trainable parameters')

    x = torch.zeros(16, 3, 32, 32) # (batch_size, n_channels, img_width, img_height)
    y = module(x) # (batch_size, output_dim)
