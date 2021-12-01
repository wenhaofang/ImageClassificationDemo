# AlexNet

import torch
import torch.nn as nn

class AlexNet(nn.Module):
    def __init__(self, config):
        super(AlexNet, self).__init__()

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(
                in_channels = config.feature_extractor['conv1']['in_channels'],
                out_channels = config.feature_extractor['conv1']['out_channels'],
                kernel_size = config.feature_extractor['conv1']['kernel_size'],
                stride = config.feature_extractor['conv1']['stride'],
                padding = config.feature_extractor['conv1']['padding'],
                dilation = config.feature_extractor['conv1']['dilation']
            ),
            nn.MaxPool2d(
                kernel_size = config.feature_extractor['pool1']['kernel_size'],
                stride = config.feature_extractor['pool1']['stride'],
                padding = config.feature_extractor['pool1']['padding'],
                dilation = config.feature_extractor['pool1']['dilation']
            ),
            nn.ReLU(inplace = True),
            nn.Conv2d(
                in_channels = config.feature_extractor['conv2']['in_channels'],
                out_channels = config.feature_extractor['conv2']['out_channels'],
                kernel_size = config.feature_extractor['conv2']['kernel_size'],
                stride = config.feature_extractor['conv2']['stride'],
                padding = config.feature_extractor['conv2']['padding'],
                dilation = config.feature_extractor['conv2']['dilation']
            ),
            nn.MaxPool2d(
                kernel_size = config.feature_extractor['pool2']['kernel_size'],
                stride = config.feature_extractor['pool2']['stride'],
                padding = config.feature_extractor['pool2']['padding'],
                dilation = config.feature_extractor['pool2']['dilation']
            ),
            nn.ReLU(inplace = True),
            nn.Conv2d(
                in_channels = config.feature_extractor['conv3']['in_channels'],
                out_channels = config.feature_extractor['conv3']['out_channels'],
                kernel_size = config.feature_extractor['conv3']['kernel_size'],
                stride = config.feature_extractor['conv3']['stride'],
                padding = config.feature_extractor['conv3']['padding'],
                dilation = config.feature_extractor['conv3']['dilation']
            ),
            nn.ReLU(inplace = True),
            nn.Conv2d(
                in_channels = config.feature_extractor['conv4']['in_channels'],
                out_channels = config.feature_extractor['conv4']['out_channels'],
                kernel_size = config.feature_extractor['conv4']['kernel_size'],
                stride = config.feature_extractor['conv4']['stride'],
                padding = config.feature_extractor['conv4']['padding'],
                dilation = config.feature_extractor['conv4']['dilation']
            ),
            nn.ReLU(inplace = True),
            nn.Conv2d(
                in_channels = config.feature_extractor['conv5']['in_channels'],
                out_channels = config.feature_extractor['conv5']['out_channels'],
                kernel_size = config.feature_extractor['conv5']['kernel_size'],
                stride = config.feature_extractor['conv5']['stride'],
                padding = config.feature_extractor['conv5']['padding'],
                dilation = config.feature_extractor['conv5']['dilation']
            ),
            nn.MaxPool2d(
                kernel_size = config.feature_extractor['pool5']['kernel_size'],
                stride = config.feature_extractor['pool5']['stride'],
                padding = config.feature_extractor['pool5']['padding'],
                dilation = config.feature_extractor['pool5']['dilation']
            ),
            nn.ReLU(inplace = True)
        )

        self.images_classifier = nn.Sequential(
            nn.Dropout(0.5),
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
            nn.Linear(
                config.images_classifier['mid_dim_2'],
                config.images_classifier['output_dim']
            )
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.view(x.shape[0], -1)
        x = self.images_classifier(x)
        return x

def get_module(config):
    module = AlexNet(config)

    def init_weight(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight.data, nonlinearity = 'relu')
            nn.init.constant_(m.bias.data, 0)
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_ (m.weight.data, gain = nn.init.calculate_gain('relu'))
            nn.init.constant_(m.bias.data, 0)

    module.apply(init_weight)
    return module

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
