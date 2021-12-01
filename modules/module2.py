# LeNet

import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNet(nn.Module):
    def __init__(self, config):
        super(LeNet, self).__init__()

        # feature_extractor

        self.conv1 = nn.Conv2d(
            in_channels = config.feature_extractor['conv1']['in_channels'],
            out_channels = config.feature_extractor['conv1']['out_channels'],
            kernel_size = config.feature_extractor['conv1']['kernel_size'],
            stride = config.feature_extractor['conv1']['stride'],
            padding = config.feature_extractor['conv1']['padding'],
            dilation = config.feature_extractor['conv1']['dilation']
        )
        self.pool1 = nn.MaxPool2d(
            kernel_size = config.feature_extractor['pool1']['kernel_size'],
            stride = config.feature_extractor['pool1']['stride'],
            padding = config.feature_extractor['pool1']['padding'],
            dilation = config.feature_extractor['pool1']['dilation']
        )
        self.conv2 = nn.Conv2d(
            in_channels = config.feature_extractor['conv2']['in_channels'],
            out_channels = config.feature_extractor['conv2']['out_channels'],
            kernel_size = config.feature_extractor['conv2']['kernel_size'],
            stride = config.feature_extractor['conv2']['stride'],
            padding = config.feature_extractor['conv2']['padding'],
            dilation = config.feature_extractor['conv2']['dilation']
        )
        self.pool2 = nn.MaxPool2d(
            kernel_size = config.feature_extractor['pool2']['kernel_size'],
            stride = config.feature_extractor['pool2']['stride'],
            padding = config.feature_extractor['pool2']['padding'],
            dilation = config.feature_extractor['pool2']['dilation']
        )

        # images_classifier

        self.linear1 = nn.Linear(
            config.images_classifier['input_dim'],
            config.images_classifier['mid_dim_1']
        )
        self.linear2 = nn.Linear(
            config.images_classifier['mid_dim_1'],
            config.images_classifier['mid_dim_2']
        )
        self.linear3 = nn.Linear(
            config.images_classifier['mid_dim_2'],
            config.images_classifier['output_dim']
        )

    def forward(self, x):

        # feature_extractor

        x = self.conv1(x)
        x = F.max_pool2d(x, kernel_size = 2)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.max_pool2d(x, kernel_size = 2)
        x = F.relu(x)

        # images_classifier

        x = x.view(x.shape[0], -1)

        x = self.linear1(x)
        x = F.relu(x)

        x = self.linear2(x)
        x = F.relu(x)

        x = self.linear3(x)

        return x

def get_module(config):
    return LeNet(config)

if __name__ == '__main__':
    from utils.parser import get_config

    config = get_config()

    module = get_module(config)

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    para_num = count_parameters(module)
    print(f'The module has {para_num} trainable parameters')

    x = torch.zeros(16, 1, 28, 28) # (batch_size, n_channels, img_width, img_height)
    y = module(x) # (batch_size, output_dim)
