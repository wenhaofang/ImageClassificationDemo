# ResNet

import torch
import torch.nn as nn

import torchvision.models as models

class BasicBlock(nn.Module):

    expansion = 1

    def __init__(self, in_channels, out_channels, block_config, down_sample):
        super(BasicBlock, self).__init__()

        self.down_sample = down_sample

        self.conv1 = nn.Conv2d(
            in_channels = in_channels,
            out_channels = out_channels,
            kernel_size = block_config['conv1']['kernel_size'],
            stride = block_config['conv1']['stride'],
            padding = block_config['conv1']['padding'],
            dilation = block_config['conv1']['dilation'],
            bias = block_config['conv1']['bias']
        )
        self.bn1 = nn.BatchNorm2d(
            num_features = out_channels
        )
        self.conv2 = nn.Conv2d(
            in_channels = out_channels,
            out_channels = out_channels,
            kernel_size = block_config['conv2']['kernel_size'],
            stride = block_config['conv2']['stride'],
            padding = block_config['conv2']['padding'],
            dilation = block_config['conv2']['dilation'],
            bias = block_config['conv2']['bias']
        )
        self.bn2 = nn.BatchNorm2d(
            num_features = out_channels
        )

        self.relu = nn.ReLU(inplace = True)

        if down_sample:
            self.conv = nn.Conv2d(
                in_channels = in_channels,
                out_channels = out_channels,
                kernel_size = block_config['conv']['kernel_size'],
                stride = block_config['conv']['stride'],
                padding = block_config['conv']['padding'],
                dilation = block_config['conv']['dilation'],
                bias = block_config['conv']['bias']
            )
            self.bn = nn.BatchNorm2d(
                num_features = out_channels
            )

    def forward(self, x):
        i = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)

        if self.down_sample:
            i = self.conv(i)
            i = self.bn(i)

        x += i
        x = self.relu(x)

        return x

class BottleNeck(nn.Module):

    expansion = 4

    def __init__(self, in_channels, out_channels, block_config, down_sample):
        super(BottleNeck, self).__init__()

        self.down_sample = down_sample

        self.conv1 = nn.Conv2d(
            in_channels = in_channels,
            out_channels = out_channels,
            kernel_size = block_config['conv1']['kernel_size'],
            stride = block_config['conv1']['stride'],
            padding = block_config['conv1']['padding'],
            dilation = block_config['conv1']['dilation'],
            bias = block_config['conv1']['bias']
        )
        self.bn1 = nn.BatchNorm2d(
            num_features = out_channels
        )
        self.conv2 = nn.Conv2d(
            in_channels = out_channels,
            out_channels = out_channels,
            kernel_size = block_config['conv2']['kernel_size'],
            stride = block_config['conv2']['stride'],
            padding = block_config['conv2']['padding'],
            dilation = block_config['conv2']['dilation'],
            bias = block_config['conv2']['bias']
        )
        self.bn2 = nn.BatchNorm2d(
            num_features = out_channels
        )
        self.conv3 = nn.Conv2d(
            in_channels = out_channels,
            out_channels = out_channels * self.expansion,
            kernel_size = block_config['conv3']['kernel_size'],
            stride = block_config['conv3']['stride'],
            padding = block_config['conv3']['padding'],
            dilation = block_config['conv3']['dilation'],
            bias = block_config['conv3']['bias']
        )
        self.bn3 = nn.BatchNorm2d(
            num_features = out_channels * self.expansion
        )

        self.relu = nn.ReLU(inplace = True)

        if down_sample:
            self.conv = nn.Conv2d(
                in_channels = in_channels,
                out_channels = out_channels * self.expansion,
                kernel_size = block_config['conv']['kernel_size'],
                stride = block_config['conv']['stride'],
                padding = block_config['conv']['padding'],
                dilation = block_config['conv']['dilation'],
                bias = block_config['conv']['bias']
            )
            self.bn = nn.BatchNorm2d(
                num_features = out_channels * self.expansion
            )

    def forward(self, x):
        i = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)

        if self.down_sample:
            i = self.conv(i)
            i = self.bn(i)

        x += i
        x = self.relu(x)

        return x

class ResNet(nn.Module):
    def __init__(self, config):
        super(ResNet, self).__init__()

        block_mapping = {
            'BasicBlock': BasicBlock,
            'BottleNeck': BottleNeck
        }

        if config.res_net_arch == 18:
            block = block_mapping[config.feature_extractor['ResNet18']['block']]
            n_blocks = config.feature_extractor['ResNet18']['n_blocks']
            channels = config.feature_extractor['ResNet18']['channels']
        elif config.res_net_arch == 34:
            block = block_mapping[config.feature_extractor['ResNet34']['block']]
            n_blocks = config.feature_extractor['ResNet34']['n_blocks']
            channels = config.feature_extractor['ResNet34']['channels']
        elif config.res_net_arch == 50:
            block = block_mapping[config.feature_extractor['ResNet50']['block']]
            n_blocks = config.feature_extractor['ResNet50']['n_blocks']
            channels = config.feature_extractor['ResNet50']['channels']
        elif config.res_net_arch == 101:
            block = block_mapping[config.feature_extractor['ResNet101']['block']]
            n_blocks = config.feature_extractor['ResNet101']['n_blocks']
            channels = config.feature_extractor['ResNet101']['channels']
        elif config.res_net_arch == 152:
            block = block_mapping[config.feature_extractor['ResNet152']['block']]
            n_blocks = config.feature_extractor['ResNet152']['n_blocks']
            channels = config.feature_extractor['ResNet152']['channels']

        assert len(n_blocks) == len(channels) == 4

        self.in_channels = channels[0]
        image_channels = config.image_channels

        self.conv = nn.Conv2d(
            in_channels = image_channels,
            out_channels = self.in_channels,
            kernel_size = config.feature_extractor['conv']['kernel_size'],
            stride = config.feature_extractor['conv']['stride'],
            padding = config.feature_extractor['conv']['padding'],
            dilation = config.feature_extractor['conv']['dilation'],
            bias = config.feature_extractor['conv']['bias']
        )
        self.bn = nn.BatchNorm2d(
            num_features = self.in_channels
        )
        self.relu = nn.ReLU(inplace = True)

        self.max_pool = nn.MaxPool2d(
            kernel_size = config.feature_extractor['max_pool']['kernel_size'],
            stride = config.feature_extractor['max_pool']['stride'],
            padding = config.feature_extractor['max_pool']['padding'],
            dilation = config.feature_extractor['max_pool']['dilation']
        )

        if block == BasicBlock:
            self.layer1 = self.get_resnet_layers(block, n_blocks[0], channels[0], config.feature_extractor['basic_block']['layer_first'])
            self.layer2 = self.get_resnet_layers(block, n_blocks[1], channels[1], config.feature_extractor['basic_block']['layer_other'])
            self.layer3 = self.get_resnet_layers(block, n_blocks[2], channels[2], config.feature_extractor['basic_block']['layer_other'])
            self.layer4 = self.get_resnet_layers(block, n_blocks[3], channels[3], config.feature_extractor['basic_block']['layer_other'])
        elif block == BottleNeck:
            self.layer1 = self.get_resnet_layers(block, n_blocks[0], channels[0], config.feature_extractor['bottle_neck']['layer_first'])
            self.layer2 = self.get_resnet_layers(block, n_blocks[1], channels[1], config.feature_extractor['bottle_neck']['layer_other'])
            self.layer3 = self.get_resnet_layers(block, n_blocks[2], channels[2], config.feature_extractor['bottle_neck']['layer_other'])
            self.layer4 = self.get_resnet_layers(block, n_blocks[3], channels[3], config.feature_extractor['bottle_neck']['layer_other'])

        self.avg_pool = nn.AdaptiveAvgPool2d(config.feature_extractor['avg_pool']['output_size'])

        self.linear = nn.Linear(
            self.in_channels,
            config.num_classes
        )

    def get_resnet_layers(self, block, n_blocks, channels, layer_config):
        layers = []

        down_sample = True if self.in_channels != block.expansion * channels else False

        for i in range(n_blocks):
            if i == 0:
                layers.append(block(self.in_channels, channels, layer_config['block_first'], down_sample))
            else:
                layers.append(block(block.expansion * channels, channels, layer_config['block_other'], False))

        self.in_channels = block.expansion * channels

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        x = self.max_pool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avg_pool(x)

        x = x.view(x.shape[0], -1)
        x = self.linear(x)

        return x

def get_module(config):
    if config.use_pretrain:
        if config.res_net_arch == 18:
            module = models.resnet18(pretrained = True)
        if config.res_net_arch == 34:
            module = models.resnet34(pretrained = True)
        if config.res_net_arch == 50:
            module = models.resnet50(pretrained = True)
        if config.res_net_arch == 101:
            module = models.resnet101(pretrained = True)
        if config.res_net_arch == 152:
            module = models.resnet152(pretrained = True)

        module.fc = nn.Linear(module.fc.in_features, config.num_classes)
        return module
    else:
        return ResNet(config)

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
