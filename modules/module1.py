# Multilayer Perceptron

import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, config):
        super(MLP, self).__init__()

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
        h0 = x.view(x.shape[0], -1)
        h1 = F.relu(self.linear1(h0))
        h2 = F.relu(self.linear2(h1))
        h3 = self.linear3(h2)
        return h3

def get_module(config):
    return MLP(config)

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
