# Multilayer Perceptron

import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, ly1_in, ly1_out_ly2_in, ly2_out_ly3_in, ly3_out):
        super(MLP, self).__init__()

        self.linear1 = nn.Linear(ly1_in, ly1_out_ly2_in)
        self.linear2 = nn.Linear(ly1_out_ly2_in, ly2_out_ly3_in)
        self.linear3 = nn.Linear(ly2_out_ly3_in, ly3_out)

    def forward(self, x):
        h0 = x.view(x.shape[0], -1)
        h1 = F.relu(self.linear1(h0))
        h2 = F.relu(self.linear2(h1))
        h3 = self.linear3(h2)
        return h3

def get_module(option):
    module = MLP(
        option.input_dim, option.mid_dim_1, option.mid_dim_2, option.output_dim
    )

    return module

if __name__ == '__main__':
    from utils.parser import get_parser

    parser = get_parser()
    option = parser.parse_args()

    module = get_module(option)

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    count_parameters(module)

    para_num = count_parameters(module)
    print(f'The module has {para_num} trainable parameters')

    x = torch.zeros(16, 1, 28, 28) # (batch_size, n_channels, img_width, img_height)
    y = module(x) # (batch_size, output_dim)
