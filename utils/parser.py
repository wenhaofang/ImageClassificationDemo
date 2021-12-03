import yaml
import argparse

def get_parser():
    parser = argparse.ArgumentParser()

    # For Basic
    parser.add_argument('--name', default = 'main', help = '')

    # For Loader
    parser.add_argument('--loader', type = int, choices = range(1, 3), default = 1, help = '')
    parser.add_argument('--loader_config_path', default = 'configs/loader1.yaml')

    # For Module
    parser.add_argument('--module', type = int, choices = range(1, 6), default = 1, help = '')
    parser.add_argument('--module_config_path', default = 'configs/module1.yaml')

    parser.add_argument('--vgg_net_arch', type = int, choices = [11, 13, 16, 19], default = 11, help = '')
    parser.add_argument('--res_net_arch', type = int, choices = [18, 34, 50, 101, 152], default = 18, help = '')

    # For Pre-Train
    parser.add_argument('--use_pretrain', action = 'store_true', help = '')

    parser.add_argument('--pretrain_image_shape', type = int  , nargs = '+', default = (224, 224), help = '')
    parser.add_argument('--pretrain_image_mean' , type = float, nargs = '+', default = (0.485, 0.456, 0.406), help = '')
    parser.add_argument('--pretrain_image_std'  , type = float, nargs = '+', default = (0.229, 0.224, 0.225), help = '')

    # For Train
    parser.add_argument('--batch_size', type = int, default = 64, help = '')
    parser.add_argument('--num_epochs', type = int, default = 10, help = '')

    return parser

def get_config():
    parser = get_parser()
    option = parser.parse_args()

    with open(option.loader_config_path, 'r', encoding = 'utf-8') as config_file:
        loader_config = yaml.full_load(config_file)

    with open(option.module_config_path ,'r', encoding = 'utf-8') as config_file:
        module_config = yaml.full_load(config_file)

    config = {}
    config.update(vars(option))
    config.update(loader_config)
    config.update(module_config)

    config = argparse.Namespace(**config)
    return config

if __name__ == '__main__':
    config = get_config()
    print(config)
