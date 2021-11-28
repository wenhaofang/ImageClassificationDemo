import argparse

def get_parser():
    parser = argparse.ArgumentParser()

    # For Basic
    parser.add_argument('--name', default = 'main', help = '')

    # For Loader
    parser.add_argument('--sources_path', default = 'datasources', help = '')
    parser.add_argument('--targets_path', default = 'datatargets', help = '')

    parser.add_argument('--train_file', default = 'train_data.tsv', help = '')
    parser.add_argument('--valid_file', default = 'valid_data.tsv', help = '')
    parser.add_argument('--test_file' , default = 'test_data.tsv' , help = '')

    parser.add_argument('--image_folder', default = 'images' , help = '')

    # For Module

    parser.add_argument('--input_dim', type = int, default = 784, help = '')
    parser.add_argument('--output_dim', type = int, default = 10, help = '')

    parser.add_argument('--mid_dim_1', type = int, default = 250, help = '')
    parser.add_argument('--mid_dim_2', type = int, default = 100, help = '')

    # For Train
    parser.add_argument('--module', type = int, choices = range(1, 7), default = 1, help = '')

    parser.add_argument('--batch_size', type = int, default = 64, help = '')
    parser.add_argument('--num_epochs', type = int, default = 10, help = '')

    return parser

if __name__ == '__main__':
    parser = get_parser()
    option = parser.parse_args()
