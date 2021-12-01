import os
import subprocess

import pickle

import numpy as np
from PIL import Image

from utils.parser import get_config

config = get_config()

subprocess.run('mkdir -p %s' % config.sources_path, shell = True)
subprocess.run('mkdir -p %s' % config.targets_path, shell = True)

subprocess.run('mkdir -p %s' % os.path.join(config.targets_path, config.image_folder), shell = True)

# The official website is https://www.cs.toronto.edu/~kriz/cifar.html

onweb_path = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'

ziped_path = os.path.join(config.sources_path, 'cifar-10-python.tar.gz')
unzip_path = os.path.join(config.sources_path, 'cifar-10-batches-py')

if not os.path.exists(ziped_path):
    os.system('wget %s -O %s' % (onweb_path, ziped_path))

if not os.path.exists(unzip_path):
    os.system('mkdir %s' % (unzip_path))
    os.system('tar -zxvf %s -C %s --strip-components 1' % (ziped_path, unzip_path))

meta_data_path = os.path.join(unzip_path, 'batches.meta')

origin_train_paths = [
    os.path.join(unzip_path, 'data_batch_1'),
    os.path.join(unzip_path, 'data_batch_2'),
    os.path.join(unzip_path, 'data_batch_3'),
    os.path.join(unzip_path, 'data_batch_4'),
    os.path.join(unzip_path, 'data_batch_5')
]
origin_valid_paths = [
    os.path.join(unzip_path, 'test_batch')
]

with open(meta_data_path, 'rb') as pickle_file:
    data = pickle.load(pickle_file, encoding = 'latin1')
    batch_n_image = data['num_cases_per_batch']
    image_n_pixel = data['num_vis']

def read_data_file(
    file_path,
    batch_n_image,
    image_n_pixel,
    image_channels,
    image_height,
    image_width
):
    assert image_n_pixel == image_channels * image_height * image_width

    with open(file_path, 'rb') as pickle_file:
        data = pickle.load(pickle_file, encoding = 'latin1')

    return (
        data['data'].reshape(batch_n_image, image_channels, image_height, image_width).transpose(0, 2, 3, 1),
        np.array(data['labels'])
    )

train_datas = [read_data_file(
    file_path,
    batch_n_image,
    image_n_pixel,
    config.image_channels,
    config.image_height,
    config.image_width
) for file_path in origin_train_paths]

train_image = np.concatenate([datas[0] for datas in train_datas])
train_label = np.concatenate([datas[1] for datas in train_datas])

valid_datas = [read_data_file(
    file_path,
    batch_n_image,
    image_n_pixel,
    config.image_channels,
    config.image_height,
    config.image_width
) for file_path in origin_valid_paths]

valid_image = np.concatenate([datas[0] for datas in valid_datas])
valid_label = np.concatenate([datas[1] for datas in valid_datas])

assert len(train_image) == len(train_label)
assert len(valid_image) == len(valid_label)

radio = 0.9

split_train_image = train_image[:int(len(train_image) * radio)]
split_train_label = train_label[:int(len(train_label) * radio)]

split_valid_image = train_image[int(len(train_image) * radio):]
split_valid_label = train_label[int(len(train_label) * radio):]

split_test_image = valid_image
split_test_label = valid_label

image_datas = [split_train_image, split_valid_image, split_test_image]
label_datas = [split_train_label, split_valid_label, split_test_label]

train_path = os.path.join(config.targets_path, config.train_file)
valid_path = os.path.join(config.targets_path, config.valid_file)
test_path  = os.path.join(config.targets_path, config.test_file )

file_paths = [train_path, valid_path, test_path]

image_folder = os.path.join(config.targets_path, config.image_folder)

counter = 0
for file_path, images, labels in zip(file_paths, image_datas, label_datas):
    with open(file_path, 'w', encoding = 'utf-8') as tsv_file:
        tsv_file.write('image_id' + '\t' + 'label_id' + '\n')
        for image, label in zip(images, labels):
            counter += 1
            image_id = str(counter).zfill(7)
            label_id = str(label)
            tsv_file.write(image_id + '\t' + label_id + '\n')
            image_obj = Image.fromarray(np.uint8(image))
            image_obj.save(os.path.join(image_folder, image_id + '.png'))
