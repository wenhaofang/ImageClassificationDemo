import os
import subprocess

import struct

import numpy as np
from PIL import Image

from utils.parser import get_parser

parser = get_parser()
option = parser.parse_args()

subprocess.run('mkdir -p %s' % option.sources_path, shell = True)
subprocess.run('mkdir -p %s' % option.targets_path, shell = True)

subprocess.run('mkdir -p %s' % os.path.join(option.targets_path, option.image_folder), shell = True)

# The official website is http://yann.lecun.com/exdb/mnist/

onweb_paths = [
    'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz', # train set image
    'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz', # train set label
    'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz' , # valid set image
    'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz'   # valid set label
]

ziped_paths = [
    os.path.join(option.sources_path, onweb_path.split('/')[-1]) for onweb_path in onweb_paths
]

unzip_paths = [
    os.path.join(option.sources_path, onweb_path.split('/')[-1].split('.')[0]) for onweb_path in onweb_paths
]

for onweb_path, ziped_path, unzip_path in zip(onweb_paths, ziped_paths, unzip_paths):

    if not os.path.exists(ziped_path):
        os.system('wget %s -O %s' % (onweb_path, ziped_path))

    if not os.path.exists(unzip_path):
        os.system('gzip -k -d %s' % (ziped_path))
        os.system('mv %s %s' % (ziped_path.split('.')[0], unzip_path))

def read_image_file(file_path):
    imgs = []
    with open(file_path, 'rb') as bin_file:
        idx = 0
        buf = bin_file.read()
        # meta
        _, n_img, n_row, n_col = struct.unpack_from('>IIII', buf, idx)
        idx += struct.calcsize('>IIII')
        # data
        ins  = '>{}B'.format(n_row * n_col)
        for _ in range(n_img):
            img  = struct.unpack_from(ins, buf, idx)
            idx += struct.calcsize(ins)
            imgs.append(img)
    imgs = [np.array(img).reshape(n_row, n_col) for img in imgs]
    return imgs

def read_label_file(file_path):
    labs = []
    with open(file_path, 'rb') as bin_file:
        idx = 0
        buf = bin_file.read()
        # meta
        _, n_img = struct.unpack_from('>II', buf, idx)
        idx += struct.calcsize('>II')
        # data
        ins  = '>{}B'.format(1)
        for _ in range(n_img):
            lab  = struct.unpack_from(ins, buf, idx)
            idx += struct.calcsize(ins)
            labs.append(lab)
    labs = [lab[0] for lab in labs]
    return labs

train_image = read_image_file(unzip_paths[0])
train_label = read_label_file(unzip_paths[1])

valid_image = read_image_file(unzip_paths[2])
valid_label = read_label_file(unzip_paths[3])

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

train_path = os.path.join(option.targets_path, option.train_file)
valid_path = os.path.join(option.targets_path, option.valid_file)
test_path  = os.path.join(option.targets_path, option.test_file )

file_paths = [train_path, valid_path, test_path]

image_folder = os.path.join(option.targets_path, option.image_folder)

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
