import os

from PIL import Image
import numpy as np

import torch
import torchvision.transforms as transforms

def read_file(file_path):
    image_ids = []
    label_ids = []
    with open(file_path, 'r', encoding = 'utf-8') as tsv_file:
        next (tsv_file)
        for line in tsv_file:
            image_id, label_id = line.split('\t')
            image_id = image_id.strip()
            label_id = label_id.strip()
            image_ids.append(image_id)
            label_ids.append(label_id)
    assert len(image_ids) == len(label_ids)
    return image_ids, label_ids

def read_images(image_folder, image_ids, is_train = True, mean_value = None, std_value = None):
    if not is_train:
        assert mean_value is not None
        assert std_value  is not None
    image_pths = [os.path.join(image_folder, image_id + '.png') for image_id in image_ids]
    image_objs = [Image.open(image_pth) for image_pth in image_pths]
    image_nums = [np.asarray(image_obj) for image_obj in image_objs]
    if is_train:
        image_mean = np.array(image_nums).mean() / 255
        image_std  = np.array(image_nums).std () / 255
        image_transform = transforms.Compose([
            transforms.RandomRotation(5, fill = (0,)),
            transforms.RandomCrop(28, padding = 2),
            transforms.ToTensor(),
            transforms.Normalize(mean = [image_mean], std = [image_std])
        ])
        images = [image_transform(image_obj) for image_obj in image_objs]
        return images, image_mean, image_std
    else:
        image_mean = mean_value
        image_std  = std_value
        image_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean = [image_mean], std = [image_std])
        ])
        images = [image_transform(image_obj) for image_obj in image_objs]
        return images

def get_loader(option):

    train_path = os.path.join(option.targets_path, option.train_file)
    valid_path = os.path.join(option.targets_path, option.valid_file)
    test_path  = os.path.join(option.targets_path, option.test_file )

    image_folder = os.path.join(option.targets_path, option.image_folder)

    train_image_ids, train_labels = read_file(train_path)
    valid_image_ids, valid_labels = read_file(valid_path)
    test_image_ids , test_labels  = read_file(test_path )

    train_images , mean, std = read_images(image_folder, train_image_ids, True)
    valid_images = read_images(image_folder, valid_image_ids, False, mean, std)
    test_images  = read_images(image_folder, test_image_ids , False, mean, std)

    train_images_tensor = torch.stack(train_images)
    valid_images_tensor = torch.stack(valid_images)
    test_images_tensor  = torch.stack(test_images )

    train_labels_tensor = torch.tensor([int(label) for label in train_labels])
    valid_labels_tensor = torch.tensor([int(label) for label in valid_labels])
    test_labels_tensor  = torch.tensor([int(label) for label in test_labels ])

    train_dataset = torch.utils.data.TensorDataset(train_images_tensor, train_labels_tensor)
    valid_dataset = torch.utils.data.TensorDataset(valid_images_tensor, valid_labels_tensor)
    test_dataset  = torch.utils.data.TensorDataset(test_images_tensor , test_labels_tensor )

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size = option.batch_size, shuffle = True )
    valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size = option.batch_size, shuffle = False)
    test_dataloader  = torch.utils.data.DataLoader(test_dataset , batch_size = option.batch_size, shuffle = False)

    return train_dataloader, valid_dataloader, test_dataloader

if __name__ == '__main__':
    from utils.parser import get_parser

    parser = get_parser()
    option = parser.parse_args()

    train_loader, valid_loader, test_loader = get_loader(option)

    print(len(train_loader.dataset)) # 54000
    print(len(valid_loader.dataset)) # 6000
    print(len(test_loader .dataset)) # 10000

    for mini_batch in train_loader:
        images, labels = mini_batch
        print(images.shape) # (batch_size, n_channels, img_width, img_height)
        print(labels.shape) # (batch_size)
        break
