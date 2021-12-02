import os
import subprocess

os.environ['PYTHONIOENCODING'] = 'utf-8'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

seed = 77

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True

from utils.parser import get_config
from utils.logger import get_logger

config = get_config()

root_path = 'result'

logs_folder = os.path.join(root_path, 'logs', config.name)
save_folder = os.path.join(root_path, 'save', config.name)
sample_folder = os.path.join(root_path, 'sample', config.name)
result_folder = os.path.join(root_path, 'result', config.name)

subprocess.run('mkdir -p %s' % logs_folder, shell = True)
subprocess.run('mkdir -p %s' % save_folder, shell = True)
subprocess.run('mkdir -p %s' % sample_folder, shell = True)
subprocess.run('mkdir -p %s' % result_folder, shell = True)

logs_path = os.path.join(logs_folder, 'main.log' )
save_path = os.path.join(save_folder, 'best.ckpt')

logger = get_logger(config.name, logs_path)

from loaders.loader1 import get_loader as get_loader1
from loaders.loader2 import get_loader as get_loader2

from modules.module1 import get_module as get_module1
from modules.module2 import get_module as get_module2
from modules.module3 import get_module as get_module3
from modules.module4 import get_module as get_module4

from utils.misc import train, valid, save_checkpoint, load_checkpoint, save_sample

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

logger.info('prepare loader')

if config.loader == 1:
    train_loader, valid_loader, test_loader = get_loader1(config)
if config.loader == 2:
    train_loader, valid_loader, test_loader = get_loader2(config)

logger.info('prepare module')

if config.module == 1:
    module = get_module1(config).to(device)
if config.module == 2:
    module = get_module2(config).to(device)
if config.module == 3:
    module = get_module3(config).to(device)
if config.module == 4:
    module = get_module4(config).to(device)

logger.info('prepare envs')

optimizer = optim.Adam(module.parameters())
criterion = nn.CrossEntropyLoss()

logger.info('start training!')

best_valid_loss = float('inf')
for epoch in range(config.num_epochs):
    train_info = train(module, train_loader, criterion, optimizer, device)
    valid_info = valid(module, valid_loader, criterion, optimizer, device)
    logger.info(
        '[Epoch %d] Train Loss: %f, Valid Loss: %f, Valid Macro F1: %f, Valid Micro F1: %f' %
        (epoch, train_info['loss'], valid_info['loss'], valid_info['macro_f1'], valid_info['micro_f1'])
    )
    if  best_valid_loss > valid_info['loss']:
        best_valid_loss = valid_info['loss']
        save_checkpoint(save_path, module, optimizer, epoch)
        save_sample(sample_folder, valid_info['true_fold'], valid_info['prob_fold'], valid_info['pred_fold'])

logger.info('start testing!')

load_checkpoint(save_path, module, optimizer)

test_info = valid(module, test_loader, criterion, optimizer, device)

logger.info(
    'Test Loss: %f, Test Macro F1: %f, Test Micro F1: %f' %
    (test_info['loss'], test_info['macro_f1'], test_info['micro_f1'])
)

save_sample(result_folder, test_info['true_fold'], test_info['prob_fold'], test_info['pred_fold'])
