import os
import tqdm
import numpy as np
import torch

def evaluate(true , pred):
    true_min, true_max = true.min(), true.max()
    pred_min, pred_max = pred.min(), pred.max()
    matrix = np.zeros((true_max - true_min + 1, pred_max - pred_min + 1))
    for i, j in zip(true , pred):
        matrix[i - true_min][j - pred_min] += 1

    TP = np.diag(matrix)
    FP = np.sum(matrix, axis = 0) - TP
    FN = np.sum(matrix, axis = 1) - TP
    TN = np.sum(matrix) - (FP + FN + TP)

    with np.errstate(divide = 'ignore', invalid = 'ignore'):
        p_i = TP / (TP + FP)
        r_i = TP / (TP + FN)
        f_i = 2 * p_i * r_i / (p_i + r_i)

    np.nan_to_num(p_i, copy = False, nan = 0)
    np.nan_to_num(r_i, copy = False, nan = 0)
    np.nan_to_num(f_i, copy = False, nan = 0)

    p_macro = np.sum(p_i) / len(p_i)
    r_macro = np.sum(r_i) / len(r_i)
    f_macro = 2 * p_macro * r_macro / (p_macro + r_macro)

    p_micro = np.sum(TP) / (np.sum(TP) + np.sum(FP))
    r_micro = np.sum(TP) / (np.sum(TP) + np.sum(FN))
    f_micro = 2 * p_micro * r_micro / (p_micro + r_micro)

    return f_i, f_macro, f_micro

def save_checkpoint(save_path, model, optim, epoch):
    checkpoint = {
        'model': model.state_dict(),
        'optim': optim.state_dict(),
        'epoch': epoch
    }
    torch.save(checkpoint, save_path)

def load_checkpoint(load_path, model, optim):
    checkpoint = torch.load(load_path)
    model.load_state_dict(checkpoint['model'])
    optim.load_state_dict(checkpoint['optim'])
    return checkpoint['epoch']

def save_sample(folder, true_label, prob_label, pred_label):
    true_label_path = os.path.join(folder, 'true_label.txt')
    prob_label_path = os.path.join(folder, 'prob_label.txt')
    pred_label_path = os.path.join(folder, 'pred_label.txt')
    np.savetxt(true_label_path, true_label)
    np.savetxt(prob_label_path, prob_label)
    np.savetxt(pred_label_path, pred_label)

def train(module_id, module, loader, criterion, optimizer, device):
    module.train()
    epoch_loss = 0
    for mini_batch in tqdm.tqdm(loader):
        mini_batch = [data_item.to(device) for data_item in mini_batch]
        images, true_labels = mini_batch
        pred_labels = module(images)
        loss = criterion(pred_labels, true_labels)
        epoch_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return {
        'loss': epoch_loss / len(loader)
    }

def valid(module_id, module, loader, criterion, optimizer, device):
    module.eval()
    epoch_loss = 0
    prob_fold = []
    true_fold = []
    with torch.no_grad():
        for mini_batch in tqdm.tqdm(loader):
            mini_batch = [data_item.to(device) for data_item in mini_batch]
            images, true_labels = mini_batch
            pred_labels = module(images)
            loss = criterion(pred_labels, true_labels)
            epoch_loss += loss.item()
            prob_fold.append(pred_labels)
            true_fold.append(true_labels)
    true_fold = torch.cat(true_fold).cpu().numpy()
    prob_fold = torch.cat(prob_fold).softmax(dim = -1).cpu().numpy()
    pred_fold = prob_fold.argmax(axis = 1)
    info = evaluate(true_fold, pred_fold)[0]
    return {
        'loss': epoch_loss / len(loader),
        'macro_f1': info[1],
        'micro_f1': info[2],
        'true_fold': true_fold,
        'prob_fold': prob_fold,
        'pred_fold': pred_fold
    }
