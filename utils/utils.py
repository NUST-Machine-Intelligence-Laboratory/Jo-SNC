import os
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import torch.backends.cudnn as cudnn
import numpy as np
import random
import math
import matplotlib.pyplot as plt
from json import dump
from tqdm import tqdm


# Helper Functions -----------------------------------------------------------------------
def set_seed(seed=0):
    random.seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_checkpoint(state, filename='checkpoint.pth'):
    torch.save(state, filename)


def get_smoothed_label_distribution(labels, num_class, epsilon):
    smoothed_label = torch.full(size=(labels.size(0), num_class), fill_value=epsilon / (num_class - 1))
    smoothed_label.scatter_(dim=1, index=torch.unsqueeze(labels, dim=1).cpu(), value=1 - epsilon)
    return smoothed_label.to(labels.device)


def init_weights(module, init_method='He'):
    for _, m in module.named_modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            if init_method == 'He':
                nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None: nn.init.constant_(m.bias.data, val=0)
            elif init_method == 'Xavier':
                nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None: nn.init.constant_(m.bias.data, val=0)
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


def freeze_layer(module):
    for parameters in module.parameters():
        parameters.requires_grad = False


def unfreeze_layer(module):
    for parameters in module.parameters():
        parameters.requires_grad = True


def adjust_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def adjust_beta(optimizer, beta):
    for param_group in optimizer.param_groups:
        if beta is not None:
            param_group['betas'] = (beta, 0.999)


def adjust_weight_decay(optimizer, weight_decay):
    for param_group in optimizer.param_groups:
        if weight_decay is not None:
            param_group['weight_decay'] = weight_decay


def make_linear_values(init_v, last, end_v):
    return list(np.linspace(init_v, end_v, last))


def make_linear_lr(init_lr, last, end_lr):
    return list(np.linspace(init_lr, end_lr, last))


def make_cosine_lr(init_lr, last, end_lr=0.0):
    assert end_lr < init_lr
    lrs = [init_lr] * last
    for i in range(last):
        lrs[i] = (init_lr - end_lr) * 0.5 * (1 + math.cos(i / last * math.pi)) + end_lr
    return lrs


def build_lr_plan(lr, total_epochs, warmup_epochs, warmup_lr=0.1, decay='linear', warmup_rampup=False):
    if warmup_rampup:
        warmup_lr_plan = make_linear_lr(warmup_lr * 1e-5, warmup_epochs, warmup_lr)
    else:
        warmup_lr_plan = [warmup_lr] * warmup_epochs
    if decay.startswith('step:'):   # example:  step:10,20,30,40 / step: 10,20,30,40,0.1
        lr_plan = [lr] * total_epochs
        ele = decay.split(':')[1].split(',')
        step = [int(i) for i in ele[:-1]]
        last_ele = float(ele[-1])
        if last_ele >= 1.0:
            step.append(int(last_ele))
            step_factor = 0.1
        else:
            step_factor = last_ele
        for ep in range(total_epochs):
            decay_factor = 1.0
            for i in range(len(step)):
                decay_factor *= (step_factor ** int(ep >= step[i]))
            lr_plan[ep] = lr * decay_factor
        lr_plan[:warmup_epochs] = warmup_lr_plan
        return lr_plan
    elif decay.startswith('linear:'):  # example:  linear:10 / linear:10,1e-6 / linear:10,1e-6,60
        lr_plan = warmup_lr_plan
        ele = decay.split(':')[1].split(',')
        lr_decay_start = int(ele[0])
        end_lr = float(ele[1]) if len(ele) == 2 else lr * 0.00001
        lr_decay_end = int(ele[2]) if len(ele) == 3 and int(ele[2]) > lr_decay_start else total_epochs
        lr_plan += [lr] * (lr_decay_start - warmup_epochs)
        lr_plan += make_linear_lr(lr, lr_decay_end - lr_decay_start, end_lr)
        lr_last = lr_plan[-1]
        lr_plan += [lr_last] * (total_epochs - lr_decay_end)
        return lr_plan
    elif decay.startswith('cosine:'):  # example:  cosine:10 / cosine:10,1e-6 / cosine:10,1e-6,60
        lr_plan = warmup_lr_plan
        ele = decay.split(':')[1].split(',')
        lr_decay_start = int(ele[0])
        end_lr = float(ele[1]) if len(ele) >= 2 else 0.0
        lr_decay_end = int(ele[2]) if len(ele) == 3 and int(ele[2]) > lr_decay_start else total_epochs
        lr_plan += [lr] * (lr_decay_start - warmup_epochs)
        lr_plan += make_cosine_lr(lr, lr_decay_end - lr_decay_start, end_lr)
        lr_last = lr_plan[-1]
        lr_plan += [lr_last] * (total_epochs - lr_decay_end)
        return lr_plan
    else:
        raise AssertionError(f'lr decay method: {decay} is not implemented yet.')


def check_nan(tensor):
    if isinstance(tensor, torch.Tensor):
        return torch.isnan(tensor).any()
    else:
        return np.isnan(tensor).any()


def check_inf(tensor):
    if isinstance(tensor, torch.Tensor):
        return torch.isinf(tensor).any()
    else:
        return np.isinf(tensor).any()


def check_nan_inf(tensor):
    return check_nan(tensor) or check_inf(tensor)


def linear_rampup(current, rampup_length):
    assert current >= 0 and rampup_length >= 0
    if current >= rampup_length:
        return 1.0
    else:
        return current / rampup_length


def approx_dist(p, delta=1e-8):
    bs, nc = p.shape
    assert torch.abs(p.sum() - bs) < 1e-3, f'before approx: |{p.sum()}-{bs}| = {torch.abs(p.sum() - bs)} >=1e-3'
    p += delta
    p[torch.arange(bs), p.argmax(dim=1)] -= delta * nc
    # assert torch.abs(p.sum() - bs) < 1e-3, f'after approx : |{p.sum()}-{bs}| = torch.abs(p.sum() - bs) >=1e-3'
    return p


def kl_div(p, q, base=2):
    # p, q is in shape (batch_size, n_classes)
    if base == 2:
        return (p * (p+1e-6).log2() - p * (q+1e-6).log2()).sum(dim=1)
    else:
        return (p * (p+1e-6).log() - p * (q+1e-6).log()).sum(dim=1)


def symmetric_kl_div(p, q, base=2):
    return kl_div(p, q, base) + kl_div(q, p, base)


def js_div(p, q, base=2):
    # Jensen-Shannon divergence, value is in (0, 1)
    m = 0.5 * (p + q)
    return 0.5 * kl_div(p, m, base) + 0.5 * kl_div(q, m, base)


def entropy(p):
    return Categorical(probs=p).entropy()


def set_union_1d(tensor1, tensor2):
    return torch.cat((tensor1, tensor2), dim=0).unique()


def set_diff_1d(tensor1, tensor2, assume_unique=False):
    if not assume_unique:
        tensor1 = torch.unique(tensor1)
        tensor2 = torch.unique(tensor2)
    return tensor1[(tensor1[:, None] != tensor2).all(dim=1)]


def set_overlap_1d(tensor1, tensor2, assume_unique=False):
    if not assume_unique:
        tensor1 = torch.unique(tensor1)
        tensor2 = torch.unique(tensor2)
    temp = torch.cat((tensor1, tensor2), dim=0)
    utemp, count = temp.unique(return_counts=True)
    return utemp[torch.where(count.gt(1))[0]]


def indices_list_to_indicator_vector(indices_list, num_samples):
    assert isinstance(indices_list, list)
    indicator_vector = torch.zeros(num_samples).int()
    try:
        indicator_vector[indices_list] = 1
    except:
        print(len(indices_list), indicator_vector.shape)
        print(indices_list)
        raise AssertionError()
    return indicator_vector


def get_stats(result_file):
    with open(result_file, 'r') as f:
        lines = f.readlines()
    test_acc_list = []
    test_acc_list2 = []
    valid_epoch = []
    # valid_epoch = [191, 192, 193, 194, 195, 196, 197, 198, 199, 200]
    for idx in range(1, 11):
        line = lines[-idx].strip()
        if 'test loss' in line:
            epoch, train_loss, train_acc, test_loss, test_acc = line.split(' | ')[:5]
        else:
            epoch, train_loss, train_acc, test_acc = line.split(' | ')[:4]
        ep = int(epoch.split(': ')[1])
        valid_epoch.append(ep)
        # assert ep in valid_epoch, ep
        if '/' not in test_acc:
            test_acc_list.append(float(test_acc.split(': ')[1]))
        else:
            test_acc1, test_acc2 = map(lambda x: float(x), test_acc.split(': ')[1].lstrip('(').rstrip(')').split('/'))
            test_acc_list.append(test_acc1)
            test_acc_list2.append(test_acc2)
    if len(test_acc_list2) == 0:
        test_acc_list = np.array(test_acc_list)
        print(valid_epoch)
        print(f'mean: {test_acc_list.mean():.2f}, std: {test_acc_list.std():.2f}')
        return {'mean': test_acc_list.mean(), 'std': test_acc_list.std(), 'valid_epoch': valid_epoch}
    else:
        test_acc_list = np.array(test_acc_list)
        test_acc_list2 = np.array(test_acc_list2)
        print(valid_epoch)
        print(f'mean: {test_acc_list.mean():.2f}, std: {test_acc_list.std():.2f}')
        print(f'mean: {test_acc_list2.mean():.2f}, std: {test_acc_list2.std():.2f}')
        return {'mean1': test_acc_list.mean(), 'std1': test_acc_list.std(),
                'mean2': test_acc_list2.mean(), 'std2': test_acc_list2.std(),
                'valid_epoch': valid_epoch}


def plot_results(result_file):
    fig, ax = plt.subplots(1, 3, figsize=(12, 6), tight_layout=True)
    ax = ax.ravel()
    metrics = ['Train Loss', 'Train Accuracy', 'Test Accuracy']
    with open(result_file, 'r') as f:
        lines = f.readlines()
    epoch_list = []
    train_loss_list = []
    train_acc_list = []
    test_acc_list = []
    best_epoch, best_acc = 0, 0.0
    for line in lines:
        if not line.startswith('epoch'):
            continue
        line = line.strip()
        epoch, train_loss, train_acc, test_acc = line.split(' | ')[:4]
        epoch_list.append(int(epoch.split(': ')[1]))
        train_loss_list.append(float(train_loss.split(': ')[1]))
        train_acc_list.append(float(train_acc.split(': ')[1]))
        test_acc_list.append(float(test_acc.split(': ')[1]))
        best_accuracy_epoch = line.split(' | ')[-1]
        best_acc = float(best_accuracy_epoch.split(' @ ')[0].split(': ')[1])
        best_epoch = int(best_accuracy_epoch.split(' @ ')[1].split(': ')[1])
    results = [
        np.array(train_loss_list),
        np.array(train_acc_list),
        np.array(test_acc_list)
    ]

    for i in range(len(metrics)):
        ax[i].plot(epoch_list, results[i], '-', label=metrics[i], linewidth=2)
        ax[i].set_title(metrics[i])
        ax[i].set_xlim(0, np.max(epoch_list))
        # ax[i].legend()
    ax[2].hlines(best_acc, 0, np.max(epoch_list), colors='red', linestyles='dashed')
    ax[2].plot(best_epoch, best_acc, 'ro')
    ax[2].annotate(f'({best_epoch}, {best_acc:.2f}%)', xy=(best_epoch, best_acc), xytext=(-30, -15), textcoords='offset points', color='red')

    result_dir = result_file.rsplit('/', 1)[0]
    fig.savefig(f'{result_dir}/loss_acc_curve.png', dpi=300)
    plt.close(fig)


def plot_precision_recall(result_file):
    fig, ax = plt.subplots(1, 4, figsize=(12, 6), tight_layout=True)
    ax = ax.ravel()
    ntype = ['Clean', 'ID Noise', 'OOD Noise']
    metrics = ['Precision', 'Recall', 'F1 Score', 'AUROC']
    colors = ['g', 'b', 'r', 'orange']
    with open(result_file, 'r') as f:
        lines = f.readlines()
    epoch_list = []
    p_clean, p_id, p_ood = [], [], []
    r_clean, r_id, r_ood = [], [], []
    f_clean, f_id, f_ood = [], [], []
    a_clean, a_id, a_ood = [], [], []
    for line in lines:
        if not line.startswith('epoch'):
            continue
        line = line.strip()
        epoch, clean, id, ood = line.split(' | ')[:4]
        epoch_list.append(int(epoch.split(': ')[1]))
        clean_prf = list(map(lambda x: float(x), clean.split(': ')[1].split('/')[1:]))  # [P, R, F1, AUROC]
        id_prf = list(map(lambda x: float(x), id.split(': ')[1].split('/')[1:]))        # [P, R, F1, AUROC]
        ood_prf = list(map(lambda x: float(x), ood.split(': ')[1].split('/')[1:]))      # [P, R, F1, AUROC]
        p_clean.append(clean_prf[0])
        r_clean.append(clean_prf[1])
        f_clean.append(clean_prf[2])
        a_clean.append(clean_prf[3])
        p_id.append(id_prf[0])
        r_id.append(id_prf[1])
        f_id.append(id_prf[2])
        a_id.append(id_prf[3])
        p_ood.append(ood_prf[0])
        r_ood.append(ood_prf[1])
        f_ood.append(ood_prf[2])
        a_ood.append(ood_prf[3])
    results = [
        [np.array(p_clean), np.array(r_clean), np.array(f_clean), np.array(a_clean)],
        [np.array(p_id), np.array(r_id), np.array(f_id), np.array(a_id)],
        [np.array(p_ood), np.array(r_ood), np.array(f_ood), np.array(a_ood)]
    ]
    for i in range(len(ntype)):
        for j in range(len(metrics)):
            ax[i].plot(epoch_list, results[i][j], '-', label=metrics[j], linewidth=2, color=colors[j])
            ax[i].set_title(ntype[i])
            ax[i].set_xlim(0, np.max(epoch_list))
            ax[i].legend()

    result_dir = result_file.rsplit('/', 1)[0]
    fig.savefig(f'{result_dir}/pr_curve.png', dpi=300)
    plt.close(fig)


# Helper Classes -----------------------------------------------------------------------
class AverageMeter(object):
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / (self.count+1e-8)


class MultiDataTransform(object):
    def __init__(self, transforms):
        assert isinstance(transforms, list) and len(transforms) > 0
        if len(transforms) == 1:
            self.transforms = [transforms[0], transforms[0]]
        else:
            self.transforms = transforms
        self.n_aug = len(self.transforms)

    def __call__(self, sample):
        views = [self.transforms[i](sample) for i in range(self.n_aug)]
        return views


class EMA(object):
    """
    https://github.com/wvangansbeke/Unsupervised-Classification/blob/master/utils/ema.py
    Usage:
        model = ResNet(config)
        ema = EMA(model, alpha=0.999)
        ... # train an epoch
        ema.update_params(model)
        ema.apply_shadow(model)
    """
    def __init__(self, model, alpha=0.999):
        self.shadow = {k: v.clone().detach() for k, v in model.state_dict().items()}
        self.param_keys = [k for k, _ in model.named_parameters()]
        self.alpha = alpha

    def init_params(self, model):
        self.shadow = {k: v.clone().detach() for k, v in model.state_dict().items()}
        self.param_keys = [k for k, _ in model.named_parameters()]

    def update_params(self, model):
        state = model.state_dict()
        for name in self.param_keys:
            self.shadow[name].copy_(self.alpha * self.shadow[name] + (1 - self.alpha) * state[name])

    def apply_shadow(self, model):
        model.load_state_dict(self.shadow, strict=True)

