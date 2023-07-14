# -*- coding: utf-8 -*-
# ================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   @File        : loss.py
#   @Author      : Zeren Sun
#   @Created date: 2023/3/17 09:28
#   @Description :
#
# ================================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.utils import kl_div, check_nan_inf


def pairwise_kl_loss_deprecated(logits, memory_logits, knn_indices, knn_similarities, temperature, sample_weights, epsilon=1e-8):
    bs, nc = logits.shape
    k = knn_indices.shape[1]
    knn_logits = memory_logits[knn_indices.view(-1)].view([bs, k, nc])            # (bs, k, nc)
    t_softmax_temp = F.softmax(knn_logits / temperature, dim=-1) + epsilon        # (bs, k, nc)
    s_softmax_temp = F.log_softmax(logits / temperature, dim=-1)                  # (bs, nc)
    # s_softmax_temp = F.softmax(logits / temperature, dim=-1)                    # (bs, nc)

    normalized_sim = knn_similarities / knn_similarities.sum(dim=-1).view(-1, 1)  # (bs, k)
    weighted_t_softmax = torch.squeeze(torch.matmul(normalized_sim.unsqueeze(dim=1), t_softmax_temp))  # (bs, nc)
    kldiv_loss_per_pair = weighted_t_softmax * (torch.log(weighted_t_softmax+1e-8) - s_softmax_temp)        # (bs, nc)
    # kldiv_loss_per_pair = s_softmax_temp * (torch.log(s_softmax_temp+1e-8) - torch.log(weighted_t_softmax))        # (bs, nc)
    kldiv_loss_per_sample = (np.power(temperature, 2) * torch.sum(kldiv_loss_per_pair, 1))             # (bs, 1 )
    if sample_weights is not None:
        normalization = sample_weights.sum()
    else:
        normalization = bs
    return kldiv_loss_per_sample.sum() / (normalization + epsilon)


def pairwise_kl_loss(logits, memory_logits, knn_indices, knn_similarities, temperature, epsilon=1e-8, knn_merge_method='sim'):
    bs, nc = logits.shape
    k = knn_indices.shape[1]
    knn_logits = memory_logits[knn_indices.view(-1)].view([bs, k, nc])            # (bs, k, nc)
    s_softmax_temp = F.softmax(logits / temperature, dim=-1)                      # (bs, nc)
    t_softmax_temp = F.softmax(knn_logits / temperature, dim=-1)                  # (bs, k, nc)
    if knn_merge_method == 'sim':
        normalized_sim = knn_similarities / (knn_similarities.sum(dim=-1).view(-1, 1) + epsilon)  # (bs, k)
    else:
        knn_temp = torch.ones_like(knn_similarities)
        normalized_sim = knn_temp / (knn_temp.sum(dim=-1).view(-1, 1) + epsilon)  # (bs, k)
    weighted_t_softmax = torch.squeeze(torch.matmul(normalized_sim.unsqueeze(dim=1), t_softmax_temp))  # (bs, nc)
    kldiv_loss_per_sample = kl_div(s_softmax_temp, weighted_t_softmax)

    if check_nan_inf(kldiv_loss_per_sample):
        print(f'{kldiv_loss_per_sample}')
        print(f'{check_nan_inf(s_softmax_temp)}')
        print(f'{check_nan_inf(t_softmax_temp)}')
        print(f'{check_nan_inf(weighted_t_softmax)}')
        print(f's->{torch.where(s_softmax_temp==0)}')
        print(f't->{torch.where(weighted_t_softmax==0)}')
        print(f'{check_nan_inf(knn_similarities)}')
        raise AssertionError()

    kldiv_loss_per_sample = np.power(temperature, 2) * kldiv_loss_per_sample
    return kldiv_loss_per_sample.mean()


def pairwise_ce_loss(logits, memory_logits, knn_indices, knn_similarities, temperature, epsilon=1e-8, knn_merge_method='sim'):
    bs, nc = logits.shape
    k = knn_indices.shape[1]
    knn_logits = memory_logits[knn_indices.view(-1)].view([bs, k, nc])  # (bs, k, nc)
    s_softmax_temp = F.softmax(logits / temperature, dim=-1)  # (bs, nc)
    t_softmax_temp = F.softmax(knn_logits / temperature, dim=-1)  # (bs, k, nc)
    if knn_merge_method == 'sim':
        normalized_sim = knn_similarities / (knn_similarities.sum(dim=-1).view(-1, 1) + epsilon)  # (bs, k)
    else:
        knn_temp = torch.ones_like(knn_similarities)
        normalized_sim = knn_temp / (knn_temp.sum(dim=-1).view(-1, 1) + epsilon)  # (bs, k)
    weighted_t_softmax = torch.squeeze(torch.matmul(normalized_sim.unsqueeze(dim=1), t_softmax_temp))  # (bs, nc)
    ce_loss_per_sample = - torch.sum(torch.log(s_softmax_temp+epsilon) * weighted_t_softmax, dim=1)  # (bs, )

    if check_nan_inf(ce_loss_per_sample):
        print(f'{ce_loss_per_sample}')
        print(f'{check_nan_inf(weighted_t_softmax)}')
        print(f'{check_nan_inf(normalized_sim)}')
        print(f'{check_nan_inf(knn_similarities)}')
        print(f'{knn_similarities}')
        raise AssertionError()

    # ce_loss_per_sample = np.power(temperature, 2) * ce_loss_per_sample
    return ce_loss_per_sample.mean()


def ncr_loss(logits, features, memory_logits, memory_features, number_neighbors=10, smoothing_gamma=0.3, temperature=2.0,
             zero_negative_similarity=True, loss_func='kldiv'):
    # features & memory features should have been l2 normalized
    # NCR code implementation is modified from https://github.com/google-research/scenic/blob/main/scenic/projects/ncr/loss.py

    similarity = torch.mm(features, memory_features.t())  # (batch_size, len_memory_pool)
    if zero_negative_similarity:
        similarity = F.relu(similarity, inplace=False)
    neighbor_similarities, neighbor_indices = similarity.topk(number_neighbors+1, dim=1, largest=True, sorted=True)
    # Remove itself from obtained nearest neighbours.
    neighbor_indices = neighbor_indices[:, 1:].contiguous()              # (batch_size, number_neighbors)
    neighbor_similarities = neighbor_similarities[:, 1:].contiguous()    # (batch_size, number_neighbors)
    if zero_negative_similarity:
        neighbor_similarities = torch.pow(neighbor_similarities, smoothing_gamma)
    if loss_func == 'kldiv':
        loss = pairwise_kl_loss(logits, memory_logits, neighbor_indices, neighbor_similarities, temperature)
    elif loss_func == 'kldiv_mean':
        loss = pairwise_kl_loss(logits, memory_logits, neighbor_indices, neighbor_similarities, temperature, knn_merge_method='mean')
    elif loss_func == 'ce':
        loss = pairwise_ce_loss(logits, memory_logits, neighbor_indices, neighbor_similarities, temperature)
    elif loss_func == 'ce_mean':
        loss = pairwise_ce_loss(logits, memory_logits, neighbor_indices, neighbor_similarities, temperature, knn_merge_method='mean')
    else:
        raise AssertionError('loss_func is not valid')
    return loss


def neg_entropy_loss(logits, reduction='mean'):
    logits = logits.clamp(min=1e-12)
    probs = torch.softmax(logits, dim=1)
    losses = torch.sum(probs.log() * probs, dim=1)
    if reduction == 'none':
        return losses
    elif reduction == 'mean':
        return torch.mean(losses)
    elif reduction == 'sum':
        return torch.sum(losses)
    else:
        raise AssertionError('reduction has to be none, mean or sum')


def entropy_loss(logits):
    return - neg_entropy_loss(logits)


def negative_cross_entropy_loss(logits, false_labels, reduction='mean'):
    N, C = logits.shape
    probs = F.softmax(logits, dim=1)
    neg_probs = 1 - probs
    neg_probs.clamp(min=1e-12)
    if false_labels.dim() == 1:
        labels = F.one_hot(false_labels, C)
    elif false_labels.dim() == 2:
        labels = false_labels
    else:
        raise AssertionError(f'dim of false_labels is {false_labels.dim()}')
    losses = - torch.sum(torch.log(neg_probs) * labels, dim=1)
    if reduction == 'none':
        return losses
    elif reduction == 'mean':
        return torch.sum(losses) / logits.size(0)
    elif reduction == 'sum':
        return torch.sum(losses)
    else:
        raise AssertionError('reduction has to be none, mean or sum')


def reversed_cross_entropy(logits, labels, reduction='none'):
    """
    :param logits: shape: (N, C)
    :param labels: shape: (N, C)
    :param reduction: options: "none", "mean", "sum"
    :return: loss or losses
    """
    N, C = logits.shape
    assert labels.size(0) == N and labels.size(1) == C, f'label tensor shape is {labels.shape}, while logits tensor shape is {logits.shape}'
    pred = F.softmax(logits, dim=1)
    pred = torch.clamp(pred, min=1e-7, max=1.0)
    labels = torch.clamp(labels, min=1e-4, max=1.0)
    losses = -torch.sum(pred * torch.log(labels), dim=1)

    if reduction == 'none':
        return losses
    elif reduction == 'mean':
        return torch.sum(losses) / logits.size(0)
    elif reduction == 'sum':
        return torch.sum(losses)
    else:
        raise AssertionError('reduction has to be none, mean or sum')


def normalized_cross_entropy(logits, labels, reduction='none'):
    """
    :param logits: shape: (N, C)
    :param labels: shape: (N, C)
    :param reduction: options: "none", "mean", "sum"
    :return: loss or losses
    """
    N, C = logits.shape
    assert labels.size(0) == N and labels.size(1) == C, f'label tensor shape is {labels.shape}, while logits tensor shape is {logits.shape}'
    log_logits = F.log_softmax(logits, dim=1)
    losses = - torch.sum(labels * log_logits, dim=1) / (- torch.bmm(labels.unsqueeze(dim=2), log_logits.unsqueeze(dim=1)).sum(dim=(1, 2)))

    if reduction == 'none':
        return losses
    elif reduction == 'mean':
        return torch.sum(losses) / logits.size(0)
    elif reduction == 'sum':
        return torch.sum(losses)
    else:
        raise AssertionError('reduction has to be none, mean or sum')


def symmetric_cross_entropy(logits, labels, alpha, beta, reduction='none'):
    """
    :param logits: shape: (N, C)
    :param labels: shape: (N, C)
    :param reduction: options: "none", "mean", "sum"
    :return: loss or losses
    """
    N, C = logits.shape
    assert labels.size(0) == N and labels.size(1) == C, f'label tensor shape is {labels.shape}, while logits tensor shape is {logits.shape}'
    ce = F.cross_entropy(logits, labels, reduction=reduction)
    rce = reversed_cross_entropy(logits, labels, reduction=reduction)
    return alpha * ce + beta * rce


def generalized_cross_entropy(logits, labels, rho=0.7, reduction='none'):
    """
    :param logits: shape: (N, C)
    :param labels: shape: (N, C)
    :param reduction: options: "none", "mean", "sum"
    :return: loss or losses
    """
    N, C = logits.shape
    assert labels.size(0) == N and labels.size(1) == C, f'label tensor shape is {labels.shape}, while logits tensor shape is {logits.shape}'
    pred = F.softmax(logits, dim=1)
    pred = torch.clamp(pred, min=1e-7, max=1.0)
    losses = torch.sum(labels * ((1.0 - torch.pow(pred, rho)) / rho), dim=1)

    if reduction == 'none':
        return losses
    elif reduction == 'mean':
        return torch.sum(losses) / logits.size(0)
    elif reduction == 'sum':
        return torch.sum(losses)
    else:
        raise AssertionError('reduction has to be none, mean or sum')


def normalized_generalized_cross_entropy(logits, labels, rho=0.7, reduction='none'):
    """
    :param logits: shape: (N, C)
    :param labels: shape: (N, C)
    :param reduction: options: "none", "mean", "sum"
    :return: loss or losses
    """
    N, C = logits.shape
    assert labels.size(0) == N and labels.size(1) == C, f'label tensor shape is {labels.shape}, while logits tensor shape is {logits.shape}'
    pred = F.softmax(logits, dim=1)
    pred = torch.clamp(pred, min=1e-7, max=1.0)
    pred_pow = torch.pow(pred, rho)
    losses = (1 - torch.sum(labels * pred_pow, dim=1)) / (C - torch.bmm(labels.unsqueeze(dim=2), pred_pow.unsqueeze(dim=1)).sum(dim=(1, 2)))

    if reduction == 'none':
        return losses
    elif reduction == 'mean':
        return torch.sum(losses) / logits.size(0)
    elif reduction == 'sum':
        return torch.sum(losses)
    else:
        raise AssertionError('reduction has to be none, mean or sum')


def mae_loss(logits, labels, reduction='none'):
    """
    :param logits: shape: (N, C)
    :param labels: shape: (N, C)
    :param reduction: options: "none", "mean", "sum"
    :return: loss or losses
    """
    N, C = logits.shape
    assert labels.size(0) == N and labels.size(1) == C, f'label tensor shape is {labels.shape}, while logits tensor shape is {logits.shape}'
    pred = logits.softmax(dim=1)
    losses = torch.abs(pred - labels).sum(dim=1)
    if reduction == 'none':
        return losses
    elif reduction == 'mean':
        return torch.sum(losses) / N
    elif reduction == 'sum':
        return torch.sum(losses)
    else:
        raise AssertionError('reduction has to be none, mean or sum')


def mse_loss(logits, labels, reduction='none'):
    """
    :param logits: shape: (N, C)
    :param labels: shape: (N, C)
    :param reduction: options: "none", "mean", "sum"
    :return: loss or losses
    """
    N, C = logits.shape
    assert labels.size(0) == N and labels.size(1) == C, f'label tensor shape is {labels.shape}, while logits tensor shape is {logits.shape}'
    pred = logits.softmax(dim=1)
    losses = torch.sum((pred - labels) ** 2, dim=1)
    if reduction == 'none':
        return losses
    elif reduction == 'mean':
        return torch.sum(losses) / N
    elif reduction == 'sum':
        return torch.sum(losses)
    else:
        raise AssertionError('reduction has to be none, mean or sum')


def active_passive_loss(logits, labels, alpha=10.0, beta=1.0, active='ce', passive='mae', rho=0.7, reduction='none'):
    """
    ICML 2020 - Normalized Loss Functions for Deep Learning with Noisy Labels
    https://github.com/HanxunH/Active-Passive-Losses/blob/master/loss.py

    a loss is defined “Active” if it only optimizes at q(k=y|x)=1, otherwise, a loss is deﬁned as “Passive”

    :param logits: shape: (N, C)
    :param labels: shape: (N)
    :param reduction: options: "none", "mean", "sum"
    :return: loss or losses
    """
    if active == 'ce':
        active_loss = F.cross_entropy(logits, labels, reduction=reduction)
    elif active == 'nce':
        active_loss = normalized_cross_entropy(logits, labels, reduction=reduction)
    elif active == 'gce':
        active_loss = generalized_cross_entropy(logits, labels, rho=rho, reduction=reduction)
    elif active == 'ngce':
        active_loss = normalized_generalized_cross_entropy(logits, labels, rho=rho, reduction=reduction)
    else:
        raise AssertionError(f'active loss: {active} is not supported yet')

    if passive == 'mae':
        passive_loss = mae_loss(logits, labels, reduction=reduction)
    elif passive == 'mse':
        passive_loss = mse_loss(logits, labels, reduction=reduction)
    elif passive == 'rce':
        passive_loss = reversed_cross_entropy(logits, labels, reduction=reduction)
    else:
        raise AssertionError(f'passive loss: {passive} is not supported yet')

    return alpha * active_loss + beta * passive_loss


class SupConLoss(nn.Module):
    """Following Supervised Contrastive Learning:
        https://arxiv.org/pdf/2004.11362.pdf."""

    def __init__(self, temperature=0.07, base_temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(self, features, mask=None, batch_size=-1):
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if mask is not None:
            # SupCon loss (Partial Label Mode)
            mask = mask.float().detach().to(device)
            # compute logits
            anchor_dot_contrast = torch.div(
                torch.matmul(features[:batch_size], features.T),
                self.temperature)
            # for numerical stability
            logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
            logits = anchor_dot_contrast - logits_max.detach()

            # mask-out self-contrast cases
            logits_mask = torch.scatter(
                torch.ones_like(mask),
                1,
                torch.arange(batch_size).view(-1, 1).to(device),
                0
            )
            mask = mask * logits_mask

            # compute log_prob
            exp_logits = torch.exp(logits) * logits_mask
            log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)

            # compute mean of log-likelihood over positive
            mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

            # loss
            loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
            loss = loss.mean()
        else:
            # MoCo loss (unsupervised)
            # compute logits
            # Einstein sum is more intuitive
            # positive logits: Nx1
            q = features[:batch_size]
            k = features[batch_size:batch_size * 2]
            queue = features[batch_size * 2:]
            l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
            # negative logits: NxK
            l_neg = torch.einsum('nc,kc->nk', [q, queue])
            # logits: Nx(1+K)
            logits = torch.cat([l_pos, l_neg], dim=1)

            # apply temperature
            logits /= self.temperature

            # labels: positive key indicators
            labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
            loss = F.cross_entropy(logits, labels)

        return loss


class SimSiamLoss(nn.Module):
    def __init__(self, temperature=0.07, base_temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.cosine_sim = nn.CosineSimilarity(dim=1)

    def forward(self, z, p):
        z = z.detach()
        # p and z should be pre-l2normalized
        loss = - self.cosine_sim(p, z)
        return loss.mean()


class ByolLoss(nn.Module):
    def __init__(self, temperature=0.07, base_temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.cosine_sim = nn.CosineSimilarity(dim=1)

    def forward(self, z, p):
        # x, y are in shape (N, C)
        z = F.normalize(z, dim=1)
        p = F.normalize(p, dim=1)
        return 2 - 2 * (z * p).sum(dim=-1)
