import torch
from torch.nn.functional import cross_entropy
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score


def multilabel_crossentropy_loss(x_conf_scores, x_current_cat, cat_vocab_size):
    # x_conf_scores (logits) - batch_size x cat_vocab_size
    # x_current_cat - batch_size x max_cat_len
    multi_loss = torch.sum(torch.stack([torch.sum(torch.stack([cross_entropy(x_conf_scores[b, :].reshape(1, -1),
                                                                             label.reshape(-1))
                                                  for label in x_current_cat[b, :] if label != cat_vocab_size], dim=0))
                           for b in range(x_conf_scores.shape[0])]), dim=0)
    return multi_loss

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def valid_thr(gt_valid, scores_valid):
    # gt_valid [sample_size, cat_vocab_size] - one-hot
    # scores_valid [sample_size, cat_vocab_size]

    sigmoid_scores_valid = sigmoid(scores_valid)
    possible_thr = np.arange(0, 1, 0.01)
    final_thr = np.array([possible_thr[np.argmax(np.array([f1_score(gt_valid[:, j], sigmoid_scores_valid[:, j] >= thr)
                                                 for thr in possible_thr]))] for j in range(gt_valid.shape[1])])
    # final_thr [cat_vocab_size]
    k = np.median(np.sum(gt_valid, axis=1))
    return final_thr, k


def calculate_all_metrics(all_gt, all_scores, final_thr, k, kind='thr'):
    # all_gt [sample_size, cat_vocab_size] - one-hot
    # all_scores [sample_size, cat_vocab_size] - confidence-scores
    tasks_with_non_trivial_targets = np.where(all_gt.sum(axis=0) != 0)[0]
    all_gt = all_gt[:, tasks_with_non_trivial_targets]
    all_scores = all_scores[:, tasks_with_non_trivial_targets]
    all_scores_sigmoid = sigmoid(all_scores)

    #final_thr, k = valid_thr(gt_valid, scores_valid)
    final_thr = final_thr[tasks_with_non_trivial_targets]
    #print(final_thr.shape)
    if kind == 'thr':
        all_preds = (all_scores_sigmoid >= final_thr).astype(np.int64)
        #print('thr ---- ', all_preds)
    if kind == 'topk':
        all_preds = np.array([np.insert(np.zeros(all_gt.shape[1]), sorted(range(len(all_scores[b, :])), key=lambda i: all_scores[b, :][i])[-k:], 1).tolist()
                    for b in range(all_gt.shape[0])])
    #print(all_gt, all_gt.shape,  all_preds, all_preds.shape)
    metrics_dict = {'precision_micro': precision_score(all_gt, all_preds, average='micro'),
                    'precision_macro': precision_score(all_gt, all_preds, average='macro'),
                    'recall_micro': recall_score(all_gt, all_preds, average='micro'),
                    'recall_macro': recall_score(all_gt, all_preds, average='macro'),
                    'f1_micro': f1_score(all_gt, all_preds, average='micro'),
                    'f1_macro': f1_score(all_gt, all_preds, average='macro'),
                    'roc_auc_micro': roc_auc_score(all_gt, all_scores, average='micro'),
                    'roc_auc_macro': roc_auc_score(all_gt, all_scores, average='macro')}
    return metrics_dict
