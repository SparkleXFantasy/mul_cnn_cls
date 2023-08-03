import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score, precision_score, f1_score, recall_score, roc_auc_score


def eval_metrics(metrics, data):
    result = dict()
    scores, labels = data
    pred_cls = np.argmax(scores, axis=1)
    if 'AUC' in metrics:
        if scores.shape[1] > 2:    # multi-class
            result['AUC'] = roc_auc_score(labels, scores, multi_class='ovr')
        else:    # binary
            result['AUC'] = roc_auc_score(labels, scores[:, 1].ravel())
    if 'F1' in metrics:
        f1 = f1_score(labels, pred_cls, average='macro')
        result['F1'] = f1
    return result
