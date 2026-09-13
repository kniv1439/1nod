import torch
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def get_link_prediction_metrics(predicts: torch.Tensor, labels: torch.Tensor):
    """
    get metrics for the link prediction task
    :param predicts: Tensor, shape (num_samples, )
    :param labels: Tensor, shape (num_samples, )
    :return:
        dictionary of metrics {'metric_name_1': metric_1, ...}
    """
    predicts = predicts.cpu().detach().numpy()
    labels = labels.cpu().numpy()

    # 检查是否只有一个类别
    unique_labels = np.unique(labels)
    if len(unique_labels) == 1:
        # 如果只有一个类别，返回默认值
        return {'average_precision': 0.5, 'roc_auc': 0.5}
    
    try:
        average_precision = average_precision_score(y_true=labels, y_score=predicts)
        roc_auc = roc_auc_score(y_true=labels, y_score=predicts)
        return {'average_precision': average_precision, 'roc_auc': roc_auc}
    except ValueError as e:
        # 如果计算失败，返回默认值
        print(f"Warning: Metrics calculation failed: {e}")
        return {'average_precision': 0.5, 'roc_auc': 0.5}


def get_node_classification_metrics(predicts: torch.Tensor, labels: torch.Tensor):
    """
    get metrics for the node classification task
    :param predicts: Tensor, shape (num_samples, )
    :param labels: Tensor, shape (num_samples, )
    :return:
        dictionary of metrics {'metric_name_1': metric_1, ...}
    """
    predicts = predicts.cpu().detach().numpy()
    labels = labels.cpu().numpy()

    # 检查是否只有一个类别
    unique_labels = np.unique(labels)
    if len(unique_labels) == 1:
        # 如果只有一个类别，返回默认值
        return {'roc_auc': 0.5}
    
    try:
        roc_auc = roc_auc_score(y_true=labels, y_score=predicts)
        return {'roc_auc': roc_auc}
    except ValueError as e:
        # 如果计算失败，返回默认值
        print(f"Warning: ROC AUC calculation failed: {e}")
        return {'roc_auc': 0.5}
