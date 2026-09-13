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

    # 过滤掉 NaN 和 Inf 值，只使用有效值计算指标
    valid_mask = ~(np.isnan(predicts) | np.isinf(predicts))
    valid_predicts = predicts[valid_mask]
    valid_labels = labels[valid_mask]
    
    # 检查是否有足够的有效值
    if len(valid_predicts) == 0:
        # 如果所有值都是 NaN/Inf，返回 NaN
        return {'average_precision': np.nan, 'roc_auc': np.nan}
    
    # 检查标签中是否有多个类别（在有效值中）
    unique_labels = np.unique(valid_labels)
    if len(unique_labels) < 2:
        # 如果只有一个类别，无法计算 ROC-AUC
        return {'average_precision': np.nan, 'roc_auc': np.nan}
    
    # 如果有 NaN 值被过滤，记录警告（但只在有 NaN 时警告，不影响其他模型）
    nan_count = np.sum(~valid_mask)
    if nan_count > 0:
        import warnings
        warnings.warn(f"过滤了 {nan_count}/{len(predicts)} 个 NaN/Inf 值，使用剩余 {len(valid_predicts)} 个有效值计算指标", UserWarning)
    
    try:
        average_precision = average_precision_score(y_true=valid_labels, y_score=valid_predicts)
        roc_auc = roc_auc_score(y_true=valid_labels, y_score=valid_predicts)
        return {'average_precision': average_precision, 'roc_auc': roc_auc}
    except ValueError as e:
        # 处理 sklearn 抛出的异常
        error_msg = str(e).lower()
        if "only one class present" in error_msg or "need at least 2 classes" in error_msg:
            return {'average_precision': np.nan, 'roc_auc': np.nan}
        else:
            raise e


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

    # 检查预测值中是否有 nan 或 inf（这通常表示模型问题，不应该静默处理）
    has_nan = np.any(np.isnan(predicts))
    has_inf = np.any(np.isinf(predicts))
    if has_nan or has_inf:
        # 如果有 nan 或 inf，这是模型问题，应该抛出异常或返回 nan
        # 但为了不中断训练，我们返回 nan 让上层处理
        import warnings
        if has_nan:
            warnings.warn(f"预测值中包含 {np.sum(np.isnan(predicts))} 个 nan 值")
        if has_inf:
            warnings.warn(f"预测值中包含 {np.sum(np.isinf(predicts))} 个 inf 值")
        return {'roc_auc': np.nan}
    
    # 检查标签中是否有多个类别
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        # 如果只有一个类别，roc_auc 无法计算，这是数据问题
        # 返回 nan 而不是默认值，让上层决定如何处理
        return {'roc_auc': np.nan}
    
    # 检查预测值是否完全相同（没有区分度）
    # 注意：即使预测值相同，如果标签有多个类别，roc_auc 仍然可以计算（但结果可能无意义）
    # 所以我们不在这里返回默认值，让 sklearn 来处理
    
    try:
        roc_auc = roc_auc_score(y_true=labels, y_score=predicts)
        # 如果结果是 nan，直接返回（可能是数据问题，如所有正样本的预测值都相同）
        return {'roc_auc': roc_auc}
    except ValueError as e:
        # 处理 sklearn 抛出的异常
        error_msg = str(e).lower()
        if "only one class present" in error_msg or "need at least 2 classes" in error_msg:
            # 只有一个类别，无法计算
            return {'roc_auc': np.nan}
        else:
            # 其他未知错误，重新抛出
            raise e
