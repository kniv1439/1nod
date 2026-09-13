import os
import torch
import torch.nn as nn
import logging


class EarlyStopping(object):

    def __init__(self, patience: int, save_model_folder: str, save_model_name: str, logger: logging.Logger, model_name: str = None):
        """
        Early stop strategy.
        :param patience: int, max patience
        :param save_model_folder: str, save model folder
        :param save_model_name: str, save model name
        :param logger: Logger
        :param model_name: str, model name
        """
        self.patience = patience
        self.counter = 0
        self.best_metrics = {}
        self.early_stop = False
        self.logger = logger
        self.save_model_path = os.path.join(save_model_folder, f"{save_model_name}.pkl")
        self.model_name = model_name
        if self.model_name in ['JODIE', 'DyRep', 'TGN']:
            # path to additionally save the nonparametric data (e.g., tensors) in memory-based models (e.g., JODIE, DyRep, TGN)
            self.save_model_nonparametric_data_path = os.path.join(save_model_folder, f"{save_model_name}_nonparametric_data.pkl")

    def step(self, metrics: list, model: nn.Module):
        """
        execute the early stop strategy for each evaluation process
        :param metrics: list, list of metrics, each element is a tuple (str, float, boolean) -> (metric_name, metric_value, whether higher means better)
        :param model: nn.Module
        :return:
        """
        metrics_compare_results = []
        for metric_tuple in metrics:
            metric_name, metric_value, higher_better = metric_tuple[0], metric_tuple[1], metric_tuple[2]

            if higher_better:
                if self.best_metrics.get(metric_name) is None or metric_value >= self.best_metrics.get(metric_name):
                    metrics_compare_results.append(True)
                else:
                    metrics_compare_results.append(False)
            else:
                if self.best_metrics.get(metric_name) is None or metric_value <= self.best_metrics.get(metric_name):
                    metrics_compare_results.append(True)
                else:
                    metrics_compare_results.append(False)
        # all the computed metrics are better than the best metrics
        if torch.all(torch.tensor(metrics_compare_results)):
            for metric_tuple in metrics:
                metric_name, metric_value = metric_tuple[0], metric_tuple[1]
                self.best_metrics[metric_name] = metric_value
            self.save_checkpoint(model)
            self.counter = 0
        # metrics are not better at the epoch
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop

    def save_checkpoint(self, model: nn.Module):
        """
        saves model at self.save_model_path
        :param model: nn.Module
        :return:
        """
        self.logger.info(f"save model {self.save_model_path}")
        # 兼容DataParallel
        if isinstance(model, nn.DataParallel):
            torch.save(model.state_dict(), self.save_model_path)
        else:
            torch.save(model.state_dict(), self.save_model_path)
        if self.model_name in ['JODIE', 'DyRep', 'TGN']:
            torch.save(model[0].memory_bank.node_raw_messages, self.save_model_nonparametric_data_path)

    def load_checkpoint(self, model: nn.Module, map_location: str = None):
        """
        load model at self.save_model_path
        :param model: nn.Module
        :param map_location: str, how to remap the storage locations
        :return:
        """
        self.logger.info(f"load model {self.save_model_path}")
        try:
            state_dict = torch.load(self.save_model_path, map_location=map_location)
            # 兼容DataParallel
            if isinstance(model, nn.DataParallel):
                model.load_state_dict(state_dict)
            else:
                model.load_state_dict(state_dict)
            if self.model_name in ['JODIE', 'DyRep', 'TGN']:
                model[0].memory_bank.node_raw_messages = torch.load(self.save_model_nonparametric_data_path, map_location=map_location)
        except Exception as e:
            error_msg = str(e)
            self.logger.warning(f"完整加载模型失败，尝试只加载backbone部分: {error_msg}")
            
            # 检查是否是分类器部分的尺寸不匹配
            if "size mismatch for 1." in error_msg:
                self.logger.info("检测到分类器部分尺寸不匹配，将只加载backbone部分")
                self.load_backbone_only(model, map_location=map_location)
            else:
                # 对于其他错误，依然抛出异常
                raise

    def load_backbone_only(self, model: nn.Module, map_location: str = None):
        """
        只加载模型的backbone部分参数，忽略分类器部分
        :param model: nn.Module, 仅包含backbone部分的模型
        :param map_location: str, 如何重新映射存储位置
        :return:
        """
        self.logger.info(f"只加载backbone部分: {self.save_model_path}")
        
        try:
            pretrained_state_dict = torch.load(self.save_model_path, map_location=map_location)
            
            # 只保留backbone部分的参数（以"0."开头的键）
            backbone_state_dict = {}
            model_state_dict = model.state_dict()
            
            # 检查维度匹配的参数
            dimension_mismatch = []
            compatible_params = 0
            total_params = 0
            
            for key, value in pretrained_state_dict.items():
                if key.startswith('0.'):
                    total_params += 1
                    # 检查是否存在于当前模型且维度匹配
                    if key in model_state_dict and model_state_dict[key].shape == value.shape:
                        backbone_state_dict[key] = value
                        compatible_params += 1
                    elif key in model_state_dict:
                        dimension_mismatch.append(f"{key}: pretrained={value.shape}, current={model_state_dict[key].shape}")
            
            # 检查是否有backbone参数被找到
            if len(backbone_state_dict) == 0:
                self.logger.warning(f"在预训练模型中未找到可直接加载的backbone参数，尝试其他加载策略")
                
                # 尝试更宽松的加载方式 - 按名称匹配参数，允许部分层加载
                # 这种方法适用于相似但不完全相同的模型架构
                for key, param in model_state_dict.items():
                    if key.startswith('0.') and key in pretrained_state_dict and pretrained_state_dict[key].shape == param.shape:
                        backbone_state_dict[key] = pretrained_state_dict[key]
                        compatible_params += 1
            
            # 如果存在维度不匹配，记录日志 - 但不重复打印
            if dimension_mismatch:
                self.logger.warning(f"发现{len(dimension_mismatch)}个维度不匹配的参数")
                # 只打印第一个不匹配的参数作为示例
                self.logger.warning(dimension_mismatch[0])
                
                if self.model_name == 'PathDyG':
                    self.logger.info("对于PathDyG，W_pos维度不匹配是正常的，已通过配置修复")
                else:
                    self.logger.warning("请检查模型配置是否与预训练模型一致")
            
            # 设置strict=False允许加载部分参数
            missing_keys, unexpected_keys = model.load_state_dict(backbone_state_dict, strict=False)
            
            # 检查并记录加载状态
            loaded_params = len(backbone_state_dict)
            total_model_params = sum(1 for _ in model_state_dict.keys() if _.startswith('0.'))
            compatibility_rate = (compatible_params / total_params * 100) if total_params > 0 else 0
            
            self.logger.info(f"成功加载 {loaded_params}/{total_model_params} 个参数到backbone")
            self.logger.info(f"参数兼容率: {compatibility_rate:.2f}%")
            
            if compatibility_rate < 50 and self.model_name != 'PathDyG':
                self.logger.warning("参数兼容率低于50%，模型可能无法正常工作")
            
            if self.model_name in ['JODIE', 'DyRep', 'TGN']:
                model[0].memory_bank.node_raw_messages = torch.load(self.save_model_nonparametric_data_path, map_location=map_location)
        
        except Exception as e:
            self.logger.error(f"加载backbone时出错: {str(e)}")
            raise
