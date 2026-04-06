import os
import json
import time
import logging
from datetime import datetime
import torch
import numpy as np
from typing import Dict, Any, Optional


class ExperimentLogger:
    """
    实验日志记录器 - 用于记录训练、测试和模型性能的完整日志
    """
    
    def __init__(self, log_dir: str = "./logs", experiment_name: str = None):
        """
        初始化日志记录器
        
        Args:
            log_dir: 日志保存目录
            experiment_name: 实验名称，如果为None则自动生成
        """
        self.log_dir = log_dir
        self.experiment_name = experiment_name or f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # 创建日志目录
        self.experiment_dir = os.path.join(log_dir, self.experiment_name)
        os.makedirs(self.experiment_dir, exist_ok=True)
        
        # 初始化日志文件路径
        self.train_log_file = os.path.join(self.experiment_dir, "training.log")
        self.test_log_file = os.path.join(self.experiment_dir, "testing.log")
        self.metrics_file = os.path.join(self.experiment_dir, "metrics.json")
        self.config_file = os.path.join(self.experiment_dir, "config.json")
        
        # 设置日志格式
        self.log_format = '%(asctime)s - %(levelname)s - %(message)s'
        
        # 初始化训练日志记录器
        self.train_logger = self._setup_logger('train', self.train_log_file)
        
        # 初始化测试日志记录器
        self.test_logger = self._setup_logger('test', self.test_log_file)
        
        # 初始化指标存储
        self.metrics_data = {
            'experiment_info': {},
            'training_history': [],
            'validation_history': [],
            'test_results': {},
            'model_info': {},
            'performance_summary': {}
        }
        
        # 记录实验开始时间
        self.start_time = time.time()
        
    def _setup_logger(self, name: str, log_file: str) -> logging.Logger:
        """设置日志记录器"""
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        
        # 清除已有的处理器
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        # 文件处理器
        file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        file_formatter = logging.Formatter(self.log_format)
        file_handler.setFormatter(file_formatter)
        
        # 控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter('%(levelname)s - %(message)s')
        console_handler.setFormatter(console_formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    
    def log_experiment_config(self, args: Any):
        """记录实验配置"""
        config_dict = {}
        if hasattr(args, '__dict__'):
            config_dict = vars(args)
        elif isinstance(args, dict):
            config_dict = args
        
        # 转换不可序列化的对象
        serializable_config = {}
        for key, value in config_dict.items():
            if isinstance(value, (str, int, float, bool, list, dict, type(None))):
                serializable_config[key] = value
            else:
                serializable_config[key] = str(value)
        
        self.metrics_data['experiment_info'] = {
            'experiment_name': self.experiment_name,
            'start_time': datetime.now().isoformat(),
            'config': serializable_config
        }
        
        # 保存配置到文件
        with open(self.config_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_config, f, indent=2, ensure_ascii=False)
        
        self.train_logger.info(f"实验开始: {self.experiment_name}")
        self.train_logger.info(f"配置参数: {json.dumps(serializable_config, indent=2, ensure_ascii=False)}")
    
    def log_model_info(self, model: torch.nn.Module, input_shape: tuple = None):
        """记录模型信息"""
        model_info = {
            'model_name': model.__class__.__name__,
            'total_params': sum(p.numel() for p in model.parameters()),
            'trainable_params': sum(p.numel() for p in model.parameters() if p.requires_grad),
            'model_size_mb': sum(p.numel() * p.element_size() for p in model.parameters()) / 1024 / 1024
        }

        if hasattr(model, 'head_parameter_summary'):
            model_info['head_summary'] = model.head_parameter_summary()
        
        if input_shape:
            model_info['input_shape'] = input_shape
        
        self.metrics_data['model_info'] = model_info
        
        self.train_logger.info(f"模型信息:")
        self.train_logger.info(f"  - 模型名称: {model_info['model_name']}")
        self.train_logger.info(f"  - 总参数量: {model_info['total_params']:,}")
        self.train_logger.info(f"  - 可训练参数: {model_info['trainable_params']:,}")
        self.train_logger.info(f"  - 模型大小: {model_info['model_size_mb']:.2f} MB")
        if 'head_summary' in model_info:
            for branch_idx, branch_summary in enumerate(model_info['head_summary']):
                self.train_logger.info(
                    f"  - 分支{branch_idx}头部: {branch_summary['head_type']} "
                    f"(params={branch_summary['head_parameters']:,}, patch_num={branch_summary['patch_num']}, pred_seq={branch_summary['pred_seq']})"
                )
    
    def log_epoch_start(self, epoch: int, total_epochs: int):
        """记录epoch开始"""
        self.current_epoch = epoch
        self.epoch_start_time = time.time()
        self.train_logger.info(f"开始训练 Epoch {epoch + 1}/{total_epochs}")
    
    def log_epoch_metrics(self, epoch: int, train_loss: float, vali_loss: float, 
                         vali_mae: float, test_loss: float, test_mae: float, 
                         epoch_time: float, learning_rate: float = None):
        """记录epoch指标"""
        epoch_data = {
            'epoch': epoch + 1,
            'train_loss': float(train_loss),
            'vali_loss': float(vali_loss),
            'vali_mae': float(vali_mae),
            'test_loss': float(test_loss),
            'test_mae': float(test_mae),
            'epoch_time': epoch_time,
            'timestamp': datetime.now().isoformat()
        }
        
        if learning_rate is not None:
            epoch_data['learning_rate'] = learning_rate
        
        self.metrics_data['training_history'].append(epoch_data)
        
        # 记录到训练日志
        self.train_logger.info(
            f"Epoch {epoch + 1} 完成 - "
            f"训练损失: {train_loss:.6f}, "
            f"验证损失: {vali_loss:.6f}, "
            f"验证MAE: {vali_mae:.6f}, "
            f"测试损失: {test_loss:.6f}, "
            f"测试MAE: {test_mae:.6f}, "
            f"耗时: {epoch_time:.2f}s"
        )
        
        if learning_rate is not None:
            self.train_logger.info(f"当前学习率: {learning_rate}")
    
    def log_early_stopping(self, epoch: int, best_loss: float):
        """记录早停信息"""
        self.train_logger.info(f"早停触发 - Epoch {epoch + 1}, 最佳验证损失: {best_loss:.6f}")
    
    def log_training_complete(self, best_epoch: int, best_val_loss: float, best_test_loss_at_best_val: float, total_time: float):
        """记录训练完成信息"""
        training_summary = {
            'total_epochs': len(self.metrics_data['training_history']),
            'best_epoch': best_epoch + 1 if best_epoch >= 0 else None,
            'best_validation_loss': float(best_val_loss),
            'best_test_loss_at_best_val': float(best_test_loss_at_best_val),
            'total_training_time': total_time,
            'average_epoch_time': total_time / len(self.metrics_data['training_history']) if self.metrics_data['training_history'] else 0
        }
        
        self.metrics_data['performance_summary']['training'] = training_summary
        
        self.train_logger.info("=" * 50)
        self.train_logger.info("训练完成!")
        self.train_logger.info(f"总训练轮数: {training_summary['total_epochs']}")
        self.train_logger.info(f"最佳轮数: {training_summary['best_epoch']}")
        self.train_logger.info(f"最佳验证损失: {training_summary['best_validation_loss']:.6f}")
        self.train_logger.info(f"最佳验证轮对应测试损失: {training_summary['best_test_loss_at_best_val']:.6f}")
        self.train_logger.info(f"总训练时间: {training_summary['total_training_time']:.2f}s")
        self.train_logger.info(f"平均每轮时间: {training_summary['average_epoch_time']:.2f}s")
        self.train_logger.info("=" * 50)
    
    def log_test_start(self):
        """记录测试开始"""
        self.test_start_time = time.time()
        self.test_logger.info("开始模型测试...")
    
    def log_test_results(self, mse: float, mae: float, rmse: float = None, 
                        mape: float = None, mspe: float = None):
        """记录测试结果"""
        test_time = time.time() - self.test_start_time
        
        test_results = {
            'mse': float(mse),
            'mae': float(mae),
            'test_time': test_time,
            'timestamp': datetime.now().isoformat()
        }
        
        if rmse is not None:
            test_results['rmse'] = float(rmse)
        if mape is not None:
            test_results['mape'] = float(mape)
        if mspe is not None:
            test_results['mspe'] = float(mspe)
        
        self.metrics_data['test_results'] = test_results
        self.metrics_data['performance_summary']['testing'] = test_results
        
        # 记录到测试日志
        self.test_logger.info("=" * 50)
        self.test_logger.info("测试结果:")
        self.test_logger.info(f"  - MSE: {mse:.6f}")
        self.test_logger.info(f"  - MAE: {mae:.6f}")
        if rmse is not None:
            self.test_logger.info(f"  - RMSE: {rmse:.6f}")
        if mape is not None:
            self.test_logger.info(f"  - MAPE: {mape:.6f}")
        if mspe is not None:
            self.test_logger.info(f"  - MSPE: {mspe:.6f}")
        self.test_logger.info(f"  - 测试时间: {test_time:.2f}s")
        self.test_logger.info("=" * 50)
    
    def save_metrics(self):
        """保存所有指标到JSON文件"""
        # 添加实验总结信息
        total_time = time.time() - self.start_time
        self.metrics_data['experiment_info']['end_time'] = datetime.now().isoformat()
        self.metrics_data['experiment_info']['total_duration'] = total_time
        
        with open(self.metrics_file, 'w', encoding='utf-8') as f:
            json.dump(self.metrics_data, f, indent=2, ensure_ascii=False)
        
        self.train_logger.info(f"实验指标已保存到: {self.metrics_file}")
    
    def log_custom_message(self, message: str, level: str = 'info', log_type: str = 'train'):
        """记录自定义消息"""
        logger = self.train_logger if log_type == 'train' else self.test_logger
        
        if level.lower() == 'info':
            logger.info(message)
        elif level.lower() == 'warning':
            logger.warning(message)
        elif level.lower() == 'error':
            logger.error(message)
        elif level.lower() == 'debug':
            logger.debug(message)
    
    def get_experiment_dir(self) -> str:
        """获取实验目录路径"""
        return self.experiment_dir
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """获取指标摘要"""
        return self.metrics_data.get('performance_summary', {})


def create_logger(log_dir: str = "./logs", experiment_name: str = None) -> ExperimentLogger:
    """
    创建实验日志记录器的便捷函数
    
    Args:
        log_dir: 日志保存目录
        experiment_name: 实验名称
    
    Returns:
        ExperimentLogger实例
    """
    return ExperimentLogger(log_dir, experiment_name)
