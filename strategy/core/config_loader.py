# core/config_loader.py
"""配置加载模块 - 从YAML文件加载因子配置"""

import os
import yaml
from typing import Dict, Any, Optional


class ConfigLoader:
    """配置加载器 - 统一管理策略配置"""

    _instance = None
    _config = None

    def __new__(cls, config_path: Optional[str] = None):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, config_path: Optional[str] = None):
        if config_path and self._config is None:
            self.load(config_path)

    def load(self, config_path: str) -> Dict[str, Any]:
        """加载YAML配置文件"""
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"配置文件不存在: {config_path}")

        with open(config_path, 'r', encoding='utf-8') as f:
            self._config = yaml.safe_load(f)

        return self._config

    def get(self, key: str, default: Any = None) -> Any:
        """获取配置项"""
        if self._config is None:
            return default

        keys = key.split('.')
        value = self._config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def get_technical_weights(self) -> Dict[str, float]:
        """获取技术面因子权重"""
        return self.get('technical_weights', {
            'volatility_10': 0.30,
            'rsi_average': 0.25,
            'bb_width': 0.15,
            'momentum': 0.30,
        })

    def get_signal_thresholds(self) -> Dict[str, float]:
        """获取交易信号阈值"""
        return self.get('signal_thresholds', {
            'buy': 0.15,
            'adjusted_buy': 0.05,
            'sell': -0.05,
        })

    def get_portfolio_config(self) -> Dict[str, Any]:
        """获取组合配置"""
        return {
            'max_position': self.get('portfolio.max_position', 10),
            'target_volatility': self.get('portfolio.target_volatility', 0.20),
            'entry_speed': self.get('portfolio.entry_speed', 1.0),
            'exit_speed': self.get('portfolio.exit_speed', 1.0),
            'position_stop_loss': self.get('portfolio.position_stop_loss', 0.10),
            'portfolio_stop_loss': self.get('portfolio.portfolio_stop_loss', 0.08),
            'volatility_control_enabled': self.get('volatility_control.enabled', False),
            'portfolio_stop_loss_enabled': self.get('portfolio_stop_loss.enabled', False),
            'emergency_exposure': self.get('portfolio_stop_loss.emergency_exposure', 0.30),
        }

    def get_industry_factor_weights(self) -> Dict[str, Dict[str, float]]:
        """获取行业因子权重"""
        return self.get('industry_factor_weights', {
            '科技/成长': {'volatility_10': 0.25, 'rsi_average': 0.35, 'bb_width': 0.15, 'momentum': 0.25},
            '周期/资源': {'volatility_10': 0.40, 'rsi_average': 0.30, 'bb_width': 0.10, 'momentum': 0.20},
            '消费/稳定': {'volatility_10': 0.30, 'rsi_average': 0.20, 'bb_width': 0.30, 'momentum': 0.20},
            '金融/大盘': {'volatility_10': 0.35, 'rsi_average': 0.20, 'bb_width': 0.15, 'momentum': 0.30},
            'default': {'volatility_10': 0.25, 'rsi_average': 0.30, 'bb_width': 0.20, 'momentum': 0.25},
        })

    def get_industry_category(self) -> Dict[str, list]:
        """获取行业分类映射"""
        return self.get('industry_category', {})

    @property
    def config(self) -> Dict[str, Any]:
        """获取完整配置"""
        return self._config


def load_config(config_path: str = None) -> ConfigLoader:
    """加载配置的便捷函数"""
    if config_path is None:
        # 从项目根目录加载
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        config_path = os.path.join(base_dir, 'config', 'factor_config.yaml')

    loader = ConfigLoader()
    loader.load(config_path)
    return loader
