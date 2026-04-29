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

    def get_portfolio_config(self) -> Dict[str, Any]:
        """获取组合配置"""
        return {
            'target_volatility': self.get('portfolio.target_volatility', 0.25),
            'entry_speed': self.get('portfolio.entry_speed', 1.0),
            'exit_speed': self.get('portfolio.exit_speed', 1.0),
            'position_stop_loss': self.get('portfolio.position_stop_loss', 0.15),
            'portfolio_stop_loss': self.get('portfolio.portfolio_stop_loss', 0.12),
            'max_single_weight': self.get('portfolio.max_single_weight', 0.15),
            'enable_industry_weighting': self.get('portfolio.enable_industry_weighting', True),
            'volatility_control_enabled': self.get('volatility_control.enabled', False),
            'portfolio_stop_loss_enabled': self.get('portfolio_stop_loss.enabled', False),
            'emergency_exposure': self.get('portfolio_stop_loss.emergency_exposure', 0.50),
        }

    def get_industry_factor_config(self) -> Dict[str, Any]:
        """获取行业因子配置"""
        return self.get('industry_factor_config', {
            'enabled': True,
            'lookback_days': 120,
            'forward_days': 20,
        })

    def get_indicator_params(self) -> Dict[str, Any]:
        """获取技术指标参数"""
        return self.get('indicator_params', {
            'ema_periods': [5, 10, 20, 60],
            'ma_periods': [5, 10, 20, 30, 60],
            'rsi_periods': [6, 8, 10, 14],
            'bb_window': 20,
            'bb_std': 2,
            'momentum_periods': [3, 5, 10, 20, 30],
            'volatility_periods': [5, 10, 20],
            'atr_periods': [10, 14, 20],
            'volume_ma_period': 20,
        })

    @property
    def config(self) -> Dict[str, Any]:
        """获取完整配置"""
        return self._config


def load_config(config_path: str = None) -> ConfigLoader:
    """加载配置的便捷函数"""
    if config_path is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_path = os.path.join(base_dir, 'config', 'factor_config.yaml')

    loader = ConfigLoader()
    loader.load(config_path)
    return loader
