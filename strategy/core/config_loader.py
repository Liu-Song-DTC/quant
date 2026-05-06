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
            'entry_speed': self.get('portfolio.entry_speed', 0.5),
            'exit_speed': self.get('portfolio.exit_speed', 0.5),
            'position_stop_loss': self.get('portfolio.position_stop_loss', 0.15),
            'portfolio_stop_loss': self.get('portfolio.portfolio_stop_loss', 0.12),
            'max_single_weight': self.get('portfolio.max_single_weight', 0.15),
            'enable_industry_weighting': self.get('portfolio.enable_industry_weighting', True),
            'volatility_control_enabled': self.get('volatility_control.enabled', False),
            'volatility_control_lookback': self.get('volatility_control.lookback_period', 20),
            'portfolio_stop_loss_enabled': self.get('portfolio_stop_loss.enabled', False),
            'emergency_exposure': self.get('portfolio_stop_loss.emergency_exposure', 0.50),
            'fv_exposure_params': self.get('portfolio.fv_exposure_params', {
                'fv_low': -0.03, 'fv_high': 0.05, 'exposure_min': 0.3, 'exposure_max': 1.0,
            }),
            'turnover_bonus': self.get('portfolio.turnover_bonus', 0.02),
            # === 小说优化新增 ===
            'dynamic_min_positions': self.get('portfolio.dynamic_min_positions', {
                'bull': 2, 'neutral': 1, 'bear': 0,
            }),
            'selection': self.get('portfolio.selection', {
                'min_rank_pct': 0.5, 'min_absolute_score': 0.15, 'min_confidence': 0.80,
            }),
        }

    def get_industry_factor_config(self) -> Dict[str, Any]:
        """获取行业因子配置"""
        return self.get('industry_factor_config', {
            'enabled': True,
            'lookback_days': 120,
            'forward_days': 20,
        })

    def get_indicator_params(self) -> Dict[str, Any]:
        """获取技术指标参数（含缠论参数）"""
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
            # 缠论参数
            'divergence': {
                'lookback': 20,
                'peak_trough_lookback': 5,
                'strength_threshold': 0.3,
                'verify_trend': True,
            },
            'structure': {
                'pivot_min_overlap': 3,
                'pivot_zone_buffer': 0.02,
                'min_trend_bars': 8,
                'zhongyin_threshold': 0.02,
            },
        })

    def get_chan_theory_params(self) -> Dict[str, Any]:
        """获取缠论参数配置"""
        return self.get('chan_theory', {
            'enabled': True,
            'divergence': {
                'enabled': True,
                'lookback': 20,
                'peak_trough_lookback': 5,
                'strength_threshold': 0.3,
                'verify_trend': True,
            },
            'structure': {
                'enabled': True,
                'pivot_min_overlap': 3,
                'pivot_zone_buffer': 0.02,
                'min_trend_bars': 8,
                'zhongyin_threshold': 0.02,
            },
            'signal_boost': {
                'bottom_divergence_mult': 1.25,
                'top_divergence_mult': 0.70,
                'alignment_boost': 0.10,
                'zhongyin_penalty': 0.85,
                'pivot_breakout_buy_mult': 1.15,
                'pivot_breakout_sell_mult': 0.75,
            },
            'exit': {
                'top_divergence_exit': 0.4,
                'trend_exhaustion_exit': -0.4,
                'buy_zone_stop_protection': 2.0,
                'volume_divergence_exit': True,
            },
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
