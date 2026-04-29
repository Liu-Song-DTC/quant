"""实盘交易配置"""
from pathlib import Path

import yaml

ROOT = Path(__file__).parent.parent.resolve()


class TradeConfig:
    """实盘配置 — 独立于回测策略配置"""

    def __init__(self):
        config_path = Path(__file__).parent / "config.yaml"
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                self._cfg = yaml.safe_load(f) or {}
        else:
            self._cfg = {}

    @property
    def init_cash(self) -> int:
        return self._cfg.get("portfolio", {}).get("init_cash", 100000)

    @property
    def max_position(self) -> int:
        return self._cfg.get("portfolio", {}).get("max_position", 10)

    @property
    def stock_data_dir(self) -> Path:
        return ROOT / "data" / "stock_data"

    @property
    def bt_data_dir(self) -> Path:
        return self.stock_data_dir / "backtrader_data"

    @property
    def fund_data_dir(self) -> Path:
        return self.stock_data_dir / "fundamental_data"

    @property
    def state_file(self) -> Path:
        return Path(__file__).parent / self._cfg.get("state_file", "portfolio_state.json")

    @property
    def report_dir(self) -> Path:
        return Path(__file__).parent / self._cfg.get("report_dir", "reports")

    @property
    def rec_file(self) -> Path:
        return Path(__file__).parent / "last_recommendations.json"

    @property
    def notification_enabled(self) -> bool:
        return self._cfg.get("notification", {}).get("enabled", False)

    @property
    def notification_sckey(self) -> str:
        return self._cfg.get("notification", {}).get("sckey", "")
