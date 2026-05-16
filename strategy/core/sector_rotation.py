# core/sector_rotation.py
import pandas as pd
import numpy as np
import os

class SectorRotation:
    """行业轮动策略"""

    def __init__(self, data_path):
        self.data_path = data_path
        self.sector_momentum = {}
        self.industry_map = self._load_industry_map()

    def _load_industry_map(self):
        """加载行业映射"""
        industry_map = {}
        fundamental_path = self.data_path
        if not os.path.exists(fundamental_path):
            return industry_map

        for file in os.listdir(fundamental_path):
            if not file.endswith('.csv'):
                continue
            code = file[:-4]
            try:
                df = pd.read_csv(fundamental_path + file)
                if len(df) > 0:
                    industry = df.iloc[0].get('行业', None)
                    if industry:
                        industry_map[code] = industry
            except Exception:
                continue
        return industry_map

    def calculate_sector_momentum(self, market_data_dict, lookback=20):
        """计算各行业动量"""
        sector_returns = {}

        for code, df in market_data_dict.items():
            if code not in self.industry_map:
                continue
            if len(df) < lookback:
                continue

            industry = self.industry_map[code]
            close = df['close'].values
            ret = (close[-1] / close[-lookback]) - 1

            if industry not in sector_returns:
                sector_returns[industry] = []
            sector_returns[industry].append(ret)

        # 计算行业平均收益
        for industry, returns in sector_returns.items():
            self.sector_momentum[industry] = np.mean(returns)

        return self.sector_momentum

    def get_sector_score(self, code):
        """获取某股票所属行业的动量分数"""
        if code not in self.industry_map:
            return 0.0

        industry = self.industry_map[code]
        momentum = self.sector_momentum.get(industry, 0.0)

        # 行业动量归一化到0-0.3
        max_momentum = max(abs(v) for v in self.sector_momentum.values()) if self.sector_momentum else 0.01
        score = (momentum / max_momentum) * 0.3 if max_momentum > 0 else 0.0

        return score

    def get_strong_sectors(self, top_n=5):
        """获取强势行业"""
        if not self.sector_momentum:
            return []
        sorted_sectors = sorted(self.sector_momentum.items(), key=lambda x: x[1], reverse=True)
        return [s[0] for s in sorted_sectors[:top_n]]

    def get_weak_sectors(self, bottom_n=5):
        """获取弱势行业"""
        if not self.sector_momentum:
            return []
        sorted_sectors = sorted(self.sector_momentum.items(), key=lambda x: x[1])
        return [s[0] for s in sorted_sectors[:bottom_n]]
