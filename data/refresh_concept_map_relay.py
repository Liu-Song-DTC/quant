#!/usr/bin/env python3
"""概念映射刷新 — 中继代理版 (2026-09-30):
push2delay.eastmoney.com 直连被远端切断, 经 akshare_proxy_patch 中继代理拉取.
逻辑与 refresh_concept_map_0903.py 完全一致, 仅前置安装补丁."""
import yaml, runpy
import akshare_proxy_patch

cfg = yaml.safe_load(open('/mnt/d/quant/trade/config.yaml', encoding='utf-8'))['proxy']
akshare_proxy_patch.install_patch(cfg['host'], auth_token=cfg['auth_token'], retry=8,
    hook_domains=['push2delay.eastmoney.com'])
runpy.run_path('/mnt/d/quant/data/refresh_concept_map_0903.py', run_name='__main__')
