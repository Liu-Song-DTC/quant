# 调优实验计划 — 2026-06-24 (持续优化中)

**目标**: Sharpe > 1.30, 收益越高越好。所有实验跑完验证有效才停。

## 当前基线: 69367c8 (B2/B3/BP8 数据驱动门控)

等待回测中...

## 已做改动 (未提交, 待回测验证)

### 1. BP8因子修复: 慢速路径覆盖 (P4)
- **问题**: BP8的 tanh(-mom_60d*2) 仅fast path(latest_only=True)生效
- **修复**: 慢速路径(latest_only=False)循环中检测 buy_point==8 → 覆盖 fval
- **影响**: 回测中BP8信号之前一直用错误因子(MOM/REV/SHARPE), 现正确使用低动量因子
- **文件**: signal_engine.py _collect_bar_scalars slow path + fast path last bar

### 2. Gate质量系数范围收窄 [0.5,2.0] → [0.7,1.4]
- **数据依据**: gate_quality赢家1.686 vs 输家1.637 (Δ=-0.05, 无预测力)
- **逻辑**: gate仅做安全网, 2x范围引入噪声; 收窄减少假阳性/假阴性
- **文件**: gate_scorer.py compute_gate_quality

### 3. B4门控基础设施 (默认关闭)
- 新增 b4_gate config: enabled/min_signal_level/min_confidence
- buy_confidence 字段加入result dict
- 等回测数据验证B4质量分层后开启
- **文件**: signal_engine.py + factor_config.yaml

### 4. Config清理
- 删除 b3_filter 25个死字段(旧直觉条件, 已改为数据驱动)
- _T 从 banned → restricted (1.3x门槛, 原完全禁止过度激进)
- B2 min_div_strength: 0.03 → 0.02 (Exp D2显示0.03略严, Sharpe 1.30→1.25)

## 待跑实验 (按优先级)

### 回测1: 基线 (进行中)
```
改动: BP8 slow path fix only (backtest已启动)
预期: 与 f92a263 Sharpe 1.30 对比, B3/B2/BP8 gate + BP8 fix
```

### 回测2: Gate范围收窄 + FQG放松 + B2放松
```
改动: gate [0.7,1.4] + _T restricted + B2 0.02
预期: 减少假阳性过滤, Sharpe应≥基线
命令: rm backtest_signals.csv && python bt_execution.py
```

### 回测3: B4门控开启 (需回测2数据验证后)
```
先跑 signal_validator → analysis_framework 获取B4质量分层数据
根据 B4 signal_level vs 胜率 决定阈值
然后: b4_gate.enabled=true + 调参
```

### 回测4: ML blend_weight 调优
```
当前: 0.70, ML IC=0.199唯一有效信号
候选: 0.80, 0.85
需回测2结果评估ML贡献后决定
```

## 待分析项 (回测完成后执行)

1. B4 buy_point质量分层: signal_level/trend_type/buy_confidence vs future_ret
2. B1 拒绝路径: 确认70%拒绝原因 (FQG? price_ok? hard_reject?)
3. BP7 拒绝路径: 同B1分析
4. 死配置清理: portfolio.py mom_60d_fomo_* 加载但未使用

## 关键数据监控

每次回测后记录:
| 指标 | 回测1 | 回测2 | 回测3 | 回测4 |
|------|-------|-------|-------|-------|
| Sharpe | ? | ? | ? | ? |
| 总收益 | ? | ? | ? | ? |
| 最大回撤 | ? | ? | ? | ? |
| B1占信号% | ? | ? | ? | ? |
| B4占信号% | ? | ? | ? | ? |
| 年化收益 | ? | ? | ? | ? |
