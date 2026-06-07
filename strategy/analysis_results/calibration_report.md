# 离线标定报告

生成时间: 2026-06-07 13:39

## 汇总

- 行业数: 376
- 总标定组合: 10162
- IC>5%的组合: 6498 (63.9%)

## 详细结果

### 2025三季报预增

- **Neutral**: ['trend_lowvol', 'momentum_reversal'] (单因子IC=0.0532, 组合IC=0.0588)
  - weights: [0.5181, 0.4819]
- **Bull**: ['low_downside', 'fund_pb'] (单因子IC=0.0627, 组合IC=0.0784)
  - bull_weights: [0.5389, 0.4611]
- **Bear**: ['mom_x_lowvol_20_20', 'top_fractal_volume'] (单因子IC=0.049, 组合IC=0.0726)
  - bear_weights: [0.57, 0.43]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| trend_lowvol | neutral | 0.0574 | 0.2013 | 0.2853 | 0.2109 | 0.1727 |
| momentum_reversal | neutral | 0.0490 | 0.1775 | 0.2758 | 0.1649 | 0.1606 |
| mom_x_lowvol_20_20 | neutral | 0.0421 | 0.1671 | 0.2519 | 0.1566 | 0.1457 |
| fund_pb | neutral | 0.0520 | 0.2135 | 0.2436 | 0.1879 | 0.1447 |
| fund_profit_growth | neutral | 0.0390 | 0.1655 | 0.2359 | 0.1260 | 0.1328 |
| rsi_vol_combo | neutral | 0.0351 | 0.1754 | 0.2000 | 0.1441 | 0.1144 |
| volatility | neutral | 0.0348 | 0.1934 | 0.1801 | 0.1733 | 0.1056 |
| fund_pe | neutral | 0.0282 | 0.1783 | 0.1579 | 0.1253 | 0.0888 |
| low_downside | bull | 0.0581 | 0.1797 | 0.3233 | 0.2652 | 0.2045 |
| fund_pb | bull | 0.0673 | 0.2447 | 0.2749 | 0.2727 | 0.1750 |
| volatility | bull | 0.0403 | 0.1781 | 0.2265 | 0.1818 | 0.1338 |
| wash_sale_score | bull | 0.0215 | 0.1234 | 0.1744 | 0.1583 | 0.1010 |
| turnover_stability | bull | 0.0214 | 0.1391 | 0.1542 | 0.1515 | 0.0888 |
| fund_pe | bull | 0.0281 | 0.1809 | 0.1555 | 0.1136 | 0.0866 |
| fund_profit_growth | bull | 0.0196 | 0.1653 | 0.1186 | 0.1058 | 0.0655 |
| mom_x_lowvol_20_20 | bear | 0.0679 | 0.2205 | 0.3080 | 0.3151 | 0.2025 |
| top_fractal_volume | bear | 0.0300 | 0.1186 | 0.2530 | 0.2075 | 0.1528 |

### 2025年报扭亏

- **Neutral**: ['fund_pb', 'volatility', 'mom_x_lowvol_20_20'] (单因子IC=0.068, 组合IC=0.087)
  - weights: [0.3489, 0.343, 0.3081]
- **Bull**: ['low_downside', 'volatility', 'trend_lowvol'] (单因子IC=0.1018, 组合IC=0.1193)
  - bull_weights: [0.4029, 0.3589, 0.2381]
- **Bear**: ['momentum_reversal'] (单因子IC=0.1289, 组合IC=0.1289)
  - bear_weights: [1.0]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| fund_pb | neutral | 0.0606 | 0.1257 | 0.4822 | 0.3779 | 0.3322 |
| volatility | neutral | 0.0767 | 0.1635 | 0.4691 | 0.3925 | 0.3266 |
| mom_x_lowvol_20_20 | neutral | 0.0666 | 0.1503 | 0.4432 | 0.3236 | 0.2933 |
| trend_lowvol | neutral | 0.0754 | 0.1705 | 0.4423 | 0.3173 | 0.2913 |
| turnover_stability | neutral | 0.0369 | 0.0857 | 0.4309 | 0.3340 | 0.2874 |
| momentum_reversal | neutral | 0.0673 | 0.1536 | 0.4381 | 0.3027 | 0.2854 |
| rsi_vol_combo | neutral | 0.0539 | 0.1462 | 0.3688 | 0.2651 | 0.2333 |
| fund_profit_growth | neutral | 0.0361 | 0.1002 | 0.3603 | 0.2714 | 0.2290 |
| low_downside | neutral | 0.0480 | 0.1508 | 0.3180 | 0.2693 | 0.2018 |
| fund_score | neutral | 0.0347 | 0.1232 | 0.2819 | 0.1524 | 0.1624 |
| fund_pe | neutral | 0.0296 | 0.1197 | 0.2469 | 0.1809 | 0.1458 |
| fund_revenue_growth | neutral | 0.0210 | 0.0975 | 0.2159 | 0.2025 | 0.1298 |
| low_downside | bull | 0.1043 | 0.1110 | 0.9389 | 0.6439 | 0.7717 |
| volatility | bull | 0.1150 | 0.1349 | 0.8520 | 0.6136 | 0.6874 |
| trend_lowvol | bull | 0.0863 | 0.1404 | 0.6143 | 0.4848 | 0.4561 |
| fund_pb | bull | 0.0833 | 0.1371 | 0.6078 | 0.4545 | 0.4420 |
| turnover_stability | bull | 0.0331 | 0.0715 | 0.4632 | 0.4167 | 0.3281 |
| mom_x_lowvol_20_20 | bull | 0.0609 | 0.1334 | 0.4567 | 0.3106 | 0.2993 |
| momentum_reversal | bull | 0.0618 | 0.1337 | 0.4619 | 0.2879 | 0.2975 |
| rsi_vol_combo | bull | 0.0391 | 0.1261 | 0.3098 | 0.1515 | 0.1784 |
| fund_pe | bull | 0.0240 | 0.1209 | 0.1989 | 0.1855 | 0.1179 |
| stroke_phase | bull | 0.0172 | 0.0896 | 0.1920 | 0.1136 | 0.1069 |
| momentum_reversal | bear | 0.1289 | 0.1798 | 0.7167 | 0.3973 | 0.5007 |
| mom_x_lowvol_20_20 | bear | 0.1237 | 0.1840 | 0.6724 | 0.4795 | 0.4974 |
| rsi_vol_combo | bear | 0.0838 | 0.1335 | 0.6280 | 0.2877 | 0.4043 |
| trend_lowvol | bear | 0.1154 | 0.1926 | 0.5992 | 0.3425 | 0.4022 |
| turnover_stability | bear | 0.0467 | 0.0877 | 0.5325 | 0.3699 | 0.3647 |
| fund_pe | bear | 0.0358 | 0.1545 | 0.2316 | 0.1045 | 0.1279 |

### 2025年报预减

- **Neutral**: ['mom_x_lowvol_20_20', 'trend_lowvol'] (单因子IC=0.0796, 组合IC=0.0925)
  - weights: [0.5047, 0.4953]
- **Bull**: ['low_downside', 'volatility', 'turnover_stability'] (单因子IC=0.0751, 组合IC=0.0971)
  - bull_weights: [0.3781, 0.3383, 0.2836]
- **Bear**: ['mom_x_lowvol_20_20', 'momentum_reversal', 'trend_lowvol'] (单因子IC=0.1153, 组合IC=0.1318)
  - bear_weights: [0.3877, 0.3341, 0.2782]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| mom_x_lowvol_20_20 | neutral | 0.0757 | 0.1541 | 0.4909 | 0.3570 | 0.3331 |
| trend_lowvol | neutral | 0.0836 | 0.1716 | 0.4869 | 0.3424 | 0.3268 |
| momentum_reversal | neutral | 0.0709 | 0.1619 | 0.4382 | 0.3069 | 0.2864 |
| fund_pb | neutral | 0.0674 | 0.1565 | 0.4309 | 0.2902 | 0.2780 |
| turnover_stability | neutral | 0.0333 | 0.0931 | 0.3571 | 0.2839 | 0.2292 |
| volatility | neutral | 0.0592 | 0.1690 | 0.3503 | 0.2735 | 0.2231 |
| rsi_vol_combo | neutral | 0.0457 | 0.1468 | 0.3113 | 0.2234 | 0.1904 |
| fund_pe | neutral | 0.0379 | 0.1522 | 0.2491 | 0.1816 | 0.1472 |
| fund_profit_growth | neutral | 0.0259 | 0.1054 | 0.2454 | 0.1628 | 0.1427 |
| low_downside | neutral | 0.0319 | 0.1574 | 0.2026 | 0.1921 | 0.1207 |
| low_downside | bull | 0.0887 | 0.1271 | 0.6980 | 0.4773 | 0.5156 |
| volatility | bull | 0.0949 | 0.1542 | 0.6150 | 0.5000 | 0.4613 |
| turnover_stability | bull | 0.0418 | 0.0753 | 0.5548 | 0.3939 | 0.3867 |
| fund_pb | bull | 0.0855 | 0.1589 | 0.5379 | 0.3712 | 0.3688 |
| momentum_reversal | bull | 0.0652 | 0.1441 | 0.4527 | 0.4318 | 0.3241 |
| trend_lowvol | bull | 0.0697 | 0.1538 | 0.4536 | 0.3409 | 0.3041 |
| mom_x_lowvol_20_20 | bull | 0.0560 | 0.1440 | 0.3889 | 0.3939 | 0.2711 |
| rsi_vol_combo | bull | 0.0380 | 0.1282 | 0.2962 | 0.3030 | 0.1930 |
| fund_pe | bull | 0.0431 | 0.1373 | 0.3142 | 0.1818 | 0.1857 |
| mom_x_lowvol_20_20 | bear | 0.1268 | 0.2013 | 0.6300 | 0.3151 | 0.4143 |
| momentum_reversal | bear | 0.1220 | 0.2059 | 0.5923 | 0.2055 | 0.3570 |
| trend_lowvol | bear | 0.0971 | 0.2104 | 0.4616 | 0.2877 | 0.2972 |
| turnover_stability | bear | 0.0319 | 0.0828 | 0.3858 | 0.3699 | 0.2642 |
| fund_revenue_growth | bear | 0.0423 | 0.1023 | 0.4138 | 0.2329 | 0.2551 |
| rsi_vol_combo | bear | 0.0727 | 0.1696 | 0.4282 | 0.1781 | 0.2523 |
| bb_width_20 | bear | 0.0698 | 0.1756 | 0.3972 | 0.2603 | 0.2503 |
| fund_score | bear | 0.0392 | 0.1265 | 0.3098 | 0.1781 | 0.1825 |

### 2026一季报预增

- **Neutral**: ['fund_pe', 'fund_pb', 'mom_x_lowvol_20_20'] (单因子IC=0.048, 组合IC=0.06)
  - weights: [0.3593, 0.3323, 0.3083]
- **Bull**: ['low_downside', 'fund_pb'] (单因子IC=0.0471, 组合IC=0.0578)
  - bull_weights: [0.5085, 0.4915]
- **Bear**: ['bb_width_20', 'mom_x_lowvol_20_20'] (单因子IC=0.0711, 组合IC=0.0738)
  - bear_weights: [0.5105, 0.4895]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| fund_pe | neutral | 0.0465 | 0.1532 | 0.3037 | 0.2380 | 0.1880 |
| fund_pb | neutral | 0.0531 | 0.1778 | 0.2985 | 0.1649 | 0.1739 |
| mom_x_lowvol_20_20 | neutral | 0.0445 | 0.1639 | 0.2716 | 0.1879 | 0.1613 |
| fund_profit_growth | neutral | 0.0389 | 0.1496 | 0.2598 | 0.1921 | 0.1548 |
| momentum_reversal | neutral | 0.0455 | 0.1772 | 0.2569 | 0.1962 | 0.1536 |
| trend_lowvol | neutral | 0.0488 | 0.2014 | 0.2421 | 0.1420 | 0.1382 |
| fund_score | neutral | 0.0311 | 0.1551 | 0.2004 | 0.1942 | 0.1197 |
| rsi_vol_combo | neutral | 0.0349 | 0.1775 | 0.1966 | 0.1315 | 0.1112 |
| volatility | neutral | 0.0341 | 0.1931 | 0.1765 | 0.1441 | 0.1010 |
| turnover_stability | neutral | 0.0138 | 0.1293 | 0.1066 | 0.1106 | 0.0592 |
| low_downside | bull | 0.0475 | 0.1694 | 0.2803 | 0.1742 | 0.1646 |
| fund_pb | bull | 0.0467 | 0.1823 | 0.2561 | 0.2424 | 0.1591 |
| fund_pe | bull | 0.0362 | 0.1479 | 0.2447 | 0.1667 | 0.1427 |
| rsi_vol_combo | bull | 0.0194 | 0.1524 | 0.1271 | 0.1439 | 0.0727 |
| momentum_reversal | bull | 0.0242 | 0.1868 | 0.1294 | 0.1136 | 0.0721 |
| bb_width_20 | bear | 0.0723 | 0.1765 | 0.4097 | 0.2603 | 0.2582 |
| mom_x_lowvol_20_20 | bear | 0.0699 | 0.1935 | 0.3614 | 0.3699 | 0.2475 |
| vol_confirm | bear | 0.0366 | 0.1129 | 0.3240 | 0.3425 | 0.2175 |
| momentum_reversal | bear | 0.0686 | 0.2039 | 0.3363 | 0.2329 | 0.2073 |
| limit_pullback_score | bear | 0.0384 | 0.1198 | 0.3206 | 0.1831 | 0.1897 |
| trend_lowvol | bear | 0.0574 | 0.2580 | 0.2227 | 0.2055 | 0.1342 |
| top_fractal_volume | bear | 0.0236 | 0.1093 | 0.2156 | 0.2143 | 0.1309 |
| fund_gross_margin | bear | 0.0270 | 0.1350 | 0.2001 | 0.2329 | 0.1233 |
| wash_sale_score | bear | 0.0229 | 0.1121 | 0.2043 | 0.1803 | 0.1206 |
| fund_pe | bear | 0.0382 | 0.2030 | 0.1879 | 0.2603 | 0.1184 |

### 3D打印

- **Neutral**: ['momentum_reversal', 'mom_x_lowvol_20_20', 'fund_pb'] (单因子IC=0.0806, 组合IC=0.1176)
  - weights: [0.3635, 0.3197, 0.3168]
- **Bull**: ['volatility', 'fund_revenue_growth', 'low_downside'] (单因子IC=0.0855, 组合IC=0.1191)
  - bull_weights: [0.3436, 0.3386, 0.3179]
- **Bear**: ['fund_profit_growth', 'trend_lowvol', 'rsi_vol_combo'] (单因子IC=0.0443, 组合IC=0.0716)
  - bear_weights: [0.4602, 0.283, 0.2569]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| momentum_reversal | neutral | 0.0833 | 0.1933 | 0.4310 | 0.3570 | 0.2924 |
| mom_x_lowvol_20_20 | neutral | 0.0717 | 0.1854 | 0.3866 | 0.3309 | 0.2572 |
| fund_pb | neutral | 0.0867 | 0.2145 | 0.4043 | 0.2610 | 0.2549 |
| volatility | neutral | 0.0737 | 0.1910 | 0.3861 | 0.2797 | 0.2470 |
| trend_lowvol | neutral | 0.0785 | 0.2111 | 0.3721 | 0.2923 | 0.2404 |
| rsi_vol_combo | neutral | 0.0622 | 0.1827 | 0.3406 | 0.3152 | 0.2240 |
| fund_pe | neutral | 0.0554 | 0.2078 | 0.2666 | 0.2443 | 0.1658 |
| turnover_stability | neutral | 0.0379 | 0.1415 | 0.2678 | 0.2192 | 0.1632 |
| fund_profit_growth | neutral | 0.0404 | 0.1622 | 0.2491 | 0.1816 | 0.1471 |
| fund_score | neutral | 0.0397 | 0.1853 | 0.2142 | 0.1566 | 0.1239 |
| low_downside | neutral | 0.0318 | 0.1862 | 0.1707 | 0.1367 | 0.0970 |
| fund_revenue_growth | neutral | 0.0213 | 0.1561 | 0.1362 | 0.1023 | 0.0751 |
| volatility | bull | 0.0872 | 0.1808 | 0.4820 | 0.4242 | 0.3433 |
| fund_revenue_growth | bull | 0.0856 | 0.1803 | 0.4750 | 0.4242 | 0.3382 |
| low_downside | bull | 0.0838 | 0.1790 | 0.4683 | 0.3561 | 0.3176 |
| trend_lowvol | bull | 0.1005 | 0.2239 | 0.4489 | 0.3106 | 0.2941 |
| turnover_stability | bull | 0.0599 | 0.1421 | 0.4217 | 0.3674 | 0.2883 |
| fund_pb | bull | 0.0764 | 0.1977 | 0.3862 | 0.3258 | 0.2560 |
| mom_x_lowvol_20_20 | bull | 0.0701 | 0.1886 | 0.3716 | 0.2803 | 0.2378 |
| fund_score | bull | 0.0597 | 0.1746 | 0.3420 | 0.2955 | 0.2215 |
| momentum_reversal | bull | 0.0657 | 0.1913 | 0.3436 | 0.2273 | 0.2108 |
| rsi_vol_combo | bull | 0.0409 | 0.1555 | 0.2630 | 0.1970 | 0.1574 |
| fund_profit_growth | bull | 0.0377 | 0.1505 | 0.2507 | 0.2500 | 0.1567 |
| fund_gross_margin | bull | 0.0371 | 0.1567 | 0.2367 | 0.2121 | 0.1434 |
| fund_pe | bull | 0.0341 | 0.1843 | 0.1851 | 0.1515 | 0.1066 |
| fund_roe | bull | 0.0196 | 0.1712 | 0.1147 | 0.1288 | 0.0647 |
| fund_profit_growth | bear | 0.0493 | 0.1649 | 0.2991 | 0.2603 | 0.1885 |
| trend_lowvol | bear | 0.0477 | 0.2482 | 0.1923 | 0.2055 | 0.1159 |
| rsi_vol_combo | bear | 0.0360 | 0.2014 | 0.1786 | 0.1781 | 0.1052 |

### 5G概念

- **Neutral**: ['fund_pb', 'mom_x_lowvol_20_20', 'momentum_reversal'] (单因子IC=0.0815, 组合IC=0.1097)
  - weights: [0.3418, 0.3397, 0.3185]
- **Bull**: ['fund_pb', 'volatility', 'low_downside'] (单因子IC=0.0853, 组合IC=0.1079)
  - bull_weights: [0.3569, 0.3486, 0.2945]
- **Bear**: ['mom_x_lowvol_20_20'] (单因子IC=0.1004, 组合IC=0.1004)
  - bear_weights: [1.0]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| fund_pb | neutral | 0.0751 | 0.1352 | 0.5553 | 0.3967 | 0.3878 |
| mom_x_lowvol_20_20 | neutral | 0.0852 | 0.1555 | 0.5478 | 0.4071 | 0.3854 |
| momentum_reversal | neutral | 0.0843 | 0.1617 | 0.5214 | 0.3862 | 0.3614 |
| trend_lowvol | neutral | 0.0883 | 0.1768 | 0.4994 | 0.3758 | 0.3436 |
| rsi_vol_combo | neutral | 0.0666 | 0.1581 | 0.4213 | 0.3111 | 0.2762 |
| fund_pe | neutral | 0.0574 | 0.1404 | 0.4085 | 0.3424 | 0.2742 |
| volatility | neutral | 0.0627 | 0.1740 | 0.3605 | 0.2944 | 0.2333 |
| fund_profit_growth | neutral | 0.0445 | 0.1298 | 0.3428 | 0.2505 | 0.2143 |
| turnover_stability | neutral | 0.0319 | 0.0978 | 0.3263 | 0.2985 | 0.2119 |
| fund_score | neutral | 0.0444 | 0.1727 | 0.2569 | 0.1962 | 0.1537 |
| fund_revenue_growth | neutral | 0.0300 | 0.1369 | 0.2192 | 0.1942 | 0.1309 |
| fund_roe | neutral | 0.0317 | 0.1858 | 0.1706 | 0.1399 | 0.0972 |
| fund_pb | bull | 0.0866 | 0.1469 | 0.5898 | 0.4318 | 0.4223 |
| volatility | bull | 0.0916 | 0.1623 | 0.5642 | 0.4621 | 0.4125 |
| low_downside | bull | 0.0778 | 0.1547 | 0.5027 | 0.3864 | 0.3484 |
| fund_pe | bull | 0.0699 | 0.1402 | 0.4984 | 0.3712 | 0.3417 |
| turnover_stability | bull | 0.0343 | 0.0812 | 0.4222 | 0.3409 | 0.2831 |
| fund_profit_growth | bull | 0.0501 | 0.1361 | 0.3678 | 0.3561 | 0.2494 |
| trend_lowvol | bull | 0.0579 | 0.1813 | 0.3195 | 0.2803 | 0.2045 |
| mom_x_lowvol_20_20 | bull | 0.0425 | 0.1402 | 0.3031 | 0.2121 | 0.1837 |
| fund_revenue_growth | bull | 0.0464 | 0.1596 | 0.2908 | 0.2576 | 0.1828 |
| fund_score | bull | 0.0551 | 0.1830 | 0.3010 | 0.1970 | 0.1801 |
| momentum_reversal | bull | 0.0273 | 0.1460 | 0.1868 | 0.1364 | 0.1062 |
| mom_x_lowvol_20_20 | bear | 0.1004 | 0.1623 | 0.6182 | 0.3973 | 0.4319 |
| momentum_reversal | bear | 0.0986 | 0.1624 | 0.6069 | 0.3425 | 0.4074 |
| rsi_vol_combo | bear | 0.0704 | 0.1367 | 0.5149 | 0.4521 | 0.3738 |
| trend_lowvol | bear | 0.1065 | 0.1854 | 0.5745 | 0.2877 | 0.3699 |
| fund_profit_growth | bear | 0.0407 | 0.1246 | 0.3267 | 0.1233 | 0.1835 |

### 6G概念

- **Neutral**: ['fund_pb', 'mom_x_lowvol_20_20', 'momentum_reversal'] (单因子IC=0.0919, 组合IC=0.1188)
  - weights: [0.3563, 0.3277, 0.3159]
- **Bull**: ['fund_pb', 'low_downside'] (单因子IC=0.1264, 组合IC=0.1565)
  - bull_weights: [0.6141, 0.3859]
- **Bear**: ['mom_x_lowvol_20_20', 'momentum_reversal', 'trend_lowvol'] (单因子IC=0.1571, 组合IC=0.1841)
  - bear_weights: [0.3615, 0.3276, 0.311]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| fund_pb | neutral | 0.0901 | 0.2286 | 0.3943 | 0.3090 | 0.2581 |
| mom_x_lowvol_20_20 | neutral | 0.0940 | 0.2506 | 0.3753 | 0.2651 | 0.2374 |
| momentum_reversal | neutral | 0.0914 | 0.2531 | 0.3611 | 0.2672 | 0.2288 |
| fund_profit_growth | neutral | 0.0569 | 0.1994 | 0.2853 | 0.2276 | 0.1751 |
| trend_lowvol | neutral | 0.0723 | 0.2631 | 0.2750 | 0.2338 | 0.1697 |
| rsi_vol_combo | neutral | 0.0646 | 0.2343 | 0.2759 | 0.2004 | 0.1656 |
| fund_pe | neutral | 0.0727 | 0.2739 | 0.2653 | 0.2234 | 0.1623 |
| volatility | neutral | 0.0607 | 0.2869 | 0.2116 | 0.1795 | 0.1248 |
| fund_revenue_growth | neutral | 0.0249 | 0.2041 | 0.1222 | 0.1106 | 0.0679 |
| fund_pb | bull | 0.1476 | 0.2282 | 0.6466 | 0.4394 | 0.4654 |
| low_downside | bull | 0.1052 | 0.2412 | 0.4361 | 0.3409 | 0.2924 |
| volatility | bull | 0.0975 | 0.2530 | 0.3853 | 0.3409 | 0.2583 |
| fund_score | bull | 0.0920 | 0.2746 | 0.3351 | 0.2803 | 0.2145 |
| fund_pe | bull | 0.0799 | 0.2610 | 0.3062 | 0.2500 | 0.1914 |
| vol_confirm | bull | 0.0557 | 0.2096 | 0.2656 | 0.1667 | 0.1549 |
| fund_profit_growth | bull | 0.0556 | 0.2192 | 0.2534 | 0.1894 | 0.1507 |
| fund_revenue_growth | bull | 0.0443 | 0.2306 | 0.1920 | 0.1515 | 0.1105 |
| exhaustion_risk | bull | 0.0373 | 0.2044 | 0.1822 | 0.1281 | 0.1028 |
| fund_roe | bull | 0.0382 | 0.2331 | 0.1638 | 0.1515 | 0.0943 |
| mom_x_lowvol_20_20 | bear | 0.1570 | 0.2421 | 0.6486 | 0.5616 | 0.5064 |
| momentum_reversal | bear | 0.1603 | 0.2584 | 0.6203 | 0.4795 | 0.4589 |
| trend_lowvol | bear | 0.1538 | 0.2612 | 0.5889 | 0.4795 | 0.4356 |
| bb_width_20 | bear | 0.0993 | 0.1923 | 0.5163 | 0.3699 | 0.3537 |
| rsi_vol_combo | bear | 0.1231 | 0.2246 | 0.5478 | 0.2877 | 0.3527 |
| fund_gross_margin | bear | 0.0975 | 0.2562 | 0.3806 | 0.2877 | 0.2451 |
| fund_profit_growth | bear | 0.0444 | 0.1832 | 0.2426 | 0.1781 | 0.1429 |

### AB股

- **Neutral**: ['mom_x_lowvol_20_20', 'trend_lowvol'] (单因子IC=0.0824, 组合IC=0.0999)
  - weights: [0.5016, 0.4984]
- **Bull**: ['low_downside', 'fund_pb', 'volatility'] (单因子IC=0.0876, 组合IC=0.1108)
  - bull_weights: [0.3964, 0.3832, 0.2203]
- **Bear**: ['mom_x_lowvol_20_20', 'momentum_reversal'] (单因子IC=0.1527, 组合IC=0.1602)
  - bear_weights: [0.5971, 0.4029]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| mom_x_lowvol_20_20 | neutral | 0.0767 | 0.1723 | 0.4455 | 0.3132 | 0.2925 |
| trend_lowvol | neutral | 0.0882 | 0.2020 | 0.4364 | 0.3319 | 0.2906 |
| momentum_reversal | neutral | 0.0806 | 0.1939 | 0.4157 | 0.3069 | 0.2716 |
| fund_pb | neutral | 0.0632 | 0.1635 | 0.3863 | 0.2839 | 0.2480 |
| rsi_vol_combo | neutral | 0.0594 | 0.1856 | 0.3202 | 0.2286 | 0.1967 |
| turnover_stability | neutral | 0.0462 | 0.1537 | 0.3009 | 0.2296 | 0.1850 |
| low_downside | neutral | 0.0527 | 0.1831 | 0.2881 | 0.2484 | 0.1798 |
| volatility | neutral | 0.0488 | 0.1851 | 0.2634 | 0.2171 | 0.1603 |
| fund_pe | neutral | 0.0423 | 0.1932 | 0.2190 | 0.2025 | 0.1317 |
| fund_score | neutral | 0.0293 | 0.1471 | 0.1990 | 0.1712 | 0.1165 |
| fund_roe | neutral | 0.0317 | 0.1690 | 0.1874 | 0.1399 | 0.1068 |
| fund_profit_growth | neutral | 0.0202 | 0.1328 | 0.1523 | 0.1232 | 0.0856 |
| low_downside | bull | 0.1032 | 0.1728 | 0.5969 | 0.4242 | 0.4250 |
| fund_pb | bull | 0.0934 | 0.1645 | 0.5680 | 0.4470 | 0.4109 |
| volatility | bull | 0.0661 | 0.1802 | 0.3669 | 0.2879 | 0.2362 |
| turnover_stability | bull | 0.0575 | 0.1568 | 0.3669 | 0.2803 | 0.2349 |
| momentum_reversal | bull | 0.0580 | 0.1671 | 0.3468 | 0.3409 | 0.2325 |
| trend_lowvol | bull | 0.0650 | 0.2056 | 0.3162 | 0.2121 | 0.1917 |
| rsi_vol_combo | bull | 0.0375 | 0.1624 | 0.2308 | 0.1667 | 0.1347 |
| mom_x_lowvol_20_20 | bull | 0.0385 | 0.1705 | 0.2260 | 0.1288 | 0.1275 |
| fund_pe | bull | 0.0276 | 0.2123 | 0.1302 | 0.1061 | 0.0720 |
| mom_x_lowvol_20_20 | bear | 0.1508 | 0.1733 | 0.8706 | 0.5342 | 0.6678 |
| momentum_reversal | bear | 0.1546 | 0.2257 | 0.6853 | 0.3151 | 0.4506 |
| rsi_vol_combo | bear | 0.1144 | 0.2169 | 0.5276 | 0.3699 | 0.3614 |
| trend_lowvol | bear | 0.0840 | 0.1865 | 0.4507 | 0.1507 | 0.2593 |
| turnover_stability | bear | 0.0406 | 0.1818 | 0.2233 | 0.1781 | 0.1315 |
| fund_score | bear | 0.0359 | 0.1599 | 0.2248 | 0.1507 | 0.1293 |

### AH股

- **Neutral**: ['fund_profit_growth', 'fund_pe', 'fund_pb'] (单因子IC=0.0611, 组合IC=0.0745)
  - weights: [0.3741, 0.3429, 0.283]
- **Bull**: ['low_downside'] (单因子IC=0.0533, 组合IC=0.0533)
  - bull_weights: [1.0]
- **Bear**: ['mom_x_lowvol_20_20', 'momentum_reversal', 'bb_width_20'] (单因子IC=0.1241, 组合IC=0.1377)
  - bear_weights: [0.3818, 0.3265, 0.2917]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| fund_profit_growth | neutral | 0.0404 | 0.1342 | 0.3008 | 0.2693 | 0.1909 |
| fund_pe | neutral | 0.0775 | 0.2635 | 0.2941 | 0.1900 | 0.1750 |
| fund_pb | neutral | 0.0654 | 0.2662 | 0.2458 | 0.1754 | 0.1444 |
| volatility | neutral | 0.0438 | 0.2045 | 0.2141 | 0.1962 | 0.1281 |
| trend_lowvol | neutral | 0.0497 | 0.2293 | 0.2167 | 0.1754 | 0.1273 |
| low_downside | neutral | 0.0436 | 0.2096 | 0.2079 | 0.1461 | 0.1192 |
| fund_score | neutral | 0.0318 | 0.1587 | 0.2001 | 0.1148 | 0.1116 |
| turnover_stability | neutral | 0.0234 | 0.1293 | 0.1813 | 0.1712 | 0.1062 |
| stroke_phase | neutral | 0.0182 | 0.1325 | 0.1375 | 0.1002 | 0.0757 |
| top_fractal_volume | neutral | 0.0113 | 0.0905 | 0.1249 | 0.1288 | 0.0705 |
| low_downside | bull | 0.0533 | 0.2004 | 0.2658 | 0.1212 | 0.1490 |
| turnover_stability | bull | 0.0297 | 0.1288 | 0.2302 | 0.2576 | 0.1447 |
| volatility | bull | 0.0473 | 0.2106 | 0.2247 | 0.1742 | 0.1319 |
| relative_strength | bull | 0.0250 | 0.2019 | 0.1239 | 0.2576 | 0.0779 |
| stroke_phase | bull | 0.0144 | 0.1261 | 0.1139 | 0.1591 | 0.0660 |
| mom_x_lowvol_20_20 | bear | 0.1233 | 0.2226 | 0.5537 | 0.5068 | 0.4172 |
| momentum_reversal | bear | 0.1245 | 0.2486 | 0.5008 | 0.4247 | 0.3567 |
| bb_width_20 | bear | 0.1245 | 0.2568 | 0.4848 | 0.3151 | 0.3188 |
| fund_profit_growth | bear | 0.0440 | 0.1104 | 0.3984 | 0.4795 | 0.2947 |
| trend_lowvol | bear | 0.1242 | 0.2997 | 0.4146 | 0.2877 | 0.2669 |
| fund_revenue_growth | bear | 0.0492 | 0.1590 | 0.3095 | 0.1233 | 0.1738 |
| fund_score | bear | 0.0308 | 0.1210 | 0.2543 | 0.2603 | 0.1603 |
| wash_sale_score | bear | 0.0217 | 0.0906 | 0.2399 | 0.2571 | 0.1508 |
| rsi_vol_combo | bear | 0.0535 | 0.2157 | 0.2478 | 0.2055 | 0.1493 |

### AIGC概念

- **Neutral**: ['momentum_reversal', 'volatility', 'fund_pb'] (单因子IC=0.1108, 组合IC=0.1533)
  - weights: [0.3521, 0.3251, 0.3227]
- **Bull**: ['fund_pb', 'fund_pe', 'volatility'] (单因子IC=0.0961, 组合IC=0.1108)
  - bull_weights: [0.4382, 0.2902, 0.2716]
- **Bear**: ['fund_profit_growth', 'fund_score', 'mom_x_lowvol_20_20'] (单因子IC=0.1017, 组合IC=0.1237)
  - bear_weights: [0.364, 0.328, 0.308]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| momentum_reversal | neutral | 0.1149 | 0.1878 | 0.6118 | 0.4280 | 0.4368 |
| volatility | neutral | 0.1194 | 0.2101 | 0.5682 | 0.4196 | 0.4033 |
| fund_pb | neutral | 0.0982 | 0.1760 | 0.5583 | 0.4342 | 0.4003 |
| mom_x_lowvol_20_20 | neutral | 0.1068 | 0.1902 | 0.5614 | 0.4175 | 0.3979 |
| trend_lowvol | neutral | 0.1125 | 0.2012 | 0.5589 | 0.4092 | 0.3938 |
| rsi_vol_combo | neutral | 0.0851 | 0.1772 | 0.4802 | 0.3716 | 0.3293 |
| fund_pe | neutral | 0.0569 | 0.1531 | 0.3717 | 0.2735 | 0.2367 |
| low_downside | neutral | 0.0649 | 0.1939 | 0.3349 | 0.2422 | 0.2080 |
| turnover_stability | neutral | 0.0359 | 0.1291 | 0.2784 | 0.2505 | 0.1741 |
| fund_score | neutral | 0.0424 | 0.1598 | 0.2652 | 0.2443 | 0.1650 |
| fund_profit_growth | neutral | 0.0353 | 0.1400 | 0.2523 | 0.1816 | 0.1490 |
| fund_revenue_growth | neutral | 0.0223 | 0.1288 | 0.1735 | 0.1670 | 0.1012 |
| fund_roe | neutral | 0.0279 | 0.1577 | 0.1769 | 0.1357 | 0.1005 |
| fund_pb | bull | 0.1106 | 0.1455 | 0.7601 | 0.5985 | 0.6075 |
| fund_pe | bull | 0.0839 | 0.1493 | 0.5620 | 0.4318 | 0.4023 |
| volatility | bull | 0.0938 | 0.1717 | 0.5462 | 0.3788 | 0.3766 |
| trend_lowvol | bull | 0.0967 | 0.1832 | 0.5277 | 0.4091 | 0.3718 |
| low_downside | bull | 0.0684 | 0.1665 | 0.4106 | 0.3712 | 0.2815 |
| fund_profit_growth | bull | 0.0448 | 0.1260 | 0.3553 | 0.3106 | 0.2328 |
| fund_score | bull | 0.0495 | 0.1435 | 0.3451 | 0.2576 | 0.2170 |
| mom_x_lowvol_20_20 | bull | 0.0595 | 0.1734 | 0.3430 | 0.1894 | 0.2040 |
| momentum_reversal | bull | 0.0518 | 0.1703 | 0.3041 | 0.1591 | 0.1763 |
| fund_roe | bull | 0.0405 | 0.1507 | 0.2690 | 0.3030 | 0.1752 |
| fund_revenue_growth | bull | 0.0286 | 0.1121 | 0.2554 | 0.2348 | 0.1577 |
| fund_profit_growth | bear | 0.0951 | 0.1456 | 0.6528 | 0.5616 | 0.5097 |
| fund_score | bear | 0.1007 | 0.1742 | 0.5782 | 0.5890 | 0.4594 |
| mom_x_lowvol_20_20 | bear | 0.1092 | 0.1873 | 0.5831 | 0.4795 | 0.4313 |
| trend_lowvol | bear | 0.0908 | 0.1703 | 0.5332 | 0.4521 | 0.3871 |
| momentum_reversal | bear | 0.0960 | 0.2027 | 0.4738 | 0.3973 | 0.3310 |
| fund_revenue_growth | bear | 0.0737 | 0.1584 | 0.4656 | 0.3151 | 0.3061 |
| turnover_stability | bear | 0.0433 | 0.1067 | 0.4061 | 0.1781 | 0.2392 |
| fund_roe | bear | 0.0502 | 0.1836 | 0.2732 | 0.1233 | 0.1535 |

### AIPC

- **Neutral**: ['momentum_reversal', 'mom_x_lowvol_20_20', 'fund_pb'] (单因子IC=0.0691, 组合IC=0.0927)
  - weights: [0.3386, 0.3323, 0.329]
- **Bull**: ['fund_pb', 'volatility'] (单因子IC=0.0815, 组合IC=0.1077)
  - bull_weights: [0.5637, 0.4363]
- **Bear**: ['fund_gross_margin', 'rsi_vol_combo', 'fund_profit_growth'] (单因子IC=0.0975, 组合IC=0.1579)
  - bear_weights: [0.5921, 0.2291, 0.1788]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| momentum_reversal | neutral | 0.0702 | 0.2440 | 0.2877 | 0.1931 | 0.1716 |
| mom_x_lowvol_20_20 | neutral | 0.0688 | 0.2457 | 0.2799 | 0.2035 | 0.1684 |
| fund_pb | neutral | 0.0683 | 0.2397 | 0.2848 | 0.1712 | 0.1668 |
| rsi_vol_combo | neutral | 0.0684 | 0.2460 | 0.2781 | 0.1879 | 0.1652 |
| volatility | neutral | 0.0642 | 0.2489 | 0.2582 | 0.1795 | 0.1523 |
| trend_lowvol | neutral | 0.0673 | 0.2670 | 0.2519 | 0.1848 | 0.1492 |
| fund_pe | neutral | 0.0425 | 0.2217 | 0.1916 | 0.1441 | 0.1096 |
| low_downside | neutral | 0.0405 | 0.2154 | 0.1880 | 0.1430 | 0.1075 |
| fund_pb | bull | 0.0922 | 0.2397 | 0.3846 | 0.2538 | 0.2411 |
| volatility | bull | 0.0708 | 0.2377 | 0.2977 | 0.2538 | 0.1866 |
| turnover_stability | bull | 0.0355 | 0.1766 | 0.2009 | 0.1818 | 0.1187 |
| fund_gross_margin | bear | 0.1476 | 0.1858 | 0.7946 | 0.6438 | 0.6531 |
| rsi_vol_combo | bear | 0.0729 | 0.1896 | 0.3843 | 0.3151 | 0.2527 |
| fund_profit_growth | bear | 0.0719 | 0.2347 | 0.3062 | 0.2877 | 0.1972 |
| mom_x_lowvol_20_20 | bear | 0.0628 | 0.2369 | 0.2651 | 0.2055 | 0.1598 |
| momentum_reversal | bear | 0.0594 | 0.2336 | 0.2545 | 0.1918 | 0.1517 |
| bb_width_20 | bear | 0.0632 | 0.2502 | 0.2528 | 0.1507 | 0.1454 |

### AI制药（医疗）

- **Neutral**: ['fund_pb', 'volatility', 'momentum_reversal'] (单因子IC=0.0725, 组合IC=0.1102)
  - weights: [0.3675, 0.317, 0.3155]
- **Bull**: ['turnover_stability', 'fund_pb'] (单因子IC=0.0721, 组合IC=0.0959)
  - bull_weights: [0.5047, 0.4953]
- **Bear**: ['trend_lowvol'] (单因子IC=0.1653, 组合IC=0.1653)
  - bear_weights: [1.0]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| fund_pb | neutral | 0.0677 | 0.1886 | 0.3590 | 0.3340 | 0.2395 |
| volatility | neutral | 0.0782 | 0.2364 | 0.3309 | 0.2484 | 0.2065 |
| momentum_reversal | neutral | 0.0715 | 0.2158 | 0.3315 | 0.2401 | 0.2056 |
| trend_lowvol | neutral | 0.0656 | 0.2233 | 0.2937 | 0.1900 | 0.1748 |
| mom_x_lowvol_20_20 | neutral | 0.0611 | 0.2155 | 0.2836 | 0.2276 | 0.1741 |
| rsi_vol_combo | neutral | 0.0442 | 0.1975 | 0.2236 | 0.1587 | 0.1296 |
| low_downside | neutral | 0.0458 | 0.2120 | 0.2160 | 0.1795 | 0.1274 |
| fund_pe | neutral | 0.0404 | 0.2163 | 0.1870 | 0.1211 | 0.1048 |
| turnover_stability | neutral | 0.0201 | 0.1585 | 0.1267 | 0.1065 | 0.0701 |
| turnover_stability | bull | 0.0623 | 0.1529 | 0.4078 | 0.3182 | 0.2688 |
| fund_pb | bull | 0.0818 | 0.2044 | 0.4002 | 0.3182 | 0.2638 |
| volatility | bull | 0.0736 | 0.2328 | 0.3160 | 0.2121 | 0.1915 |
| rsi_vol_combo | bull | 0.0507 | 0.1733 | 0.2924 | 0.1894 | 0.1739 |
| low_downside | bull | 0.0508 | 0.1998 | 0.2541 | 0.2652 | 0.1608 |
| momentum_reversal | bull | 0.0515 | 0.2030 | 0.2539 | 0.1591 | 0.1471 |
| mom_x_lowvol_20_20 | bull | 0.0497 | 0.2055 | 0.2417 | 0.1515 | 0.1392 |
| fund_profit_growth | bull | 0.0415 | 0.1802 | 0.2305 | 0.1894 | 0.1371 |
| fund_pe | bull | 0.0444 | 0.2206 | 0.2013 | 0.1136 | 0.1121 |
| fund_score | bull | 0.0309 | 0.1909 | 0.1619 | 0.1212 | 0.0908 |
| fund_revenue_growth | bull | 0.0164 | 0.1691 | 0.0972 | 0.2045 | 0.0585 |
| trend_lowvol | bear | 0.1653 | 0.1976 | 0.8365 | 0.5342 | 0.6417 |
| fund_revenue_growth | bear | 0.1073 | 0.1426 | 0.7524 | 0.6164 | 0.6081 |
| turnover_stability | bear | 0.0908 | 0.1590 | 0.5709 | 0.4795 | 0.4223 |
| mom_x_lowvol_20_20 | bear | 0.1102 | 0.2116 | 0.5206 | 0.3699 | 0.3566 |
| momentum_reversal | bear | 0.1193 | 0.2458 | 0.4852 | 0.2877 | 0.3124 |
| fund_score | bear | 0.0787 | 0.1798 | 0.4377 | 0.3973 | 0.3058 |
| fund_profit_growth | bear | 0.0500 | 0.1394 | 0.3586 | 0.4247 | 0.2554 |
| fund_roe | bear | 0.0710 | 0.1899 | 0.3737 | 0.3425 | 0.2508 |
| rsi_vol_combo | bear | 0.0586 | 0.2072 | 0.2827 | 0.3151 | 0.1859 |
| fund_gross_margin | bear | 0.0441 | 0.1829 | 0.2411 | 0.2877 | 0.1552 |
| fund_pe | bear | 0.0537 | 0.2405 | 0.2235 | 0.1781 | 0.1316 |
| fund_pb | bear | 0.0254 | 0.1203 | 0.2108 | 0.1233 | 0.1184 |

### AI应用

- **Neutral**: ['fund_pb', 'momentum_reversal', 'mom_x_lowvol_20_20'] (单因子IC=0.0996, 组合IC=0.1329)
  - weights: [0.3614, 0.3335, 0.3051]
- **Bull**: ['fund_pb', 'volatility'] (单因子IC=0.1074, 组合IC=0.1291)
  - bull_weights: [0.5456, 0.4544]
- **Bear**: ['trend_lowvol', 'mom_x_lowvol_20_20', 'fund_profit_growth'] (单因子IC=0.0886, 组合IC=0.1184)
  - bear_weights: [0.432, 0.3092, 0.2588]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| fund_pb | neutral | 0.0884 | 0.1233 | 0.7173 | 0.5073 | 0.5406 |
| momentum_reversal | neutral | 0.1089 | 0.1612 | 0.6758 | 0.4760 | 0.4988 |
| mom_x_lowvol_20_20 | neutral | 0.1014 | 0.1622 | 0.6253 | 0.4593 | 0.4563 |
| trend_lowvol | neutral | 0.1100 | 0.1791 | 0.6145 | 0.4280 | 0.4387 |
| rsi_vol_combo | neutral | 0.0907 | 0.1517 | 0.5981 | 0.4217 | 0.4252 |
| volatility | neutral | 0.0973 | 0.1833 | 0.5306 | 0.4280 | 0.3789 |
| fund_pe | neutral | 0.0653 | 0.1417 | 0.4610 | 0.3278 | 0.3061 |
| fund_profit_growth | neutral | 0.0452 | 0.1195 | 0.3783 | 0.2944 | 0.2448 |
| low_downside | neutral | 0.0498 | 0.1651 | 0.3019 | 0.2735 | 0.1922 |
| fund_score | neutral | 0.0427 | 0.1487 | 0.2875 | 0.2463 | 0.1792 |
| turnover_stability | neutral | 0.0275 | 0.1149 | 0.2391 | 0.1670 | 0.1395 |
| fund_revenue_growth | neutral | 0.0251 | 0.1144 | 0.2197 | 0.1691 | 0.1284 |
| fund_roe | neutral | 0.0290 | 0.1583 | 0.1834 | 0.1524 | 0.1057 |
| fund_pb | bull | 0.1063 | 0.1233 | 0.8625 | 0.6136 | 0.6959 |
| volatility | bull | 0.1086 | 0.1511 | 0.7183 | 0.6136 | 0.5796 |
| low_downside | bull | 0.0753 | 0.1267 | 0.5941 | 0.4848 | 0.4411 |
| trend_lowvol | bull | 0.0942 | 0.1659 | 0.5679 | 0.4167 | 0.4023 |
| fund_pe | bull | 0.0708 | 0.1405 | 0.5038 | 0.4091 | 0.3550 |
| mom_x_lowvol_20_20 | bull | 0.0749 | 0.1588 | 0.4718 | 0.3864 | 0.3271 |
| momentum_reversal | bull | 0.0643 | 0.1594 | 0.4032 | 0.3712 | 0.2764 |
| rsi_vol_combo | bull | 0.0385 | 0.1548 | 0.2491 | 0.2045 | 0.1500 |
| fund_score | bull | 0.0303 | 0.1382 | 0.2193 | 0.1439 | 0.1254 |
| turnover_stability | bull | 0.0215 | 0.1048 | 0.2052 | 0.1288 | 0.1158 |
| fund_profit_growth | bull | 0.0192 | 0.1075 | 0.1788 | 0.1970 | 0.1070 |
| trend_lowvol | bear | 0.1045 | 0.1547 | 0.6756 | 0.5342 | 0.5183 |
| mom_x_lowvol_20_20 | bear | 0.0962 | 0.1811 | 0.5310 | 0.3973 | 0.3710 |
| fund_profit_growth | bear | 0.0652 | 0.1352 | 0.4823 | 0.2877 | 0.3105 |
| momentum_reversal | bear | 0.0831 | 0.1910 | 0.4352 | 0.3973 | 0.3040 |
| fund_revenue_growth | bear | 0.0528 | 0.1401 | 0.3764 | 0.2877 | 0.2424 |
| fund_score | bear | 0.0666 | 0.1729 | 0.3851 | 0.1781 | 0.2269 |
| turnover_stability | bear | 0.0299 | 0.1139 | 0.2623 | 0.2877 | 0.1689 |
| rsi_vol_combo | bear | 0.0306 | 0.1740 | 0.1759 | 0.1507 | 0.1012 |

### AI手机

- **Neutral**: ['fund_pb', 'momentum_reversal', 'mom_x_lowvol_20_20'] (单因子IC=0.0867, 组合IC=0.1232)
  - weights: [0.3994, 0.3047, 0.2959]
- **Bull**: ['turnover_stability', 'volatility', 'fund_pb'] (单因子IC=0.1134, 组合IC=0.1661)
  - bull_weights: [0.3643, 0.3258, 0.3099]
- **Bear**: ['fund_pb', 'momentum_reversal'] (单因子IC=0.1103, 组合IC=0.1587)
  - bear_weights: [0.5766, 0.4234]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| fund_pb | neutral | 0.1019 | 0.2470 | 0.4123 | 0.2860 | 0.2651 |
| momentum_reversal | neutral | 0.0799 | 0.2491 | 0.3209 | 0.2610 | 0.2023 |
| mom_x_lowvol_20_20 | neutral | 0.0784 | 0.2460 | 0.3187 | 0.2328 | 0.1964 |
| volatility | neutral | 0.0783 | 0.2483 | 0.3153 | 0.2255 | 0.1932 |
| trend_lowvol | neutral | 0.0813 | 0.2584 | 0.3146 | 0.2192 | 0.1918 |
| rsi_vol_combo | neutral | 0.0652 | 0.2388 | 0.2729 | 0.2630 | 0.1723 |
| fund_pe | neutral | 0.0519 | 0.2369 | 0.2191 | 0.1754 | 0.1288 |
| low_downside | neutral | 0.0394 | 0.2260 | 0.1742 | 0.1357 | 0.0989 |
| turnover_stability | bull | 0.1096 | 0.1842 | 0.5949 | 0.3939 | 0.4146 |
| volatility | bull | 0.1227 | 0.2306 | 0.5321 | 0.3939 | 0.3708 |
| fund_pb | bull | 0.1080 | 0.2146 | 0.5033 | 0.4015 | 0.3527 |
| mom_x_lowvol_20_20 | bull | 0.0844 | 0.2358 | 0.3580 | 0.3106 | 0.2346 |
| fund_profit_growth | bull | 0.0692 | 0.1927 | 0.3594 | 0.2955 | 0.2328 |
| momentum_reversal | bull | 0.0822 | 0.2452 | 0.3351 | 0.2500 | 0.2095 |
| trend_lowvol | bull | 0.0755 | 0.2587 | 0.2920 | 0.2803 | 0.1869 |
| fund_score | bull | 0.0672 | 0.2427 | 0.2767 | 0.2576 | 0.1740 |
| low_downside | bull | 0.0611 | 0.2245 | 0.2720 | 0.1591 | 0.1576 |
| rsi_vol_combo | bull | 0.0432 | 0.2461 | 0.1754 | 0.1629 | 0.1020 |
| fund_revenue_growth | bull | 0.0364 | 0.2254 | 0.1617 | 0.2121 | 0.0980 |
| fund_pb | bear | 0.1289 | 0.2402 | 0.5366 | 0.3973 | 0.3749 |
| momentum_reversal | bear | 0.0917 | 0.2237 | 0.4101 | 0.3425 | 0.2753 |
| rsi_vol_combo | bear | 0.0795 | 0.1956 | 0.4065 | 0.2603 | 0.2562 |
| wash_sale_score | bear | 0.0809 | 0.1929 | 0.4192 | 0.1333 | 0.2376 |
| fund_profit_growth | bear | 0.0754 | 0.2687 | 0.2804 | 0.2329 | 0.1728 |
| fund_score | bear | 0.0648 | 0.2728 | 0.2377 | 0.2329 | 0.1465 |

### AI智能体

- **Neutral**: ['momentum_reversal', 'mom_x_lowvol_20_20', 'trend_lowvol'] (单因子IC=0.1019, 组合IC=0.1155)
  - weights: [0.3597, 0.3238, 0.3165]
- **Bull**: ['fund_pb', 'volatility', 'trend_lowvol'] (单因子IC=0.1038, 组合IC=0.1316)
  - bull_weights: [0.4096, 0.3077, 0.2827]
- **Bear**: ['trend_lowvol', 'fund_profit_growth'] (单因子IC=0.0829, 组合IC=0.1145)
  - bear_weights: [0.5383, 0.4617]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| momentum_reversal | neutral | 0.1062 | 0.1676 | 0.6338 | 0.4572 | 0.4618 |
| mom_x_lowvol_20_20 | neutral | 0.0989 | 0.1709 | 0.5788 | 0.4363 | 0.4157 |
| trend_lowvol | neutral | 0.1007 | 0.1779 | 0.5659 | 0.4363 | 0.4064 |
| fund_pb | neutral | 0.0796 | 0.1467 | 0.5425 | 0.3904 | 0.3772 |
| rsi_vol_combo | neutral | 0.0812 | 0.1589 | 0.5114 | 0.3779 | 0.3523 |
| volatility | neutral | 0.0792 | 0.1876 | 0.4222 | 0.3424 | 0.2834 |
| fund_pe | neutral | 0.0634 | 0.1599 | 0.3965 | 0.3278 | 0.2632 |
| fund_profit_growth | neutral | 0.0481 | 0.1322 | 0.3636 | 0.2818 | 0.2331 |
| turnover_stability | neutral | 0.0356 | 0.1126 | 0.3163 | 0.2797 | 0.2024 |
| fund_score | neutral | 0.0436 | 0.1639 | 0.2663 | 0.1649 | 0.1551 |
| low_downside | neutral | 0.0395 | 0.1804 | 0.2192 | 0.1942 | 0.1309 |
| fund_roe | neutral | 0.0376 | 0.1691 | 0.2224 | 0.1649 | 0.1295 |
| fund_revenue_growth | neutral | 0.0233 | 0.1247 | 0.1866 | 0.1795 | 0.1100 |
| wash_sale_score | neutral | 0.0128 | 0.1053 | 0.1212 | 0.1362 | 0.0689 |
| fund_pb | bull | 0.1099 | 0.1317 | 0.8349 | 0.5530 | 0.6483 |
| volatility | bull | 0.1041 | 0.1563 | 0.6662 | 0.4621 | 0.4870 |
| trend_lowvol | bull | 0.0974 | 0.1599 | 0.6091 | 0.4697 | 0.4476 |
| fund_pe | bull | 0.0762 | 0.1463 | 0.5211 | 0.4242 | 0.3711 |
| low_downside | bull | 0.0853 | 0.1600 | 0.5335 | 0.3485 | 0.3597 |
| mom_x_lowvol_20_20 | bull | 0.0610 | 0.1516 | 0.4023 | 0.2803 | 0.2575 |
| momentum_reversal | bull | 0.0558 | 0.1574 | 0.3545 | 0.2424 | 0.2202 |
| turnover_stability | bull | 0.0275 | 0.1038 | 0.2651 | 0.1970 | 0.1586 |
| wash_sale_score | bull | 0.0136 | 0.1039 | 0.1306 | 0.1069 | 0.0723 |
| trend_lowvol | bear | 0.0892 | 0.1612 | 0.5532 | 0.4795 | 0.4092 |
| fund_profit_growth | bear | 0.0766 | 0.1464 | 0.5230 | 0.3425 | 0.3510 |
| fund_score | bear | 0.0701 | 0.1717 | 0.4082 | 0.2603 | 0.2572 |
| mom_x_lowvol_20_20 | bear | 0.0751 | 0.2033 | 0.3695 | 0.3425 | 0.2480 |
| momentum_reversal | bear | 0.0625 | 0.1946 | 0.3211 | 0.2329 | 0.1979 |
| turnover_stability | bear | 0.0332 | 0.1141 | 0.2907 | 0.2603 | 0.1832 |
| wash_sale_score | bear | 0.0347 | 0.1266 | 0.2744 | 0.1579 | 0.1589 |

### AI眼镜

- **Neutral**: ['fund_pb', 'momentum_reversal', 'mom_x_lowvol_20_20'] (单因子IC=0.0687, 组合IC=0.0978)
  - weights: [0.3636, 0.3244, 0.312]
- **Bull**: ['fund_pb', 'trend_lowvol', 'volatility'] (单因子IC=0.0836, 组合IC=0.1136)
  - bull_weights: [0.408, 0.3027, 0.2893]
- **Bear**: ['fund_pb', 'fund_pe', 'turnover_stability'] (单因子IC=0.0704, 组合IC=0.1145)
  - bear_weights: [0.4143, 0.3173, 0.2684]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| fund_pb | neutral | 0.0696 | 0.1664 | 0.4185 | 0.3299 | 0.2783 |
| momentum_reversal | neutral | 0.0693 | 0.1809 | 0.3830 | 0.2965 | 0.2483 |
| mom_x_lowvol_20_20 | neutral | 0.0672 | 0.1768 | 0.3799 | 0.2568 | 0.2387 |
| fund_pe | neutral | 0.0561 | 0.1612 | 0.3479 | 0.2150 | 0.2114 |
| rsi_vol_combo | neutral | 0.0557 | 0.1781 | 0.3126 | 0.2526 | 0.1958 |
| trend_lowvol | neutral | 0.0645 | 0.2107 | 0.3060 | 0.2432 | 0.1902 |
| fund_profit_growth | neutral | 0.0474 | 0.1584 | 0.2991 | 0.2651 | 0.1892 |
| volatility | neutral | 0.0517 | 0.1834 | 0.2817 | 0.1962 | 0.1685 |
| low_downside | neutral | 0.0353 | 0.1655 | 0.2133 | 0.1503 | 0.1227 |
| fund_revenue_growth | neutral | 0.0312 | 0.1554 | 0.2004 | 0.1900 | 0.1193 |
| fund_score | neutral | 0.0304 | 0.1833 | 0.1657 | 0.1503 | 0.0953 |
| fund_pb | bull | 0.0928 | 0.1614 | 0.5749 | 0.3864 | 0.3985 |
| trend_lowvol | bull | 0.0819 | 0.1848 | 0.4434 | 0.3333 | 0.2956 |
| volatility | bull | 0.0762 | 0.1797 | 0.4239 | 0.3333 | 0.2826 |
| mom_x_lowvol_20_20 | bull | 0.0518 | 0.1744 | 0.2969 | 0.2727 | 0.1890 |
| low_downside | bull | 0.0452 | 0.1740 | 0.2598 | 0.1591 | 0.1506 |
| fund_pe | bull | 0.0429 | 0.1622 | 0.2645 | 0.1364 | 0.1503 |
| momentum_reversal | bull | 0.0458 | 0.1850 | 0.2474 | 0.1667 | 0.1443 |
| turnover_stability | bull | 0.0216 | 0.1260 | 0.1712 | 0.1288 | 0.0966 |
| fund_gross_margin | bull | 0.0168 | 0.1345 | 0.1252 | 0.1136 | 0.0697 |
| fund_pb | bear | 0.1052 | 0.2186 | 0.4813 | 0.2329 | 0.2967 |
| fund_pe | bear | 0.0673 | 0.1907 | 0.3530 | 0.2877 | 0.2272 |
| turnover_stability | bear | 0.0388 | 0.1298 | 0.2986 | 0.2877 | 0.1922 |

### AI芯片

- **Neutral**: ['rsi_vol_combo', 'volatility', 'fund_pb'] (单因子IC=0.0623, 组合IC=0.0796)
  - weights: [0.3498, 0.3272, 0.323]
- **Bull**: ['low_downside', 'fund_pe'] (单因子IC=0.0622, 组合IC=0.0735)
  - bull_weights: [0.503, 0.497]
- **Bear**: ['momentum_reversal', 'mom_x_lowvol_20_20'] (单因子IC=0.219, 组合IC=0.2253)
  - bear_weights: [0.5435, 0.4565]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| rsi_vol_combo | neutral | 0.0579 | 0.2294 | 0.2524 | 0.1795 | 0.1489 |
| volatility | neutral | 0.0631 | 0.2707 | 0.2332 | 0.1942 | 0.1392 |
| fund_pb | neutral | 0.0658 | 0.2859 | 0.2302 | 0.1942 | 0.1375 |
| momentum_reversal | neutral | 0.0567 | 0.2475 | 0.2292 | 0.1900 | 0.1363 |
| mom_x_lowvol_20_20 | neutral | 0.0523 | 0.2539 | 0.2059 | 0.1670 | 0.1202 |
| fund_score | neutral | 0.0458 | 0.2565 | 0.1787 | 0.1503 | 0.1028 |
| fund_pe | neutral | 0.0445 | 0.2774 | 0.1604 | 0.1211 | 0.0899 |
| trend_lowvol | neutral | 0.0364 | 0.2671 | 0.1364 | 0.1148 | 0.0760 |
| fund_profit_growth | neutral | 0.0317 | 0.2471 | 0.1285 | 0.1347 | 0.0729 |
| low_downside | bull | 0.0566 | 0.1951 | 0.2901 | 0.1894 | 0.1725 |
| fund_pe | bull | 0.0679 | 0.2474 | 0.2744 | 0.2424 | 0.1705 |
| volatility | bull | 0.0568 | 0.2061 | 0.2755 | 0.2273 | 0.1691 |
| fund_pb | bull | 0.0521 | 0.2529 | 0.2060 | 0.1515 | 0.1186 |
| trend_lowvol | bull | 0.0374 | 0.2549 | 0.1466 | 0.1591 | 0.0849 |
| momentum_reversal | bear | 0.2229 | 0.2523 | 0.8835 | 0.5890 | 0.7020 |
| mom_x_lowvol_20_20 | bear | 0.2152 | 0.2750 | 0.7825 | 0.5068 | 0.5895 |
| rsi_vol_combo | bear | 0.1253 | 0.2166 | 0.5787 | 0.3699 | 0.3963 |
| trend_lowvol | bear | 0.1476 | 0.2535 | 0.5823 | 0.2603 | 0.3669 |

### AI语料

- **Neutral**: ['volatility', 'fund_pb', 'trend_lowvol'] (单因子IC=0.1073, 组合IC=0.1452)
  - weights: [0.3923, 0.3438, 0.264]
- **Bull**: ['trend_lowvol', 'volatility', 'fund_pb'] (单因子IC=0.1285, 组合IC=0.156)
  - bull_weights: [0.3605, 0.3581, 0.2814]
- **Bear**: ['trend_lowvol'] (单因子IC=0.1514, 组合IC=0.1514)
  - bear_weights: [1.0]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| volatility | neutral | 0.1294 | 0.2826 | 0.4578 | 0.3716 | 0.3139 |
| fund_pb | neutral | 0.0995 | 0.2424 | 0.4105 | 0.3403 | 0.2751 |
| trend_lowvol | neutral | 0.0931 | 0.2670 | 0.3486 | 0.2119 | 0.2112 |
| low_downside | neutral | 0.0879 | 0.2691 | 0.3268 | 0.2714 | 0.2077 |
| momentum_reversal | neutral | 0.0838 | 0.2604 | 0.3219 | 0.2255 | 0.1973 |
| mom_x_lowvol_20_20 | neutral | 0.0746 | 0.2684 | 0.2779 | 0.1795 | 0.1639 |
| rsi_vol_combo | neutral | 0.0671 | 0.2512 | 0.2670 | 0.1420 | 0.1525 |
| fund_pe | neutral | 0.0567 | 0.2512 | 0.2256 | 0.1503 | 0.1297 |
| fund_profit_growth | neutral | 0.0299 | 0.2076 | 0.1439 | 0.1347 | 0.0816 |
| trend_lowvol | bull | 0.1351 | 0.2480 | 0.5448 | 0.4394 | 0.3921 |
| volatility | bull | 0.1461 | 0.2629 | 0.5559 | 0.4015 | 0.3895 |
| fund_pb | bull | 0.1043 | 0.2356 | 0.4428 | 0.3826 | 0.3061 |
| mom_x_lowvol_20_20 | bull | 0.1102 | 0.2368 | 0.4655 | 0.3106 | 0.3050 |
| low_downside | bull | 0.0951 | 0.2310 | 0.4117 | 0.3371 | 0.2753 |
| momentum_reversal | bull | 0.0978 | 0.2316 | 0.4221 | 0.2955 | 0.2734 |
| fund_profit_growth | bull | 0.0563 | 0.1859 | 0.3027 | 0.2689 | 0.1921 |
| turnover_stability | bull | 0.0591 | 0.2061 | 0.2865 | 0.1780 | 0.1687 |
| fund_score | bull | 0.0421 | 0.1813 | 0.2321 | 0.2424 | 0.1442 |
| rsi_vol_combo | bull | 0.0535 | 0.2456 | 0.2179 | 0.1212 | 0.1221 |
| fund_pe | bull | 0.0379 | 0.2353 | 0.1611 | 0.1136 | 0.0897 |
| fund_revenue_growth | bull | 0.0267 | 0.1877 | 0.1421 | 0.1288 | 0.0802 |
| trend_lowvol | bear | 0.1514 | 0.2967 | 0.5102 | 0.4247 | 0.3634 |
| momentum_reversal | bear | 0.1033 | 0.2836 | 0.3642 | 0.3973 | 0.2545 |
| fund_gross_margin | bear | 0.0914 | 0.2447 | 0.3736 | 0.3425 | 0.2508 |
| mom_x_lowvol_20_20 | bear | 0.1086 | 0.3135 | 0.3463 | 0.4247 | 0.2467 |
| fund_score | bear | 0.0786 | 0.2272 | 0.3461 | 0.2329 | 0.2133 |
| turnover_stability | bear | 0.0543 | 0.1922 | 0.2823 | 0.3151 | 0.1856 |
| bb_width_20 | bear | 0.0784 | 0.2636 | 0.2975 | 0.1781 | 0.1752 |
| fund_profit_growth | bear | 0.0547 | 0.2797 | 0.1956 | 0.2329 | 0.1205 |

### BC电池

- **Neutral**: ['fund_pb', 'momentum_reversal', 'trend_lowvol'] (单因子IC=0.0676, 组合IC=0.0876)
  - weights: [0.3551, 0.3357, 0.3092]
- **Bull**: ['fund_score', 'fund_gross_margin'] (单因子IC=0.0793, 组合IC=0.085)
  - bull_weights: [0.5762, 0.4238]
- **Bear**: ['top_fractal_volume'] (单因子IC=0.1315, 组合IC=0.1315)
  - bear_weights: [1.0]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| fund_pb | neutral | 0.0681 | 0.2858 | 0.2383 | 0.1743 | 0.1399 |
| momentum_reversal | neutral | 0.0666 | 0.2981 | 0.2234 | 0.1837 | 0.1322 |
| trend_lowvol | neutral | 0.0680 | 0.3190 | 0.2132 | 0.1430 | 0.1218 |
| mom_x_lowvol_20_20 | neutral | 0.0585 | 0.2923 | 0.2002 | 0.1691 | 0.1170 |
| fund_gross_margin | neutral | 0.0338 | 0.2956 | 0.1143 | 0.1106 | 0.0635 |
| wash_sale_score | neutral | 0.0230 | 0.2452 | 0.0937 | 0.1079 | 0.0519 |
| fund_score | bull | 0.0821 | 0.2813 | 0.2917 | 0.1553 | 0.1685 |
| fund_gross_margin | bull | 0.0766 | 0.3420 | 0.2241 | 0.1061 | 0.1239 |
| fund_revenue_growth | bull | 0.0519 | 0.2386 | 0.2177 | 0.1326 | 0.1233 |
| fund_profit_growth | bull | 0.0556 | 0.2607 | 0.2133 | 0.1477 | 0.1224 |
| fund_pe | bull | 0.0547 | 0.2912 | 0.1878 | 0.1590 | 0.1088 |
| fund_roe | bull | 0.0570 | 0.2953 | 0.1929 | 0.1136 | 0.1074 |
| wash_sale_score | bull | 0.0319 | 0.2426 | 0.1314 | 0.1202 | 0.0736 |
| bb_width_20 | bull | 0.0305 | 0.2861 | 0.1067 | 0.1250 | 0.0600 |
| top_fractal_volume | bear | 0.1315 | 0.2486 | 0.5291 | 0.2800 | 0.3386 |
| wash_sale_score | bear | 0.0934 | 0.2218 | 0.4209 | 0.3824 | 0.2909 |
| fund_profit_growth | bear | 0.1086 | 0.2719 | 0.3994 | 0.3973 | 0.2790 |
| vol_opening_confirm | bear | 0.1170 | 0.2965 | 0.3944 | 0.2414 | 0.2448 |
| momentum_reversal | bear | 0.1117 | 0.2881 | 0.3878 | 0.2603 | 0.2444 |
| vol_opening_strength | bear | 0.1160 | 0.2978 | 0.3895 | 0.2414 | 0.2417 |
| trend_lowvol | bear | 0.1188 | 0.3228 | 0.3681 | 0.2877 | 0.2370 |
| mom_x_lowvol_20_20 | bear | 0.1011 | 0.2823 | 0.3581 | 0.2877 | 0.2306 |
| rsi_vol_combo | bear | 0.0837 | 0.2998 | 0.2793 | 0.2055 | 0.1683 |
| bb_width_20 | bear | 0.0664 | 0.2536 | 0.2618 | 0.2603 | 0.1650 |
| fund_score | bear | 0.0566 | 0.2734 | 0.2069 | 0.2055 | 0.1247 |

### C2M概念

- **Neutral**: ['fund_profit_growth', 'turnover_stability', 'fund_pb'] (单因子IC=0.0657, 组合IC=0.0995)
  - weights: [0.3682, 0.3345, 0.2973]
- **Bull**: ['low_downside', 'volatility', 'fund_pb'] (单因子IC=0.1234, 组合IC=0.153)
  - bull_weights: [0.4617, 0.328, 0.2103]
- **Bear**: ['fund_revenue_growth'] (单因子IC=0.0582, 组合IC=0.0582)
  - bear_weights: [1.0]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| fund_profit_growth | neutral | 0.0648 | 0.1985 | 0.3265 | 0.2923 | 0.2110 |
| turnover_stability | neutral | 0.0528 | 0.1720 | 0.3071 | 0.2484 | 0.1917 |
| fund_pb | neutral | 0.0793 | 0.2698 | 0.2941 | 0.1587 | 0.1704 |
| volatility | neutral | 0.0651 | 0.2519 | 0.2585 | 0.2046 | 0.1557 |
| mom_x_lowvol_20_20 | neutral | 0.0557 | 0.2268 | 0.2456 | 0.1451 | 0.1406 |
| low_downside | neutral | 0.0471 | 0.2329 | 0.2024 | 0.1900 | 0.1204 |
| fund_pe | neutral | 0.0535 | 0.2927 | 0.1828 | 0.1952 | 0.1093 |
| trend_lowvol | neutral | 0.0506 | 0.2597 | 0.1949 | 0.1169 | 0.1088 |
| momentum_reversal | neutral | 0.0405 | 0.2244 | 0.1805 | 0.1169 | 0.1008 |
| rsi_vol_combo | neutral | 0.0336 | 0.2131 | 0.1577 | 0.1002 | 0.0868 |
| low_downside | bull | 0.1371 | 0.1916 | 0.7154 | 0.5303 | 0.5474 |
| volatility | bull | 0.1235 | 0.2262 | 0.5460 | 0.4242 | 0.3888 |
| fund_pb | bull | 0.1097 | 0.2734 | 0.4013 | 0.2424 | 0.2493 |
| trend_lowvol | bull | 0.0713 | 0.2251 | 0.3168 | 0.2879 | 0.2040 |
| turnover_stability | bull | 0.0517 | 0.1889 | 0.2738 | 0.2727 | 0.1742 |
| momentum_reversal | bull | 0.0562 | 0.2179 | 0.2578 | 0.1742 | 0.1514 |
| mom_x_lowvol_20_20 | bull | 0.0550 | 0.2345 | 0.2347 | 0.1667 | 0.1369 |
| fund_gross_margin | bull | 0.0366 | 0.1751 | 0.2093 | 0.1212 | 0.1173 |
| rsi_vol_combo | bull | 0.0418 | 0.2144 | 0.1952 | 0.1136 | 0.1087 |
| fund_revenue_growth | bear | 0.0582 | 0.1695 | 0.3434 | 0.3151 | 0.2258 |
| bb_width_20 | bear | 0.0662 | 0.2249 | 0.2946 | 0.2877 | 0.1897 |
| momentum_reversal | bear | 0.0611 | 0.2342 | 0.2608 | 0.2329 | 0.1608 |
| wash_sale_score | bear | 0.0568 | 0.2408 | 0.2360 | 0.1556 | 0.1364 |
| trend_lowvol | bear | 0.0585 | 0.2499 | 0.2341 | 0.1507 | 0.1347 |
| mom_x_lowvol_20_20 | bear | 0.0529 | 0.2511 | 0.2106 | 0.1781 | 0.1240 |

### CAR-T细胞疗法

- **Neutral**: ['fund_pb', 'mom_x_lowvol_20_20', 'momentum_reversal'] (单因子IC=0.0697, 组合IC=0.0921)
  - weights: [0.3532, 0.3315, 0.3153]
- **Bull**: ['turnover_stability', 'fund_revenue_growth', 'low_downside'] (单因子IC=0.0686, 组合IC=0.1021)
  - bull_weights: [0.4224, 0.3429, 0.2348]
- **Bear**: ['trend_lowvol', 'momentum_reversal'] (单因子IC=0.1713, 组合IC=0.1862)
  - bear_weights: [0.518, 0.482]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| fund_pb | neutral | 0.0720 | 0.2799 | 0.2573 | 0.1472 | 0.1476 |
| mom_x_lowvol_20_20 | neutral | 0.0689 | 0.2915 | 0.2365 | 0.1712 | 0.1385 |
| momentum_reversal | neutral | 0.0680 | 0.3030 | 0.2246 | 0.1733 | 0.1317 |
| fund_gross_margin | neutral | 0.0505 | 0.2303 | 0.2191 | 0.1670 | 0.1278 |
| volatility | neutral | 0.0553 | 0.2602 | 0.2127 | 0.1430 | 0.1216 |
| fund_profit_growth | neutral | 0.0539 | 0.2807 | 0.1922 | 0.1534 | 0.1108 |
| turnover_stability | bull | 0.0840 | 0.2398 | 0.3504 | 0.2197 | 0.2137 |
| fund_revenue_growth | bull | 0.0700 | 0.2553 | 0.2743 | 0.2652 | 0.1735 |
| low_downside | bull | 0.0518 | 0.2496 | 0.2077 | 0.1439 | 0.1188 |
| rsi_vol_combo | bull | 0.0569 | 0.2729 | 0.2086 | 0.1364 | 0.1185 |
| fund_pb | bull | 0.0575 | 0.2995 | 0.1919 | 0.2008 | 0.1152 |
| fund_pe | bull | 0.0588 | 0.2843 | 0.2068 | 0.1136 | 0.1152 |
| momentum_reversal | bull | 0.0517 | 0.2626 | 0.1968 | 0.1591 | 0.1141 |
| volatility | bull | 0.0466 | 0.2522 | 0.1848 | 0.1553 | 0.1068 |
| fund_score | bull | 0.0457 | 0.2871 | 0.1591 | 0.2841 | 0.1022 |
| mom_x_lowvol_20_20 | bull | 0.0433 | 0.2574 | 0.1682 | 0.1061 | 0.0930 |
| trend_lowvol | bull | 0.0423 | 0.2745 | 0.1542 | 0.1136 | 0.0858 |
| stroke_phase | bull | 0.0360 | 0.2511 | 0.1432 | 0.1439 | 0.0819 |
| fund_gross_margin | bull | 0.0296 | 0.2310 | 0.1284 | 0.1364 | 0.0729 |
| trend_lowvol | bear | 0.1819 | 0.2940 | 0.6190 | 0.4795 | 0.4579 |
| momentum_reversal | bear | 0.1607 | 0.2815 | 0.5706 | 0.4932 | 0.4260 |
| mom_x_lowvol_20_20 | bear | 0.1498 | 0.2807 | 0.5335 | 0.4795 | 0.3946 |
| fund_pb | bear | 0.0885 | 0.2747 | 0.3223 | 0.5205 | 0.2450 |
| wash_sale_score | bear | 0.0889 | 0.2737 | 0.3246 | 0.2941 | 0.2101 |
| bb_width_20 | bear | 0.0739 | 0.2544 | 0.2907 | 0.2603 | 0.1832 |
| rsi_vol_combo | bear | 0.0840 | 0.2711 | 0.3099 | 0.1507 | 0.1783 |

### CPO概念

- **Neutral**: ['fund_pe', 'trend_lowvol', 'momentum_reversal'] (单因子IC=0.0759, 组合IC=0.0878)
  - weights: [0.3548, 0.3526, 0.2925]
- **Bull**: ['fund_pe'] (单因子IC=0.0708, 组合IC=0.0708)
  - bull_weights: [1.0]
- **Bear**: ['trend_lowvol', 'mom_x_lowvol_20_20'] (单因子IC=0.1342, 组合IC=0.1547)
  - bear_weights: [0.5006, 0.4994]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| fund_pe | neutral | 0.0672 | 0.1894 | 0.3546 | 0.2714 | 0.2254 |
| trend_lowvol | neutral | 0.0868 | 0.2512 | 0.3456 | 0.2965 | 0.2240 |
| momentum_reversal | neutral | 0.0739 | 0.2444 | 0.3023 | 0.2296 | 0.1858 |
| mom_x_lowvol_20_20 | neutral | 0.0689 | 0.2309 | 0.2985 | 0.2088 | 0.1804 |
| rsi_vol_combo | neutral | 0.0640 | 0.2347 | 0.2727 | 0.1983 | 0.1634 |
| fund_pb | neutral | 0.0585 | 0.2333 | 0.2508 | 0.1712 | 0.1468 |
| fund_profit_growth | neutral | 0.0358 | 0.2126 | 0.1682 | 0.1169 | 0.0939 |
| fund_roe | neutral | 0.0372 | 0.2310 | 0.1609 | 0.1482 | 0.0924 |
| volatility | neutral | 0.0413 | 0.2630 | 0.1570 | 0.1503 | 0.0903 |
| fund_score | neutral | 0.0331 | 0.2379 | 0.1392 | 0.1190 | 0.0779 |
| fund_pe | bull | 0.0708 | 0.2044 | 0.3464 | 0.2879 | 0.2231 |
| fund_pb | bull | 0.0629 | 0.2314 | 0.2720 | 0.2197 | 0.1659 |
| low_downside | bull | 0.0499 | 0.2099 | 0.2375 | 0.1818 | 0.1403 |
| fund_gross_margin | bull | 0.0406 | 0.1928 | 0.2107 | 0.2121 | 0.1277 |
| fund_profit_growth | bull | 0.0345 | 0.2043 | 0.1688 | 0.1212 | 0.0946 |
| exhaustion_risk | bull | 0.0247 | 0.1700 | 0.1451 | 0.1074 | 0.0803 |
| trend_lowvol | bear | 0.1463 | 0.2504 | 0.5842 | 0.3973 | 0.4082 |
| mom_x_lowvol_20_20 | bear | 0.1221 | 0.2259 | 0.5404 | 0.5068 | 0.4071 |
| rsi_vol_combo | bear | 0.1143 | 0.2186 | 0.5229 | 0.3973 | 0.3653 |
| momentum_reversal | bear | 0.1151 | 0.2212 | 0.5206 | 0.3973 | 0.3637 |
| fund_pb | bear | 0.0626 | 0.2308 | 0.2711 | 0.1781 | 0.1597 |

### CRO

- **Neutral**: ['volatility', 'momentum_reversal', 'fund_pb'] (单因子IC=0.0658, 组合IC=0.0914)
  - weights: [0.3483, 0.3266, 0.3251]
- **Bull**: ['volatility'] (单因子IC=0.1029, 组合IC=0.1029)
  - bull_weights: [1.0]
- **Bear**: ['trend_lowvol', 'momentum_reversal'] (单因子IC=0.2213, 组合IC=0.2484)
  - bear_weights: [0.5711, 0.4289]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| volatility | neutral | 0.0706 | 0.2980 | 0.2368 | 0.1712 | 0.1387 |
| momentum_reversal | neutral | 0.0638 | 0.2771 | 0.2301 | 0.1305 | 0.1300 |
| fund_pb | neutral | 0.0631 | 0.2811 | 0.2247 | 0.1524 | 0.1295 |
| trend_lowvol | neutral | 0.0569 | 0.2786 | 0.2044 | 0.1441 | 0.1169 |
| mom_x_lowvol_20_20 | neutral | 0.0560 | 0.2810 | 0.1993 | 0.1033 | 0.1100 |
| low_downside | neutral | 0.0430 | 0.2749 | 0.1564 | 0.1461 | 0.0896 |
| rsi_vol_combo | neutral | 0.0429 | 0.2690 | 0.1594 | 0.1086 | 0.0884 |
| fund_pe | neutral | 0.0340 | 0.2753 | 0.1235 | 0.1294 | 0.0697 |
| volatility | bull | 0.1029 | 0.2946 | 0.3494 | 0.2121 | 0.2118 |
| low_downside | bull | 0.0705 | 0.2494 | 0.2827 | 0.1894 | 0.1681 |
| mom_x_lowvol_20_20 | bull | 0.0840 | 0.3178 | 0.2644 | 0.2121 | 0.1602 |
| momentum_reversal | bull | 0.0705 | 0.2971 | 0.2374 | 0.2008 | 0.1425 |
| turnover_stability | bull | 0.0506 | 0.2371 | 0.2133 | 0.2273 | 0.1309 |
| rsi_vol_combo | bull | 0.0547 | 0.2649 | 0.2066 | 0.1894 | 0.1229 |
| fund_gross_margin | bull | 0.0361 | 0.1804 | 0.2000 | 0.1439 | 0.1144 |
| fund_pb | bull | 0.0401 | 0.3224 | 0.1245 | 0.1061 | 0.0689 |
| trend_lowvol | bear | 0.2376 | 0.2709 | 0.8770 | 0.7534 | 0.7689 |
| momentum_reversal | bear | 0.2050 | 0.2626 | 0.7807 | 0.4795 | 0.5775 |
| mom_x_lowvol_20_20 | bear | 0.1649 | 0.2401 | 0.6868 | 0.4795 | 0.5080 |
| rsi_vol_combo | bear | 0.1473 | 0.2419 | 0.6090 | 0.3973 | 0.4255 |
| fund_pe | bear | 0.1268 | 0.2417 | 0.5244 | 0.5068 | 0.3951 |
| fund_revenue_growth | bear | 0.1206 | 0.2353 | 0.5126 | 0.3973 | 0.3581 |
| fund_pb | bear | 0.0975 | 0.2059 | 0.4736 | 0.2055 | 0.2855 |
| bb_width_20 | bear | 0.0771 | 0.2890 | 0.2666 | 0.1644 | 0.1552 |

### ChatGPT概念

- **Neutral**: ['trend_lowvol', 'fund_pb', 'momentum_reversal'] (单因子IC=0.1085, 组合IC=0.1488)
  - weights: [0.3481, 0.3382, 0.3137]
- **Bull**: ['fund_pb', 'volatility', 'trend_lowvol'] (单因子IC=0.1066, 组合IC=0.1287)
  - bull_weights: [0.4097, 0.2994, 0.291]
- **Bear**: ['trend_lowvol', 'mom_x_lowvol_20_20', 'fund_profit_growth'] (单因子IC=0.1055, 组合IC=0.1636)
  - bear_weights: [0.5894, 0.2153, 0.1953]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| trend_lowvol | neutral | 0.1240 | 0.2080 | 0.5963 | 0.4739 | 0.4394 |
| fund_pb | neutral | 0.0947 | 0.1600 | 0.5918 | 0.4426 | 0.4269 |
| momentum_reversal | neutral | 0.1068 | 0.1915 | 0.5580 | 0.4196 | 0.3961 |
| volatility | neutral | 0.1074 | 0.2026 | 0.5300 | 0.4092 | 0.3735 |
| mom_x_lowvol_20_20 | neutral | 0.1002 | 0.1945 | 0.5149 | 0.3779 | 0.3547 |
| rsi_vol_combo | neutral | 0.0833 | 0.1862 | 0.4474 | 0.3591 | 0.3040 |
| low_downside | neutral | 0.0571 | 0.1811 | 0.3151 | 0.2672 | 0.1996 |
| fund_profit_growth | neutral | 0.0484 | 0.1583 | 0.3060 | 0.1921 | 0.1824 |
| fund_score | neutral | 0.0510 | 0.1741 | 0.2933 | 0.2338 | 0.1809 |
| fund_pe | neutral | 0.0549 | 0.1933 | 0.2840 | 0.1879 | 0.1687 |
| fund_roe | neutral | 0.0385 | 0.1848 | 0.2086 | 0.1962 | 0.1248 |
| turnover_stability | neutral | 0.0304 | 0.1566 | 0.1943 | 0.1190 | 0.1087 |
| fund_pb | bull | 0.1019 | 0.1380 | 0.7384 | 0.5606 | 0.5762 |
| volatility | bull | 0.1072 | 0.1803 | 0.5944 | 0.4167 | 0.4211 |
| trend_lowvol | bull | 0.1107 | 0.1906 | 0.5808 | 0.4091 | 0.4092 |
| low_downside | bull | 0.0889 | 0.1757 | 0.5058 | 0.3939 | 0.3525 |
| fund_pe | bull | 0.0670 | 0.1647 | 0.4068 | 0.2955 | 0.2635 |
| fund_revenue_growth | bull | 0.0459 | 0.1527 | 0.3006 | 0.2424 | 0.1867 |
| mom_x_lowvol_20_20 | bull | 0.0558 | 0.1996 | 0.2797 | 0.1667 | 0.1631 |
| fund_score | bull | 0.0417 | 0.1610 | 0.2589 | 0.2348 | 0.1598 |
| momentum_reversal | bull | 0.0519 | 0.2009 | 0.2581 | 0.1818 | 0.1525 |
| fund_roe | bull | 0.0369 | 0.1798 | 0.2052 | 0.2424 | 0.1275 |
| rsi_vol_combo | bull | 0.0387 | 0.2045 | 0.1891 | 0.1061 | 0.1046 |
| fund_profit_growth | bull | 0.0277 | 0.1513 | 0.1832 | 0.1212 | 0.1027 |
| stroke_phase | bull | 0.0253 | 0.1565 | 0.1614 | 0.1364 | 0.0917 |
| trend_lowvol | bear | 0.1474 | 0.1678 | 0.8786 | 0.5890 | 0.6980 |
| mom_x_lowvol_20_20 | bear | 0.0926 | 0.2387 | 0.3878 | 0.3151 | 0.2550 |
| fund_profit_growth | bear | 0.0766 | 0.1952 | 0.3926 | 0.1781 | 0.2313 |
| fund_score | bear | 0.0720 | 0.2197 | 0.3275 | 0.3699 | 0.2243 |
| momentum_reversal | bear | 0.0750 | 0.2458 | 0.3050 | 0.2055 | 0.1838 |

### DeepSeek概念

- **Neutral**: ['momentum_reversal', 'trend_lowvol'] (单因子IC=0.0935, 组合IC=0.1043)
  - weights: [0.5133, 0.4867]
- **Bull**: ['volatility', 'low_downside', 'fund_pb'] (单因子IC=0.1083, 组合IC=0.1321)
  - bull_weights: [0.3683, 0.3385, 0.2932]
- **Bear**: ['trend_lowvol', 'fund_profit_growth', 'mom_x_lowvol_20_20'] (单因子IC=0.0658, 组合IC=0.0909)
  - bear_weights: [0.3973, 0.3875, 0.2152]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| momentum_reversal | neutral | 0.0926 | 0.1573 | 0.5882 | 0.4280 | 0.4200 |
| trend_lowvol | neutral | 0.0945 | 0.1689 | 0.5594 | 0.4238 | 0.3982 |
| mom_x_lowvol_20_20 | neutral | 0.0865 | 0.1568 | 0.5516 | 0.3883 | 0.3829 |
| rsi_vol_combo | neutral | 0.0725 | 0.1485 | 0.4882 | 0.3319 | 0.3251 |
| fund_pb | neutral | 0.0751 | 0.1652 | 0.4545 | 0.3048 | 0.2965 |
| fund_profit_growth | neutral | 0.0460 | 0.1113 | 0.4132 | 0.3424 | 0.2774 |
| volatility | neutral | 0.0775 | 0.1898 | 0.4086 | 0.3069 | 0.2670 |
| turnover_stability | neutral | 0.0365 | 0.0926 | 0.3941 | 0.3090 | 0.2579 |
| fund_score | neutral | 0.0450 | 0.1532 | 0.2939 | 0.1795 | 0.1733 |
| low_downside | neutral | 0.0463 | 0.1791 | 0.2583 | 0.2505 | 0.1615 |
| fund_pe | neutral | 0.0468 | 0.1952 | 0.2399 | 0.1837 | 0.1420 |
| fund_revenue_growth | neutral | 0.0151 | 0.1001 | 0.1512 | 0.1232 | 0.0849 |
| volatility | bull | 0.1166 | 0.1534 | 0.7599 | 0.5758 | 0.5987 |
| low_downside | bull | 0.0988 | 0.1401 | 0.7053 | 0.5606 | 0.5503 |
| fund_pb | bull | 0.1096 | 0.1681 | 0.6519 | 0.4621 | 0.4765 |
| trend_lowvol | bull | 0.1050 | 0.1695 | 0.6193 | 0.4773 | 0.4575 |
| fund_pe | bull | 0.0885 | 0.1811 | 0.4887 | 0.4015 | 0.3425 |
| mom_x_lowvol_20_20 | bull | 0.0707 | 0.1477 | 0.4786 | 0.3333 | 0.3191 |
| momentum_reversal | bull | 0.0682 | 0.1543 | 0.4421 | 0.3106 | 0.2897 |
| fund_profit_growth | bull | 0.0448 | 0.1083 | 0.4138 | 0.3485 | 0.2790 |
| fund_score | bull | 0.0598 | 0.1511 | 0.3955 | 0.2500 | 0.2472 |
| fund_roe | bull | 0.0574 | 0.1707 | 0.3361 | 0.1591 | 0.1948 |
| fund_revenue_growth | bull | 0.0273 | 0.1033 | 0.2639 | 0.1515 | 0.1520 |
| rsi_vol_combo | bull | 0.0360 | 0.1484 | 0.2428 | 0.1364 | 0.1380 |
| turnover_stability | bull | 0.0214 | 0.1047 | 0.2047 | 0.1667 | 0.1194 |
| trend_lowvol | bear | 0.0790 | 0.1584 | 0.4991 | 0.3425 | 0.3350 |
| fund_profit_growth | bear | 0.0550 | 0.1152 | 0.4771 | 0.3699 | 0.3268 |
| mom_x_lowvol_20_20 | bear | 0.0634 | 0.2104 | 0.3011 | 0.2055 | 0.1815 |
| momentum_reversal | bear | 0.0586 | 0.2041 | 0.2870 | 0.1507 | 0.1651 |

### ERP概念

- **Neutral**: ['fund_score', 'mom_x_lowvol_20_20'] (单因子IC=0.0826, 组合IC=0.104)
  - weights: [0.574, 0.426]
- **Bull**: ['volatility', 'trend_lowvol', 'fund_pb'] (单因子IC=0.1057, 组合IC=0.1336)
  - bull_weights: [0.3791, 0.313, 0.3079]
- **Bear**: ['fund_revenue_growth', 'wash_sale_score'] (单因子IC=0.1538, 组合IC=0.1771)
  - bear_weights: [0.5467, 0.4533]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| fund_score | neutral | 0.0888 | 0.2545 | 0.3488 | 0.3006 | 0.2268 |
| mom_x_lowvol_20_20 | neutral | 0.0764 | 0.2758 | 0.2771 | 0.2150 | 0.1683 |
| fund_profit_growth | neutral | 0.0652 | 0.2398 | 0.2717 | 0.2150 | 0.1651 |
| fund_roe | neutral | 0.0676 | 0.2538 | 0.2664 | 0.2150 | 0.1618 |
| momentum_reversal | neutral | 0.0731 | 0.2845 | 0.2570 | 0.2109 | 0.1556 |
| rsi_vol_combo | neutral | 0.0682 | 0.2715 | 0.2514 | 0.1942 | 0.1501 |
| fund_pb | neutral | 0.0743 | 0.2967 | 0.2504 | 0.1754 | 0.1471 |
| trend_lowvol | neutral | 0.0710 | 0.2835 | 0.2506 | 0.1722 | 0.1469 |
| volatility | neutral | 0.0681 | 0.2991 | 0.2278 | 0.2119 | 0.1380 |
| fund_pe | neutral | 0.0510 | 0.2871 | 0.1776 | 0.1806 | 0.1049 |
| fund_revenue_growth | neutral | 0.0484 | 0.2720 | 0.1779 | 0.1263 | 0.1002 |
| wash_sale_score | neutral | 0.0339 | 0.2317 | 0.1461 | 0.1132 | 0.0813 |
| volatility | bull | 0.1060 | 0.2671 | 0.3970 | 0.3712 | 0.2722 |
| trend_lowvol | bull | 0.1021 | 0.2737 | 0.3732 | 0.2045 | 0.2247 |
| fund_pb | bull | 0.1090 | 0.3232 | 0.3374 | 0.3106 | 0.2211 |
| low_downside | bull | 0.0891 | 0.2776 | 0.3210 | 0.1970 | 0.1921 |
| mom_x_lowvol_20_20 | bull | 0.0749 | 0.2669 | 0.2808 | 0.1970 | 0.1680 |
| momentum_reversal | bull | 0.0795 | 0.2767 | 0.2871 | 0.1667 | 0.1675 |
| fund_score | bull | 0.0348 | 0.2047 | 0.1701 | 0.1212 | 0.0953 |
| fund_roe | bull | 0.0306 | 0.2327 | 0.1314 | 0.1174 | 0.0734 |
| stroke_phase | bull | 0.0251 | 0.2491 | 0.1008 | 0.1098 | 0.0559 |
| fund_revenue_growth | bear | 0.1757 | 0.2734 | 0.6427 | 0.4110 | 0.4534 |
| wash_sale_score | bear | 0.1319 | 0.2580 | 0.5112 | 0.4706 | 0.3759 |
| mom_x_lowvol_20_20 | bear | 0.1185 | 0.2802 | 0.4229 | 0.3151 | 0.2781 |
| fund_profit_growth | bear | 0.0934 | 0.2545 | 0.3670 | 0.4247 | 0.2614 |
| fund_score | bear | 0.0930 | 0.2333 | 0.3989 | 0.2055 | 0.2404 |

### ETC

- **Neutral**: ['mom_x_lowvol_20_20', 'momentum_reversal', 'turnover_stability'] (单因子IC=0.0901, 组合IC=0.11)
  - weights: [0.3402, 0.339, 0.3208]
- **Bull**: ['low_downside', 'volatility', 'fund_pb'] (单因子IC=0.1329, 组合IC=0.1529)
  - bull_weights: [0.3913, 0.3323, 0.2764]
- **Bear**: ['fund_revenue_growth', 'fund_score', 'mom_x_lowvol_20_20'] (单因子IC=0.1139, 组合IC=0.1432)
  - bear_weights: [0.5287, 0.2387, 0.2326]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| mom_x_lowvol_20_20 | neutral | 0.0983 | 0.2283 | 0.4304 | 0.3194 | 0.2840 |
| momentum_reversal | neutral | 0.0992 | 0.2315 | 0.4284 | 0.3215 | 0.2830 |
| turnover_stability | neutral | 0.0728 | 0.1798 | 0.4047 | 0.3236 | 0.2678 |
| fund_pb | neutral | 0.0909 | 0.2239 | 0.4060 | 0.2860 | 0.2611 |
| trend_lowvol | neutral | 0.0924 | 0.2373 | 0.3895 | 0.3132 | 0.2557 |
| rsi_vol_combo | neutral | 0.0814 | 0.2229 | 0.3651 | 0.3006 | 0.2374 |
| volatility | neutral | 0.0851 | 0.2545 | 0.3342 | 0.2965 | 0.2166 |
| fund_pe | neutral | 0.0812 | 0.2417 | 0.3362 | 0.2693 | 0.2134 |
| fund_profit_growth | neutral | 0.0512 | 0.2126 | 0.2406 | 0.1503 | 0.1384 |
| fund_score | neutral | 0.0491 | 0.2348 | 0.2091 | 0.1441 | 0.1196 |
| fund_revenue_growth | neutral | 0.0357 | 0.1955 | 0.1824 | 0.1336 | 0.1034 |
| low_downside | neutral | 0.0416 | 0.2363 | 0.1762 | 0.1503 | 0.1014 |
| fund_roe | neutral | 0.0386 | 0.2314 | 0.1667 | 0.1545 | 0.0962 |
| low_downside | bull | 0.1468 | 0.2334 | 0.6290 | 0.4697 | 0.4622 |
| volatility | bull | 0.1419 | 0.2575 | 0.5511 | 0.4242 | 0.3924 |
| fund_pb | bull | 0.1100 | 0.2260 | 0.4869 | 0.3409 | 0.3265 |
| fund_pe | bull | 0.0848 | 0.2418 | 0.3508 | 0.3712 | 0.2405 |
| trend_lowvol | bull | 0.0859 | 0.2497 | 0.3441 | 0.2955 | 0.2229 |
| fund_profit_growth | bull | 0.0708 | 0.1968 | 0.3601 | 0.2348 | 0.2223 |
| turnover_stability | bull | 0.0595 | 0.1899 | 0.3132 | 0.2841 | 0.2011 |
| mom_x_lowvol_20_20 | bull | 0.0674 | 0.2426 | 0.2777 | 0.2576 | 0.1746 |
| fund_score | bull | 0.0546 | 0.2257 | 0.2419 | 0.1894 | 0.1439 |
| fund_gross_margin | bull | 0.0337 | 0.1514 | 0.2229 | 0.2197 | 0.1359 |
| fund_roe | bull | 0.0371 | 0.2108 | 0.1761 | 0.1818 | 0.1040 |
| momentum_reversal | bull | 0.0366 | 0.2263 | 0.1616 | 0.2273 | 0.0992 |
| fund_revenue_growth | bear | 0.1282 | 0.1574 | 0.8140 | 0.6164 | 0.6579 |
| fund_score | bear | 0.0996 | 0.2113 | 0.4712 | 0.2603 | 0.2969 |
| mom_x_lowvol_20_20 | bear | 0.1139 | 0.2479 | 0.4593 | 0.2603 | 0.2894 |
| momentum_reversal | bear | 0.1010 | 0.2461 | 0.4104 | 0.2603 | 0.2586 |
| trend_lowvol | bear | 0.0835 | 0.2193 | 0.3810 | 0.2055 | 0.2296 |
| rsi_vol_combo | bear | 0.0532 | 0.2183 | 0.2437 | 0.2329 | 0.1502 |
| fund_profit_growth | bear | 0.0488 | 0.1911 | 0.2554 | 0.1507 | 0.1469 |
| fund_gross_margin | bear | 0.0411 | 0.1771 | 0.2321 | 0.1507 | 0.1336 |

### F5G概念

- **Neutral**: ['trend_lowvol', 'mom_x_lowvol_20_20'] (单因子IC=0.0793, 组合IC=0.0925)
  - weights: [0.5379, 0.4621]
- **Bull**: ['fund_profit_growth', 'fund_revenue_growth'] (单因子IC=0.0701, 组合IC=0.0838)
  - bull_weights: [0.5082, 0.4918]
- **Bear**: ['trend_lowvol', 'momentum_reversal'] (单因子IC=0.1184, 组合IC=0.1389)
  - bear_weights: [0.591, 0.409]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| trend_lowvol | neutral | 0.0886 | 0.2746 | 0.3227 | 0.2171 | 0.1964 |
| mom_x_lowvol_20_20 | neutral | 0.0700 | 0.2430 | 0.2881 | 0.1712 | 0.1687 |
| volatility | neutral | 0.0594 | 0.2627 | 0.2262 | 0.1858 | 0.1341 |
| fund_pb | neutral | 0.0677 | 0.3000 | 0.2256 | 0.1775 | 0.1328 |
| fund_pe | neutral | 0.0548 | 0.2705 | 0.2025 | 0.1441 | 0.1159 |
| fund_profit_growth | neutral | 0.0458 | 0.2296 | 0.1993 | 0.1211 | 0.1117 |
| momentum_reversal | neutral | 0.0516 | 0.2597 | 0.1986 | 0.1148 | 0.1107 |
| fund_revenue_growth | neutral | 0.0433 | 0.2253 | 0.1922 | 0.1216 | 0.1078 |
| fund_roe | neutral | 0.0388 | 0.2656 | 0.1462 | 0.1033 | 0.0806 |
| fund_profit_growth | bull | 0.0650 | 0.2358 | 0.2759 | 0.2879 | 0.1776 |
| fund_revenue_growth | bull | 0.0751 | 0.2525 | 0.2973 | 0.1566 | 0.1719 |
| fund_score | bull | 0.0786 | 0.2873 | 0.2737 | 0.2045 | 0.1649 |
| volatility | bull | 0.0742 | 0.2627 | 0.2823 | 0.1667 | 0.1647 |
| mom_x_lowvol_20_20 | bull | 0.0443 | 0.2345 | 0.1887 | 0.1174 | 0.1054 |
| relative_strength | bull | 0.0416 | 0.2320 | 0.1795 | 0.1515 | 0.1033 |
| exhaustion_risk | bull | 0.0352 | 0.2183 | 0.1610 | 0.2381 | 0.0997 |
| low_downside | bull | 0.0426 | 0.2778 | 0.1532 | 0.1856 | 0.0908 |
| trend_lowvol | bear | 0.1274 | 0.2224 | 0.5728 | 0.5068 | 0.4316 |
| momentum_reversal | bear | 0.1094 | 0.2410 | 0.4541 | 0.3151 | 0.2986 |
| mom_x_lowvol_20_20 | bear | 0.1179 | 0.2651 | 0.4446 | 0.2877 | 0.2862 |
| bb_width_20 | bear | 0.0990 | 0.2433 | 0.4068 | 0.3425 | 0.2730 |
| rsi_vol_combo | bear | 0.0481 | 0.2100 | 0.2288 | 0.2055 | 0.1379 |

### HJT电池

- **Neutral**: ['trend_lowvol'] (单因子IC=0.0649, 组合IC=0.0649)
  - weights: [1.0]
- **Bull**: ['trend_lowvol', 'volatility', 'fund_gross_margin'] (单因子IC=0.0959, 组合IC=0.1305)
  - bull_weights: [0.3807, 0.3257, 0.2936]
- **Bear**: ['mom_x_lowvol_20_20'] (单因子IC=0.1764, 组合IC=0.1764)
  - bear_weights: [1.0]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| trend_lowvol | neutral | 0.0649 | 0.2663 | 0.2439 | 0.2578 | 0.1534 |
| volatility | neutral | 0.0392 | 0.2446 | 0.1603 | 0.1743 | 0.0941 |
| trend_lowvol | bull | 0.1077 | 0.2039 | 0.5280 | 0.3636 | 0.3600 |
| volatility | bull | 0.0971 | 0.2186 | 0.4442 | 0.3864 | 0.3079 |
| fund_gross_margin | bull | 0.0831 | 0.1972 | 0.4212 | 0.3182 | 0.2776 |
| low_downside | bull | 0.0714 | 0.1786 | 0.3998 | 0.3106 | 0.2620 |
| momentum_reversal | bull | 0.0373 | 0.2015 | 0.1850 | 0.1818 | 0.1093 |
| mom_x_lowvol_20_20 | bull | 0.0260 | 0.2011 | 0.1293 | 0.1364 | 0.0735 |
| rsi_vol_combo | bull | 0.0214 | 0.2060 | 0.1037 | 0.1439 | 0.0593 |
| mom_x_lowvol_20_20 | bear | 0.1764 | 0.3096 | 0.5698 | 0.5342 | 0.4371 |
| momentum_reversal | bear | 0.1402 | 0.2883 | 0.4864 | 0.3973 | 0.3398 |
| rsi_vol_combo | bear | 0.1114 | 0.2662 | 0.4184 | 0.3151 | 0.2751 |
| trend_lowvol | bear | 0.0671 | 0.3275 | 0.2049 | 0.1781 | 0.1207 |
| turnover_stability | bear | 0.0453 | 0.2221 | 0.2040 | 0.1781 | 0.1202 |
| fund_profit_growth | bear | 0.0370 | 0.2008 | 0.1840 | 0.1233 | 0.1034 |

### IGBT概念

- **Neutral**: ['turnover_stability', 'trend_lowvol', 'mom_x_lowvol_20_20'] (单因子IC=0.0672, 组合IC=0.1045)
  - weights: [0.3402, 0.3368, 0.323]
- **Bull**: ['fund_pb', 'fund_pe', 'low_downside'] (单因子IC=0.0728, 组合IC=0.0948)
  - bull_weights: [0.3762, 0.3409, 0.2829]
- **Bear**: ['mom_x_lowvol_20_20', 'momentum_reversal', 'bb_width_20'] (单因子IC=0.11, 组合IC=0.141)
  - bear_weights: [0.4035, 0.3673, 0.2293]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| turnover_stability | neutral | 0.0592 | 0.2066 | 0.2864 | 0.2359 | 0.1770 |
| trend_lowvol | neutral | 0.0697 | 0.2511 | 0.2774 | 0.2630 | 0.1752 |
| mom_x_lowvol_20_20 | neutral | 0.0727 | 0.2546 | 0.2855 | 0.1775 | 0.1681 |
| momentum_reversal | neutral | 0.0730 | 0.2597 | 0.2812 | 0.1639 | 0.1636 |
| volatility | neutral | 0.0696 | 0.2680 | 0.2598 | 0.1649 | 0.1513 |
| fund_pb | neutral | 0.0580 | 0.2464 | 0.2356 | 0.2411 | 0.1462 |
| fund_pe | neutral | 0.0539 | 0.2377 | 0.2267 | 0.1921 | 0.1351 |
| fund_profit_growth | neutral | 0.0463 | 0.2373 | 0.1950 | 0.1597 | 0.1131 |
| fund_score | neutral | 0.0370 | 0.2484 | 0.1491 | 0.1660 | 0.0869 |
| low_downside | neutral | 0.0304 | 0.2544 | 0.1193 | 0.1044 | 0.0659 |
| fund_pb | bull | 0.0898 | 0.3031 | 0.2963 | 0.3106 | 0.1942 |
| fund_pe | bull | 0.0789 | 0.2618 | 0.3016 | 0.1667 | 0.1759 |
| low_downside | bull | 0.0496 | 0.2157 | 0.2301 | 0.2689 | 0.1460 |
| turnover_stability | bull | 0.0397 | 0.1941 | 0.2044 | 0.2348 | 0.1262 |
| volatility | bull | 0.0522 | 0.2648 | 0.1971 | 0.2121 | 0.1194 |
| fund_profit_growth | bull | 0.0450 | 0.2202 | 0.2045 | 0.1288 | 0.1154 |
| trend_lowvol | bull | 0.0307 | 0.2524 | 0.1217 | 0.1061 | 0.0673 |
| mom_x_lowvol_20_20 | bear | 0.1225 | 0.2195 | 0.5582 | 0.4795 | 0.4129 |
| momentum_reversal | bear | 0.1328 | 0.2371 | 0.5600 | 0.3425 | 0.3759 |
| bb_width_20 | bear | 0.0747 | 0.2092 | 0.3569 | 0.3151 | 0.2347 |
| rsi_vol_combo | bear | 0.0737 | 0.2592 | 0.2846 | 0.2055 | 0.1715 |

### IPO受益

- **Neutral**: ['fund_pb', 'trend_lowvol'] (单因子IC=0.0775, 组合IC=0.1044)
  - weights: [0.5696, 0.4304]
- **Bull**: ['low_downside', 'trend_lowvol', 'momentum_reversal'] (单因子IC=0.0885, 组合IC=0.1335)
  - bull_weights: [0.4434, 0.2812, 0.2753]
- **Bear**: ['momentum_reversal', 'fund_pe', 'fund_profit_growth'] (单因子IC=0.0764, 组合IC=0.106)
  - bear_weights: [0.391, 0.3291, 0.2798]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| fund_pb | neutral | 0.0793 | 0.1781 | 0.4452 | 0.3434 | 0.2990 |
| trend_lowvol | neutral | 0.0757 | 0.2151 | 0.3519 | 0.2839 | 0.2259 |
| fund_pe | neutral | 0.0537 | 0.1813 | 0.2965 | 0.2714 | 0.1885 |
| low_downside | neutral | 0.0589 | 0.2048 | 0.2874 | 0.2735 | 0.1830 |
| momentum_reversal | neutral | 0.0550 | 0.2130 | 0.2582 | 0.1962 | 0.1544 |
| turnover_stability | neutral | 0.0396 | 0.1586 | 0.2500 | 0.2088 | 0.1511 |
| volatility | neutral | 0.0417 | 0.1761 | 0.2370 | 0.2276 | 0.1455 |
| mom_x_lowvol_20_20 | neutral | 0.0346 | 0.1804 | 0.1918 | 0.1190 | 0.1073 |
| fund_profit_growth | neutral | 0.0272 | 0.1501 | 0.1813 | 0.1628 | 0.1054 |
| low_downside | bull | 0.1063 | 0.1654 | 0.6428 | 0.4848 | 0.4773 |
| trend_lowvol | bull | 0.0793 | 0.1736 | 0.4566 | 0.3258 | 0.3027 |
| momentum_reversal | bull | 0.0801 | 0.1750 | 0.4575 | 0.2955 | 0.2963 |
| turnover_stability | bull | 0.0594 | 0.1439 | 0.4128 | 0.3030 | 0.2690 |
| fund_pb | bull | 0.0562 | 0.1891 | 0.2971 | 0.2121 | 0.1800 |
| rsi_vol_combo | bull | 0.0503 | 0.1754 | 0.2870 | 0.1667 | 0.1674 |
| fund_pe | bull | 0.0427 | 0.1807 | 0.2362 | 0.2803 | 0.1512 |
| volatility | bull | 0.0287 | 0.1840 | 0.1561 | 0.1288 | 0.0881 |
| momentum_reversal | bear | 0.1032 | 0.2157 | 0.4785 | 0.2877 | 0.3080 |
| fund_pe | bear | 0.0704 | 0.1786 | 0.3943 | 0.3151 | 0.2593 |
| fund_profit_growth | bear | 0.0557 | 0.1558 | 0.3576 | 0.2329 | 0.2204 |
| trend_lowvol | bear | 0.0726 | 0.2251 | 0.3227 | 0.2603 | 0.2033 |
| fund_score | bear | 0.0541 | 0.1810 | 0.2988 | 0.2055 | 0.1801 |
| mom_x_lowvol_20_20 | bear | 0.0638 | 0.2090 | 0.3054 | 0.1233 | 0.1715 |
| fund_roe | bear | 0.0542 | 0.2108 | 0.2573 | 0.1507 | 0.1480 |
| bb_width_20 | bear | 0.0584 | 0.2719 | 0.2147 | 0.1233 | 0.1206 |

### LED概念

- **Neutral**: ['trend_lowvol', 'mom_x_lowvol_20_20'] (单因子IC=0.0761, 组合IC=0.0876)
  - weights: [0.5113, 0.4887]
- **Bull**: ['fund_pb', 'low_downside', 'trend_lowvol'] (单因子IC=0.0758, 组合IC=0.1007)
  - bull_weights: [0.3626, 0.3258, 0.3116]
- **Bear**: ['mom_x_lowvol_20_20'] (单因子IC=0.1176, 组合IC=0.1176)
  - bear_weights: [1.0]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| trend_lowvol | neutral | 0.0812 | 0.2055 | 0.3949 | 0.3288 | 0.2624 |
| mom_x_lowvol_20_20 | neutral | 0.0711 | 0.1799 | 0.3953 | 0.2693 | 0.2509 |
| momentum_reversal | neutral | 0.0710 | 0.1879 | 0.3781 | 0.2881 | 0.2435 |
| fund_pb | neutral | 0.0689 | 0.1853 | 0.3718 | 0.2693 | 0.2360 |
| fund_pe | neutral | 0.0580 | 0.1716 | 0.3380 | 0.2756 | 0.2156 |
| turnover_stability | neutral | 0.0429 | 0.1359 | 0.3153 | 0.2756 | 0.2011 |
| volatility | neutral | 0.0617 | 0.1990 | 0.3099 | 0.2004 | 0.1860 |
| rsi_vol_combo | neutral | 0.0561 | 0.1882 | 0.2980 | 0.2046 | 0.1795 |
| low_downside | neutral | 0.0405 | 0.1846 | 0.2195 | 0.1649 | 0.1279 |
| fund_profit_growth | neutral | 0.0303 | 0.1674 | 0.1808 | 0.1503 | 0.1040 |
| fund_score | neutral | 0.0310 | 0.1878 | 0.1649 | 0.1148 | 0.0919 |
| fund_revenue_growth | neutral | 0.0197 | 0.1468 | 0.1345 | 0.1169 | 0.0751 |
| fund_pb | bull | 0.0896 | 0.1926 | 0.4654 | 0.2727 | 0.2962 |
| low_downside | bull | 0.0677 | 0.1715 | 0.3947 | 0.3485 | 0.2661 |
| trend_lowvol | bull | 0.0700 | 0.1824 | 0.3840 | 0.3258 | 0.2546 |
| turnover_stability | bull | 0.0441 | 0.1150 | 0.3831 | 0.2879 | 0.2467 |
| volatility | bull | 0.0675 | 0.1894 | 0.3566 | 0.2424 | 0.2215 |
| fund_pe | bull | 0.0549 | 0.1805 | 0.3041 | 0.2652 | 0.1923 |
| rsi_vol_combo | bull | 0.0296 | 0.1765 | 0.1676 | 0.1061 | 0.0927 |
| stroke_phase | bull | 0.0199 | 0.1267 | 0.1573 | 0.1515 | 0.0906 |
| mom_x_lowvol_20_20 | bear | 0.1176 | 0.2046 | 0.5744 | 0.3973 | 0.4013 |
| momentum_reversal | bear | 0.0724 | 0.1881 | 0.3851 | 0.2603 | 0.2427 |

### MLCC

- **Neutral**: ['mom_x_lowvol_20_20', 'momentum_reversal', 'trend_lowvol'] (单因子IC=0.0833, 组合IC=0.0909)
  - weights: [0.3506, 0.3282, 0.3212]
- **Bull**: ['trend_lowvol', 'fund_pb', 'low_downside'] (单因子IC=0.0624, 组合IC=0.1076)
  - bull_weights: [0.4071, 0.2997, 0.2932]
- **Bear**: ['top_fractal_volume'] (单因子IC=0.1329, 组合IC=0.1329)
  - bear_weights: [1.0]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| mom_x_lowvol_20_20 | neutral | 0.0872 | 0.2825 | 0.3086 | 0.2589 | 0.1943 |
| momentum_reversal | neutral | 0.0856 | 0.2927 | 0.2926 | 0.2432 | 0.1819 |
| trend_lowvol | neutral | 0.0770 | 0.2748 | 0.2802 | 0.2704 | 0.1780 |
| fund_pb | neutral | 0.0733 | 0.2969 | 0.2469 | 0.2088 | 0.1492 |
| rsi_vol_combo | neutral | 0.0651 | 0.2830 | 0.2299 | 0.1837 | 0.1361 |
| fund_pe | neutral | 0.0475 | 0.2952 | 0.1609 | 0.1399 | 0.0917 |
| volatility | neutral | 0.0407 | 0.2782 | 0.1462 | 0.1117 | 0.0813 |
| trend_lowvol | bull | 0.0741 | 0.2566 | 0.2889 | 0.1326 | 0.1636 |
| fund_pb | bull | 0.0572 | 0.2942 | 0.1945 | 0.2386 | 0.1204 |
| low_downside | bull | 0.0559 | 0.2711 | 0.2060 | 0.1439 | 0.1178 |
| top_fractal_volume | bull | 0.0335 | 0.2278 | 0.1472 | 0.1173 | 0.0822 |
| fund_pe | bull | 0.0353 | 0.2763 | 0.1276 | 0.1591 | 0.0740 |
| vol_confirm | bull | 0.0311 | 0.2900 | 0.1072 | 0.1061 | 0.0593 |
| top_fractal_volume | bear | 0.1329 | 0.2461 | 0.5400 | 0.4286 | 0.3857 |
| rsi_vol_combo | bear | 0.1120 | 0.2492 | 0.4493 | 0.3425 | 0.3016 |
| trend_lowvol | bear | 0.0842 | 0.2468 | 0.3410 | 0.2329 | 0.2102 |
| limit_pullback_score | bear | 0.0640 | 0.2272 | 0.2815 | 0.3333 | 0.1877 |
| fund_profit_growth | bear | 0.0585 | 0.2423 | 0.2415 | 0.1370 | 0.1373 |

### MicroLED

- **Neutral**: ['trend_lowvol', 'momentum_reversal', 'turnover_stability'] (单因子IC=0.0647, 组合IC=0.0877)
  - weights: [0.3438, 0.3282, 0.3279]
- **Bull**: ['fund_pb', 'wash_sale_score', 'exhaustion_risk'] (单因子IC=0.0583, 组合IC=0.089)
  - bull_weights: [0.3833, 0.3794, 0.2373]
- **Bear**: ['mom_x_lowvol_20_20'] (单因子IC=0.1769, 组合IC=0.1769)
  - bear_weights: [1.0]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| trend_lowvol | neutral | 0.0741 | 0.2698 | 0.2746 | 0.2150 | 0.1668 |
| momentum_reversal | neutral | 0.0671 | 0.2502 | 0.2681 | 0.1879 | 0.1592 |
| turnover_stability | neutral | 0.0530 | 0.2066 | 0.2568 | 0.2390 | 0.1591 |
| mom_x_lowvol_20_20 | neutral | 0.0540 | 0.2511 | 0.2150 | 0.1566 | 0.1243 |
| rsi_vol_combo | neutral | 0.0519 | 0.2519 | 0.2060 | 0.1399 | 0.1174 |
| fund_pb | neutral | 0.0521 | 0.2779 | 0.1874 | 0.1357 | 0.1064 |
| volatility | neutral | 0.0525 | 0.3011 | 0.1742 | 0.1253 | 0.0980 |
| fund_pe | neutral | 0.0374 | 0.2782 | 0.1343 | 0.1096 | 0.0745 |
| fund_revenue_growth | neutral | 0.0278 | 0.2125 | 0.1310 | 0.1181 | 0.0732 |
| low_downside | neutral | 0.0233 | 0.2561 | 0.0909 | 0.1127 | 0.0506 |
| fund_pb | bull | 0.0695 | 0.2478 | 0.2804 | 0.2348 | 0.1731 |
| wash_sale_score | bull | 0.0652 | 0.2263 | 0.2881 | 0.1896 | 0.1714 |
| exhaustion_risk | bull | 0.0402 | 0.2131 | 0.1887 | 0.1358 | 0.1072 |
| stroke_phase | bull | 0.0358 | 0.1997 | 0.1790 | 0.1439 | 0.1024 |
| low_downside | bull | 0.0468 | 0.2750 | 0.1703 | 0.1705 | 0.0997 |
| fund_revenue_growth | bull | 0.0338 | 0.2105 | 0.1608 | 0.1288 | 0.0908 |
| volatility | bull | 0.0328 | 0.2635 | 0.1246 | 0.1212 | 0.0698 |
| mom_x_lowvol_20_20 | bear | 0.1769 | 0.2347 | 0.7539 | 0.6438 | 0.6196 |
| momentum_reversal | bear | 0.1527 | 0.2485 | 0.6145 | 0.3699 | 0.4209 |
| rsi_vol_combo | bear | 0.0958 | 0.2057 | 0.4656 | 0.3425 | 0.3125 |
| fund_gross_margin | bear | 0.1084 | 0.2334 | 0.4646 | 0.3425 | 0.3119 |
| turnover_stability | bear | 0.0775 | 0.1849 | 0.4189 | 0.2603 | 0.2640 |
| fund_profit_growth | bear | 0.0947 | 0.2286 | 0.4142 | 0.2603 | 0.2610 |
| wash_sale_score | bear | 0.0720 | 0.2733 | 0.2633 | 0.2222 | 0.1609 |
| trend_lowvol | bear | 0.0660 | 0.2886 | 0.2286 | 0.1507 | 0.1316 |

### MiniLED

- **Neutral**: ['fund_pb', 'momentum_reversal', 'trend_lowvol'] (单因子IC=0.0843, 组合IC=0.1255)
  - weights: [0.344, 0.3305, 0.3255]
- **Bull**: ['volatility'] (单因子IC=0.0878, 组合IC=0.0878)
  - bull_weights: [1.0]
- **Bear**: ['mom_x_lowvol_20_20'] (单因子IC=0.1313, 组合IC=0.1313)
  - bear_weights: [1.0]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| fund_pb | neutral | 0.0784 | 0.1795 | 0.4368 | 0.3382 | 0.2923 |
| momentum_reversal | neutral | 0.0839 | 0.1999 | 0.4197 | 0.3382 | 0.2808 |
| trend_lowvol | neutral | 0.0905 | 0.2218 | 0.4082 | 0.3549 | 0.2765 |
| mom_x_lowvol_20_20 | neutral | 0.0761 | 0.1956 | 0.3891 | 0.3132 | 0.2555 |
| rsi_vol_combo | neutral | 0.0795 | 0.1979 | 0.4016 | 0.2714 | 0.2553 |
| volatility | neutral | 0.0739 | 0.2076 | 0.3560 | 0.3006 | 0.2315 |
| turnover_stability | neutral | 0.0524 | 0.1535 | 0.3413 | 0.3111 | 0.2237 |
| fund_pe | neutral | 0.0558 | 0.1724 | 0.3236 | 0.2610 | 0.2040 |
| low_downside | neutral | 0.0482 | 0.1835 | 0.2627 | 0.2234 | 0.1607 |
| fund_profit_growth | neutral | 0.0399 | 0.1762 | 0.2265 | 0.1837 | 0.1340 |
| fund_score | neutral | 0.0426 | 0.1892 | 0.2251 | 0.1733 | 0.1320 |
| fund_gross_margin | neutral | 0.0266 | 0.1423 | 0.1867 | 0.1576 | 0.1080 |
| fund_revenue_growth | neutral | 0.0311 | 0.1667 | 0.1863 | 0.1211 | 0.1045 |
| fund_roe | neutral | 0.0286 | 0.1785 | 0.1601 | 0.1545 | 0.0924 |
| volatility | bull | 0.0878 | 0.2214 | 0.3968 | 0.3258 | 0.2630 |
| trend_lowvol | bull | 0.0681 | 0.2055 | 0.3313 | 0.3030 | 0.2158 |
| fund_pb | bull | 0.0647 | 0.1919 | 0.3369 | 0.2576 | 0.2118 |
| low_downside | bull | 0.0569 | 0.2028 | 0.2804 | 0.2955 | 0.1816 |
| mom_x_lowvol_20_20 | bull | 0.0558 | 0.1906 | 0.2929 | 0.2273 | 0.1797 |
| turnover_stability | bull | 0.0360 | 0.1499 | 0.2402 | 0.1212 | 0.1347 |
| momentum_reversal | bull | 0.0451 | 0.1987 | 0.2270 | 0.1667 | 0.1324 |
| fund_revenue_growth | bull | 0.0299 | 0.1657 | 0.1805 | 0.2273 | 0.1107 |
| rsi_vol_combo | bull | 0.0273 | 0.1831 | 0.1488 | 0.1515 | 0.0857 |
| wash_sale_score | bull | 0.0216 | 0.1532 | 0.1413 | 0.1130 | 0.0787 |
| mom_x_lowvol_20_20 | bear | 0.1313 | 0.2072 | 0.6340 | 0.3973 | 0.4429 |
| momentum_reversal | bear | 0.1054 | 0.2086 | 0.5054 | 0.3699 | 0.3462 |
| fund_gross_margin | bear | 0.0486 | 0.1485 | 0.3275 | 0.2603 | 0.2064 |
| turnover_stability | bear | 0.0326 | 0.1425 | 0.2284 | 0.1507 | 0.1314 |

### OLED

- **Neutral**: ['mom_x_lowvol_20_20', 'momentum_reversal', 'fund_profit_growth'] (单因子IC=0.0626, 组合IC=0.0801)
  - weights: [0.3552, 0.3376, 0.3072]
- **Bull**: ['fund_pe', 'volatility', 'mom_x_lowvol_20_20'] (单因子IC=0.0794, 组合IC=0.0944)
  - bull_weights: [0.4179, 0.3161, 0.266]
- **Bear**: ['momentum_reversal', 'mom_x_lowvol_20_20', 'fund_profit_growth'] (单因子IC=0.0925, 组合IC=0.1147)
  - bear_weights: [0.3563, 0.3388, 0.3049]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| mom_x_lowvol_20_20 | neutral | 0.0695 | 0.1754 | 0.3966 | 0.3215 | 0.2620 |
| momentum_reversal | neutral | 0.0693 | 0.1795 | 0.3861 | 0.2902 | 0.2491 |
| fund_profit_growth | neutral | 0.0489 | 0.1399 | 0.3496 | 0.2965 | 0.2266 |
| fund_pe | neutral | 0.0547 | 0.1565 | 0.3497 | 0.2777 | 0.2234 |
| trend_lowvol | neutral | 0.0677 | 0.1945 | 0.3481 | 0.2526 | 0.2180 |
| rsi_vol_combo | neutral | 0.0541 | 0.1698 | 0.3185 | 0.2505 | 0.1991 |
| fund_pb | neutral | 0.0573 | 0.1844 | 0.3107 | 0.2150 | 0.1888 |
| fund_score | neutral | 0.0509 | 0.1783 | 0.2855 | 0.2088 | 0.1725 |
| turnover_stability | neutral | 0.0351 | 0.1264 | 0.2775 | 0.2380 | 0.1718 |
| fund_revenue_growth | neutral | 0.0371 | 0.1389 | 0.2670 | 0.2129 | 0.1619 |
| volatility | neutral | 0.0455 | 0.1884 | 0.2415 | 0.2150 | 0.1467 |
| low_downside | neutral | 0.0363 | 0.1783 | 0.2036 | 0.1399 | 0.1160 |
| fund_roe | neutral | 0.0299 | 0.1767 | 0.1691 | 0.1127 | 0.0941 |
| fund_pe | bull | 0.0840 | 0.1352 | 0.6212 | 0.4697 | 0.4565 |
| volatility | bull | 0.0856 | 0.1738 | 0.4927 | 0.4015 | 0.3452 |
| mom_x_lowvol_20_20 | bull | 0.0686 | 0.1592 | 0.4309 | 0.3485 | 0.2905 |
| turnover_stability | bull | 0.0471 | 0.1088 | 0.4335 | 0.2879 | 0.2791 |
| low_downside | bull | 0.0626 | 0.1606 | 0.3898 | 0.2803 | 0.2496 |
| trend_lowvol | bull | 0.0701 | 0.1852 | 0.3787 | 0.2955 | 0.2453 |
| fund_pb | bull | 0.0730 | 0.1962 | 0.3720 | 0.2955 | 0.2410 |
| momentum_reversal | bull | 0.0574 | 0.1641 | 0.3499 | 0.2197 | 0.2134 |
| rsi_vol_combo | bull | 0.0473 | 0.1563 | 0.3027 | 0.2576 | 0.1904 |
| fund_score | bull | 0.0421 | 0.1736 | 0.2423 | 0.1061 | 0.1340 |
| momentum_reversal | bear | 0.1029 | 0.1850 | 0.5562 | 0.3699 | 0.3809 |
| mom_x_lowvol_20_20 | bear | 0.1059 | 0.2043 | 0.5184 | 0.3973 | 0.3622 |
| fund_profit_growth | bear | 0.0687 | 0.1473 | 0.4666 | 0.3973 | 0.3260 |
| trend_lowvol | bear | 0.0961 | 0.2005 | 0.4792 | 0.2603 | 0.3020 |
| rsi_vol_combo | bear | 0.0695 | 0.1467 | 0.4741 | 0.2603 | 0.2988 |
| fund_gross_margin | bear | 0.0478 | 0.1573 | 0.3040 | 0.1233 | 0.1707 |
| fund_score | bear | 0.0479 | 0.1780 | 0.2692 | 0.1233 | 0.1512 |
| fund_pe | bear | 0.0406 | 0.1642 | 0.2470 | 0.1507 | 0.1421 |
| fund_revenue_growth | bear | 0.0339 | 0.1411 | 0.2399 | 0.1781 | 0.1413 |

### PCB

- **Neutral**: ['mom_x_lowvol_20_20', 'momentum_reversal', 'fund_pb'] (单因子IC=0.0702, 组合IC=0.0946)
  - weights: [0.3404, 0.339, 0.3205]
- **Bull**: ['turnover_stability', 'low_downside', 'volatility'] (单因子IC=0.0609, 组合IC=0.0739)
  - bull_weights: [0.3659, 0.3357, 0.2984]
- **Bear**: ['mom_x_lowvol_20_20', 'trend_lowvol'] (单因子IC=0.1104, 组合IC=0.1252)
  - bear_weights: [0.5096, 0.4904]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| mom_x_lowvol_20_20 | neutral | 0.0737 | 0.1715 | 0.4297 | 0.3486 | 0.2898 |
| momentum_reversal | neutral | 0.0750 | 0.1753 | 0.4280 | 0.3486 | 0.2886 |
| fund_pb | neutral | 0.0619 | 0.1490 | 0.4156 | 0.3132 | 0.2728 |
| volatility | neutral | 0.0719 | 0.1841 | 0.3906 | 0.2965 | 0.2532 |
| trend_lowvol | neutral | 0.0767 | 0.1985 | 0.3864 | 0.2735 | 0.2460 |
| fund_profit_growth | neutral | 0.0493 | 0.1342 | 0.3677 | 0.2839 | 0.2360 |
| fund_pe | neutral | 0.0524 | 0.1430 | 0.3662 | 0.2443 | 0.2278 |
| rsi_vol_combo | neutral | 0.0584 | 0.1705 | 0.3427 | 0.2651 | 0.2168 |
| fund_score | neutral | 0.0435 | 0.1623 | 0.2680 | 0.1837 | 0.1586 |
| fund_revenue_growth | neutral | 0.0306 | 0.1322 | 0.2311 | 0.1628 | 0.1344 |
| low_downside | neutral | 0.0352 | 0.1757 | 0.2001 | 0.1357 | 0.1136 |
| turnover_stability | neutral | 0.0245 | 0.1321 | 0.1858 | 0.1148 | 0.1036 |
| turnover_stability | bull | 0.0408 | 0.0854 | 0.4779 | 0.3636 | 0.3258 |
| low_downside | bull | 0.0674 | 0.1581 | 0.4267 | 0.4015 | 0.2990 |
| volatility | bull | 0.0745 | 0.1837 | 0.4056 | 0.3106 | 0.2658 |
| fund_pb | bull | 0.0565 | 0.1582 | 0.3574 | 0.3636 | 0.2437 |
| mom_x_lowvol_20_20 | bull | 0.0438 | 0.1578 | 0.2777 | 0.1894 | 0.1652 |
| trend_lowvol | bull | 0.0501 | 0.1962 | 0.2552 | 0.1894 | 0.1518 |
| fund_profit_growth | bull | 0.0304 | 0.1410 | 0.2157 | 0.1667 | 0.1258 |
| momentum_reversal | bull | 0.0302 | 0.1597 | 0.1891 | 0.1591 | 0.1096 |
| fund_revenue_growth | bull | 0.0267 | 0.1514 | 0.1762 | 0.1061 | 0.0975 |
| rsi_vol_combo | bull | 0.0179 | 0.1598 | 0.1118 | 0.1364 | 0.0635 |
| mom_x_lowvol_20_20 | bear | 0.1010 | 0.2208 | 0.4572 | 0.3699 | 0.3131 |
| trend_lowvol | bear | 0.1199 | 0.2670 | 0.4490 | 0.3425 | 0.3014 |
| momentum_reversal | bear | 0.0939 | 0.2152 | 0.4363 | 0.3699 | 0.2989 |
| rsi_vol_combo | bear | 0.0699 | 0.1653 | 0.4227 | 0.3973 | 0.2953 |
| fund_profit_growth | bear | 0.0670 | 0.1528 | 0.4381 | 0.2603 | 0.2761 |
| fund_gross_margin | bear | 0.0478 | 0.1714 | 0.2787 | 0.2055 | 0.1680 |
| fund_pe | bear | 0.0270 | 0.1479 | 0.1828 | 0.1507 | 0.1052 |

### PEEK材料概念

- **Neutral**: ['fund_pb', 'fund_pe'] (单因子IC=0.0903, 组合IC=0.0997)
  - weights: [0.503, 0.497]
- **Bull**: ['fund_pb', 'low_downside', 'volatility'] (单因子IC=0.1257, 组合IC=0.1556)
  - bull_weights: [0.3543, 0.3359, 0.3099]
- **Bear**: ['mom_x_lowvol_20_20', 'momentum_reversal', 'rsi_vol_combo'] (单因子IC=0.1403, 组合IC=0.155)
  - bear_weights: [0.408, 0.3256, 0.2664]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| fund_pb | neutral | 0.0956 | 0.2363 | 0.4044 | 0.2975 | 0.2624 |
| fund_pe | neutral | 0.0850 | 0.2159 | 0.3939 | 0.3163 | 0.2592 |
| fund_score | neutral | 0.0607 | 0.2265 | 0.2682 | 0.2756 | 0.1711 |
| fund_profit_growth | neutral | 0.0591 | 0.2303 | 0.2567 | 0.2401 | 0.1592 |
| mom_x_lowvol_20_20 | neutral | 0.0492 | 0.2264 | 0.2174 | 0.1743 | 0.1276 |
| momentum_reversal | neutral | 0.0441 | 0.2235 | 0.1974 | 0.1545 | 0.1139 |
| trend_lowvol | neutral | 0.0450 | 0.2507 | 0.1796 | 0.1430 | 0.1026 |
| rsi_vol_combo | neutral | 0.0324 | 0.2186 | 0.1483 | 0.1503 | 0.0853 |
| volatility | neutral | 0.0386 | 0.2574 | 0.1500 | 0.1065 | 0.0830 |
| fund_revenue_growth | neutral | 0.0240 | 0.1867 | 0.1285 | 0.1106 | 0.0713 |
| fund_pb | bull | 0.1475 | 0.2418 | 0.6102 | 0.4470 | 0.4415 |
| low_downside | bull | 0.1066 | 0.1872 | 0.5695 | 0.4697 | 0.4185 |
| volatility | bull | 0.1231 | 0.2246 | 0.5480 | 0.4091 | 0.3861 |
| fund_pe | bull | 0.0825 | 0.2261 | 0.3650 | 0.3258 | 0.2419 |
| mom_x_lowvol_20_20 | bull | 0.0714 | 0.2088 | 0.3417 | 0.2045 | 0.2058 |
| trend_lowvol | bull | 0.0755 | 0.2272 | 0.3321 | 0.1894 | 0.1975 |
| fund_profit_growth | bull | 0.0345 | 0.1754 | 0.1965 | 0.2159 | 0.1195 |
| fund_score | bull | 0.0250 | 0.1693 | 0.1477 | 0.2424 | 0.0917 |
| exhaustion_risk | bull | 0.0272 | 0.1882 | 0.1446 | 0.1411 | 0.0825 |
| mom_x_lowvol_20_20 | bear | 0.1481 | 0.2073 | 0.7145 | 0.5342 | 0.5481 |
| momentum_reversal | bear | 0.1448 | 0.2493 | 0.5806 | 0.5068 | 0.4375 |
| rsi_vol_combo | bear | 0.1278 | 0.2398 | 0.5332 | 0.3425 | 0.3579 |
| trend_lowvol | bear | 0.0695 | 0.2437 | 0.2852 | 0.1507 | 0.1641 |
| turnover_stability | bear | 0.0515 | 0.1885 | 0.2733 | 0.1781 | 0.1610 |
| fund_pb | bear | 0.0684 | 0.3113 | 0.2197 | 0.2055 | 0.1324 |

### PPP模式

- **Neutral**: ['mom_x_lowvol_20_20', 'volatility'] (单因子IC=0.081, 组合IC=0.0961)
  - weights: [0.5054, 0.4946]
- **Bull**: ['low_downside', 'turnover_stability', 'volatility'] (单因子IC=0.0782, 组合IC=0.0987)
  - bull_weights: [0.3811, 0.3178, 0.3011]
- **Bear**: ['momentum_reversal', 'mom_x_lowvol_20_20'] (单因子IC=0.1101, 组合IC=0.1136)
  - bear_weights: [0.5263, 0.4737]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| mom_x_lowvol_20_20 | neutral | 0.0800 | 0.1913 | 0.4181 | 0.3173 | 0.2754 |
| volatility | neutral | 0.0821 | 0.2089 | 0.3930 | 0.3716 | 0.2695 |
| fund_profit_growth | neutral | 0.0510 | 0.1241 | 0.4112 | 0.3090 | 0.2691 |
| momentum_reversal | neutral | 0.0817 | 0.2012 | 0.4062 | 0.3006 | 0.2641 |
| turnover_stability | neutral | 0.0470 | 0.1194 | 0.3940 | 0.2777 | 0.2517 |
| rsi_vol_combo | neutral | 0.0629 | 0.1812 | 0.3474 | 0.2902 | 0.2241 |
| fund_score | neutral | 0.0588 | 0.1863 | 0.3155 | 0.2965 | 0.2045 |
| trend_lowvol | neutral | 0.0718 | 0.2275 | 0.3157 | 0.2276 | 0.1938 |
| fund_pb | neutral | 0.0530 | 0.1963 | 0.2697 | 0.2088 | 0.1630 |
| low_downside | neutral | 0.0465 | 0.1968 | 0.2363 | 0.2547 | 0.1482 |
| fund_gross_margin | neutral | 0.0329 | 0.1522 | 0.2160 | 0.2359 | 0.1335 |
| fund_pe | neutral | 0.0448 | 0.2207 | 0.2030 | 0.1545 | 0.1172 |
| fund_roe | neutral | 0.0386 | 0.2088 | 0.1847 | 0.1649 | 0.1076 |
| fund_revenue_growth | neutral | 0.0191 | 0.1269 | 0.1504 | 0.1524 | 0.0867 |
| low_downside | bull | 0.0920 | 0.1560 | 0.5897 | 0.4773 | 0.4356 |
| turnover_stability | bull | 0.0523 | 0.1015 | 0.5156 | 0.4091 | 0.3633 |
| volatility | bull | 0.0902 | 0.1797 | 0.5019 | 0.3712 | 0.3441 |
| fund_pb | bull | 0.0819 | 0.1781 | 0.4599 | 0.3409 | 0.3083 |
| trend_lowvol | bull | 0.0702 | 0.1978 | 0.3550 | 0.2955 | 0.2299 |
| momentum_reversal | bull | 0.0593 | 0.1695 | 0.3498 | 0.3030 | 0.2279 |
| fund_profit_growth | bull | 0.0408 | 0.1087 | 0.3758 | 0.2121 | 0.2278 |
| fund_score | bull | 0.0440 | 0.1618 | 0.2720 | 0.1894 | 0.1618 |
| fund_pe | bull | 0.0561 | 0.2096 | 0.2678 | 0.1742 | 0.1572 |
| mom_x_lowvol_20_20 | bull | 0.0403 | 0.1680 | 0.2401 | 0.1970 | 0.1437 |
| rsi_vol_combo | bull | 0.0405 | 0.1669 | 0.2424 | 0.1591 | 0.1405 |
| fund_gross_margin | bull | 0.0270 | 0.1214 | 0.2221 | 0.1894 | 0.1321 |
| stroke_phase | bull | 0.0253 | 0.1285 | 0.1967 | 0.1970 | 0.1178 |
| momentum_reversal | bear | 0.1125 | 0.2143 | 0.5253 | 0.4521 | 0.3814 |
| mom_x_lowvol_20_20 | bear | 0.1076 | 0.2233 | 0.4819 | 0.4247 | 0.3433 |
| rsi_vol_combo | bear | 0.0707 | 0.1814 | 0.3899 | 0.1507 | 0.2243 |
| fund_revenue_growth | bear | 0.0373 | 0.1330 | 0.2802 | 0.2055 | 0.1689 |

### REITs概念

- **Neutral**: ['mom_x_lowvol_20_20', 'momentum_reversal', 'turnover_stability'] (单因子IC=0.0856, 组合IC=0.1097)
  - weights: [0.3999, 0.3098, 0.2903]
- **Bull**: ['momentum_reversal', 'low_downside', 'trend_lowvol'] (单因子IC=0.0904, 组合IC=0.1078)
  - bull_weights: [0.3373, 0.3367, 0.326]
- **Bear**: ['mom_x_lowvol_20_20', 'trend_lowvol'] (单因子IC=0.2181, 组合IC=0.2694)
  - bear_weights: [0.5109, 0.4891]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| mom_x_lowvol_20_20 | neutral | 0.0958 | 0.2457 | 0.3901 | 0.2662 | 0.2470 |
| momentum_reversal | neutral | 0.0905 | 0.2912 | 0.3107 | 0.2317 | 0.1914 |
| turnover_stability | neutral | 0.0704 | 0.2442 | 0.2882 | 0.2443 | 0.1793 |
| trend_lowvol | neutral | 0.0902 | 0.3168 | 0.2849 | 0.2359 | 0.1760 |
| fund_profit_growth | neutral | 0.0629 | 0.2211 | 0.2845 | 0.2035 | 0.1712 |
| fund_score | neutral | 0.0651 | 0.2570 | 0.2532 | 0.2056 | 0.1526 |
| fund_pe | neutral | 0.0719 | 0.2933 | 0.2450 | 0.1775 | 0.1442 |
| rsi_vol_combo | neutral | 0.0624 | 0.2863 | 0.2178 | 0.1493 | 0.1252 |
| fund_roe | neutral | 0.0656 | 0.3021 | 0.2172 | 0.1472 | 0.1246 |
| fund_revenue_growth | neutral | 0.0486 | 0.2399 | 0.2027 | 0.1670 | 0.1183 |
| low_downside | neutral | 0.0568 | 0.2874 | 0.1977 | 0.1503 | 0.1137 |
| fund_pb | neutral | 0.0435 | 0.2352 | 0.1848 | 0.1336 | 0.1047 |
| volatility | neutral | 0.0283 | 0.2325 | 0.1218 | 0.1837 | 0.0721 |
| momentum_reversal | bull | 0.0853 | 0.2669 | 0.3198 | 0.2727 | 0.2035 |
| low_downside | bull | 0.0875 | 0.2759 | 0.3173 | 0.2803 | 0.2031 |
| trend_lowvol | bull | 0.0984 | 0.3032 | 0.3246 | 0.2121 | 0.1967 |
| turnover_stability | bull | 0.0689 | 0.2280 | 0.3021 | 0.2197 | 0.1842 |
| mom_x_lowvol_20_20 | bull | 0.0552 | 0.2131 | 0.2591 | 0.2348 | 0.1600 |
| fund_revenue_growth | bull | 0.0312 | 0.2607 | 0.1195 | 0.1212 | 0.0670 |
| fund_profit_growth | bull | 0.0216 | 0.2099 | 0.1031 | 0.1212 | 0.0578 |
| mom_x_lowvol_20_20 | bear | 0.2064 | 0.2282 | 0.9044 | 0.5616 | 0.7062 |
| trend_lowvol | bear | 0.2298 | 0.2747 | 0.8366 | 0.6164 | 0.6761 |
| momentum_reversal | bear | 0.2272 | 0.2819 | 0.8061 | 0.5068 | 0.6073 |
| rsi_vol_combo | bear | 0.1610 | 0.2896 | 0.5558 | 0.3425 | 0.3731 |
| fund_pe | bear | 0.1016 | 0.2932 | 0.3465 | 0.3973 | 0.2421 |
| bb_width_20 | bear | 0.0686 | 0.2680 | 0.2562 | 0.2329 | 0.1579 |

### ST股

- **Neutral**: ['volatility', 'momentum_reversal'] (单因子IC=0.0953, 组合IC=0.1148)
  - weights: [0.5231, 0.4769]
- **Bull**: ['fund_pb', 'volatility', 'low_downside'] (单因子IC=0.0907, 组合IC=0.1107)
  - bull_weights: [0.3852, 0.309, 0.3058]
- **Bear**: ['turnover_stability', 'momentum_reversal', 'mom_x_lowvol_20_20'] (单因子IC=0.1004, 组合IC=0.1372)
  - bear_weights: [0.3862, 0.3353, 0.2785]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| volatility | neutral | 0.0987 | 0.1640 | 0.6020 | 0.4927 | 0.4493 |
| momentum_reversal | neutral | 0.0918 | 0.1600 | 0.5738 | 0.4280 | 0.4097 |
| mom_x_lowvol_20_20 | neutral | 0.0887 | 0.1612 | 0.5500 | 0.4134 | 0.3887 |
| trend_lowvol | neutral | 0.0896 | 0.1780 | 0.5037 | 0.3528 | 0.3407 |
| rsi_vol_combo | neutral | 0.0693 | 0.1488 | 0.4657 | 0.3695 | 0.3189 |
| fund_pb | neutral | 0.0501 | 0.1124 | 0.4454 | 0.2881 | 0.2868 |
| turnover_stability | neutral | 0.0439 | 0.1115 | 0.3942 | 0.3090 | 0.2580 |
| low_downside | neutral | 0.0567 | 0.1577 | 0.3598 | 0.3382 | 0.2408 |
| fund_profit_growth | neutral | 0.0341 | 0.1176 | 0.2898 | 0.2589 | 0.1824 |
| fund_score | neutral | 0.0281 | 0.1371 | 0.2048 | 0.1712 | 0.1199 |
| fund_roe | neutral | 0.0220 | 0.1378 | 0.1599 | 0.1420 | 0.0913 |
| fund_pb | bull | 0.0824 | 0.1115 | 0.7393 | 0.4829 | 0.5481 |
| volatility | bull | 0.1042 | 0.1658 | 0.6285 | 0.3992 | 0.4397 |
| low_downside | bull | 0.0856 | 0.1443 | 0.5929 | 0.4677 | 0.4351 |
| trend_lowvol | bull | 0.0868 | 0.1551 | 0.5594 | 0.3688 | 0.3829 |
| momentum_reversal | bull | 0.0699 | 0.1576 | 0.4435 | 0.3764 | 0.3052 |
| mom_x_lowvol_20_20 | bull | 0.0707 | 0.1585 | 0.4462 | 0.3080 | 0.2918 |
| fund_profit_growth | bull | 0.0417 | 0.1039 | 0.4012 | 0.3460 | 0.2700 |
| rsi_vol_combo | bull | 0.0596 | 0.1543 | 0.3862 | 0.2624 | 0.2437 |
| turnover_stability | bull | 0.0314 | 0.1274 | 0.2464 | 0.2243 | 0.1508 |
| fund_pe | bull | 0.0462 | 0.1807 | 0.2558 | 0.1179 | 0.1430 |
| turnover_stability | bear | 0.0681 | 0.0868 | 0.7846 | 0.5068 | 0.5911 |
| momentum_reversal | bear | 0.1224 | 0.1764 | 0.6938 | 0.4795 | 0.5132 |
| mom_x_lowvol_20_20 | bear | 0.1107 | 0.1813 | 0.6102 | 0.3973 | 0.4263 |
| trend_lowvol | bear | 0.0947 | 0.1728 | 0.5481 | 0.3699 | 0.3754 |
| fund_revenue_growth | bear | 0.0492 | 0.0989 | 0.4972 | 0.3973 | 0.3473 |
| rsi_vol_combo | bear | 0.0817 | 0.1660 | 0.4921 | 0.3151 | 0.3235 |
| fund_score | bear | 0.0480 | 0.1366 | 0.3514 | 0.3151 | 0.2310 |
| volatility | bear | 0.0593 | 0.1658 | 0.3575 | 0.2603 | 0.2253 |
| vol_confirm | bear | 0.0244 | 0.0961 | 0.2533 | 0.3151 | 0.1666 |
| fund_pb | bear | 0.0317 | 0.1398 | 0.2268 | 0.1507 | 0.1305 |

### TOPCon电池

- **Neutral**: ['volatility', 'momentum_reversal', 'fund_pb'] (单因子IC=0.0566, 组合IC=0.079)
  - weights: [0.3619, 0.3221, 0.316]
- **Bull**: ['low_downside', 'volatility', 'fund_pb'] (单因子IC=0.091, 组合IC=0.0942)
  - bull_weights: [0.3601, 0.3368, 0.3031]
- **Bear**: ['mom_x_lowvol_20_20'] (单因子IC=0.1498, 组合IC=0.1498)
  - bear_weights: [1.0]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| volatility | neutral | 0.0624 | 0.2443 | 0.2555 | 0.2390 | 0.1583 |
| momentum_reversal | neutral | 0.0566 | 0.2466 | 0.2297 | 0.2265 | 0.1408 |
| fund_pb | neutral | 0.0508 | 0.2216 | 0.2294 | 0.2046 | 0.1382 |
| mom_x_lowvol_20_20 | neutral | 0.0540 | 0.2411 | 0.2240 | 0.2192 | 0.1366 |
| fund_pe | neutral | 0.0528 | 0.2406 | 0.2195 | 0.2025 | 0.1320 |
| rsi_vol_combo | neutral | 0.0424 | 0.2350 | 0.1806 | 0.1691 | 0.1056 |
| trend_lowvol | neutral | 0.0481 | 0.2770 | 0.1738 | 0.1691 | 0.1016 |
| low_downside | neutral | 0.0361 | 0.2313 | 0.1559 | 0.1900 | 0.0928 |
| wash_sale_score | neutral | 0.0222 | 0.1821 | 0.1221 | 0.1006 | 0.0672 |
| low_downside | bull | 0.0881 | 0.1895 | 0.4652 | 0.3258 | 0.3084 |
| volatility | bull | 0.1041 | 0.2365 | 0.4402 | 0.3106 | 0.2885 |
| fund_pb | bull | 0.0809 | 0.2065 | 0.3916 | 0.3258 | 0.2596 |
| mom_x_lowvol_20_20 | bull | 0.0728 | 0.2306 | 0.3157 | 0.2348 | 0.1949 |
| trend_lowvol | bull | 0.0693 | 0.2517 | 0.2753 | 0.2879 | 0.1773 |
| momentum_reversal | bull | 0.0650 | 0.2321 | 0.2800 | 0.1742 | 0.1644 |
| rsi_vol_combo | bull | 0.0578 | 0.2116 | 0.2732 | 0.1818 | 0.1614 |
| turnover_stability | bull | 0.0250 | 0.2008 | 0.1246 | 0.1402 | 0.0711 |
| fund_score | bull | 0.0328 | 0.2736 | 0.1199 | 0.1288 | 0.0677 |
| fund_revenue_growth | bull | 0.0255 | 0.2249 | 0.1132 | 0.1515 | 0.0652 |
| mom_x_lowvol_20_20 | bear | 0.1498 | 0.2833 | 0.5287 | 0.3973 | 0.3693 |
| momentum_reversal | bear | 0.1186 | 0.2506 | 0.4732 | 0.4795 | 0.3501 |
| wash_sale_score | bear | 0.0820 | 0.1796 | 0.4567 | 0.3500 | 0.3083 |
| trend_lowvol | bear | 0.1051 | 0.2406 | 0.4367 | 0.3151 | 0.2872 |
| fund_pb | bear | 0.0818 | 0.1987 | 0.4118 | 0.2877 | 0.2651 |
| volatility | bear | 0.0690 | 0.2402 | 0.2872 | 0.2329 | 0.1770 |
| rsi_vol_combo | bear | 0.0562 | 0.2397 | 0.2345 | 0.2877 | 0.1510 |
| fund_profit_growth | bear | 0.0519 | 0.2073 | 0.2506 | 0.1507 | 0.1442 |
| fund_score | bear | 0.0568 | 0.2593 | 0.2189 | 0.1233 | 0.1229 |

### Web3.0

- **Neutral**: ['fund_pb', 'momentum_reversal', 'trend_lowvol'] (单因子IC=0.1165, 组合IC=0.1627)
  - weights: [0.3526, 0.3286, 0.3188]
- **Bull**: ['fund_pb', 'trend_lowvol', 'volatility'] (单因子IC=0.1401, 组合IC=0.1701)
  - bull_weights: [0.383, 0.3124, 0.3046]
- **Bear**: ['fund_profit_growth', 'fund_score', 'mom_x_lowvol_20_20'] (单因子IC=0.0953, 组合IC=0.1243)
  - bear_weights: [0.3988, 0.3642, 0.237]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| fund_pb | neutral | 0.1068 | 0.1811 | 0.5898 | 0.4196 | 0.4187 |
| momentum_reversal | neutral | 0.1194 | 0.2185 | 0.5466 | 0.4280 | 0.3902 |
| trend_lowvol | neutral | 0.1234 | 0.2296 | 0.5372 | 0.4092 | 0.3785 |
| mom_x_lowvol_20_20 | neutral | 0.1135 | 0.2255 | 0.5033 | 0.3925 | 0.3504 |
| volatility | neutral | 0.1207 | 0.2422 | 0.4983 | 0.3779 | 0.3433 |
| rsi_vol_combo | neutral | 0.0937 | 0.2068 | 0.4531 | 0.3048 | 0.2956 |
| low_downside | neutral | 0.0634 | 0.2168 | 0.2926 | 0.2192 | 0.1783 |
| fund_profit_growth | neutral | 0.0461 | 0.1678 | 0.2746 | 0.2516 | 0.1718 |
| fund_pe | neutral | 0.0546 | 0.2148 | 0.2539 | 0.1628 | 0.1476 |
| fund_roe | neutral | 0.0494 | 0.2036 | 0.2428 | 0.1900 | 0.1444 |
| fund_score | neutral | 0.0472 | 0.2013 | 0.2343 | 0.2213 | 0.1431 |
| turnover_stability | neutral | 0.0324 | 0.1651 | 0.1965 | 0.1681 | 0.1148 |
| fund_revenue_growth | neutral | 0.0105 | 0.1609 | 0.0652 | 0.1148 | 0.0364 |
| fund_pb | bull | 0.1277 | 0.1623 | 0.7866 | 0.5227 | 0.5989 |
| trend_lowvol | bull | 0.1508 | 0.2269 | 0.6648 | 0.4697 | 0.4885 |
| volatility | bull | 0.1418 | 0.2244 | 0.6318 | 0.5076 | 0.4762 |
| mom_x_lowvol_20_20 | bull | 0.1165 | 0.2303 | 0.5060 | 0.3864 | 0.3507 |
| momentum_reversal | bull | 0.0985 | 0.2234 | 0.4410 | 0.3485 | 0.2973 |
| low_downside | bull | 0.0814 | 0.2153 | 0.3781 | 0.3258 | 0.2506 |
| fund_pe | bull | 0.0539 | 0.1997 | 0.2698 | 0.1970 | 0.1615 |
| fund_score | bull | 0.0404 | 0.1893 | 0.2135 | 0.2197 | 0.1302 |
| fund_profit_growth | bull | 0.0347 | 0.1631 | 0.2124 | 0.2197 | 0.1296 |
| rsi_vol_combo | bull | 0.0496 | 0.2413 | 0.2054 | 0.2045 | 0.1237 |
| fund_revenue_growth | bull | 0.0348 | 0.1658 | 0.2100 | 0.1364 | 0.1193 |
| fund_profit_growth | bear | 0.0994 | 0.1849 | 0.5376 | 0.3973 | 0.3756 |
| fund_score | bear | 0.1119 | 0.2279 | 0.4909 | 0.3973 | 0.3429 |
| mom_x_lowvol_20_20 | bear | 0.0747 | 0.2156 | 0.3467 | 0.2877 | 0.2232 |
| trend_lowvol | bear | 0.0558 | 0.1724 | 0.3235 | 0.1781 | 0.1906 |
| fund_revenue_growth | bear | 0.0341 | 0.1337 | 0.2550 | 0.2329 | 0.1572 |
| momentum_reversal | bear | 0.0503 | 0.2264 | 0.2220 | 0.2055 | 0.1338 |
| fund_gross_margin | bear | 0.0341 | 0.1816 | 0.1879 | 0.1507 | 0.1081 |

### WiFi

- **Neutral**: ['fund_pe', 'fund_pb', 'mom_x_lowvol_20_20'] (单因子IC=0.0911, 组合IC=0.1058)
  - weights: [0.3715, 0.3314, 0.2971]
- **Bull**: ['volatility', 'fund_pb'] (单因子IC=0.0978, 组合IC=0.1263)
  - bull_weights: [0.5089, 0.4911]
- **Bear**: ['wash_sale_score', 'mom_x_lowvol_20_20', 'momentum_reversal'] (单因子IC=0.0905, 组合IC=0.1079)
  - bear_weights: [0.441, 0.2963, 0.2627]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| fund_pe | neutral | 0.0963 | 0.2330 | 0.4131 | 0.2996 | 0.2685 |
| fund_pb | neutral | 0.0925 | 0.2434 | 0.3801 | 0.2599 | 0.2395 |
| mom_x_lowvol_20_20 | neutral | 0.0846 | 0.2450 | 0.3454 | 0.2432 | 0.2147 |
| momentum_reversal | neutral | 0.0833 | 0.2528 | 0.3296 | 0.2349 | 0.2035 |
| volatility | neutral | 0.0702 | 0.2339 | 0.3001 | 0.2380 | 0.1858 |
| trend_lowvol | neutral | 0.0761 | 0.2619 | 0.2906 | 0.2140 | 0.1764 |
| low_downside | neutral | 0.0506 | 0.2512 | 0.2015 | 0.1524 | 0.1161 |
| rsi_vol_combo | neutral | 0.0470 | 0.2359 | 0.1993 | 0.1482 | 0.1144 |
| fund_profit_growth | neutral | 0.0438 | 0.2190 | 0.2000 | 0.1232 | 0.1123 |
| fund_roe | neutral | 0.0450 | 0.2446 | 0.1842 | 0.1451 | 0.1055 |
| fund_score | neutral | 0.0427 | 0.2386 | 0.1789 | 0.1378 | 0.1017 |
| volatility | bull | 0.0978 | 0.2650 | 0.3689 | 0.4015 | 0.2585 |
| fund_pb | bull | 0.0978 | 0.2674 | 0.3659 | 0.3636 | 0.2495 |
| low_downside | bull | 0.0749 | 0.2360 | 0.3172 | 0.1894 | 0.1886 |
| mom_x_lowvol_20_20 | bull | 0.0694 | 0.2438 | 0.2846 | 0.2879 | 0.1832 |
| momentum_reversal | bull | 0.0676 | 0.2469 | 0.2739 | 0.2879 | 0.1764 |
| fund_pe | bull | 0.0529 | 0.2181 | 0.2425 | 0.1742 | 0.1424 |
| trend_lowvol | bull | 0.0501 | 0.2377 | 0.2108 | 0.1894 | 0.1254 |
| fund_revenue_growth | bull | 0.0355 | 0.2429 | 0.1462 | 0.1174 | 0.0817 |
| wash_sale_score | bear | 0.0986 | 0.2164 | 0.4555 | 0.4667 | 0.3340 |
| mom_x_lowvol_20_20 | bear | 0.0896 | 0.2515 | 0.3562 | 0.2603 | 0.2244 |
| momentum_reversal | bear | 0.0835 | 0.2586 | 0.3227 | 0.2329 | 0.1989 |
| rsi_vol_combo | bear | 0.0759 | 0.2495 | 0.3042 | 0.1233 | 0.1709 |
| fund_pb | bear | 0.0924 | 0.3052 | 0.3027 | 0.1233 | 0.1700 |
| volatility | bear | 0.0551 | 0.2387 | 0.2307 | 0.1507 | 0.1327 |

### 一带一路

- **Neutral**: ['turnover_stability', 'mom_x_lowvol_20_20', 'momentum_reversal'] (单因子IC=0.0662, 组合IC=0.085)
  - weights: [0.3552, 0.3423, 0.3025]
- **Bull**: ['low_downside', 'volatility', 'turnover_stability'] (单因子IC=0.0725, 组合IC=0.0938)
  - bull_weights: [0.4015, 0.3541, 0.2443]
- **Bear**: ['mom_x_lowvol_20_20'] (单因子IC=0.1323, 组合IC=0.1323)
  - bear_weights: [1.0]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| turnover_stability | neutral | 0.0462 | 0.0896 | 0.5153 | 0.3862 | 0.3571 |
| mom_x_lowvol_20_20 | neutral | 0.0772 | 0.1534 | 0.5034 | 0.3674 | 0.3442 |
| momentum_reversal | neutral | 0.0751 | 0.1637 | 0.4588 | 0.3257 | 0.3041 |
| trend_lowvol | neutral | 0.0847 | 0.1889 | 0.4484 | 0.3194 | 0.2958 |
| fund_profit_growth | neutral | 0.0403 | 0.1039 | 0.3881 | 0.3612 | 0.2641 |
| volatility | neutral | 0.0643 | 0.1763 | 0.3647 | 0.3319 | 0.2429 |
| rsi_vol_combo | neutral | 0.0518 | 0.1566 | 0.3310 | 0.2484 | 0.2066 |
| fund_pb | neutral | 0.0623 | 0.1822 | 0.3420 | 0.2025 | 0.2056 |
| fund_score | neutral | 0.0385 | 0.1342 | 0.2868 | 0.1962 | 0.1716 |
| fund_pe | neutral | 0.0500 | 0.1850 | 0.2704 | 0.1879 | 0.1606 |
| fund_revenue_growth | neutral | 0.0230 | 0.0917 | 0.2504 | 0.2109 | 0.1516 |
| low_downside | neutral | 0.0371 | 0.1774 | 0.2093 | 0.1900 | 0.1246 |
| fund_gross_margin | neutral | 0.0083 | 0.0984 | 0.0841 | 0.1086 | 0.0466 |
| low_downside | bull | 0.0910 | 0.1298 | 0.7012 | 0.4924 | 0.5233 |
| volatility | bull | 0.0878 | 0.1434 | 0.6123 | 0.5076 | 0.4615 |
| turnover_stability | bull | 0.0388 | 0.0836 | 0.4645 | 0.3712 | 0.3184 |
| fund_pb | bull | 0.0750 | 0.1629 | 0.4606 | 0.2879 | 0.2966 |
| trend_lowvol | bull | 0.0659 | 0.1585 | 0.4155 | 0.3106 | 0.2723 |
| mom_x_lowvol_20_20 | bull | 0.0436 | 0.1222 | 0.3565 | 0.2576 | 0.2241 |
| momentum_reversal | bull | 0.0495 | 0.1371 | 0.3614 | 0.1894 | 0.2149 |
| fund_pe | bull | 0.0538 | 0.1614 | 0.3330 | 0.2197 | 0.2031 |
| fund_revenue_growth | bull | 0.0200 | 0.0956 | 0.2088 | 0.1894 | 0.1242 |
| stroke_phase | bull | 0.0165 | 0.0865 | 0.1909 | 0.1742 | 0.1121 |
| rsi_vol_combo | bull | 0.0216 | 0.1299 | 0.1664 | 0.1212 | 0.0933 |
| fund_score | bull | 0.0220 | 0.1419 | 0.1551 | 0.1136 | 0.0864 |
| mom_x_lowvol_20_20 | bear | 0.1323 | 0.2085 | 0.6344 | 0.5616 | 0.4954 |
| momentum_reversal | bear | 0.1164 | 0.2078 | 0.5604 | 0.5342 | 0.4299 |
| fund_revenue_growth | bear | 0.0486 | 0.0898 | 0.5414 | 0.4795 | 0.4005 |
| fund_profit_growth | bear | 0.0545 | 0.1112 | 0.4904 | 0.3973 | 0.3426 |
| fund_score | bear | 0.0618 | 0.1445 | 0.4277 | 0.3973 | 0.2988 |
| rsi_vol_combo | bear | 0.0672 | 0.1568 | 0.4288 | 0.2055 | 0.2585 |
| trend_lowvol | bear | 0.0747 | 0.1901 | 0.3930 | 0.2055 | 0.2369 |
| fund_gross_margin | bear | 0.0232 | 0.1069 | 0.2166 | 0.2603 | 0.1365 |
| bb_width_20 | bear | 0.0432 | 0.2080 | 0.2077 | 0.1781 | 0.1223 |

### 上海自贸

- **Neutral**: ['trend_lowvol', 'volatility'] (单因子IC=0.0805, 组合IC=0.0981)
  - weights: [0.5444, 0.4556]
- **Bull**: ['low_downside', 'turnover_stability', 'rsi_vol_combo'] (单因子IC=0.0569, 组合IC=0.0905)
  - bull_weights: [0.4201, 0.3527, 0.2272]
- **Bear**: ['mom_x_lowvol_20_20'] (单因子IC=0.0537, 组合IC=0.0537)
  - bear_weights: [1.0]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| trend_lowvol | neutral | 0.0914 | 0.2606 | 0.3506 | 0.3330 | 0.2337 |
| volatility | neutral | 0.0696 | 0.2240 | 0.3107 | 0.2589 | 0.1955 |
| low_downside | neutral | 0.0661 | 0.2454 | 0.2695 | 0.2338 | 0.1663 |
| fund_pb | neutral | 0.0651 | 0.2380 | 0.2735 | 0.2004 | 0.1641 |
| mom_x_lowvol_20_20 | neutral | 0.0597 | 0.2321 | 0.2573 | 0.2234 | 0.1574 |
| momentum_reversal | neutral | 0.0636 | 0.2567 | 0.2479 | 0.1973 | 0.1484 |
| turnover_stability | neutral | 0.0354 | 0.2135 | 0.1657 | 0.1409 | 0.0945 |
| rsi_vol_combo | neutral | 0.0390 | 0.2485 | 0.1568 | 0.1127 | 0.0872 |
| low_downside | bull | 0.0755 | 0.2315 | 0.3261 | 0.2879 | 0.2100 |
| turnover_stability | bull | 0.0532 | 0.1970 | 0.2699 | 0.3068 | 0.1763 |
| rsi_vol_combo | bull | 0.0421 | 0.2079 | 0.2026 | 0.1212 | 0.1136 |
| volatility | bull | 0.0421 | 0.2234 | 0.1883 | 0.1439 | 0.1077 |
| fund_revenue_growth | bull | 0.0313 | 0.2036 | 0.1538 | 0.2197 | 0.0938 |
| mom_x_lowvol_20_20 | bear | 0.0537 | 0.2861 | 0.1878 | 0.1233 | 0.1055 |

### 上证180_

- **Neutral**: ['fund_pe', 'fund_pb', 'trend_lowvol'] (单因子IC=0.0638, 组合IC=0.0797)
  - weights: [0.3703, 0.3678, 0.2619]
- **Bull**: ['exhaustion_risk', 'relative_strength', 'ema20_slope'] (单因子IC=0.0361, 组合IC=0.0449)
  - bull_weights: [0.3792, 0.3231, 0.2977]
- **Bear**: ['mom_x_lowvol_20_20', 'bb_width_20'] (单因子IC=0.1214, 组合IC=0.1514)
  - bear_weights: [0.5069, 0.4931]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| fund_pe | neutral | 0.0754 | 0.2743 | 0.2748 | 0.1712 | 0.1609 |
| fund_pb | neutral | 0.0720 | 0.2665 | 0.2700 | 0.1837 | 0.1598 |
| trend_lowvol | neutral | 0.0440 | 0.2274 | 0.1933 | 0.1775 | 0.1138 |
| volatility | neutral | 0.0277 | 0.1563 | 0.1772 | 0.1190 | 0.0992 |
| fund_profit_growth | neutral | 0.0247 | 0.1452 | 0.1698 | 0.1441 | 0.0971 |
| fund_roe | neutral | 0.0152 | 0.1615 | 0.0944 | 0.1148 | 0.0526 |
| exhaustion_risk | bull | 0.0267 | 0.1212 | 0.2206 | 0.2424 | 0.1370 |
| relative_strength | bull | 0.0414 | 0.2228 | 0.1857 | 0.2576 | 0.1168 |
| ema20_slope | bull | 0.0403 | 0.2313 | 0.1742 | 0.2348 | 0.1076 |
| ma_alignment | bull | 0.0346 | 0.2342 | 0.1478 | 0.1894 | 0.0879 |
| fund_gross_margin | bull | 0.0213 | 0.1389 | 0.1535 | 0.1364 | 0.0872 |
| low_downside | bull | 0.0280 | 0.2024 | 0.1381 | 0.1212 | 0.0774 |
| top_fractal_volume | bull | 0.0146 | 0.1057 | 0.1378 | 0.1197 | 0.0771 |
| volatility | bull | 0.0242 | 0.1879 | 0.1287 | 0.1439 | 0.0736 |
| fund_pe | bull | 0.0337 | 0.3139 | 0.1074 | 0.1136 | 0.0598 |
| mom_x_lowvol_20_20 | bear | 0.1005 | 0.1718 | 0.5846 | 0.5890 | 0.4645 |
| bb_width_20 | bear | 0.1423 | 0.2200 | 0.6468 | 0.3973 | 0.4519 |
| momentum_reversal | bear | 0.1318 | 0.2348 | 0.5613 | 0.4521 | 0.4075 |
| wash_sale_score | bear | 0.0466 | 0.1082 | 0.4308 | 0.2113 | 0.2609 |
| trend_lowvol | bear | 0.1146 | 0.3115 | 0.3678 | 0.2603 | 0.2318 |
| rsi_vol_combo | bear | 0.0613 | 0.2138 | 0.2868 | 0.1507 | 0.1650 |
| vol_confirm | bear | 0.0298 | 0.1358 | 0.2196 | 0.1507 | 0.1263 |
| fund_profit_growth | bear | 0.0275 | 0.1347 | 0.2042 | 0.2329 | 0.1259 |

### 上证50_

- **Neutral**: ['fund_pb', 'fund_pe', 'trend_lowvol'] (单因子IC=0.0724, 组合IC=0.0906)
  - weights: [0.3853, 0.3673, 0.2474]
- **Bull**: ['ema20_slope', 'fund_gross_margin', 'relative_strength'] (单因子IC=0.0682, 组合IC=0.0871)
  - bull_weights: [0.3543, 0.3234, 0.3223]
- **Bear**: ['bb_width_20', 'momentum_reversal', 'wash_sale_score'] (单因子IC=0.1491, 组合IC=0.2252)
  - bear_weights: [0.4952, 0.2716, 0.2332]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| fund_pb | neutral | 0.0867 | 0.3126 | 0.2774 | 0.2067 | 0.1674 |
| fund_pe | neutral | 0.0835 | 0.3143 | 0.2658 | 0.2004 | 0.1595 |
| trend_lowvol | neutral | 0.0470 | 0.2572 | 0.1828 | 0.1754 | 0.1074 |
| top_fractal_volume | neutral | 0.0198 | 0.1569 | 0.1261 | 0.1370 | 0.0717 |
| turnover_stability | neutral | 0.0115 | 0.2053 | 0.0559 | 0.1138 | 0.0311 |
| ema20_slope | bull | 0.0805 | 0.2663 | 0.3024 | 0.3182 | 0.1993 |
| fund_gross_margin | bull | 0.0522 | 0.1794 | 0.2911 | 0.2500 | 0.1819 |
| relative_strength | bull | 0.0719 | 0.2629 | 0.2735 | 0.3258 | 0.1813 |
| ma_alignment | bull | 0.0671 | 0.2608 | 0.2574 | 0.2727 | 0.1638 |
| exhaustion_risk | bull | 0.0477 | 0.1805 | 0.2643 | 0.2385 | 0.1637 |
| vol_confirm | bull | 0.0426 | 0.2365 | 0.1801 | 0.1061 | 0.0996 |
| stroke_phase | bull | 0.0314 | 0.1910 | 0.1644 | 0.1591 | 0.0953 |
| turnover_stability | bull | 0.0271 | 0.1977 | 0.1369 | 0.1288 | 0.0773 |
| low_downside | bull | 0.0269 | 0.2663 | 0.1009 | 0.1364 | 0.0574 |
| bb_width_20 | bear | 0.2073 | 0.2399 | 0.8643 | 0.5890 | 0.6867 |
| momentum_reversal | bear | 0.1443 | 0.2729 | 0.5287 | 0.4247 | 0.3766 |
| wash_sale_score | bear | 0.0957 | 0.1836 | 0.5210 | 0.2414 | 0.3234 |
| mom_x_lowvol_20_20 | bear | 0.1057 | 0.2358 | 0.4482 | 0.3973 | 0.3131 |
| fund_revenue_growth | bear | 0.0671 | 0.1530 | 0.4387 | 0.3151 | 0.2885 |
| fund_profit_growth | bear | 0.0634 | 0.1810 | 0.3503 | 0.2329 | 0.2160 |
| trend_lowvol | bear | 0.1079 | 0.3228 | 0.3343 | 0.2329 | 0.2061 |
| rsi_vol_combo | bear | 0.0691 | 0.2536 | 0.2725 | 0.2877 | 0.1754 |

### 东北振兴

- **Neutral**: ['trend_lowvol', 'low_downside'] (单因子IC=0.0837, 组合IC=0.1039)
  - weights: [0.5506, 0.4494]
- **Bull**: ['low_downside', 'turnover_stability'] (单因子IC=0.1146, 组合IC=0.1448)
  - bull_weights: [0.53, 0.47]
- **Bear**: ['mom_x_lowvol_20_20', 'momentum_reversal', 'trend_lowvol'] (单因子IC=0.105, 组合IC=0.1136)
  - bear_weights: [0.3653, 0.3418, 0.2929]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| trend_lowvol | neutral | 0.0944 | 0.2147 | 0.4397 | 0.3820 | 0.3039 |
| low_downside | neutral | 0.0730 | 0.1949 | 0.3748 | 0.3236 | 0.2480 |
| volatility | neutral | 0.0756 | 0.2011 | 0.3762 | 0.2610 | 0.2372 |
| turnover_stability | neutral | 0.0555 | 0.1574 | 0.3525 | 0.3006 | 0.2293 |
| momentum_reversal | neutral | 0.0679 | 0.2019 | 0.3363 | 0.3278 | 0.2232 |
| fund_pb | neutral | 0.0629 | 0.1905 | 0.3299 | 0.2276 | 0.2025 |
| mom_x_lowvol_20_20 | neutral | 0.0574 | 0.1989 | 0.2888 | 0.2860 | 0.1857 |
| fund_pe | neutral | 0.0547 | 0.2297 | 0.2380 | 0.1983 | 0.1426 |
| rsi_vol_combo | neutral | 0.0473 | 0.1974 | 0.2396 | 0.1681 | 0.1399 |
| fund_profit_growth | neutral | 0.0261 | 0.1644 | 0.1584 | 0.1608 | 0.0920 |
| fund_score | neutral | 0.0185 | 0.1836 | 0.1008 | 0.1023 | 0.0556 |
| low_downside | bull | 0.1259 | 0.1684 | 0.7480 | 0.5530 | 0.5808 |
| turnover_stability | bull | 0.1033 | 0.1489 | 0.6937 | 0.4848 | 0.5150 |
| volatility | bull | 0.1043 | 0.1716 | 0.6079 | 0.4848 | 0.4513 |
| trend_lowvol | bull | 0.0756 | 0.1902 | 0.3975 | 0.2955 | 0.2575 |
| fund_pb | bull | 0.0782 | 0.1963 | 0.3983 | 0.2576 | 0.2505 |
| momentum_reversal | bull | 0.0577 | 0.1726 | 0.3344 | 0.2652 | 0.2115 |
| fund_pe | bull | 0.0590 | 0.2324 | 0.2539 | 0.2348 | 0.1568 |
| fund_revenue_growth | bull | 0.0340 | 0.1370 | 0.2482 | 0.1970 | 0.1485 |
| mom_x_lowvol_20_20 | bull | 0.0322 | 0.1653 | 0.1949 | 0.1515 | 0.1122 |
| stroke_phase | bull | 0.0272 | 0.1494 | 0.1820 | 0.1212 | 0.1020 |
| rsi_vol_combo | bull | 0.0317 | 0.1768 | 0.1793 | 0.1136 | 0.0999 |
| fund_score | bull | 0.0238 | 0.1728 | 0.1378 | 0.1364 | 0.0783 |
| fund_gross_margin | bull | 0.0172 | 0.1462 | 0.1178 | 0.1288 | 0.0665 |
| mom_x_lowvol_20_20 | bear | 0.1013 | 0.1655 | 0.6124 | 0.3699 | 0.4195 |
| momentum_reversal | bear | 0.1070 | 0.1868 | 0.5730 | 0.3699 | 0.3925 |
| trend_lowvol | bear | 0.1067 | 0.1999 | 0.5338 | 0.2603 | 0.3364 |
| fund_revenue_growth | bear | 0.0535 | 0.1245 | 0.4292 | 0.3973 | 0.2999 |
| fund_score | bear | 0.0641 | 0.1603 | 0.3998 | 0.2877 | 0.2574 |
| rsi_vol_combo | bear | 0.0618 | 0.1525 | 0.4051 | 0.1644 | 0.2359 |
| vol_confirm | bear | 0.0554 | 0.1812 | 0.3060 | 0.2603 | 0.1928 |

### 东数西算

- **Neutral**: ['fund_pb', 'volatility', 'trend_lowvol'] (单因子IC=0.0832, 组合IC=0.1083)
  - weights: [0.3896, 0.3221, 0.2883]
- **Bull**: ['volatility'] (单因子IC=0.1409, 组合IC=0.1409)
  - bull_weights: [1.0]
- **Bear**: ['trend_lowvol', 'mom_x_lowvol_20_20'] (单因子IC=0.1502, 组合IC=0.1717)
  - bear_weights: [0.5466, 0.4534]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| fund_pb | neutral | 0.0874 | 0.1978 | 0.4418 | 0.3236 | 0.2924 |
| volatility | neutral | 0.0873 | 0.2351 | 0.3711 | 0.3027 | 0.2417 |
| trend_lowvol | neutral | 0.0750 | 0.2232 | 0.3360 | 0.2881 | 0.2164 |
| momentum_reversal | neutral | 0.0658 | 0.2038 | 0.3230 | 0.2693 | 0.2050 |
| mom_x_lowvol_20_20 | neutral | 0.0648 | 0.2071 | 0.3127 | 0.2839 | 0.2007 |
| fund_pe | neutral | 0.0514 | 0.2293 | 0.2240 | 0.1399 | 0.1277 |
| fund_roe | neutral | 0.0468 | 0.2094 | 0.2235 | 0.1315 | 0.1265 |
| turnover_stability | neutral | 0.0322 | 0.1493 | 0.2160 | 0.1503 | 0.1243 |
| fund_score | neutral | 0.0414 | 0.1932 | 0.2145 | 0.1461 | 0.1229 |
| rsi_vol_combo | neutral | 0.0413 | 0.1958 | 0.2109 | 0.1441 | 0.1207 |
| low_downside | neutral | 0.0471 | 0.2310 | 0.2038 | 0.1795 | 0.1202 |
| fund_profit_growth | neutral | 0.0270 | 0.1606 | 0.1682 | 0.1357 | 0.0955 |
| volatility | bull | 0.1409 | 0.1946 | 0.7240 | 0.5379 | 0.5567 |
| low_downside | bull | 0.0959 | 0.1920 | 0.4994 | 0.3636 | 0.3405 |
| mom_x_lowvol_20_20 | bull | 0.0729 | 0.1904 | 0.3826 | 0.3333 | 0.2551 |
| fund_pb | bull | 0.0741 | 0.1943 | 0.3815 | 0.3258 | 0.2529 |
| trend_lowvol | bull | 0.0763 | 0.2122 | 0.3595 | 0.3409 | 0.2410 |
| momentum_reversal | bull | 0.0491 | 0.1795 | 0.2735 | 0.2348 | 0.1688 |
| fund_pe | bull | 0.0507 | 0.1965 | 0.2582 | 0.2121 | 0.1565 |
| rsi_vol_combo | bull | 0.0377 | 0.1590 | 0.2368 | 0.2273 | 0.1453 |
| fund_roe | bull | 0.0361 | 0.1789 | 0.2016 | 0.2121 | 0.1222 |
| turnover_stability | bull | 0.0277 | 0.1497 | 0.1854 | 0.1591 | 0.1075 |
| fund_score | bull | 0.0328 | 0.1804 | 0.1815 | 0.1591 | 0.1052 |
| stroke_phase | bull | 0.0243 | 0.1355 | 0.1792 | 0.1439 | 0.1025 |
| fund_revenue_growth | bull | 0.0186 | 0.1626 | 0.1143 | 0.1136 | 0.0637 |
| trend_lowvol | bear | 0.1586 | 0.1868 | 0.8491 | 0.6164 | 0.6862 |
| mom_x_lowvol_20_20 | bear | 0.1419 | 0.1946 | 0.7290 | 0.5616 | 0.5692 |
| momentum_reversal | bear | 0.1358 | 0.1862 | 0.7294 | 0.5342 | 0.5596 |
| rsi_vol_combo | bear | 0.0873 | 0.1669 | 0.5229 | 0.3425 | 0.3510 |
| fund_profit_growth | bear | 0.0623 | 0.1427 | 0.4362 | 0.2329 | 0.2689 |

### 东方财富热股

- **Neutral**: ['trend_lowvol', 'momentum_reversal'] (单因子IC=0.0671, 组合IC=0.0719)
  - weights: [0.5253, 0.4747]
- **Bull**: ['low_downside'] (单因子IC=0.0736, 组合IC=0.0736)
  - bull_weights: [1.0]
- **Bear**: ['trend_lowvol', 'mom_x_lowvol_20_20'] (单因子IC=0.1056, 组合IC=0.1211)
  - bear_weights: [0.5433, 0.4567]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| trend_lowvol | neutral | 0.0717 | 0.2059 | 0.3483 | 0.2693 | 0.2210 |
| momentum_reversal | neutral | 0.0626 | 0.1972 | 0.3174 | 0.2589 | 0.1998 |
| mom_x_lowvol_20_20 | neutral | 0.0571 | 0.1851 | 0.3084 | 0.2568 | 0.1938 |
| volatility | neutral | 0.0473 | 0.1864 | 0.2536 | 0.2046 | 0.1528 |
| fund_pe | neutral | 0.0440 | 0.1756 | 0.2508 | 0.2025 | 0.1508 |
| fund_pb | neutral | 0.0468 | 0.1888 | 0.2481 | 0.1420 | 0.1416 |
| rsi_vol_combo | neutral | 0.0422 | 0.1944 | 0.2169 | 0.1712 | 0.1270 |
| fund_profit_growth | neutral | 0.0332 | 0.1764 | 0.1883 | 0.1190 | 0.1054 |
| fund_roe | neutral | 0.0225 | 0.2019 | 0.1112 | 0.1127 | 0.0619 |
| fund_revenue_growth | neutral | 0.0182 | 0.1758 | 0.1032 | 0.1148 | 0.0576 |
| fund_score | neutral | 0.0198 | 0.2079 | 0.0952 | 0.1002 | 0.0524 |
| low_downside | bull | 0.0736 | 0.1661 | 0.4432 | 0.2803 | 0.2837 |
| volatility | bull | 0.0565 | 0.1604 | 0.3521 | 0.3864 | 0.2441 |
| fund_pb | bull | 0.0393 | 0.2100 | 0.1869 | 0.1591 | 0.1083 |
| turnover_stability | bull | 0.0214 | 0.1511 | 0.1416 | 0.1061 | 0.0783 |
| relative_strength | bull | 0.0200 | 0.1738 | 0.1151 | 0.1136 | 0.0641 |
| trend_lowvol | bear | 0.1050 | 0.1775 | 0.5918 | 0.3425 | 0.3972 |
| mom_x_lowvol_20_20 | bear | 0.1063 | 0.2136 | 0.4974 | 0.3425 | 0.3339 |
| momentum_reversal | bear | 0.0941 | 0.2079 | 0.4524 | 0.4247 | 0.3223 |
| rsi_vol_combo | bear | 0.0787 | 0.1739 | 0.4524 | 0.4247 | 0.3223 |
| bb_width_20 | bear | 0.0570 | 0.1661 | 0.3430 | 0.3425 | 0.2302 |
| fund_revenue_growth | bear | 0.0433 | 0.1509 | 0.2870 | 0.2877 | 0.1847 |
| fund_pb | bear | 0.0499 | 0.1622 | 0.3075 | 0.1781 | 0.1812 |
| wash_sale_score | bear | 0.0313 | 0.1172 | 0.2674 | 0.1186 | 0.1496 |

### 中俄贸易概念

- **Neutral**: ['trend_lowvol', 'fund_pb', 'mom_x_lowvol_20_20'] (单因子IC=0.1047, 组合IC=0.148)
  - weights: [0.3416, 0.3335, 0.325]
- **Bull**: ['fund_pb', 'volatility', 'turnover_stability'] (单因子IC=0.1076, 组合IC=0.1495)
  - bull_weights: [0.3393, 0.3315, 0.3292]
- **Bear**: ['rsi_vol_combo', 'momentum_reversal'] (单因子IC=0.1712, 组合IC=0.1799)
  - bear_weights: [0.5364, 0.4636]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| trend_lowvol | neutral | 0.1174 | 0.2013 | 0.5831 | 0.4509 | 0.4230 |
| fund_pb | neutral | 0.0910 | 0.1543 | 0.5896 | 0.4008 | 0.4130 |
| mom_x_lowvol_20_20 | neutral | 0.1057 | 0.1903 | 0.5556 | 0.4489 | 0.4025 |
| momentum_reversal | neutral | 0.1041 | 0.1895 | 0.5491 | 0.4238 | 0.3909 |
| turnover_stability | neutral | 0.0637 | 0.1501 | 0.4244 | 0.2860 | 0.2729 |
| rsi_vol_combo | neutral | 0.0702 | 0.1914 | 0.3668 | 0.3445 | 0.2466 |
| volatility | neutral | 0.0787 | 0.2141 | 0.3674 | 0.2547 | 0.2305 |
| fund_pe | neutral | 0.0463 | 0.1811 | 0.2559 | 0.1962 | 0.1530 |
| fund_profit_growth | neutral | 0.0325 | 0.1468 | 0.2213 | 0.1608 | 0.1284 |
| low_downside | neutral | 0.0371 | 0.2010 | 0.1845 | 0.1670 | 0.1077 |
| fund_score | neutral | 0.0274 | 0.1663 | 0.1650 | 0.1086 | 0.0915 |
| fund_gross_margin | neutral | 0.0113 | 0.1341 | 0.0841 | 0.1461 | 0.0482 |
| fund_pb | bull | 0.1115 | 0.1656 | 0.6733 | 0.4924 | 0.5024 |
| volatility | bull | 0.1196 | 0.1846 | 0.6480 | 0.5152 | 0.4909 |
| turnover_stability | bull | 0.0917 | 0.1375 | 0.6668 | 0.4621 | 0.4875 |
| low_downside | bull | 0.0915 | 0.1641 | 0.5573 | 0.4697 | 0.4096 |
| momentum_reversal | bull | 0.0932 | 0.1650 | 0.5648 | 0.4470 | 0.4086 |
| mom_x_lowvol_20_20 | bull | 0.0861 | 0.1655 | 0.5203 | 0.4167 | 0.3685 |
| fund_pe | bull | 0.0808 | 0.1779 | 0.4544 | 0.4394 | 0.3270 |
| rsi_vol_combo | bull | 0.0647 | 0.1605 | 0.4033 | 0.3409 | 0.2704 |
| trend_lowvol | bull | 0.0705 | 0.1839 | 0.3832 | 0.2576 | 0.2409 |
| fund_profit_growth | bull | 0.0340 | 0.1207 | 0.2817 | 0.2121 | 0.1707 |
| fund_revenue_growth | bull | 0.0224 | 0.1250 | 0.1791 | 0.1439 | 0.1024 |
| fund_gross_margin | bull | 0.0152 | 0.1147 | 0.1329 | 0.1439 | 0.0760 |
| rsi_vol_combo | bear | 0.1626 | 0.1826 | 0.8905 | 0.6986 | 0.7563 |
| momentum_reversal | bear | 0.1797 | 0.2185 | 0.8225 | 0.5890 | 0.6535 |
| mom_x_lowvol_20_20 | bear | 0.1676 | 0.2255 | 0.7431 | 0.5616 | 0.5802 |
| fund_revenue_growth | bear | 0.0913 | 0.1318 | 0.6925 | 0.4795 | 0.5122 |
| trend_lowvol | bear | 0.1302 | 0.2109 | 0.6172 | 0.3973 | 0.4312 |
| fund_pb | bear | 0.0956 | 0.1675 | 0.5706 | 0.4247 | 0.4065 |
| fund_score | bear | 0.0909 | 0.1761 | 0.5161 | 0.3973 | 0.3606 |
| fund_profit_growth | bear | 0.0732 | 0.1371 | 0.5340 | 0.3425 | 0.3584 |
| fund_pe | bear | 0.0776 | 0.1858 | 0.4174 | 0.3973 | 0.2916 |
| bb_width_20 | bear | 0.0822 | 0.2017 | 0.4074 | 0.3699 | 0.2790 |
| fund_roe | bear | 0.0609 | 0.1949 | 0.3125 | 0.2055 | 0.1884 |

### 中字头

- **Neutral**: ['trend_lowvol'] (单因子IC=0.0854, 组合IC=0.0854)
  - weights: [1.0]
- **Bull**: ['low_downside', 'volatility', 'turnover_stability'] (单因子IC=0.0911, 组合IC=0.1194)
  - bull_weights: [0.4536, 0.3027, 0.2437]
- **Bear**: ['bb_width_20', 'mom_x_lowvol_20_20', 'momentum_reversal'] (单因子IC=0.1365, 组合IC=0.1582)
  - bear_weights: [0.3731, 0.3152, 0.3116]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| trend_lowvol | neutral | 0.0854 | 0.2306 | 0.3706 | 0.2944 | 0.2399 |
| momentum_reversal | neutral | 0.0695 | 0.2076 | 0.3350 | 0.2484 | 0.2091 |
| mom_x_lowvol_20_20 | neutral | 0.0629 | 0.1965 | 0.3201 | 0.2380 | 0.1982 |
| turnover_stability | neutral | 0.0443 | 0.1475 | 0.3006 | 0.1879 | 0.1785 |
| fund_pb | neutral | 0.0650 | 0.2607 | 0.2494 | 0.1837 | 0.1476 |
| rsi_vol_combo | neutral | 0.0470 | 0.2001 | 0.2348 | 0.1566 | 0.1358 |
| fund_pe | neutral | 0.0604 | 0.2564 | 0.2354 | 0.1524 | 0.1357 |
| volatility | neutral | 0.0400 | 0.1884 | 0.2125 | 0.2025 | 0.1277 |
| fund_score | neutral | 0.0286 | 0.1649 | 0.1736 | 0.1587 | 0.1006 |
| low_downside | neutral | 0.0300 | 0.2081 | 0.1440 | 0.1670 | 0.0840 |
| fund_roe | neutral | 0.0239 | 0.1722 | 0.1389 | 0.1002 | 0.0764 |
| fund_profit_growth | neutral | 0.0186 | 0.1461 | 0.1274 | 0.1587 | 0.0738 |
| fund_revenue_growth | neutral | 0.0140 | 0.1401 | 0.0996 | 0.1106 | 0.0553 |
| low_downside | bull | 0.1127 | 0.1645 | 0.6853 | 0.5152 | 0.5191 |
| volatility | bull | 0.0972 | 0.1923 | 0.5053 | 0.3712 | 0.3465 |
| turnover_stability | bull | 0.0633 | 0.1479 | 0.4282 | 0.3030 | 0.2790 |
| trend_lowvol | bull | 0.0841 | 0.2145 | 0.3924 | 0.3030 | 0.2556 |
| momentum_reversal | bull | 0.0686 | 0.1958 | 0.3501 | 0.2879 | 0.2255 |
| mom_x_lowvol_20_20 | bull | 0.0582 | 0.1793 | 0.3246 | 0.2727 | 0.2066 |
| fund_pb | bull | 0.0697 | 0.2182 | 0.3195 | 0.2879 | 0.2057 |
| fund_pe | bull | 0.0670 | 0.2320 | 0.2889 | 0.1667 | 0.1685 |
| rsi_vol_combo | bull | 0.0294 | 0.1874 | 0.1570 | 0.1591 | 0.0910 |
| bb_width_20 | bear | 0.1359 | 0.2278 | 0.5964 | 0.4795 | 0.4412 |
| mom_x_lowvol_20_20 | bear | 0.1290 | 0.2417 | 0.5335 | 0.3973 | 0.3727 |
| momentum_reversal | bear | 0.1447 | 0.2743 | 0.5273 | 0.3973 | 0.3684 |
| rsi_vol_combo | bear | 0.1059 | 0.2256 | 0.4694 | 0.3151 | 0.3087 |
| trend_lowvol | bear | 0.1029 | 0.2790 | 0.3688 | 0.3425 | 0.2476 |
| fund_profit_growth | bear | 0.0501 | 0.1554 | 0.3224 | 0.2877 | 0.2076 |

### 中特估

- **Neutral**: ['fund_roe', 'fund_pe'] (单因子IC=0.0649, 组合IC=0.0709)
  - weights: [0.5084, 0.4916]
- **Bull**: ['low_downside', 'turnover_stability', 'exhaustion_risk'] (单因子IC=0.054, 组合IC=0.0879)
  - bull_weights: [0.4139, 0.3535, 0.2326]
- **Bear**: ['bb_width_20', 'fund_profit_growth', 'mom_x_lowvol_20_20'] (单因子IC=0.0976, 组合IC=0.1205)
  - bear_weights: [0.3628, 0.352, 0.2852]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| fund_roe | neutral | 0.0560 | 0.2167 | 0.2584 | 0.2171 | 0.1572 |
| fund_pe | neutral | 0.0737 | 0.2906 | 0.2538 | 0.1983 | 0.1520 |
| fund_score | neutral | 0.0417 | 0.1726 | 0.2414 | 0.2025 | 0.1451 |
| trend_lowvol | neutral | 0.0585 | 0.2640 | 0.2215 | 0.1524 | 0.1276 |
| volatility | neutral | 0.0329 | 0.1520 | 0.2165 | 0.1566 | 0.1252 |
| turnover_stability | neutral | 0.0327 | 0.1565 | 0.2091 | 0.1691 | 0.1223 |
| fund_gross_margin | neutral | 0.0406 | 0.1959 | 0.2070 | 0.1670 | 0.1208 |
| fund_pb | neutral | 0.0571 | 0.2801 | 0.2039 | 0.1587 | 0.1181 |
| fund_profit_growth | neutral | 0.0286 | 0.1572 | 0.1821 | 0.1482 | 0.1045 |
| mom_x_lowvol_20_20 | neutral | 0.0353 | 0.2014 | 0.1751 | 0.1357 | 0.0994 |
| fund_revenue_growth | neutral | 0.0232 | 0.1549 | 0.1500 | 0.1670 | 0.0875 |
| top_fractal_volume | neutral | 0.0189 | 0.1342 | 0.1407 | 0.1248 | 0.0791 |
| low_downside | neutral | 0.0275 | 0.2449 | 0.1124 | 0.1044 | 0.0621 |
| low_downside | bull | 0.0801 | 0.2369 | 0.3381 | 0.2424 | 0.2100 |
| turnover_stability | bull | 0.0527 | 0.1690 | 0.3116 | 0.1515 | 0.1794 |
| exhaustion_risk | bull | 0.0292 | 0.1539 | 0.1897 | 0.2443 | 0.1180 |
| fund_score | bull | 0.0337 | 0.1680 | 0.2007 | 0.1742 | 0.1178 |
| relative_strength | bull | 0.0406 | 0.2458 | 0.1652 | 0.2803 | 0.1057 |
| fund_pe | bull | 0.0510 | 0.3144 | 0.1621 | 0.1212 | 0.0909 |
| stroke_phase | bull | 0.0242 | 0.1696 | 0.1427 | 0.1364 | 0.0811 |
| volatility | bull | 0.0249 | 0.1871 | 0.1333 | 0.1439 | 0.0763 |
| fund_revenue_growth | bull | 0.0194 | 0.1442 | 0.1343 | 0.1212 | 0.0753 |
| bb_width_20 | bear | 0.1126 | 0.2217 | 0.5080 | 0.3699 | 0.3479 |
| fund_profit_growth | bear | 0.0973 | 0.1933 | 0.5030 | 0.3425 | 0.3376 |
| mom_x_lowvol_20_20 | bear | 0.0829 | 0.2075 | 0.3995 | 0.3699 | 0.2736 |
| momentum_reversal | bear | 0.1070 | 0.2856 | 0.3748 | 0.2603 | 0.2362 |
| trend_lowvol | bear | 0.0877 | 0.2946 | 0.2978 | 0.3425 | 0.1999 |
| limit_pullback_score | bear | 0.0435 | 0.1281 | 0.3398 | 0.1250 | 0.1911 |
| rsi_vol_combo | bear | 0.0764 | 0.2542 | 0.3008 | 0.2603 | 0.1895 |
| wash_sale_score | bear | 0.0448 | 0.1440 | 0.3112 | 0.1481 | 0.1787 |
| fund_score | bear | 0.0652 | 0.2074 | 0.3142 | 0.1233 | 0.1765 |
| fund_revenue_growth | bear | 0.0303 | 0.1437 | 0.2111 | 0.2603 | 0.1330 |

### 中盘价值

- **Neutral**: ['fund_pe', 'fund_pb'] (单因子IC=0.0859, 组合IC=0.0976)
  - weights: [0.5213, 0.4787]
- **Bull**: ['fund_pb', 'fund_pe'] (单因子IC=0.0835, 组合IC=0.0955)
  - bull_weights: [0.5906, 0.4094]
- **Bear**: ['turnover_stability', 'fund_revenue_growth', 'fund_profit_growth'] (单因子IC=0.0669, 组合IC=0.0876)
  - bear_weights: [0.4051, 0.3099, 0.2851]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| fund_pe | neutral | 0.0886 | 0.2300 | 0.3852 | 0.3278 | 0.2557 |
| fund_pb | neutral | 0.0833 | 0.2259 | 0.3688 | 0.2735 | 0.2348 |
| volatility | neutral | 0.0331 | 0.1338 | 0.2472 | 0.1816 | 0.1461 |
| fund_profit_growth | neutral | 0.0367 | 0.1658 | 0.2213 | 0.2109 | 0.1340 |
| trend_lowvol | neutral | 0.0336 | 0.2317 | 0.1450 | 0.1691 | 0.0847 |
| fund_score | neutral | 0.0233 | 0.1635 | 0.1426 | 0.1461 | 0.0817 |
| fund_revenue_growth | neutral | 0.0173 | 0.1278 | 0.1352 | 0.1628 | 0.0786 |
| low_downside | neutral | 0.0243 | 0.1913 | 0.1270 | 0.1023 | 0.0700 |
| fund_roe | neutral | 0.0107 | 0.1739 | 0.0616 | 0.1148 | 0.0343 |
| fund_pb | bull | 0.0881 | 0.1730 | 0.5091 | 0.3864 | 0.3529 |
| fund_pe | bull | 0.0790 | 0.2055 | 0.3844 | 0.2727 | 0.2446 |
| volatility | bull | 0.0152 | 0.1264 | 0.1205 | 0.1136 | 0.0671 |
| top_fractal_volume | bull | 0.0107 | 0.1081 | 0.0993 | 0.1225 | 0.0558 |
| turnover_stability | bear | 0.0566 | 0.0839 | 0.6743 | 0.5068 | 0.5080 |
| fund_revenue_growth | bear | 0.0735 | 0.1348 | 0.5456 | 0.4247 | 0.3886 |
| fund_profit_growth | bear | 0.0706 | 0.1433 | 0.4925 | 0.4521 | 0.3576 |
| momentum_reversal | bear | 0.0952 | 0.2176 | 0.4375 | 0.3699 | 0.2996 |
| fund_score | bear | 0.0621 | 0.1451 | 0.4278 | 0.3425 | 0.2872 |
| rsi_vol_combo | bear | 0.0706 | 0.2048 | 0.3447 | 0.3699 | 0.2361 |
| trend_lowvol | bear | 0.0889 | 0.2464 | 0.3608 | 0.2329 | 0.2224 |
| bb_width_20 | bear | 0.0654 | 0.1850 | 0.3536 | 0.2329 | 0.2180 |
| mom_x_lowvol_20_20 | bear | 0.0376 | 0.1393 | 0.2702 | 0.2603 | 0.1703 |
| top_fractal_volume | bear | 0.0301 | 0.1081 | 0.2786 | 0.1579 | 0.1613 |

### 中盘成长

- **Neutral**: ['fund_pe', 'trend_lowvol', 'fund_profit_growth'] (单因子IC=0.0516, 组合IC=0.0661)
  - weights: [0.3696, 0.3242, 0.3062]
- **Bull**: ['turnover_stability', 'fund_pe', 'fund_pb'] (单因子IC=0.0501, 组合IC=0.0642)
  - bull_weights: [0.3809, 0.3441, 0.275]
- **Bear**: ['mom_x_lowvol_20_20'] (单因子IC=0.1329, 组合IC=0.1329)
  - bear_weights: [1.0]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| fund_pe | neutral | 0.0507 | 0.1454 | 0.3487 | 0.2630 | 0.2202 |
| trend_lowvol | neutral | 0.0606 | 0.1904 | 0.3185 | 0.2129 | 0.1932 |
| fund_profit_growth | neutral | 0.0435 | 0.1480 | 0.2937 | 0.2422 | 0.1824 |
| momentum_reversal | neutral | 0.0514 | 0.1704 | 0.3015 | 0.2004 | 0.1810 |
| mom_x_lowvol_20_20 | neutral | 0.0487 | 0.1651 | 0.2949 | 0.2192 | 0.1798 |
| fund_pb | neutral | 0.0504 | 0.1944 | 0.2591 | 0.1086 | 0.1436 |
| fund_score | neutral | 0.0396 | 0.1762 | 0.2247 | 0.1858 | 0.1332 |
| fund_revenue_growth | neutral | 0.0320 | 0.1455 | 0.2198 | 0.1837 | 0.1301 |
| rsi_vol_combo | neutral | 0.0364 | 0.1625 | 0.2237 | 0.1461 | 0.1282 |
| fund_roe | neutral | 0.0318 | 0.1602 | 0.1987 | 0.1378 | 0.1130 |
| turnover_stability | neutral | 0.0164 | 0.1257 | 0.1303 | 0.1106 | 0.0724 |
| turnover_stability | bull | 0.0400 | 0.1005 | 0.3980 | 0.4015 | 0.2789 |
| fund_pe | bull | 0.0544 | 0.1407 | 0.3867 | 0.3030 | 0.2519 |
| fund_pb | bull | 0.0558 | 0.1671 | 0.3343 | 0.2045 | 0.2013 |
| volatility | bull | 0.0492 | 0.1507 | 0.3266 | 0.1742 | 0.1917 |
| fund_profit_growth | bull | 0.0315 | 0.1215 | 0.2595 | 0.2424 | 0.1612 |
| fund_revenue_growth | bull | 0.0294 | 0.1199 | 0.2451 | 0.1970 | 0.1467 |
| low_downside | bull | 0.0364 | 0.1493 | 0.2440 | 0.1894 | 0.1451 |
| mom_x_lowvol_20_20 | bear | 0.1329 | 0.2614 | 0.5083 | 0.3973 | 0.3551 |
| momentum_reversal | bear | 0.1208 | 0.2556 | 0.4727 | 0.3699 | 0.3238 |
| fund_gross_margin | bear | 0.0569 | 0.1697 | 0.3353 | 0.2877 | 0.2159 |
| bb_width_20 | bear | 0.0865 | 0.2660 | 0.3251 | 0.3151 | 0.2138 |
| rsi_vol_combo | bear | 0.0688 | 0.1996 | 0.3450 | 0.2329 | 0.2127 |
| fund_profit_growth | bear | 0.0476 | 0.1777 | 0.2679 | 0.3425 | 0.1798 |
| trend_lowvol | bear | 0.0890 | 0.2916 | 0.3050 | 0.1781 | 0.1797 |
| fund_revenue_growth | bear | 0.0561 | 0.2004 | 0.2802 | 0.1233 | 0.1574 |
| fund_pe | bear | 0.0291 | 0.1404 | 0.2076 | 0.1507 | 0.1194 |

### 中芯概念

- **Neutral**: ['mom_x_lowvol_20_20'] (单因子IC=0.0842, 组合IC=0.0842)
  - weights: [1.0]
- **Bull**: ['trend_lowvol', 'low_downside', 'momentum_reversal'] (单因子IC=0.0607, 组合IC=0.0788)
  - bull_weights: [0.3565, 0.3246, 0.319]
- **Bear**: ['mom_x_lowvol_20_20'] (单因子IC=0.1358, 组合IC=0.1358)
  - bear_weights: [1.0]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| mom_x_lowvol_20_20 | neutral | 0.0842 | 0.2119 | 0.3971 | 0.3027 | 0.2587 |
| momentum_reversal | neutral | 0.0808 | 0.2049 | 0.3942 | 0.2526 | 0.2469 |
| rsi_vol_combo | neutral | 0.0636 | 0.1998 | 0.3183 | 0.2338 | 0.1964 |
| fund_pe | neutral | 0.0574 | 0.1994 | 0.2878 | 0.2067 | 0.1736 |
| trend_lowvol | neutral | 0.0603 | 0.2309 | 0.2610 | 0.2161 | 0.1587 |
| fund_pb | neutral | 0.0582 | 0.2261 | 0.2575 | 0.1587 | 0.1492 |
| fund_profit_growth | neutral | 0.0358 | 0.2054 | 0.1741 | 0.1670 | 0.1016 |
| fund_score | neutral | 0.0343 | 0.2214 | 0.1549 | 0.2004 | 0.0930 |
| fund_revenue_growth | neutral | 0.0269 | 0.1956 | 0.1374 | 0.1889 | 0.0817 |
| volatility | neutral | 0.0347 | 0.2390 | 0.1450 | 0.1253 | 0.0816 |
| fund_roe | neutral | 0.0238 | 0.1814 | 0.1312 | 0.1545 | 0.0757 |
| trend_lowvol | bull | 0.0700 | 0.2100 | 0.3331 | 0.2652 | 0.2107 |
| low_downside | bull | 0.0590 | 0.1957 | 0.3015 | 0.2727 | 0.1918 |
| momentum_reversal | bull | 0.0530 | 0.1779 | 0.2980 | 0.2652 | 0.1885 |
| fund_pe | bull | 0.0558 | 0.1914 | 0.2915 | 0.2803 | 0.1866 |
| rsi_vol_combo | bull | 0.0501 | 0.1757 | 0.2852 | 0.2500 | 0.1782 |
| fund_revenue_growth | bull | 0.0514 | 0.1967 | 0.2611 | 0.2045 | 0.1572 |
| fund_pb | bull | 0.0524 | 0.2141 | 0.2448 | 0.2576 | 0.1539 |
| volatility | bull | 0.0468 | 0.2169 | 0.2157 | 0.2121 | 0.1307 |
| turnover_stability | bull | 0.0344 | 0.1588 | 0.2168 | 0.1818 | 0.1281 |
| fund_profit_growth | bull | 0.0394 | 0.1789 | 0.2200 | 0.1212 | 0.1233 |
| fund_gross_margin | bull | 0.0345 | 0.1731 | 0.1993 | 0.1818 | 0.1178 |
| fund_roe | bull | 0.0260 | 0.1617 | 0.1607 | 0.1136 | 0.0895 |
| mom_x_lowvol_20_20 | bear | 0.1358 | 0.2656 | 0.5114 | 0.4521 | 0.3713 |
| rsi_vol_combo | bear | 0.0949 | 0.1930 | 0.4919 | 0.3699 | 0.3369 |
| momentum_reversal | bear | 0.1082 | 0.2292 | 0.4719 | 0.3699 | 0.3232 |
| fund_pe | bear | 0.0738 | 0.2098 | 0.3519 | 0.2877 | 0.2266 |
| fund_revenue_growth | bear | 0.0584 | 0.2021 | 0.2892 | 0.2329 | 0.1782 |
| fund_profit_growth | bear | 0.0556 | 0.2134 | 0.2608 | 0.1781 | 0.1536 |
| bb_width_20 | bear | 0.0487 | 0.2075 | 0.2347 | 0.2055 | 0.1415 |
| fund_roe | bear | 0.0366 | 0.1585 | 0.2308 | 0.2055 | 0.1391 |

### 中药概念

- **Neutral**: ['fund_pb', 'volatility', 'trend_lowvol'] (单因子IC=0.0718, 组合IC=0.0937)
  - weights: [0.4359, 0.2848, 0.2792]
- **Bull**: ['low_downside', 'fund_pb', 'turnover_stability'] (单因子IC=0.0426, 组合IC=0.0675)
  - bull_weights: [0.3764, 0.3196, 0.304]
- **Bear**: ['trend_lowvol', 'mom_x_lowvol_20_20'] (单因子IC=0.1162, 组合IC=0.1406)
  - bear_weights: [0.532, 0.468]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| fund_pb | neutral | 0.0716 | 0.1422 | 0.5037 | 0.3466 | 0.3391 |
| volatility | neutral | 0.0691 | 0.2005 | 0.3446 | 0.2860 | 0.2216 |
| trend_lowvol | neutral | 0.0748 | 0.2247 | 0.3330 | 0.3048 | 0.2173 |
| momentum_reversal | neutral | 0.0650 | 0.1926 | 0.3373 | 0.2610 | 0.2127 |
| turnover_stability | neutral | 0.0400 | 0.1224 | 0.3272 | 0.2735 | 0.2083 |
| fund_pe | neutral | 0.0605 | 0.1947 | 0.3107 | 0.2797 | 0.1988 |
| low_downside | neutral | 0.0594 | 0.2008 | 0.2957 | 0.2651 | 0.1870 |
| mom_x_lowvol_20_20 | neutral | 0.0560 | 0.1805 | 0.3100 | 0.2004 | 0.1860 |
| rsi_vol_combo | neutral | 0.0496 | 0.1795 | 0.2763 | 0.2150 | 0.1678 |
| fund_profit_growth | neutral | 0.0338 | 0.1383 | 0.2443 | 0.2234 | 0.1494 |
| fund_revenue_growth | neutral | 0.0264 | 0.1196 | 0.2206 | 0.1816 | 0.1304 |
| fund_score | neutral | 0.0311 | 0.1831 | 0.1700 | 0.1461 | 0.0974 |
| low_downside | bull | 0.0566 | 0.1637 | 0.3459 | 0.3409 | 0.2319 |
| fund_pb | bull | 0.0396 | 0.1235 | 0.3208 | 0.2273 | 0.1969 |
| turnover_stability | bull | 0.0316 | 0.1024 | 0.3091 | 0.2121 | 0.1873 |
| rsi_vol_combo | bull | 0.0400 | 0.1580 | 0.2534 | 0.1742 | 0.1488 |
| fund_revenue_growth | bull | 0.0158 | 0.1061 | 0.1486 | 0.1818 | 0.0878 |
| trend_lowvol | bear | 0.1200 | 0.1882 | 0.6376 | 0.4795 | 0.4716 |
| mom_x_lowvol_20_20 | bear | 0.1124 | 0.2079 | 0.5409 | 0.5342 | 0.4149 |
| momentum_reversal | bear | 0.1261 | 0.2363 | 0.5338 | 0.3699 | 0.3656 |
| fund_revenue_growth | bear | 0.0666 | 0.1278 | 0.5208 | 0.2603 | 0.3282 |
| fund_pe | bear | 0.0789 | 0.1876 | 0.4203 | 0.5068 | 0.3166 |
| turnover_stability | bear | 0.0403 | 0.0921 | 0.4374 | 0.3151 | 0.2876 |
| fund_pb | bear | 0.0441 | 0.1353 | 0.3257 | 0.3973 | 0.2276 |
| rsi_vol_combo | bear | 0.0802 | 0.2439 | 0.3291 | 0.2603 | 0.2074 |
| fund_roe | bear | 0.0681 | 0.2283 | 0.2982 | 0.1507 | 0.1716 |
| fund_score | bear | 0.0554 | 0.1957 | 0.2829 | 0.1781 | 0.1667 |
| fund_profit_growth | bear | 0.0363 | 0.1440 | 0.2521 | 0.1507 | 0.1450 |

### 乡村振兴

- **Neutral**: ['momentum_reversal', 'turnover_stability', 'mom_x_lowvol_20_20'] (单因子IC=0.0834, 组合IC=0.1097)
  - weights: [0.3434, 0.3316, 0.3251]
- **Bull**: ['fund_pb', 'low_downside', 'trend_lowvol'] (单因子IC=0.0997, 组合IC=0.1296)
  - bull_weights: [0.3849, 0.3399, 0.2753]
- **Bear**: ['trend_lowvol', 'momentum_reversal'] (单因子IC=0.1208, 组合IC=0.1328)
  - bear_weights: [0.5205, 0.4795]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| momentum_reversal | neutral | 0.0980 | 0.1753 | 0.5589 | 0.4363 | 0.4014 |
| turnover_stability | neutral | 0.0598 | 0.1086 | 0.5509 | 0.4071 | 0.3876 |
| mom_x_lowvol_20_20 | neutral | 0.0923 | 0.1726 | 0.5346 | 0.4217 | 0.3800 |
| trend_lowvol | neutral | 0.1012 | 0.1915 | 0.5283 | 0.4238 | 0.3761 |
| fund_pb | neutral | 0.0685 | 0.1529 | 0.4481 | 0.3278 | 0.2975 |
| rsi_vol_combo | neutral | 0.0742 | 0.1713 | 0.4334 | 0.3528 | 0.2932 |
| volatility | neutral | 0.0755 | 0.1962 | 0.3847 | 0.3528 | 0.2602 |
| fund_profit_growth | neutral | 0.0324 | 0.1179 | 0.2747 | 0.2484 | 0.1715 |
| fund_score | neutral | 0.0368 | 0.1446 | 0.2546 | 0.1900 | 0.1515 |
| fund_pe | neutral | 0.0441 | 0.1797 | 0.2455 | 0.1733 | 0.1440 |
| low_downside | neutral | 0.0436 | 0.1896 | 0.2298 | 0.2234 | 0.1406 |
| fund_revenue_growth | neutral | 0.0154 | 0.1094 | 0.1409 | 0.1336 | 0.0799 |
| fund_pb | bull | 0.1061 | 0.1311 | 0.8088 | 0.5985 | 0.6464 |
| low_downside | bull | 0.0988 | 0.1312 | 0.7535 | 0.5152 | 0.5709 |
| trend_lowvol | bull | 0.0943 | 0.1552 | 0.6072 | 0.5227 | 0.4623 |
| volatility | bull | 0.1005 | 0.1649 | 0.6096 | 0.3409 | 0.4087 |
| mom_x_lowvol_20_20 | bull | 0.0807 | 0.1534 | 0.5261 | 0.4091 | 0.3707 |
| momentum_reversal | bull | 0.0770 | 0.1531 | 0.5030 | 0.3864 | 0.3487 |
| turnover_stability | bull | 0.0473 | 0.1237 | 0.3824 | 0.3409 | 0.2564 |
| fund_pe | bull | 0.0654 | 0.1619 | 0.4042 | 0.2045 | 0.2435 |
| rsi_vol_combo | bull | 0.0508 | 0.1389 | 0.3657 | 0.1894 | 0.2175 |
| stroke_phase | bull | 0.0231 | 0.1134 | 0.2037 | 0.2273 | 0.1250 |
| trend_lowvol | bear | 0.1098 | 0.1652 | 0.6645 | 0.3699 | 0.4551 |
| momentum_reversal | bear | 0.1319 | 0.2111 | 0.6247 | 0.3425 | 0.4193 |
| fund_profit_growth | bear | 0.0656 | 0.1126 | 0.5827 | 0.3699 | 0.3991 |
| mom_x_lowvol_20_20 | bear | 0.1220 | 0.2130 | 0.5726 | 0.3425 | 0.3843 |
| fund_score | bear | 0.0716 | 0.1472 | 0.4865 | 0.3973 | 0.3399 |
| rsi_vol_combo | bear | 0.0865 | 0.1763 | 0.4904 | 0.2603 | 0.3090 |
| turnover_stability | bear | 0.0457 | 0.1084 | 0.4220 | 0.3425 | 0.2833 |
| bb_width_20 | bear | 0.0610 | 0.2287 | 0.2668 | 0.2603 | 0.1681 |
| fund_gross_margin | bear | 0.0392 | 0.1611 | 0.2434 | 0.1507 | 0.1401 |

### 乳业

- **Neutral**: ['fund_pb', 'mom_x_lowvol_20_20', 'trend_lowvol'] (单因子IC=0.0974, 组合IC=0.1377)
  - weights: [0.409, 0.299, 0.2921]
- **Bull**: ['fund_pb', 'fund_pe', 'volatility'] (单因子IC=0.0894, 组合IC=0.1244)
  - bull_weights: [0.4189, 0.3112, 0.27]
- **Bear**: ['momentum_reversal', 'rsi_vol_combo'] (单因子IC=0.1395, 组合IC=0.1568)
  - bear_weights: [0.5141, 0.4859]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| fund_pb | neutral | 0.0962 | 0.2036 | 0.4723 | 0.3392 | 0.3163 |
| mom_x_lowvol_20_20 | neutral | 0.0954 | 0.2701 | 0.3532 | 0.3090 | 0.2312 |
| trend_lowvol | neutral | 0.1007 | 0.2800 | 0.3598 | 0.2557 | 0.2259 |
| momentum_reversal | neutral | 0.0917 | 0.2662 | 0.3444 | 0.2568 | 0.2164 |
| volatility | neutral | 0.0861 | 0.2883 | 0.2987 | 0.2484 | 0.1864 |
| fund_pe | neutral | 0.0717 | 0.2433 | 0.2946 | 0.2150 | 0.1790 |
| rsi_vol_combo | neutral | 0.0611 | 0.2476 | 0.2469 | 0.1743 | 0.1450 |
| low_downside | neutral | 0.0512 | 0.2569 | 0.1993 | 0.1816 | 0.1177 |
| turnover_stability | neutral | 0.0294 | 0.2152 | 0.1365 | 0.1065 | 0.0755 |
| fund_profit_growth | neutral | 0.0271 | 0.2170 | 0.1250 | 0.1013 | 0.0689 |
| fund_pb | bull | 0.0956 | 0.2069 | 0.4621 | 0.3712 | 0.3168 |
| fund_pe | bull | 0.0785 | 0.2123 | 0.3699 | 0.2727 | 0.2354 |
| volatility | bull | 0.0940 | 0.2730 | 0.3445 | 0.1856 | 0.2042 |
| turnover_stability | bull | 0.0638 | 0.2069 | 0.3082 | 0.2121 | 0.1868 |
| mom_x_lowvol_20_20 | bull | 0.0614 | 0.2567 | 0.2391 | 0.1894 | 0.1422 |
| momentum_reversal | bull | 0.0546 | 0.2552 | 0.2141 | 0.1894 | 0.1273 |
| low_downside | bull | 0.0552 | 0.2670 | 0.2067 | 0.1515 | 0.1190 |
| momentum_reversal | bear | 0.1518 | 0.3296 | 0.4605 | 0.3699 | 0.3154 |
| rsi_vol_combo | bear | 0.1273 | 0.2925 | 0.4353 | 0.3699 | 0.2981 |
| mom_x_lowvol_20_20 | bear | 0.1438 | 0.3464 | 0.4152 | 0.3973 | 0.2901 |
| turnover_stability | bear | 0.0972 | 0.2453 | 0.3962 | 0.2329 | 0.2442 |
| fund_pb | bear | 0.0640 | 0.1812 | 0.3532 | 0.3151 | 0.2323 |
| bb_width_20 | bear | 0.0806 | 0.3239 | 0.2489 | 0.2329 | 0.1534 |
| fund_revenue_growth | bear | 0.0395 | 0.2073 | 0.1904 | 0.1507 | 0.1096 |

### 云计算

- **Neutral**: ['fund_pb', 'momentum_reversal', 'trend_lowvol'] (单因子IC=0.0869, 组合IC=0.114)
  - weights: [0.3713, 0.3274, 0.3013]
- **Bull**: ['volatility', 'low_downside', 'fund_pb'] (单因子IC=0.1012, 组合IC=0.1178)
  - bull_weights: [0.3618, 0.3228, 0.3154]
- **Bear**: ['mom_x_lowvol_20_20', 'momentum_reversal', 'trend_lowvol'] (单因子IC=0.1351, 组合IC=0.1483)
  - bear_weights: [0.3698, 0.3364, 0.2938]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| fund_pb | neutral | 0.0795 | 0.1376 | 0.5774 | 0.4092 | 0.4068 |
| momentum_reversal | neutral | 0.0864 | 0.1662 | 0.5198 | 0.3800 | 0.3587 |
| trend_lowvol | neutral | 0.0949 | 0.1969 | 0.4821 | 0.3695 | 0.3301 |
| mom_x_lowvol_20_20 | neutral | 0.0796 | 0.1672 | 0.4758 | 0.3111 | 0.3119 |
| volatility | neutral | 0.0754 | 0.1925 | 0.3915 | 0.3278 | 0.2599 |
| fund_profit_growth | neutral | 0.0528 | 0.1345 | 0.3926 | 0.2985 | 0.2549 |
| rsi_vol_combo | neutral | 0.0595 | 0.1602 | 0.3716 | 0.2505 | 0.2323 |
| fund_score | neutral | 0.0595 | 0.1712 | 0.3475 | 0.2610 | 0.2191 |
| fund_pe | neutral | 0.0547 | 0.1682 | 0.3250 | 0.2651 | 0.2056 |
| turnover_stability | neutral | 0.0302 | 0.1136 | 0.2658 | 0.2484 | 0.1659 |
| fund_roe | neutral | 0.0493 | 0.1824 | 0.2706 | 0.1962 | 0.1618 |
| low_downside | neutral | 0.0488 | 0.1898 | 0.2569 | 0.2192 | 0.1566 |
| fund_revenue_growth | neutral | 0.0257 | 0.1082 | 0.2377 | 0.1879 | 0.1412 |
| volatility | bull | 0.1098 | 0.1614 | 0.6804 | 0.5076 | 0.5129 |
| low_downside | bull | 0.1005 | 0.1655 | 0.6070 | 0.5076 | 0.4576 |
| fund_pb | bull | 0.0934 | 0.1534 | 0.6084 | 0.4697 | 0.4471 |
| trend_lowvol | bull | 0.1022 | 0.1869 | 0.5466 | 0.5303 | 0.4182 |
| fund_pe | bull | 0.0829 | 0.1731 | 0.4788 | 0.3182 | 0.3156 |
| mom_x_lowvol_20_20 | bull | 0.0602 | 0.1540 | 0.3907 | 0.2955 | 0.2531 |
| turnover_stability | bull | 0.0407 | 0.1062 | 0.3828 | 0.3030 | 0.2494 |
| momentum_reversal | bull | 0.0558 | 0.1586 | 0.3519 | 0.2652 | 0.2226 |
| fund_score | bull | 0.0496 | 0.1624 | 0.3052 | 0.2652 | 0.1930 |
| fund_profit_growth | bull | 0.0394 | 0.1250 | 0.3153 | 0.1970 | 0.1887 |
| fund_roe | bull | 0.0449 | 0.1662 | 0.2703 | 0.2121 | 0.1638 |
| fund_revenue_growth | bull | 0.0143 | 0.1132 | 0.1262 | 0.1136 | 0.0703 |
| mom_x_lowvol_20_20 | bear | 0.1459 | 0.1820 | 0.8018 | 0.6712 | 0.6700 |
| momentum_reversal | bear | 0.1448 | 0.1919 | 0.7542 | 0.6164 | 0.6095 |
| trend_lowvol | bear | 0.1146 | 0.1680 | 0.6819 | 0.5616 | 0.5324 |
| fund_profit_growth | bear | 0.0817 | 0.1344 | 0.6077 | 0.5342 | 0.4661 |
| rsi_vol_combo | bear | 0.0861 | 0.1662 | 0.5180 | 0.4521 | 0.3761 |
| fund_pb | bear | 0.0690 | 0.1681 | 0.4105 | 0.2603 | 0.2586 |

### 互联医疗

- **Neutral**: ['fund_profit_growth', 'trend_lowvol', 'fund_pb'] (单因子IC=0.0708, 组合IC=0.0954)
  - weights: [0.3378, 0.3349, 0.3273]
- **Bull**: ['low_downside', 'turnover_stability'] (单因子IC=0.0921, 组合IC=0.1249)
  - bull_weights: [0.5425, 0.4575]
- **Bear**: ['trend_lowvol', 'momentum_reversal', 'mom_x_lowvol_20_20'] (单因子IC=0.1285, 组合IC=0.1525)
  - bear_weights: [0.4909, 0.2579, 0.2512]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| fund_profit_growth | neutral | 0.0448 | 0.1135 | 0.3948 | 0.3173 | 0.2601 |
| trend_lowvol | neutral | 0.0876 | 0.2245 | 0.3901 | 0.3215 | 0.2578 |
| fund_pb | neutral | 0.0800 | 0.2052 | 0.3899 | 0.2923 | 0.2519 |
| mom_x_lowvol_20_20 | neutral | 0.0667 | 0.1958 | 0.3405 | 0.2443 | 0.2118 |
| momentum_reversal | neutral | 0.0703 | 0.2068 | 0.3397 | 0.2443 | 0.2113 |
| turnover_stability | neutral | 0.0408 | 0.1346 | 0.3029 | 0.2672 | 0.1919 |
| low_downside | neutral | 0.0701 | 0.2453 | 0.2859 | 0.2693 | 0.1815 |
| volatility | neutral | 0.0682 | 0.2378 | 0.2870 | 0.2610 | 0.1809 |
| fund_score | neutral | 0.0392 | 0.1809 | 0.2167 | 0.1691 | 0.1267 |
| fund_pe | neutral | 0.0537 | 0.2572 | 0.2088 | 0.1524 | 0.1203 |
| rsi_vol_combo | neutral | 0.0410 | 0.1976 | 0.2077 | 0.1503 | 0.1194 |
| low_downside | bull | 0.1230 | 0.1926 | 0.6388 | 0.4167 | 0.4525 |
| turnover_stability | bull | 0.0611 | 0.1134 | 0.5388 | 0.4167 | 0.3816 |
| volatility | bull | 0.0946 | 0.2069 | 0.4572 | 0.3636 | 0.3117 |
| fund_profit_growth | bull | 0.0447 | 0.1167 | 0.3833 | 0.3258 | 0.2541 |
| trend_lowvol | bull | 0.0752 | 0.2412 | 0.3117 | 0.2197 | 0.1901 |
| fund_pb | bull | 0.0685 | 0.2254 | 0.3039 | 0.1515 | 0.1750 |
| fund_revenue_growth | bull | 0.0347 | 0.1218 | 0.2845 | 0.2121 | 0.1724 |
| momentum_reversal | bull | 0.0573 | 0.2194 | 0.2611 | 0.2121 | 0.1582 |
| fund_score | bull | 0.0379 | 0.1896 | 0.2001 | 0.1061 | 0.1107 |
| mom_x_lowvol_20_20 | bull | 0.0399 | 0.2136 | 0.1866 | 0.1288 | 0.1053 |
| fund_gross_margin | bull | 0.0102 | 0.0997 | 0.1019 | 0.1742 | 0.0598 |
| trend_lowvol | bear | 0.1521 | 0.2051 | 0.7418 | 0.5068 | 0.5589 |
| momentum_reversal | bear | 0.1187 | 0.2713 | 0.4373 | 0.3425 | 0.2935 |
| mom_x_lowvol_20_20 | bear | 0.1147 | 0.2582 | 0.4442 | 0.2877 | 0.2860 |
| fund_revenue_growth | bear | 0.0372 | 0.1043 | 0.3565 | 0.2329 | 0.2197 |
| fund_gross_margin | bear | 0.0268 | 0.1004 | 0.2670 | 0.2329 | 0.1646 |
| bb_width_20 | bear | 0.0541 | 0.2438 | 0.2217 | 0.1233 | 0.1245 |
| rsi_vol_combo | bear | 0.0454 | 0.2240 | 0.2026 | 0.2055 | 0.1221 |

### 互联网服务

- **Neutral**: ['fund_pb', 'momentum_reversal', 'mom_x_lowvol_20_20'] (单因子IC=0.0875, 组合IC=0.1197)
  - weights: [0.3511, 0.3392, 0.3098]
- **Bull**: ['fund_pb', 'volatility', 'trend_lowvol'] (单因子IC=0.1079, 组合IC=0.1405)
  - bull_weights: [0.3981, 0.3466, 0.2553]
- **Bear**: ['mom_x_lowvol_20_20', 'trend_lowvol'] (单因子IC=0.1202, 组合IC=0.1478)
  - bear_weights: [0.5151, 0.4849]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| fund_pb | neutral | 0.0702 | 0.1101 | 0.6376 | 0.4614 | 0.4659 |
| momentum_reversal | neutral | 0.0992 | 0.1624 | 0.6107 | 0.4739 | 0.4501 |
| mom_x_lowvol_20_20 | neutral | 0.0931 | 0.1624 | 0.5733 | 0.4342 | 0.4111 |
| volatility | neutral | 0.0949 | 0.1781 | 0.5330 | 0.4259 | 0.3800 |
| trend_lowvol | neutral | 0.0913 | 0.1843 | 0.4953 | 0.4071 | 0.3484 |
| rsi_vol_combo | neutral | 0.0798 | 0.1585 | 0.5038 | 0.3674 | 0.3445 |
| turnover_stability | neutral | 0.0525 | 0.1156 | 0.4546 | 0.3674 | 0.3108 |
| fund_profit_growth | neutral | 0.0497 | 0.1296 | 0.3833 | 0.2797 | 0.2453 |
| fund_pe | neutral | 0.0514 | 0.1440 | 0.3567 | 0.2902 | 0.2301 |
| fund_score | neutral | 0.0503 | 0.1584 | 0.3173 | 0.2380 | 0.1964 |
| fund_roe | neutral | 0.0470 | 0.1602 | 0.2934 | 0.1712 | 0.1718 |
| low_downside | neutral | 0.0376 | 0.1698 | 0.2216 | 0.2276 | 0.1360 |
| fund_pb | bull | 0.0994 | 0.1148 | 0.8662 | 0.6364 | 0.7087 |
| volatility | bull | 0.1260 | 0.1601 | 0.7870 | 0.5682 | 0.6171 |
| trend_lowvol | bull | 0.0983 | 0.1638 | 0.6000 | 0.5152 | 0.4545 |
| mom_x_lowvol_20_20 | bull | 0.0874 | 0.1488 | 0.5875 | 0.4242 | 0.4184 |
| low_downside | bull | 0.0864 | 0.1513 | 0.5713 | 0.4545 | 0.4155 |
| momentum_reversal | bull | 0.0779 | 0.1523 | 0.5115 | 0.3258 | 0.3391 |
| fund_pe | bull | 0.0552 | 0.1223 | 0.4512 | 0.3258 | 0.2991 |
| turnover_stability | bull | 0.0371 | 0.1140 | 0.3254 | 0.3182 | 0.2145 |
| rsi_vol_combo | bull | 0.0499 | 0.1497 | 0.3337 | 0.2500 | 0.2086 |
| fund_profit_growth | bull | 0.0317 | 0.1199 | 0.2644 | 0.1970 | 0.1582 |
| fund_score | bull | 0.0374 | 0.1560 | 0.2397 | 0.2045 | 0.1444 |
| fund_roe | bull | 0.0328 | 0.1669 | 0.1965 | 0.1439 | 0.1124 |
| mom_x_lowvol_20_20 | bear | 0.1093 | 0.1638 | 0.6672 | 0.4795 | 0.4935 |
| trend_lowvol | bear | 0.1312 | 0.2012 | 0.6521 | 0.4247 | 0.4645 |
| momentum_reversal | bear | 0.1055 | 0.1723 | 0.6124 | 0.4247 | 0.4362 |
| fund_score | bear | 0.1013 | 0.1740 | 0.5823 | 0.4247 | 0.4148 |
| fund_profit_growth | bear | 0.0774 | 0.1425 | 0.5432 | 0.4521 | 0.3944 |
| fund_gross_margin | bear | 0.0495 | 0.0934 | 0.5294 | 0.4247 | 0.3771 |
| fund_revenue_growth | bear | 0.0656 | 0.1500 | 0.4372 | 0.2055 | 0.2635 |
| fund_pe | bear | 0.0658 | 0.1705 | 0.3857 | 0.2603 | 0.2430 |
| rsi_vol_combo | bear | 0.0558 | 0.1630 | 0.3424 | 0.3973 | 0.2392 |
| fund_roe | bear | 0.0653 | 0.1687 | 0.3871 | 0.2055 | 0.2333 |
| fund_pb | bear | 0.0454 | 0.1325 | 0.3427 | 0.2329 | 0.2112 |
| turnover_stability | bear | 0.0369 | 0.1222 | 0.3015 | 0.3151 | 0.1982 |

### 互联网金融

- **Neutral**: ['turnover_stability', 'trend_lowvol'] (单因子IC=0.0734, 组合IC=0.1004)
  - weights: [0.5653, 0.4347]
- **Bull**: ['low_downside', 'volatility', 'fund_pb'] (单因子IC=0.1204, 组合IC=0.1336)
  - bull_weights: [0.3917, 0.3742, 0.2342]
- **Bear**: ['trend_lowvol', 'momentum_reversal', 'mom_x_lowvol_20_20'] (单因子IC=0.1692, 组合IC=0.184)
  - bear_weights: [0.3739, 0.3143, 0.3119]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| turnover_stability | neutral | 0.0636 | 0.1548 | 0.4105 | 0.3340 | 0.2738 |
| trend_lowvol | neutral | 0.0832 | 0.2483 | 0.3350 | 0.2568 | 0.2105 |
| volatility | neutral | 0.0613 | 0.2641 | 0.2320 | 0.1879 | 0.1378 |
| low_downside | neutral | 0.0609 | 0.2796 | 0.2179 | 0.1670 | 0.1271 |
| fund_pb | neutral | 0.0692 | 0.3434 | 0.2015 | 0.1461 | 0.1155 |
| fund_score | neutral | 0.0423 | 0.2144 | 0.1972 | 0.1608 | 0.1145 |
| mom_x_lowvol_20_20 | neutral | 0.0447 | 0.2189 | 0.2044 | 0.1148 | 0.1139 |
| fund_roe | neutral | 0.0449 | 0.2430 | 0.1847 | 0.1211 | 0.1035 |
| fund_profit_growth | neutral | 0.0278 | 0.1582 | 0.1757 | 0.1378 | 0.1000 |
| low_downside | bull | 0.1290 | 0.2399 | 0.5377 | 0.3712 | 0.3687 |
| volatility | bull | 0.1163 | 0.2252 | 0.5165 | 0.3636 | 0.3522 |
| fund_pb | bull | 0.1160 | 0.3268 | 0.3549 | 0.2424 | 0.2204 |
| mom_x_lowvol_20_20 | bull | 0.0618 | 0.1727 | 0.3580 | 0.2197 | 0.2183 |
| turnover_stability | bull | 0.0560 | 0.1824 | 0.3071 | 0.3182 | 0.2024 |
| momentum_reversal | bull | 0.0639 | 0.1943 | 0.3288 | 0.2045 | 0.1980 |
| fund_pe | bull | 0.1079 | 0.3526 | 0.3061 | 0.2273 | 0.1878 |
| trend_lowvol | bull | 0.0666 | 0.2463 | 0.2704 | 0.1970 | 0.1618 |
| fund_score | bull | 0.0525 | 0.2170 | 0.2420 | 0.1364 | 0.1375 |
| rsi_vol_combo | bull | 0.0455 | 0.2003 | 0.2272 | 0.1742 | 0.1334 |
| fund_revenue_growth | bull | 0.0186 | 0.1193 | 0.1562 | 0.1970 | 0.0935 |
| trend_lowvol | bear | 0.1768 | 0.2749 | 0.6432 | 0.2603 | 0.4053 |
| momentum_reversal | bear | 0.1688 | 0.3189 | 0.5291 | 0.2877 | 0.3407 |
| mom_x_lowvol_20_20 | bear | 0.1621 | 0.2890 | 0.5610 | 0.2055 | 0.3381 |
| bb_width_20 | bear | 0.1558 | 0.3257 | 0.4783 | 0.3151 | 0.3145 |
| fund_revenue_growth | bear | 0.0314 | 0.1227 | 0.2560 | 0.2603 | 0.1613 |

### 交运设备

- **Neutral**: ['mom_x_lowvol_20_20', 'fund_pb', 'momentum_reversal'] (单因子IC=0.0752, 组合IC=0.1078)
  - weights: [0.3396, 0.3374, 0.323]
- **Bull**: ['fund_pb', 'trend_lowvol', 'turnover_stability'] (单因子IC=0.0893, 组合IC=0.1314)
  - bull_weights: [0.4818, 0.2648, 0.2534]
- **Bear**: ['fund_revenue_growth', 'momentum_reversal'] (单因子IC=0.1, 组合IC=0.1422)
  - bear_weights: [0.5327, 0.4673]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| mom_x_lowvol_20_20 | neutral | 0.0796 | 0.2248 | 0.3542 | 0.2735 | 0.2256 |
| fund_pb | neutral | 0.0690 | 0.1978 | 0.3490 | 0.2839 | 0.2241 |
| momentum_reversal | neutral | 0.0769 | 0.2269 | 0.3388 | 0.2662 | 0.2145 |
| rsi_vol_combo | neutral | 0.0623 | 0.2288 | 0.2723 | 0.2109 | 0.1648 |
| turnover_stability | neutral | 0.0516 | 0.1898 | 0.2720 | 0.2035 | 0.1637 |
| trend_lowvol | neutral | 0.0605 | 0.2473 | 0.2447 | 0.2505 | 0.1530 |
| volatility | neutral | 0.0534 | 0.2185 | 0.2442 | 0.2276 | 0.1499 |
| fund_profit_growth | neutral | 0.0389 | 0.1909 | 0.2035 | 0.1461 | 0.1166 |
| low_downside | neutral | 0.0386 | 0.2220 | 0.1740 | 0.1973 | 0.1041 |
| fund_pb | bull | 0.1139 | 0.1757 | 0.6481 | 0.4773 | 0.4787 |
| trend_lowvol | bull | 0.0849 | 0.2139 | 0.3969 | 0.3258 | 0.2631 |
| turnover_stability | bull | 0.0691 | 0.1788 | 0.3865 | 0.3030 | 0.2518 |
| low_downside | bull | 0.0832 | 0.2169 | 0.3835 | 0.2576 | 0.2412 |
| volatility | bull | 0.0604 | 0.2059 | 0.2932 | 0.2500 | 0.1832 |
| momentum_reversal | bull | 0.0687 | 0.2210 | 0.3110 | 0.1667 | 0.1814 |
| rsi_vol_combo | bull | 0.0426 | 0.2280 | 0.1869 | 0.1818 | 0.1104 |
| fund_gross_margin | bull | 0.0244 | 0.1754 | 0.1391 | 0.1894 | 0.0827 |
| fund_revenue_growth | bear | 0.1014 | 0.2064 | 0.4914 | 0.4247 | 0.3500 |
| momentum_reversal | bear | 0.0987 | 0.2201 | 0.4483 | 0.3699 | 0.3070 |
| mom_x_lowvol_20_20 | bear | 0.0920 | 0.2745 | 0.3352 | 0.4521 | 0.2434 |
| fund_pb | bear | 0.0745 | 0.2259 | 0.3297 | 0.1781 | 0.1942 |
| fund_score | bear | 0.0627 | 0.2302 | 0.2724 | 0.1233 | 0.1530 |
| fund_gross_margin | bear | 0.0309 | 0.1654 | 0.1866 | 0.2055 | 0.1125 |

### 京津冀

- **Neutral**: ['fund_score', 'trend_lowvol'] (单因子IC=0.0754, 组合IC=0.0891)
  - weights: [0.5047, 0.4953]
- **Bull**: ['volatility', 'mom_x_lowvol_20_20'] (单因子IC=0.11, 组合IC=0.1206)
  - bull_weights: [0.5075, 0.4925]
- **Bear**: ['momentum_reversal', 'mom_x_lowvol_20_20', 'trend_lowvol'] (单因子IC=0.1402, 组合IC=0.1596)
  - bear_weights: [0.3804, 0.3148, 0.3048]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| fund_score | neutral | 0.0629 | 0.1842 | 0.3417 | 0.2818 | 0.2190 |
| trend_lowvol | neutral | 0.0880 | 0.2507 | 0.3508 | 0.2255 | 0.2150 |
| fund_profit_growth | neutral | 0.0496 | 0.1530 | 0.3238 | 0.2359 | 0.2001 |
| volatility | neutral | 0.0675 | 0.2286 | 0.2951 | 0.3069 | 0.1928 |
| mom_x_lowvol_20_20 | neutral | 0.0719 | 0.2282 | 0.3152 | 0.2150 | 0.1915 |
| momentum_reversal | neutral | 0.0722 | 0.2332 | 0.3098 | 0.2317 | 0.1908 |
| fund_roe | neutral | 0.0523 | 0.2037 | 0.2568 | 0.1712 | 0.1504 |
| fund_pe | neutral | 0.0518 | 0.2158 | 0.2403 | 0.2109 | 0.1455 |
| fund_revenue_growth | neutral | 0.0314 | 0.1429 | 0.2200 | 0.1461 | 0.1261 |
| rsi_vol_combo | neutral | 0.0484 | 0.2240 | 0.2161 | 0.1482 | 0.1241 |
| low_downside | neutral | 0.0406 | 0.2173 | 0.1869 | 0.1983 | 0.1120 |
| turnover_stability | neutral | 0.0335 | 0.1718 | 0.1949 | 0.1409 | 0.1112 |
| volatility | bull | 0.1133 | 0.2015 | 0.5624 | 0.4318 | 0.4026 |
| mom_x_lowvol_20_20 | bull | 0.1066 | 0.1964 | 0.5428 | 0.4394 | 0.3907 |
| low_downside | bull | 0.0925 | 0.1768 | 0.5231 | 0.4394 | 0.3765 |
| momentum_reversal | bull | 0.1091 | 0.2065 | 0.5283 | 0.4015 | 0.3702 |
| rsi_vol_combo | bull | 0.0913 | 0.1964 | 0.4650 | 0.3409 | 0.3118 |
| trend_lowvol | bull | 0.0760 | 0.1954 | 0.3890 | 0.2879 | 0.2505 |
| turnover_stability | bull | 0.0508 | 0.1873 | 0.2713 | 0.1515 | 0.1562 |
| fund_pb | bull | 0.0488 | 0.2276 | 0.2146 | 0.2424 | 0.1333 |
| momentum_reversal | bear | 0.1545 | 0.2669 | 0.5790 | 0.5068 | 0.4362 |
| mom_x_lowvol_20_20 | bear | 0.1391 | 0.2850 | 0.4880 | 0.4795 | 0.3610 |
| trend_lowvol | bear | 0.1269 | 0.2488 | 0.5102 | 0.3699 | 0.3494 |
| rsi_vol_combo | bear | 0.0899 | 0.2310 | 0.3891 | 0.2603 | 0.2452 |
| fund_revenue_growth | bear | 0.0438 | 0.1350 | 0.3248 | 0.3151 | 0.2136 |
| turnover_stability | bear | 0.0343 | 0.1419 | 0.2415 | 0.2603 | 0.1522 |
| bb_width_20 | bear | 0.0458 | 0.2207 | 0.2076 | 0.2055 | 0.1251 |
| fund_score | bear | 0.0344 | 0.1735 | 0.1984 | 0.1781 | 0.1169 |

### 人工智能

- **Neutral**: ['momentum_reversal', 'mom_x_lowvol_20_20', 'trend_lowvol'] (单因子IC=0.0927, 组合IC=0.1026)
  - weights: [0.346, 0.3279, 0.3261]
- **Bull**: ['volatility', 'low_downside', 'fund_pb'] (单因子IC=0.0986, 组合IC=0.1198)
  - bull_weights: [0.3717, 0.3498, 0.2784]
- **Bear**: ['mom_x_lowvol_20_20', 'trend_lowvol'] (单因子IC=0.1109, 组合IC=0.1247)
  - bear_weights: [0.5062, 0.4938]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| momentum_reversal | neutral | 0.0922 | 0.1463 | 0.6301 | 0.4635 | 0.4611 |
| mom_x_lowvol_20_20 | neutral | 0.0887 | 0.1463 | 0.6065 | 0.4405 | 0.4368 |
| trend_lowvol | neutral | 0.0973 | 0.1633 | 0.5955 | 0.4593 | 0.4345 |
| rsi_vol_combo | neutral | 0.0731 | 0.1388 | 0.5269 | 0.3883 | 0.3658 |
| fund_pb | neutral | 0.0760 | 0.1424 | 0.5340 | 0.3424 | 0.3584 |
| volatility | neutral | 0.0834 | 0.1761 | 0.4739 | 0.3653 | 0.3235 |
| turnover_stability | neutral | 0.0337 | 0.0755 | 0.4461 | 0.3674 | 0.3050 |
| fund_profit_growth | neutral | 0.0420 | 0.1020 | 0.4119 | 0.2714 | 0.2618 |
| fund_pe | neutral | 0.0496 | 0.1410 | 0.3518 | 0.2672 | 0.2229 |
| low_downside | neutral | 0.0465 | 0.1626 | 0.2862 | 0.2630 | 0.1808 |
| fund_score | neutral | 0.0388 | 0.1369 | 0.2834 | 0.1962 | 0.1695 |
| fund_revenue_growth | neutral | 0.0238 | 0.0975 | 0.2437 | 0.1921 | 0.1453 |
| volatility | bull | 0.1116 | 0.1483 | 0.7523 | 0.5833 | 0.5956 |
| low_downside | bull | 0.0899 | 0.1239 | 0.7253 | 0.5455 | 0.5605 |
| fund_pb | bull | 0.0943 | 0.1465 | 0.6435 | 0.3864 | 0.4461 |
| trend_lowvol | bull | 0.0875 | 0.1588 | 0.5509 | 0.4470 | 0.3986 |
| turnover_stability | bull | 0.0424 | 0.0755 | 0.5615 | 0.3939 | 0.3913 |
| mom_x_lowvol_20_20 | bull | 0.0665 | 0.1413 | 0.4708 | 0.3939 | 0.3282 |
| momentum_reversal | bull | 0.0605 | 0.1415 | 0.4274 | 0.3788 | 0.2946 |
| fund_pe | bull | 0.0593 | 0.1412 | 0.4203 | 0.3409 | 0.2818 |
| fund_revenue_growth | bull | 0.0308 | 0.1012 | 0.3042 | 0.2273 | 0.1867 |
| fund_profit_growth | bull | 0.0312 | 0.1015 | 0.3070 | 0.1818 | 0.1814 |
| rsi_vol_combo | bull | 0.0343 | 0.1309 | 0.2621 | 0.1591 | 0.1519 |
| fund_score | bull | 0.0353 | 0.1354 | 0.2606 | 0.1364 | 0.1481 |
| fund_roe | bull | 0.0210 | 0.1420 | 0.1478 | 0.1364 | 0.0840 |
| mom_x_lowvol_20_20 | bear | 0.1130 | 0.1640 | 0.6886 | 0.5068 | 0.5188 |
| trend_lowvol | bear | 0.1088 | 0.1619 | 0.6717 | 0.5068 | 0.5061 |
| momentum_reversal | bear | 0.1067 | 0.1588 | 0.6717 | 0.4521 | 0.4877 |
| rsi_vol_combo | bear | 0.0581 | 0.1285 | 0.4524 | 0.2877 | 0.2913 |
| turnover_stability | bear | 0.0242 | 0.0698 | 0.3459 | 0.2877 | 0.2227 |
| fund_revenue_growth | bear | 0.0426 | 0.1173 | 0.3632 | 0.1507 | 0.2090 |
| fund_profit_growth | bear | 0.0441 | 0.1268 | 0.3480 | 0.1781 | 0.2050 |

### 人形机器人

- **Neutral**: ['fund_pb', 'fund_pe', 'momentum_reversal'] (单因子IC=0.0745, 组合IC=0.104)
  - weights: [0.3406, 0.3329, 0.3266]
- **Bull**: ['low_downside', 'volatility', 'fund_pb'] (单因子IC=0.0804, 组合IC=0.0992)
  - bull_weights: [0.3659, 0.3251, 0.309]
- **Bear**: ['momentum_reversal', 'fund_pe'] (单因子IC=0.0595, 组合IC=0.1118)
  - bear_weights: [0.5071, 0.4929]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| fund_pb | neutral | 0.0787 | 0.1564 | 0.5034 | 0.3507 | 0.3400 |
| fund_pe | neutral | 0.0673 | 0.1344 | 0.5006 | 0.3278 | 0.3323 |
| momentum_reversal | neutral | 0.0775 | 0.1621 | 0.4783 | 0.3633 | 0.3260 |
| mom_x_lowvol_20_20 | neutral | 0.0760 | 0.1568 | 0.4850 | 0.3361 | 0.3240 |
| rsi_vol_combo | neutral | 0.0628 | 0.1435 | 0.4377 | 0.3570 | 0.2970 |
| fund_profit_growth | neutral | 0.0557 | 0.1259 | 0.4424 | 0.3236 | 0.2928 |
| trend_lowvol | neutral | 0.0654 | 0.1852 | 0.3534 | 0.2463 | 0.2202 |
| volatility | neutral | 0.0596 | 0.1863 | 0.3202 | 0.2484 | 0.1998 |
| fund_score | neutral | 0.0447 | 0.1372 | 0.3260 | 0.2192 | 0.1987 |
| fund_revenue_growth | neutral | 0.0265 | 0.1139 | 0.2328 | 0.1921 | 0.1388 |
| fund_roe | neutral | 0.0299 | 0.1307 | 0.2285 | 0.1336 | 0.1295 |
| low_downside | neutral | 0.0285 | 0.1673 | 0.1706 | 0.1211 | 0.0956 |
| low_downside | bull | 0.0762 | 0.1335 | 0.5703 | 0.3939 | 0.3975 |
| volatility | bull | 0.0881 | 0.1673 | 0.5266 | 0.3409 | 0.3531 |
| fund_pb | bull | 0.0769 | 0.1589 | 0.4842 | 0.3864 | 0.3356 |
| fund_pe | bull | 0.0562 | 0.1261 | 0.4452 | 0.3561 | 0.3019 |
| turnover_stability | bull | 0.0394 | 0.0846 | 0.4659 | 0.2803 | 0.2982 |
| mom_x_lowvol_20_20 | bull | 0.0522 | 0.1493 | 0.3495 | 0.2576 | 0.2197 |
| trend_lowvol | bull | 0.0626 | 0.1747 | 0.3585 | 0.1818 | 0.2118 |
| fund_profit_growth | bull | 0.0395 | 0.1264 | 0.3126 | 0.3030 | 0.2037 |
| momentum_reversal | bull | 0.0425 | 0.1500 | 0.2836 | 0.1818 | 0.1676 |
| rsi_vol_combo | bull | 0.0319 | 0.1337 | 0.2386 | 0.1136 | 0.1329 |
| fund_score | bull | 0.0290 | 0.1402 | 0.2067 | 0.1061 | 0.1143 |
| fund_revenue_growth | bull | 0.0210 | 0.1212 | 0.1733 | 0.1288 | 0.0978 |
| momentum_reversal | bear | 0.0700 | 0.1994 | 0.3512 | 0.2055 | 0.2117 |
| fund_pe | bear | 0.0491 | 0.1437 | 0.3414 | 0.2055 | 0.2058 |
| fund_gross_margin | bear | 0.0460 | 0.1488 | 0.3094 | 0.2603 | 0.1949 |
| fund_roe | bear | 0.0465 | 0.1461 | 0.3183 | 0.1233 | 0.1788 |
| fund_profit_growth | bear | 0.0436 | 0.1536 | 0.2839 | 0.1507 | 0.1634 |
| trend_lowvol | bear | 0.0572 | 0.2426 | 0.2356 | 0.1781 | 0.1388 |
| bb_width_20 | bear | 0.0434 | 0.2217 | 0.1959 | 0.1507 | 0.1127 |

### 人脑工程

- **Neutral**: ['fund_pb', 'mom_x_lowvol_20_20', 'momentum_reversal'] (单因子IC=0.0827, 组合IC=0.1218)
  - weights: [0.4094, 0.3001, 0.2905]
- **Bull**: ['low_downside', 'volatility', 'fund_pb'] (单因子IC=0.1133, 组合IC=0.1577)
  - bull_weights: [0.3655, 0.3519, 0.2826]
- **Bear**: ['mom_x_lowvol_20_20'] (单因子IC=0.1055, 组合IC=0.1055)
  - bear_weights: [1.0]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| fund_pb | neutral | 0.0952 | 0.1926 | 0.4942 | 0.3173 | 0.3255 |
| mom_x_lowvol_20_20 | neutral | 0.0767 | 0.2040 | 0.3759 | 0.2693 | 0.2386 |
| momentum_reversal | neutral | 0.0762 | 0.2094 | 0.3639 | 0.2693 | 0.2310 |
| trend_lowvol | neutral | 0.0813 | 0.2386 | 0.3406 | 0.2610 | 0.2147 |
| volatility | neutral | 0.0652 | 0.2229 | 0.2924 | 0.2046 | 0.1761 |
| fund_pe | neutral | 0.0552 | 0.2145 | 0.2574 | 0.1816 | 0.1521 |
| rsi_vol_combo | neutral | 0.0524 | 0.2116 | 0.2478 | 0.1691 | 0.1448 |
| low_downside | neutral | 0.0484 | 0.2177 | 0.2223 | 0.1712 | 0.1301 |
| fund_profit_growth | neutral | 0.0277 | 0.1970 | 0.1408 | 0.1294 | 0.0795 |
| low_downside | bull | 0.1130 | 0.1830 | 0.6174 | 0.4773 | 0.4561 |
| volatility | bull | 0.1351 | 0.2156 | 0.6267 | 0.4015 | 0.4392 |
| fund_pb | bull | 0.0918 | 0.1775 | 0.5173 | 0.3636 | 0.3527 |
| trend_lowvol | bull | 0.1132 | 0.2536 | 0.4464 | 0.2273 | 0.2739 |
| mom_x_lowvol_20_20 | bull | 0.0761 | 0.2126 | 0.3579 | 0.2045 | 0.2155 |
| momentum_reversal | bull | 0.0705 | 0.2154 | 0.3272 | 0.1667 | 0.1909 |
| turnover_stability | bull | 0.0460 | 0.1844 | 0.2495 | 0.1591 | 0.1446 |
| rsi_vol_combo | bull | 0.0427 | 0.1933 | 0.2211 | 0.2045 | 0.1331 |
| fund_score | bull | 0.0375 | 0.1955 | 0.1920 | 0.1591 | 0.1113 |
| fund_revenue_growth | bull | 0.0328 | 0.2011 | 0.1631 | 0.1515 | 0.0939 |
| mom_x_lowvol_20_20 | bear | 0.1055 | 0.2256 | 0.4675 | 0.2603 | 0.2946 |
| momentum_reversal | bear | 0.0699 | 0.1948 | 0.3588 | 0.1781 | 0.2113 |
| trend_lowvol | bear | 0.0471 | 0.2178 | 0.2161 | 0.1233 | 0.1214 |

### 价值股

- **Neutral**: ['fund_pe', 'fund_pb'] (单因子IC=0.1007, 组合IC=0.1168)
  - weights: [0.5496, 0.4504]
- **Bull**: ['low_downside', 'fund_pe'] (单因子IC=0.0391, 组合IC=0.04)
  - bull_weights: [0.5171, 0.4829]
- **Bear**: ['mom_x_lowvol_20_20', 'bb_width_20', 'trend_lowvol'] (单因子IC=0.0927, 组合IC=0.1225)
  - bear_weights: [0.373, 0.3482, 0.2788]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| fund_pe | neutral | 0.1108 | 0.1985 | 0.5581 | 0.3841 | 0.3863 |
| fund_pb | neutral | 0.0906 | 0.1953 | 0.4638 | 0.3653 | 0.3166 |
| volatility | neutral | 0.0399 | 0.1314 | 0.3038 | 0.2401 | 0.1884 |
| trend_lowvol | neutral | 0.0620 | 0.2297 | 0.2700 | 0.1942 | 0.1612 |
| fund_profit_growth | neutral | 0.0312 | 0.1471 | 0.2120 | 0.1441 | 0.1213 |
| momentum_reversal | neutral | 0.0330 | 0.2056 | 0.1603 | 0.1106 | 0.0890 |
| low_downside | neutral | 0.0231 | 0.1838 | 0.1258 | 0.1044 | 0.0694 |
| low_downside | bull | 0.0353 | 0.1504 | 0.2347 | 0.1591 | 0.1360 |
| fund_pe | bull | 0.0429 | 0.1920 | 0.2236 | 0.1364 | 0.1270 |
| mom_x_lowvol_20_20 | bear | 0.0792 | 0.1805 | 0.4387 | 0.5616 | 0.3426 |
| bb_width_20 | bear | 0.0910 | 0.1870 | 0.4863 | 0.3151 | 0.3198 |
| trend_lowvol | bear | 0.1078 | 0.2770 | 0.3893 | 0.3151 | 0.2560 |
| fund_profit_growth | bear | 0.0489 | 0.1322 | 0.3696 | 0.2877 | 0.2379 |
| momentum_reversal | bear | 0.0914 | 0.2722 | 0.3358 | 0.3973 | 0.2346 |
| fund_pb | bear | 0.0509 | 0.2192 | 0.2324 | 0.1781 | 0.1369 |
| rsi_vol_combo | bear | 0.0458 | 0.2336 | 0.1959 | 0.3425 | 0.1315 |
| wash_sale_score | bear | 0.0274 | 0.1237 | 0.2216 | 0.1562 | 0.1281 |

### 传感器

- **Neutral**: ['trend_lowvol', 'momentum_reversal'] (单因子IC=0.101, 组合IC=0.1108)
  - weights: [0.5418, 0.4582]
- **Bull**: ['volatility', 'trend_lowvol'] (单因子IC=0.1032, 组合IC=0.1124)
  - bull_weights: [0.5319, 0.4681]
- **Bear**: ['momentum_reversal'] (单因子IC=0.0941, 组合IC=0.0941)
  - bear_weights: [1.0]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| trend_lowvol | neutral | 0.1082 | 0.1883 | 0.5745 | 0.4259 | 0.4096 |
| momentum_reversal | neutral | 0.0938 | 0.1854 | 0.5059 | 0.3695 | 0.3464 |
| mom_x_lowvol_20_20 | neutral | 0.0906 | 0.1835 | 0.4937 | 0.3528 | 0.3339 |
| volatility | neutral | 0.0851 | 0.1853 | 0.4592 | 0.3278 | 0.3049 |
| rsi_vol_combo | neutral | 0.0741 | 0.1789 | 0.4140 | 0.3215 | 0.2736 |
| fund_pe | neutral | 0.0578 | 0.1570 | 0.3685 | 0.2985 | 0.2392 |
| fund_profit_growth | neutral | 0.0539 | 0.1534 | 0.3511 | 0.2777 | 0.2243 |
| fund_pb | neutral | 0.0669 | 0.1926 | 0.3471 | 0.2651 | 0.2196 |
| turnover_stability | neutral | 0.0388 | 0.1258 | 0.3083 | 0.2359 | 0.1905 |
| fund_score | neutral | 0.0436 | 0.1646 | 0.2646 | 0.2276 | 0.1624 |
| low_downside | neutral | 0.0417 | 0.1771 | 0.2353 | 0.2109 | 0.1424 |
| fund_revenue_growth | neutral | 0.0238 | 0.1402 | 0.1699 | 0.1253 | 0.0956 |
| fund_roe | neutral | 0.0273 | 0.1689 | 0.1615 | 0.1336 | 0.0916 |
| volatility | bull | 0.0987 | 0.1593 | 0.6192 | 0.4697 | 0.4550 |
| trend_lowvol | bull | 0.1078 | 0.1846 | 0.5839 | 0.3712 | 0.4004 |
| low_downside | bull | 0.0816 | 0.1613 | 0.5056 | 0.4167 | 0.3581 |
| fund_pb | bull | 0.0963 | 0.1917 | 0.5025 | 0.3864 | 0.3483 |
| mom_x_lowvol_20_20 | bull | 0.0703 | 0.1635 | 0.4300 | 0.2879 | 0.2769 |
| momentum_reversal | bull | 0.0695 | 0.1722 | 0.4038 | 0.2803 | 0.2585 |
| fund_pe | bull | 0.0593 | 0.1706 | 0.3475 | 0.2955 | 0.2251 |
| turnover_stability | bull | 0.0370 | 0.1227 | 0.3015 | 0.2727 | 0.1919 |
| rsi_vol_combo | bull | 0.0477 | 0.1793 | 0.2660 | 0.2045 | 0.1602 |
| stroke_phase | bull | 0.0169 | 0.1298 | 0.1298 | 0.1667 | 0.0757 |
| momentum_reversal | bear | 0.0941 | 0.1900 | 0.4955 | 0.3425 | 0.3326 |
| mom_x_lowvol_20_20 | bear | 0.0924 | 0.1847 | 0.5000 | 0.3151 | 0.3288 |
| rsi_vol_combo | bear | 0.0484 | 0.1746 | 0.2770 | 0.2329 | 0.1708 |
| fund_profit_growth | bear | 0.0395 | 0.1654 | 0.2388 | 0.2329 | 0.1472 |

### 低价股

- **Neutral**: ['turnover_stability', 'fund_score', 'volatility'] (单因子IC=0.0684, 组合IC=0.088)
  - weights: [0.3769, 0.3131, 0.31]
- **Bull**: ['low_downside'] (单因子IC=0.119, 组合IC=0.119)
  - bull_weights: [1.0]
- **Bear**: ['momentum_reversal', 'mom_x_lowvol_20_20'] (单因子IC=0.1508, 组合IC=0.1622)
  - bear_weights: [0.52, 0.48]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| turnover_stability | neutral | 0.0644 | 0.1288 | 0.4994 | 0.3841 | 0.3456 |
| fund_score | neutral | 0.0623 | 0.1479 | 0.4213 | 0.3633 | 0.2872 |
| volatility | neutral | 0.0787 | 0.1936 | 0.4064 | 0.3987 | 0.2843 |
| mom_x_lowvol_20_20 | neutral | 0.0677 | 0.1740 | 0.3888 | 0.3257 | 0.2577 |
| trend_lowvol | neutral | 0.0779 | 0.2111 | 0.3688 | 0.3048 | 0.2406 |
| fund_roe | neutral | 0.0558 | 0.1593 | 0.3506 | 0.3236 | 0.2320 |
| rsi_vol_combo | neutral | 0.0626 | 0.1711 | 0.3659 | 0.2589 | 0.2303 |
| momentum_reversal | neutral | 0.0691 | 0.1925 | 0.3588 | 0.2568 | 0.2254 |
| fund_profit_growth | neutral | 0.0404 | 0.1201 | 0.3366 | 0.3278 | 0.2235 |
| low_downside | neutral | 0.0674 | 0.2115 | 0.3187 | 0.2965 | 0.2066 |
| fund_revenue_growth | neutral | 0.0251 | 0.1046 | 0.2397 | 0.2401 | 0.1486 |
| fund_pb | neutral | 0.0472 | 0.1856 | 0.2544 | 0.1649 | 0.1482 |
| fund_pe | neutral | 0.0433 | 0.1985 | 0.2179 | 0.1733 | 0.1278 |
| low_downside | bull | 0.1190 | 0.1646 | 0.7231 | 0.4394 | 0.5204 |
| volatility | bull | 0.0923 | 0.1639 | 0.5631 | 0.4470 | 0.4074 |
| turnover_stability | bull | 0.0452 | 0.1198 | 0.3770 | 0.2424 | 0.2342 |
| trend_lowvol | bull | 0.0665 | 0.1986 | 0.3348 | 0.2197 | 0.2042 |
| momentum_reversal | bull | 0.0552 | 0.1803 | 0.3059 | 0.2273 | 0.1877 |
| mom_x_lowvol_20_20 | bull | 0.0404 | 0.1651 | 0.2448 | 0.1591 | 0.1419 |
| rsi_vol_combo | bull | 0.0422 | 0.1730 | 0.2438 | 0.1591 | 0.1413 |
| fund_pe | bull | 0.0299 | 0.2103 | 0.1423 | 0.1288 | 0.0803 |
| stroke_phase | bull | 0.0170 | 0.1320 | 0.1286 | 0.1742 | 0.0755 |
| momentum_reversal | bear | 0.1622 | 0.2134 | 0.7599 | 0.5068 | 0.5725 |
| mom_x_lowvol_20_20 | bear | 0.1395 | 0.1952 | 0.7144 | 0.4795 | 0.5285 |
| rsi_vol_combo | bear | 0.1272 | 0.1916 | 0.6640 | 0.4247 | 0.4730 |
| trend_lowvol | bear | 0.1209 | 0.1943 | 0.6225 | 0.3699 | 0.4264 |
| turnover_stability | bear | 0.0437 | 0.0893 | 0.4894 | 0.3699 | 0.3352 |
| fund_revenue_growth | bear | 0.0413 | 0.1147 | 0.3599 | 0.2877 | 0.2317 |
| fund_roe | bear | 0.0486 | 0.1828 | 0.2655 | 0.2329 | 0.1637 |
| volatility | bear | 0.0406 | 0.1680 | 0.2418 | 0.2055 | 0.1457 |
| fund_gross_margin | bear | 0.0194 | 0.0841 | 0.2313 | 0.1507 | 0.1331 |
| fund_score | bear | 0.0388 | 0.1719 | 0.2256 | 0.1781 | 0.1329 |

### 低空经济

- **Neutral**: ['momentum_reversal', 'trend_lowvol'] (单因子IC=0.085, 组合IC=0.0929)
  - weights: [0.5018, 0.4982]
- **Bull**: ['low_downside', 'fund_pb', 'volatility'] (单因子IC=0.1207, 组合IC=0.1445)
  - bull_weights: [0.3657, 0.322, 0.3123]
- **Bear**: ['momentum_reversal'] (单因子IC=0.0926, 组合IC=0.0926)
  - bear_weights: [1.0]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| momentum_reversal | neutral | 0.0809 | 0.1620 | 0.4995 | 0.3570 | 0.3389 |
| trend_lowvol | neutral | 0.0891 | 0.1752 | 0.5084 | 0.3236 | 0.3364 |
| mom_x_lowvol_20_20 | neutral | 0.0738 | 0.1578 | 0.4676 | 0.3403 | 0.3133 |
| rsi_vol_combo | neutral | 0.0638 | 0.1511 | 0.4226 | 0.2985 | 0.2744 |
| volatility | neutral | 0.0677 | 0.1711 | 0.3958 | 0.3194 | 0.2611 |
| fund_pb | neutral | 0.0641 | 0.1608 | 0.3987 | 0.2610 | 0.2514 |
| turnover_stability | neutral | 0.0337 | 0.0940 | 0.3591 | 0.3486 | 0.2421 |
| fund_profit_growth | neutral | 0.0447 | 0.1198 | 0.3728 | 0.2317 | 0.2296 |
| fund_pe | neutral | 0.0454 | 0.1457 | 0.3117 | 0.2255 | 0.1910 |
| fund_score | neutral | 0.0398 | 0.1441 | 0.2766 | 0.1921 | 0.1648 |
| low_downside | neutral | 0.0406 | 0.1738 | 0.2337 | 0.2067 | 0.1410 |
| fund_revenue_growth | neutral | 0.0228 | 0.1142 | 0.2000 | 0.1775 | 0.1178 |
| fund_roe | neutral | 0.0250 | 0.1490 | 0.1676 | 0.1315 | 0.0948 |
| low_downside | bull | 0.1235 | 0.1372 | 0.9000 | 0.6061 | 0.7227 |
| fund_pb | bull | 0.1134 | 0.1431 | 0.7923 | 0.6061 | 0.6363 |
| volatility | bull | 0.1254 | 0.1570 | 0.7986 | 0.5455 | 0.6171 |
| trend_lowvol | bull | 0.1087 | 0.1514 | 0.7183 | 0.5152 | 0.5441 |
| momentum_reversal | bull | 0.0731 | 0.1479 | 0.4941 | 0.3864 | 0.3425 |
| fund_pe | bull | 0.0640 | 0.1448 | 0.4418 | 0.4167 | 0.3129 |
| mom_x_lowvol_20_20 | bull | 0.0652 | 0.1429 | 0.4560 | 0.3258 | 0.3022 |
| turnover_stability | bull | 0.0320 | 0.0902 | 0.3547 | 0.2576 | 0.2230 |
| rsi_vol_combo | bull | 0.0401 | 0.1439 | 0.2785 | 0.1818 | 0.1646 |
| fund_score | bull | 0.0200 | 0.1284 | 0.1562 | 0.1136 | 0.0869 |
| stroke_phase | bull | 0.0130 | 0.0951 | 0.1369 | 0.1515 | 0.0788 |
| momentum_reversal | bear | 0.0926 | 0.1747 | 0.5297 | 0.3699 | 0.3628 |
| rsi_vol_combo | bear | 0.0635 | 0.1361 | 0.4665 | 0.3425 | 0.3132 |
| mom_x_lowvol_20_20 | bear | 0.0786 | 0.1693 | 0.4643 | 0.2877 | 0.2989 |
| bb_width_20 | bear | 0.0396 | 0.1934 | 0.2046 | 0.2329 | 0.1261 |

### 体外诊断概念

- **Neutral**: ['volatility', 'fund_pb', 'turnover_stability'] (单因子IC=0.0685, 组合IC=0.1045)
  - weights: [0.3821, 0.3245, 0.2934]
- **Bull**: ['volatility', 'fund_profit_growth'] (单因子IC=0.0694, 组合IC=0.0968)
  - bull_weights: [0.556, 0.444]
- **Bear**: ['momentum_reversal', 'mom_x_lowvol_20_20'] (单因子IC=0.1208, 组合IC=0.1219)
  - bear_weights: [0.5082, 0.4918]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| volatility | neutral | 0.0846 | 0.1995 | 0.4241 | 0.3466 | 0.2855 |
| fund_pb | neutral | 0.0703 | 0.1879 | 0.3741 | 0.2965 | 0.2425 |
| turnover_stability | neutral | 0.0506 | 0.1479 | 0.3424 | 0.2808 | 0.2193 |
| momentum_reversal | neutral | 0.0577 | 0.2107 | 0.2740 | 0.1775 | 0.1613 |
| trend_lowvol | neutral | 0.0598 | 0.2323 | 0.2574 | 0.1733 | 0.1510 |
| mom_x_lowvol_20_20 | neutral | 0.0528 | 0.2054 | 0.2569 | 0.1618 | 0.1492 |
| fund_profit_growth | neutral | 0.0424 | 0.1745 | 0.2429 | 0.2171 | 0.1478 |
| low_downside | neutral | 0.0463 | 0.2089 | 0.2219 | 0.1545 | 0.1281 |
| rsi_vol_combo | neutral | 0.0393 | 0.1930 | 0.2036 | 0.1357 | 0.1156 |
| volatility | bull | 0.0844 | 0.1917 | 0.4401 | 0.3712 | 0.3018 |
| fund_profit_growth | bull | 0.0544 | 0.1419 | 0.3832 | 0.2576 | 0.2410 |
| low_downside | bull | 0.0582 | 0.1664 | 0.3498 | 0.3106 | 0.2293 |
| fund_pb | bull | 0.0645 | 0.2104 | 0.3066 | 0.2879 | 0.1974 |
| fund_pe | bull | 0.0597 | 0.2048 | 0.2916 | 0.1667 | 0.1701 |
| turnover_stability | bull | 0.0418 | 0.1731 | 0.2413 | 0.2121 | 0.1462 |
| fund_revenue_growth | bull | 0.0339 | 0.1506 | 0.2250 | 0.2121 | 0.1364 |
| fund_score | bull | 0.0380 | 0.1967 | 0.1930 | 0.1136 | 0.1075 |
| rsi_vol_combo | bull | 0.0298 | 0.1730 | 0.1722 | 0.1894 | 0.1024 |
| mom_x_lowvol_20_20 | bull | 0.0263 | 0.1828 | 0.1438 | 0.2045 | 0.0866 |
| momentum_reversal | bull | 0.0277 | 0.1887 | 0.1468 | 0.1591 | 0.0851 |
| momentum_reversal | bear | 0.1208 | 0.2291 | 0.5272 | 0.3425 | 0.3539 |
| mom_x_lowvol_20_20 | bear | 0.1208 | 0.2271 | 0.5319 | 0.2877 | 0.3425 |
| fund_pb | bear | 0.0490 | 0.1231 | 0.3981 | 0.1233 | 0.2236 |
| fund_gross_margin | bear | 0.0709 | 0.2058 | 0.3443 | 0.2055 | 0.2075 |
| trend_lowvol | bear | 0.1118 | 0.3105 | 0.3600 | 0.1507 | 0.2071 |
| fund_roe | bear | 0.0759 | 0.2646 | 0.2869 | 0.2603 | 0.1808 |
| wash_sale_score | bear | 0.0303 | 0.1202 | 0.2524 | 0.3269 | 0.1674 |
| rsi_vol_combo | bear | 0.0621 | 0.2168 | 0.2864 | 0.1507 | 0.1648 |
| fund_score | bear | 0.0475 | 0.2352 | 0.2020 | 0.2877 | 0.1300 |

### 体育产业

- **Neutral**: ['volatility', 'fund_pb', 'trend_lowvol'] (单因子IC=0.0876, 组合IC=0.1084)
  - weights: [0.338, 0.3334, 0.3285]
- **Bull**: ['volatility', 'trend_lowvol'] (单因子IC=0.1011, 组合IC=0.1185)
  - bull_weights: [0.5682, 0.4318]
- **Bear**: ['momentum_reversal'] (单因子IC=0.0823, 组合IC=0.0823)
  - bear_weights: [1.0]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| volatility | neutral | 0.0878 | 0.1954 | 0.4493 | 0.3967 | 0.3138 |
| fund_pb | neutral | 0.0722 | 0.1555 | 0.4647 | 0.3319 | 0.3095 |
| trend_lowvol | neutral | 0.1028 | 0.2295 | 0.4480 | 0.3612 | 0.3049 |
| momentum_reversal | neutral | 0.0759 | 0.2051 | 0.3700 | 0.3090 | 0.2421 |
| mom_x_lowvol_20_20 | neutral | 0.0725 | 0.1993 | 0.3639 | 0.2714 | 0.2313 |
| low_downside | neutral | 0.0485 | 0.1915 | 0.2534 | 0.2401 | 0.1571 |
| rsi_vol_combo | neutral | 0.0480 | 0.1987 | 0.2415 | 0.1921 | 0.1439 |
| fund_pe | neutral | 0.0370 | 0.1879 | 0.1971 | 0.1086 | 0.1092 |
| turnover_stability | neutral | 0.0255 | 0.1358 | 0.1877 | 0.1284 | 0.1059 |
| wash_sale_score | neutral | 0.0093 | 0.1477 | 0.0627 | 0.1079 | 0.0348 |
| volatility | bull | 0.1123 | 0.1810 | 0.6203 | 0.5303 | 0.4746 |
| trend_lowvol | bull | 0.0899 | 0.1803 | 0.4985 | 0.4470 | 0.3606 |
| fund_pb | bull | 0.0794 | 0.1548 | 0.5128 | 0.3333 | 0.3419 |
| low_downside | bull | 0.0765 | 0.1633 | 0.4688 | 0.3864 | 0.3249 |
| momentum_reversal | bull | 0.0450 | 0.1560 | 0.2885 | 0.1667 | 0.1683 |
| stroke_phase | bull | 0.0341 | 0.1382 | 0.2465 | 0.2083 | 0.1489 |
| mom_x_lowvol_20_20 | bull | 0.0341 | 0.1636 | 0.2085 | 0.2197 | 0.1272 |
| rsi_vol_combo | bull | 0.0261 | 0.1513 | 0.1729 | 0.1326 | 0.0979 |
| momentum_reversal | bear | 0.0823 | 0.1987 | 0.4141 | 0.3425 | 0.2779 |
| mom_x_lowvol_20_20 | bear | 0.0664 | 0.1733 | 0.3831 | 0.3425 | 0.2571 |
| fund_gross_margin | bear | 0.0358 | 0.1014 | 0.3527 | 0.1233 | 0.1981 |
| fund_pb | bear | 0.0428 | 0.1540 | 0.2777 | 0.1233 | 0.1560 |
| rsi_vol_combo | bear | 0.0377 | 0.1747 | 0.2158 | 0.2055 | 0.1301 |
| trend_lowvol | bear | 0.0513 | 0.2405 | 0.2133 | 0.2055 | 0.1285 |

### 信创

- **Neutral**: ['momentum_reversal', 'mom_x_lowvol_20_20', 'fund_pb'] (单因子IC=0.0925, 组合IC=0.123)
  - weights: [0.3543, 0.3252, 0.3205]
- **Bull**: ['volatility', 'fund_pb', 'low_downside'] (单因子IC=0.1176, 组合IC=0.1478)
  - bull_weights: [0.382, 0.3201, 0.2979]
- **Bear**: ['mom_x_lowvol_20_20', 'momentum_reversal', 'trend_lowvol'] (单因子IC=0.1115, 组合IC=0.125)
  - bear_weights: [0.3669, 0.3249, 0.3082]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| momentum_reversal | neutral | 0.0995 | 0.1484 | 0.6710 | 0.4948 | 0.5015 |
| mom_x_lowvol_20_20 | neutral | 0.0942 | 0.1484 | 0.6345 | 0.4509 | 0.4603 |
| fund_pb | neutral | 0.0838 | 0.1332 | 0.6290 | 0.4426 | 0.4537 |
| trend_lowvol | neutral | 0.0979 | 0.1734 | 0.5647 | 0.4154 | 0.3997 |
| rsi_vol_combo | neutral | 0.0774 | 0.1417 | 0.5463 | 0.4342 | 0.3917 |
| volatility | neutral | 0.0776 | 0.1690 | 0.4588 | 0.3758 | 0.3156 |
| fund_pe | neutral | 0.0559 | 0.1256 | 0.4449 | 0.3737 | 0.3056 |
| fund_profit_growth | neutral | 0.0472 | 0.1142 | 0.4135 | 0.3152 | 0.2719 |
| turnover_stability | neutral | 0.0391 | 0.1037 | 0.3768 | 0.2818 | 0.2415 |
| fund_score | neutral | 0.0515 | 0.1402 | 0.3670 | 0.2443 | 0.2283 |
| fund_roe | neutral | 0.0373 | 0.1452 | 0.2570 | 0.2109 | 0.1556 |
| low_downside | neutral | 0.0347 | 0.1611 | 0.2154 | 0.2129 | 0.1306 |
| fund_revenue_growth | neutral | 0.0203 | 0.1004 | 0.2024 | 0.1482 | 0.1162 |
| volatility | bull | 0.1365 | 0.1462 | 0.9332 | 0.6439 | 0.7671 |
| fund_pb | bull | 0.1151 | 0.1411 | 0.8157 | 0.5758 | 0.6427 |
| low_downside | bull | 0.1011 | 0.1326 | 0.7628 | 0.5682 | 0.5981 |
| trend_lowvol | bull | 0.1126 | 0.1626 | 0.6925 | 0.5530 | 0.5377 |
| fund_pe | bull | 0.0873 | 0.1257 | 0.6945 | 0.5227 | 0.5287 |
| mom_x_lowvol_20_20 | bull | 0.0965 | 0.1497 | 0.6447 | 0.4545 | 0.4689 |
| momentum_reversal | bull | 0.0849 | 0.1537 | 0.5520 | 0.4167 | 0.3910 |
| turnover_stability | bull | 0.0395 | 0.1021 | 0.3869 | 0.3333 | 0.2580 |
| fund_profit_growth | bull | 0.0361 | 0.1031 | 0.3502 | 0.1515 | 0.2016 |
| rsi_vol_combo | bull | 0.0436 | 0.1484 | 0.2935 | 0.1894 | 0.1745 |
| fund_roe | bull | 0.0286 | 0.1427 | 0.2005 | 0.1212 | 0.1124 |
| mom_x_lowvol_20_20 | bear | 0.1177 | 0.1595 | 0.7378 | 0.5342 | 0.5660 |
| momentum_reversal | bear | 0.1123 | 0.1595 | 0.7037 | 0.4247 | 0.5012 |
| trend_lowvol | bear | 0.1047 | 0.1598 | 0.6548 | 0.4521 | 0.4754 |
| fund_profit_growth | bear | 0.0723 | 0.1335 | 0.5419 | 0.4247 | 0.3860 |
| fund_score | bear | 0.0809 | 0.1554 | 0.5210 | 0.3973 | 0.3640 |
| rsi_vol_combo | bear | 0.0676 | 0.1443 | 0.4685 | 0.3973 | 0.3273 |
| fund_pe | bear | 0.0645 | 0.1570 | 0.4111 | 0.3151 | 0.2703 |
| fund_pb | bear | 0.0596 | 0.1688 | 0.3532 | 0.2329 | 0.2178 |
| fund_revenue_growth | bear | 0.0420 | 0.1186 | 0.3544 | 0.1781 | 0.2087 |
| fund_roe | bear | 0.0580 | 0.1791 | 0.3238 | 0.1507 | 0.1863 |
| fund_gross_margin | bear | 0.0248 | 0.1311 | 0.1891 | 0.1233 | 0.1062 |

### 储能概念

- **Neutral**: ['momentum_reversal', 'mom_x_lowvol_20_20', 'fund_profit_growth'] (单因子IC=0.0617, 组合IC=0.076)
  - weights: [0.3412, 0.3301, 0.3287]
- **Bull**: ['low_downside', 'volatility', 'fund_pb'] (单因子IC=0.0757, 组合IC=0.0896)
  - bull_weights: [0.4067, 0.3031, 0.2902]
- **Bear**: ['rsi_vol_combo', 'momentum_reversal', 'mom_x_lowvol_20_20'] (单因子IC=0.1159, 组合IC=0.1263)
  - bear_weights: [0.3758, 0.3414, 0.2828]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| momentum_reversal | neutral | 0.0701 | 0.1665 | 0.4208 | 0.3194 | 0.2776 |
| mom_x_lowvol_20_20 | neutral | 0.0676 | 0.1619 | 0.4176 | 0.2860 | 0.2685 |
| fund_profit_growth | neutral | 0.0474 | 0.1149 | 0.4125 | 0.2965 | 0.2674 |
| trend_lowvol | neutral | 0.0758 | 0.1895 | 0.4000 | 0.3048 | 0.2609 |
| fund_pb | neutral | 0.0656 | 0.1619 | 0.4054 | 0.2777 | 0.2590 |
| fund_pe | neutral | 0.0532 | 0.1524 | 0.3493 | 0.2422 | 0.2169 |
| volatility | neutral | 0.0657 | 0.1972 | 0.3331 | 0.2714 | 0.2118 |
| rsi_vol_combo | neutral | 0.0512 | 0.1526 | 0.3355 | 0.2610 | 0.2115 |
| fund_score | neutral | 0.0365 | 0.1458 | 0.2504 | 0.1399 | 0.1427 |
| turnover_stability | neutral | 0.0235 | 0.0972 | 0.2422 | 0.1712 | 0.1418 |
| low_downside | neutral | 0.0427 | 0.1888 | 0.2263 | 0.1733 | 0.1328 |
| fund_revenue_growth | neutral | 0.0165 | 0.1097 | 0.1503 | 0.1065 | 0.0832 |
| low_downside | bull | 0.0823 | 0.1432 | 0.5749 | 0.4545 | 0.4181 |
| volatility | bull | 0.0770 | 0.1648 | 0.4675 | 0.3333 | 0.3117 |
| fund_pb | bull | 0.0678 | 0.1540 | 0.4401 | 0.3561 | 0.2984 |
| fund_pe | bull | 0.0561 | 0.1425 | 0.3934 | 0.3258 | 0.2607 |
| turnover_stability | bull | 0.0316 | 0.0915 | 0.3457 | 0.2273 | 0.2121 |
| trend_lowvol | bull | 0.0355 | 0.1632 | 0.2173 | 0.1061 | 0.1202 |
| fund_profit_growth | bull | 0.0206 | 0.1178 | 0.1750 | 0.1818 | 0.1034 |
| relative_strength | bull | 0.0213 | 0.1440 | 0.1482 | 0.1667 | 0.0865 |
| rsi_vol_combo | bear | 0.0997 | 0.1377 | 0.7239 | 0.5616 | 0.5653 |
| momentum_reversal | bear | 0.1294 | 0.1864 | 0.6942 | 0.4795 | 0.5135 |
| mom_x_lowvol_20_20 | bear | 0.1186 | 0.2100 | 0.5647 | 0.5068 | 0.4254 |
| trend_lowvol | bear | 0.1012 | 0.2054 | 0.4927 | 0.3425 | 0.3307 |
| turnover_stability | bear | 0.0274 | 0.0882 | 0.3113 | 0.2055 | 0.1876 |
| bb_width_20 | bear | 0.0489 | 0.2161 | 0.2263 | 0.1781 | 0.1333 |

### 元宇宙概念

- **Neutral**: ['trend_lowvol', 'momentum_reversal'] (单因子IC=0.1101, 组合IC=0.1214)
  - weights: [0.5311, 0.4689]
- **Bull**: ['fund_pb', 'volatility', 'trend_lowvol'] (单因子IC=0.1035, 组合IC=0.1216)
  - bull_weights: [0.3661, 0.3436, 0.2903]
- **Bear**: ['turnover_stability', 'fund_profit_growth', 'trend_lowvol'] (单因子IC=0.0671, 组合IC=0.1019)
  - bear_weights: [0.373, 0.3328, 0.2941]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| trend_lowvol | neutral | 0.1208 | 0.2071 | 0.5833 | 0.4468 | 0.4219 |
| momentum_reversal | neutral | 0.0994 | 0.1829 | 0.5431 | 0.3716 | 0.3725 |
| mom_x_lowvol_20_20 | neutral | 0.0977 | 0.1858 | 0.5258 | 0.3800 | 0.3628 |
| fund_pb | neutral | 0.0801 | 0.1590 | 0.5037 | 0.3820 | 0.3480 |
| rsi_vol_combo | neutral | 0.0801 | 0.1689 | 0.4740 | 0.3445 | 0.3187 |
| volatility | neutral | 0.0958 | 0.2105 | 0.4550 | 0.3152 | 0.2992 |
| fund_profit_growth | neutral | 0.0403 | 0.1289 | 0.3127 | 0.2777 | 0.1998 |
| fund_score | neutral | 0.0467 | 0.1645 | 0.2838 | 0.2296 | 0.1745 |
| low_downside | neutral | 0.0516 | 0.1871 | 0.2756 | 0.2505 | 0.1723 |
| fund_pe | neutral | 0.0482 | 0.1693 | 0.2845 | 0.2046 | 0.1714 |
| turnover_stability | neutral | 0.0336 | 0.1194 | 0.2815 | 0.2109 | 0.1704 |
| fund_roe | neutral | 0.0372 | 0.1704 | 0.2183 | 0.1858 | 0.1294 |
| fund_revenue_growth | neutral | 0.0254 | 0.1279 | 0.1986 | 0.1837 | 0.1175 |
| fund_pb | bull | 0.0902 | 0.1367 | 0.6598 | 0.5076 | 0.4973 |
| volatility | bull | 0.1075 | 0.1701 | 0.6320 | 0.4773 | 0.4668 |
| trend_lowvol | bull | 0.1128 | 0.2025 | 0.5568 | 0.4167 | 0.3944 |
| low_downside | bull | 0.0796 | 0.1561 | 0.5097 | 0.3485 | 0.3437 |
| fund_profit_growth | bull | 0.0449 | 0.1183 | 0.3798 | 0.3106 | 0.2489 |
| mom_x_lowvol_20_20 | bull | 0.0604 | 0.1800 | 0.3356 | 0.2424 | 0.2085 |
| momentum_reversal | bull | 0.0574 | 0.1780 | 0.3225 | 0.2727 | 0.2052 |
| turnover_stability | bull | 0.0338 | 0.1104 | 0.3060 | 0.3106 | 0.2005 |
| fund_score | bull | 0.0477 | 0.1592 | 0.2996 | 0.1212 | 0.1680 |
| fund_roe | bull | 0.0414 | 0.1654 | 0.2503 | 0.2727 | 0.1593 |
| fund_pe | bull | 0.0382 | 0.1607 | 0.2375 | 0.2121 | 0.1440 |
| rsi_vol_combo | bull | 0.0352 | 0.1654 | 0.2129 | 0.1439 | 0.1218 |
| fund_gross_margin | bull | 0.0257 | 0.1193 | 0.2156 | 0.1288 | 0.1217 |
| wash_sale_score | bull | 0.0157 | 0.0939 | 0.1675 | 0.1615 | 0.0973 |
| fund_revenue_growth | bull | 0.0193 | 0.1281 | 0.1503 | 0.1439 | 0.0860 |
| stroke_phase | bull | 0.0146 | 0.1165 | 0.1250 | 0.1136 | 0.0696 |
| turnover_stability | bear | 0.0626 | 0.1176 | 0.5326 | 0.4521 | 0.3867 |
| fund_profit_growth | bear | 0.0708 | 0.1377 | 0.5140 | 0.3425 | 0.3450 |
| trend_lowvol | bear | 0.0678 | 0.1463 | 0.4637 | 0.3151 | 0.3049 |
| fund_revenue_growth | bear | 0.0546 | 0.1167 | 0.4677 | 0.2603 | 0.2947 |
| fund_score | bear | 0.0569 | 0.1450 | 0.3921 | 0.2877 | 0.2524 |
| mom_x_lowvol_20_20 | bear | 0.0609 | 0.1939 | 0.3141 | 0.2329 | 0.1936 |
| fund_gross_margin | bear | 0.0361 | 0.1239 | 0.2910 | 0.2055 | 0.1754 |
| momentum_reversal | bear | 0.0407 | 0.2125 | 0.1917 | 0.2603 | 0.1208 |

### 充电桩

- **Neutral**: ['mom_x_lowvol_20_20', 'momentum_reversal', 'trend_lowvol'] (单因子IC=0.0845, 组合IC=0.0937)
  - weights: [0.3579, 0.3363, 0.3058]
- **Bull**: ['low_downside', 'volatility', 'fund_pb'] (单因子IC=0.0897, 组合IC=0.1083)
  - bull_weights: [0.3999, 0.341, 0.2591]
- **Bear**: ['mom_x_lowvol_20_20'] (单因子IC=0.1164, 组合IC=0.1164)
  - bear_weights: [1.0]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| mom_x_lowvol_20_20 | neutral | 0.0832 | 0.1651 | 0.5037 | 0.3904 | 0.3502 |
| momentum_reversal | neutral | 0.0822 | 0.1711 | 0.4805 | 0.3695 | 0.3290 |
| trend_lowvol | neutral | 0.0882 | 0.1948 | 0.4528 | 0.3215 | 0.2992 |
| fund_profit_growth | neutral | 0.0537 | 0.1274 | 0.4211 | 0.3048 | 0.2747 |
| fund_pb | neutral | 0.0788 | 0.1828 | 0.4311 | 0.2651 | 0.2727 |
| fund_pe | neutral | 0.0646 | 0.1609 | 0.4016 | 0.2797 | 0.2570 |
| rsi_vol_combo | neutral | 0.0612 | 0.1611 | 0.3801 | 0.2902 | 0.2452 |
| volatility | neutral | 0.0789 | 0.2103 | 0.3750 | 0.2965 | 0.2431 |
| turnover_stability | neutral | 0.0370 | 0.0991 | 0.3728 | 0.2965 | 0.2417 |
| fund_score | neutral | 0.0462 | 0.1500 | 0.3080 | 0.1900 | 0.1833 |
| low_downside | neutral | 0.0511 | 0.1990 | 0.2569 | 0.1942 | 0.1534 |
| fund_revenue_growth | neutral | 0.0273 | 0.1169 | 0.2331 | 0.1253 | 0.1312 |
| low_downside | bull | 0.0918 | 0.1555 | 0.5908 | 0.4773 | 0.4364 |
| volatility | bull | 0.0971 | 0.1820 | 0.5338 | 0.3939 | 0.3720 |
| fund_pb | bull | 0.0802 | 0.1772 | 0.4524 | 0.2500 | 0.2827 |
| fund_pe | bull | 0.0634 | 0.1594 | 0.3980 | 0.3106 | 0.2608 |
| turnover_stability | bull | 0.0414 | 0.1055 | 0.3927 | 0.3030 | 0.2559 |
| trend_lowvol | bull | 0.0567 | 0.1689 | 0.3359 | 0.2576 | 0.2112 |
| stroke_phase | bull | 0.0180 | 0.0967 | 0.1857 | 0.1288 | 0.1048 |
| fund_gross_margin | bull | 0.0122 | 0.0887 | 0.1381 | 0.2197 | 0.0842 |
| relative_strength | bull | 0.0192 | 0.1583 | 0.1215 | 0.1667 | 0.0709 |
| mom_x_lowvol_20_20 | bear | 0.1164 | 0.2305 | 0.5051 | 0.3699 | 0.3460 |
| momentum_reversal | bear | 0.1061 | 0.2201 | 0.4818 | 0.3425 | 0.3234 |
| fund_profit_growth | bear | 0.0626 | 0.1300 | 0.4814 | 0.3425 | 0.3231 |
| rsi_vol_combo | bear | 0.0727 | 0.1684 | 0.4317 | 0.2877 | 0.2780 |
| trend_lowvol | bear | 0.0692 | 0.2254 | 0.3071 | 0.2055 | 0.1851 |

### 先进封装

- **Neutral**: ['trend_lowvol'] (单因子IC=0.0964, 组合IC=0.0964)
  - weights: [1.0]
- **Bull**: ['volatility', 'fund_pe', 'trend_lowvol'] (单因子IC=0.1049, 组合IC=0.1443)
  - bull_weights: [0.3813, 0.3122, 0.3065]
- **Bear**: ['mom_x_lowvol_20_20'] (单因子IC=0.1873, 组合IC=0.1873)
  - bear_weights: [1.0]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| trend_lowvol | neutral | 0.0964 | 0.2836 | 0.3401 | 0.2265 | 0.2085 |
| momentum_reversal | neutral | 0.0837 | 0.2746 | 0.3049 | 0.2693 | 0.1935 |
| mom_x_lowvol_20_20 | neutral | 0.0811 | 0.2784 | 0.2912 | 0.2912 | 0.1880 |
| rsi_vol_combo | neutral | 0.0762 | 0.2646 | 0.2881 | 0.2610 | 0.1816 |
| volatility | neutral | 0.0684 | 0.2616 | 0.2616 | 0.2067 | 0.1578 |
| fund_pb | neutral | 0.0540 | 0.2560 | 0.2109 | 0.1545 | 0.1217 |
| low_downside | neutral | 0.0348 | 0.2441 | 0.1424 | 0.1159 | 0.0795 |
| fund_revenue_growth | neutral | 0.0227 | 0.2249 | 0.1010 | 0.1054 | 0.0558 |
| volatility | bull | 0.1158 | 0.2447 | 0.4731 | 0.3636 | 0.3225 |
| fund_pe | bull | 0.0977 | 0.2537 | 0.3852 | 0.3712 | 0.2641 |
| trend_lowvol | bull | 0.1011 | 0.2556 | 0.3957 | 0.3106 | 0.2593 |
| fund_pb | bull | 0.0997 | 0.2749 | 0.3628 | 0.2727 | 0.2309 |
| turnover_stability | bull | 0.0635 | 0.1886 | 0.3365 | 0.2424 | 0.2090 |
| low_downside | bull | 0.0706 | 0.2141 | 0.3299 | 0.2500 | 0.2062 |
| fund_revenue_growth | bull | 0.0673 | 0.2380 | 0.2829 | 0.3333 | 0.1886 |
| mom_x_lowvol_20_20 | bull | 0.0591 | 0.2449 | 0.2413 | 0.1856 | 0.1430 |
| momentum_reversal | bull | 0.0506 | 0.2420 | 0.2090 | 0.1856 | 0.1239 |
| rsi_vol_combo | bull | 0.0295 | 0.2380 | 0.1240 | 0.1591 | 0.0719 |
| mom_x_lowvol_20_20 | bear | 0.1873 | 0.2580 | 0.7261 | 0.4795 | 0.5371 |
| momentum_reversal | bear | 0.1727 | 0.2612 | 0.6612 | 0.5616 | 0.5163 |
| bb_width_20 | bear | 0.1033 | 0.2211 | 0.4674 | 0.3973 | 0.3266 |
| trend_lowvol | bear | 0.0958 | 0.2039 | 0.4697 | 0.3425 | 0.3153 |
| rsi_vol_combo | bear | 0.0867 | 0.2167 | 0.4003 | 0.3973 | 0.2797 |
| fund_profit_growth | bear | 0.0454 | 0.1983 | 0.2290 | 0.1233 | 0.1286 |

### 光伏概念

- **Neutral**: ['volatility', 'trend_lowvol', 'momentum_reversal'] (单因子IC=0.0698, 组合IC=0.0885)
  - weights: [0.3388, 0.3317, 0.3295]
- **Bull**: ['low_downside', 'volatility', 'turnover_stability'] (单因子IC=0.0677, 组合IC=0.0882)
  - bull_weights: [0.3912, 0.3514, 0.2575]
- **Bear**: ['mom_x_lowvol_20_20'] (单因子IC=0.13, 组合IC=0.13)
  - bear_weights: [1.0]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| volatility | neutral | 0.0743 | 0.2042 | 0.3641 | 0.2756 | 0.2322 |
| trend_lowvol | neutral | 0.0713 | 0.1997 | 0.3570 | 0.2735 | 0.2273 |
| momentum_reversal | neutral | 0.0637 | 0.1772 | 0.3594 | 0.2568 | 0.2258 |
| mom_x_lowvol_20_20 | neutral | 0.0608 | 0.1701 | 0.3575 | 0.2484 | 0.2231 |
| fund_profit_growth | neutral | 0.0409 | 0.1207 | 0.3390 | 0.2714 | 0.2155 |
| fund_pb | neutral | 0.0665 | 0.1976 | 0.3365 | 0.2234 | 0.2058 |
| fund_pe | neutral | 0.0523 | 0.1638 | 0.3196 | 0.2046 | 0.1925 |
| turnover_stability | neutral | 0.0268 | 0.0990 | 0.2702 | 0.2422 | 0.1678 |
| rsi_vol_combo | neutral | 0.0406 | 0.1651 | 0.2460 | 0.1962 | 0.1471 |
| low_downside | neutral | 0.0477 | 0.1963 | 0.2430 | 0.1608 | 0.1410 |
| fund_score | neutral | 0.0335 | 0.1560 | 0.2145 | 0.1566 | 0.1240 |
| fund_revenue_growth | neutral | 0.0172 | 0.1111 | 0.1552 | 0.1190 | 0.0868 |
| low_downside | bull | 0.0789 | 0.1452 | 0.5436 | 0.3864 | 0.3768 |
| volatility | bull | 0.0856 | 0.1706 | 0.5020 | 0.3485 | 0.3384 |
| turnover_stability | bull | 0.0387 | 0.1051 | 0.3678 | 0.3485 | 0.2480 |
| fund_pe | bull | 0.0578 | 0.1501 | 0.3853 | 0.2576 | 0.2422 |
| trend_lowvol | bull | 0.0524 | 0.1570 | 0.3339 | 0.2576 | 0.2100 |
| fund_pb | bull | 0.0630 | 0.1901 | 0.3316 | 0.1818 | 0.1960 |
| exhaustion_risk | bull | 0.0154 | 0.0835 | 0.1846 | 0.1439 | 0.1056 |
| fund_profit_growth | bull | 0.0144 | 0.1135 | 0.1267 | 0.1894 | 0.0753 |
| fund_score | bull | 0.0185 | 0.1561 | 0.1184 | 0.1439 | 0.0677 |
| mom_x_lowvol_20_20 | bear | 0.1300 | 0.1912 | 0.6797 | 0.4247 | 0.4842 |
| momentum_reversal | bear | 0.1256 | 0.1872 | 0.6711 | 0.4247 | 0.4780 |
| rsi_vol_combo | bear | 0.0737 | 0.1469 | 0.5014 | 0.3151 | 0.3297 |
| trend_lowvol | bear | 0.0940 | 0.1968 | 0.4778 | 0.2329 | 0.2946 |
| fund_profit_growth | bear | 0.0369 | 0.1226 | 0.3011 | 0.2329 | 0.1856 |
| turnover_stability | bear | 0.0274 | 0.1010 | 0.2712 | 0.2603 | 0.1709 |

### 光刻机(胶)

- **Neutral**: ['mom_x_lowvol_20_20', 'momentum_reversal', 'trend_lowvol'] (单因子IC=0.0898, 组合IC=0.0971)
  - weights: [0.3465, 0.3417, 0.3118]
- **Bull**: ['volatility', 'low_downside', 'fund_pe'] (单因子IC=0.0849, 组合IC=0.113)
  - bull_weights: [0.3751, 0.357, 0.2679]
- **Bear**: ['momentum_reversal', 'mom_x_lowvol_20_20', 'fund_gross_margin'] (单因子IC=0.1112, 组合IC=0.1616)
  - bear_weights: [0.3815, 0.3531, 0.2654]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| mom_x_lowvol_20_20 | neutral | 0.0900 | 0.1965 | 0.4578 | 0.3674 | 0.3130 |
| momentum_reversal | neutral | 0.0916 | 0.1976 | 0.4635 | 0.3319 | 0.3087 |
| trend_lowvol | neutral | 0.0878 | 0.2085 | 0.4209 | 0.3382 | 0.2816 |
| rsi_vol_combo | neutral | 0.0704 | 0.1989 | 0.3541 | 0.2547 | 0.2221 |
| fund_pe | neutral | 0.0570 | 0.1862 | 0.3063 | 0.2651 | 0.1938 |
| volatility | neutral | 0.0613 | 0.2031 | 0.3018 | 0.2359 | 0.1865 |
| fund_pb | neutral | 0.0698 | 0.2326 | 0.3001 | 0.2338 | 0.1851 |
| fund_profit_growth | neutral | 0.0415 | 0.1636 | 0.2536 | 0.1649 | 0.1477 |
| fund_score | neutral | 0.0397 | 0.1929 | 0.2060 | 0.1336 | 0.1167 |
| turnover_stability | neutral | 0.0265 | 0.1427 | 0.1858 | 0.1378 | 0.1057 |
| low_downside | neutral | 0.0262 | 0.1814 | 0.1446 | 0.1169 | 0.0808 |
| volatility | bull | 0.0975 | 0.1764 | 0.5526 | 0.4015 | 0.3872 |
| low_downside | bull | 0.0869 | 0.1687 | 0.5148 | 0.4318 | 0.3686 |
| fund_pe | bull | 0.0704 | 0.1659 | 0.4246 | 0.3030 | 0.2766 |
| turnover_stability | bull | 0.0541 | 0.1430 | 0.3784 | 0.3258 | 0.2508 |
| mom_x_lowvol_20_20 | bull | 0.0646 | 0.1852 | 0.3489 | 0.3030 | 0.2273 |
| fund_pb | bull | 0.0706 | 0.2248 | 0.3139 | 0.3030 | 0.2045 |
| trend_lowvol | bull | 0.0614 | 0.2020 | 0.3042 | 0.2273 | 0.1867 |
| fund_profit_growth | bull | 0.0411 | 0.1383 | 0.2973 | 0.2045 | 0.1790 |
| fund_score | bull | 0.0371 | 0.1464 | 0.2535 | 0.1364 | 0.1440 |
| momentum_reversal | bull | 0.0445 | 0.1916 | 0.2321 | 0.2121 | 0.1407 |
| rsi_vol_combo | bull | 0.0348 | 0.1744 | 0.1995 | 0.1970 | 0.1194 |
| momentum_reversal | bear | 0.1190 | 0.1664 | 0.7153 | 0.5616 | 0.5585 |
| mom_x_lowvol_20_20 | bear | 0.1214 | 0.1802 | 0.6738 | 0.5342 | 0.5169 |
| fund_gross_margin | bear | 0.0931 | 0.1674 | 0.5563 | 0.3973 | 0.3886 |
| rsi_vol_combo | bear | 0.0926 | 0.1698 | 0.5455 | 0.3699 | 0.3737 |
| trend_lowvol | bear | 0.1081 | 0.2209 | 0.4893 | 0.2329 | 0.3016 |
| fund_profit_growth | bear | 0.0733 | 0.2022 | 0.3628 | 0.3151 | 0.2385 |
| fund_revenue_growth | bear | 0.0468 | 0.1438 | 0.3254 | 0.2329 | 0.2006 |
| fund_score | bear | 0.0576 | 0.2180 | 0.2640 | 0.3151 | 0.1736 |

### 光纤概念

- **Neutral**: ['mom_x_lowvol_20_20', 'momentum_reversal', 'trend_lowvol'] (单因子IC=0.0916, 组合IC=0.1017)
  - weights: [0.366, 0.3369, 0.2971]
- **Bull**: ['fund_profit_growth', 'low_downside'] (单因子IC=0.0641, 组合IC=0.0861)
  - bull_weights: [0.5051, 0.4949]
- **Bear**: ['mom_x_lowvol_20_20'] (单因子IC=0.1355, 组合IC=0.1355)
  - bear_weights: [1.0]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| mom_x_lowvol_20_20 | neutral | 0.0949 | 0.2174 | 0.4365 | 0.2923 | 0.2820 |
| momentum_reversal | neutral | 0.0912 | 0.2255 | 0.4045 | 0.2839 | 0.2597 |
| trend_lowvol | neutral | 0.0886 | 0.2403 | 0.3687 | 0.2422 | 0.2290 |
| rsi_vol_combo | neutral | 0.0663 | 0.2128 | 0.3118 | 0.2484 | 0.1946 |
| fund_pb | neutral | 0.0508 | 0.2208 | 0.2300 | 0.1848 | 0.1362 |
| fund_profit_growth | neutral | 0.0373 | 0.1656 | 0.2251 | 0.1775 | 0.1325 |
| turnover_stability | neutral | 0.0325 | 0.1627 | 0.1997 | 0.1420 | 0.1140 |
| fund_pe | neutral | 0.0369 | 0.2058 | 0.1793 | 0.1566 | 0.1037 |
| fund_revenue_growth | neutral | 0.0276 | 0.1612 | 0.1710 | 0.1482 | 0.0982 |
| volatility | neutral | 0.0387 | 0.2570 | 0.1505 | 0.1273 | 0.0848 |
| fund_score | neutral | 0.0296 | 0.2016 | 0.1468 | 0.1169 | 0.0820 |
| fund_profit_growth | bull | 0.0527 | 0.1558 | 0.3382 | 0.3030 | 0.2203 |
| low_downside | bull | 0.0756 | 0.2215 | 0.3412 | 0.2652 | 0.2159 |
| fund_score | bull | 0.0481 | 0.1813 | 0.2655 | 0.2424 | 0.1649 |
| fund_revenue_growth | bull | 0.0422 | 0.1697 | 0.2487 | 0.1818 | 0.1470 |
| volatility | bull | 0.0535 | 0.2254 | 0.2376 | 0.2045 | 0.1431 |
| fund_pe | bull | 0.0416 | 0.1970 | 0.2114 | 0.2424 | 0.1313 |
| fund_pb | bull | 0.0355 | 0.2487 | 0.1429 | 0.1439 | 0.0817 |
| top_fractal_volume | bull | 0.0164 | 0.1487 | 0.1102 | 0.1504 | 0.0634 |
| mom_x_lowvol_20_20 | bear | 0.1355 | 0.2157 | 0.6283 | 0.3973 | 0.4389 |
| momentum_reversal | bear | 0.0963 | 0.2026 | 0.4753 | 0.3425 | 0.3190 |
| trend_lowvol | bear | 0.0853 | 0.2168 | 0.3933 | 0.3151 | 0.2586 |
| bb_width_20 | bear | 0.0558 | 0.1802 | 0.3098 | 0.3425 | 0.2080 |
| rsi_vol_combo | bear | 0.0465 | 0.1820 | 0.2556 | 0.2603 | 0.1611 |
| fund_revenue_growth | bear | 0.0343 | 0.1597 | 0.2148 | 0.1233 | 0.1206 |

### 光通信模块

- **Neutral**: ['fund_pe', 'mom_x_lowvol_20_20', 'momentum_reversal'] (单因子IC=0.0741, 组合IC=0.0913)
  - weights: [0.3479, 0.334, 0.3181]
- **Bull**: ['fund_pe'] (单因子IC=0.0725, 组合IC=0.0724)
  - bull_weights: [1.0]
- **Bear**: ['mom_x_lowvol_20_20', 'rsi_vol_combo'] (单因子IC=0.1134, 组合IC=0.1213)
  - bear_weights: [0.5106, 0.4894]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| fund_pe | neutral | 0.0647 | 0.1718 | 0.3769 | 0.3069 | 0.2463 |
| mom_x_lowvol_20_20 | neutral | 0.0783 | 0.2088 | 0.3749 | 0.2610 | 0.2364 |
| momentum_reversal | neutral | 0.0794 | 0.2185 | 0.3631 | 0.2401 | 0.2251 |
| trend_lowvol | neutral | 0.0801 | 0.2228 | 0.3595 | 0.2276 | 0.2207 |
| rsi_vol_combo | neutral | 0.0647 | 0.2014 | 0.3213 | 0.2505 | 0.2009 |
| fund_profit_growth | neutral | 0.0509 | 0.1590 | 0.3199 | 0.2401 | 0.1983 |
| volatility | neutral | 0.0627 | 0.2193 | 0.2860 | 0.2129 | 0.1735 |
| fund_pb | neutral | 0.0615 | 0.2291 | 0.2685 | 0.1461 | 0.1539 |
| fund_score | neutral | 0.0399 | 0.1903 | 0.2095 | 0.1858 | 0.1242 |
| fund_roe | neutral | 0.0377 | 0.2005 | 0.1880 | 0.1461 | 0.1077 |
| turnover_stability | neutral | 0.0244 | 0.1569 | 0.1556 | 0.1608 | 0.0903 |
| fund_revenue_growth | neutral | 0.0221 | 0.1535 | 0.1440 | 0.1106 | 0.0800 |
| fund_pe | bull | 0.0725 | 0.1695 | 0.4276 | 0.3864 | 0.2964 |
| volatility | bull | 0.0569 | 0.1819 | 0.3128 | 0.2424 | 0.1943 |
| fund_pb | bull | 0.0713 | 0.2262 | 0.3153 | 0.2273 | 0.1935 |
| low_downside | bull | 0.0534 | 0.1952 | 0.2738 | 0.2879 | 0.1763 |
| fund_revenue_growth | bull | 0.0369 | 0.1468 | 0.2515 | 0.1818 | 0.1486 |
| fund_profit_growth | bull | 0.0413 | 0.1664 | 0.2481 | 0.1894 | 0.1475 |
| fund_score | bull | 0.0373 | 0.1782 | 0.2090 | 0.3030 | 0.1362 |
| mom_x_lowvol_20_20 | bull | 0.0240 | 0.1442 | 0.1662 | 0.1742 | 0.0976 |
| fund_roe | bull | 0.0199 | 0.1751 | 0.1136 | 0.1439 | 0.0650 |
| mom_x_lowvol_20_20 | bear | 0.1211 | 0.1687 | 0.7177 | 0.4795 | 0.5309 |
| rsi_vol_combo | bear | 0.1057 | 0.1621 | 0.6518 | 0.5616 | 0.5089 |
| momentum_reversal | bear | 0.1021 | 0.1615 | 0.6320 | 0.3425 | 0.4242 |
| bb_width_20 | bear | 0.0549 | 0.1519 | 0.3612 | 0.1507 | 0.2078 |
| fund_gross_margin | bear | 0.0594 | 0.1926 | 0.3086 | 0.1507 | 0.1775 |

### 免疫治疗

- **Neutral**: ['fund_pb', 'momentum_reversal'] (单因子IC=0.0736, 组合IC=0.1026)
  - weights: [0.5362, 0.4638]
- **Bull**: ['turnover_stability', 'low_downside'] (单因子IC=0.0647, 组合IC=0.07)
  - bull_weights: [0.503, 0.497]
- **Bear**: ['trend_lowvol', 'fund_revenue_growth', 'momentum_reversal'] (单因子IC=0.1303, 组合IC=0.1692)
  - bear_weights: [0.3622, 0.3199, 0.3179]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| fund_pb | neutral | 0.0764 | 0.2311 | 0.3307 | 0.2380 | 0.2047 |
| momentum_reversal | neutral | 0.0707 | 0.2399 | 0.2949 | 0.2004 | 0.1770 |
| volatility | neutral | 0.0556 | 0.2213 | 0.2514 | 0.2651 | 0.1590 |
| mom_x_lowvol_20_20 | neutral | 0.0582 | 0.2246 | 0.2590 | 0.1816 | 0.1530 |
| fund_pe | neutral | 0.0536 | 0.2126 | 0.2522 | 0.1900 | 0.1500 |
| trend_lowvol | neutral | 0.0633 | 0.2653 | 0.2385 | 0.1597 | 0.1383 |
| rsi_vol_combo | neutral | 0.0491 | 0.2275 | 0.2157 | 0.1649 | 0.1256 |
| fund_profit_growth | neutral | 0.0411 | 0.2026 | 0.2026 | 0.1858 | 0.1202 |
| fund_score | neutral | 0.0382 | 0.2090 | 0.1827 | 0.1524 | 0.1053 |
| low_downside | neutral | 0.0359 | 0.2339 | 0.1537 | 0.1670 | 0.0897 |
| turnover_stability | bull | 0.0594 | 0.2002 | 0.2964 | 0.1970 | 0.1774 |
| low_downside | bull | 0.0700 | 0.2391 | 0.2930 | 0.1970 | 0.1753 |
| fund_pe | bull | 0.0483 | 0.1839 | 0.2625 | 0.1818 | 0.1551 |
| trend_lowvol | bull | 0.0392 | 0.2085 | 0.1879 | 0.1667 | 0.1096 |
| fund_revenue_growth | bull | 0.0290 | 0.1591 | 0.1820 | 0.1591 | 0.1055 |
| fund_pb | bull | 0.0386 | 0.2097 | 0.1841 | 0.1402 | 0.1049 |
| stroke_phase | bull | 0.0316 | 0.1856 | 0.1703 | 0.1515 | 0.0980 |
| momentum_reversal | bull | 0.0376 | 0.2216 | 0.1696 | 0.1288 | 0.0957 |
| volatility | bull | 0.0329 | 0.2243 | 0.1469 | 0.1288 | 0.0829 |
| fund_roe | bull | 0.0235 | 0.1730 | 0.1360 | 0.1288 | 0.0768 |
| rsi_vol_combo | bull | 0.0228 | 0.1901 | 0.1197 | 0.1591 | 0.0694 |
| trend_lowvol | bear | 0.1530 | 0.2134 | 0.7169 | 0.5342 | 0.5499 |
| fund_revenue_growth | bear | 0.1099 | 0.1611 | 0.6819 | 0.4247 | 0.4857 |
| momentum_reversal | bear | 0.1280 | 0.2088 | 0.6128 | 0.5753 | 0.4827 |
| fund_score | bear | 0.1155 | 0.2132 | 0.5419 | 0.4247 | 0.3860 |
| mom_x_lowvol_20_20 | bear | 0.1026 | 0.2150 | 0.4773 | 0.4521 | 0.3465 |
| fund_profit_growth | bear | 0.0854 | 0.2070 | 0.4126 | 0.4521 | 0.2996 |
| fund_roe | bear | 0.0794 | 0.1873 | 0.4238 | 0.3699 | 0.2903 |
| fund_pe | bear | 0.0779 | 0.2023 | 0.3848 | 0.1781 | 0.2267 |
| rsi_vol_combo | bear | 0.0764 | 0.2241 | 0.3410 | 0.2329 | 0.2102 |
| bb_width_20 | bear | 0.0597 | 0.1937 | 0.3083 | 0.2877 | 0.1985 |
| turnover_stability | bear | 0.0518 | 0.1701 | 0.3047 | 0.2877 | 0.1962 |
| fund_gross_margin | bear | 0.0645 | 0.2483 | 0.2597 | 0.1781 | 0.1530 |

### 养老概念

- **Neutral**: ['trend_lowvol', 'mom_x_lowvol_20_20'] (单因子IC=0.0845, 组合IC=0.1003)
  - weights: [0.5484, 0.4516]
- **Bull**: ['low_downside', 'volatility', 'trend_lowvol'] (单因子IC=0.0852, 组合IC=0.0942)
  - bull_weights: [0.3809, 0.3206, 0.2986]
- **Bear**: ['trend_lowvol'] (单因子IC=0.1366, 组合IC=0.1366)
  - bear_weights: [1.0]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| trend_lowvol | neutral | 0.0966 | 0.2077 | 0.4651 | 0.4029 | 0.3263 |
| mom_x_lowvol_20_20 | neutral | 0.0723 | 0.1755 | 0.4119 | 0.3048 | 0.2687 |
| momentum_reversal | neutral | 0.0725 | 0.1870 | 0.3875 | 0.2860 | 0.2492 |
| turnover_stability | neutral | 0.0430 | 0.1157 | 0.3715 | 0.3132 | 0.2439 |
| fund_pb | neutral | 0.0660 | 0.1824 | 0.3619 | 0.2380 | 0.2240 |
| volatility | neutral | 0.0665 | 0.2053 | 0.3238 | 0.3132 | 0.2126 |
| low_downside | neutral | 0.0607 | 0.1976 | 0.3070 | 0.2735 | 0.1955 |
| fund_profit_growth | neutral | 0.0347 | 0.1267 | 0.2735 | 0.2150 | 0.1661 |
| rsi_vol_combo | neutral | 0.0468 | 0.1864 | 0.2513 | 0.1816 | 0.1485 |
| fund_score | neutral | 0.0324 | 0.1505 | 0.2155 | 0.1775 | 0.1269 |
| fund_pe | neutral | 0.0410 | 0.2127 | 0.1929 | 0.1628 | 0.1122 |
| low_downside | bull | 0.0884 | 0.1567 | 0.5639 | 0.4091 | 0.3973 |
| volatility | bull | 0.0850 | 0.1772 | 0.4797 | 0.3939 | 0.3344 |
| trend_lowvol | bull | 0.0821 | 0.1748 | 0.4698 | 0.3258 | 0.3114 |
| momentum_reversal | bull | 0.0758 | 0.1766 | 0.4289 | 0.2424 | 0.2664 |
| turnover_stability | bull | 0.0401 | 0.1147 | 0.3496 | 0.2348 | 0.2158 |
| fund_profit_growth | bull | 0.0395 | 0.1187 | 0.3333 | 0.2576 | 0.2096 |
| fund_pb | bull | 0.0528 | 0.1806 | 0.2923 | 0.2348 | 0.1805 |
| rsi_vol_combo | bull | 0.0495 | 0.1715 | 0.2888 | 0.2121 | 0.1750 |
| mom_x_lowvol_20_20 | bull | 0.0452 | 0.1715 | 0.2634 | 0.1439 | 0.1506 |
| stroke_phase | bull | 0.0239 | 0.1245 | 0.1923 | 0.2045 | 0.1158 |
| fund_score | bull | 0.0233 | 0.1477 | 0.1574 | 0.1364 | 0.0895 |
| fund_pe | bull | 0.0291 | 0.2190 | 0.1329 | 0.1061 | 0.0735 |
| trend_lowvol | bear | 0.1366 | 0.2260 | 0.6045 | 0.3973 | 0.4223 |
| momentum_reversal | bear | 0.1117 | 0.2482 | 0.4499 | 0.2055 | 0.2712 |
| mom_x_lowvol_20_20 | bear | 0.1015 | 0.2513 | 0.4038 | 0.2329 | 0.2489 |
| fund_revenue_growth | bear | 0.0473 | 0.1452 | 0.3258 | 0.1233 | 0.1830 |
| bb_width_20 | bear | 0.0629 | 0.2429 | 0.2588 | 0.1781 | 0.1524 |
| rsi_vol_combo | bear | 0.0414 | 0.2232 | 0.1856 | 0.1507 | 0.1068 |

### 养老金

- **Neutral**: ['fund_pb', 'fund_pe', 'fund_profit_growth'] (单因子IC=0.0596, 组合IC=0.084)
  - weights: [0.3813, 0.3805, 0.2382]
- **Bull**: ['fund_pb', 'low_downside'] (单因子IC=0.0538, 组合IC=0.0688)
  - bull_weights: [0.6027, 0.3973]
- **Bear**: ['bb_width_20', 'momentum_reversal'] (单因子IC=0.0864, 组合IC=0.1022)
  - bear_weights: [0.5078, 0.4922]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| fund_pb | neutral | 0.0701 | 0.1891 | 0.3707 | 0.2610 | 0.2337 |
| fund_pe | neutral | 0.0748 | 0.2045 | 0.3657 | 0.2756 | 0.2333 |
| fund_profit_growth | neutral | 0.0339 | 0.1376 | 0.2463 | 0.1858 | 0.1460 |
| trend_lowvol | neutral | 0.0464 | 0.1997 | 0.2326 | 0.1545 | 0.1343 |
| fund_revenue_growth | neutral | 0.0237 | 0.1188 | 0.1998 | 0.2025 | 0.1201 |
| fund_score | neutral | 0.0318 | 0.1543 | 0.2058 | 0.1670 | 0.1201 |
| volatility | neutral | 0.0394 | 0.1970 | 0.1999 | 0.1670 | 0.1167 |
| mom_x_lowvol_20_20 | neutral | 0.0303 | 0.1679 | 0.1803 | 0.1002 | 0.0992 |
| momentum_reversal | neutral | 0.0304 | 0.1746 | 0.1739 | 0.1211 | 0.0975 |
| turnover_stability | neutral | 0.0160 | 0.1021 | 0.1566 | 0.1273 | 0.0883 |
| fund_pb | bull | 0.0681 | 0.1787 | 0.3815 | 0.2197 | 0.2326 |
| low_downside | bull | 0.0395 | 0.1530 | 0.2579 | 0.1894 | 0.1534 |
| volatility | bull | 0.0391 | 0.1919 | 0.2039 | 0.1591 | 0.1182 |
| fund_pe | bull | 0.0346 | 0.1873 | 0.1849 | 0.1136 | 0.1030 |
| turnover_stability | bull | 0.0173 | 0.1037 | 0.1672 | 0.1136 | 0.0931 |
| top_fractal_volume | bull | 0.0115 | 0.0894 | 0.1288 | 0.1473 | 0.0739 |
| bb_width_20 | bear | 0.0858 | 0.2235 | 0.3838 | 0.3151 | 0.2523 |
| momentum_reversal | bear | 0.0870 | 0.2242 | 0.3881 | 0.2603 | 0.2445 |
| mom_x_lowvol_20_20 | bear | 0.0921 | 0.2290 | 0.4020 | 0.2055 | 0.2423 |
| fund_gross_margin | bear | 0.0390 | 0.1052 | 0.3708 | 0.1233 | 0.2083 |
| trend_lowvol | bear | 0.0827 | 0.2370 | 0.3490 | 0.1233 | 0.1960 |
| rsi_vol_combo | bear | 0.0350 | 0.1760 | 0.1989 | 0.2055 | 0.1199 |

### 内贸流通

- **Neutral**: ['fund_pb', 'trend_lowvol', 'turnover_stability'] (单因子IC=0.0821, 组合IC=0.1241)
  - weights: [0.3661, 0.3272, 0.3068]
- **Bull**: ['low_downside', 'turnover_stability', 'momentum_reversal'] (单因子IC=0.0907, 组合IC=0.1427)
  - bull_weights: [0.3461, 0.336, 0.3179]
- **Bear**: ['turnover_stability', 'fund_profit_growth', 'momentum_reversal'] (单因子IC=0.0671, 组合IC=0.1052)
  - bear_weights: [0.4325, 0.2924, 0.2751]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| fund_pb | neutral | 0.0790 | 0.1258 | 0.6277 | 0.4718 | 0.4619 |
| trend_lowvol | neutral | 0.1053 | 0.1838 | 0.5731 | 0.4405 | 0.4128 |
| turnover_stability | neutral | 0.0621 | 0.1148 | 0.5405 | 0.4322 | 0.3871 |
| mom_x_lowvol_20_20 | neutral | 0.0817 | 0.1644 | 0.4971 | 0.4029 | 0.3487 |
| momentum_reversal | neutral | 0.0887 | 0.1788 | 0.4962 | 0.3549 | 0.3362 |
| volatility | neutral | 0.0677 | 0.1652 | 0.4096 | 0.3987 | 0.2865 |
| low_downside | neutral | 0.0621 | 0.1685 | 0.3689 | 0.3486 | 0.2488 |
| rsi_vol_combo | neutral | 0.0610 | 0.1770 | 0.3448 | 0.2296 | 0.2120 |
| fund_pe | neutral | 0.0510 | 0.1616 | 0.3159 | 0.2192 | 0.1925 |
| fund_profit_growth | neutral | 0.0300 | 0.1124 | 0.2669 | 0.1858 | 0.1583 |
| fund_score | neutral | 0.0239 | 0.1469 | 0.1626 | 0.1294 | 0.0918 |
| low_downside | bull | 0.1003 | 0.1400 | 0.7165 | 0.5530 | 0.5564 |
| turnover_stability | bull | 0.0786 | 0.1086 | 0.7239 | 0.4924 | 0.5402 |
| momentum_reversal | bull | 0.0930 | 0.1393 | 0.6679 | 0.5303 | 0.5110 |
| volatility | bull | 0.0786 | 0.1428 | 0.5505 | 0.3712 | 0.3775 |
| trend_lowvol | bull | 0.0902 | 0.1740 | 0.5186 | 0.4167 | 0.3673 |
| rsi_vol_combo | bull | 0.0675 | 0.1422 | 0.4746 | 0.3561 | 0.3218 |
| mom_x_lowvol_20_20 | bull | 0.0523 | 0.1326 | 0.3943 | 0.3182 | 0.2599 |
| fund_pb | bull | 0.0549 | 0.1409 | 0.3896 | 0.3333 | 0.2597 |
| fund_pe | bull | 0.0413 | 0.1634 | 0.2527 | 0.1061 | 0.1398 |
| turnover_stability | bear | 0.0582 | 0.1020 | 0.5712 | 0.3699 | 0.3912 |
| fund_profit_growth | bear | 0.0609 | 0.1324 | 0.4598 | 0.1507 | 0.2645 |
| momentum_reversal | bear | 0.0822 | 0.2173 | 0.3784 | 0.3151 | 0.2488 |
| fund_score | bear | 0.0825 | 0.1960 | 0.4210 | 0.1507 | 0.2422 |
| fund_pe | bear | 0.0598 | 0.1670 | 0.3581 | 0.3151 | 0.2355 |
| fund_pb | bear | 0.0489 | 0.1467 | 0.3334 | 0.2055 | 0.2010 |
| mom_x_lowvol_20_20 | bear | 0.0651 | 0.1996 | 0.3263 | 0.2055 | 0.1967 |
| fund_revenue_growth | bear | 0.0547 | 0.1624 | 0.3366 | 0.1507 | 0.1937 |
| rsi_vol_combo | bear | 0.0592 | 0.2062 | 0.2870 | 0.2603 | 0.1808 |
| trend_lowvol | bear | 0.0539 | 0.1938 | 0.2783 | 0.1233 | 0.1563 |
| volatility | bear | 0.0388 | 0.1688 | 0.2302 | 0.1233 | 0.1293 |

### 军工

- **Neutral**: ['mom_x_lowvol_20_20', 'momentum_reversal', 'trend_lowvol'] (单因子IC=0.0915, 组合IC=0.1017)
  - weights: [0.352, 0.3461, 0.3019]
- **Bull**: ['volatility', 'low_downside', 'trend_lowvol'] (单因子IC=0.109, 组合IC=0.1228)
  - bull_weights: [0.3467, 0.3412, 0.312]
- **Bear**: ['rsi_vol_combo', 'momentum_reversal', 'turnover_stability'] (单因子IC=0.0879, 组合IC=0.1176)
  - bear_weights: [0.3517, 0.3371, 0.3112]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| mom_x_lowvol_20_20 | neutral | 0.0915 | 0.1446 | 0.6328 | 0.4718 | 0.4657 |
| momentum_reversal | neutral | 0.0936 | 0.1487 | 0.6293 | 0.4551 | 0.4579 |
| trend_lowvol | neutral | 0.0894 | 0.1580 | 0.5660 | 0.4113 | 0.3994 |
| rsi_vol_combo | neutral | 0.0735 | 0.1422 | 0.5166 | 0.3883 | 0.3586 |
| fund_pb | neutral | 0.0781 | 0.1576 | 0.4958 | 0.3779 | 0.3416 |
| volatility | neutral | 0.0735 | 0.1652 | 0.4451 | 0.3633 | 0.3034 |
| turnover_stability | neutral | 0.0369 | 0.0889 | 0.4153 | 0.3466 | 0.2796 |
| fund_pe | neutral | 0.0546 | 0.1357 | 0.4020 | 0.2797 | 0.2572 |
| fund_profit_growth | neutral | 0.0405 | 0.1156 | 0.3501 | 0.2944 | 0.2266 |
| fund_revenue_growth | neutral | 0.0256 | 0.1028 | 0.2487 | 0.1628 | 0.1446 |
| fund_score | neutral | 0.0370 | 0.1518 | 0.2437 | 0.1566 | 0.1409 |
| low_downside | neutral | 0.0342 | 0.1604 | 0.2133 | 0.1816 | 0.1260 |
| volatility | bull | 0.1169 | 0.1574 | 0.7425 | 0.5682 | 0.5822 |
| low_downside | bull | 0.1015 | 0.1362 | 0.7451 | 0.5379 | 0.5730 |
| trend_lowvol | bull | 0.1087 | 0.1563 | 0.6951 | 0.5076 | 0.5240 |
| fund_pb | bull | 0.1067 | 0.1545 | 0.6906 | 0.4773 | 0.5101 |
| fund_pe | bull | 0.0729 | 0.1602 | 0.4550 | 0.3258 | 0.3016 |
| mom_x_lowvol_20_20 | bull | 0.0607 | 0.1410 | 0.4308 | 0.3333 | 0.2872 |
| momentum_reversal | bull | 0.0584 | 0.1450 | 0.4027 | 0.2955 | 0.2608 |
| rsi_vol_combo | bull | 0.0408 | 0.1311 | 0.3109 | 0.1515 | 0.1790 |
| fund_profit_growth | bull | 0.0141 | 0.1073 | 0.1315 | 0.1364 | 0.0747 |
| turnover_stability | bull | 0.0107 | 0.0856 | 0.1253 | 0.1061 | 0.0693 |
| rsi_vol_combo | bear | 0.0953 | 0.1282 | 0.7436 | 0.5342 | 0.5704 |
| momentum_reversal | bear | 0.1197 | 0.1649 | 0.7257 | 0.5068 | 0.5468 |
| turnover_stability | bear | 0.0487 | 0.0714 | 0.6822 | 0.4795 | 0.5046 |
| mom_x_lowvol_20_20 | bear | 0.1121 | 0.1693 | 0.6623 | 0.5068 | 0.4990 |
| trend_lowvol | bear | 0.0931 | 0.1967 | 0.4735 | 0.2329 | 0.2919 |
| fund_revenue_growth | bear | 0.0457 | 0.1321 | 0.3460 | 0.2603 | 0.2180 |
| bb_width_20 | bear | 0.0530 | 0.1735 | 0.3055 | 0.1507 | 0.1758 |
| fund_profit_growth | bear | 0.0428 | 0.1508 | 0.2838 | 0.1781 | 0.1671 |
| fund_gross_margin | bear | 0.0509 | 0.1970 | 0.2582 | 0.1233 | 0.1450 |

### 军民融合

- **Neutral**: ['momentum_reversal', 'mom_x_lowvol_20_20', 'trend_lowvol'] (单因子IC=0.0831, 组合IC=0.0932)
  - weights: [0.3579, 0.3576, 0.2845]
- **Bull**: ['low_downside', 'volatility', 'fund_pb'] (单因子IC=0.1227, 组合IC=0.1504)
  - bull_weights: [0.3499, 0.3431, 0.307]
- **Bear**: ['momentum_reversal'] (单因子IC=0.1275, 组合IC=0.1275)
  - bear_weights: [1.0]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| momentum_reversal | neutral | 0.0875 | 0.1650 | 0.5305 | 0.3883 | 0.3683 |
| mom_x_lowvol_20_20 | neutral | 0.0847 | 0.1586 | 0.5341 | 0.3779 | 0.3680 |
| trend_lowvol | neutral | 0.0770 | 0.1750 | 0.4397 | 0.3319 | 0.2928 |
| rsi_vol_combo | neutral | 0.0693 | 0.1618 | 0.4285 | 0.3257 | 0.2840 |
| turnover_stability | neutral | 0.0425 | 0.1067 | 0.3986 | 0.3132 | 0.2617 |
| fund_pb | neutral | 0.0703 | 0.1847 | 0.3807 | 0.2735 | 0.2424 |
| volatility | neutral | 0.0664 | 0.1790 | 0.3711 | 0.3006 | 0.2413 |
| fund_pe | neutral | 0.0483 | 0.1415 | 0.3412 | 0.2693 | 0.2166 |
| fund_profit_growth | neutral | 0.0441 | 0.1374 | 0.3210 | 0.2380 | 0.1987 |
| fund_score | neutral | 0.0363 | 0.1651 | 0.2198 | 0.1733 | 0.1289 |
| fund_revenue_growth | neutral | 0.0247 | 0.1226 | 0.2016 | 0.1628 | 0.1172 |
| low_downside | neutral | 0.0291 | 0.1718 | 0.1696 | 0.1670 | 0.0990 |
| low_downside | bull | 0.1214 | 0.1572 | 0.7720 | 0.5076 | 0.5819 |
| volatility | bull | 0.1314 | 0.1780 | 0.7383 | 0.5455 | 0.5705 |
| fund_pb | bull | 0.1153 | 0.1643 | 0.7019 | 0.4545 | 0.5105 |
| trend_lowvol | bull | 0.1078 | 0.1637 | 0.6585 | 0.4697 | 0.4839 |
| fund_pe | bull | 0.0676 | 0.1584 | 0.4268 | 0.2273 | 0.2619 |
| mom_x_lowvol_20_20 | bull | 0.0577 | 0.1520 | 0.3797 | 0.2424 | 0.2359 |
| momentum_reversal | bull | 0.0551 | 0.1560 | 0.3530 | 0.1970 | 0.2113 |
| fund_profit_growth | bull | 0.0152 | 0.1316 | 0.1158 | 0.1364 | 0.0658 |
| momentum_reversal | bear | 0.1275 | 0.1721 | 0.7404 | 0.4247 | 0.5274 |
| rsi_vol_combo | bear | 0.0872 | 0.1371 | 0.6357 | 0.4795 | 0.4703 |
| mom_x_lowvol_20_20 | bear | 0.1148 | 0.1924 | 0.5965 | 0.3699 | 0.4085 |
| turnover_stability | bear | 0.0433 | 0.0944 | 0.4593 | 0.3151 | 0.3020 |
| trend_lowvol | bear | 0.0954 | 0.2114 | 0.4514 | 0.2603 | 0.2844 |
| fund_pe | bear | 0.0403 | 0.1336 | 0.3014 | 0.2055 | 0.1817 |
| bb_width_20 | bear | 0.0451 | 0.2048 | 0.2201 | 0.1507 | 0.1266 |
| fund_profit_growth | bear | 0.0375 | 0.1937 | 0.1934 | 0.2329 | 0.1192 |

### 农业种植

- **Neutral**: ['fund_pb', 'volatility', 'momentum_reversal'] (单因子IC=0.0734, 组合IC=0.115)
  - weights: [0.3604, 0.3311, 0.3085]
- **Bull**: ['fund_pb', 'low_downside', 'trend_lowvol'] (单因子IC=0.0726, 组合IC=0.1026)
  - bull_weights: [0.4599, 0.3429, 0.1972]
- **Bear**: ['fund_profit_growth', 'bb_width_20', 'turnover_stability'] (单因子IC=0.0638, 组合IC=0.0938)
  - bear_weights: [0.3726, 0.3214, 0.306]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| fund_pb | neutral | 0.0784 | 0.1959 | 0.4003 | 0.3027 | 0.2607 |
| volatility | neutral | 0.0717 | 0.1995 | 0.3591 | 0.3340 | 0.2395 |
| momentum_reversal | neutral | 0.0702 | 0.1973 | 0.3558 | 0.2547 | 0.2232 |
| mom_x_lowvol_20_20 | neutral | 0.0649 | 0.1927 | 0.3367 | 0.2693 | 0.2137 |
| rsi_vol_combo | neutral | 0.0592 | 0.1822 | 0.3252 | 0.2171 | 0.1979 |
| trend_lowvol | neutral | 0.0668 | 0.2165 | 0.3084 | 0.2296 | 0.1896 |
| turnover_stability | neutral | 0.0379 | 0.1494 | 0.2538 | 0.1795 | 0.1497 |
| fund_pe | neutral | 0.0472 | 0.1934 | 0.2443 | 0.1879 | 0.1451 |
| low_downside | neutral | 0.0430 | 0.2022 | 0.2127 | 0.2463 | 0.1326 |
| fund_profit_growth | neutral | 0.0328 | 0.1521 | 0.2159 | 0.1921 | 0.1287 |
| fund_pb | bull | 0.0928 | 0.1762 | 0.5265 | 0.3864 | 0.3650 |
| low_downside | bull | 0.0743 | 0.1819 | 0.4082 | 0.3333 | 0.2721 |
| trend_lowvol | bull | 0.0507 | 0.1914 | 0.2648 | 0.1818 | 0.1565 |
| rsi_vol_combo | bull | 0.0389 | 0.1532 | 0.2541 | 0.1894 | 0.1511 |
| fund_revenue_growth | bull | 0.0308 | 0.1259 | 0.2446 | 0.2348 | 0.1510 |
| volatility | bull | 0.0452 | 0.2013 | 0.2244 | 0.1818 | 0.1326 |
| fund_pe | bull | 0.0521 | 0.2285 | 0.2280 | 0.1212 | 0.1278 |
| momentum_reversal | bull | 0.0329 | 0.1782 | 0.1845 | 0.1212 | 0.1034 |
| fund_score | bull | 0.0242 | 0.1519 | 0.1595 | 0.1061 | 0.0882 |
| exhaustion_risk | bull | 0.0160 | 0.1398 | 0.1144 | 0.1077 | 0.0634 |
| fund_profit_growth | bear | 0.0571 | 0.1475 | 0.3874 | 0.3151 | 0.2547 |
| bb_width_20 | bear | 0.0796 | 0.2232 | 0.3565 | 0.2329 | 0.2198 |
| turnover_stability | bear | 0.0546 | 0.1502 | 0.3637 | 0.1507 | 0.2092 |
| fund_gross_margin | bear | 0.0337 | 0.1156 | 0.2917 | 0.3425 | 0.1958 |
| mom_x_lowvol_20_20 | bear | 0.0702 | 0.2487 | 0.2824 | 0.1233 | 0.1586 |

### 农药兽药

- **Neutral**: ['fund_pb', 'mom_x_lowvol_20_20'] (单因子IC=0.0759, 组合IC=0.101)
  - weights: [0.5884, 0.4116]
- **Bull**: ['fund_pb', 'rsi_vol_combo', 'fund_pe'] (单因子IC=0.0624, 组合IC=0.1051)
  - bull_weights: [0.5298, 0.236, 0.2342]
- **Bear**: ['mom_x_lowvol_20_20', 'fund_pb'] (单因子IC=0.1027, 组合IC=0.1425)
  - bear_weights: [0.516, 0.484]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| fund_pb | neutral | 0.0850 | 0.1889 | 0.4497 | 0.3027 | 0.2929 |
| mom_x_lowvol_20_20 | neutral | 0.0668 | 0.2032 | 0.3288 | 0.2463 | 0.2049 |
| fund_pe | neutral | 0.0584 | 0.2214 | 0.2640 | 0.1608 | 0.1532 |
| momentum_reversal | neutral | 0.0481 | 0.2270 | 0.2120 | 0.1357 | 0.1204 |
| trend_lowvol | neutral | 0.0483 | 0.2277 | 0.2120 | 0.1263 | 0.1194 |
| volatility | neutral | 0.0296 | 0.2015 | 0.1469 | 0.1044 | 0.0811 |
| fund_profit_growth | neutral | 0.0282 | 0.2113 | 0.1335 | 0.1127 | 0.0743 |
| rsi_vol_combo | neutral | 0.0274 | 0.2140 | 0.1282 | 0.1232 | 0.0720 |
| fund_pb | bull | 0.0821 | 0.1661 | 0.4944 | 0.3485 | 0.3333 |
| rsi_vol_combo | bull | 0.0492 | 0.1935 | 0.2545 | 0.1667 | 0.1485 |
| fund_pe | bull | 0.0558 | 0.2207 | 0.2526 | 0.1667 | 0.1474 |
| trend_lowvol | bull | 0.0550 | 0.2498 | 0.2204 | 0.1515 | 0.1269 |
| low_downside | bull | 0.0333 | 0.1859 | 0.1789 | 0.1742 | 0.1050 |
| fund_score | bull | 0.0368 | 0.2091 | 0.1759 | 0.1591 | 0.1020 |
| fund_revenue_growth | bull | 0.0291 | 0.1928 | 0.1510 | 0.1212 | 0.0846 |
| mom_x_lowvol_20_20 | bear | 0.1131 | 0.2135 | 0.5299 | 0.3699 | 0.3629 |
| fund_pb | bear | 0.0923 | 0.1821 | 0.5071 | 0.3425 | 0.3404 |
| fund_revenue_growth | bear | 0.0472 | 0.2203 | 0.2144 | 0.1507 | 0.1234 |

### 冰雪经济

- **Neutral**: ['trend_lowvol', 'mom_x_lowvol_20_20'] (单因子IC=0.0903, 组合IC=0.1059)
  - weights: [0.5091, 0.4909]
- **Bull**: ['momentum_reversal', 'mom_x_lowvol_20_20', 'trend_lowvol'] (单因子IC=0.1145, 组合IC=0.1369)
  - bull_weights: [0.4411, 0.2954, 0.2635]
- **Bear**: ['momentum_reversal', 'trend_lowvol', 'mom_x_lowvol_20_20'] (单因子IC=0.1318, 组合IC=0.1503)
  - bear_weights: [0.3568, 0.3225, 0.3206]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| trend_lowvol | neutral | 0.0966 | 0.2593 | 0.3727 | 0.2839 | 0.2393 |
| mom_x_lowvol_20_20 | neutral | 0.0839 | 0.2342 | 0.3582 | 0.2881 | 0.2307 |
| momentum_reversal | neutral | 0.0767 | 0.2529 | 0.3032 | 0.2171 | 0.1845 |
| fund_profit_growth | neutral | 0.0501 | 0.1790 | 0.2797 | 0.2547 | 0.1755 |
| fund_pe | neutral | 0.0506 | 0.2049 | 0.2470 | 0.1733 | 0.1449 |
| low_downside | neutral | 0.0425 | 0.2328 | 0.1827 | 0.1482 | 0.1049 |
| wash_sale_score | neutral | 0.0307 | 0.1761 | 0.1741 | 0.1031 | 0.0960 |
| fund_score | neutral | 0.0308 | 0.2129 | 0.1448 | 0.1441 | 0.0828 |
| momentum_reversal | bull | 0.1358 | 0.1860 | 0.7298 | 0.5303 | 0.5584 |
| mom_x_lowvol_20_20 | bull | 0.1021 | 0.1903 | 0.5367 | 0.3939 | 0.3740 |
| trend_lowvol | bull | 0.1057 | 0.2172 | 0.4867 | 0.3712 | 0.3337 |
| turnover_stability | bull | 0.0720 | 0.1577 | 0.4567 | 0.3333 | 0.3044 |
| rsi_vol_combo | bull | 0.0816 | 0.1995 | 0.4090 | 0.2955 | 0.2649 |
| volatility | bull | 0.0593 | 0.1950 | 0.3043 | 0.2576 | 0.1913 |
| fund_gross_margin | bull | 0.0406 | 0.1802 | 0.2251 | 0.2955 | 0.1458 |
| fund_pe | bull | 0.0490 | 0.2048 | 0.2395 | 0.1894 | 0.1424 |
| low_downside | bull | 0.0342 | 0.1896 | 0.1804 | 0.1970 | 0.1080 |
| stroke_phase | bull | 0.0212 | 0.1980 | 0.1072 | 0.1364 | 0.0609 |
| momentum_reversal | bear | 0.1388 | 0.2285 | 0.6077 | 0.4795 | 0.4495 |
| trend_lowvol | bear | 0.1197 | 0.2220 | 0.5393 | 0.5068 | 0.4063 |
| mom_x_lowvol_20_20 | bear | 0.1369 | 0.2414 | 0.5671 | 0.4247 | 0.4039 |
| rsi_vol_combo | bear | 0.1119 | 0.2304 | 0.4858 | 0.3151 | 0.3194 |

### 冷链物流

- **Neutral**: ['trend_lowvol', 'fund_pb'] (单因子IC=0.0959, 组合IC=0.1232)
  - weights: [0.5295, 0.4705]
- **Bull**: ['low_downside', 'turnover_stability', 'momentum_reversal'] (单因子IC=0.0702, 组合IC=0.1053)
  - bull_weights: [0.4026, 0.33, 0.2675]
- **Bear**: ['mom_x_lowvol_20_20', 'momentum_reversal', 'rsi_vol_combo'] (单因子IC=0.1045, 组合IC=0.109)
  - bear_weights: [0.3473, 0.3276, 0.325]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| trend_lowvol | neutral | 0.1045 | 0.1996 | 0.5234 | 0.3779 | 0.3606 |
| fund_pb | neutral | 0.0873 | 0.1823 | 0.4788 | 0.3382 | 0.3204 |
| volatility | neutral | 0.0738 | 0.1732 | 0.4258 | 0.3299 | 0.2831 |
| mom_x_lowvol_20_20 | neutral | 0.0637 | 0.1556 | 0.4096 | 0.3006 | 0.2663 |
| momentum_reversal | neutral | 0.0715 | 0.1790 | 0.3997 | 0.2839 | 0.2566 |
| fund_pe | neutral | 0.0670 | 0.1755 | 0.3814 | 0.2839 | 0.2449 |
| turnover_stability | neutral | 0.0452 | 0.1232 | 0.3668 | 0.2902 | 0.2366 |
| fund_profit_growth | neutral | 0.0399 | 0.1289 | 0.3097 | 0.2129 | 0.1879 |
| low_downside | neutral | 0.0505 | 0.1803 | 0.2799 | 0.1858 | 0.1660 |
| rsi_vol_combo | neutral | 0.0429 | 0.1756 | 0.2442 | 0.1378 | 0.1389 |
| fund_score | neutral | 0.0330 | 0.1572 | 0.2102 | 0.1336 | 0.1191 |
| low_downside | bull | 0.0927 | 0.1451 | 0.6390 | 0.5227 | 0.4865 |
| turnover_stability | bull | 0.0546 | 0.1002 | 0.5454 | 0.4621 | 0.3987 |
| momentum_reversal | bull | 0.0634 | 0.1411 | 0.4492 | 0.4394 | 0.3233 |
| rsi_vol_combo | bull | 0.0520 | 0.1323 | 0.3929 | 0.3333 | 0.2619 |
| volatility | bull | 0.0600 | 0.1535 | 0.3906 | 0.3409 | 0.2619 |
| trend_lowvol | bull | 0.0607 | 0.1814 | 0.3348 | 0.2197 | 0.2042 |
| fund_pb | bull | 0.0601 | 0.1765 | 0.3405 | 0.1591 | 0.1974 |
| mom_x_lowvol_20_20 | bull | 0.0406 | 0.1437 | 0.2824 | 0.3182 | 0.1861 |
| mom_x_lowvol_20_20 | bear | 0.1057 | 0.1797 | 0.5880 | 0.4795 | 0.4349 |
| momentum_reversal | bear | 0.1050 | 0.1893 | 0.5547 | 0.4795 | 0.4103 |
| rsi_vol_combo | bear | 0.1027 | 0.1832 | 0.5606 | 0.4521 | 0.4070 |
| trend_lowvol | bear | 0.1066 | 0.1887 | 0.5651 | 0.3425 | 0.3793 |
| fund_pb | bear | 0.0837 | 0.1855 | 0.4511 | 0.3699 | 0.3090 |
| fund_pe | bear | 0.0765 | 0.2153 | 0.3554 | 0.2603 | 0.2240 |
| fund_gross_margin | bear | 0.0332 | 0.1247 | 0.2665 | 0.1507 | 0.1533 |

### 减肥药

- **Neutral**: ['fund_pb', 'volatility', 'low_downside'] (单因子IC=0.0749, 组合IC=0.0929)
  - weights: [0.3929, 0.3041, 0.303]
- **Bull**: ['low_downside', 'volatility'] (单因子IC=0.0898, 组合IC=0.0942)
  - bull_weights: [0.5272, 0.4728]
- **Bear**: ['fund_revenue_growth', 'trend_lowvol', 'fund_pb'] (单因子IC=0.0666, 组合IC=0.1116)
  - bear_weights: [0.413, 0.347, 0.2399]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| fund_pb | neutral | 0.0734 | 0.2098 | 0.3498 | 0.2610 | 0.2206 |
| volatility | neutral | 0.0794 | 0.2785 | 0.2849 | 0.1983 | 0.1707 |
| low_downside | neutral | 0.0720 | 0.2586 | 0.2785 | 0.2213 | 0.1701 |
| fund_pe | neutral | 0.0634 | 0.2325 | 0.2725 | 0.2401 | 0.1690 |
| trend_lowvol | neutral | 0.0624 | 0.2595 | 0.2406 | 0.2025 | 0.1447 |
| fund_profit_growth | neutral | 0.0479 | 0.2048 | 0.2340 | 0.1670 | 0.1365 |
| momentum_reversal | neutral | 0.0439 | 0.2525 | 0.1738 | 0.1117 | 0.0966 |
| fund_revenue_growth | neutral | 0.0316 | 0.2020 | 0.1563 | 0.1232 | 0.0878 |
| low_downside | bull | 0.0903 | 0.2239 | 0.4034 | 0.3258 | 0.2674 |
| volatility | bull | 0.0894 | 0.2428 | 0.3680 | 0.3030 | 0.2398 |
| fund_pe | bull | 0.0639 | 0.2273 | 0.2811 | 0.1591 | 0.1629 |
| turnover_stability | bull | 0.0570 | 0.2176 | 0.2621 | 0.2045 | 0.1579 |
| fund_pb | bull | 0.0588 | 0.2429 | 0.2421 | 0.1591 | 0.1403 |
| rsi_vol_combo | bull | 0.0376 | 0.1883 | 0.1995 | 0.1136 | 0.1111 |
| trend_lowvol | bull | 0.0409 | 0.2456 | 0.1666 | 0.1591 | 0.0965 |
| mom_x_lowvol_20_20 | bull | 0.0338 | 0.2428 | 0.1392 | 0.1288 | 0.0786 |
| momentum_reversal | bull | 0.0315 | 0.2311 | 0.1364 | 0.1212 | 0.0765 |
| fund_revenue_growth | bear | 0.0604 | 0.1654 | 0.3651 | 0.2877 | 0.2351 |
| trend_lowvol | bear | 0.0885 | 0.2701 | 0.3277 | 0.2055 | 0.1975 |
| fund_pb | bear | 0.0508 | 0.2139 | 0.2374 | 0.1507 | 0.1366 |
| rsi_vol_combo | bear | 0.0573 | 0.2977 | 0.1924 | 0.1233 | 0.1080 |

### 减速器

- **Neutral**: ['fund_pb', 'mom_x_lowvol_20_20', 'momentum_reversal'] (单因子IC=0.093, 组合IC=0.1332)
  - weights: [0.3971, 0.3048, 0.2981]
- **Bull**: ['volatility', 'fund_pb', 'low_downside'] (单因子IC=0.0992, 组合IC=0.1212)
  - bull_weights: [0.371, 0.3435, 0.2855]
- **Bear**: ['fund_roe', 'fund_profit_growth', 'fund_pe'] (单因子IC=0.0816, 组合IC=0.1066)
  - bear_weights: [0.3668, 0.3227, 0.3104]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| fund_pb | neutral | 0.1040 | 0.1813 | 0.5737 | 0.4175 | 0.4067 |
| mom_x_lowvol_20_20 | neutral | 0.0876 | 0.1882 | 0.4658 | 0.3403 | 0.3121 |
| momentum_reversal | neutral | 0.0873 | 0.1908 | 0.4577 | 0.3340 | 0.3053 |
| fund_pe | neutral | 0.0731 | 0.1770 | 0.4130 | 0.3382 | 0.2763 |
| rsi_vol_combo | neutral | 0.0675 | 0.1838 | 0.3669 | 0.2923 | 0.2371 |
| volatility | neutral | 0.0770 | 0.2301 | 0.3347 | 0.2276 | 0.2054 |
| fund_profit_growth | neutral | 0.0499 | 0.1569 | 0.3180 | 0.2463 | 0.1982 |
| trend_lowvol | neutral | 0.0723 | 0.2311 | 0.3128 | 0.1900 | 0.1861 |
| low_downside | neutral | 0.0446 | 0.1992 | 0.2240 | 0.1670 | 0.1307 |
| fund_score | neutral | 0.0358 | 0.1663 | 0.2154 | 0.1628 | 0.1252 |
| volatility | bull | 0.1141 | 0.2069 | 0.5513 | 0.4470 | 0.3989 |
| fund_pb | bull | 0.0975 | 0.1840 | 0.5299 | 0.3939 | 0.3693 |
| low_downside | bull | 0.0859 | 0.1812 | 0.4739 | 0.2955 | 0.3070 |
| trend_lowvol | bull | 0.0714 | 0.1979 | 0.3610 | 0.2424 | 0.2242 |
| turnover_stability | bull | 0.0523 | 0.1503 | 0.3478 | 0.2424 | 0.2160 |
| fund_profit_growth | bull | 0.0461 | 0.1639 | 0.2810 | 0.2689 | 0.1783 |
| fund_score | bull | 0.0439 | 0.1728 | 0.2540 | 0.2197 | 0.1549 |
| fund_revenue_growth | bull | 0.0321 | 0.1433 | 0.2239 | 0.2273 | 0.1374 |
| mom_x_lowvol_20_20 | bull | 0.0387 | 0.1739 | 0.2225 | 0.1136 | 0.1239 |
| momentum_reversal | bull | 0.0317 | 0.1793 | 0.1769 | 0.1212 | 0.0992 |
| stroke_phase | bull | 0.0174 | 0.1344 | 0.1298 | 0.2121 | 0.0786 |
| fund_roe | bear | 0.0932 | 0.1725 | 0.5402 | 0.3425 | 0.3626 |
| fund_profit_growth | bear | 0.0764 | 0.1771 | 0.4313 | 0.4795 | 0.3190 |
| fund_pe | bear | 0.0753 | 0.1579 | 0.4766 | 0.2877 | 0.3069 |
| turnover_stability | bear | 0.0510 | 0.1265 | 0.4032 | 0.3699 | 0.2761 |
| mom_x_lowvol_20_20 | bear | 0.0816 | 0.2237 | 0.3647 | 0.3699 | 0.2498 |
| fund_score | bear | 0.0761 | 0.1950 | 0.3903 | 0.2055 | 0.2353 |
| fund_revenue_growth | bear | 0.0569 | 0.1916 | 0.2967 | 0.2603 | 0.1870 |
| fund_gross_margin | bear | 0.0552 | 0.1702 | 0.3241 | 0.1507 | 0.1864 |

### 创业成份

- **Neutral**: ['fund_profit_growth', 'fund_pe', 'fund_pb'] (单因子IC=0.0442, 组合IC=0.06)
  - weights: [0.4287, 0.3137, 0.2576]
- **Bull**: ['fund_score', 'turnover_stability'] (单因子IC=0.058, 组合IC=0.0777)
  - bull_weights: [0.5441, 0.4559]
- **Bear**: ['momentum_reversal', 'mom_x_lowvol_20_20', 'bb_width_20'] (单因子IC=0.0994, 组合IC=0.1179)
  - bear_weights: [0.3638, 0.3609, 0.2753]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| fund_profit_growth | neutral | 0.0542 | 0.1773 | 0.3054 | 0.2025 | 0.1836 |
| fund_pe | neutral | 0.0413 | 0.1770 | 0.2332 | 0.1524 | 0.1344 |
| fund_pb | neutral | 0.0373 | 0.1939 | 0.1922 | 0.1482 | 0.1104 |
| fund_score | neutral | 0.0281 | 0.1954 | 0.1440 | 0.1127 | 0.0801 |
| fund_roe | neutral | 0.0206 | 0.1754 | 0.1176 | 0.1211 | 0.0659 |
| fund_score | bull | 0.0725 | 0.1805 | 0.4014 | 0.2197 | 0.2448 |
| turnover_stability | bull | 0.0435 | 0.1350 | 0.3223 | 0.2727 | 0.2051 |
| fund_profit_growth | bull | 0.0464 | 0.1636 | 0.2839 | 0.1364 | 0.1613 |
| fund_revenue_growth | bull | 0.0479 | 0.1810 | 0.2647 | 0.1742 | 0.1554 |
| fund_pb | bull | 0.0535 | 0.2248 | 0.2380 | 0.2955 | 0.1542 |
| fund_pe | bull | 0.0555 | 0.2131 | 0.2606 | 0.1742 | 0.1530 |
| volatility | bull | 0.0454 | 0.1772 | 0.2565 | 0.1364 | 0.1457 |
| fund_roe | bull | 0.0398 | 0.1624 | 0.2450 | 0.1742 | 0.1438 |
| low_downside | bull | 0.0213 | 0.1675 | 0.1274 | 0.1061 | 0.0705 |
| momentum_reversal | bear | 0.1082 | 0.2314 | 0.4678 | 0.3699 | 0.3204 |
| mom_x_lowvol_20_20 | bear | 0.1051 | 0.2356 | 0.4462 | 0.4247 | 0.3178 |
| bb_width_20 | bear | 0.0848 | 0.2253 | 0.3765 | 0.2877 | 0.2424 |
| rsi_vol_combo | bear | 0.0658 | 0.2006 | 0.3282 | 0.2329 | 0.2023 |
| limit_pullback_score | bear | 0.0420 | 0.1595 | 0.2635 | 0.3151 | 0.1732 |
| fund_pb | bear | 0.0421 | 0.1627 | 0.2587 | 0.1781 | 0.1524 |

### 创新医疗服务

- **Neutral**: ['fund_pb', 'volatility', 'momentum_reversal'] (单因子IC=0.067, 组合IC=0.0987)
  - weights: [0.3475, 0.3415, 0.311]
- **Bull**: ['turnover_stability', 'low_downside', 'rsi_vol_combo'] (单因子IC=0.0527, 组合IC=0.0773)
  - bull_weights: [0.3634, 0.3244, 0.3122]
- **Bear**: ['trend_lowvol', 'mom_x_lowvol_20_20'] (单因子IC=0.153, 组合IC=0.1776)
  - bear_weights: [0.521, 0.479]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| fund_pb | neutral | 0.0630 | 0.2113 | 0.2983 | 0.2797 | 0.1909 |
| volatility | neutral | 0.0733 | 0.2378 | 0.3082 | 0.2171 | 0.1875 |
| momentum_reversal | neutral | 0.0646 | 0.2306 | 0.2801 | 0.2192 | 0.1708 |
| mom_x_lowvol_20_20 | neutral | 0.0614 | 0.2235 | 0.2747 | 0.2109 | 0.1663 |
| trend_lowvol | neutral | 0.0659 | 0.2418 | 0.2725 | 0.1649 | 0.1587 |
| low_downside | neutral | 0.0564 | 0.2251 | 0.2506 | 0.2359 | 0.1548 |
| fund_profit_growth | neutral | 0.0287 | 0.1496 | 0.1916 | 0.1461 | 0.1098 |
| fund_pe | neutral | 0.0394 | 0.2321 | 0.1697 | 0.1534 | 0.0979 |
| rsi_vol_combo | neutral | 0.0383 | 0.2279 | 0.1680 | 0.1138 | 0.0936 |
| turnover_stability | neutral | 0.0248 | 0.1601 | 0.1548 | 0.1534 | 0.0893 |
| turnover_stability | bull | 0.0431 | 0.1238 | 0.3481 | 0.2121 | 0.2110 |
| low_downside | bull | 0.0589 | 0.1929 | 0.3050 | 0.2348 | 0.1883 |
| rsi_vol_combo | bull | 0.0563 | 0.1893 | 0.2972 | 0.2197 | 0.1813 |
| fund_pb | bull | 0.0586 | 0.2430 | 0.2413 | 0.2576 | 0.1517 |
| momentum_reversal | bull | 0.0615 | 0.2338 | 0.2631 | 0.1364 | 0.1495 |
| volatility | bull | 0.0589 | 0.2404 | 0.2448 | 0.2197 | 0.1493 |
| mom_x_lowvol_20_20 | bull | 0.0617 | 0.2410 | 0.2562 | 0.1061 | 0.1417 |
| fund_profit_growth | bull | 0.0309 | 0.1415 | 0.2180 | 0.2045 | 0.1313 |
| fund_score | bull | 0.0282 | 0.2175 | 0.1299 | 0.1136 | 0.0723 |
| trend_lowvol | bear | 0.1741 | 0.2520 | 0.6908 | 0.3151 | 0.4542 |
| mom_x_lowvol_20_20 | bear | 0.1320 | 0.2381 | 0.5543 | 0.5068 | 0.4176 |
| momentum_reversal | bear | 0.1433 | 0.2697 | 0.5315 | 0.3425 | 0.3567 |
| fund_pb | bear | 0.0628 | 0.1553 | 0.4043 | 0.3699 | 0.2769 |
| rsi_vol_combo | bear | 0.0928 | 0.2368 | 0.3922 | 0.2877 | 0.2525 |
| fund_gross_margin | bear | 0.0336 | 0.1258 | 0.2673 | 0.2603 | 0.1684 |

### 创新药

- **Neutral**: ['volatility', 'fund_pb', 'fund_pe'] (单因子IC=0.0734, 组合IC=0.0922)
  - weights: [0.3456, 0.3276, 0.3267]
- **Bull**: ['low_downside', 'volatility', 'fund_pb'] (单因子IC=0.0751, 组合IC=0.0836)
  - bull_weights: [0.3905, 0.3305, 0.279]
- **Bear**: ['mom_x_lowvol_20_20', 'momentum_reversal', 'fund_pe'] (单因子IC=0.0974, 组合IC=0.1404)
  - bear_weights: [0.3571, 0.3241, 0.3189]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| volatility | neutral | 0.0753 | 0.1897 | 0.3973 | 0.3528 | 0.2687 |
| fund_pb | neutral | 0.0784 | 0.1963 | 0.3993 | 0.2756 | 0.2547 |
| fund_pe | neutral | 0.0665 | 0.1706 | 0.3900 | 0.3027 | 0.2540 |
| fund_profit_growth | neutral | 0.0464 | 0.1337 | 0.3473 | 0.2735 | 0.2211 |
| trend_lowvol | neutral | 0.0630 | 0.1948 | 0.3235 | 0.2860 | 0.2080 |
| low_downside | neutral | 0.0488 | 0.1853 | 0.2632 | 0.2568 | 0.1654 |
| momentum_reversal | neutral | 0.0501 | 0.1812 | 0.2763 | 0.1921 | 0.1647 |
| fund_score | neutral | 0.0420 | 0.1611 | 0.2605 | 0.1649 | 0.1517 |
| mom_x_lowvol_20_20 | neutral | 0.0396 | 0.1687 | 0.2345 | 0.1775 | 0.1381 |
| fund_revenue_growth | neutral | 0.0270 | 0.1328 | 0.2030 | 0.1900 | 0.1208 |
| turnover_stability | neutral | 0.0237 | 0.1152 | 0.2060 | 0.1628 | 0.1198 |
| rsi_vol_combo | neutral | 0.0342 | 0.1673 | 0.2041 | 0.1524 | 0.1176 |
| low_downside | bull | 0.0763 | 0.1653 | 0.4618 | 0.4091 | 0.3254 |
| volatility | bull | 0.0715 | 0.1731 | 0.4130 | 0.3333 | 0.2753 |
| fund_pb | bull | 0.0776 | 0.2100 | 0.3697 | 0.2576 | 0.2325 |
| fund_pe | bull | 0.0553 | 0.1685 | 0.3282 | 0.2197 | 0.2002 |
| turnover_stability | bull | 0.0324 | 0.1179 | 0.2744 | 0.2121 | 0.1663 |
| trend_lowvol | bull | 0.0404 | 0.1820 | 0.2219 | 0.2045 | 0.1337 |
| fund_roe | bull | 0.0360 | 0.1686 | 0.2134 | 0.1970 | 0.1277 |
| fund_score | bull | 0.0256 | 0.1550 | 0.1654 | 0.1136 | 0.0921 |
| mom_x_lowvol_20_20 | bear | 0.0990 | 0.1783 | 0.5554 | 0.5616 | 0.4337 |
| momentum_reversal | bear | 0.1011 | 0.1900 | 0.5321 | 0.4795 | 0.3936 |
| fund_pe | bear | 0.0920 | 0.1692 | 0.5437 | 0.4247 | 0.3873 |
| fund_revenue_growth | bear | 0.0533 | 0.1037 | 0.5138 | 0.2877 | 0.3308 |
| trend_lowvol | bear | 0.0819 | 0.1957 | 0.4186 | 0.3425 | 0.2810 |
| fund_profit_growth | bear | 0.0408 | 0.1302 | 0.3134 | 0.2877 | 0.2018 |
| fund_score | bear | 0.0483 | 0.1576 | 0.3064 | 0.2055 | 0.1847 |
| fund_pb | bear | 0.0577 | 0.1929 | 0.2993 | 0.1781 | 0.1763 |
| turnover_stability | bear | 0.0228 | 0.1028 | 0.2215 | 0.2329 | 0.1366 |
| fund_roe | bear | 0.0436 | 0.1865 | 0.2340 | 0.1233 | 0.1314 |
| rsi_vol_combo | bear | 0.0401 | 0.1809 | 0.2216 | 0.1507 | 0.1275 |

### 券商概念

- **Neutral**: ['fund_pe', 'fund_pb', 'fund_score'] (单因子IC=0.0948, 组合IC=0.1238)
  - weights: [0.3808, 0.3288, 0.2905]
- **Bull**: ['low_downside'] (单因子IC=0.1443, 组合IC=0.1443)
  - bull_weights: [1.0]
- **Bear**: ['mom_x_lowvol_20_20', 'momentum_reversal', 'bb_width_20'] (单因子IC=0.1458, 组合IC=0.1857)
  - bear_weights: [0.3628, 0.3506, 0.2865]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| fund_pe | neutral | 0.1102 | 0.2526 | 0.4362 | 0.3445 | 0.2932 |
| fund_pb | neutral | 0.0945 | 0.2339 | 0.4042 | 0.2526 | 0.2531 |
| fund_score | neutral | 0.0796 | 0.2257 | 0.3527 | 0.2683 | 0.2237 |
| fund_profit_growth | neutral | 0.0603 | 0.1845 | 0.3267 | 0.2599 | 0.2058 |
| volatility | neutral | 0.0657 | 0.2117 | 0.3105 | 0.2150 | 0.1886 |
| trend_lowvol | neutral | 0.0646 | 0.2372 | 0.2725 | 0.2359 | 0.1684 |
| momentum_reversal | neutral | 0.0569 | 0.2119 | 0.2685 | 0.2213 | 0.1640 |
| rsi_vol_combo | neutral | 0.0545 | 0.2040 | 0.2669 | 0.1795 | 0.1574 |
| fund_roe | neutral | 0.0559 | 0.2487 | 0.2249 | 0.1775 | 0.1324 |
| low_downside | neutral | 0.0528 | 0.2381 | 0.2218 | 0.1837 | 0.1313 |
| fund_revenue_growth | neutral | 0.0380 | 0.1893 | 0.2007 | 0.1649 | 0.1169 |
| mom_x_lowvol_20_20 | neutral | 0.0344 | 0.1971 | 0.1744 | 0.1399 | 0.0994 |
| low_downside | bull | 0.1443 | 0.2158 | 0.6685 | 0.4924 | 0.4988 |
| volatility | bull | 0.1201 | 0.1833 | 0.6556 | 0.5000 | 0.4917 |
| mom_x_lowvol_20_20 | bull | 0.1005 | 0.2009 | 0.5003 | 0.4091 | 0.3525 |
| fund_pb | bull | 0.0990 | 0.2500 | 0.3960 | 0.3030 | 0.2580 |
| momentum_reversal | bull | 0.0961 | 0.2468 | 0.3894 | 0.2955 | 0.2522 |
| trend_lowvol | bull | 0.0868 | 0.2473 | 0.3509 | 0.3485 | 0.2366 |
| fund_pe | bull | 0.0971 | 0.2635 | 0.3686 | 0.2348 | 0.2276 |
| turnover_stability | bull | 0.0570 | 0.1782 | 0.3199 | 0.2879 | 0.2060 |
| fund_roe | bull | 0.0564 | 0.1882 | 0.2998 | 0.2500 | 0.1874 |
| fund_profit_growth | bull | 0.0374 | 0.1463 | 0.2553 | 0.2273 | 0.1567 |
| rsi_vol_combo | bull | 0.0568 | 0.2252 | 0.2524 | 0.1212 | 0.1415 |
| stroke_phase | bull | 0.0298 | 0.1998 | 0.1489 | 0.1667 | 0.0869 |
| fund_revenue_growth | bull | 0.0259 | 0.1705 | 0.1520 | 0.1288 | 0.0858 |
| fund_score | bull | 0.0257 | 0.1847 | 0.1394 | 0.1136 | 0.0776 |
| mom_x_lowvol_20_20 | bear | 0.1598 | 0.2146 | 0.7443 | 0.5342 | 0.5710 |
| momentum_reversal | bear | 0.1711 | 0.2421 | 0.7066 | 0.5616 | 0.5517 |
| bb_width_20 | bear | 0.1064 | 0.1810 | 0.5878 | 0.5342 | 0.4509 |
| fund_revenue_growth | bear | 0.1123 | 0.2019 | 0.5560 | 0.3699 | 0.3808 |
| trend_lowvol | bear | 0.1238 | 0.2518 | 0.4918 | 0.3973 | 0.3436 |
| rsi_vol_combo | bear | 0.1052 | 0.2448 | 0.4297 | 0.3151 | 0.2825 |
| fund_score | bear | 0.0890 | 0.2565 | 0.3470 | 0.4247 | 0.2472 |
| fund_profit_growth | bear | 0.0643 | 0.2189 | 0.2937 | 0.3151 | 0.1931 |
| fund_gross_margin | bear | 0.0572 | 0.1882 | 0.3040 | 0.2055 | 0.1832 |
| fund_roe | bear | 0.0579 | 0.2617 | 0.2212 | 0.1781 | 0.1303 |

### 动力电池回收

- **Neutral**: ['trend_lowvol', 'fund_profit_growth'] (单因子IC=0.0734, 组合IC=0.1068)
  - weights: [0.5215, 0.4785]
- **Bull**: ['low_downside'] (单因子IC=0.1176, 组合IC=0.1176)
  - bull_weights: [1.0]
- **Bear**: ['momentum_reversal'] (单因子IC=0.1408, 组合IC=0.1408)
  - bear_weights: [1.0]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| trend_lowvol | neutral | 0.0837 | 0.2643 | 0.3166 | 0.2484 | 0.1976 |
| fund_profit_growth | neutral | 0.0632 | 0.2142 | 0.2949 | 0.2296 | 0.1813 |
| fund_pe | neutral | 0.0678 | 0.2433 | 0.2788 | 0.2150 | 0.1694 |
| volatility | neutral | 0.0588 | 0.2262 | 0.2600 | 0.2015 | 0.1562 |
| fund_pb | neutral | 0.0662 | 0.2686 | 0.2465 | 0.2359 | 0.1523 |
| momentum_reversal | neutral | 0.0530 | 0.2352 | 0.2255 | 0.1722 | 0.1322 |
| mom_x_lowvol_20_20 | neutral | 0.0492 | 0.2199 | 0.2237 | 0.1764 | 0.1316 |
| fund_score | neutral | 0.0491 | 0.2514 | 0.1953 | 0.1608 | 0.1133 |
| rsi_vol_combo | neutral | 0.0310 | 0.2282 | 0.1358 | 0.1044 | 0.0750 |
| turnover_stability | neutral | 0.0170 | 0.1946 | 0.0872 | 0.1013 | 0.0480 |
| low_downside | bull | 0.1176 | 0.2238 | 0.5253 | 0.3864 | 0.3641 |
| volatility | bull | 0.0938 | 0.2064 | 0.4545 | 0.3333 | 0.3030 |
| fund_profit_growth | bull | 0.0550 | 0.2055 | 0.2677 | 0.1591 | 0.1551 |
| fund_pb | bull | 0.0610 | 0.2485 | 0.2457 | 0.2197 | 0.1498 |
| fund_score | bull | 0.0588 | 0.2416 | 0.2433 | 0.1894 | 0.1447 |
| fund_pe | bull | 0.0424 | 0.2495 | 0.1699 | 0.1970 | 0.1017 |
| relative_strength | bull | 0.0311 | 0.2078 | 0.1497 | 0.1970 | 0.0896 |
| fund_gross_margin | bull | 0.0262 | 0.1710 | 0.1530 | 0.1439 | 0.0875 |
| exhaustion_risk | bull | 0.0247 | 0.1799 | 0.1371 | 0.1844 | 0.0812 |
| momentum_reversal | bear | 0.1408 | 0.2332 | 0.6037 | 0.5616 | 0.4714 |
| mom_x_lowvol_20_20 | bear | 0.1110 | 0.2282 | 0.4865 | 0.3699 | 0.3332 |
| rsi_vol_combo | bear | 0.0965 | 0.2035 | 0.4742 | 0.3425 | 0.3183 |
| bb_width_20 | bear | 0.0908 | 0.2190 | 0.4145 | 0.4247 | 0.2953 |

### 包装材料

- **Neutral**: ['fund_pb', 'trend_lowvol'] (单因子IC=0.0874, 组合IC=0.1189)
  - weights: [0.5881, 0.4119]
- **Bull**: ['volatility', 'fund_pb', 'rsi_vol_combo'] (单因子IC=0.0797, 组合IC=0.1009)
  - bull_weights: [0.3497, 0.3339, 0.3165]
- **Bear**: ['trend_lowvol', 'mom_x_lowvol_20_20'] (单因子IC=0.1888, 组合IC=0.2343)
  - bear_weights: [0.5659, 0.4341]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| fund_pb | neutral | 0.0924 | 0.2238 | 0.4130 | 0.3664 | 0.2822 |
| trend_lowvol | neutral | 0.0823 | 0.2491 | 0.3303 | 0.1962 | 0.1976 |
| fund_score | neutral | 0.0650 | 0.2155 | 0.3017 | 0.2735 | 0.1921 |
| volatility | neutral | 0.0690 | 0.2546 | 0.2709 | 0.2505 | 0.1694 |
| momentum_reversal | neutral | 0.0657 | 0.2348 | 0.2801 | 0.2035 | 0.1685 |
| mom_x_lowvol_20_20 | neutral | 0.0616 | 0.2312 | 0.2663 | 0.1994 | 0.1597 |
| fund_profit_growth | neutral | 0.0538 | 0.2046 | 0.2629 | 0.2067 | 0.1586 |
| fund_pe | neutral | 0.0628 | 0.2579 | 0.2436 | 0.2537 | 0.1527 |
| fund_roe | neutral | 0.0512 | 0.2459 | 0.2082 | 0.1879 | 0.1237 |
| rsi_vol_combo | neutral | 0.0456 | 0.2348 | 0.1944 | 0.1597 | 0.1127 |
| low_downside | neutral | 0.0417 | 0.2525 | 0.1650 | 0.1347 | 0.0936 |
| volatility | bull | 0.0801 | 0.2013 | 0.3978 | 0.2879 | 0.2562 |
| fund_pb | bull | 0.0809 | 0.2047 | 0.3950 | 0.2386 | 0.2446 |
| rsi_vol_combo | bull | 0.0781 | 0.2055 | 0.3801 | 0.2197 | 0.2318 |
| momentum_reversal | bull | 0.0802 | 0.2340 | 0.3426 | 0.2576 | 0.2154 |
| trend_lowvol | bull | 0.0762 | 0.2337 | 0.3263 | 0.1970 | 0.1953 |
| turnover_stability | bull | 0.0610 | 0.2048 | 0.2980 | 0.2348 | 0.1840 |
| mom_x_lowvol_20_20 | bull | 0.0591 | 0.2114 | 0.2795 | 0.2424 | 0.1736 |
| fund_pe | bull | 0.0459 | 0.1976 | 0.2321 | 0.2386 | 0.1438 |
| low_downside | bull | 0.0371 | 0.2318 | 0.1602 | 0.1439 | 0.0916 |
| trend_lowvol | bear | 0.1959 | 0.1999 | 0.9800 | 0.6164 | 0.7921 |
| mom_x_lowvol_20_20 | bear | 0.1817 | 0.2377 | 0.7646 | 0.5890 | 0.6075 |
| momentum_reversal | bear | 0.1917 | 0.2470 | 0.7762 | 0.5616 | 0.6060 |
| rsi_vol_combo | bear | 0.1499 | 0.2418 | 0.6197 | 0.5342 | 0.4754 |
| turnover_stability | bear | 0.0986 | 0.2148 | 0.4592 | 0.3288 | 0.3051 |
| fund_pb | bear | 0.0582 | 0.2346 | 0.2482 | 0.3151 | 0.1632 |
| volatility | bear | 0.0695 | 0.2706 | 0.2570 | 0.2603 | 0.1619 |
| fund_gross_margin | bear | 0.0427 | 0.2045 | 0.2087 | 0.1781 | 0.1229 |

### 化债(AMC)概念

- **Neutral**: ['momentum_reversal', 'mom_x_lowvol_20_20', 'trend_lowvol'] (单因子IC=0.0829, 组合IC=0.0937)
  - weights: [0.3558, 0.325, 0.3192]
- **Bull**: ['volatility', 'low_downside', 'fund_pb'] (单因子IC=0.1219, 组合IC=0.1462)
  - bull_weights: [0.3872, 0.356, 0.2568]
- **Bear**: ['trend_lowvol', 'mom_x_lowvol_20_20'] (单因子IC=0.0934, 组合IC=0.1211)
  - bear_weights: [0.6129, 0.3871]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| momentum_reversal | neutral | 0.0847 | 0.2090 | 0.4054 | 0.3027 | 0.2640 |
| mom_x_lowvol_20_20 | neutral | 0.0773 | 0.2048 | 0.3776 | 0.2777 | 0.2412 |
| trend_lowvol | neutral | 0.0865 | 0.2360 | 0.3666 | 0.2923 | 0.2369 |
| fund_score | neutral | 0.0589 | 0.1932 | 0.3047 | 0.2693 | 0.1934 |
| rsi_vol_combo | neutral | 0.0607 | 0.2101 | 0.2892 | 0.2422 | 0.1796 |
| turnover_stability | neutral | 0.0445 | 0.1741 | 0.2558 | 0.1816 | 0.1511 |
| volatility | neutral | 0.0572 | 0.2424 | 0.2359 | 0.2255 | 0.1446 |
| fund_profit_growth | neutral | 0.0377 | 0.1565 | 0.2410 | 0.1879 | 0.1431 |
| fund_pe | neutral | 0.0527 | 0.2462 | 0.2141 | 0.1524 | 0.1234 |
| fund_revenue_growth | neutral | 0.0269 | 0.1525 | 0.1762 | 0.2067 | 0.1063 |
| fund_pb | neutral | 0.0526 | 0.2847 | 0.1848 | 0.1044 | 0.1020 |
| fund_roe | neutral | 0.0410 | 0.2395 | 0.1711 | 0.1816 | 0.1011 |
| low_downside | neutral | 0.0347 | 0.2301 | 0.1509 | 0.1232 | 0.0847 |
| volatility | bull | 0.1245 | 0.1928 | 0.6456 | 0.5000 | 0.4842 |
| low_downside | bull | 0.1264 | 0.2075 | 0.6088 | 0.4621 | 0.4451 |
| fund_pb | bull | 0.1150 | 0.2333 | 0.4928 | 0.3030 | 0.3211 |
| rsi_vol_combo | bull | 0.0740 | 0.1741 | 0.4249 | 0.3409 | 0.2849 |
| momentum_reversal | bull | 0.0905 | 0.2139 | 0.4229 | 0.3409 | 0.2835 |
| turnover_stability | bull | 0.0590 | 0.1428 | 0.4135 | 0.3030 | 0.2694 |
| mom_x_lowvol_20_20 | bull | 0.0760 | 0.1897 | 0.4004 | 0.3333 | 0.2669 |
| trend_lowvol | bull | 0.0973 | 0.2367 | 0.4108 | 0.2803 | 0.2630 |
| fund_pe | bull | 0.0896 | 0.2604 | 0.3441 | 0.2045 | 0.2073 |
| fund_revenue_growth | bull | 0.0427 | 0.1549 | 0.2756 | 0.2500 | 0.1723 |
| fund_roe | bull | 0.0656 | 0.2504 | 0.2621 | 0.1439 | 0.1499 |
| fund_score | bull | 0.0527 | 0.2180 | 0.2416 | 0.1515 | 0.1391 |
| trend_lowvol | bear | 0.1080 | 0.2483 | 0.4348 | 0.3699 | 0.2978 |
| mom_x_lowvol_20_20 | bear | 0.0789 | 0.2643 | 0.2986 | 0.2603 | 0.1881 |
| momentum_reversal | bear | 0.0846 | 0.2899 | 0.2918 | 0.2877 | 0.1879 |

### 化妆品概念

- **Neutral**: ['fund_pb', 'volatility', 'fund_pe'] (单因子IC=0.0797, 组合IC=0.1073)
  - weights: [0.4056, 0.3044, 0.2901]
- **Bull**: ['fund_pb', 'fund_pe', 'volatility'] (单因子IC=0.0938, 组合IC=0.1421)
  - bull_weights: [0.3791, 0.34, 0.2808]
- **Bear**: ['fund_pe', 'fund_pb', 'turnover_stability'] (单因子IC=0.0864, 组合IC=0.1396)
  - bear_weights: [0.3943, 0.3908, 0.2149]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| fund_pb | neutral | 0.0988 | 0.2161 | 0.4569 | 0.3006 | 0.2971 |
| volatility | neutral | 0.0730 | 0.2169 | 0.3364 | 0.3257 | 0.2230 |
| fund_pe | neutral | 0.0673 | 0.2034 | 0.3311 | 0.2839 | 0.2125 |
| fund_profit_growth | neutral | 0.0432 | 0.1821 | 0.2371 | 0.1921 | 0.1413 |
| fund_revenue_growth | neutral | 0.0390 | 0.1621 | 0.2405 | 0.1733 | 0.1411 |
| low_downside | neutral | 0.0521 | 0.2261 | 0.2302 | 0.2203 | 0.1405 |
| turnover_stability | neutral | 0.0343 | 0.1517 | 0.2262 | 0.1639 | 0.1316 |
| momentum_reversal | neutral | 0.0465 | 0.2203 | 0.2110 | 0.1482 | 0.1212 |
| mom_x_lowvol_20_20 | neutral | 0.0452 | 0.2167 | 0.2087 | 0.1357 | 0.1185 |
| trend_lowvol | neutral | 0.0459 | 0.2348 | 0.1956 | 0.1795 | 0.1154 |
| rsi_vol_combo | neutral | 0.0386 | 0.2154 | 0.1791 | 0.1461 | 0.1026 |
| fund_pb | bull | 0.1088 | 0.2056 | 0.5291 | 0.4318 | 0.3788 |
| fund_pe | bull | 0.0868 | 0.1781 | 0.4874 | 0.3939 | 0.3397 |
| volatility | bull | 0.0857 | 0.1944 | 0.4409 | 0.2727 | 0.2806 |
| momentum_reversal | bull | 0.0731 | 0.1833 | 0.3989 | 0.3636 | 0.2720 |
| low_downside | bull | 0.0612 | 0.1779 | 0.3438 | 0.2727 | 0.2188 |
| mom_x_lowvol_20_20 | bull | 0.0568 | 0.1779 | 0.3192 | 0.3106 | 0.2092 |
| rsi_vol_combo | bull | 0.0644 | 0.1833 | 0.3511 | 0.1667 | 0.2048 |
| trend_lowvol | bull | 0.0582 | 0.2049 | 0.2841 | 0.1742 | 0.1668 |
| fund_pe | bear | 0.1048 | 0.2005 | 0.5226 | 0.3699 | 0.3580 |
| fund_pb | bear | 0.1157 | 0.2168 | 0.5339 | 0.3288 | 0.3547 |
| turnover_stability | bear | 0.0386 | 0.1327 | 0.2906 | 0.3425 | 0.1951 |
| fund_revenue_growth | bear | 0.0412 | 0.1799 | 0.2287 | 0.1507 | 0.1316 |
| mom_x_lowvol_20_20 | bear | 0.0562 | 0.2725 | 0.2063 | 0.1507 | 0.1187 |

### 化工原料

- **Neutral**: ['fund_pb', 'mom_x_lowvol_20_20', 'volatility'] (单因子IC=0.0665, 组合IC=0.0888)
  - weights: [0.356, 0.323, 0.321]
- **Bull**: ['volatility', 'low_downside', 'fund_pb'] (单因子IC=0.0969, 组合IC=0.1177)
  - bull_weights: [0.4263, 0.297, 0.2767]
- **Bear**: ['fund_score', 'momentum_reversal', 'fund_profit_growth'] (单因子IC=0.1377, 组合IC=0.163)
  - bear_weights: [0.3663, 0.3265, 0.3071]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| fund_pb | neutral | 0.0661 | 0.1875 | 0.3524 | 0.2902 | 0.2273 |
| mom_x_lowvol_20_20 | neutral | 0.0633 | 0.1926 | 0.3287 | 0.2547 | 0.2062 |
| volatility | neutral | 0.0700 | 0.2180 | 0.3213 | 0.2756 | 0.2049 |
| trend_lowvol | neutral | 0.0670 | 0.2382 | 0.2813 | 0.2526 | 0.1762 |
| momentum_reversal | neutral | 0.0596 | 0.2097 | 0.2844 | 0.1900 | 0.1692 |
| low_downside | neutral | 0.0530 | 0.2176 | 0.2436 | 0.2317 | 0.1500 |
| turnover_stability | neutral | 0.0405 | 0.1716 | 0.2359 | 0.2380 | 0.1460 |
| rsi_vol_combo | neutral | 0.0406 | 0.1867 | 0.2173 | 0.2004 | 0.1304 |
| fund_pe | neutral | 0.0492 | 0.2379 | 0.2069 | 0.1942 | 0.1236 |
| fund_profit_growth | neutral | 0.0280 | 0.1714 | 0.1632 | 0.1336 | 0.0925 |
| fund_revenue_growth | neutral | 0.0197 | 0.1369 | 0.1436 | 0.1587 | 0.0832 |
| volatility | bull | 0.1140 | 0.1965 | 0.5799 | 0.4091 | 0.4086 |
| low_downside | bull | 0.0930 | 0.2118 | 0.4394 | 0.2955 | 0.2846 |
| fund_pb | bull | 0.0838 | 0.2046 | 0.4094 | 0.2955 | 0.2652 |
| turnover_stability | bull | 0.0502 | 0.1527 | 0.3288 | 0.2652 | 0.2080 |
| fund_pe | bull | 0.0702 | 0.2260 | 0.3107 | 0.2500 | 0.1942 |
| mom_x_lowvol_20_20 | bull | 0.0560 | 0.1888 | 0.2965 | 0.2727 | 0.1887 |
| trend_lowvol | bull | 0.0587 | 0.2029 | 0.2894 | 0.1970 | 0.1732 |
| momentum_reversal | bull | 0.0537 | 0.2025 | 0.2652 | 0.2576 | 0.1668 |
| rsi_vol_combo | bull | 0.0486 | 0.1884 | 0.2577 | 0.2045 | 0.1552 |
| fund_roe | bull | 0.0273 | 0.2237 | 0.1221 | 0.1061 | 0.0675 |
| fund_score | bear | 0.1327 | 0.1612 | 0.8233 | 0.6438 | 0.6767 |
| momentum_reversal | bear | 0.1502 | 0.1945 | 0.7724 | 0.5616 | 0.6031 |
| fund_profit_growth | bear | 0.1301 | 0.1728 | 0.7529 | 0.5068 | 0.5672 |
| rsi_vol_combo | bear | 0.1117 | 0.1669 | 0.6696 | 0.5890 | 0.5320 |
| mom_x_lowvol_20_20 | bear | 0.1582 | 0.2348 | 0.6738 | 0.4521 | 0.4892 |
| turnover_stability | bear | 0.0742 | 0.1331 | 0.5577 | 0.6164 | 0.4508 |
| trend_lowvol | bear | 0.1161 | 0.1778 | 0.6530 | 0.3151 | 0.4294 |
| bb_width_20 | bear | 0.0966 | 0.2221 | 0.4349 | 0.4521 | 0.3158 |
| fund_roe | bear | 0.0721 | 0.2484 | 0.2903 | 0.1781 | 0.1710 |
| vol_confirm | bear | 0.0280 | 0.1470 | 0.1903 | 0.1507 | 0.1095 |

### 北交所概念

- **Neutral**: ['trend_lowvol', 'fund_pb', 'momentum_reversal'] (单因子IC=0.0758, 组合IC=0.1096)
  - weights: [0.3776, 0.3262, 0.2961]
- **Bull**: ['fund_pb', 'low_downside', 'trend_lowvol'] (单因子IC=0.1154, 组合IC=0.154)
  - bull_weights: [0.3973, 0.355, 0.2477]
- **Bear**: ['trend_lowvol', 'rsi_vol_combo', 'mom_x_lowvol_20_20'] (单因子IC=0.1452, 组合IC=0.1844)
  - bear_weights: [0.4202, 0.2979, 0.2819]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| trend_lowvol | neutral | 0.0851 | 0.2622 | 0.3246 | 0.2714 | 0.2064 |
| fund_pb | neutral | 0.0746 | 0.2572 | 0.2900 | 0.2296 | 0.1783 |
| momentum_reversal | neutral | 0.0679 | 0.2537 | 0.2675 | 0.2098 | 0.1618 |
| mom_x_lowvol_20_20 | neutral | 0.0577 | 0.2423 | 0.2381 | 0.2015 | 0.1430 |
| turnover_stability | neutral | 0.0493 | 0.2185 | 0.2257 | 0.1952 | 0.1349 |
| volatility | neutral | 0.0535 | 0.2434 | 0.2198 | 0.1681 | 0.1284 |
| rsi_vol_combo | neutral | 0.0509 | 0.2509 | 0.2027 | 0.1566 | 0.1172 |
| fund_pe | neutral | 0.0481 | 0.2639 | 0.1822 | 0.1576 | 0.1054 |
| low_downside | neutral | 0.0369 | 0.2617 | 0.1410 | 0.1190 | 0.0789 |
| fund_pb | bull | 0.1275 | 0.2354 | 0.5418 | 0.4848 | 0.4022 |
| low_downside | bull | 0.1172 | 0.2297 | 0.5101 | 0.4091 | 0.3594 |
| trend_lowvol | bull | 0.1015 | 0.2585 | 0.3928 | 0.2765 | 0.2507 |
| fund_pe | bull | 0.0878 | 0.2498 | 0.3516 | 0.2803 | 0.2251 |
| volatility | bull | 0.0762 | 0.2362 | 0.3225 | 0.2424 | 0.2003 |
| momentum_reversal | bull | 0.0714 | 0.2269 | 0.3147 | 0.2424 | 0.1955 |
| fund_profit_growth | bull | 0.0555 | 0.1962 | 0.2828 | 0.2121 | 0.1714 |
| rsi_vol_combo | bull | 0.0579 | 0.2252 | 0.2573 | 0.1894 | 0.1530 |
| mom_x_lowvol_20_20 | bull | 0.0282 | 0.2124 | 0.1328 | 0.1364 | 0.0755 |
| turnover_stability | bull | 0.0271 | 0.2145 | 0.1265 | 0.1061 | 0.0700 |
| trend_lowvol | bear | 0.1667 | 0.2489 | 0.6698 | 0.5068 | 0.5047 |
| rsi_vol_combo | bear | 0.1322 | 0.2531 | 0.5224 | 0.3699 | 0.3578 |
| mom_x_lowvol_20_20 | bear | 0.1367 | 0.2875 | 0.4754 | 0.4247 | 0.3386 |
| momentum_reversal | bear | 0.1490 | 0.3166 | 0.4707 | 0.2055 | 0.2837 |
| bb_width_20 | bear | 0.0641 | 0.2792 | 0.2295 | 0.1233 | 0.1289 |
| fund_gross_margin | bear | 0.0557 | 0.2780 | 0.2002 | 0.2329 | 0.1234 |
| fund_score | bear | 0.0436 | 0.2433 | 0.1791 | 0.1233 | 0.1006 |

### 北斗导航

- **Neutral**: ['mom_x_lowvol_20_20', 'momentum_reversal', 'trend_lowvol'] (单因子IC=0.0949, 组合IC=0.1028)
  - weights: [0.3494, 0.3369, 0.3137]
- **Bull**: ['volatility'] (单因子IC=0.1819, 组合IC=0.1819)
  - bull_weights: [1.0]
- **Bear**: ['momentum_reversal'] (单因子IC=0.1358, 组合IC=0.1358)
  - bear_weights: [1.0]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| mom_x_lowvol_20_20 | neutral | 0.0963 | 0.1959 | 0.4915 | 0.3382 | 0.3288 |
| momentum_reversal | neutral | 0.0939 | 0.1970 | 0.4768 | 0.3299 | 0.3171 |
| trend_lowvol | neutral | 0.0945 | 0.2085 | 0.4529 | 0.3038 | 0.2952 |
| fund_pb | neutral | 0.0732 | 0.1660 | 0.4413 | 0.2944 | 0.2856 |
| volatility | neutral | 0.0799 | 0.2153 | 0.3714 | 0.3152 | 0.2442 |
| rsi_vol_combo | neutral | 0.0585 | 0.1877 | 0.3118 | 0.2317 | 0.1920 |
| fund_profit_growth | neutral | 0.0432 | 0.1542 | 0.2800 | 0.2317 | 0.1725 |
| turnover_stability | neutral | 0.0370 | 0.1393 | 0.2657 | 0.2380 | 0.1645 |
| fund_pe | neutral | 0.0468 | 0.2008 | 0.2331 | 0.1587 | 0.1350 |
| fund_score | neutral | 0.0384 | 0.1787 | 0.2150 | 0.1503 | 0.1237 |
| fund_roe | neutral | 0.0357 | 0.1784 | 0.2000 | 0.1628 | 0.1163 |
| volatility | bull | 0.1819 | 0.1977 | 0.9201 | 0.6439 | 0.7563 |
| low_downside | bull | 0.1193 | 0.1662 | 0.7176 | 0.5833 | 0.5681 |
| trend_lowvol | bull | 0.1503 | 0.1995 | 0.7532 | 0.5076 | 0.5677 |
| fund_pb | bull | 0.1153 | 0.1562 | 0.7384 | 0.4318 | 0.5286 |
| mom_x_lowvol_20_20 | bull | 0.1111 | 0.1855 | 0.5987 | 0.5152 | 0.4536 |
| momentum_reversal | bull | 0.0995 | 0.1830 | 0.5436 | 0.4621 | 0.3974 |
| fund_pe | bull | 0.1042 | 0.1928 | 0.5407 | 0.3712 | 0.3707 |
| rsi_vol_combo | bull | 0.0606 | 0.1632 | 0.3711 | 0.2576 | 0.2334 |
| turnover_stability | bull | 0.0489 | 0.1374 | 0.3557 | 0.2803 | 0.2277 |
| fund_roe | bull | 0.0248 | 0.1402 | 0.1767 | 0.1288 | 0.0997 |
| fund_revenue_growth | bull | 0.0172 | 0.1392 | 0.1234 | 0.1136 | 0.0687 |
| momentum_reversal | bear | 0.1358 | 0.1618 | 0.8391 | 0.5616 | 0.6552 |
| rsi_vol_combo | bear | 0.1109 | 0.1408 | 0.7876 | 0.6438 | 0.6474 |
| mom_x_lowvol_20_20 | bear | 0.1391 | 0.1771 | 0.7859 | 0.4795 | 0.5813 |
| trend_lowvol | bear | 0.0926 | 0.2140 | 0.4327 | 0.1781 | 0.2549 |
| fund_pe | bear | 0.0428 | 0.1776 | 0.2412 | 0.2055 | 0.1454 |
| turnover_stability | bear | 0.0256 | 0.1144 | 0.2241 | 0.2055 | 0.1351 |
| fund_gross_margin | bear | 0.0394 | 0.1865 | 0.2112 | 0.2329 | 0.1302 |

### 区块链

- **Neutral**: ['trend_lowvol', 'momentum_reversal'] (单因子IC=0.0948, 组合IC=0.1035)
  - weights: [0.5152, 0.4848]
- **Bull**: ['volatility', 'low_downside', 'fund_pb'] (单因子IC=0.1167, 组合IC=0.1398)
  - bull_weights: [0.3634, 0.3287, 0.3079]
- **Bear**: ['trend_lowvol', 'mom_x_lowvol_20_20'] (单因子IC=0.1435, 组合IC=0.1674)
  - bear_weights: [0.6204, 0.3796]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| trend_lowvol | neutral | 0.0983 | 0.1845 | 0.5327 | 0.4342 | 0.3820 |
| momentum_reversal | neutral | 0.0913 | 0.1721 | 0.5306 | 0.3549 | 0.3595 |
| mom_x_lowvol_20_20 | neutral | 0.0820 | 0.1696 | 0.4839 | 0.3278 | 0.3212 |
| fund_pb | neutral | 0.0805 | 0.1725 | 0.4668 | 0.2965 | 0.3026 |
| turnover_stability | neutral | 0.0439 | 0.0990 | 0.4431 | 0.3090 | 0.2900 |
| rsi_vol_combo | neutral | 0.0675 | 0.1569 | 0.4303 | 0.3173 | 0.2834 |
| fund_profit_growth | neutral | 0.0461 | 0.1066 | 0.4323 | 0.2213 | 0.2640 |
| volatility | neutral | 0.0795 | 0.2090 | 0.3803 | 0.3027 | 0.2477 |
| fund_score | neutral | 0.0440 | 0.1408 | 0.3126 | 0.2171 | 0.1902 |
| low_downside | neutral | 0.0520 | 0.1980 | 0.2627 | 0.2505 | 0.1643 |
| fund_pe | neutral | 0.0525 | 0.1905 | 0.2758 | 0.1733 | 0.1618 |
| fund_revenue_growth | neutral | 0.0179 | 0.0957 | 0.1871 | 0.1712 | 0.1096 |
| fund_roe | neutral | 0.0323 | 0.1653 | 0.1955 | 0.1023 | 0.1077 |
| volatility | bull | 0.1303 | 0.1637 | 0.7961 | 0.5379 | 0.6122 |
| low_downside | bull | 0.1144 | 0.1557 | 0.7346 | 0.5076 | 0.5537 |
| fund_pb | bull | 0.1053 | 0.1508 | 0.6987 | 0.4848 | 0.5187 |
| turnover_stability | bull | 0.0549 | 0.0938 | 0.5850 | 0.5000 | 0.4388 |
| trend_lowvol | bull | 0.1053 | 0.1854 | 0.5679 | 0.4621 | 0.4152 |
| mom_x_lowvol_20_20 | bull | 0.0761 | 0.1530 | 0.4972 | 0.3939 | 0.3466 |
| momentum_reversal | bull | 0.0727 | 0.1526 | 0.4766 | 0.3939 | 0.3322 |
| fund_pe | bull | 0.0764 | 0.1687 | 0.4531 | 0.2576 | 0.2849 |
| fund_profit_growth | bull | 0.0435 | 0.1088 | 0.4001 | 0.2803 | 0.2561 |
| rsi_vol_combo | bull | 0.0423 | 0.1485 | 0.2846 | 0.2424 | 0.1768 |
| trend_lowvol | bear | 0.1573 | 0.1436 | 1.0958 | 0.6986 | 0.9307 |
| mom_x_lowvol_20_20 | bear | 0.1296 | 0.1746 | 0.7422 | 0.5342 | 0.5694 |
| momentum_reversal | bear | 0.1260 | 0.1851 | 0.6807 | 0.5068 | 0.5129 |
| fund_profit_growth | bear | 0.0547 | 0.0925 | 0.5914 | 0.4521 | 0.4294 |
| fund_revenue_growth | bear | 0.0450 | 0.0977 | 0.4609 | 0.2329 | 0.2841 |
| rsi_vol_combo | bear | 0.0621 | 0.1680 | 0.3695 | 0.4521 | 0.2683 |
| fund_score | bear | 0.0498 | 0.1408 | 0.3537 | 0.1781 | 0.2083 |
| bb_width_20 | bear | 0.0445 | 0.1692 | 0.2630 | 0.2055 | 0.1585 |

### 医废处理

- **Neutral**: ['fund_pb', 'mom_x_lowvol_20_20'] (单因子IC=0.0868, 组合IC=0.1241)
  - weights: [0.5038, 0.4962]
- **Bull**: ['low_downside', 'volatility'] (单因子IC=0.0956, 组合IC=0.1033)
  - bull_weights: [0.5669, 0.4331]
- **Bear**: ['trend_lowvol'] (单因子IC=0.2649, 组合IC=0.2649)
  - bear_weights: [1.0]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| fund_pb | neutral | 0.0910 | 0.2774 | 0.3281 | 0.2129 | 0.1990 |
| mom_x_lowvol_20_20 | neutral | 0.0826 | 0.2653 | 0.3113 | 0.2589 | 0.1960 |
| momentum_reversal | neutral | 0.0793 | 0.2663 | 0.2978 | 0.2526 | 0.1865 |
| fund_profit_growth | neutral | 0.0620 | 0.2078 | 0.2983 | 0.2307 | 0.1835 |
| turnover_stability | neutral | 0.0524 | 0.2040 | 0.2570 | 0.2380 | 0.1591 |
| trend_lowvol | neutral | 0.0729 | 0.2792 | 0.2610 | 0.1712 | 0.1528 |
| fund_pe | neutral | 0.0712 | 0.3163 | 0.2250 | 0.1273 | 0.1268 |
| volatility | neutral | 0.0659 | 0.3039 | 0.2169 | 0.1566 | 0.1255 |
| rsi_vol_combo | neutral | 0.0540 | 0.2544 | 0.2124 | 0.1712 | 0.1244 |
| fund_score | neutral | 0.0478 | 0.2343 | 0.2040 | 0.1399 | 0.1163 |
| fund_gross_margin | neutral | 0.0317 | 0.2129 | 0.1491 | 0.1211 | 0.0836 |
| low_downside | neutral | 0.0303 | 0.2649 | 0.1144 | 0.1086 | 0.0634 |
| low_downside | bull | 0.0998 | 0.2177 | 0.4585 | 0.2576 | 0.2883 |
| volatility | bull | 0.0913 | 0.2600 | 0.3513 | 0.2538 | 0.2202 |
| fund_pe | bull | 0.1005 | 0.2982 | 0.3371 | 0.1742 | 0.1979 |
| wash_sale_score | bull | 0.0547 | 0.1941 | 0.2820 | 0.2584 | 0.1774 |
| fund_profit_growth | bull | 0.0546 | 0.1941 | 0.2811 | 0.2576 | 0.1767 |
| fund_roe | bull | 0.0700 | 0.2467 | 0.2838 | 0.2424 | 0.1763 |
| fund_gross_margin | bull | 0.0564 | 0.2066 | 0.2727 | 0.1591 | 0.1581 |
| fund_score | bull | 0.0571 | 0.2208 | 0.2587 | 0.2159 | 0.1573 |
| fund_pb | bull | 0.0574 | 0.2573 | 0.2232 | 0.1439 | 0.1277 |
| fund_revenue_growth | bull | 0.0346 | 0.1675 | 0.2068 | 0.1477 | 0.1187 |
| trend_lowvol | bull | 0.0568 | 0.2798 | 0.2028 | 0.1439 | 0.1160 |
| trend_lowvol | bear | 0.2649 | 0.2390 | 1.1083 | 0.7808 | 0.9869 |
| momentum_reversal | bear | 0.2154 | 0.2495 | 0.8634 | 0.6164 | 0.6978 |
| mom_x_lowvol_20_20 | bear | 0.1649 | 0.2661 | 0.6197 | 0.3973 | 0.4330 |
| rsi_vol_combo | bear | 0.1269 | 0.2244 | 0.5656 | 0.3699 | 0.3874 |
| bb_width_20 | bear | 0.1265 | 0.2645 | 0.4781 | 0.3151 | 0.3144 |
| fund_pb | bear | 0.1168 | 0.2837 | 0.4116 | 0.1507 | 0.2368 |

### 医疗器械概念

- **Neutral**: ['fund_pb', 'volatility', 'trend_lowvol'] (单因子IC=0.0709, 组合IC=0.0969)
  - weights: [0.3487, 0.3268, 0.3245]
- **Bull**: ['turnover_stability', 'low_downside', 'volatility'] (单因子IC=0.0676, 组合IC=0.0787)
  - bull_weights: [0.3815, 0.3135, 0.305]
- **Bear**: ['momentum_reversal'] (单因子IC=0.1051, 组合IC=0.1051)
  - bear_weights: [1.0]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| fund_pb | neutral | 0.0716 | 0.1630 | 0.4392 | 0.2923 | 0.2838 |
| volatility | neutral | 0.0688 | 0.1708 | 0.4026 | 0.3215 | 0.2660 |
| trend_lowvol | neutral | 0.0725 | 0.1762 | 0.4114 | 0.2839 | 0.2641 |
| momentum_reversal | neutral | 0.0620 | 0.1689 | 0.3673 | 0.2484 | 0.2293 |
| turnover_stability | neutral | 0.0345 | 0.0955 | 0.3611 | 0.2630 | 0.2281 |
| mom_x_lowvol_20_20 | neutral | 0.0582 | 0.1606 | 0.3627 | 0.2380 | 0.2245 |
| low_downside | neutral | 0.0473 | 0.1671 | 0.2827 | 0.2484 | 0.1765 |
| fund_profit_growth | neutral | 0.0352 | 0.1221 | 0.2885 | 0.1816 | 0.1705 |
| rsi_vol_combo | neutral | 0.0416 | 0.1525 | 0.2726 | 0.2276 | 0.1673 |
| fund_pe | neutral | 0.0339 | 0.1376 | 0.2460 | 0.1461 | 0.1410 |
| turnover_stability | bull | 0.0496 | 0.0829 | 0.5980 | 0.4242 | 0.4259 |
| low_downside | bull | 0.0724 | 0.1441 | 0.5022 | 0.3939 | 0.3500 |
| volatility | bull | 0.0808 | 0.1581 | 0.5108 | 0.3333 | 0.3405 |
| fund_pb | bull | 0.0737 | 0.1690 | 0.4361 | 0.2879 | 0.2808 |
| trend_lowvol | bull | 0.0506 | 0.1715 | 0.2950 | 0.1591 | 0.1710 |
| momentum_reversal | bull | 0.0362 | 0.1484 | 0.2442 | 0.2273 | 0.1499 |
| fund_pe | bull | 0.0377 | 0.1459 | 0.2581 | 0.1288 | 0.1457 |
| mom_x_lowvol_20_20 | bull | 0.0336 | 0.1467 | 0.2289 | 0.1364 | 0.1300 |
| fund_profit_growth | bull | 0.0221 | 0.1111 | 0.1990 | 0.1439 | 0.1138 |
| rsi_vol_combo | bull | 0.0158 | 0.1360 | 0.1162 | 0.1591 | 0.0673 |
| momentum_reversal | bear | 0.1051 | 0.1982 | 0.5304 | 0.3425 | 0.3560 |
| mom_x_lowvol_20_20 | bear | 0.0998 | 0.1957 | 0.5101 | 0.3151 | 0.3354 |
| turnover_stability | bear | 0.0310 | 0.0701 | 0.4426 | 0.3699 | 0.3031 |
| trend_lowvol | bear | 0.0874 | 0.2015 | 0.4338 | 0.2329 | 0.2674 |
| fund_profit_growth | bear | 0.0426 | 0.1140 | 0.3732 | 0.3973 | 0.2608 |
| rsi_vol_combo | bear | 0.0673 | 0.1682 | 0.4003 | 0.2329 | 0.2468 |
| fund_pb | bear | 0.0392 | 0.1391 | 0.2822 | 0.1233 | 0.1585 |
| fund_score | bear | 0.0381 | 0.1600 | 0.2379 | 0.2329 | 0.1467 |
| fund_pe | bear | 0.0252 | 0.1375 | 0.1830 | 0.2877 | 0.1178 |

### 医美概念

- **Neutral**: ['fund_pb', 'trend_lowvol', 'turnover_stability'] (单因子IC=0.0721, 组合IC=0.1193)
  - weights: [0.3441, 0.3387, 0.3173]
- **Bull**: ['momentum_reversal', 'trend_lowvol', 'turnover_stability'] (单因子IC=0.0961, 组合IC=0.1218)
  - bull_weights: [0.354, 0.3243, 0.3217]
- **Bear**: ['momentum_reversal', 'fund_pe', 'rsi_vol_combo'] (单因子IC=0.0923, 组合IC=0.1202)
  - bear_weights: [0.3447, 0.3428, 0.3125]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| fund_pb | neutral | 0.0818 | 0.2005 | 0.4078 | 0.2902 | 0.2630 |
| trend_lowvol | neutral | 0.0833 | 0.2129 | 0.3912 | 0.3236 | 0.2589 |
| turnover_stability | neutral | 0.0513 | 0.1400 | 0.3665 | 0.3236 | 0.2425 |
| low_downside | neutral | 0.0711 | 0.2048 | 0.3473 | 0.3361 | 0.2320 |
| volatility | neutral | 0.0705 | 0.2033 | 0.3465 | 0.3194 | 0.2286 |
| fund_pe | neutral | 0.0670 | 0.1947 | 0.3442 | 0.2777 | 0.2199 |
| fund_profit_growth | neutral | 0.0412 | 0.1522 | 0.2708 | 0.1712 | 0.1586 |
| momentum_reversal | neutral | 0.0493 | 0.1876 | 0.2629 | 0.1482 | 0.1509 |
| mom_x_lowvol_20_20 | neutral | 0.0374 | 0.1767 | 0.2119 | 0.1148 | 0.1181 |
| fund_score | neutral | 0.0308 | 0.1669 | 0.1848 | 0.1921 | 0.1102 |
| rsi_vol_combo | neutral | 0.0339 | 0.1818 | 0.1867 | 0.1148 | 0.1040 |
| momentum_reversal | bull | 0.1093 | 0.1931 | 0.5662 | 0.4394 | 0.4075 |
| trend_lowvol | bull | 0.1064 | 0.1943 | 0.5475 | 0.3636 | 0.3733 |
| turnover_stability | bull | 0.0725 | 0.1358 | 0.5342 | 0.3864 | 0.3703 |
| volatility | bull | 0.0886 | 0.1766 | 0.5019 | 0.3561 | 0.3403 |
| rsi_vol_combo | bull | 0.0806 | 0.1724 | 0.4678 | 0.3333 | 0.3119 |
| mom_x_lowvol_20_20 | bull | 0.0790 | 0.1885 | 0.4190 | 0.3258 | 0.2778 |
| low_downside | bull | 0.0731 | 0.1727 | 0.4235 | 0.2955 | 0.2743 |
| fund_pb | bull | 0.0737 | 0.2058 | 0.3581 | 0.2614 | 0.2258 |
| fund_pe | bull | 0.0503 | 0.2136 | 0.2354 | 0.1818 | 0.1391 |
| momentum_reversal | bear | 0.1030 | 0.2153 | 0.4784 | 0.2877 | 0.3080 |
| fund_pe | bear | 0.0802 | 0.1793 | 0.4473 | 0.3699 | 0.3064 |
| rsi_vol_combo | bear | 0.0938 | 0.1978 | 0.4741 | 0.1781 | 0.2793 |
| mom_x_lowvol_20_20 | bear | 0.0866 | 0.2109 | 0.4103 | 0.3151 | 0.2698 |
| trend_lowvol | bear | 0.0715 | 0.2258 | 0.3167 | 0.2877 | 0.2039 |
| fund_pb | bear | 0.0686 | 0.2027 | 0.3382 | 0.1781 | 0.1992 |
| turnover_stability | bear | 0.0314 | 0.1188 | 0.2646 | 0.2603 | 0.1668 |

### 半导体概念

- **Neutral**: ['mom_x_lowvol_20_20', 'momentum_reversal', 'trend_lowvol'] (单因子IC=0.0734, 组合IC=0.0824)
  - weights: [0.3588, 0.3471, 0.2941]
- **Bull**: ['volatility', 'fund_pb', 'low_downside'] (单因子IC=0.0743, 组合IC=0.0954)
  - bull_weights: [0.3747, 0.3156, 0.3097]
- **Bear**: ['mom_x_lowvol_20_20'] (单因子IC=0.1414, 组合IC=0.1414)
  - bear_weights: [1.0]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| mom_x_lowvol_20_20 | neutral | 0.0750 | 0.1547 | 0.4849 | 0.3006 | 0.3153 |
| momentum_reversal | neutral | 0.0756 | 0.1580 | 0.4783 | 0.2756 | 0.3050 |
| trend_lowvol | neutral | 0.0695 | 0.1699 | 0.4092 | 0.2630 | 0.2584 |
| rsi_vol_combo | neutral | 0.0605 | 0.1543 | 0.3920 | 0.2756 | 0.2500 |
| fund_pe | neutral | 0.0559 | 0.1483 | 0.3769 | 0.2860 | 0.2423 |
| turnover_stability | neutral | 0.0350 | 0.0959 | 0.3648 | 0.2526 | 0.2284 |
| volatility | neutral | 0.0602 | 0.1700 | 0.3542 | 0.2672 | 0.2244 |
| fund_pb | neutral | 0.0602 | 0.1758 | 0.3425 | 0.2401 | 0.2124 |
| fund_profit_growth | neutral | 0.0405 | 0.1363 | 0.2975 | 0.2380 | 0.1841 |
| fund_revenue_growth | neutral | 0.0279 | 0.1247 | 0.2235 | 0.1775 | 0.1316 |
| fund_score | neutral | 0.0354 | 0.1674 | 0.2115 | 0.1879 | 0.1256 |
| low_downside | neutral | 0.0276 | 0.1609 | 0.1717 | 0.1566 | 0.0993 |
| fund_roe | neutral | 0.0253 | 0.1552 | 0.1630 | 0.1002 | 0.0897 |
| volatility | bull | 0.0861 | 0.1555 | 0.5539 | 0.3561 | 0.3756 |
| fund_pb | bull | 0.0754 | 0.1617 | 0.4666 | 0.3561 | 0.3163 |
| low_downside | bull | 0.0613 | 0.1338 | 0.4579 | 0.3561 | 0.3105 |
| trend_lowvol | bull | 0.0707 | 0.1652 | 0.4278 | 0.3561 | 0.2900 |
| turnover_stability | bull | 0.0361 | 0.0864 | 0.4183 | 0.3409 | 0.2805 |
| mom_x_lowvol_20_20 | bull | 0.0559 | 0.1510 | 0.3706 | 0.2652 | 0.2344 |
| momentum_reversal | bull | 0.0543 | 0.1532 | 0.3544 | 0.2348 | 0.2188 |
| fund_pe | bull | 0.0487 | 0.1402 | 0.3473 | 0.2576 | 0.2184 |
| rsi_vol_combo | bull | 0.0425 | 0.1441 | 0.2950 | 0.2424 | 0.1833 |
| fund_profit_growth | bull | 0.0249 | 0.1326 | 0.1876 | 0.1970 | 0.1123 |
| fund_revenue_growth | bull | 0.0240 | 0.1379 | 0.1743 | 0.1439 | 0.0997 |
| fund_score | bull | 0.0240 | 0.1600 | 0.1498 | 0.1212 | 0.0840 |
| mom_x_lowvol_20_20 | bear | 0.1414 | 0.1702 | 0.8308 | 0.5068 | 0.6260 |
| momentum_reversal | bear | 0.1306 | 0.1629 | 0.8015 | 0.4521 | 0.5819 |
| rsi_vol_combo | bear | 0.0861 | 0.1314 | 0.6556 | 0.5068 | 0.4940 |
| trend_lowvol | bear | 0.0844 | 0.1944 | 0.4340 | 0.1781 | 0.2556 |
| bb_width_20 | bear | 0.0592 | 0.1579 | 0.3752 | 0.1781 | 0.2210 |
| turnover_stability | bear | 0.0189 | 0.1069 | 0.1770 | 0.1781 | 0.1042 |

### 华为昇腾

- **Neutral**: ['momentum_reversal', 'trend_lowvol', 'rsi_vol_combo'] (单因子IC=0.0827, 组合IC=0.0988)
  - weights: [0.3409, 0.3382, 0.321]
- **Bull**: ['fund_pb', 'volatility', 'trend_lowvol'] (单因子IC=0.118, 组合IC=0.1445)
  - bull_weights: [0.3703, 0.3293, 0.3004]
- **Bear**: ['fund_profit_growth', 'fund_revenue_growth'] (单因子IC=0.0994, 组合IC=0.1086)
  - bear_weights: [0.5063, 0.4937]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| momentum_reversal | neutral | 0.0846 | 0.2061 | 0.4105 | 0.2985 | 0.2665 |
| trend_lowvol | neutral | 0.0879 | 0.2164 | 0.4063 | 0.3017 | 0.2644 |
| rsi_vol_combo | neutral | 0.0756 | 0.1988 | 0.3804 | 0.3194 | 0.2510 |
| volatility | neutral | 0.0787 | 0.2166 | 0.3634 | 0.3486 | 0.2451 |
| fund_pb | neutral | 0.0585 | 0.1736 | 0.3371 | 0.3069 | 0.2203 |
| mom_x_lowvol_20_20 | neutral | 0.0679 | 0.2046 | 0.3319 | 0.2192 | 0.2023 |
| fund_profit_growth | neutral | 0.0521 | 0.1762 | 0.2953 | 0.2276 | 0.1813 |
| low_downside | neutral | 0.0558 | 0.2042 | 0.2733 | 0.2610 | 0.1723 |
| fund_score | neutral | 0.0525 | 0.1930 | 0.2719 | 0.2338 | 0.1678 |
| fund_pe | neutral | 0.0631 | 0.2313 | 0.2728 | 0.2276 | 0.1674 |
| fund_roe | neutral | 0.0528 | 0.2024 | 0.2610 | 0.1983 | 0.1564 |
| turnover_stability | neutral | 0.0319 | 0.1806 | 0.1769 | 0.1524 | 0.1019 |
| fund_pb | bull | 0.1039 | 0.1459 | 0.7122 | 0.5000 | 0.5341 |
| volatility | bull | 0.1208 | 0.1888 | 0.6398 | 0.4848 | 0.4750 |
| trend_lowvol | bull | 0.1294 | 0.2160 | 0.5990 | 0.4470 | 0.4333 |
| momentum_reversal | bull | 0.1048 | 0.2245 | 0.4665 | 0.3712 | 0.3199 |
| low_downside | bull | 0.0790 | 0.1757 | 0.4496 | 0.3636 | 0.3065 |
| mom_x_lowvol_20_20 | bull | 0.0985 | 0.2274 | 0.4331 | 0.3106 | 0.2838 |
| fund_pe | bull | 0.0599 | 0.1974 | 0.3037 | 0.2197 | 0.1852 |
| rsi_vol_combo | bull | 0.0583 | 0.2180 | 0.2673 | 0.3485 | 0.1802 |
| turnover_stability | bull | 0.0397 | 0.1868 | 0.2127 | 0.1970 | 0.1273 |
| fund_profit_growth | bear | 0.1074 | 0.2182 | 0.4922 | 0.3425 | 0.3304 |
| fund_revenue_growth | bear | 0.0914 | 0.1982 | 0.4612 | 0.3973 | 0.3222 |
| fund_score | bear | 0.0941 | 0.2530 | 0.3722 | 0.2603 | 0.2345 |
| volatility | bear | 0.0330 | 0.1852 | 0.1780 | 0.1507 | 0.1024 |

### 华为概念

- **Neutral**: ['momentum_reversal', 'fund_pb', 'mom_x_lowvol_20_20'] (单因子IC=0.0772, 组合IC=0.1043)
  - weights: [0.3363, 0.3351, 0.3286]
- **Bull**: ['low_downside', 'volatility', 'fund_pb'] (单因子IC=0.0907, 组合IC=0.1103)
  - bull_weights: [0.3524, 0.3351, 0.3124]
- **Bear**: ['momentum_reversal', 'mom_x_lowvol_20_20', 'trend_lowvol'] (单因子IC=0.0897, 组合IC=0.0994)
  - bear_weights: [0.3416, 0.3348, 0.3236]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| momentum_reversal | neutral | 0.0811 | 0.1514 | 0.5359 | 0.3820 | 0.3703 |
| fund_pb | neutral | 0.0721 | 0.1333 | 0.5412 | 0.3633 | 0.3689 |
| mom_x_lowvol_20_20 | neutral | 0.0785 | 0.1483 | 0.5292 | 0.3674 | 0.3619 |
| trend_lowvol | neutral | 0.0859 | 0.1709 | 0.5028 | 0.4071 | 0.3538 |
| rsi_vol_combo | neutral | 0.0627 | 0.1404 | 0.4464 | 0.3236 | 0.2954 |
| volatility | neutral | 0.0724 | 0.1733 | 0.4177 | 0.3361 | 0.2791 |
| turnover_stability | neutral | 0.0320 | 0.0777 | 0.4114 | 0.3382 | 0.2752 |
| fund_pe | neutral | 0.0510 | 0.1319 | 0.3868 | 0.2881 | 0.2491 |
| fund_profit_growth | neutral | 0.0439 | 0.1150 | 0.3816 | 0.3027 | 0.2485 |
| fund_score | neutral | 0.0430 | 0.1533 | 0.2802 | 0.1733 | 0.1644 |
| low_downside | neutral | 0.0401 | 0.1613 | 0.2485 | 0.2630 | 0.1569 |
| fund_revenue_growth | neutral | 0.0238 | 0.1131 | 0.2105 | 0.1816 | 0.1244 |
| fund_roe | neutral | 0.0313 | 0.1558 | 0.2007 | 0.1044 | 0.1108 |
| low_downside | bull | 0.0885 | 0.1279 | 0.6922 | 0.5152 | 0.5244 |
| volatility | bull | 0.0989 | 0.1496 | 0.6616 | 0.5076 | 0.4987 |
| fund_pb | bull | 0.0845 | 0.1364 | 0.6199 | 0.5000 | 0.4649 |
| turnover_stability | bull | 0.0390 | 0.0709 | 0.5507 | 0.3712 | 0.3776 |
| fund_pe | bull | 0.0619 | 0.1385 | 0.4468 | 0.4318 | 0.3199 |
| trend_lowvol | bull | 0.0743 | 0.1689 | 0.4400 | 0.4015 | 0.3083 |
| mom_x_lowvol_20_20 | bull | 0.0573 | 0.1408 | 0.4070 | 0.3636 | 0.2775 |
| momentum_reversal | bull | 0.0499 | 0.1430 | 0.3490 | 0.3712 | 0.2393 |
| fund_profit_growth | bull | 0.0365 | 0.1152 | 0.3170 | 0.2576 | 0.1993 |
| fund_score | bull | 0.0455 | 0.1630 | 0.2790 | 0.1364 | 0.1585 |
| fund_revenue_growth | bull | 0.0331 | 0.1240 | 0.2665 | 0.1364 | 0.1514 |
| fund_roe | bull | 0.0339 | 0.1572 | 0.2159 | 0.1439 | 0.1235 |
| rsi_vol_combo | bull | 0.0252 | 0.1313 | 0.1920 | 0.1591 | 0.1113 |
| momentum_reversal | bear | 0.0891 | 0.1637 | 0.5446 | 0.4247 | 0.3879 |
| mom_x_lowvol_20_20 | bear | 0.0885 | 0.1658 | 0.5338 | 0.4247 | 0.3802 |
| trend_lowvol | bear | 0.0916 | 0.1638 | 0.5590 | 0.3151 | 0.3676 |
| rsi_vol_combo | bear | 0.0559 | 0.1239 | 0.4509 | 0.4795 | 0.3335 |
| turnover_stability | bear | 0.0207 | 0.0772 | 0.2684 | 0.2055 | 0.1618 |

### 华为汽车

- **Neutral**: ['trend_lowvol', 'momentum_reversal', 'fund_pb'] (单因子IC=0.0825, 组合IC=0.1152)
  - weights: [0.3445, 0.3433, 0.3122]
- **Bull**: ['trend_lowvol'] (单因子IC=0.0816, 组合IC=0.0816)
  - bull_weights: [1.0]
- **Bear**: ['fund_revenue_growth', 'fund_pe', 'momentum_reversal'] (单因子IC=0.0618, 组合IC=0.0984)
  - bear_weights: [0.3825, 0.3141, 0.3034]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| trend_lowvol | neutral | 0.0909 | 0.1763 | 0.5157 | 0.3445 | 0.3467 |
| momentum_reversal | neutral | 0.0826 | 0.1664 | 0.4962 | 0.3925 | 0.3455 |
| fund_pb | neutral | 0.0739 | 0.1589 | 0.4652 | 0.3507 | 0.3142 |
| mom_x_lowvol_20_20 | neutral | 0.0738 | 0.1625 | 0.4541 | 0.3528 | 0.3071 |
| fund_pe | neutral | 0.0729 | 0.1600 | 0.4554 | 0.3194 | 0.3004 |
| rsi_vol_combo | neutral | 0.0645 | 0.1589 | 0.4063 | 0.3299 | 0.2701 |
| fund_profit_growth | neutral | 0.0531 | 0.1422 | 0.3732 | 0.2735 | 0.2376 |
| volatility | neutral | 0.0632 | 0.1782 | 0.3544 | 0.2756 | 0.2261 |
| fund_score | neutral | 0.0541 | 0.1578 | 0.3427 | 0.2505 | 0.2143 |
| fund_revenue_growth | neutral | 0.0305 | 0.1345 | 0.2270 | 0.1608 | 0.1317 |
| low_downside | neutral | 0.0349 | 0.1704 | 0.2049 | 0.1795 | 0.1208 |
| fund_roe | neutral | 0.0322 | 0.1691 | 0.1907 | 0.1482 | 0.1095 |
| turnover_stability | neutral | 0.0231 | 0.1252 | 0.1845 | 0.1754 | 0.1084 |
| trend_lowvol | bull | 0.0816 | 0.1867 | 0.4373 | 0.2803 | 0.2799 |
| volatility | bull | 0.0679 | 0.1750 | 0.3879 | 0.3636 | 0.2645 |
| momentum_reversal | bull | 0.0617 | 0.1660 | 0.3718 | 0.2652 | 0.2352 |
| low_downside | bull | 0.0548 | 0.1504 | 0.3641 | 0.2348 | 0.2248 |
| turnover_stability | bull | 0.0341 | 0.1022 | 0.3341 | 0.3258 | 0.2215 |
| rsi_vol_combo | bull | 0.0431 | 0.1397 | 0.3083 | 0.2576 | 0.1939 |
| mom_x_lowvol_20_20 | bull | 0.0484 | 0.1599 | 0.3027 | 0.2424 | 0.1880 |
| fund_gross_margin | bull | 0.0258 | 0.1301 | 0.1984 | 0.1439 | 0.1135 |
| fund_score | bull | 0.0150 | 0.1547 | 0.0972 | 0.1364 | 0.0552 |
| fund_revenue_growth | bear | 0.0562 | 0.1197 | 0.4694 | 0.2603 | 0.2958 |
| fund_pe | bear | 0.0483 | 0.1362 | 0.3546 | 0.3699 | 0.2429 |
| momentum_reversal | bear | 0.0809 | 0.2079 | 0.3892 | 0.2055 | 0.2346 |
| fund_pb | bear | 0.0625 | 0.1849 | 0.3381 | 0.1781 | 0.1991 |
| fund_gross_margin | bear | 0.0407 | 0.1611 | 0.2528 | 0.2877 | 0.1627 |
| mom_x_lowvol_20_20 | bear | 0.0596 | 0.2138 | 0.2789 | 0.1233 | 0.1566 |
| fund_score | bear | 0.0507 | 0.1896 | 0.2674 | 0.1233 | 0.1502 |
| fund_profit_growth | bear | 0.0391 | 0.1543 | 0.2533 | 0.1507 | 0.1457 |
| rsi_vol_combo | bear | 0.0424 | 0.1684 | 0.2520 | 0.1507 | 0.1450 |
| trend_lowvol | bear | 0.0561 | 0.2249 | 0.2495 | 0.1507 | 0.1436 |

### 华为海思

- **Neutral**: ['trend_lowvol', 'momentum_reversal'] (单因子IC=0.0999, 组合IC=0.1129)
  - weights: [0.5119, 0.4881]
- **Bull**: ['fund_pe', 'low_downside', 'fund_pb'] (单因子IC=0.0931, 组合IC=0.1224)
  - bull_weights: [0.3555, 0.3517, 0.2928]
- **Bear**: ['rsi_vol_combo', 'fund_revenue_growth', 'volatility'] (单因子IC=0.0573, 组合IC=0.094)
  - bear_weights: [0.3802, 0.3443, 0.2755]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| trend_lowvol | neutral | 0.1003 | 0.2326 | 0.4312 | 0.3267 | 0.2861 |
| momentum_reversal | neutral | 0.0995 | 0.2381 | 0.4181 | 0.3048 | 0.2727 |
| mom_x_lowvol_20_20 | neutral | 0.0874 | 0.2389 | 0.3657 | 0.2568 | 0.2298 |
| fund_pb | neutral | 0.0788 | 0.2283 | 0.3453 | 0.2129 | 0.2094 |
| rsi_vol_combo | neutral | 0.0782 | 0.2336 | 0.3350 | 0.2390 | 0.2075 |
| volatility | neutral | 0.0578 | 0.2347 | 0.2463 | 0.1712 | 0.1443 |
| fund_pe | neutral | 0.0544 | 0.2277 | 0.2388 | 0.1691 | 0.1396 |
| turnover_stability | neutral | 0.0435 | 0.1928 | 0.2256 | 0.1649 | 0.1314 |
| fund_score | neutral | 0.0320 | 0.2129 | 0.1504 | 0.1576 | 0.0870 |
| fund_profit_growth | neutral | 0.0298 | 0.2071 | 0.1440 | 0.1200 | 0.0806 |
| low_downside | neutral | 0.0252 | 0.2198 | 0.1146 | 0.1169 | 0.0640 |
| fund_pe | bull | 0.0930 | 0.1997 | 0.4659 | 0.3030 | 0.3036 |
| low_downside | bull | 0.0867 | 0.1904 | 0.4556 | 0.3182 | 0.3003 |
| fund_pb | bull | 0.0995 | 0.2533 | 0.3928 | 0.2727 | 0.2500 |
| trend_lowvol | bull | 0.0747 | 0.2564 | 0.2914 | 0.2045 | 0.1755 |
| momentum_reversal | bull | 0.0670 | 0.2272 | 0.2951 | 0.1894 | 0.1755 |
| volatility | bull | 0.0555 | 0.2203 | 0.2520 | 0.1667 | 0.1470 |
| mom_x_lowvol_20_20 | bull | 0.0471 | 0.2140 | 0.2202 | 0.1970 | 0.1318 |
| fund_gross_margin | bull | 0.0417 | 0.1938 | 0.2155 | 0.1818 | 0.1273 |
| turnover_stability | bull | 0.0379 | 0.1937 | 0.1957 | 0.2045 | 0.1179 |
| fund_roe | bull | 0.0212 | 0.1920 | 0.1106 | 0.1364 | 0.0628 |
| rsi_vol_combo | bull | 0.0257 | 0.2411 | 0.1068 | 0.1591 | 0.0619 |
| rsi_vol_combo | bear | 0.0625 | 0.1915 | 0.3264 | 0.2055 | 0.1967 |
| fund_revenue_growth | bear | 0.0479 | 0.1621 | 0.2956 | 0.2055 | 0.1781 |
| volatility | bear | 0.0614 | 0.2537 | 0.2420 | 0.1781 | 0.1426 |
| fund_gross_margin | bear | 0.0389 | 0.2137 | 0.1822 | 0.1233 | 0.1023 |

### 单抗概念

- **Neutral**: ['volatility', 'low_downside', 'fund_pb'] (单因子IC=0.0563, 组合IC=0.0654)
  - weights: [0.3587, 0.3409, 0.3004]
- **Bull**: ['turnover_stability', 'fund_pe', 'low_downside'] (单因子IC=0.0941, 组合IC=0.1044)
  - bull_weights: [0.3969, 0.3112, 0.2919]
- **Bear**: ['fund_revenue_growth', 'fund_profit_growth', 'momentum_reversal'] (单因子IC=0.1, 组合IC=0.1451)
  - bear_weights: [0.4301, 0.357, 0.2129]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| volatility | neutral | 0.0606 | 0.2507 | 0.2415 | 0.2171 | 0.1470 |
| low_downside | neutral | 0.0522 | 0.2304 | 0.2267 | 0.2328 | 0.1397 |
| fund_pb | neutral | 0.0563 | 0.2605 | 0.2160 | 0.1399 | 0.1231 |
| fund_pe | neutral | 0.0457 | 0.2361 | 0.1935 | 0.1733 | 0.1135 |
| fund_profit_growth | neutral | 0.0287 | 0.1962 | 0.1464 | 0.1169 | 0.0817 |
| fund_revenue_growth | neutral | 0.0183 | 0.2196 | 0.0831 | 0.1065 | 0.0460 |
| turnover_stability | bull | 0.0890 | 0.1574 | 0.5655 | 0.4280 | 0.4038 |
| fund_pe | bull | 0.0926 | 0.2005 | 0.4617 | 0.3712 | 0.3166 |
| low_downside | bull | 0.1008 | 0.2226 | 0.4531 | 0.3106 | 0.2969 |
| volatility | bull | 0.0884 | 0.2262 | 0.3909 | 0.3409 | 0.2621 |
| fund_pb | bull | 0.0693 | 0.2730 | 0.2538 | 0.1818 | 0.1500 |
| fund_roe | bull | 0.0458 | 0.1813 | 0.2528 | 0.1212 | 0.1417 |
| stroke_phase | bull | 0.0256 | 0.2031 | 0.1260 | 0.1212 | 0.0706 |
| rsi_vol_combo | bull | 0.0242 | 0.1998 | 0.1211 | 0.1364 | 0.0688 |
| top_fractal_volume | bull | 0.0185 | 0.1701 | 0.1085 | 0.1327 | 0.0615 |
| fund_revenue_growth | bear | 0.1202 | 0.1955 | 0.6147 | 0.4521 | 0.4463 |
| fund_profit_growth | bear | 0.0955 | 0.1872 | 0.5102 | 0.4521 | 0.3704 |
| momentum_reversal | bear | 0.0843 | 0.2301 | 0.3666 | 0.2055 | 0.2210 |
| trend_lowvol | bear | 0.0755 | 0.2385 | 0.3167 | 0.2877 | 0.2039 |
| mom_x_lowvol_20_20 | bear | 0.0634 | 0.2124 | 0.2984 | 0.2055 | 0.1799 |
| fund_score | bear | 0.0627 | 0.2366 | 0.2649 | 0.1781 | 0.1561 |

### 卫星互联网

- **Neutral**: ['trend_lowvol', 'momentum_reversal'] (单因子IC=0.0996, 组合IC=0.1099)
  - weights: [0.5167, 0.4833]
- **Bull**: ['volatility', 'fund_pb', 'low_downside'] (单因子IC=0.1015, 组合IC=0.1273)
  - bull_weights: [0.3683, 0.3641, 0.2676]
- **Bear**: ['momentum_reversal', 'mom_x_lowvol_20_20', 'trend_lowvol'] (单因子IC=0.0993, 组合IC=0.1078)
  - bear_weights: [0.3614, 0.3254, 0.3132]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| trend_lowvol | neutral | 0.1028 | 0.2121 | 0.4844 | 0.4008 | 0.3393 |
| momentum_reversal | neutral | 0.0964 | 0.2041 | 0.4721 | 0.3445 | 0.3174 |
| mom_x_lowvol_20_20 | neutral | 0.0942 | 0.1995 | 0.4720 | 0.3048 | 0.3080 |
| fund_pb | neutral | 0.0752 | 0.1641 | 0.4584 | 0.3058 | 0.2993 |
| volatility | neutral | 0.0769 | 0.1973 | 0.3896 | 0.3069 | 0.2546 |
| fund_pe | neutral | 0.0615 | 0.1742 | 0.3531 | 0.2860 | 0.2270 |
| fund_profit_growth | neutral | 0.0541 | 0.1476 | 0.3666 | 0.2317 | 0.2258 |
| rsi_vol_combo | neutral | 0.0686 | 0.1943 | 0.3529 | 0.2589 | 0.2222 |
| turnover_stability | neutral | 0.0421 | 0.1340 | 0.3144 | 0.2443 | 0.1956 |
| fund_score | neutral | 0.0489 | 0.1934 | 0.2530 | 0.1983 | 0.1516 |
| low_downside | neutral | 0.0322 | 0.1852 | 0.1738 | 0.1273 | 0.0980 |
| fund_roe | neutral | 0.0350 | 0.1992 | 0.1757 | 0.1148 | 0.0980 |
| fund_revenue_growth | neutral | 0.0240 | 0.1519 | 0.1580 | 0.1628 | 0.0919 |
| volatility | bull | 0.1166 | 0.1808 | 0.6448 | 0.5076 | 0.4860 |
| fund_pb | bull | 0.0973 | 0.1465 | 0.6640 | 0.4470 | 0.4804 |
| low_downside | bull | 0.0906 | 0.1778 | 0.5094 | 0.3864 | 0.3531 |
| fund_pe | bull | 0.0704 | 0.1808 | 0.3893 | 0.3864 | 0.2698 |
| fund_revenue_growth | bull | 0.0499 | 0.1361 | 0.3666 | 0.3258 | 0.2430 |
| trend_lowvol | bull | 0.0799 | 0.2063 | 0.3871 | 0.2197 | 0.2361 |
| mom_x_lowvol_20_20 | bull | 0.0666 | 0.1906 | 0.3496 | 0.2576 | 0.2198 |
| fund_profit_growth | bull | 0.0443 | 0.1438 | 0.3081 | 0.2500 | 0.1926 |
| momentum_reversal | bull | 0.0596 | 0.1934 | 0.3079 | 0.2273 | 0.1890 |
| fund_score | bull | 0.0481 | 0.1664 | 0.2892 | 0.1364 | 0.1643 |
| turnover_stability | bull | 0.0365 | 0.1514 | 0.2409 | 0.2273 | 0.1478 |
| stroke_phase | bull | 0.0287 | 0.1415 | 0.2030 | 0.1818 | 0.1200 |
| momentum_reversal | bear | 0.0980 | 0.2132 | 0.4597 | 0.2603 | 0.2896 |
| mom_x_lowvol_20_20 | bear | 0.0922 | 0.2131 | 0.4326 | 0.2055 | 0.2607 |
| trend_lowvol | bear | 0.1077 | 0.2410 | 0.4469 | 0.1233 | 0.2510 |
| rsi_vol_combo | bear | 0.0542 | 0.1843 | 0.2939 | 0.2603 | 0.1852 |
| top_fractal_volume | bear | 0.0413 | 0.1426 | 0.2900 | 0.2683 | 0.1839 |
| fund_gross_margin | bear | 0.0446 | 0.1829 | 0.2440 | 0.1507 | 0.1404 |
| fund_profit_growth | bear | 0.0327 | 0.1705 | 0.1916 | 0.1781 | 0.1129 |

### 历史新高

- **Neutral**: ['trend_lowvol', 'mom_x_lowvol_20_20'] (单因子IC=0.1001, 组合IC=0.1183)
  - weights: [0.5546, 0.4454]
- **Bull**: ['low_downside', 'volatility', 'fund_pb'] (单因子IC=0.0741, 组合IC=0.0844)
  - bull_weights: [0.4002, 0.3397, 0.2601]
- **Bear**: ['momentum_reversal', 'mom_x_lowvol_20_20', 'trend_lowvol'] (单因子IC=0.1347, 组合IC=0.1402)
  - bear_weights: [0.355, 0.3408, 0.3042]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| trend_lowvol | neutral | 0.1105 | 0.2039 | 0.5417 | 0.3998 | 0.3792 |
| mom_x_lowvol_20_20 | neutral | 0.0897 | 0.1973 | 0.4547 | 0.3392 | 0.3044 |
| momentum_reversal | neutral | 0.0930 | 0.2057 | 0.4521 | 0.3257 | 0.2997 |
| rsi_vol_combo | neutral | 0.0665 | 0.2046 | 0.3250 | 0.2359 | 0.2008 |
| fund_pb | neutral | 0.0573 | 0.1977 | 0.2900 | 0.1983 | 0.1738 |
| volatility | neutral | 0.0563 | 0.2023 | 0.2782 | 0.2276 | 0.1707 |
| fund_profit_growth | neutral | 0.0422 | 0.1827 | 0.2309 | 0.1983 | 0.1383 |
| fund_pe | neutral | 0.0439 | 0.2064 | 0.2127 | 0.1482 | 0.1221 |
| turnover_stability | neutral | 0.0308 | 0.1622 | 0.1900 | 0.1795 | 0.1121 |
| low_downside | neutral | 0.0364 | 0.2251 | 0.1618 | 0.1357 | 0.0919 |
| fund_score | neutral | 0.0286 | 0.2067 | 0.1385 | 0.1044 | 0.0765 |
| low_downside | bull | 0.0741 | 0.1761 | 0.4208 | 0.3788 | 0.2901 |
| volatility | bull | 0.0785 | 0.2003 | 0.3917 | 0.2576 | 0.2463 |
| fund_pb | bull | 0.0697 | 0.2338 | 0.2981 | 0.2652 | 0.1886 |
| mom_x_lowvol_20_20 | bull | 0.0441 | 0.1805 | 0.2442 | 0.1364 | 0.1387 |
| rsi_vol_combo | bull | 0.0406 | 0.1781 | 0.2278 | 0.1061 | 0.1260 |
| turnover_stability | bull | 0.0295 | 0.1578 | 0.1871 | 0.2348 | 0.1155 |
| fund_pe | bull | 0.0315 | 0.1841 | 0.1714 | 0.1515 | 0.0987 |
| fund_gross_margin | bull | 0.0251 | 0.1616 | 0.1554 | 0.1894 | 0.0924 |
| momentum_reversal | bear | 0.1338 | 0.1991 | 0.6722 | 0.5616 | 0.5249 |
| mom_x_lowvol_20_20 | bear | 0.1468 | 0.2154 | 0.6814 | 0.4795 | 0.5040 |
| trend_lowvol | bear | 0.1236 | 0.1994 | 0.6196 | 0.4521 | 0.4498 |
| rsi_vol_combo | bear | 0.1006 | 0.1877 | 0.5358 | 0.3973 | 0.3743 |
| fund_gross_margin | bear | 0.0649 | 0.1587 | 0.4087 | 0.2877 | 0.2631 |
| fund_profit_growth | bear | 0.0572 | 0.1784 | 0.3206 | 0.3425 | 0.2152 |
| fund_pb | bear | 0.0499 | 0.1577 | 0.3161 | 0.2603 | 0.1992 |
| fund_revenue_growth | bear | 0.0409 | 0.1429 | 0.2860 | 0.2055 | 0.1724 |
| turnover_stability | bear | 0.0334 | 0.1171 | 0.2849 | 0.2055 | 0.1717 |

### 参股保险

- **Neutral**: ['low_downside', 'fund_pb', 'trend_lowvol'] (单因子IC=0.0797, 组合IC=0.1046)
  - weights: [0.3535, 0.3368, 0.3097]
- **Bull**: ['low_downside', 'turnover_stability'] (单因子IC=0.0702, 组合IC=0.0943)
  - bull_weights: [0.5953, 0.4047]
- **Bear**: ['mom_x_lowvol_20_20', 'momentum_reversal'] (单因子IC=0.1086, 组合IC=0.1154)
  - bear_weights: [0.5382, 0.4618]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| low_downside | neutral | 0.0804 | 0.2169 | 0.3708 | 0.3257 | 0.2458 |
| fund_pb | neutral | 0.0834 | 0.2287 | 0.3649 | 0.2839 | 0.2342 |
| trend_lowvol | neutral | 0.0752 | 0.2255 | 0.3333 | 0.2923 | 0.2154 |
| turnover_stability | neutral | 0.0502 | 0.1513 | 0.3319 | 0.2484 | 0.2072 |
| mom_x_lowvol_20_20 | neutral | 0.0448 | 0.1780 | 0.2517 | 0.1795 | 0.1484 |
| fund_pe | neutral | 0.0543 | 0.2259 | 0.2403 | 0.2088 | 0.1453 |
| momentum_reversal | neutral | 0.0501 | 0.2046 | 0.2449 | 0.1357 | 0.1391 |
| fund_profit_growth | neutral | 0.0280 | 0.1358 | 0.2062 | 0.1503 | 0.1186 |
| fund_score | neutral | 0.0242 | 0.1612 | 0.1500 | 0.1148 | 0.0836 |
| stroke_phase | neutral | 0.0099 | 0.1566 | 0.0632 | 0.1044 | 0.0349 |
| low_downside | bull | 0.0895 | 0.1849 | 0.4837 | 0.3864 | 0.3353 |
| turnover_stability | bull | 0.0510 | 0.1338 | 0.3809 | 0.1970 | 0.2280 |
| volatility | bull | 0.0436 | 0.1855 | 0.2350 | 0.2273 | 0.1442 |
| fund_pb | bull | 0.0518 | 0.2333 | 0.2218 | 0.1364 | 0.1260 |
| trend_lowvol | bull | 0.0376 | 0.2374 | 0.1586 | 0.1288 | 0.0895 |
| mom_x_lowvol_20_20 | bear | 0.1057 | 0.2032 | 0.5201 | 0.3973 | 0.3634 |
| momentum_reversal | bear | 0.1115 | 0.2352 | 0.4742 | 0.3151 | 0.3118 |
| fund_revenue_growth | bear | 0.0508 | 0.1281 | 0.3966 | 0.2055 | 0.2391 |
| fund_profit_growth | bear | 0.0425 | 0.1287 | 0.3305 | 0.2055 | 0.1992 |
| trend_lowvol | bear | 0.0717 | 0.2709 | 0.2646 | 0.2603 | 0.1667 |
| fund_score | bear | 0.0376 | 0.1690 | 0.2225 | 0.2329 | 0.1372 |
| bb_width_20 | bear | 0.0641 | 0.2756 | 0.2324 | 0.1781 | 0.1369 |
| rsi_vol_combo | bear | 0.0467 | 0.1956 | 0.2387 | 0.1233 | 0.1341 |

### 参股券商

- **Neutral**: ['trend_lowvol', 'momentum_reversal', 'fund_pb'] (单因子IC=0.0787, 组合IC=0.1134)
  - weights: [0.432, 0.285, 0.283]
- **Bull**: ['momentum_reversal', 'low_downside', 'trend_lowvol'] (单因子IC=0.0937, 组合IC=0.1238)
  - bull_weights: [0.3461, 0.3453, 0.3086]
- **Bear**: ['momentum_reversal', 'mom_x_lowvol_20_20', 'trend_lowvol'] (单因子IC=0.0992, 组合IC=0.1214)
  - bear_weights: [0.3733, 0.3211, 0.3056]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| trend_lowvol | neutral | 0.0969 | 0.1927 | 0.5028 | 0.4071 | 0.3538 |
| momentum_reversal | neutral | 0.0702 | 0.1839 | 0.3816 | 0.2234 | 0.2334 |
| fund_pb | neutral | 0.0690 | 0.1818 | 0.3796 | 0.2213 | 0.2318 |
| mom_x_lowvol_20_20 | neutral | 0.0551 | 0.1557 | 0.3537 | 0.2589 | 0.2226 |
| fund_pe | neutral | 0.0651 | 0.1845 | 0.3527 | 0.2317 | 0.2172 |
| low_downside | neutral | 0.0552 | 0.1763 | 0.3130 | 0.3048 | 0.2042 |
| turnover_stability | neutral | 0.0389 | 0.1223 | 0.3181 | 0.2213 | 0.1943 |
| rsi_vol_combo | neutral | 0.0545 | 0.1773 | 0.3072 | 0.1900 | 0.1828 |
| fund_profit_growth | neutral | 0.0356 | 0.1182 | 0.3013 | 0.1983 | 0.1805 |
| volatility | neutral | 0.0419 | 0.1476 | 0.2839 | 0.2484 | 0.1772 |
| fund_score | neutral | 0.0261 | 0.1378 | 0.1897 | 0.1378 | 0.1079 |
| momentum_reversal | bull | 0.0926 | 0.1573 | 0.5885 | 0.3788 | 0.4057 |
| low_downside | bull | 0.0926 | 0.1595 | 0.5807 | 0.3939 | 0.4048 |
| trend_lowvol | bull | 0.0957 | 0.1854 | 0.5163 | 0.4015 | 0.3618 |
| turnover_stability | bull | 0.0518 | 0.1100 | 0.4715 | 0.3333 | 0.3144 |
| rsi_vol_combo | bull | 0.0642 | 0.1458 | 0.4400 | 0.3106 | 0.2883 |
| fund_pb | bull | 0.0686 | 0.1597 | 0.4298 | 0.3030 | 0.2800 |
| volatility | bull | 0.0470 | 0.1261 | 0.3728 | 0.3258 | 0.2471 |
| fund_pe | bull | 0.0660 | 0.1839 | 0.3589 | 0.2879 | 0.2311 |
| mom_x_lowvol_20_20 | bull | 0.0350 | 0.1269 | 0.2755 | 0.1742 | 0.1617 |
| fund_score | bull | 0.0250 | 0.1253 | 0.1998 | 0.1136 | 0.1113 |
| fund_profit_growth | bull | 0.0235 | 0.1200 | 0.1961 | 0.1212 | 0.1100 |
| momentum_reversal | bear | 0.1138 | 0.2185 | 0.5209 | 0.3151 | 0.3425 |
| mom_x_lowvol_20_20 | bear | 0.0877 | 0.1917 | 0.4577 | 0.2877 | 0.2947 |
| trend_lowvol | bear | 0.0961 | 0.2253 | 0.4265 | 0.3151 | 0.2805 |
| rsi_vol_combo | bear | 0.0559 | 0.1893 | 0.2954 | 0.1781 | 0.1740 |
| bb_width_20 | bear | 0.0670 | 0.2281 | 0.2936 | 0.1781 | 0.1729 |

### 参股新三板

- **Neutral**: ['trend_lowvol', 'momentum_reversal', 'volatility'] (单因子IC=0.0764, 组合IC=0.0951)
  - weights: [0.3674, 0.3194, 0.3132]
- **Bull**: ['turnover_stability', 'low_downside', 'fund_pb'] (单因子IC=0.0741, 组合IC=0.1063)
  - bull_weights: [0.3612, 0.3499, 0.2889]
- **Bear**: ['mom_x_lowvol_20_20', 'momentum_reversal', 'trend_lowvol'] (单因子IC=0.1166, 组合IC=0.1378)
  - bear_weights: [0.363, 0.3536, 0.2833]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| trend_lowvol | neutral | 0.0876 | 0.2055 | 0.4263 | 0.3027 | 0.2777 |
| momentum_reversal | neutral | 0.0694 | 0.1861 | 0.3730 | 0.2944 | 0.2414 |
| volatility | neutral | 0.0721 | 0.1959 | 0.3681 | 0.2860 | 0.2367 |
| mom_x_lowvol_20_20 | neutral | 0.0628 | 0.1804 | 0.3481 | 0.2965 | 0.2256 |
| fund_pb | neutral | 0.0714 | 0.2033 | 0.3511 | 0.2526 | 0.2199 |
| turnover_stability | neutral | 0.0379 | 0.1161 | 0.3263 | 0.2589 | 0.2054 |
| fund_pe | neutral | 0.0536 | 0.1939 | 0.2766 | 0.2150 | 0.1680 |
| rsi_vol_combo | neutral | 0.0501 | 0.1825 | 0.2747 | 0.1942 | 0.1640 |
| fund_profit_growth | neutral | 0.0318 | 0.1307 | 0.2435 | 0.1858 | 0.1444 |
| low_downside | neutral | 0.0444 | 0.1895 | 0.2343 | 0.1795 | 0.1382 |
| fund_score | neutral | 0.0357 | 0.1650 | 0.2165 | 0.1816 | 0.1279 |
| fund_roe | neutral | 0.0287 | 0.1821 | 0.1574 | 0.1065 | 0.0871 |
| fund_revenue_growth | neutral | 0.0148 | 0.1367 | 0.1086 | 0.1106 | 0.0603 |
| turnover_stability | bull | 0.0610 | 0.1144 | 0.5332 | 0.3409 | 0.3575 |
| low_downside | bull | 0.0818 | 0.1602 | 0.5106 | 0.3561 | 0.3462 |
| fund_pb | bull | 0.0796 | 0.1815 | 0.4388 | 0.3030 | 0.2859 |
| trend_lowvol | bull | 0.0854 | 0.1962 | 0.4352 | 0.2955 | 0.2819 |
| volatility | bull | 0.0620 | 0.1816 | 0.3413 | 0.2348 | 0.2107 |
| momentum_reversal | bull | 0.0547 | 0.1735 | 0.3154 | 0.2273 | 0.1936 |
| fund_pe | bull | 0.0481 | 0.1780 | 0.2702 | 0.2197 | 0.1648 |
| rsi_vol_combo | bull | 0.0401 | 0.1657 | 0.2423 | 0.2273 | 0.1487 |
| mom_x_lowvol_20_20 | bear | 0.1246 | 0.2308 | 0.5398 | 0.4247 | 0.3845 |
| momentum_reversal | bear | 0.1291 | 0.2361 | 0.5469 | 0.3699 | 0.3746 |
| trend_lowvol | bear | 0.0961 | 0.2280 | 0.4214 | 0.4247 | 0.3001 |
| bb_width_20 | bear | 0.0818 | 0.2146 | 0.3812 | 0.2877 | 0.2455 |
| rsi_vol_combo | bear | 0.0512 | 0.1819 | 0.2817 | 0.1781 | 0.1659 |

### 参股期货

- **Neutral**: ['trend_lowvol', 'fund_pb', 'momentum_reversal'] (单因子IC=0.0677, 组合IC=0.1091)
  - weights: [0.4468, 0.2812, 0.272]
- **Bull**: ['turnover_stability', 'low_downside'] (单因子IC=0.0865, 组合IC=0.117)
  - bull_weights: [0.5245, 0.4755]
- **Bear**: ['mom_x_lowvol_20_20', 'bb_width_20', 'momentum_reversal'] (单因子IC=0.0857, 组合IC=0.1085)
  - bear_weights: [0.3942, 0.3073, 0.2985]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| trend_lowvol | neutral | 0.0860 | 0.2301 | 0.3738 | 0.3027 | 0.2435 |
| fund_pb | neutral | 0.0590 | 0.2229 | 0.2645 | 0.1587 | 0.1532 |
| momentum_reversal | neutral | 0.0582 | 0.2342 | 0.2483 | 0.1942 | 0.1483 |
| fund_pe | neutral | 0.0574 | 0.2315 | 0.2478 | 0.1795 | 0.1462 |
| mom_x_lowvol_20_20 | neutral | 0.0462 | 0.1949 | 0.2371 | 0.1775 | 0.1396 |
| turnover_stability | neutral | 0.0432 | 0.1854 | 0.2331 | 0.1795 | 0.1375 |
| fund_score | neutral | 0.0414 | 0.1984 | 0.2086 | 0.1712 | 0.1222 |
| volatility | neutral | 0.0390 | 0.1941 | 0.2006 | 0.1514 | 0.1155 |
| low_downside | neutral | 0.0416 | 0.2268 | 0.1832 | 0.1294 | 0.1035 |
| fund_revenue_growth | neutral | 0.0316 | 0.1854 | 0.1706 | 0.1618 | 0.0991 |
| rsi_vol_combo | neutral | 0.0299 | 0.2220 | 0.1346 | 0.1336 | 0.0763 |
| turnover_stability | bull | 0.0825 | 0.1698 | 0.4859 | 0.4545 | 0.3534 |
| low_downside | bull | 0.0905 | 0.1904 | 0.4751 | 0.3485 | 0.3203 |
| volatility | bull | 0.0681 | 0.1724 | 0.3949 | 0.3409 | 0.2648 |
| fund_pe | bull | 0.0720 | 0.2023 | 0.3560 | 0.2803 | 0.2279 |
| fund_pb | bull | 0.0577 | 0.1966 | 0.2938 | 0.2424 | 0.1825 |
| momentum_reversal | bull | 0.0610 | 0.2241 | 0.2722 | 0.2045 | 0.1639 |
| mom_x_lowvol_20_20 | bull | 0.0373 | 0.1770 | 0.2106 | 0.1894 | 0.1252 |
| rsi_vol_combo | bull | 0.0411 | 0.2009 | 0.2045 | 0.1742 | 0.1201 |
| fund_revenue_growth | bull | 0.0201 | 0.1649 | 0.1221 | 0.1212 | 0.0685 |
| mom_x_lowvol_20_20 | bear | 0.0900 | 0.2437 | 0.3693 | 0.2877 | 0.2378 |
| bb_width_20 | bear | 0.0799 | 0.2599 | 0.3075 | 0.2055 | 0.1853 |
| momentum_reversal | bear | 0.0873 | 0.2856 | 0.3056 | 0.1781 | 0.1800 |
| fund_pe | bear | 0.0538 | 0.2059 | 0.2611 | 0.2055 | 0.1574 |
| fund_pb | bear | 0.0341 | 0.1469 | 0.2324 | 0.1781 | 0.1369 |

### 参股银行

- **Neutral**: ['fund_pb', 'turnover_stability', 'trend_lowvol'] (单因子IC=0.0678, 组合IC=0.1104)
  - weights: [0.4013, 0.3178, 0.2809]
- **Bull**: ['fund_pb', 'low_downside', 'turnover_stability'] (单因子IC=0.0731, 组合IC=0.1052)
  - bull_weights: [0.377, 0.3117, 0.3113]
- **Bear**: ['mom_x_lowvol_20_20'] (单因子IC=0.0989, 组合IC=0.0989)
  - bear_weights: [1.0]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| fund_pb | neutral | 0.0814 | 0.1539 | 0.5291 | 0.3549 | 0.3584 |
| turnover_stability | neutral | 0.0447 | 0.1053 | 0.4248 | 0.3361 | 0.2838 |
| trend_lowvol | neutral | 0.0772 | 0.2017 | 0.3827 | 0.3111 | 0.2509 |
| mom_x_lowvol_20_20 | neutral | 0.0543 | 0.1445 | 0.3756 | 0.2589 | 0.2364 |
| fund_pe | neutral | 0.0690 | 0.1824 | 0.3785 | 0.2150 | 0.2300 |
| fund_profit_growth | neutral | 0.0426 | 0.1164 | 0.3660 | 0.2443 | 0.2277 |
| low_downside | neutral | 0.0607 | 0.1807 | 0.3359 | 0.3006 | 0.2184 |
| momentum_reversal | neutral | 0.0586 | 0.1710 | 0.3426 | 0.2109 | 0.2074 |
| fund_score | neutral | 0.0456 | 0.1432 | 0.3181 | 0.2443 | 0.1979 |
| volatility | neutral | 0.0482 | 0.1657 | 0.2910 | 0.2484 | 0.1817 |
| fund_roe | neutral | 0.0390 | 0.1736 | 0.2245 | 0.1858 | 0.1331 |
| rsi_vol_combo | neutral | 0.0349 | 0.1580 | 0.2207 | 0.1378 | 0.1255 |
| fund_pb | bull | 0.0806 | 0.1268 | 0.6360 | 0.4773 | 0.4698 |
| low_downside | bull | 0.0872 | 0.1513 | 0.5761 | 0.3485 | 0.3884 |
| turnover_stability | bull | 0.0514 | 0.0929 | 0.5535 | 0.4015 | 0.3879 |
| volatility | bull | 0.0662 | 0.1389 | 0.4768 | 0.2803 | 0.3052 |
| fund_pe | bull | 0.0648 | 0.1783 | 0.3636 | 0.2500 | 0.2273 |
| momentum_reversal | bull | 0.0406 | 0.1400 | 0.2896 | 0.2348 | 0.1788 |
| trend_lowvol | bull | 0.0421 | 0.1833 | 0.2295 | 0.1818 | 0.1356 |
| rsi_vol_combo | bull | 0.0274 | 0.1224 | 0.2240 | 0.2045 | 0.1349 |
| mom_x_lowvol_20_20 | bull | 0.0229 | 0.1264 | 0.1813 | 0.1667 | 0.1057 |
| fund_gross_margin | bull | 0.0116 | 0.0962 | 0.1202 | 0.1136 | 0.0669 |
| mom_x_lowvol_20_20 | bear | 0.0989 | 0.1763 | 0.5608 | 0.3699 | 0.3841 |
| fund_revenue_growth | bear | 0.0464 | 0.0969 | 0.4790 | 0.3699 | 0.3281 |
| momentum_reversal | bear | 0.1129 | 0.2177 | 0.5184 | 0.2329 | 0.3196 |
| trend_lowvol | bear | 0.0887 | 0.1991 | 0.4453 | 0.2877 | 0.2867 |
| fund_profit_growth | bear | 0.0467 | 0.1192 | 0.3916 | 0.1781 | 0.2306 |
| turnover_stability | bear | 0.0293 | 0.0893 | 0.3285 | 0.2877 | 0.2115 |
| fund_score | bear | 0.0427 | 0.1401 | 0.3046 | 0.2055 | 0.1836 |
| rsi_vol_combo | bear | 0.0494 | 0.1743 | 0.2832 | 0.1507 | 0.1629 |
| bb_width_20 | bear | 0.0527 | 0.2394 | 0.2200 | 0.1233 | 0.1236 |

### 反内卷概念

- **Neutral**: ['fund_pe', 'turnover_stability'] (单因子IC=0.0504, 组合IC=0.074)
  - weights: [0.5307, 0.4693]
- **Bull**: ['low_downside', 'volatility', 'fund_pb'] (单因子IC=0.0768, 组合IC=0.0875)
  - bull_weights: [0.3711, 0.3557, 0.2732]
- **Bear**: ['momentum_reversal', 'bb_width_20', 'turnover_stability'] (单因子IC=0.1141, 组合IC=0.1748)
  - bear_weights: [0.3585, 0.3382, 0.3033]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| fund_pe | neutral | 0.0686 | 0.2783 | 0.2464 | 0.2004 | 0.1479 |
| turnover_stability | neutral | 0.0322 | 0.1498 | 0.2149 | 0.2171 | 0.1308 |
| fund_pb | neutral | 0.0606 | 0.2682 | 0.2258 | 0.1524 | 0.1301 |
| volatility | neutral | 0.0478 | 0.2112 | 0.2261 | 0.1503 | 0.1300 |
| fund_gross_margin | neutral | 0.0365 | 0.1998 | 0.1828 | 0.1921 | 0.1090 |
| mom_x_lowvol_20_20 | neutral | 0.0302 | 0.1840 | 0.1640 | 0.1190 | 0.0917 |
| fund_score | neutral | 0.0353 | 0.2388 | 0.1479 | 0.1065 | 0.0818 |
| fund_roe | neutral | 0.0304 | 0.2412 | 0.1261 | 0.1106 | 0.0700 |
| trend_lowvol | neutral | 0.0250 | 0.2515 | 0.0996 | 0.1273 | 0.0561 |
| low_downside | bull | 0.0742 | 0.1831 | 0.4050 | 0.2879 | 0.2608 |
| volatility | bull | 0.0768 | 0.1989 | 0.3859 | 0.2955 | 0.2500 |
| fund_pb | bull | 0.0794 | 0.2398 | 0.3312 | 0.1591 | 0.1920 |
| stroke_phase | bull | 0.0338 | 0.1285 | 0.2632 | 0.2576 | 0.1655 |
| fund_pe | bull | 0.0621 | 0.2410 | 0.2576 | 0.2273 | 0.1581 |
| turnover_stability | bull | 0.0274 | 0.1674 | 0.1635 | 0.1364 | 0.0929 |
| momentum_reversal | bear | 0.1492 | 0.2435 | 0.6126 | 0.4521 | 0.4448 |
| bb_width_20 | bear | 0.1183 | 0.2048 | 0.5778 | 0.4521 | 0.4195 |
| turnover_stability | bear | 0.0749 | 0.1337 | 0.5606 | 0.3425 | 0.3763 |
| mom_x_lowvol_20_20 | bear | 0.1182 | 0.2352 | 0.5027 | 0.3699 | 0.3443 |
| rsi_vol_combo | bear | 0.1054 | 0.2307 | 0.4570 | 0.2603 | 0.2880 |
| trend_lowvol | bear | 0.1095 | 0.2794 | 0.3920 | 0.1507 | 0.2256 |
| top_fractal_volume | bear | 0.0275 | 0.1072 | 0.2562 | 0.2759 | 0.1635 |
| fund_score | bear | 0.0459 | 0.2357 | 0.1948 | 0.2055 | 0.1174 |

### 发电机概念

- **Neutral**: ['momentum_reversal', 'mom_x_lowvol_20_20', 'trend_lowvol'] (单因子IC=0.0936, 组合IC=0.1045)
  - weights: [0.3513, 0.328, 0.3207]
- **Bull**: ['fund_pb', 'volatility'] (单因子IC=0.0848, 组合IC=0.0939)
  - bull_weights: [0.5098, 0.4902]
- **Bear**: ['momentum_reversal', 'mom_x_lowvol_20_20'] (单因子IC=0.1657, 组合IC=0.1774)
  - bear_weights: [0.5339, 0.4661]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| momentum_reversal | neutral | 0.0943 | 0.2848 | 0.3311 | 0.2599 | 0.2086 |
| mom_x_lowvol_20_20 | neutral | 0.0901 | 0.2903 | 0.3103 | 0.2547 | 0.1947 |
| trend_lowvol | neutral | 0.0965 | 0.3069 | 0.3144 | 0.2109 | 0.1903 |
| fund_revenue_growth | neutral | 0.0713 | 0.2497 | 0.2855 | 0.2751 | 0.1820 |
| rsi_vol_combo | neutral | 0.0747 | 0.2779 | 0.2689 | 0.1931 | 0.1604 |
| turnover_stability | neutral | 0.0668 | 0.2543 | 0.2629 | 0.1952 | 0.1571 |
| volatility | neutral | 0.0744 | 0.2891 | 0.2574 | 0.2077 | 0.1554 |
| fund_score | neutral | 0.0665 | 0.2773 | 0.2400 | 0.2578 | 0.1509 |
| fund_profit_growth | neutral | 0.0569 | 0.2665 | 0.2134 | 0.2344 | 0.1317 |
| fund_pb | neutral | 0.0541 | 0.2662 | 0.2032 | 0.1733 | 0.1192 |
| fund_gross_margin | neutral | 0.0331 | 0.2823 | 0.1172 | 0.1378 | 0.0667 |
| fund_pb | bull | 0.0882 | 0.2608 | 0.3383 | 0.2538 | 0.2121 |
| volatility | bull | 0.0814 | 0.2508 | 0.3244 | 0.2576 | 0.2040 |
| low_downside | bull | 0.0764 | 0.2536 | 0.3013 | 0.2727 | 0.1917 |
| rsi_vol_combo | bull | 0.0445 | 0.2805 | 0.1585 | 0.1364 | 0.0901 |
| turnover_stability | bull | 0.0294 | 0.2082 | 0.1415 | 0.1364 | 0.0804 |
| trend_lowvol | bull | 0.0336 | 0.2839 | 0.1184 | 0.1136 | 0.0659 |
| fund_profit_growth | bull | 0.0257 | 0.2648 | 0.0969 | 0.1406 | 0.0553 |
| momentum_reversal | bear | 0.1748 | 0.2475 | 0.7064 | 0.5890 | 0.5612 |
| mom_x_lowvol_20_20 | bear | 0.1566 | 0.2451 | 0.6386 | 0.5342 | 0.4899 |
| rsi_vol_combo | bear | 0.1486 | 0.2895 | 0.5134 | 0.3425 | 0.3446 |
| fund_score | bear | 0.1442 | 0.2773 | 0.5199 | 0.3151 | 0.3418 |
| fund_revenue_growth | bear | 0.1414 | 0.3131 | 0.4515 | 0.4366 | 0.3243 |
| fund_profit_growth | bear | 0.1060 | 0.2325 | 0.4559 | 0.2740 | 0.2904 |
| fund_roe | bear | 0.1314 | 0.2978 | 0.4411 | 0.2877 | 0.2840 |
| fund_pe | bear | 0.1045 | 0.2810 | 0.3718 | 0.3151 | 0.2445 |
| fund_pb | bear | 0.0536 | 0.2082 | 0.2576 | 0.1781 | 0.1517 |
| trend_lowvol | bear | 0.0570 | 0.2592 | 0.2200 | 0.2329 | 0.1356 |

### 可控核聚变

- **Neutral**: ['fund_profit_growth', 'momentum_reversal', 'mom_x_lowvol_20_20'] (单因子IC=0.0652, 组合IC=0.0793)
  - weights: [0.3633, 0.3192, 0.3176]
- **Bull**: ['low_downside'] (单因子IC=0.0843, 组合IC=0.0843)
  - bull_weights: [1.0]
- **Bear**: ['mom_x_lowvol_20_20', 'rsi_vol_combo', 'momentum_reversal'] (单因子IC=0.1251, 组合IC=0.136)
  - bear_weights: [0.34, 0.3373, 0.3227]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| fund_profit_growth | neutral | 0.0554 | 0.1525 | 0.3631 | 0.2724 | 0.2310 |
| momentum_reversal | neutral | 0.0731 | 0.2256 | 0.3241 | 0.2526 | 0.2030 |
| mom_x_lowvol_20_20 | neutral | 0.0671 | 0.2068 | 0.3246 | 0.2443 | 0.2020 |
| volatility | neutral | 0.0680 | 0.2197 | 0.3098 | 0.2578 | 0.1948 |
| trend_lowvol | neutral | 0.0673 | 0.2282 | 0.2952 | 0.2380 | 0.1827 |
| turnover_stability | neutral | 0.0439 | 0.1524 | 0.2884 | 0.2422 | 0.1791 |
| rsi_vol_combo | neutral | 0.0608 | 0.2095 | 0.2903 | 0.1921 | 0.1730 |
| fund_pb | neutral | 0.0646 | 0.2340 | 0.2763 | 0.1921 | 0.1647 |
| fund_pe | neutral | 0.0547 | 0.2109 | 0.2595 | 0.2422 | 0.1612 |
| fund_score | neutral | 0.0400 | 0.1628 | 0.2455 | 0.2192 | 0.1496 |
| fund_roe | neutral | 0.0378 | 0.1743 | 0.2172 | 0.1503 | 0.1249 |
| low_downside | neutral | 0.0518 | 0.2469 | 0.2098 | 0.1503 | 0.1207 |
| fund_revenue_growth | neutral | 0.0226 | 0.1387 | 0.1631 | 0.1190 | 0.0913 |
| low_downside | bull | 0.0843 | 0.1892 | 0.4454 | 0.3788 | 0.3070 |
| volatility | bull | 0.0669 | 0.1998 | 0.3347 | 0.3561 | 0.2269 |
| fund_profit_growth | bull | 0.0506 | 0.1662 | 0.3047 | 0.3258 | 0.2020 |
| fund_pb | bull | 0.0649 | 0.2157 | 0.3011 | 0.2576 | 0.1893 |
| fund_pe | bull | 0.0582 | 0.2092 | 0.2784 | 0.2652 | 0.1761 |
| fund_roe | bull | 0.0421 | 0.1882 | 0.2234 | 0.1742 | 0.1312 |
| turnover_stability | bull | 0.0309 | 0.1491 | 0.2070 | 0.2348 | 0.1278 |
| fund_score | bull | 0.0340 | 0.1796 | 0.1894 | 0.2803 | 0.1213 |
| fund_revenue_growth | bull | 0.0319 | 0.1548 | 0.2060 | 0.1591 | 0.1194 |
| trend_lowvol | bull | 0.0449 | 0.2152 | 0.2087 | 0.1364 | 0.1186 |
| exhaustion_risk | bull | 0.0271 | 0.1589 | 0.1708 | 0.1163 | 0.0953 |
| top_fractal_volume | bull | 0.0230 | 0.1437 | 0.1603 | 0.1466 | 0.0919 |
| mom_x_lowvol_20_20 | bear | 0.1208 | 0.2125 | 0.5685 | 0.3425 | 0.3816 |
| rsi_vol_combo | bear | 0.1170 | 0.2032 | 0.5758 | 0.3151 | 0.3786 |
| momentum_reversal | bear | 0.1374 | 0.2442 | 0.5626 | 0.2877 | 0.3622 |
| bb_width_20 | bear | 0.1153 | 0.2475 | 0.4658 | 0.3425 | 0.3126 |
| turnover_stability | bear | 0.0520 | 0.1337 | 0.3889 | 0.4247 | 0.2770 |
| trend_lowvol | bear | 0.1341 | 0.2926 | 0.4584 | 0.1781 | 0.2700 |
| fund_gross_margin | bear | 0.0718 | 0.2279 | 0.3150 | 0.3151 | 0.2072 |

### 合成生物

- **Neutral**: ['fund_pb', 'volatility', 'trend_lowvol'] (单因子IC=0.0773, 组合IC=0.1179)
  - weights: [0.5216, 0.2545, 0.2239]
- **Bull**: ['low_downside', 'volatility', 'fund_pb'] (单因子IC=0.0914, 组合IC=0.1134)
  - bull_weights: [0.3588, 0.3296, 0.3116]
- **Bear**: ['trend_lowvol', 'mom_x_lowvol_20_20', 'fund_pb'] (单因子IC=0.1193, 组合IC=0.1725)
  - bear_weights: [0.3896, 0.331, 0.2794]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| fund_pb | neutral | 0.1025 | 0.1590 | 0.6444 | 0.4530 | 0.4681 |
| volatility | neutral | 0.0679 | 0.1939 | 0.3501 | 0.3048 | 0.2284 |
| trend_lowvol | neutral | 0.0615 | 0.1875 | 0.3279 | 0.2255 | 0.2009 |
| fund_pe | neutral | 0.0535 | 0.1716 | 0.3119 | 0.2589 | 0.1963 |
| momentum_reversal | neutral | 0.0577 | 0.1808 | 0.3190 | 0.1754 | 0.1875 |
| mom_x_lowvol_20_20 | neutral | 0.0530 | 0.1723 | 0.3078 | 0.2109 | 0.1863 |
| fund_profit_growth | neutral | 0.0435 | 0.1481 | 0.2938 | 0.2171 | 0.1788 |
| low_downside | neutral | 0.0424 | 0.1740 | 0.2439 | 0.2025 | 0.1467 |
| rsi_vol_combo | neutral | 0.0367 | 0.1698 | 0.2163 | 0.1921 | 0.1289 |
| fund_score | neutral | 0.0316 | 0.1695 | 0.1866 | 0.1420 | 0.1065 |
| turnover_stability | neutral | 0.0219 | 0.1280 | 0.1712 | 0.1023 | 0.0943 |
| low_downside | bull | 0.0817 | 0.1365 | 0.5981 | 0.5455 | 0.4622 |
| volatility | bull | 0.0999 | 0.1729 | 0.5778 | 0.4697 | 0.4246 |
| fund_pb | bull | 0.0928 | 0.1655 | 0.5608 | 0.4318 | 0.4015 |
| momentum_reversal | bull | 0.0661 | 0.1757 | 0.3760 | 0.3030 | 0.2450 |
| mom_x_lowvol_20_20 | bull | 0.0555 | 0.1771 | 0.3135 | 0.2424 | 0.1948 |
| turnover_stability | bull | 0.0409 | 0.1344 | 0.3046 | 0.2727 | 0.1939 |
| rsi_vol_combo | bull | 0.0465 | 0.1469 | 0.3168 | 0.1970 | 0.1896 |
| trend_lowvol | bull | 0.0496 | 0.1815 | 0.2735 | 0.1136 | 0.1523 |
| fund_pe | bull | 0.0347 | 0.1643 | 0.2114 | 0.1591 | 0.1225 |
| trend_lowvol | bear | 0.1390 | 0.1743 | 0.7978 | 0.5890 | 0.6338 |
| mom_x_lowvol_20_20 | bear | 0.1374 | 0.1993 | 0.6896 | 0.5616 | 0.5384 |
| fund_pb | bear | 0.0816 | 0.1279 | 0.6380 | 0.4247 | 0.4545 |
| momentum_reversal | bear | 0.1208 | 0.2073 | 0.5825 | 0.3151 | 0.3830 |
| rsi_vol_combo | bear | 0.0866 | 0.1826 | 0.4743 | 0.2329 | 0.2924 |
| fund_revenue_growth | bear | 0.0422 | 0.1130 | 0.3733 | 0.3973 | 0.2608 |
| turnover_stability | bear | 0.0387 | 0.1154 | 0.3355 | 0.2055 | 0.2022 |
| fund_pe | bear | 0.0569 | 0.1826 | 0.3118 | 0.2603 | 0.1965 |

### 周期股

- **Neutral**: ['fund_pe', 'fund_gross_margin', 'top_fractal_volume'] (单因子IC=0.0346, 组合IC=0.054)
  - weights: [0.3636, 0.3541, 0.2823]
- **Bull**: ['fund_roe', 'ema20_slope'] (单因子IC=0.0354, 组合IC=0.0379)
  - bull_weights: [0.5029, 0.4971]
- **Bear**: ['bb_width_20', 'limit_pullback_score', 'wash_sale_score'] (单因子IC=0.0736, 组合IC=0.1328)
  - bear_weights: [0.433, 0.3248, 0.2422]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| fund_pe | neutral | 0.0517 | 0.2448 | 0.2113 | 0.1712 | 0.1238 |
| fund_gross_margin | neutral | 0.0321 | 0.1581 | 0.2033 | 0.1858 | 0.1205 |
| top_fractal_volume | neutral | 0.0199 | 0.1183 | 0.1683 | 0.1417 | 0.0961 |
| trend_lowvol | neutral | 0.0393 | 0.2329 | 0.1687 | 0.1294 | 0.0953 |
| vol_confirm | neutral | 0.0215 | 0.1882 | 0.1142 | 0.1169 | 0.0638 |
| fund_roe | bull | 0.0350 | 0.1998 | 0.1753 | 0.1515 | 0.1010 |
| ema20_slope | bull | 0.0358 | 0.2270 | 0.1578 | 0.2652 | 0.0998 |
| exhaustion_risk | bull | 0.0266 | 0.1556 | 0.1708 | 0.1591 | 0.0990 |
| ma_alignment | bull | 0.0327 | 0.2338 | 0.1398 | 0.2576 | 0.0879 |
| top_fractal_volume | bull | 0.0185 | 0.1311 | 0.1408 | 0.2400 | 0.0873 |
| relative_strength | bull | 0.0318 | 0.2188 | 0.1455 | 0.1515 | 0.0838 |
| bb_width_20 | bull | 0.0217 | 0.2080 | 0.1044 | 0.1288 | 0.0589 |
| bb_width_20 | bear | 0.1064 | 0.1961 | 0.5427 | 0.3973 | 0.3791 |
| limit_pullback_score | bear | 0.0737 | 0.1506 | 0.4896 | 0.1618 | 0.2844 |
| wash_sale_score | bear | 0.0406 | 0.1120 | 0.3628 | 0.1695 | 0.2121 |
| fund_profit_growth | bear | 0.0571 | 0.1773 | 0.3220 | 0.1781 | 0.1896 |
| momentum_reversal | bear | 0.0630 | 0.2302 | 0.2739 | 0.3151 | 0.1801 |
| trend_lowvol | bear | 0.0807 | 0.2910 | 0.2775 | 0.2329 | 0.1711 |
| vol_confirm | bear | 0.0372 | 0.1509 | 0.2462 | 0.1507 | 0.1416 |

### 味蕾经济

- **Neutral**: ['fund_pb', 'mom_x_lowvol_20_20', 'momentum_reversal'] (单因子IC=0.0696, 组合IC=0.115)
  - weights: [0.4, 0.3201, 0.2799]
- **Bull**: ['volatility', 'fund_pb'] (单因子IC=0.0669, 组合IC=0.0906)
  - bull_weights: [0.5257, 0.4743]
- **Bear**: ['fund_pb', 'trend_lowvol', 'mom_x_lowvol_20_20'] (单因子IC=0.1318, 组合IC=0.1973)
  - bear_weights: [0.365, 0.3243, 0.3106]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| fund_pb | neutral | 0.0870 | 0.2363 | 0.3683 | 0.3090 | 0.2411 |
| mom_x_lowvol_20_20 | neutral | 0.0623 | 0.1954 | 0.3186 | 0.2109 | 0.1929 |
| momentum_reversal | neutral | 0.0596 | 0.2078 | 0.2865 | 0.1775 | 0.1687 |
| trend_lowvol | neutral | 0.0645 | 0.2347 | 0.2749 | 0.1858 | 0.1630 |
| fund_pe | neutral | 0.0508 | 0.1957 | 0.2595 | 0.2276 | 0.1593 |
| fund_profit_growth | neutral | 0.0466 | 0.1780 | 0.2616 | 0.1879 | 0.1554 |
| rsi_vol_combo | neutral | 0.0436 | 0.2059 | 0.2115 | 0.2025 | 0.1272 |
| volatility | neutral | 0.0289 | 0.1973 | 0.1465 | 0.1023 | 0.0807 |
| low_downside | neutral | 0.0235 | 0.2100 | 0.1120 | 0.1294 | 0.0633 |
| volatility | bull | 0.0660 | 0.1926 | 0.3424 | 0.2727 | 0.2179 |
| fund_pb | bull | 0.0678 | 0.2180 | 0.3108 | 0.2652 | 0.1966 |
| turnover_stability | bull | 0.0400 | 0.1491 | 0.2683 | 0.1364 | 0.1525 |
| low_downside | bull | 0.0460 | 0.1842 | 0.2497 | 0.1970 | 0.1494 |
| rsi_vol_combo | bull | 0.0446 | 0.1884 | 0.2370 | 0.1818 | 0.1400 |
| trend_lowvol | bull | 0.0528 | 0.2263 | 0.2335 | 0.1667 | 0.1362 |
| fund_pe | bull | 0.0426 | 0.2031 | 0.2099 | 0.1515 | 0.1208 |
| momentum_reversal | bull | 0.0438 | 0.2103 | 0.2083 | 0.1288 | 0.1176 |
| mom_x_lowvol_20_20 | bull | 0.0364 | 0.1826 | 0.1994 | 0.1591 | 0.1156 |
| fund_pb | bear | 0.1538 | 0.2710 | 0.5676 | 0.3699 | 0.3888 |
| trend_lowvol | bear | 0.1062 | 0.2106 | 0.5043 | 0.3699 | 0.3454 |
| mom_x_lowvol_20_20 | bear | 0.1353 | 0.2690 | 0.5031 | 0.3151 | 0.3308 |
| momentum_reversal | bear | 0.1331 | 0.3074 | 0.4329 | 0.2877 | 0.2787 |
| rsi_vol_combo | bear | 0.0944 | 0.2779 | 0.3396 | 0.1781 | 0.2000 |

### 商业航天

- **Neutral**: ['momentum_reversal', 'mom_x_lowvol_20_20', 'trend_lowvol'] (单因子IC=0.0865, 组合IC=0.0962)
  - weights: [0.3398, 0.3308, 0.3295]
- **Bull**: ['low_downside', 'fund_pb', 'volatility'] (单因子IC=0.0944, 组合IC=0.1193)
  - bull_weights: [0.3574, 0.3245, 0.3181]
- **Bear**: ['mom_x_lowvol_20_20'] (单因子IC=0.0974, 组合IC=0.0974)
  - bear_weights: [1.0]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| momentum_reversal | neutral | 0.0866 | 0.1626 | 0.5328 | 0.3925 | 0.3709 |
| mom_x_lowvol_20_20 | neutral | 0.0825 | 0.1569 | 0.5258 | 0.3737 | 0.3611 |
| trend_lowvol | neutral | 0.0903 | 0.1711 | 0.5277 | 0.3633 | 0.3597 |
| rsi_vol_combo | neutral | 0.0686 | 0.1541 | 0.4450 | 0.3403 | 0.2982 |
| fund_pb | neutral | 0.0715 | 0.1701 | 0.4204 | 0.3006 | 0.2734 |
| volatility | neutral | 0.0628 | 0.1800 | 0.3488 | 0.2589 | 0.2195 |
| fund_profit_growth | neutral | 0.0406 | 0.1200 | 0.3382 | 0.2463 | 0.2107 |
| turnover_stability | neutral | 0.0319 | 0.0965 | 0.3310 | 0.2317 | 0.2038 |
| fund_pe | neutral | 0.0424 | 0.1494 | 0.2839 | 0.1983 | 0.1701 |
| fund_score | neutral | 0.0357 | 0.1548 | 0.2304 | 0.1733 | 0.1352 |
| fund_revenue_growth | neutral | 0.0219 | 0.1179 | 0.1858 | 0.1566 | 0.1075 |
| low_downside | neutral | 0.0293 | 0.1712 | 0.1712 | 0.1315 | 0.0968 |
| low_downside | bull | 0.0881 | 0.1350 | 0.6529 | 0.4621 | 0.4773 |
| fund_pb | bull | 0.0982 | 0.1597 | 0.6151 | 0.4091 | 0.4334 |
| volatility | bull | 0.0968 | 0.1588 | 0.6095 | 0.3939 | 0.4248 |
| trend_lowvol | bull | 0.0842 | 0.1632 | 0.5162 | 0.4470 | 0.3735 |
| momentum_reversal | bull | 0.0561 | 0.1466 | 0.3826 | 0.2803 | 0.2449 |
| fund_pe | bull | 0.0570 | 0.1474 | 0.3865 | 0.2121 | 0.2343 |
| mom_x_lowvol_20_20 | bull | 0.0492 | 0.1386 | 0.3547 | 0.3030 | 0.2311 |
| rsi_vol_combo | bull | 0.0375 | 0.1326 | 0.2826 | 0.1667 | 0.1649 |
| fund_revenue_growth | bull | 0.0268 | 0.1093 | 0.2450 | 0.2424 | 0.1522 |
| fund_score | bull | 0.0324 | 0.1379 | 0.2346 | 0.1818 | 0.1386 |
| fund_profit_growth | bull | 0.0217 | 0.1123 | 0.1933 | 0.1364 | 0.1098 |
| fund_roe | bull | 0.0182 | 0.1493 | 0.1222 | 0.1288 | 0.0690 |
| mom_x_lowvol_20_20 | bear | 0.0974 | 0.1864 | 0.5223 | 0.3973 | 0.3649 |
| momentum_reversal | bear | 0.0959 | 0.1810 | 0.5298 | 0.3425 | 0.3556 |
| rsi_vol_combo | bear | 0.0662 | 0.1509 | 0.4391 | 0.3151 | 0.2887 |
| trend_lowvol | bear | 0.0986 | 0.2188 | 0.4507 | 0.1781 | 0.2655 |
| turnover_stability | bear | 0.0363 | 0.0959 | 0.3779 | 0.1507 | 0.2174 |
| fund_revenue_growth | bear | 0.0435 | 0.1412 | 0.3080 | 0.1781 | 0.1814 |
| fund_gross_margin | bear | 0.0446 | 0.1597 | 0.2792 | 0.2055 | 0.1683 |
| top_fractal_volume | bear | 0.0202 | 0.0823 | 0.2458 | 0.1613 | 0.1428 |

### 固态电池

- **Neutral**: ['fund_pb', 'fund_pe', 'fund_profit_growth'] (单因子IC=0.0603, 组合IC=0.0777)
  - weights: [0.3526, 0.3259, 0.3216]
- **Bull**: ['low_downside', 'volatility', 'fund_pb'] (单因子IC=0.0811, 组合IC=0.0991)
  - bull_weights: [0.3799, 0.3396, 0.2806]
- **Bear**: ['rsi_vol_combo', 'momentum_reversal', 'mom_x_lowvol_20_20'] (单因子IC=0.1116, 组合IC=0.1239)
  - bear_weights: [0.3557, 0.3529, 0.2914]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| fund_pb | neutral | 0.0686 | 0.1776 | 0.3860 | 0.3215 | 0.2551 |
| fund_pe | neutral | 0.0593 | 0.1616 | 0.3666 | 0.2860 | 0.2358 |
| fund_profit_growth | neutral | 0.0531 | 0.1453 | 0.3654 | 0.2735 | 0.2327 |
| volatility | neutral | 0.0541 | 0.1867 | 0.2900 | 0.1795 | 0.1710 |
| momentum_reversal | neutral | 0.0456 | 0.1652 | 0.2758 | 0.1608 | 0.1601 |
| mom_x_lowvol_20_20 | neutral | 0.0426 | 0.1601 | 0.2661 | 0.1524 | 0.1533 |
| trend_lowvol | neutral | 0.0477 | 0.2027 | 0.2355 | 0.1628 | 0.1369 |
| fund_score | neutral | 0.0379 | 0.1751 | 0.2165 | 0.1566 | 0.1252 |
| rsi_vol_combo | neutral | 0.0293 | 0.1538 | 0.1906 | 0.1211 | 0.1068 |
| fund_revenue_growth | neutral | 0.0237 | 0.1309 | 0.1807 | 0.1232 | 0.1015 |
| low_downside | neutral | 0.0327 | 0.1894 | 0.1726 | 0.1253 | 0.0971 |
| turnover_stability | neutral | 0.0115 | 0.1300 | 0.0886 | 0.1441 | 0.0507 |
| low_downside | bull | 0.0829 | 0.1338 | 0.6192 | 0.4394 | 0.4456 |
| volatility | bull | 0.0876 | 0.1575 | 0.5564 | 0.4318 | 0.3983 |
| fund_pb | bull | 0.0728 | 0.1516 | 0.4801 | 0.3712 | 0.3292 |
| turnover_stability | bull | 0.0348 | 0.1161 | 0.2993 | 0.2955 | 0.1939 |
| fund_gross_margin | bull | 0.0320 | 0.1114 | 0.2874 | 0.3106 | 0.1884 |
| fund_pe | bull | 0.0476 | 0.1562 | 0.3047 | 0.1894 | 0.1812 |
| trend_lowvol | bull | 0.0429 | 0.1671 | 0.2566 | 0.2045 | 0.1545 |
| fund_score | bull | 0.0343 | 0.1924 | 0.1780 | 0.1667 | 0.1038 |
| fund_profit_growth | bull | 0.0276 | 0.1562 | 0.1769 | 0.1515 | 0.1019 |
| wash_sale_score | bull | 0.0126 | 0.0837 | 0.1505 | 0.1136 | 0.0838 |
| fund_revenue_growth | bull | 0.0147 | 0.1500 | 0.0979 | 0.1667 | 0.0571 |
| rsi_vol_combo | bear | 0.0987 | 0.1310 | 0.7536 | 0.5616 | 0.5885 |
| momentum_reversal | bear | 0.1231 | 0.1647 | 0.7476 | 0.5616 | 0.5837 |
| mom_x_lowvol_20_20 | bear | 0.1131 | 0.1799 | 0.6285 | 0.5342 | 0.4821 |
| bb_width_20 | bear | 0.0828 | 0.1986 | 0.4167 | 0.2877 | 0.2683 |
| trend_lowvol | bear | 0.0694 | 0.1819 | 0.3815 | 0.1233 | 0.2143 |
| fund_pb | bear | 0.0526 | 0.2272 | 0.2313 | 0.1507 | 0.1331 |

### 国产芯片

- **Neutral**: ['momentum_reversal', 'mom_x_lowvol_20_20', 'fund_pe'] (单因子IC=0.0665, 组合IC=0.0814)
  - weights: [0.3437, 0.3379, 0.3184]
- **Bull**: ['volatility', 'turnover_stability', 'fund_pb'] (单因子IC=0.0744, 组合IC=0.1073)
  - bull_weights: [0.3524, 0.3507, 0.2969]
- **Bear**: ['mom_x_lowvol_20_20', 'momentum_reversal'] (单因子IC=0.1383, 组合IC=0.1399)
  - bear_weights: [0.5059, 0.4941]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| momentum_reversal | neutral | 0.0724 | 0.1596 | 0.4538 | 0.3111 | 0.2975 |
| mom_x_lowvol_20_20 | neutral | 0.0713 | 0.1573 | 0.4534 | 0.2902 | 0.2925 |
| fund_pe | neutral | 0.0558 | 0.1319 | 0.4231 | 0.3027 | 0.2756 |
| rsi_vol_combo | neutral | 0.0591 | 0.1523 | 0.3879 | 0.2818 | 0.2486 |
| trend_lowvol | neutral | 0.0688 | 0.1777 | 0.3873 | 0.2610 | 0.2442 |
| fund_pb | neutral | 0.0637 | 0.1700 | 0.3747 | 0.2735 | 0.2386 |
| fund_profit_growth | neutral | 0.0473 | 0.1363 | 0.3468 | 0.3194 | 0.2288 |
| volatility | neutral | 0.0593 | 0.1724 | 0.3436 | 0.2568 | 0.2159 |
| turnover_stability | neutral | 0.0326 | 0.0971 | 0.3361 | 0.2630 | 0.2122 |
| fund_score | neutral | 0.0431 | 0.1720 | 0.2508 | 0.2338 | 0.1547 |
| low_downside | neutral | 0.0319 | 0.1584 | 0.2016 | 0.1962 | 0.1206 |
| fund_roe | neutral | 0.0345 | 0.1649 | 0.2090 | 0.1357 | 0.1187 |
| fund_revenue_growth | neutral | 0.0253 | 0.1283 | 0.1970 | 0.1670 | 0.1150 |
| volatility | bull | 0.0914 | 0.1569 | 0.5825 | 0.4015 | 0.4082 |
| turnover_stability | bull | 0.0477 | 0.0867 | 0.5499 | 0.4773 | 0.4062 |
| fund_pb | bull | 0.0842 | 0.1669 | 0.5043 | 0.3636 | 0.3438 |
| low_downside | bull | 0.0672 | 0.1442 | 0.4659 | 0.3409 | 0.3124 |
| fund_pe | bull | 0.0637 | 0.1380 | 0.4614 | 0.2955 | 0.2989 |
| trend_lowvol | bull | 0.0700 | 0.1747 | 0.4007 | 0.3788 | 0.2763 |
| mom_x_lowvol_20_20 | bull | 0.0499 | 0.1558 | 0.3201 | 0.3030 | 0.2086 |
| momentum_reversal | bull | 0.0429 | 0.1586 | 0.2701 | 0.2879 | 0.1739 |
| fund_revenue_growth | bull | 0.0245 | 0.1384 | 0.1768 | 0.1061 | 0.0978 |
| fund_profit_growth | bull | 0.0208 | 0.1235 | 0.1685 | 0.1136 | 0.0938 |
| rsi_vol_combo | bull | 0.0228 | 0.1494 | 0.1524 | 0.1061 | 0.0843 |
| mom_x_lowvol_20_20 | bear | 0.1391 | 0.1607 | 0.8656 | 0.5342 | 0.6640 |
| momentum_reversal | bear | 0.1375 | 0.1597 | 0.8609 | 0.5068 | 0.6486 |
| rsi_vol_combo | bear | 0.0922 | 0.1337 | 0.6897 | 0.6438 | 0.5669 |
| trend_lowvol | bear | 0.0924 | 0.1881 | 0.4911 | 0.2055 | 0.2960 |
| bb_width_20 | bear | 0.0678 | 0.1512 | 0.4483 | 0.3151 | 0.2948 |
| turnover_stability | bear | 0.0151 | 0.0759 | 0.1990 | 0.1507 | 0.1145 |

### 国产软件

- **Neutral**: ['momentum_reversal', 'trend_lowvol'] (单因子IC=0.0982, 组合IC=0.1092)
  - weights: [0.5162, 0.4838]
- **Bull**: ['volatility', 'fund_pb'] (单因子IC=0.1196, 组合IC=0.1419)
  - bull_weights: [0.5116, 0.4884]
- **Bear**: ['trend_lowvol', 'momentum_reversal', 'mom_x_lowvol_20_20'] (单因子IC=0.1254, 组合IC=0.1372)
  - bear_weights: [0.3877, 0.3077, 0.3047]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| momentum_reversal | neutral | 0.0981 | 0.1502 | 0.6529 | 0.4572 | 0.4757 |
| trend_lowvol | neutral | 0.0983 | 0.1618 | 0.6075 | 0.4676 | 0.4458 |
| mom_x_lowvol_20_20 | neutral | 0.0933 | 0.1500 | 0.6217 | 0.4154 | 0.4400 |
| fund_pb | neutral | 0.0683 | 0.1268 | 0.5390 | 0.3695 | 0.3691 |
| rsi_vol_combo | neutral | 0.0762 | 0.1468 | 0.5189 | 0.3987 | 0.3629 |
| volatility | neutral | 0.0797 | 0.1652 | 0.4823 | 0.3633 | 0.3287 |
| fund_profit_growth | neutral | 0.0456 | 0.1107 | 0.4118 | 0.3111 | 0.2700 |
| fund_pe | neutral | 0.0558 | 0.1386 | 0.4028 | 0.3090 | 0.2636 |
| turnover_stability | neutral | 0.0403 | 0.1012 | 0.3985 | 0.2985 | 0.2588 |
| fund_score | neutral | 0.0447 | 0.1321 | 0.3385 | 0.2610 | 0.2134 |
| low_downside | neutral | 0.0424 | 0.1660 | 0.2551 | 0.2505 | 0.1595 |
| fund_revenue_growth | neutral | 0.0218 | 0.1016 | 0.2143 | 0.2067 | 0.1293 |
| fund_roe | neutral | 0.0306 | 0.1462 | 0.2094 | 0.1420 | 0.1196 |
| volatility | bull | 0.1202 | 0.1474 | 0.8157 | 0.5682 | 0.6396 |
| fund_pb | bull | 0.1189 | 0.1512 | 0.7864 | 0.5530 | 0.6106 |
| low_downside | bull | 0.0957 | 0.1360 | 0.7039 | 0.5152 | 0.5333 |
| trend_lowvol | bull | 0.1045 | 0.1608 | 0.6495 | 0.5303 | 0.4970 |
| fund_pe | bull | 0.0832 | 0.1349 | 0.6173 | 0.4773 | 0.4559 |
| mom_x_lowvol_20_20 | bull | 0.0864 | 0.1390 | 0.6213 | 0.4015 | 0.4354 |
| momentum_reversal | bull | 0.0764 | 0.1451 | 0.5267 | 0.3333 | 0.3511 |
| turnover_stability | bull | 0.0350 | 0.0988 | 0.3546 | 0.1970 | 0.2122 |
| rsi_vol_combo | bull | 0.0422 | 0.1448 | 0.2917 | 0.2045 | 0.1757 |
| fund_score | bull | 0.0246 | 0.1328 | 0.1851 | 0.1515 | 0.1066 |
| fund_roe | bull | 0.0154 | 0.1307 | 0.1178 | 0.1364 | 0.0669 |
| fund_revenue_growth | bull | 0.0109 | 0.0985 | 0.1105 | 0.1515 | 0.0636 |
| trend_lowvol | bear | 0.1361 | 0.1611 | 0.8448 | 0.6164 | 0.6827 |
| momentum_reversal | bear | 0.1199 | 0.1668 | 0.7192 | 0.5068 | 0.5419 |
| mom_x_lowvol_20_20 | bear | 0.1201 | 0.1656 | 0.7254 | 0.4795 | 0.5366 |
| rsi_vol_combo | bear | 0.0828 | 0.1434 | 0.5777 | 0.3973 | 0.4036 |
| fund_profit_growth | bear | 0.0623 | 0.1375 | 0.4530 | 0.4247 | 0.3227 |
| fund_score | bear | 0.0622 | 0.1573 | 0.3953 | 0.1781 | 0.2328 |
| fund_pb | bear | 0.0363 | 0.1166 | 0.3112 | 0.1781 | 0.1833 |
| fund_gross_margin | bear | 0.0302 | 0.1380 | 0.2191 | 0.2877 | 0.1411 |

### 国资云概念

- **Neutral**: ['trend_lowvol', 'mom_x_lowvol_20_20'] (单因子IC=0.0964, 组合IC=0.1139)
  - weights: [0.5102, 0.4898]
- **Bull**: ['volatility', 'momentum_reversal', 'fund_pb'] (单因子IC=0.1191, 组合IC=0.1606)
  - bull_weights: [0.3701, 0.3233, 0.3066]
- **Bear**: ['mom_x_lowvol_20_20'] (单因子IC=0.139, 组合IC=0.139)
  - bear_weights: [1.0]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| trend_lowvol | neutral | 0.1031 | 0.2915 | 0.3536 | 0.2578 | 0.2224 |
| mom_x_lowvol_20_20 | neutral | 0.0897 | 0.2668 | 0.3361 | 0.2704 | 0.2135 |
| momentum_reversal | neutral | 0.0884 | 0.2711 | 0.3261 | 0.2359 | 0.2015 |
| volatility | neutral | 0.0838 | 0.2861 | 0.2928 | 0.2578 | 0.1841 |
| low_downside | neutral | 0.0730 | 0.2878 | 0.2538 | 0.2119 | 0.1538 |
| fund_pb | neutral | 0.0654 | 0.2926 | 0.2237 | 0.2067 | 0.1349 |
| rsi_vol_combo | neutral | 0.0579 | 0.2702 | 0.2143 | 0.1670 | 0.1250 |
| fund_pe | neutral | 0.0618 | 0.3014 | 0.2050 | 0.1868 | 0.1216 |
| fund_roe | neutral | 0.0495 | 0.2669 | 0.1855 | 0.1232 | 0.1042 |
| volatility | bull | 0.1406 | 0.3155 | 0.4456 | 0.3788 | 0.3072 |
| momentum_reversal | bull | 0.1090 | 0.2569 | 0.4242 | 0.2652 | 0.2683 |
| fund_pb | bull | 0.1077 | 0.2782 | 0.3873 | 0.3144 | 0.2545 |
| mom_x_lowvol_20_20 | bull | 0.1112 | 0.2835 | 0.3921 | 0.2727 | 0.2495 |
| trend_lowvol | bull | 0.1151 | 0.3216 | 0.3581 | 0.2500 | 0.2238 |
| low_downside | bull | 0.0968 | 0.3112 | 0.3109 | 0.2121 | 0.1884 |
| rsi_vol_combo | bull | 0.0624 | 0.2649 | 0.2355 | 0.1402 | 0.1342 |
| fund_profit_growth | bull | 0.0302 | 0.1830 | 0.1649 | 0.1364 | 0.0937 |
| turnover_stability | bull | 0.0353 | 0.2226 | 0.1585 | 0.1098 | 0.0879 |
| fund_score | bull | 0.0403 | 0.2822 | 0.1429 | 0.1364 | 0.0812 |
| mom_x_lowvol_20_20 | bear | 0.1390 | 0.3158 | 0.4401 | 0.4658 | 0.3225 |
| trend_lowvol | bear | 0.1161 | 0.2630 | 0.4416 | 0.3151 | 0.2904 |
| momentum_reversal | bear | 0.0641 | 0.2997 | 0.2139 | 0.3151 | 0.1406 |
| fund_gross_margin | bear | 0.0393 | 0.1817 | 0.2166 | 0.1918 | 0.1290 |
| fund_score | bear | 0.0437 | 0.2343 | 0.1864 | 0.1233 | 0.1047 |

### 土地流转

- **Neutral**: ['volatility', 'momentum_reversal', 'turnover_stability'] (单因子IC=0.0821, 组合IC=0.1156)
  - weights: [0.3531, 0.3256, 0.3213]
- **Bull**: ['low_downside', 'volatility'] (单因子IC=0.0876, 组合IC=0.0986)
  - bull_weights: [0.5909, 0.4091]
- **Bear**: ['momentum_reversal', 'turnover_stability', 'mom_x_lowvol_20_20'] (单因子IC=0.1367, 组合IC=0.1925)
  - bear_weights: [0.3731, 0.3537, 0.2732]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| volatility | neutral | 0.0907 | 0.2169 | 0.4181 | 0.3445 | 0.2810 |
| momentum_reversal | neutral | 0.0887 | 0.2221 | 0.3992 | 0.2985 | 0.2592 |
| turnover_stability | neutral | 0.0671 | 0.1719 | 0.3902 | 0.3111 | 0.2558 |
| mom_x_lowvol_20_20 | neutral | 0.0755 | 0.2149 | 0.3514 | 0.2735 | 0.2238 |
| trend_lowvol | neutral | 0.0822 | 0.2372 | 0.3464 | 0.2589 | 0.2181 |
| rsi_vol_combo | neutral | 0.0714 | 0.2110 | 0.3387 | 0.2797 | 0.2167 |
| low_downside | neutral | 0.0616 | 0.2017 | 0.3055 | 0.2902 | 0.1971 |
| fund_pe | neutral | 0.0610 | 0.2064 | 0.2954 | 0.2380 | 0.1828 |
| fund_score | neutral | 0.0537 | 0.2044 | 0.2625 | 0.2390 | 0.1626 |
| fund_profit_growth | neutral | 0.0478 | 0.1810 | 0.2640 | 0.2088 | 0.1596 |
| fund_pb | neutral | 0.0521 | 0.2224 | 0.2342 | 0.1754 | 0.1376 |
| fund_revenue_growth | neutral | 0.0267 | 0.1926 | 0.1388 | 0.1211 | 0.0778 |
| fund_roe | neutral | 0.0276 | 0.2175 | 0.1267 | 0.1148 | 0.0706 |
| low_downside | bull | 0.0932 | 0.1853 | 0.5030 | 0.3864 | 0.3487 |
| volatility | bull | 0.0821 | 0.2242 | 0.3662 | 0.3182 | 0.2414 |
| fund_pb | bull | 0.0689 | 0.1861 | 0.3704 | 0.2424 | 0.2301 |
| trend_lowvol | bull | 0.0676 | 0.2210 | 0.3060 | 0.2348 | 0.1890 |
| momentum_reversal | bull | 0.0562 | 0.1921 | 0.2924 | 0.1894 | 0.1739 |
| fund_profit_growth | bull | 0.0328 | 0.1571 | 0.2086 | 0.1742 | 0.1225 |
| rsi_vol_combo | bull | 0.0321 | 0.1790 | 0.1792 | 0.1212 | 0.1005 |
| fund_gross_margin | bull | 0.0204 | 0.1447 | 0.1410 | 0.1136 | 0.0785 |
| fund_revenue_growth | bull | 0.0232 | 0.1723 | 0.1345 | 0.1667 | 0.0784 |
| fund_score | bull | 0.0212 | 0.1771 | 0.1197 | 0.1212 | 0.0671 |
| mom_x_lowvol_20_20 | bull | 0.0214 | 0.2004 | 0.1068 | 0.1364 | 0.0607 |
| momentum_reversal | bear | 0.1681 | 0.2420 | 0.6945 | 0.4795 | 0.5137 |
| turnover_stability | bear | 0.1159 | 0.1760 | 0.6584 | 0.4795 | 0.4871 |
| mom_x_lowvol_20_20 | bear | 0.1263 | 0.2207 | 0.5721 | 0.3151 | 0.3761 |
| rsi_vol_combo | bear | 0.1058 | 0.2391 | 0.4423 | 0.3151 | 0.2908 |
| fund_roe | bear | 0.0861 | 0.2054 | 0.4190 | 0.3425 | 0.2813 |
| trend_lowvol | bear | 0.0995 | 0.2408 | 0.4133 | 0.1781 | 0.2434 |
| fund_revenue_growth | bear | 0.0653 | 0.2061 | 0.3170 | 0.3014 | 0.2063 |
| fund_gross_margin | bear | 0.0414 | 0.1356 | 0.3057 | 0.2329 | 0.1885 |
| top_fractal_volume | bear | 0.0515 | 0.1802 | 0.2855 | 0.2703 | 0.1813 |
| fund_score | bear | 0.0593 | 0.2055 | 0.2888 | 0.2329 | 0.1780 |
| vol_confirm | bear | 0.0595 | 0.2458 | 0.2422 | 0.2329 | 0.1493 |
| fund_pe | bear | 0.0318 | 0.1467 | 0.2167 | 0.1781 | 0.1277 |

### 土壤修复

- **Neutral**: ['fund_pb', 'momentum_reversal', 'turnover_stability'] (单因子IC=0.0815, 组合IC=0.143)
  - weights: [0.3846, 0.3182, 0.2972]
- **Bull**: ['momentum_reversal', 'mom_x_lowvol_20_20', 'fund_pe'] (单因子IC=0.0495, 组合IC=0.0891)
  - bull_weights: [0.408, 0.2974, 0.2946]
- **Bear**: ['fund_score', 'bb_width_20', 'fund_pe'] (单因子IC=0.1039, 组合IC=0.1257)
  - bear_weights: [0.3339, 0.3335, 0.3326]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| fund_pb | neutral | 0.0922 | 0.2359 | 0.3909 | 0.2808 | 0.2504 |
| momentum_reversal | neutral | 0.0882 | 0.2695 | 0.3275 | 0.2651 | 0.2072 |
| turnover_stability | neutral | 0.0641 | 0.2118 | 0.3026 | 0.2787 | 0.1935 |
| volatility | neutral | 0.0793 | 0.2747 | 0.2887 | 0.2797 | 0.1847 |
| fund_pe | neutral | 0.0753 | 0.2619 | 0.2875 | 0.2276 | 0.1765 |
| mom_x_lowvol_20_20 | neutral | 0.0760 | 0.2693 | 0.2822 | 0.2317 | 0.1738 |
| trend_lowvol | neutral | 0.0613 | 0.2747 | 0.2231 | 0.2067 | 0.1346 |
| rsi_vol_combo | neutral | 0.0586 | 0.2544 | 0.2303 | 0.1670 | 0.1344 |
| fund_score | neutral | 0.0422 | 0.2365 | 0.1783 | 0.1566 | 0.1031 |
| fund_profit_growth | neutral | 0.0307 | 0.2049 | 0.1498 | 0.1002 | 0.0824 |
| momentum_reversal | bull | 0.0595 | 0.2396 | 0.2482 | 0.1667 | 0.1448 |
| mom_x_lowvol_20_20 | bull | 0.0463 | 0.2477 | 0.1870 | 0.1288 | 0.1055 |
| fund_pe | bull | 0.0427 | 0.2520 | 0.1693 | 0.2348 | 0.1045 |
| rsi_vol_combo | bull | 0.0378 | 0.2163 | 0.1748 | 0.1667 | 0.1020 |
| turnover_stability | bull | 0.0310 | 0.1942 | 0.1596 | 0.1439 | 0.0913 |
| trend_lowvol | bull | 0.0435 | 0.2747 | 0.1585 | 0.1402 | 0.0904 |
| low_downside | bull | 0.0243 | 0.2337 | 0.1039 | 0.1288 | 0.0587 |
| fund_gross_margin | bull | 0.0213 | 0.2207 | 0.0967 | 0.1136 | 0.0538 |
| fund_score | bear | 0.0996 | 0.2151 | 0.4630 | 0.3699 | 0.3171 |
| bb_width_20 | bear | 0.0936 | 0.2146 | 0.4363 | 0.4521 | 0.3168 |
| fund_pe | bear | 0.1185 | 0.2518 | 0.4706 | 0.3425 | 0.3159 |
| fund_roe | bear | 0.1152 | 0.2559 | 0.4503 | 0.3699 | 0.3084 |
| momentum_reversal | bear | 0.1196 | 0.2868 | 0.4170 | 0.4247 | 0.2970 |
| wash_sale_score | bear | 0.0989 | 0.2239 | 0.4418 | 0.3056 | 0.2884 |
| trend_lowvol | bear | 0.1072 | 0.2588 | 0.4142 | 0.2877 | 0.2667 |
| fund_pb | bear | 0.0782 | 0.2163 | 0.3616 | 0.3425 | 0.2427 |
| mom_x_lowvol_20_20 | bear | 0.0933 | 0.2681 | 0.3480 | 0.3151 | 0.2288 |
| fund_profit_growth | bear | 0.0502 | 0.1738 | 0.2890 | 0.3151 | 0.1901 |
| rsi_vol_combo | bear | 0.0613 | 0.2365 | 0.2594 | 0.2329 | 0.1599 |
| fund_gross_margin | bear | 0.0422 | 0.2088 | 0.2019 | 0.2603 | 0.1272 |

### 在线教育

- **Neutral**: ['volatility', 'trend_lowvol', 'fund_pb'] (单因子IC=0.1079, 组合IC=0.1376)
  - weights: [0.3425, 0.3387, 0.3188]
- **Bull**: ['low_downside', 'volatility', 'fund_pb'] (单因子IC=0.0964, 组合IC=0.112)
  - bull_weights: [0.3612, 0.3406, 0.2981]
- **Bear**: ['trend_lowvol', 'mom_x_lowvol_20_20', 'momentum_reversal'] (单因子IC=0.1552, 组合IC=0.1802)
  - bear_weights: [0.4127, 0.2966, 0.2907]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| volatility | neutral | 0.1096 | 0.2179 | 0.5028 | 0.4280 | 0.3590 |
| trend_lowvol | neutral | 0.1108 | 0.2195 | 0.5046 | 0.4071 | 0.3550 |
| fund_pb | neutral | 0.1034 | 0.2102 | 0.4917 | 0.3591 | 0.3342 |
| momentum_reversal | neutral | 0.0936 | 0.2070 | 0.4519 | 0.3653 | 0.3085 |
| mom_x_lowvol_20_20 | neutral | 0.0868 | 0.2015 | 0.4308 | 0.3403 | 0.2887 |
| rsi_vol_combo | neutral | 0.0684 | 0.1831 | 0.3735 | 0.2797 | 0.2390 |
| low_downside | neutral | 0.0749 | 0.2102 | 0.3561 | 0.2839 | 0.2286 |
| turnover_stability | neutral | 0.0394 | 0.1195 | 0.3300 | 0.2610 | 0.2080 |
| fund_profit_growth | neutral | 0.0406 | 0.1409 | 0.2882 | 0.1858 | 0.1709 |
| fund_score | neutral | 0.0521 | 0.1939 | 0.2688 | 0.2296 | 0.1652 |
| fund_pe | neutral | 0.0568 | 0.2343 | 0.2423 | 0.1837 | 0.1434 |
| fund_roe | neutral | 0.0373 | 0.2213 | 0.1686 | 0.1002 | 0.0928 |
| fund_revenue_growth | neutral | 0.0194 | 0.1396 | 0.1386 | 0.1148 | 0.0773 |
| low_downside | bull | 0.0914 | 0.1612 | 0.5672 | 0.4545 | 0.4125 |
| volatility | bull | 0.1099 | 0.1979 | 0.5551 | 0.4015 | 0.3890 |
| fund_pb | bull | 0.0881 | 0.1734 | 0.5078 | 0.3409 | 0.3405 |
| trend_lowvol | bull | 0.0848 | 0.2100 | 0.4037 | 0.3030 | 0.2630 |
| mom_x_lowvol_20_20 | bull | 0.0566 | 0.1734 | 0.3262 | 0.2576 | 0.2051 |
| momentum_reversal | bull | 0.0456 | 0.1723 | 0.2644 | 0.2197 | 0.1612 |
| trend_lowvol | bear | 0.1535 | 0.1690 | 0.9087 | 0.6712 | 0.7593 |
| mom_x_lowvol_20_20 | bear | 0.1552 | 0.2182 | 0.7115 | 0.5342 | 0.5458 |
| momentum_reversal | bear | 0.1567 | 0.2168 | 0.7230 | 0.4795 | 0.5348 |
| rsi_vol_combo | bear | 0.0869 | 0.1920 | 0.4526 | 0.3699 | 0.3100 |
| turnover_stability | bear | 0.0532 | 0.1270 | 0.4194 | 0.3973 | 0.2930 |
| fund_profit_growth | bear | 0.0773 | 0.1658 | 0.4661 | 0.2329 | 0.2873 |
| fund_gross_margin | bear | 0.0422 | 0.1286 | 0.3285 | 0.3425 | 0.2205 |
| bb_width_20 | bear | 0.0569 | 0.2309 | 0.2464 | 0.1233 | 0.1384 |

### 地下管网

- **Neutral**: ['mom_x_lowvol_20_20'] (单因子IC=0.1086, 组合IC=0.1086)
  - weights: [1.0]
- **Bull**: ['low_downside', 'fund_pb'] (单因子IC=0.0892, 组合IC=0.1092)
  - bull_weights: [0.5425, 0.4575]
- **Bear**: ['mom_x_lowvol_20_20', 'trend_lowvol'] (单因子IC=0.086, 组合IC=0.0925)
  - bear_weights: [0.5005, 0.4995]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| mom_x_lowvol_20_20 | neutral | 0.1086 | 0.2241 | 0.4845 | 0.3977 | 0.3386 |
| momentum_reversal | neutral | 0.0985 | 0.2262 | 0.4355 | 0.3570 | 0.2955 |
| trend_lowvol | neutral | 0.1012 | 0.2463 | 0.4110 | 0.3559 | 0.2786 |
| fund_pb | neutral | 0.0820 | 0.2065 | 0.3970 | 0.2651 | 0.2511 |
| turnover_stability | neutral | 0.0502 | 0.1764 | 0.2846 | 0.2630 | 0.1797 |
| rsi_vol_combo | neutral | 0.0635 | 0.2236 | 0.2839 | 0.2046 | 0.1710 |
| volatility | neutral | 0.0717 | 0.2582 | 0.2776 | 0.2004 | 0.1666 |
| low_downside | neutral | 0.0426 | 0.2357 | 0.1807 | 0.1983 | 0.1083 |
| fund_pe | neutral | 0.0467 | 0.2548 | 0.1832 | 0.1347 | 0.1039 |
| low_downside | bull | 0.0933 | 0.2038 | 0.4578 | 0.3561 | 0.3104 |
| fund_pb | bull | 0.0850 | 0.2017 | 0.4214 | 0.2424 | 0.2618 |
| fund_pe | bull | 0.0733 | 0.2248 | 0.3261 | 0.2045 | 0.1964 |
| volatility | bull | 0.0662 | 0.2242 | 0.2951 | 0.2045 | 0.1777 |
| trend_lowvol | bull | 0.0574 | 0.2117 | 0.2713 | 0.2500 | 0.1696 |
| stroke_phase | bull | 0.0347 | 0.1789 | 0.1941 | 0.2197 | 0.1184 |
| turnover_stability | bull | 0.0195 | 0.1385 | 0.1409 | 0.1667 | 0.0822 |
| mom_x_lowvol_20_20 | bear | 0.0835 | 0.2285 | 0.3653 | 0.2192 | 0.2227 |
| trend_lowvol | bear | 0.0886 | 0.2730 | 0.3245 | 0.3699 | 0.2223 |
| momentum_reversal | bear | 0.0853 | 0.2341 | 0.3644 | 0.1781 | 0.2146 |
| fund_revenue_growth | bear | 0.0450 | 0.1573 | 0.2858 | 0.1507 | 0.1644 |
| vol_confirm | bear | 0.0414 | 0.1918 | 0.2156 | 0.1233 | 0.1211 |

### 地摊经济

- **Neutral**: ['momentum_reversal', 'mom_x_lowvol_20_20', 'fund_pb'] (单因子IC=0.0916, 组合IC=0.1202)
  - weights: [0.3467, 0.3269, 0.3264]
- **Bull**: ['low_downside', 'fund_pb', 'momentum_reversal'] (单因子IC=0.0959, 组合IC=0.143)
  - bull_weights: [0.4325, 0.2905, 0.277]
- **Bear**: ['fund_pe', 'fund_profit_growth', 'momentum_reversal'] (单因子IC=0.1201, 组合IC=0.2219)
  - bear_weights: [0.3896, 0.3547, 0.2556]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| momentum_reversal | neutral | 0.0969 | 0.2507 | 0.3866 | 0.3267 | 0.2565 |
| mom_x_lowvol_20_20 | neutral | 0.0917 | 0.2464 | 0.3719 | 0.3006 | 0.2419 |
| fund_pb | neutral | 0.0861 | 0.2302 | 0.3740 | 0.2912 | 0.2414 |
| trend_lowvol | neutral | 0.0925 | 0.2721 | 0.3399 | 0.2714 | 0.2161 |
| rsi_vol_combo | neutral | 0.0701 | 0.2360 | 0.2973 | 0.2109 | 0.1800 |
| fund_pe | neutral | 0.0633 | 0.2219 | 0.2853 | 0.2380 | 0.1766 |
| turnover_stability | neutral | 0.0592 | 0.2121 | 0.2791 | 0.1952 | 0.1668 |
| low_downside | neutral | 0.0506 | 0.2243 | 0.2257 | 0.1952 | 0.1349 |
| volatility | neutral | 0.0470 | 0.2423 | 0.1941 | 0.1493 | 0.1115 |
| fund_profit_growth | neutral | 0.0393 | 0.2234 | 0.1758 | 0.1326 | 0.0996 |
| low_downside | bull | 0.1053 | 0.2008 | 0.5246 | 0.4545 | 0.3815 |
| fund_pb | bull | 0.1006 | 0.2542 | 0.3957 | 0.2955 | 0.2563 |
| momentum_reversal | bull | 0.0817 | 0.2268 | 0.3603 | 0.3561 | 0.2443 |
| turnover_stability | bull | 0.0675 | 0.2132 | 0.3167 | 0.1894 | 0.1883 |
| trend_lowvol | bull | 0.0619 | 0.2308 | 0.2683 | 0.1591 | 0.1555 |
| rsi_vol_combo | bull | 0.0582 | 0.2257 | 0.2578 | 0.1364 | 0.1465 |
| fund_gross_margin | bull | 0.0344 | 0.2376 | 0.1447 | 0.1705 | 0.0847 |
| fund_pe | bull | 0.0259 | 0.2119 | 0.1224 | 0.1439 | 0.0700 |
| mom_x_lowvol_20_20 | bull | 0.0223 | 0.2228 | 0.1003 | 0.1439 | 0.0573 |
| fund_pe | bear | 0.1670 | 0.2720 | 0.6141 | 0.3425 | 0.4122 |
| fund_profit_growth | bear | 0.0972 | 0.1845 | 0.5269 | 0.4247 | 0.3753 |
| momentum_reversal | bear | 0.0960 | 0.2334 | 0.4113 | 0.3151 | 0.2705 |
| rsi_vol_combo | bear | 0.0782 | 0.2169 | 0.3606 | 0.2877 | 0.2322 |
| mom_x_lowvol_20_20 | bear | 0.0753 | 0.2373 | 0.3172 | 0.1507 | 0.1825 |
| turnover_stability | bear | 0.0553 | 0.1877 | 0.2949 | 0.2055 | 0.1777 |
| wash_sale_score | bear | 0.0520 | 0.1932 | 0.2694 | 0.1875 | 0.1599 |
| fund_score | bear | 0.0510 | 0.2250 | 0.2267 | 0.2055 | 0.1367 |
| fund_pb | bear | 0.0471 | 0.2299 | 0.2049 | 0.1781 | 0.1207 |

### 垃圾分类

- **Neutral**: ['turnover_stability', 'momentum_reversal', 'mom_x_lowvol_20_20'] (单因子IC=0.0699, 组合IC=0.0868)
  - weights: [0.3407, 0.3304, 0.3289]
- **Bull**: ['fund_pb', 'low_downside', 'turnover_stability'] (单因子IC=0.0745, 组合IC=0.1045)
  - bull_weights: [0.3782, 0.3238, 0.2979]
- **Bear**: ['momentum_reversal', 'trend_lowvol'] (单因子IC=0.1508, 组合IC=0.1733)
  - bear_weights: [0.5184, 0.4816]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| turnover_stability | neutral | 0.0530 | 0.1274 | 0.4161 | 0.3006 | 0.2706 |
| momentum_reversal | neutral | 0.0792 | 0.1976 | 0.4010 | 0.3090 | 0.2624 |
| mom_x_lowvol_20_20 | neutral | 0.0776 | 0.1935 | 0.4010 | 0.3027 | 0.2612 |
| fund_profit_growth | neutral | 0.0522 | 0.1396 | 0.3737 | 0.2484 | 0.2333 |
| fund_pb | neutral | 0.0804 | 0.2193 | 0.3665 | 0.2630 | 0.2314 |
| trend_lowvol | neutral | 0.0742 | 0.2120 | 0.3501 | 0.2923 | 0.2262 |
| volatility | neutral | 0.0746 | 0.2296 | 0.3250 | 0.2338 | 0.2005 |
| fund_score | neutral | 0.0520 | 0.1722 | 0.3022 | 0.2317 | 0.1861 |
| rsi_vol_combo | neutral | 0.0541 | 0.1875 | 0.2885 | 0.2046 | 0.1738 |
| fund_pe | neutral | 0.0587 | 0.2222 | 0.2642 | 0.1879 | 0.1569 |
| low_downside | neutral | 0.0425 | 0.2017 | 0.2109 | 0.1795 | 0.1244 |
| fund_revenue_growth | neutral | 0.0222 | 0.1290 | 0.1719 | 0.1336 | 0.0974 |
| fund_gross_margin | neutral | 0.0257 | 0.1600 | 0.1604 | 0.1649 | 0.0934 |
| fund_pb | bull | 0.0953 | 0.1811 | 0.5262 | 0.3939 | 0.3668 |
| low_downside | bull | 0.0760 | 0.1623 | 0.4683 | 0.3409 | 0.3140 |
| turnover_stability | bull | 0.0521 | 0.1189 | 0.4383 | 0.3182 | 0.2889 |
| volatility | bull | 0.0888 | 0.1995 | 0.4451 | 0.2879 | 0.2866 |
| fund_pe | bull | 0.0648 | 0.2120 | 0.3054 | 0.1591 | 0.1770 |
| momentum_reversal | bull | 0.0480 | 0.1647 | 0.2912 | 0.1439 | 0.1666 |
| mom_x_lowvol_20_20 | bull | 0.0439 | 0.1603 | 0.2736 | 0.1667 | 0.1596 |
| trend_lowvol | bull | 0.0506 | 0.1960 | 0.2580 | 0.2045 | 0.1554 |
| fund_profit_growth | bull | 0.0314 | 0.1404 | 0.2238 | 0.1288 | 0.1263 |
| stroke_phase | bull | 0.0192 | 0.1264 | 0.1518 | 0.1818 | 0.0897 |
| fund_gross_margin | bull | 0.0172 | 0.1475 | 0.1168 | 0.1288 | 0.0659 |
| momentum_reversal | bear | 0.1603 | 0.2282 | 0.7025 | 0.5616 | 0.5485 |
| trend_lowvol | bear | 0.1414 | 0.1976 | 0.7155 | 0.4247 | 0.5096 |
| mom_x_lowvol_20_20 | bear | 0.1499 | 0.2591 | 0.5786 | 0.4795 | 0.4280 |
| rsi_vol_combo | bear | 0.0880 | 0.1739 | 0.5061 | 0.4247 | 0.3605 |
| bb_width_20 | bear | 0.0969 | 0.2466 | 0.3929 | 0.3151 | 0.2583 |
| fund_profit_growth | bear | 0.0522 | 0.1554 | 0.3363 | 0.3425 | 0.2257 |
| fund_revenue_growth | bear | 0.0337 | 0.1338 | 0.2515 | 0.1781 | 0.1482 |

### 基因测序

- **Neutral**: ['volatility', 'fund_pb', 'trend_lowvol'] (单因子IC=0.0865, 组合IC=0.1179)
  - weights: [0.4457, 0.3189, 0.2354]
- **Bull**: ['momentum_reversal'] (单因子IC=0.1115, 组合IC=0.1115)
  - bull_weights: [1.0]
- **Bear**: ['trend_lowvol', 'mom_x_lowvol_20_20'] (单因子IC=0.1607, 组合IC=0.1968)
  - bear_weights: [0.5455, 0.4545]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| volatility | neutral | 0.1094 | 0.2156 | 0.5072 | 0.4061 | 0.3566 |
| fund_pb | neutral | 0.0776 | 0.2014 | 0.3856 | 0.3236 | 0.2552 |
| trend_lowvol | neutral | 0.0726 | 0.2285 | 0.3176 | 0.1858 | 0.1883 |
| momentum_reversal | neutral | 0.0692 | 0.2240 | 0.3089 | 0.1691 | 0.1806 |
| mom_x_lowvol_20_20 | neutral | 0.0580 | 0.2211 | 0.2622 | 0.1649 | 0.1527 |
| rsi_vol_combo | neutral | 0.0541 | 0.2152 | 0.2513 | 0.2119 | 0.1523 |
| turnover_stability | neutral | 0.0456 | 0.1958 | 0.2330 | 0.1545 | 0.1345 |
| fund_profit_growth | neutral | 0.0437 | 0.2059 | 0.2123 | 0.2140 | 0.1288 |
| low_downside | neutral | 0.0453 | 0.2327 | 0.1946 | 0.1566 | 0.1126 |
| fund_score | neutral | 0.0237 | 0.2226 | 0.1064 | 0.1190 | 0.0595 |
| fund_gross_margin | neutral | 0.0203 | 0.2146 | 0.0944 | 0.1211 | 0.0529 |
| momentum_reversal | bull | 0.1115 | 0.2124 | 0.5247 | 0.4091 | 0.3697 |
| mom_x_lowvol_20_20 | bull | 0.0982 | 0.2068 | 0.4752 | 0.4242 | 0.3384 |
| volatility | bull | 0.0893 | 0.2161 | 0.4131 | 0.3485 | 0.2785 |
| rsi_vol_combo | bull | 0.0832 | 0.2043 | 0.4072 | 0.3030 | 0.2653 |
| turnover_stability | bull | 0.0696 | 0.1885 | 0.3693 | 0.3333 | 0.2462 |
| fund_profit_growth | bull | 0.0539 | 0.1726 | 0.3121 | 0.2803 | 0.1998 |
| trend_lowvol | bull | 0.0743 | 0.2347 | 0.3165 | 0.2121 | 0.1918 |
| low_downside | bull | 0.0378 | 0.1985 | 0.1907 | 0.1288 | 0.1076 |
| fund_revenue_growth | bull | 0.0337 | 0.1882 | 0.1793 | 0.1288 | 0.1012 |
| fund_score | bull | 0.0287 | 0.1883 | 0.1524 | 0.1818 | 0.0900 |
| fund_pb | bull | 0.0254 | 0.2404 | 0.1055 | 0.1288 | 0.0596 |
| trend_lowvol | bear | 0.1682 | 0.2029 | 0.8287 | 0.5890 | 0.6585 |
| mom_x_lowvol_20_20 | bear | 0.1531 | 0.2180 | 0.7026 | 0.5616 | 0.5486 |
| momentum_reversal | bear | 0.1403 | 0.2495 | 0.5624 | 0.3973 | 0.3929 |
| fund_pb | bear | 0.1051 | 0.2246 | 0.4680 | 0.2603 | 0.2949 |
| fund_revenue_growth | bear | 0.0714 | 0.1877 | 0.3805 | 0.2329 | 0.2345 |
| rsi_vol_combo | bear | 0.0926 | 0.2725 | 0.3399 | 0.1781 | 0.2002 |
| fund_score | bear | 0.0609 | 0.2114 | 0.2878 | 0.2877 | 0.1853 |
| fund_pe | bear | 0.0463 | 0.2026 | 0.2287 | 0.1233 | 0.1285 |
| fund_roe | bear | 0.0422 | 0.2170 | 0.1945 | 0.1507 | 0.1119 |

### 基金重仓

- **Neutral**: ['fund_revenue_growth', 'fund_profit_growth', 'fund_pb'] (单因子IC=0.0446, 组合IC=0.0621)
  - weights: [0.3636, 0.3504, 0.286]
- **Bull**: ['limit_pullback_score', 'ema20_slope', 'top_fractal_volume'] (单因子IC=0.0337, 组合IC=0.0511)
  - bull_weights: [0.4105, 0.2994, 0.2902]
- **Bear**: ['mom_x_lowvol_20_20', 'bb_width_20'] (单因子IC=0.1153, 组合IC=0.1336)
  - bear_weights: [0.5591, 0.4409]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| fund_revenue_growth | neutral | 0.0442 | 0.1568 | 0.2818 | 0.2380 | 0.1744 |
| fund_profit_growth | neutral | 0.0436 | 0.1568 | 0.2781 | 0.2088 | 0.1681 |
| fund_pb | neutral | 0.0462 | 0.1921 | 0.2403 | 0.1420 | 0.1372 |
| fund_pe | neutral | 0.0440 | 0.1895 | 0.2321 | 0.1691 | 0.1357 |
| mom_x_lowvol_20_20 | neutral | 0.0398 | 0.1765 | 0.2258 | 0.1148 | 0.1258 |
| fund_score | neutral | 0.0301 | 0.1732 | 0.1740 | 0.1357 | 0.0988 |
| rsi_vol_combo | neutral | 0.0310 | 0.1773 | 0.1747 | 0.1002 | 0.0961 |
| volatility | neutral | 0.0272 | 0.1844 | 0.1476 | 0.1336 | 0.0837 |
| limit_pullback_score | bull | 0.0390 | 0.1317 | 0.2962 | 0.1970 | 0.1773 |
| ema20_slope | bull | 0.0376 | 0.1719 | 0.2188 | 0.1818 | 0.1293 |
| top_fractal_volume | bull | 0.0243 | 0.1170 | 0.2082 | 0.2041 | 0.1253 |
| relative_strength | bull | 0.0353 | 0.1609 | 0.2196 | 0.1212 | 0.1231 |
| ma_alignment | bull | 0.0279 | 0.1700 | 0.1641 | 0.1212 | 0.0920 |
| fund_pb | bull | 0.0196 | 0.1979 | 0.0993 | 0.1591 | 0.0575 |
| mom_x_lowvol_20_20 | bear | 0.1293 | 0.1895 | 0.6824 | 0.5068 | 0.5141 |
| bb_width_20 | bear | 0.1012 | 0.1744 | 0.5805 | 0.3973 | 0.4055 |
| momentum_reversal | bear | 0.0991 | 0.2036 | 0.4865 | 0.3973 | 0.3399 |
| trend_lowvol | bear | 0.0860 | 0.2521 | 0.3410 | 0.2329 | 0.2102 |
| rsi_vol_combo | bear | 0.0593 | 0.1824 | 0.3251 | 0.2603 | 0.2048 |
| fund_pb | bear | 0.0565 | 0.1851 | 0.3052 | 0.1233 | 0.1714 |
| limit_pullback_score | bear | 0.0323 | 0.1268 | 0.2547 | 0.2603 | 0.1605 |
| fund_profit_growth | bear | 0.0331 | 0.1598 | 0.2072 | 0.3151 | 0.1362 |

### 增强现实

- **Neutral**: ['mom_x_lowvol_20_20', 'momentum_reversal', 'volatility'] (单因子IC=0.095, 组合IC=0.1084)
  - weights: [0.3503, 0.3415, 0.3082]
- **Bull**: ['fund_pb', 'volatility', 'turnover_stability'] (单因子IC=0.0688, 组合IC=0.1074)
  - bull_weights: [0.3874, 0.3145, 0.2981]
- **Bear**: ['fund_profit_growth', 'wash_sale_score', 'mom_x_lowvol_20_20'] (单因子IC=0.0794, 组合IC=0.1178)
  - bear_weights: [0.4945, 0.2743, 0.2312]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| mom_x_lowvol_20_20 | neutral | 0.0954 | 0.2233 | 0.4274 | 0.3424 | 0.2869 |
| momentum_reversal | neutral | 0.1022 | 0.2385 | 0.4287 | 0.3048 | 0.2797 |
| volatility | neutral | 0.0874 | 0.2256 | 0.3874 | 0.3027 | 0.2524 |
| rsi_vol_combo | neutral | 0.0885 | 0.2379 | 0.3721 | 0.2630 | 0.2350 |
| trend_lowvol | neutral | 0.0911 | 0.2561 | 0.3559 | 0.2411 | 0.2209 |
| fund_pb | neutral | 0.0699 | 0.2241 | 0.3120 | 0.2944 | 0.2019 |
| turnover_stability | neutral | 0.0498 | 0.1722 | 0.2895 | 0.2234 | 0.1771 |
| fund_score | neutral | 0.0573 | 0.2233 | 0.2565 | 0.2088 | 0.1550 |
| fund_profit_growth | neutral | 0.0490 | 0.2047 | 0.2396 | 0.1795 | 0.1413 |
| fund_pe | neutral | 0.0420 | 0.2214 | 0.1897 | 0.1420 | 0.1083 |
| low_downside | neutral | 0.0343 | 0.2090 | 0.1643 | 0.1430 | 0.0939 |
| fund_revenue_growth | neutral | 0.0268 | 0.2086 | 0.1287 | 0.1357 | 0.0731 |
| fund_pb | bull | 0.0921 | 0.2397 | 0.3841 | 0.2576 | 0.2415 |
| volatility | bull | 0.0680 | 0.2075 | 0.3276 | 0.1970 | 0.1961 |
| turnover_stability | bull | 0.0465 | 0.1602 | 0.2903 | 0.2803 | 0.1858 |
| low_downside | bull | 0.0601 | 0.2109 | 0.2848 | 0.2955 | 0.1845 |
| fund_pe | bull | 0.0665 | 0.2408 | 0.2760 | 0.1667 | 0.1610 |
| fund_profit_growth | bull | 0.0451 | 0.1892 | 0.2382 | 0.1667 | 0.1390 |
| fund_revenue_growth | bull | 0.0433 | 0.2010 | 0.2154 | 0.2121 | 0.1306 |
| fund_score | bull | 0.0438 | 0.2206 | 0.1986 | 0.1439 | 0.1136 |
| stroke_phase | bull | 0.0288 | 0.1697 | 0.1698 | 0.1061 | 0.0939 |
| fund_gross_margin | bull | 0.0197 | 0.1977 | 0.0994 | 0.1364 | 0.0565 |
| fund_profit_growth | bear | 0.1152 | 0.2026 | 0.5684 | 0.3151 | 0.3737 |
| wash_sale_score | bear | 0.0556 | 0.1927 | 0.2888 | 0.4359 | 0.2073 |
| mom_x_lowvol_20_20 | bear | 0.0675 | 0.2223 | 0.3037 | 0.1507 | 0.1747 |

### 复合集流体

- **Neutral**: ['fund_pb', 'fund_pe', 'volatility'] (单因子IC=0.0551, 组合IC=0.0708)
  - weights: [0.3835, 0.3285, 0.288]
- **Bull**: ['fund_revenue_growth', 'fund_gross_margin', 'low_downside'] (单因子IC=0.0574, 组合IC=0.0994)
  - bull_weights: [0.4901, 0.2566, 0.2534]
- **Bear**: ['relative_strength'] (单因子IC=0.0684, 组合IC=0.0684)
  - bear_weights: [1.0]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| fund_pb | neutral | 0.0601 | 0.2732 | 0.2201 | 0.2025 | 0.1324 |
| fund_pe | neutral | 0.0575 | 0.2992 | 0.1923 | 0.1795 | 0.1134 |
| volatility | neutral | 0.0477 | 0.2860 | 0.1669 | 0.1910 | 0.0994 |
| fund_profit_growth | neutral | 0.0363 | 0.2375 | 0.1530 | 0.1514 | 0.0881 |
| rsi_vol_combo | neutral | 0.0334 | 0.2422 | 0.1378 | 0.1347 | 0.0782 |
| fund_revenue_growth | bull | 0.0745 | 0.1766 | 0.4220 | 0.3182 | 0.2781 |
| fund_gross_margin | bull | 0.0449 | 0.1939 | 0.2316 | 0.2576 | 0.1456 |
| low_downside | bull | 0.0528 | 0.2136 | 0.2473 | 0.1629 | 0.1438 |
| fund_pe | bull | 0.0695 | 0.3005 | 0.2312 | 0.1515 | 0.1331 |
| volatility | bull | 0.0487 | 0.2314 | 0.2107 | 0.1591 | 0.1221 |
| trend_lowvol | bull | 0.0311 | 0.2507 | 0.1240 | 0.1136 | 0.0690 |
| fund_profit_growth | bull | 0.0264 | 0.2502 | 0.1054 | 0.1629 | 0.0613 |
| relative_strength | bear | 0.0684 | 0.2848 | 0.2401 | 0.1233 | 0.1349 |

### 多模态AI

- **Neutral**: ['momentum_reversal', 'fund_pb'] (单因子IC=0.097, 组合IC=0.1296)
  - weights: [0.5164, 0.4836]
- **Bull**: ['trend_lowvol', 'volatility', 'fund_pb'] (单因子IC=0.1156, 组合IC=0.1466)
  - bull_weights: [0.3364, 0.3358, 0.3278]
- **Bear**: ['fund_profit_growth', 'mom_x_lowvol_20_20'] (单因子IC=0.11, 组合IC=0.1509)
  - bear_weights: [0.5041, 0.4959]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| momentum_reversal | neutral | 0.0980 | 0.1932 | 0.5074 | 0.3466 | 0.3416 |
| fund_pb | neutral | 0.0960 | 0.1972 | 0.4872 | 0.3132 | 0.3199 |
| mom_x_lowvol_20_20 | neutral | 0.0948 | 0.2000 | 0.4742 | 0.3382 | 0.3173 |
| trend_lowvol | neutral | 0.0917 | 0.2014 | 0.4554 | 0.3361 | 0.3042 |
| volatility | neutral | 0.0995 | 0.2274 | 0.4378 | 0.3455 | 0.2945 |
| rsi_vol_combo | neutral | 0.0817 | 0.1907 | 0.4282 | 0.3257 | 0.2839 |
| fund_profit_growth | neutral | 0.0513 | 0.1740 | 0.2947 | 0.2056 | 0.1777 |
| fund_score | neutral | 0.0513 | 0.1845 | 0.2783 | 0.2129 | 0.1688 |
| low_downside | neutral | 0.0518 | 0.2098 | 0.2468 | 0.2338 | 0.1522 |
| fund_pe | neutral | 0.0527 | 0.2089 | 0.2524 | 0.2025 | 0.1518 |
| turnover_stability | neutral | 0.0292 | 0.1506 | 0.1940 | 0.1441 | 0.1110 |
| fund_revenue_growth | neutral | 0.0124 | 0.1667 | 0.0747 | 0.1086 | 0.0414 |
| trend_lowvol | bull | 0.1184 | 0.1851 | 0.6398 | 0.4697 | 0.4701 |
| volatility | bull | 0.1215 | 0.1972 | 0.6164 | 0.5227 | 0.4693 |
| fund_pb | bull | 0.1068 | 0.1687 | 0.6332 | 0.4470 | 0.4581 |
| fund_pe | bull | 0.0859 | 0.1679 | 0.5116 | 0.3561 | 0.3469 |
| mom_x_lowvol_20_20 | bull | 0.0773 | 0.1937 | 0.3993 | 0.3030 | 0.2602 |
| low_downside | bull | 0.0667 | 0.1752 | 0.3806 | 0.3258 | 0.2523 |
| momentum_reversal | bull | 0.0667 | 0.1936 | 0.3444 | 0.2576 | 0.2165 |
| fund_roe | bull | 0.0443 | 0.1698 | 0.2610 | 0.1364 | 0.1483 |
| wash_sale_score | bull | 0.0329 | 0.1527 | 0.2153 | 0.1525 | 0.1241 |
| fund_profit_growth | bull | 0.0320 | 0.1594 | 0.2009 | 0.1667 | 0.1172 |
| fund_score | bull | 0.0342 | 0.1760 | 0.1942 | 0.1136 | 0.1081 |
| fund_profit_growth | bear | 0.0883 | 0.1465 | 0.6022 | 0.3699 | 0.4125 |
| mom_x_lowvol_20_20 | bear | 0.1318 | 0.2269 | 0.5809 | 0.3973 | 0.4058 |
| momentum_reversal | bear | 0.1124 | 0.2182 | 0.5150 | 0.4521 | 0.3739 |
| trend_lowvol | bear | 0.0939 | 0.1891 | 0.4964 | 0.2877 | 0.3196 |
| fund_score | bear | 0.0646 | 0.1657 | 0.3899 | 0.2055 | 0.2350 |
| bb_width_20 | bear | 0.0758 | 0.1877 | 0.4039 | 0.1507 | 0.2324 |
| rsi_vol_combo | bear | 0.0489 | 0.1925 | 0.2541 | 0.2329 | 0.1566 |
| wash_sale_score | bear | 0.0332 | 0.1305 | 0.2544 | 0.1333 | 0.1442 |

### 大数据

- **Neutral**: ['momentum_reversal', 'trend_lowvol'] (单因子IC=0.095, 组合IC=0.1031)
  - weights: [0.527, 0.473]
- **Bull**: ['volatility', 'fund_pb', 'trend_lowvol'] (单因子IC=0.1161, 组合IC=0.1447)
  - bull_weights: [0.3631, 0.3601, 0.2768]
- **Bear**: ['trend_lowvol', 'mom_x_lowvol_20_20'] (单因子IC=0.132, 组合IC=0.1512)
  - bear_weights: [0.5918, 0.4082]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| momentum_reversal | neutral | 0.0929 | 0.1474 | 0.6302 | 0.4718 | 0.4638 |
| trend_lowvol | neutral | 0.0971 | 0.1702 | 0.5705 | 0.4593 | 0.4162 |
| mom_x_lowvol_20_20 | neutral | 0.0862 | 0.1467 | 0.5879 | 0.4134 | 0.4154 |
| fund_pb | neutral | 0.0772 | 0.1362 | 0.5670 | 0.4071 | 0.3989 |
| rsi_vol_combo | neutral | 0.0703 | 0.1385 | 0.5074 | 0.3841 | 0.3512 |
| volatility | neutral | 0.0775 | 0.1805 | 0.4296 | 0.3486 | 0.2897 |
| fund_profit_growth | neutral | 0.0467 | 0.1098 | 0.4254 | 0.3173 | 0.2802 |
| fund_pe | neutral | 0.0540 | 0.1430 | 0.3776 | 0.3173 | 0.2487 |
| turnover_stability | neutral | 0.0333 | 0.0903 | 0.3683 | 0.3048 | 0.2403 |
| fund_score | neutral | 0.0438 | 0.1432 | 0.3062 | 0.2150 | 0.1860 |
| low_downside | neutral | 0.0409 | 0.1707 | 0.2397 | 0.2192 | 0.1461 |
| fund_roe | neutral | 0.0350 | 0.1557 | 0.2250 | 0.1983 | 0.1348 |
| fund_revenue_growth | neutral | 0.0199 | 0.0927 | 0.2147 | 0.1754 | 0.1262 |
| volatility | bull | 0.1208 | 0.1388 | 0.8705 | 0.6894 | 0.7353 |
| fund_pb | bull | 0.1126 | 0.1240 | 0.9082 | 0.6061 | 0.7293 |
| trend_lowvol | bull | 0.1147 | 0.1566 | 0.7328 | 0.5303 | 0.5607 |
| low_downside | bull | 0.0888 | 0.1307 | 0.6797 | 0.5455 | 0.5252 |
| mom_x_lowvol_20_20 | bull | 0.0885 | 0.1396 | 0.6339 | 0.4697 | 0.4658 |
| momentum_reversal | bull | 0.0837 | 0.1399 | 0.5981 | 0.4394 | 0.4304 |
| fund_pe | bull | 0.0566 | 0.1385 | 0.4088 | 0.3409 | 0.2741 |
| turnover_stability | bull | 0.0303 | 0.0847 | 0.3582 | 0.3258 | 0.2375 |
| rsi_vol_combo | bull | 0.0520 | 0.1360 | 0.3826 | 0.2197 | 0.2333 |
| fund_profit_growth | bull | 0.0354 | 0.1044 | 0.3387 | 0.1742 | 0.1988 |
| fund_revenue_growth | bull | 0.0203 | 0.0878 | 0.2316 | 0.1667 | 0.1351 |
| stroke_phase | bull | 0.0184 | 0.1022 | 0.1803 | 0.1061 | 0.0997 |
| trend_lowvol | bear | 0.1373 | 0.1451 | 0.9458 | 0.6986 | 0.8033 |
| mom_x_lowvol_20_20 | bear | 0.1266 | 0.1785 | 0.7097 | 0.5616 | 0.5541 |
| momentum_reversal | bear | 0.1161 | 0.1861 | 0.6234 | 0.5616 | 0.4868 |
| fund_revenue_growth | bear | 0.0569 | 0.1008 | 0.5648 | 0.3151 | 0.3714 |
| rsi_vol_combo | bear | 0.0708 | 0.1596 | 0.4434 | 0.3973 | 0.3098 |
| fund_profit_growth | bear | 0.0489 | 0.1131 | 0.4325 | 0.3425 | 0.2903 |
| fund_score | bear | 0.0488 | 0.1511 | 0.3233 | 0.2329 | 0.1993 |
| turnover_stability | bear | 0.0227 | 0.1089 | 0.2081 | 0.3699 | 0.1425 |

### 大盘价值

- **Neutral**: ['fund_pe', 'fund_pb'] (单因子IC=0.0813, 组合IC=0.0945)
  - weights: [0.5681, 0.4319]
- **Bull**: ['relative_strength'] (单因子IC=0.0443, 组合IC=0.0443)
  - bull_weights: [1.0]
- **Bear**: ['momentum_reversal', 'trend_lowvol', 'bb_width_20'] (单因子IC=0.1074, 组合IC=0.1278)
  - bear_weights: [0.3864, 0.3467, 0.2669]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| fund_pe | neutral | 0.0902 | 0.2303 | 0.3916 | 0.2777 | 0.2502 |
| fund_pb | neutral | 0.0724 | 0.2375 | 0.3047 | 0.2484 | 0.1902 |
| fund_profit_growth | neutral | 0.0232 | 0.1473 | 0.1576 | 0.1336 | 0.0893 |
| top_fractal_volume | neutral | 0.0172 | 0.1169 | 0.1471 | 0.1523 | 0.0847 |
| turnover_stability | neutral | 0.0153 | 0.1488 | 0.1029 | 0.1294 | 0.0581 |
| relative_strength | bull | 0.0443 | 0.2453 | 0.1804 | 0.1894 | 0.1073 |
| ma_alignment | bull | 0.0421 | 0.2469 | 0.1705 | 0.1970 | 0.1020 |
| ema20_slope | bull | 0.0399 | 0.2416 | 0.1651 | 0.1742 | 0.0969 |
| fund_pe | bull | 0.0450 | 0.3020 | 0.1491 | 0.1894 | 0.0887 |
| momentum_reversal | bear | 0.1145 | 0.2128 | 0.5381 | 0.3699 | 0.3686 |
| trend_lowvol | bear | 0.1322 | 0.2792 | 0.4735 | 0.3973 | 0.3308 |
| bb_width_20 | bear | 0.0755 | 0.1869 | 0.4040 | 0.2603 | 0.2546 |
| wash_sale_score | bear | 0.0450 | 0.1242 | 0.3624 | 0.3846 | 0.2509 |
| mom_x_lowvol_20_20 | bear | 0.0624 | 0.1635 | 0.3816 | 0.2329 | 0.2352 |
| rsi_vol_combo | bear | 0.0744 | 0.1924 | 0.3867 | 0.1781 | 0.2278 |
| volatility | bear | 0.0332 | 0.1013 | 0.3281 | 0.1781 | 0.1932 |
| fund_pb | bear | 0.0638 | 0.2692 | 0.2370 | 0.2603 | 0.1493 |
| top_fractal_volume | bear | 0.0231 | 0.1120 | 0.2063 | 0.1864 | 0.1224 |
| fund_profit_growth | bear | 0.0264 | 0.1456 | 0.1817 | 0.2055 | 0.1095 |

### 大盘成长

- **Neutral**: ['fund_pe'] (单因子IC=0.06, 组合IC=0.0598)
  - weights: [1.0]
- **Bull**: ['ema20_slope'] (单因子IC=0.0656, 组合IC=0.0656)
  - bull_weights: [1.0]
- **Bear**: ['mom_x_lowvol_20_20', 'momentum_reversal', 'bb_width_20'] (单因子IC=0.1482, 组合IC=0.1636)
  - bear_weights: [0.3661, 0.3221, 0.3118]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| fund_pe | neutral | 0.0600 | 0.2021 | 0.2966 | 0.2422 | 0.1842 |
| fund_pb | neutral | 0.0457 | 0.1942 | 0.2354 | 0.1858 | 0.1396 |
| trend_lowvol | neutral | 0.0560 | 0.2569 | 0.2182 | 0.1253 | 0.1227 |
| fund_roe | neutral | 0.0318 | 0.1891 | 0.1680 | 0.1461 | 0.0963 |
| top_fractal_volume | neutral | 0.0194 | 0.1353 | 0.1432 | 0.1274 | 0.0807 |
| fund_profit_growth | neutral | 0.0196 | 0.1784 | 0.1099 | 0.1211 | 0.0616 |
| volatility | neutral | 0.0213 | 0.2054 | 0.1037 | 0.1127 | 0.0577 |
| fund_score | neutral | 0.0204 | 0.2051 | 0.0997 | 0.1023 | 0.0549 |
| ema20_slope | bull | 0.0656 | 0.2096 | 0.3133 | 0.2576 | 0.1970 |
| ma_alignment | bull | 0.0592 | 0.2121 | 0.2789 | 0.2727 | 0.1775 |
| relative_strength | bull | 0.0523 | 0.1898 | 0.2755 | 0.2879 | 0.1774 |
| bb_width_20 | bull | 0.0494 | 0.2172 | 0.2273 | 0.2197 | 0.1386 |
| fund_pe | bull | 0.0541 | 0.2417 | 0.2240 | 0.2045 | 0.1349 |
| fund_roe | bull | 0.0349 | 0.1685 | 0.2072 | 0.1591 | 0.1201 |
| fund_score | bull | 0.0360 | 0.1806 | 0.1993 | 0.1970 | 0.1193 |
| fund_pb | bull | 0.0340 | 0.1967 | 0.1728 | 0.1742 | 0.1015 |
| turnover_stability | bull | 0.0280 | 0.1709 | 0.1637 | 0.1288 | 0.0924 |
| vol_confirm | bull | 0.0178 | 0.1812 | 0.0983 | 0.1742 | 0.0577 |
| mom_x_lowvol_20_20 | bear | 0.1468 | 0.1875 | 0.7829 | 0.6164 | 0.6327 |
| momentum_reversal | bear | 0.1502 | 0.2144 | 0.7008 | 0.5890 | 0.5568 |
| bb_width_20 | bear | 0.1475 | 0.2174 | 0.6784 | 0.5890 | 0.5390 |
| rsi_vol_combo | bear | 0.0936 | 0.1563 | 0.5988 | 0.4521 | 0.4347 |
| trend_lowvol | bear | 0.1173 | 0.3017 | 0.3887 | 0.1507 | 0.2237 |
| fund_pb | bear | 0.0538 | 0.1663 | 0.3232 | 0.2603 | 0.2037 |

### 大飞机

- **Neutral**: ['momentum_reversal', 'mom_x_lowvol_20_20', 'trend_lowvol'] (单因子IC=0.086, 组合IC=0.1028)
  - weights: [0.3813, 0.3197, 0.299]
- **Bull**: ['low_downside', 'trend_lowvol'] (单因子IC=0.1262, 组合IC=0.1471)
  - bull_weights: [0.5626, 0.4374]
- **Bear**: ['rsi_vol_combo', 'fund_gross_margin'] (单因子IC=0.0779, 组合IC=0.0911)
  - bear_weights: [0.5245, 0.4755]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| momentum_reversal | neutral | 0.0981 | 0.1988 | 0.4935 | 0.3486 | 0.3328 |
| mom_x_lowvol_20_20 | neutral | 0.0797 | 0.1869 | 0.4263 | 0.3090 | 0.2790 |
| trend_lowvol | neutral | 0.0803 | 0.1995 | 0.4025 | 0.2965 | 0.2609 |
| rsi_vol_combo | neutral | 0.0750 | 0.1863 | 0.4027 | 0.2630 | 0.2543 |
| fund_pe | neutral | 0.0535 | 0.1610 | 0.3326 | 0.2401 | 0.2062 |
| turnover_stability | neutral | 0.0446 | 0.1449 | 0.3076 | 0.2735 | 0.1959 |
| fund_profit_growth | neutral | 0.0498 | 0.1606 | 0.3101 | 0.2234 | 0.1897 |
| fund_pb | neutral | 0.0573 | 0.2179 | 0.2631 | 0.2213 | 0.1607 |
| volatility | neutral | 0.0467 | 0.1942 | 0.2406 | 0.1837 | 0.1424 |
| fund_score | neutral | 0.0381 | 0.1928 | 0.1978 | 0.1608 | 0.1148 |
| low_downside | neutral | 0.0257 | 0.2017 | 0.1274 | 0.1378 | 0.0725 |
| fund_revenue_growth | neutral | 0.0179 | 0.1475 | 0.1212 | 0.1023 | 0.0668 |
| low_downside | bull | 0.1280 | 0.1521 | 0.8419 | 0.5909 | 0.6697 |
| trend_lowvol | bull | 0.1243 | 0.1790 | 0.6941 | 0.5000 | 0.5206 |
| volatility | bull | 0.1154 | 0.1824 | 0.6329 | 0.4848 | 0.4699 |
| fund_pb | bull | 0.0915 | 0.1910 | 0.4790 | 0.3333 | 0.3193 |
| fund_pe | bull | 0.0727 | 0.1947 | 0.3733 | 0.2576 | 0.2347 |
| momentum_reversal | bull | 0.0600 | 0.1773 | 0.3384 | 0.2879 | 0.2179 |
| mom_x_lowvol_20_20 | bull | 0.0519 | 0.1609 | 0.3226 | 0.2955 | 0.2089 |
| turnover_stability | bull | 0.0240 | 0.1246 | 0.1927 | 0.1818 | 0.1138 |
| stroke_phase | bull | 0.0163 | 0.1171 | 0.1390 | 0.1818 | 0.0821 |
| top_fractal_volume | bull | 0.0120 | 0.1198 | 0.1002 | 0.1286 | 0.0566 |
| rsi_vol_combo | bear | 0.0808 | 0.1801 | 0.4485 | 0.3425 | 0.3011 |
| fund_gross_margin | bear | 0.0751 | 0.1695 | 0.4428 | 0.2329 | 0.2730 |
| fund_pe | bear | 0.0633 | 0.1500 | 0.4221 | 0.2877 | 0.2718 |
| momentum_reversal | bear | 0.0844 | 0.2243 | 0.3763 | 0.2877 | 0.2422 |
| bb_width_20 | bear | 0.0625 | 0.2068 | 0.3022 | 0.2329 | 0.1863 |

### 天然气

- **Neutral**: ['mom_x_lowvol_20_20', 'momentum_reversal', 'fund_pb'] (单因子IC=0.0792, 组合IC=0.1103)
  - weights: [0.3661, 0.3362, 0.2977]
- **Bull**: ['fund_profit_growth', 'low_downside', 'fund_pb'] (单因子IC=0.0508, 组合IC=0.0755)
  - bull_weights: [0.4478, 0.2823, 0.2699]
- **Bear**: ['trend_lowvol', 'fund_profit_growth', 'rsi_vol_combo'] (单因子IC=0.1217, 组合IC=0.1577)
  - bear_weights: [0.3623, 0.3404, 0.2973]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| mom_x_lowvol_20_20 | neutral | 0.0837 | 0.1983 | 0.4224 | 0.3424 | 0.2835 |
| momentum_reversal | neutral | 0.0825 | 0.2065 | 0.3996 | 0.3027 | 0.2603 |
| fund_pb | neutral | 0.0713 | 0.1896 | 0.3762 | 0.2255 | 0.2305 |
| trend_lowvol | neutral | 0.0804 | 0.2199 | 0.3656 | 0.2610 | 0.2305 |
| volatility | neutral | 0.0785 | 0.2241 | 0.3501 | 0.2985 | 0.2273 |
| turnover_stability | neutral | 0.0421 | 0.1446 | 0.2912 | 0.2150 | 0.1769 |
| fund_profit_growth | neutral | 0.0370 | 0.1289 | 0.2868 | 0.2328 | 0.1768 |
| rsi_vol_combo | neutral | 0.0546 | 0.1960 | 0.2786 | 0.1983 | 0.1669 |
| fund_pe | neutral | 0.0613 | 0.2233 | 0.2746 | 0.2046 | 0.1654 |
| fund_score | neutral | 0.0423 | 0.1629 | 0.2598 | 0.2046 | 0.1565 |
| low_downside | neutral | 0.0397 | 0.2067 | 0.1919 | 0.1524 | 0.1106 |
| fund_revenue_growth | neutral | 0.0206 | 0.1227 | 0.1681 | 0.1169 | 0.0939 |
| fund_profit_growth | bull | 0.0447 | 0.0992 | 0.4504 | 0.3636 | 0.3071 |
| low_downside | bull | 0.0511 | 0.1671 | 0.3060 | 0.2652 | 0.1936 |
| fund_pb | bull | 0.0566 | 0.1864 | 0.3034 | 0.2197 | 0.1851 |
| volatility | bull | 0.0505 | 0.1950 | 0.2592 | 0.1591 | 0.1502 |
| fund_pe | bull | 0.0428 | 0.2106 | 0.2033 | 0.1439 | 0.1163 |
| exhaustion_risk | bull | 0.0152 | 0.1136 | 0.1337 | 0.1174 | 0.0747 |
| stroke_phase | bull | 0.0142 | 0.1182 | 0.1198 | 0.1288 | 0.0676 |
| trend_lowvol | bear | 0.1437 | 0.2069 | 0.6948 | 0.5342 | 0.5330 |
| fund_profit_growth | bear | 0.1001 | 0.1506 | 0.6647 | 0.5068 | 0.5008 |
| rsi_vol_combo | bear | 0.1214 | 0.2014 | 0.6025 | 0.4521 | 0.4374 |
| momentum_reversal | bear | 0.1431 | 0.2489 | 0.5747 | 0.4247 | 0.4094 |
| mom_x_lowvol_20_20 | bear | 0.1379 | 0.2545 | 0.5420 | 0.4795 | 0.4009 |
| fund_gross_margin | bear | 0.0559 | 0.1097 | 0.5099 | 0.2603 | 0.3213 |
| fund_score | bear | 0.0789 | 0.1839 | 0.4293 | 0.3699 | 0.2940 |
| turnover_stability | bear | 0.0448 | 0.1238 | 0.3619 | 0.2877 | 0.2330 |
| fund_revenue_growth | bear | 0.0326 | 0.1129 | 0.2890 | 0.2877 | 0.1860 |
| fund_roe | bear | 0.0468 | 0.2270 | 0.2061 | 0.1781 | 0.1214 |

### 央视50_

- **Neutral**: ['fund_profit_growth', 'fund_pe'] (单因子IC=0.0701, 组合IC=0.0796)
  - weights: [0.5118, 0.4882]
- **Bull**: ['low_downside', 'relative_strength'] (单因子IC=0.0552, 组合IC=0.0636)
  - bull_weights: [0.5734, 0.4266]
- **Bear**: ['trend_lowvol', 'bb_width_20'] (单因子IC=0.1563, 组合IC=0.1841)
  - bear_weights: [0.5299, 0.4701]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| fund_profit_growth | neutral | 0.0524 | 0.1823 | 0.2874 | 0.2797 | 0.1839 |
| fund_pe | neutral | 0.0878 | 0.3063 | 0.2867 | 0.2234 | 0.1754 |
| fund_score | neutral | 0.0482 | 0.2048 | 0.2352 | 0.1754 | 0.1382 |
| trend_lowvol | neutral | 0.0541 | 0.2480 | 0.2183 | 0.1608 | 0.1267 |
| volatility | neutral | 0.0419 | 0.2009 | 0.2084 | 0.1691 | 0.1218 |
| fund_pb | neutral | 0.0540 | 0.2671 | 0.2020 | 0.1712 | 0.1183 |
| fund_revenue_growth | neutral | 0.0330 | 0.1814 | 0.1819 | 0.1942 | 0.1086 |
| fund_roe | neutral | 0.0414 | 0.2384 | 0.1737 | 0.1106 | 0.0965 |
| low_downside | neutral | 0.0392 | 0.2429 | 0.1612 | 0.1232 | 0.0905 |
| top_fractal_volume | neutral | 0.0250 | 0.1537 | 0.1625 | 0.1031 | 0.0896 |
| stroke_phase | neutral | 0.0275 | 0.1904 | 0.1442 | 0.1169 | 0.0805 |
| low_downside | bull | 0.0601 | 0.2201 | 0.2731 | 0.2273 | 0.1676 |
| relative_strength | bull | 0.0503 | 0.2386 | 0.2110 | 0.1818 | 0.1247 |
| stroke_phase | bull | 0.0334 | 0.1817 | 0.1839 | 0.2045 | 0.1107 |
| ema20_slope | bull | 0.0481 | 0.2573 | 0.1871 | 0.1591 | 0.1084 |
| ma_alignment | bull | 0.0427 | 0.2494 | 0.1712 | 0.1894 | 0.1018 |
| turnover_stability | bull | 0.0329 | 0.1957 | 0.1681 | 0.1364 | 0.0955 |
| volatility | bull | 0.0278 | 0.2103 | 0.1321 | 0.1212 | 0.0741 |
| exhaustion_risk | bull | 0.0189 | 0.1746 | 0.1081 | 0.1183 | 0.0605 |
| trend_lowvol | bear | 0.1827 | 0.2856 | 0.6398 | 0.5068 | 0.4820 |
| bb_width_20 | bear | 0.1298 | 0.2204 | 0.5889 | 0.4521 | 0.4276 |
| momentum_reversal | bear | 0.1500 | 0.2748 | 0.5460 | 0.5342 | 0.4188 |
| vol_confirm | bear | 0.0776 | 0.2183 | 0.3557 | 0.2877 | 0.2290 |
| mom_x_lowvol_20_20 | bear | 0.0944 | 0.2598 | 0.3633 | 0.1507 | 0.2090 |
| rsi_vol_combo | bear | 0.0698 | 0.2895 | 0.2412 | 0.3699 | 0.1652 |
| turnover_stability | bear | 0.0315 | 0.1613 | 0.1952 | 0.2603 | 0.1230 |

### 婴童概念

- **Neutral**: ['fund_pb', 'mom_x_lowvol_20_20'] (单因子IC=0.0825, 组合IC=0.1103)
  - weights: [0.6544, 0.3456]
- **Bull**: ['volatility', 'fund_pb'] (单因子IC=0.1063, 组合IC=0.1211)
  - bull_weights: [0.5281, 0.4719]
- **Bear**: ['momentum_reversal', 'mom_x_lowvol_20_20', 'turnover_stability'] (单因子IC=0.0897, 组合IC=0.1268)
  - bear_weights: [0.3542, 0.3365, 0.3093]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| fund_pb | neutral | 0.0961 | 0.1364 | 0.7042 | 0.5282 | 0.5381 |
| mom_x_lowvol_20_20 | neutral | 0.0689 | 0.1610 | 0.4281 | 0.3278 | 0.2842 |
| volatility | neutral | 0.0711 | 0.1847 | 0.3852 | 0.3340 | 0.2569 |
| trend_lowvol | neutral | 0.0746 | 0.1840 | 0.4056 | 0.2568 | 0.2548 |
| momentum_reversal | neutral | 0.0660 | 0.1676 | 0.3939 | 0.2714 | 0.2504 |
| fund_pe | neutral | 0.0601 | 0.1799 | 0.3343 | 0.2881 | 0.2153 |
| low_downside | neutral | 0.0419 | 0.1740 | 0.2406 | 0.2651 | 0.1522 |
| rsi_vol_combo | neutral | 0.0428 | 0.1674 | 0.2554 | 0.1775 | 0.1503 |
| turnover_stability | neutral | 0.0235 | 0.1104 | 0.2124 | 0.1587 | 0.1231 |
| fund_profit_growth | neutral | 0.0301 | 0.1466 | 0.2053 | 0.1795 | 0.1211 |
| fund_score | neutral | 0.0219 | 0.1787 | 0.1223 | 0.1065 | 0.0677 |
| volatility | bull | 0.1119 | 0.1521 | 0.7354 | 0.5833 | 0.5822 |
| fund_pb | bull | 0.1008 | 0.1438 | 0.7008 | 0.4848 | 0.5203 |
| low_downside | bull | 0.0837 | 0.1332 | 0.6286 | 0.5000 | 0.4715 |
| momentum_reversal | bull | 0.0660 | 0.1320 | 0.5000 | 0.3561 | 0.3390 |
| turnover_stability | bull | 0.0548 | 0.1150 | 0.4768 | 0.3409 | 0.3197 |
| rsi_vol_combo | bull | 0.0581 | 0.1253 | 0.4635 | 0.2803 | 0.2967 |
| mom_x_lowvol_20_20 | bull | 0.0529 | 0.1355 | 0.3904 | 0.3106 | 0.2558 |
| trend_lowvol | bull | 0.0519 | 0.1679 | 0.3093 | 0.2955 | 0.2003 |
| fund_pe | bull | 0.0462 | 0.1718 | 0.2689 | 0.2197 | 0.1640 |
| momentum_reversal | bear | 0.1115 | 0.2093 | 0.5327 | 0.3425 | 0.3576 |
| mom_x_lowvol_20_20 | bear | 0.1022 | 0.2102 | 0.4862 | 0.3973 | 0.3397 |
| turnover_stability | bear | 0.0552 | 0.1261 | 0.4382 | 0.4247 | 0.3122 |
| trend_lowvol | bear | 0.0825 | 0.1743 | 0.4733 | 0.2055 | 0.2853 |
| fund_revenue_growth | bear | 0.0582 | 0.1348 | 0.4318 | 0.3151 | 0.2839 |
| rsi_vol_combo | bear | 0.0833 | 0.1814 | 0.4593 | 0.1507 | 0.2643 |
| fund_pb | bear | 0.0367 | 0.1052 | 0.3493 | 0.3425 | 0.2344 |
| fund_score | bear | 0.0443 | 0.1558 | 0.2845 | 0.3151 | 0.1870 |

### 存储芯片

- **Neutral**: ['trend_lowvol', 'fund_pe', 'momentum_reversal'] (单因子IC=0.0684, 组合IC=0.0852)
  - weights: [0.3691, 0.3299, 0.3009]
- **Bull**: ['volatility'] (单因子IC=0.097, 组合IC=0.097)
  - bull_weights: [1.0]
- **Bear**: ['momentum_reversal'] (单因子IC=0.1645, 组合IC=0.1645)
  - bear_weights: [1.0]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| trend_lowvol | neutral | 0.0810 | 0.2098 | 0.3862 | 0.3173 | 0.2543 |
| fund_pe | neutral | 0.0588 | 0.1643 | 0.3582 | 0.2693 | 0.2273 |
| momentum_reversal | neutral | 0.0653 | 0.1953 | 0.3344 | 0.2401 | 0.2074 |
| rsi_vol_combo | neutral | 0.0596 | 0.1892 | 0.3150 | 0.2338 | 0.1943 |
| mom_x_lowvol_20_20 | neutral | 0.0580 | 0.1944 | 0.2985 | 0.2526 | 0.1870 |
| fund_pb | neutral | 0.0575 | 0.1955 | 0.2940 | 0.2338 | 0.1814 |
| volatility | neutral | 0.0551 | 0.1928 | 0.2859 | 0.2213 | 0.1746 |
| fund_profit_growth | neutral | 0.0410 | 0.1772 | 0.2313 | 0.2109 | 0.1401 |
| low_downside | neutral | 0.0362 | 0.1886 | 0.1920 | 0.1733 | 0.1126 |
| turnover_stability | neutral | 0.0258 | 0.1496 | 0.1728 | 0.1336 | 0.0979 |
| fund_score | neutral | 0.0327 | 0.2179 | 0.1502 | 0.1691 | 0.0878 |
| volatility | bull | 0.0970 | 0.1692 | 0.5730 | 0.4394 | 0.4124 |
| low_downside | bull | 0.0774 | 0.1717 | 0.4510 | 0.3182 | 0.2972 |
| fund_pe | bull | 0.0622 | 0.1450 | 0.4288 | 0.3333 | 0.2858 |
| trend_lowvol | bull | 0.0779 | 0.1920 | 0.4055 | 0.3030 | 0.2642 |
| fund_pb | bull | 0.0715 | 0.1889 | 0.3787 | 0.3561 | 0.2567 |
| turnover_stability | bull | 0.0484 | 0.1269 | 0.3812 | 0.2500 | 0.2383 |
| mom_x_lowvol_20_20 | bull | 0.0559 | 0.1596 | 0.3502 | 0.2576 | 0.2202 |
| momentum_reversal | bull | 0.0485 | 0.1705 | 0.2846 | 0.2273 | 0.1746 |
| rsi_vol_combo | bull | 0.0321 | 0.1881 | 0.1705 | 0.1667 | 0.0994 |
| fund_roe | bull | 0.0289 | 0.1818 | 0.1591 | 0.1061 | 0.0880 |
| fund_gross_margin | bull | 0.0219 | 0.1661 | 0.1318 | 0.1515 | 0.0759 |
| stroke_phase | bull | 0.0130 | 0.1267 | 0.1022 | 0.1174 | 0.0571 |
| momentum_reversal | bear | 0.1645 | 0.1931 | 0.8522 | 0.6164 | 0.6887 |
| rsi_vol_combo | bear | 0.1259 | 0.1727 | 0.7289 | 0.5616 | 0.5691 |
| mom_x_lowvol_20_20 | bear | 0.1485 | 0.2018 | 0.7361 | 0.4521 | 0.5344 |
| bb_width_20 | bear | 0.1035 | 0.1659 | 0.6241 | 0.5890 | 0.4958 |
| trend_lowvol | bear | 0.0683 | 0.1937 | 0.3524 | 0.1507 | 0.2027 |
| fund_pe | bear | 0.0435 | 0.1403 | 0.3099 | 0.1781 | 0.1826 |
| fund_profit_growth | bear | 0.0348 | 0.1918 | 0.1816 | 0.1233 | 0.1020 |

### 宁组合

- **Neutral**: ['fund_profit_growth', 'fund_revenue_growth', 'fund_pe'] (单因子IC=0.0552, 组合IC=0.0692)
  - weights: [0.3658, 0.3296, 0.3045]
- **Bull**: ['fund_pb', 'stroke_phase'] (单因子IC=0.0387, 组合IC=0.0552)
  - bull_weights: [0.5731, 0.4269]
- **Bear**: ['momentum_reversal', 'rsi_vol_combo', 'trend_lowvol'] (单因子IC=0.1869, 组合IC=0.2027)
  - bear_weights: [0.3634, 0.3229, 0.3137]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| fund_profit_growth | neutral | 0.0594 | 0.2626 | 0.2261 | 0.1851 | 0.1340 |
| fund_revenue_growth | neutral | 0.0563 | 0.2694 | 0.2091 | 0.1548 | 0.1207 |
| fund_pe | neutral | 0.0500 | 0.2564 | 0.1952 | 0.1430 | 0.1115 |
| fund_score | neutral | 0.0462 | 0.2530 | 0.1826 | 0.1743 | 0.1072 |
| fund_pb | bull | 0.0445 | 0.2533 | 0.1756 | 0.1515 | 0.1011 |
| stroke_phase | bull | 0.0328 | 0.2462 | 0.1334 | 0.1288 | 0.0753 |
| momentum_reversal | bear | 0.1982 | 0.2969 | 0.6675 | 0.5342 | 0.5121 |
| rsi_vol_combo | bear | 0.1678 | 0.2778 | 0.6039 | 0.5068 | 0.4550 |
| trend_lowvol | bear | 0.1947 | 0.3198 | 0.6088 | 0.4521 | 0.4420 |
| mom_x_lowvol_20_20 | bear | 0.1208 | 0.2537 | 0.4761 | 0.4795 | 0.3522 |
| bb_width_20 | bear | 0.1331 | 0.3108 | 0.4283 | 0.4247 | 0.3051 |
| vol_opening_strength | bear | 0.0839 | 0.2220 | 0.3777 | 0.4054 | 0.2654 |
| fund_pe | bear | 0.0699 | 0.1869 | 0.3737 | 0.3151 | 0.2457 |
| fund_revenue_growth | bear | 0.0816 | 0.2486 | 0.3281 | 0.4231 | 0.2334 |
| fund_pb | bear | 0.0908 | 0.2551 | 0.3559 | 0.2329 | 0.2194 |
| vol_opening_confirm | bear | 0.0756 | 0.2237 | 0.3381 | 0.2432 | 0.2102 |
| wash_sale_score | bear | 0.0589 | 0.1774 | 0.3320 | 0.2500 | 0.2075 |

### 安防概念

- **Neutral**: ['fund_pb', 'momentum_reversal', 'trend_lowvol'] (单因子IC=0.0784, 组合IC=0.1086)
  - weights: [0.3963, 0.3039, 0.2998]
- **Bull**: ['fund_pb', 'volatility'] (单因子IC=0.123, 组合IC=0.1428)
  - bull_weights: [0.5826, 0.4174]
- **Bear**: ['trend_lowvol', 'turnover_stability', 'mom_x_lowvol_20_20'] (单因子IC=0.1077, 组合IC=0.1372)
  - bear_weights: [0.3638, 0.3392, 0.297]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| fund_pb | neutral | 0.0690 | 0.1211 | 0.5697 | 0.4259 | 0.4062 |
| momentum_reversal | neutral | 0.0801 | 0.1752 | 0.4569 | 0.3633 | 0.3114 |
| trend_lowvol | neutral | 0.0861 | 0.1897 | 0.4536 | 0.3549 | 0.3073 |
| mom_x_lowvol_20_20 | neutral | 0.0769 | 0.1711 | 0.4492 | 0.3528 | 0.3039 |
| rsi_vol_combo | neutral | 0.0642 | 0.1611 | 0.3986 | 0.3246 | 0.2640 |
| volatility | neutral | 0.0686 | 0.1774 | 0.3867 | 0.3486 | 0.2608 |
| fund_profit_growth | neutral | 0.0409 | 0.1414 | 0.2896 | 0.1754 | 0.1702 |
| low_downside | neutral | 0.0428 | 0.1805 | 0.2368 | 0.1983 | 0.1419 |
| fund_score | neutral | 0.0325 | 0.1676 | 0.1938 | 0.1587 | 0.1123 |
| turnover_stability | neutral | 0.0225 | 0.1228 | 0.1831 | 0.1461 | 0.1049 |
| fund_pe | neutral | 0.0302 | 0.1700 | 0.1779 | 0.1628 | 0.1034 |
| fund_roe | neutral | 0.0298 | 0.1723 | 0.1730 | 0.1409 | 0.0987 |
| fund_revenue_growth | neutral | 0.0151 | 0.1419 | 0.1063 | 0.1086 | 0.0589 |
| fund_pb | bull | 0.1207 | 0.1209 | 0.9981 | 0.6818 | 0.8393 |
| volatility | bull | 0.1252 | 0.1617 | 0.7744 | 0.5530 | 0.6013 |
| low_downside | bull | 0.0927 | 0.1348 | 0.6878 | 0.5303 | 0.5263 |
| trend_lowvol | bull | 0.0995 | 0.1697 | 0.5864 | 0.3788 | 0.4042 |
| mom_x_lowvol_20_20 | bull | 0.0747 | 0.1776 | 0.4208 | 0.2727 | 0.2678 |
| momentum_reversal | bull | 0.0703 | 0.1709 | 0.4117 | 0.2045 | 0.2479 |
| turnover_stability | bull | 0.0463 | 0.1247 | 0.3715 | 0.2500 | 0.2322 |
| fund_pe | bull | 0.0537 | 0.1573 | 0.3414 | 0.2955 | 0.2211 |
| trend_lowvol | bear | 0.1293 | 0.1658 | 0.7802 | 0.5890 | 0.6199 |
| turnover_stability | bear | 0.0870 | 0.1196 | 0.7273 | 0.5890 | 0.5779 |
| mom_x_lowvol_20_20 | bear | 0.1068 | 0.1532 | 0.6971 | 0.4521 | 0.5061 |
| momentum_reversal | bear | 0.1009 | 0.1564 | 0.6450 | 0.4521 | 0.4683 |
| fund_pb | bear | 0.0728 | 0.1529 | 0.4758 | 0.3973 | 0.3324 |
| rsi_vol_combo | bear | 0.0782 | 0.1578 | 0.4951 | 0.1781 | 0.2917 |
| fund_gross_margin | bear | 0.0407 | 0.1287 | 0.3161 | 0.1781 | 0.1862 |
| fund_revenue_growth | bear | 0.0418 | 0.1433 | 0.2919 | 0.2055 | 0.1759 |
| fund_pe | bear | 0.0471 | 0.1745 | 0.2696 | 0.1781 | 0.1588 |
| volatility | bear | 0.0441 | 0.1884 | 0.2340 | 0.2329 | 0.1442 |

### 宠物经济

- **Neutral**: ['fund_pb', 'trend_lowvol'] (单因子IC=0.0804, 组合IC=0.1086)
  - weights: [0.654, 0.346]
- **Bull**: ['volatility', 'fund_pe', 'fund_pb'] (单因子IC=0.0856, 组合IC=0.1195)
  - bull_weights: [0.4078, 0.3141, 0.2782]
- **Bear**: ['momentum_reversal', 'mom_x_lowvol_20_20', 'fund_pb'] (单因子IC=0.1155, 组合IC=0.1427)
  - bear_weights: [0.3795, 0.3213, 0.2992]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| fund_pb | neutral | 0.0847 | 0.1586 | 0.5344 | 0.4217 | 0.3799 |
| trend_lowvol | neutral | 0.0761 | 0.2310 | 0.3296 | 0.2192 | 0.2009 |
| volatility | neutral | 0.0645 | 0.2089 | 0.3088 | 0.2923 | 0.1995 |
| mom_x_lowvol_20_20 | neutral | 0.0650 | 0.2153 | 0.3020 | 0.1754 | 0.1775 |
| momentum_reversal | neutral | 0.0600 | 0.2154 | 0.2786 | 0.2088 | 0.1684 |
| low_downside | neutral | 0.0472 | 0.2015 | 0.2345 | 0.2818 | 0.1503 |
| rsi_vol_combo | neutral | 0.0442 | 0.1972 | 0.2241 | 0.1795 | 0.1322 |
| turnover_stability | neutral | 0.0281 | 0.1567 | 0.1793 | 0.1827 | 0.1061 |
| fund_pe | neutral | 0.0287 | 0.1886 | 0.1523 | 0.1608 | 0.0884 |
| volatility | bull | 0.1051 | 0.1806 | 0.5818 | 0.5303 | 0.4452 |
| fund_pe | bull | 0.0746 | 0.1524 | 0.4893 | 0.4015 | 0.3429 |
| fund_pb | bull | 0.0773 | 0.1765 | 0.4381 | 0.3864 | 0.3037 |
| low_downside | bull | 0.0754 | 0.1766 | 0.4271 | 0.3712 | 0.2928 |
| momentum_reversal | bull | 0.0454 | 0.1582 | 0.2867 | 0.1780 | 0.1689 |
| trend_lowvol | bull | 0.0487 | 0.1737 | 0.2806 | 0.1970 | 0.1679 |
| rsi_vol_combo | bull | 0.0386 | 0.1458 | 0.2649 | 0.2197 | 0.1616 |
| turnover_stability | bull | 0.0323 | 0.1202 | 0.2687 | 0.1515 | 0.1547 |
| mom_x_lowvol_20_20 | bull | 0.0385 | 0.1645 | 0.2338 | 0.1894 | 0.1390 |
| momentum_reversal | bear | 0.1218 | 0.1863 | 0.6540 | 0.4247 | 0.4659 |
| mom_x_lowvol_20_20 | bear | 0.1216 | 0.2111 | 0.5760 | 0.3699 | 0.3945 |
| fund_pb | bear | 0.1031 | 0.2039 | 0.5059 | 0.4521 | 0.3673 |
| trend_lowvol | bear | 0.1262 | 0.2593 | 0.4866 | 0.4795 | 0.3599 |
| rsi_vol_combo | bear | 0.0717 | 0.1904 | 0.3765 | 0.3425 | 0.2527 |
| fund_revenue_growth | bear | 0.0411 | 0.1796 | 0.2289 | 0.2603 | 0.1442 |
| volatility | bear | 0.0441 | 0.2453 | 0.1800 | 0.1781 | 0.1060 |

### 小米概念

- **Neutral**: ['fund_pb', 'momentum_reversal', 'mom_x_lowvol_20_20'] (单因子IC=0.0689, 组合IC=0.0931)
  - weights: [0.3628, 0.3227, 0.3145]
- **Bull**: ['volatility', 'trend_lowvol', 'fund_pb'] (单因子IC=0.092, 组合IC=0.1162)
  - bull_weights: [0.3622, 0.3399, 0.2979]
- **Bear**: ['momentum_reversal'] (单因子IC=0.0644, 组合IC=0.0644)
  - bear_weights: [1.0]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| fund_pb | neutral | 0.0697 | 0.1484 | 0.4698 | 0.3424 | 0.3153 |
| momentum_reversal | neutral | 0.0697 | 0.1631 | 0.4271 | 0.3132 | 0.2804 |
| mom_x_lowvol_20_20 | neutral | 0.0672 | 0.1618 | 0.4156 | 0.3152 | 0.2733 |
| trend_lowvol | neutral | 0.0753 | 0.1777 | 0.4236 | 0.2735 | 0.2697 |
| volatility | neutral | 0.0651 | 0.1832 | 0.3555 | 0.2881 | 0.2290 |
| rsi_vol_combo | neutral | 0.0515 | 0.1509 | 0.3410 | 0.2693 | 0.2164 |
| fund_profit_growth | neutral | 0.0376 | 0.1250 | 0.3007 | 0.2505 | 0.1880 |
| fund_pe | neutral | 0.0489 | 0.1554 | 0.3144 | 0.1942 | 0.1877 |
| low_downside | neutral | 0.0392 | 0.1703 | 0.2304 | 0.1837 | 0.1364 |
| fund_score | neutral | 0.0332 | 0.1612 | 0.2062 | 0.1482 | 0.1184 |
| turnover_stability | neutral | 0.0115 | 0.1056 | 0.1088 | 0.1169 | 0.0608 |
| volatility | bull | 0.0983 | 0.1721 | 0.5713 | 0.4318 | 0.4090 |
| trend_lowvol | bull | 0.0968 | 0.1776 | 0.5449 | 0.4091 | 0.3839 |
| fund_pb | bull | 0.0808 | 0.1656 | 0.4881 | 0.3788 | 0.3365 |
| low_downside | bull | 0.0760 | 0.1590 | 0.4777 | 0.3485 | 0.3221 |
| turnover_stability | bull | 0.0316 | 0.0747 | 0.4234 | 0.2955 | 0.2743 |
| mom_x_lowvol_20_20 | bull | 0.0657 | 0.1624 | 0.4044 | 0.2652 | 0.2558 |
| momentum_reversal | bull | 0.0576 | 0.1648 | 0.3494 | 0.2500 | 0.2184 |
| fund_pe | bull | 0.0456 | 0.1636 | 0.2784 | 0.2273 | 0.1708 |
| stroke_phase | bull | 0.0265 | 0.0995 | 0.2658 | 0.1288 | 0.1500 |
| fund_profit_growth | bull | 0.0199 | 0.1161 | 0.1713 | 0.1667 | 0.0999 |
| momentum_reversal | bear | 0.0644 | 0.2229 | 0.2888 | 0.3699 | 0.1978 |
| rsi_vol_combo | bear | 0.0470 | 0.1488 | 0.3157 | 0.1781 | 0.1860 |
| mom_x_lowvol_20_20 | bear | 0.0508 | 0.2227 | 0.2279 | 0.2603 | 0.1436 |
| wash_sale_score | bear | 0.0268 | 0.1211 | 0.2217 | 0.1000 | 0.1220 |

### 小米汽车

- **Neutral**: ['trend_lowvol', 'fund_pb', 'momentum_reversal'] (单因子IC=0.0792, 组合IC=0.1064)
  - weights: [0.3722, 0.3305, 0.2973]
- **Bull**: ['trend_lowvol', 'volatility', 'fund_pb'] (单因子IC=0.084, 组合IC=0.1023)
  - bull_weights: [0.3749, 0.3588, 0.2663]
- **Bear**: ['fund_gross_margin', 'fund_profit_growth', 'momentum_reversal'] (单因子IC=0.0423, 组合IC=0.0729)
  - bear_weights: [0.3614, 0.3324, 0.3062]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| trend_lowvol | neutral | 0.0955 | 0.2002 | 0.4771 | 0.3424 | 0.3202 |
| fund_pb | neutral | 0.0685 | 0.1615 | 0.4243 | 0.3403 | 0.2843 |
| momentum_reversal | neutral | 0.0734 | 0.1911 | 0.3841 | 0.3319 | 0.2558 |
| mom_x_lowvol_20_20 | neutral | 0.0658 | 0.1887 | 0.3488 | 0.2923 | 0.2254 |
| fund_profit_growth | neutral | 0.0483 | 0.1410 | 0.3426 | 0.2756 | 0.2185 |
| fund_pe | neutral | 0.0664 | 0.2045 | 0.3247 | 0.2505 | 0.2030 |
| fund_score | neutral | 0.0507 | 0.1656 | 0.3063 | 0.2276 | 0.1880 |
| rsi_vol_combo | neutral | 0.0517 | 0.1773 | 0.2916 | 0.2286 | 0.1791 |
| volatility | neutral | 0.0555 | 0.2178 | 0.2546 | 0.1921 | 0.1518 |
| fund_roe | neutral | 0.0449 | 0.1768 | 0.2541 | 0.1587 | 0.1472 |
| low_downside | neutral | 0.0342 | 0.1951 | 0.1751 | 0.1357 | 0.0994 |
| turnover_stability | neutral | 0.0171 | 0.1351 | 0.1265 | 0.1086 | 0.0701 |
| fund_gross_margin | neutral | 0.0097 | 0.1339 | 0.0725 | 0.1086 | 0.0402 |
| trend_lowvol | bull | 0.0937 | 0.1900 | 0.4929 | 0.3258 | 0.3268 |
| volatility | bull | 0.0913 | 0.1980 | 0.4612 | 0.3561 | 0.3127 |
| fund_pb | bull | 0.0669 | 0.1768 | 0.3782 | 0.2273 | 0.2321 |
| turnover_stability | bull | 0.0369 | 0.1129 | 0.3265 | 0.3258 | 0.2164 |
| low_downside | bull | 0.0589 | 0.1711 | 0.3444 | 0.2500 | 0.2153 |
| fund_pe | bull | 0.0525 | 0.1777 | 0.2957 | 0.2955 | 0.1915 |
| mom_x_lowvol_20_20 | bull | 0.0334 | 0.1905 | 0.1754 | 0.1591 | 0.1017 |
| momentum_reversal | bull | 0.0305 | 0.1887 | 0.1613 | 0.2197 | 0.0984 |
| stroke_phase | bull | 0.0205 | 0.1343 | 0.1524 | 0.1591 | 0.0883 |
| fund_gross_margin | bear | 0.0386 | 0.1720 | 0.2245 | 0.2329 | 0.1384 |
| fund_profit_growth | bear | 0.0346 | 0.1601 | 0.2161 | 0.1781 | 0.1273 |
| momentum_reversal | bear | 0.0536 | 0.2692 | 0.1991 | 0.1781 | 0.1173 |

### 小红书概念

- **Neutral**: ['volatility', 'momentum_reversal', 'fund_pb'] (单因子IC=0.0997, 组合IC=0.1398)
  - weights: [0.3534, 0.3282, 0.3184]
- **Bull**: ['volatility', 'fund_pb', 'low_downside'] (单因子IC=0.0926, 组合IC=0.1111)
  - bull_weights: [0.3816, 0.3208, 0.2976]
- **Bear**: ['momentum_reversal', 'mom_x_lowvol_20_20', 'trend_lowvol'] (单因子IC=0.151, 组合IC=0.1716)
  - bear_weights: [0.3797, 0.3346, 0.2857]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| volatility | neutral | 0.1115 | 0.2436 | 0.4576 | 0.3591 | 0.3110 |
| momentum_reversal | neutral | 0.0974 | 0.2246 | 0.4337 | 0.3319 | 0.2888 |
| fund_pb | neutral | 0.0902 | 0.2144 | 0.4208 | 0.3319 | 0.2802 |
| mom_x_lowvol_20_20 | neutral | 0.0947 | 0.2257 | 0.4194 | 0.3090 | 0.2745 |
| trend_lowvol | neutral | 0.0903 | 0.2243 | 0.4025 | 0.3319 | 0.2681 |
| rsi_vol_combo | neutral | 0.0817 | 0.2074 | 0.3938 | 0.3194 | 0.2598 |
| low_downside | neutral | 0.0786 | 0.2283 | 0.3445 | 0.2860 | 0.2215 |
| turnover_stability | neutral | 0.0377 | 0.1524 | 0.2474 | 0.2067 | 0.1492 |
| fund_profit_growth | neutral | 0.0342 | 0.1579 | 0.2168 | 0.1837 | 0.1283 |
| fund_pe | neutral | 0.0421 | 0.2507 | 0.1679 | 0.1315 | 0.0950 |
| volatility | bull | 0.1033 | 0.2088 | 0.4948 | 0.4242 | 0.3523 |
| fund_pb | bull | 0.0929 | 0.2139 | 0.4345 | 0.3636 | 0.2962 |
| low_downside | bull | 0.0817 | 0.1891 | 0.4318 | 0.2727 | 0.2748 |
| turnover_stability | bull | 0.0382 | 0.1329 | 0.2873 | 0.1780 | 0.1692 |
| rsi_vol_combo | bull | 0.0490 | 0.1847 | 0.2654 | 0.2197 | 0.1618 |
| momentum_reversal | bull | 0.0432 | 0.1900 | 0.2274 | 0.2652 | 0.1438 |
| trend_lowvol | bull | 0.0479 | 0.2310 | 0.2074 | 0.2500 | 0.1296 |
| fund_pe | bull | 0.0567 | 0.2574 | 0.2204 | 0.1212 | 0.1235 |
| mom_x_lowvol_20_20 | bull | 0.0293 | 0.1916 | 0.1532 | 0.1894 | 0.0911 |
| momentum_reversal | bear | 0.1628 | 0.1808 | 0.9005 | 0.6164 | 0.7278 |
| mom_x_lowvol_20_20 | bear | 0.1533 | 0.1932 | 0.7935 | 0.6164 | 0.6414 |
| trend_lowvol | bear | 0.1369 | 0.1918 | 0.7138 | 0.5342 | 0.5475 |
| rsi_vol_combo | bear | 0.0935 | 0.1496 | 0.6250 | 0.4247 | 0.4452 |
| bb_width_20 | bear | 0.0728 | 0.2038 | 0.3570 | 0.2055 | 0.2152 |
| turnover_stability | bear | 0.0305 | 0.1375 | 0.2216 | 0.2055 | 0.1336 |

### 小金属概念

- **Neutral**: ['volatility', 'momentum_reversal'] (单因子IC=0.0594, 组合IC=0.0687)
  - weights: [0.5161, 0.4839]
- **Bull**: ['volatility', 'low_downside', 'momentum_reversal'] (单因子IC=0.0698, 组合IC=0.0857)
  - bull_weights: [0.4282, 0.3472, 0.2246]
- **Bear**: ['mom_x_lowvol_20_20', 'momentum_reversal', 'bb_width_20'] (单因子IC=0.0731, 组合IC=0.0855)
  - bear_weights: [0.433, 0.3146, 0.2524]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| volatility | neutral | 0.0592 | 0.1887 | 0.3138 | 0.2109 | 0.1900 |
| momentum_reversal | neutral | 0.0596 | 0.2000 | 0.2978 | 0.1962 | 0.1781 |
| mom_x_lowvol_20_20 | neutral | 0.0544 | 0.1846 | 0.2945 | 0.2046 | 0.1774 |
| fund_pb | neutral | 0.0551 | 0.1870 | 0.2947 | 0.1942 | 0.1760 |
| trend_lowvol | neutral | 0.0632 | 0.2206 | 0.2863 | 0.2004 | 0.1719 |
| turnover_stability | neutral | 0.0384 | 0.1516 | 0.2534 | 0.2129 | 0.1537 |
| fund_pe | neutral | 0.0468 | 0.1838 | 0.2549 | 0.1733 | 0.1495 |
| fund_profit_growth | neutral | 0.0320 | 0.1511 | 0.2118 | 0.1273 | 0.1194 |
| fund_revenue_growth | neutral | 0.0221 | 0.1468 | 0.1502 | 0.1315 | 0.0850 |
| volatility | bull | 0.0772 | 0.1588 | 0.4864 | 0.4621 | 0.3556 |
| low_downside | bull | 0.0726 | 0.1649 | 0.4400 | 0.3106 | 0.2884 |
| momentum_reversal | bull | 0.0596 | 0.1925 | 0.3097 | 0.2045 | 0.1865 |
| trend_lowvol | bull | 0.0541 | 0.1935 | 0.2794 | 0.2121 | 0.1693 |
| turnover_stability | bull | 0.0386 | 0.1548 | 0.2495 | 0.2424 | 0.1550 |
| mom_x_lowvol_20_20 | bull | 0.0444 | 0.1724 | 0.2574 | 0.1288 | 0.1453 |
| fund_pe | bull | 0.0375 | 0.1936 | 0.1936 | 0.1970 | 0.1159 |
| fund_gross_margin | bull | 0.0154 | 0.1242 | 0.1240 | 0.1136 | 0.0691 |
| mom_x_lowvol_20_20 | bear | 0.0834 | 0.1946 | 0.4284 | 0.3151 | 0.2817 |
| momentum_reversal | bear | 0.0754 | 0.2272 | 0.3320 | 0.2329 | 0.2047 |
| bb_width_20 | bear | 0.0604 | 0.2067 | 0.2923 | 0.1233 | 0.1642 |
| turnover_stability | bear | 0.0240 | 0.1271 | 0.1887 | 0.1233 | 0.1060 |

### 尾气治理

- **Neutral**: ['mom_x_lowvol_20_20', 'momentum_reversal', 'fund_pb'] (单因子IC=0.0765, 组合IC=0.0953)
  - weights: [0.3472, 0.3337, 0.3191]
- **Bull**: ['fund_pe'] (单因子IC=0.1039, 组合IC=0.1044)
  - bull_weights: [1.0]
- **Bear**: ['momentum_reversal', 'fund_profit_growth', 'mom_x_lowvol_20_20'] (单因子IC=0.0873, 组合IC=0.1098)
  - bear_weights: [0.3514, 0.3471, 0.3015]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| mom_x_lowvol_20_20 | neutral | 0.0795 | 0.2198 | 0.3616 | 0.3079 | 0.2365 |
| momentum_reversal | neutral | 0.0771 | 0.2211 | 0.3489 | 0.3027 | 0.2273 |
| fund_pb | neutral | 0.0729 | 0.2124 | 0.3433 | 0.2662 | 0.2174 |
| fund_pe | neutral | 0.0800 | 0.2372 | 0.3371 | 0.2589 | 0.2122 |
| fund_profit_growth | neutral | 0.0630 | 0.2007 | 0.3139 | 0.2328 | 0.1935 |
| volatility | neutral | 0.0691 | 0.2327 | 0.2971 | 0.2495 | 0.1856 |
| trend_lowvol | neutral | 0.0694 | 0.2485 | 0.2793 | 0.1962 | 0.1671 |
| low_downside | neutral | 0.0563 | 0.2357 | 0.2387 | 0.2505 | 0.1493 |
| rsi_vol_combo | neutral | 0.0457 | 0.2061 | 0.2218 | 0.1879 | 0.1317 |
| fund_score | neutral | 0.0461 | 0.2204 | 0.2090 | 0.1336 | 0.1185 |
| turnover_stability | neutral | 0.0324 | 0.1794 | 0.1804 | 0.1962 | 0.1079 |
| fund_revenue_growth | neutral | 0.0238 | 0.1816 | 0.1309 | 0.1106 | 0.0727 |
| fund_pe | bull | 0.1039 | 0.2163 | 0.4804 | 0.4167 | 0.3403 |
| fund_pb | bull | 0.0857 | 0.2078 | 0.4126 | 0.3258 | 0.2735 |
| rsi_vol_combo | bull | 0.0738 | 0.1865 | 0.3957 | 0.3561 | 0.2683 |
| fund_profit_growth | bull | 0.0781 | 0.1966 | 0.3971 | 0.3144 | 0.2610 |
| volatility | bull | 0.0807 | 0.2178 | 0.3706 | 0.2652 | 0.2345 |
| fund_score | bull | 0.0637 | 0.2230 | 0.2857 | 0.2652 | 0.1807 |
| turnover_stability | bull | 0.0514 | 0.1828 | 0.2812 | 0.2500 | 0.1758 |
| momentum_reversal | bull | 0.0509 | 0.1918 | 0.2656 | 0.2500 | 0.1660 |
| fund_gross_margin | bull | 0.0585 | 0.2137 | 0.2737 | 0.1629 | 0.1591 |
| fund_revenue_growth | bull | 0.0441 | 0.1774 | 0.2486 | 0.1742 | 0.1459 |
| mom_x_lowvol_20_20 | bull | 0.0397 | 0.1996 | 0.1987 | 0.1667 | 0.1159 |
| momentum_reversal | bear | 0.0941 | 0.2347 | 0.4007 | 0.3151 | 0.2635 |
| fund_profit_growth | bear | 0.0794 | 0.1881 | 0.4222 | 0.2329 | 0.2602 |
| mom_x_lowvol_20_20 | bear | 0.0885 | 0.2573 | 0.3438 | 0.3151 | 0.2260 |

### 工业互联

- **Neutral**: ['momentum_reversal', 'mom_x_lowvol_20_20', 'trend_lowvol'] (单因子IC=0.0742, 组合IC=0.081)
  - weights: [0.3461, 0.3351, 0.3188]
- **Bull**: ['low_downside', 'fund_pb', 'trend_lowvol'] (单因子IC=0.0923, 组合IC=0.1237)
  - bull_weights: [0.3864, 0.3213, 0.2923]
- **Bear**: ['trend_lowvol', 'momentum_reversal', 'mom_x_lowvol_20_20'] (单因子IC=0.0942, 组合IC=0.0973)
  - bear_weights: [0.3527, 0.3318, 0.3155]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| momentum_reversal | neutral | 0.0736 | 0.1586 | 0.4640 | 0.3194 | 0.3061 |
| mom_x_lowvol_20_20 | neutral | 0.0708 | 0.1584 | 0.4471 | 0.3257 | 0.2964 |
| trend_lowvol | neutral | 0.0781 | 0.1792 | 0.4357 | 0.2944 | 0.2820 |
| fund_profit_growth | neutral | 0.0512 | 0.1207 | 0.4242 | 0.2860 | 0.2727 |
| fund_pb | neutral | 0.0743 | 0.1783 | 0.4167 | 0.2568 | 0.2618 |
| rsi_vol_combo | neutral | 0.0598 | 0.1485 | 0.4025 | 0.2923 | 0.2601 |
| volatility | neutral | 0.0597 | 0.1934 | 0.3087 | 0.2484 | 0.1927 |
| fund_score | neutral | 0.0482 | 0.1546 | 0.3119 | 0.1962 | 0.1866 |
| turnover_stability | neutral | 0.0256 | 0.1005 | 0.2547 | 0.2213 | 0.1555 |
| fund_revenue_growth | neutral | 0.0257 | 0.1088 | 0.2361 | 0.1441 | 0.1351 |
| fund_pe | neutral | 0.0432 | 0.1844 | 0.2346 | 0.1315 | 0.1327 |
| low_downside | neutral | 0.0414 | 0.1893 | 0.2188 | 0.1524 | 0.1261 |
| low_downside | bull | 0.0888 | 0.1431 | 0.6207 | 0.4697 | 0.4561 |
| fund_pb | bull | 0.0985 | 0.1830 | 0.5384 | 0.4091 | 0.3793 |
| trend_lowvol | bull | 0.0894 | 0.1767 | 0.5062 | 0.3636 | 0.3451 |
| volatility | bull | 0.0829 | 0.1667 | 0.4974 | 0.3864 | 0.3448 |
| momentum_reversal | bull | 0.0542 | 0.1546 | 0.3506 | 0.2576 | 0.2205 |
| turnover_stability | bull | 0.0302 | 0.1014 | 0.2981 | 0.3106 | 0.1954 |
| mom_x_lowvol_20_20 | bull | 0.0428 | 0.1506 | 0.2844 | 0.1515 | 0.1637 |
| fund_pe | bull | 0.0500 | 0.1891 | 0.2642 | 0.2045 | 0.1591 |
| rsi_vol_combo | bull | 0.0361 | 0.1325 | 0.2724 | 0.1591 | 0.1578 |
| fund_profit_growth | bull | 0.0255 | 0.1234 | 0.2068 | 0.1667 | 0.1206 |
| fund_roe | bull | 0.0197 | 0.1849 | 0.1065 | 0.1212 | 0.0597 |
| trend_lowvol | bear | 0.0898 | 0.1593 | 0.5636 | 0.3973 | 0.3938 |
| momentum_reversal | bear | 0.0919 | 0.1733 | 0.5303 | 0.3973 | 0.3705 |
| mom_x_lowvol_20_20 | bear | 0.1009 | 0.1884 | 0.5357 | 0.3151 | 0.3522 |
| fund_revenue_growth | bear | 0.0430 | 0.1157 | 0.3713 | 0.1507 | 0.2137 |
| rsi_vol_combo | bear | 0.0486 | 0.1462 | 0.3320 | 0.2603 | 0.2092 |
| turnover_stability | bear | 0.0340 | 0.1115 | 0.3046 | 0.2877 | 0.1961 |

### 工业大麻

- **Neutral**: ['trend_lowvol', 'fund_pb', 'turnover_stability'] (单因子IC=0.0689, 组合IC=0.0973)
  - weights: [0.3966, 0.325, 0.2784]
- **Bull**: ['low_downside', 'volatility', 'turnover_stability'] (单因子IC=0.0772, 组合IC=0.0992)
  - bull_weights: [0.3507, 0.344, 0.3053]
- **Bear**: ['trend_lowvol'] (单因子IC=0.1932, 组合IC=0.1932)
  - bear_weights: [1.0]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| trend_lowvol | neutral | 0.0900 | 0.2513 | 0.3582 | 0.3006 | 0.2330 |
| fund_pb | neutral | 0.0629 | 0.2039 | 0.3084 | 0.2380 | 0.1909 |
| turnover_stability | neutral | 0.0538 | 0.1980 | 0.2716 | 0.2046 | 0.1636 |
| momentum_reversal | neutral | 0.0590 | 0.2470 | 0.2387 | 0.1628 | 0.1388 |
| mom_x_lowvol_20_20 | neutral | 0.0580 | 0.2460 | 0.2360 | 0.1441 | 0.1350 |
| volatility | neutral | 0.0577 | 0.2644 | 0.2180 | 0.2088 | 0.1318 |
| fund_score | neutral | 0.0441 | 0.2201 | 0.2005 | 0.1848 | 0.1187 |
| low_downside | neutral | 0.0468 | 0.2389 | 0.1960 | 0.1441 | 0.1121 |
| fund_profit_growth | neutral | 0.0184 | 0.1900 | 0.0969 | 0.1096 | 0.0537 |
| low_downside | bull | 0.0821 | 0.2326 | 0.3532 | 0.3523 | 0.2388 |
| volatility | bull | 0.0873 | 0.2514 | 0.3473 | 0.3485 | 0.2342 |
| turnover_stability | bull | 0.0622 | 0.1938 | 0.3209 | 0.2955 | 0.2079 |
| fund_pe | bull | 0.0789 | 0.2467 | 0.3196 | 0.1591 | 0.1852 |
| fund_pb | bull | 0.0552 | 0.1915 | 0.2881 | 0.2273 | 0.1768 |
| fund_revenue_growth | bull | 0.0357 | 0.1747 | 0.2041 | 0.1212 | 0.1144 |
| trend_lowvol | bull | 0.0366 | 0.2348 | 0.1557 | 0.1818 | 0.0920 |
| rsi_vol_combo | bull | 0.0288 | 0.2249 | 0.1279 | 0.1364 | 0.0727 |
| trend_lowvol | bear | 0.1932 | 0.2319 | 0.8332 | 0.6712 | 0.6963 |
| momentum_reversal | bear | 0.1291 | 0.2676 | 0.4824 | 0.1781 | 0.2842 |
| mom_x_lowvol_20_20 | bear | 0.1142 | 0.2815 | 0.4056 | 0.2055 | 0.2445 |
| rsi_vol_combo | bear | 0.0929 | 0.2433 | 0.3817 | 0.1507 | 0.2196 |
| turnover_stability | bear | 0.0614 | 0.1765 | 0.3475 | 0.1233 | 0.1952 |
| vol_confirm | bear | 0.0604 | 0.2005 | 0.3012 | 0.2877 | 0.1939 |
| fund_pe | bear | 0.0740 | 0.2598 | 0.2848 | 0.2329 | 0.1756 |
| wash_sale_score | bear | 0.0653 | 0.2386 | 0.2738 | 0.1842 | 0.1621 |
| fund_pb | bear | 0.0370 | 0.1926 | 0.1919 | 0.1507 | 0.1104 |
| fund_profit_growth | bear | 0.0254 | 0.1427 | 0.1776 | 0.1644 | 0.1034 |

### 工业母机

- **Neutral**: ['fund_pb', 'fund_pe', 'trend_lowvol'] (单因子IC=0.0697, 组合IC=0.0834)
  - weights: [0.3623, 0.3373, 0.3005]
- **Bull**: ['turnover_stability', 'volatility', 'low_downside'] (单因子IC=0.0805, 组合IC=0.0983)
  - bull_weights: [0.3547, 0.344, 0.3013]
- **Bear**: ['trend_lowvol'] (单因子IC=0.1055, 组合IC=0.1055)
  - bear_weights: [1.0]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| fund_pb | neutral | 0.0643 | 0.1523 | 0.4222 | 0.3215 | 0.2789 |
| fund_pe | neutral | 0.0639 | 0.1622 | 0.3943 | 0.3173 | 0.2597 |
| trend_lowvol | neutral | 0.0809 | 0.2194 | 0.3688 | 0.2547 | 0.2313 |
| momentum_reversal | neutral | 0.0703 | 0.1957 | 0.3589 | 0.2860 | 0.2308 |
| mom_x_lowvol_20_20 | neutral | 0.0648 | 0.2014 | 0.3218 | 0.2526 | 0.2016 |
| rsi_vol_combo | neutral | 0.0581 | 0.1842 | 0.3153 | 0.2505 | 0.1971 |
| volatility | neutral | 0.0675 | 0.2193 | 0.3076 | 0.2276 | 0.1888 |
| fund_profit_growth | neutral | 0.0477 | 0.1552 | 0.3071 | 0.2234 | 0.1878 |
| low_downside | neutral | 0.0467 | 0.1919 | 0.2432 | 0.2380 | 0.1506 |
| fund_score | neutral | 0.0433 | 0.1713 | 0.2530 | 0.1461 | 0.1450 |
| fund_revenue_growth | neutral | 0.0328 | 0.1446 | 0.2271 | 0.1315 | 0.1285 |
| fund_roe | neutral | 0.0366 | 0.1713 | 0.2137 | 0.1420 | 0.1220 |
| turnover_stability | neutral | 0.0277 | 0.1397 | 0.1983 | 0.1503 | 0.1141 |
| turnover_stability | bull | 0.0642 | 0.1237 | 0.5191 | 0.4167 | 0.3677 |
| volatility | bull | 0.0994 | 0.1858 | 0.5349 | 0.3333 | 0.3566 |
| low_downside | bull | 0.0780 | 0.1684 | 0.4632 | 0.3485 | 0.3123 |
| fund_revenue_growth | bull | 0.0634 | 0.1468 | 0.4320 | 0.3636 | 0.2946 |
| fund_pb | bull | 0.0661 | 0.1653 | 0.3997 | 0.3333 | 0.2665 |
| fund_gross_margin | bull | 0.0340 | 0.1554 | 0.2185 | 0.1970 | 0.1308 |
| trend_lowvol | bull | 0.0436 | 0.2090 | 0.2088 | 0.2197 | 0.1273 |
| fund_score | bull | 0.0361 | 0.1759 | 0.2055 | 0.2045 | 0.1238 |
| fund_profit_growth | bull | 0.0309 | 0.1602 | 0.1930 | 0.1212 | 0.1082 |
| fund_pe | bull | 0.0300 | 0.1581 | 0.1898 | 0.1364 | 0.1079 |
| mom_x_lowvol_20_20 | bull | 0.0250 | 0.1873 | 0.1333 | 0.1667 | 0.0778 |
| relative_strength | bull | 0.0212 | 0.1785 | 0.1189 | 0.1439 | 0.0680 |
| exhaustion_risk | bull | 0.0162 | 0.1424 | 0.1138 | 0.1559 | 0.0658 |
| trend_lowvol | bear | 0.1055 | 0.2239 | 0.4711 | 0.3973 | 0.3292 |
| rsi_vol_combo | bear | 0.0787 | 0.1837 | 0.4285 | 0.3973 | 0.2993 |
| momentum_reversal | bear | 0.0936 | 0.2054 | 0.4558 | 0.2603 | 0.2872 |
| turnover_stability | bear | 0.0543 | 0.1353 | 0.4014 | 0.3425 | 0.2694 |
| mom_x_lowvol_20_20 | bear | 0.0956 | 0.2329 | 0.4106 | 0.2055 | 0.2475 |
| fund_gross_margin | bear | 0.0405 | 0.1572 | 0.2580 | 0.3699 | 0.1767 |

### 工业气体

- **Neutral**: ['momentum_reversal', 'trend_lowvol', 'volatility'] (单因子IC=0.0792, 组合IC=0.113)
  - weights: [0.3922, 0.3101, 0.2977]
- **Bull**: ['low_downside', 'volatility'] (单因子IC=0.0963, 组合IC=0.1053)
  - bull_weights: [0.5333, 0.4667]
- **Bear**: ['momentum_reversal', 'mom_x_lowvol_20_20', 'fund_profit_growth'] (单因子IC=0.135, 组合IC=0.1774)
  - bear_weights: [0.3409, 0.337, 0.3221]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| momentum_reversal | neutral | 0.0900 | 0.2608 | 0.3450 | 0.2589 | 0.2171 |
| trend_lowvol | neutral | 0.0758 | 0.2713 | 0.2794 | 0.2286 | 0.1717 |
| volatility | neutral | 0.0720 | 0.2670 | 0.2696 | 0.2223 | 0.1648 |
| rsi_vol_combo | neutral | 0.0686 | 0.2619 | 0.2619 | 0.2056 | 0.1579 |
| mom_x_lowvol_20_20 | neutral | 0.0679 | 0.2597 | 0.2615 | 0.1816 | 0.1545 |
| turnover_stability | neutral | 0.0445 | 0.2050 | 0.2172 | 0.1451 | 0.1244 |
| fund_profit_growth | neutral | 0.0456 | 0.2415 | 0.1887 | 0.1827 | 0.1116 |
| low_downside | neutral | 0.0499 | 0.2790 | 0.1790 | 0.1409 | 0.1021 |
| fund_pe | neutral | 0.0395 | 0.2565 | 0.1540 | 0.1482 | 0.0884 |
| fund_score | neutral | 0.0349 | 0.2692 | 0.1296 | 0.1023 | 0.0715 |
| low_downside | bull | 0.0959 | 0.2388 | 0.4017 | 0.3523 | 0.2716 |
| volatility | bull | 0.0967 | 0.2696 | 0.3585 | 0.3258 | 0.2377 |
| fund_pe | bull | 0.0797 | 0.2122 | 0.3755 | 0.2424 | 0.2332 |
| mom_x_lowvol_20_20 | bull | 0.0646 | 0.2417 | 0.2671 | 0.1894 | 0.1588 |
| fund_pb | bull | 0.0745 | 0.3070 | 0.2426 | 0.1894 | 0.1443 |
| trend_lowvol | bull | 0.0455 | 0.2500 | 0.1822 | 0.1364 | 0.1035 |
| stroke_phase | bull | 0.0325 | 0.2155 | 0.1508 | 0.1970 | 0.0902 |
| momentum_reversal | bear | 0.1431 | 0.2665 | 0.5368 | 0.4521 | 0.3897 |
| mom_x_lowvol_20_20 | bear | 0.1558 | 0.2880 | 0.5408 | 0.4247 | 0.3853 |
| fund_profit_growth | bear | 0.1060 | 0.2091 | 0.5071 | 0.4521 | 0.3682 |
| fund_gross_margin | bear | 0.1455 | 0.2986 | 0.4873 | 0.3699 | 0.3338 |
| fund_score | bear | 0.1019 | 0.2258 | 0.4512 | 0.2603 | 0.2843 |
| fund_roe | bear | 0.0956 | 0.2310 | 0.4141 | 0.3425 | 0.2780 |
| turnover_stability | bear | 0.0996 | 0.2456 | 0.4053 | 0.3425 | 0.2721 |
| trend_lowvol | bear | 0.1074 | 0.3069 | 0.3499 | 0.2603 | 0.2205 |
| fund_pe | bear | 0.0747 | 0.2406 | 0.3105 | 0.2329 | 0.1914 |
| rsi_vol_combo | bear | 0.0729 | 0.2551 | 0.2858 | 0.2877 | 0.1840 |
| bb_width_20 | bear | 0.0852 | 0.2909 | 0.2927 | 0.2329 | 0.1804 |

### 工程建设

- **Neutral**: ['fund_profit_growth', 'turnover_stability', 'mom_x_lowvol_20_20'] (单因子IC=0.055, 组合IC=0.078)
  - weights: [0.3563, 0.3392, 0.3044]
- **Bull**: ['fund_profit_growth', 'trend_lowvol'] (单因子IC=0.061, 组合IC=0.0896)
  - bull_weights: [0.5306, 0.4694]
- **Bear**: ['momentum_reversal', 'mom_x_lowvol_20_20'] (单因子IC=0.1437, 组合IC=0.1484)
  - bear_weights: [0.5155, 0.4845]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| fund_profit_growth | neutral | 0.0487 | 0.1564 | 0.3116 | 0.3111 | 0.2043 |
| turnover_stability | neutral | 0.0498 | 0.1588 | 0.3134 | 0.2411 | 0.1945 |
| mom_x_lowvol_20_20 | neutral | 0.0665 | 0.2324 | 0.2863 | 0.2192 | 0.1745 |
| momentum_reversal | neutral | 0.0669 | 0.2401 | 0.2785 | 0.2359 | 0.1721 |
| trend_lowvol | neutral | 0.0690 | 0.2689 | 0.2566 | 0.2276 | 0.1575 |
| volatility | neutral | 0.0529 | 0.2291 | 0.2310 | 0.2150 | 0.1403 |
| rsi_vol_combo | neutral | 0.0457 | 0.2237 | 0.2044 | 0.1733 | 0.1199 |
| fund_score | neutral | 0.0447 | 0.2268 | 0.1972 | 0.1493 | 0.1133 |
| fund_pb | neutral | 0.0503 | 0.2662 | 0.1889 | 0.1086 | 0.1047 |
| low_downside | neutral | 0.0362 | 0.2348 | 0.1540 | 0.1190 | 0.0862 |
| fund_roe | neutral | 0.0261 | 0.2321 | 0.1123 | 0.1190 | 0.0628 |
| fund_profit_growth | bull | 0.0479 | 0.1437 | 0.3331 | 0.3485 | 0.2246 |
| trend_lowvol | bull | 0.0740 | 0.2399 | 0.3086 | 0.2879 | 0.1987 |
| fund_score | bull | 0.0542 | 0.1802 | 0.3008 | 0.2197 | 0.1834 |
| volatility | bull | 0.0556 | 0.1924 | 0.2891 | 0.2045 | 0.1741 |
| low_downside | bull | 0.0465 | 0.1791 | 0.2595 | 0.1667 | 0.1514 |
| momentum_reversal | bull | 0.0491 | 0.2039 | 0.2408 | 0.2348 | 0.1486 |
| turnover_stability | bull | 0.0306 | 0.1230 | 0.2487 | 0.1591 | 0.1442 |
| fund_revenue_growth | bull | 0.0325 | 0.1417 | 0.2295 | 0.2121 | 0.1391 |
| fund_pb | bull | 0.0407 | 0.2177 | 0.1870 | 0.2121 | 0.1134 |
| wash_sale_score | bull | 0.0252 | 0.1505 | 0.1673 | 0.1765 | 0.0984 |
| mom_x_lowvol_20_20 | bull | 0.0314 | 0.1902 | 0.1648 | 0.1439 | 0.0943 |
| stroke_phase | bull | 0.0223 | 0.1573 | 0.1417 | 0.1288 | 0.0800 |
| momentum_reversal | bear | 0.1473 | 0.2287 | 0.6440 | 0.4795 | 0.4764 |
| mom_x_lowvol_20_20 | bear | 0.1401 | 0.2401 | 0.5836 | 0.5342 | 0.4477 |
| rsi_vol_combo | bear | 0.1215 | 0.2138 | 0.5683 | 0.5068 | 0.4282 |
| trend_lowvol | bear | 0.0905 | 0.2404 | 0.3764 | 0.2192 | 0.2295 |

### 工程机械概念

- **Neutral**: ['fund_pb', 'mom_x_lowvol_20_20'] (单因子IC=0.0725, 组合IC=0.0963)
  - weights: [0.5233, 0.4767]
- **Bull**: ['volatility'] (单因子IC=0.0993, 组合IC=0.0993)
  - bull_weights: [1.0]
- **Bear**: ['fund_profit_growth', 'fund_revenue_growth', 'momentum_reversal'] (单因子IC=0.0743, 组合IC=0.1004)
  - bear_weights: [0.4652, 0.2677, 0.2671]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| fund_pb | neutral | 0.0735 | 0.2057 | 0.3573 | 0.2296 | 0.2197 |
| mom_x_lowvol_20_20 | neutral | 0.0714 | 0.2195 | 0.3255 | 0.2296 | 0.2001 |
| volatility | neutral | 0.0728 | 0.2463 | 0.2956 | 0.2150 | 0.1796 |
| trend_lowvol | neutral | 0.0749 | 0.2608 | 0.2874 | 0.2088 | 0.1737 |
| momentum_reversal | neutral | 0.0628 | 0.2253 | 0.2785 | 0.2161 | 0.1694 |
| rsi_vol_combo | neutral | 0.0424 | 0.2218 | 0.1913 | 0.1315 | 0.1082 |
| low_downside | neutral | 0.0413 | 0.2266 | 0.1824 | 0.1451 | 0.1044 |
| fund_pe | neutral | 0.0479 | 0.2607 | 0.1837 | 0.1232 | 0.1032 |
| fund_profit_growth | neutral | 0.0271 | 0.1820 | 0.1489 | 0.1263 | 0.0839 |
| turnover_stability | neutral | 0.0182 | 0.1831 | 0.0996 | 0.1023 | 0.0549 |
| volatility | bull | 0.0993 | 0.2205 | 0.4502 | 0.4167 | 0.3189 |
| low_downside | bull | 0.0850 | 0.2091 | 0.4066 | 0.3258 | 0.2696 |
| fund_pb | bull | 0.0807 | 0.1993 | 0.4052 | 0.3106 | 0.2655 |
| fund_pe | bull | 0.0667 | 0.2080 | 0.3208 | 0.2652 | 0.2030 |
| fund_profit_growth | bull | 0.0579 | 0.1927 | 0.3005 | 0.1818 | 0.1776 |
| fund_revenue_growth | bull | 0.0476 | 0.2243 | 0.2121 | 0.1515 | 0.1221 |
| turnover_stability | bull | 0.0328 | 0.1671 | 0.1962 | 0.1591 | 0.1137 |
| trend_lowvol | bull | 0.0445 | 0.2652 | 0.1679 | 0.1288 | 0.0947 |
| fund_score | bull | 0.0366 | 0.2373 | 0.1541 | 0.2197 | 0.0940 |
| stroke_phase | bull | 0.0231 | 0.1562 | 0.1477 | 0.1136 | 0.0823 |
| fund_profit_growth | bear | 0.0864 | 0.1952 | 0.4424 | 0.3425 | 0.2969 |
| fund_revenue_growth | bear | 0.0628 | 0.2568 | 0.2445 | 0.3973 | 0.1708 |
| momentum_reversal | bear | 0.0737 | 0.2429 | 0.3035 | 0.1233 | 0.1705 |
| fund_score | bear | 0.0754 | 0.2990 | 0.2522 | 0.1781 | 0.1486 |
| rsi_vol_combo | bear | 0.0576 | 0.2276 | 0.2530 | 0.1507 | 0.1456 |
| fund_roe | bear | 0.0515 | 0.2708 | 0.1902 | 0.1781 | 0.1120 |

### 并购重组概念

- **Neutral**: ['momentum_reversal', 'turnover_stability', 'mom_x_lowvol_20_20'] (单因子IC=0.0753, 组合IC=0.1014)
  - weights: [0.3358, 0.3347, 0.3295]
- **Bull**: ['low_downside', 'turnover_stability', 'trend_lowvol'] (单因子IC=0.0693, 组合IC=0.1007)
  - bull_weights: [0.3619, 0.3386, 0.2995]
- **Bear**: ['momentum_reversal'] (单因子IC=0.1155, 组合IC=0.1155)
  - bear_weights: [1.0]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| momentum_reversal | neutral | 0.0855 | 0.1857 | 0.4602 | 0.3528 | 0.3113 |
| turnover_stability | neutral | 0.0577 | 0.1278 | 0.4510 | 0.3758 | 0.3102 |
| mom_x_lowvol_20_20 | neutral | 0.0828 | 0.1833 | 0.4516 | 0.3528 | 0.3055 |
| trend_lowvol | neutral | 0.0840 | 0.1840 | 0.4564 | 0.3372 | 0.3051 |
| volatility | neutral | 0.0785 | 0.1823 | 0.4305 | 0.3194 | 0.2840 |
| rsi_vol_combo | neutral | 0.0615 | 0.1795 | 0.3424 | 0.2714 | 0.2177 |
| fund_pb | neutral | 0.0434 | 0.1673 | 0.2597 | 0.1628 | 0.1510 |
| low_downside | neutral | 0.0409 | 0.1789 | 0.2286 | 0.1942 | 0.1365 |
| fund_score | neutral | 0.0329 | 0.1579 | 0.2085 | 0.2046 | 0.1256 |
| fund_profit_growth | neutral | 0.0299 | 0.1389 | 0.2155 | 0.1503 | 0.1240 |
| fund_revenue_growth | neutral | 0.0233 | 0.1205 | 0.1934 | 0.1273 | 0.1090 |
| fund_pe | neutral | 0.0340 | 0.1917 | 0.1773 | 0.1086 | 0.0983 |
| fund_roe | neutral | 0.0232 | 0.1604 | 0.1447 | 0.1190 | 0.0810 |
| wash_sale_score | neutral | 0.0168 | 0.1269 | 0.1324 | 0.1049 | 0.0732 |
| low_downside | bull | 0.0779 | 0.1516 | 0.5136 | 0.3409 | 0.3444 |
| turnover_stability | bull | 0.0565 | 0.1183 | 0.4779 | 0.3485 | 0.3222 |
| trend_lowvol | bull | 0.0734 | 0.1708 | 0.4300 | 0.3258 | 0.2850 |
| volatility | bull | 0.0701 | 0.1733 | 0.4045 | 0.3409 | 0.2712 |
| rsi_vol_combo | bull | 0.0502 | 0.1592 | 0.3154 | 0.2576 | 0.1983 |
| momentum_reversal | bull | 0.0554 | 0.1805 | 0.3068 | 0.2652 | 0.1941 |
| mom_x_lowvol_20_20 | bull | 0.0468 | 0.1790 | 0.2614 | 0.2197 | 0.1594 |
| fund_pb | bull | 0.0342 | 0.1724 | 0.1983 | 0.1439 | 0.1134 |
| momentum_reversal | bear | 0.1155 | 0.2154 | 0.5363 | 0.2603 | 0.3379 |
| trend_lowvol | bear | 0.0939 | 0.2114 | 0.4440 | 0.4521 | 0.3224 |
| rsi_vol_combo | bear | 0.0851 | 0.1711 | 0.4976 | 0.2877 | 0.3204 |
| mom_x_lowvol_20_20 | bear | 0.0952 | 0.2078 | 0.4583 | 0.2603 | 0.2888 |
| fund_revenue_growth | bear | 0.0362 | 0.1377 | 0.2626 | 0.2877 | 0.1690 |
| turnover_stability | bear | 0.0369 | 0.1494 | 0.2469 | 0.2877 | 0.1589 |
| fund_gross_margin | bear | 0.0444 | 0.1816 | 0.2445 | 0.2329 | 0.1507 |

### 幽门螺杆菌概念

- **Neutral**: ['fund_pb', 'volatility', 'momentum_reversal'] (单因子IC=0.1044, 组合IC=0.1476)
  - weights: [0.4034, 0.3525, 0.2441]
- **Bull**: ['low_downside', 'fund_pb', 'volatility'] (单因子IC=0.0966, 组合IC=0.12)
  - bull_weights: [0.3576, 0.3263, 0.3162]
- **Bear**: ['momentum_reversal', 'trend_lowvol', 'mom_x_lowvol_20_20'] (单因子IC=0.1293, 组合IC=0.1489)
  - bear_weights: [0.3416, 0.3297, 0.3287]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| fund_pb | neutral | 0.1119 | 0.2028 | 0.5520 | 0.4092 | 0.3889 |
| volatility | neutral | 0.1170 | 0.2419 | 0.4838 | 0.4050 | 0.3398 |
| momentum_reversal | neutral | 0.0842 | 0.2250 | 0.3741 | 0.2578 | 0.2353 |
| low_downside | neutral | 0.0766 | 0.2311 | 0.3316 | 0.3299 | 0.2205 |
| trend_lowvol | neutral | 0.0839 | 0.2491 | 0.3369 | 0.2370 | 0.2084 |
| mom_x_lowvol_20_20 | neutral | 0.0762 | 0.2213 | 0.3443 | 0.2035 | 0.2072 |
| rsi_vol_combo | neutral | 0.0665 | 0.2093 | 0.3180 | 0.2401 | 0.1972 |
| turnover_stability | neutral | 0.0549 | 0.1817 | 0.3022 | 0.2213 | 0.1845 |
| fund_pe | neutral | 0.0709 | 0.2292 | 0.3093 | 0.1816 | 0.1827 |
| fund_profit_growth | neutral | 0.0517 | 0.1814 | 0.2851 | 0.2129 | 0.1729 |
| fund_score | neutral | 0.0380 | 0.2057 | 0.1849 | 0.1587 | 0.1071 |
| low_downside | bull | 0.0970 | 0.1852 | 0.5236 | 0.3788 | 0.3610 |
| fund_pb | bull | 0.0960 | 0.1943 | 0.4940 | 0.3333 | 0.3293 |
| volatility | bull | 0.0969 | 0.2024 | 0.4787 | 0.3333 | 0.3191 |
| trend_lowvol | bull | 0.0618 | 0.1957 | 0.3158 | 0.3030 | 0.2057 |
| fund_pe | bull | 0.0678 | 0.2332 | 0.2909 | 0.1970 | 0.1741 |
| turnover_stability | bull | 0.0363 | 0.1676 | 0.2162 | 0.1667 | 0.1261 |
| momentum_reversal | bear | 0.1441 | 0.2275 | 0.6336 | 0.4521 | 0.4600 |
| trend_lowvol | bear | 0.1093 | 0.1755 | 0.6232 | 0.4247 | 0.4439 |
| mom_x_lowvol_20_20 | bear | 0.1343 | 0.2223 | 0.6040 | 0.4658 | 0.4427 |
| rsi_vol_combo | bear | 0.1330 | 0.2252 | 0.5907 | 0.4521 | 0.4289 |
| fund_revenue_growth | bear | 0.0840 | 0.1533 | 0.5481 | 0.4247 | 0.3904 |
| fund_pb | bear | 0.0826 | 0.1666 | 0.4954 | 0.2877 | 0.3190 |
| volatility | bear | 0.0644 | 0.2209 | 0.2915 | 0.1507 | 0.1677 |
| fund_score | bear | 0.0376 | 0.2103 | 0.1789 | 0.2055 | 0.1078 |

### 建筑节能

- **Neutral**: ['fund_pb', 'mom_x_lowvol_20_20', 'volatility'] (单因子IC=0.0698, 组合IC=0.0981)
  - weights: [0.4203, 0.2973, 0.2824]
- **Bull**: ['low_downside', 'volatility', 'fund_pb'] (单因子IC=0.1101, 组合IC=0.1285)
  - bull_weights: [0.353, 0.3476, 0.2994]
- **Bear**: ['fund_pb', 'mom_x_lowvol_20_20', 'momentum_reversal'] (单因子IC=0.1151, 组合IC=0.1569)
  - bear_weights: [0.3652, 0.3634, 0.2714]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| fund_pb | neutral | 0.0755 | 0.1867 | 0.4045 | 0.3111 | 0.2652 |
| mom_x_lowvol_20_20 | neutral | 0.0698 | 0.2217 | 0.3150 | 0.1910 | 0.1876 |
| volatility | neutral | 0.0642 | 0.2262 | 0.2836 | 0.2568 | 0.1782 |
| momentum_reversal | neutral | 0.0683 | 0.2328 | 0.2932 | 0.1754 | 0.1723 |
| rsi_vol_combo | neutral | 0.0557 | 0.2218 | 0.2512 | 0.1670 | 0.1465 |
| trend_lowvol | neutral | 0.0558 | 0.2398 | 0.2326 | 0.1931 | 0.1388 |
| fund_pe | neutral | 0.0474 | 0.2234 | 0.2123 | 0.1503 | 0.1221 |
| turnover_stability | neutral | 0.0346 | 0.1746 | 0.1982 | 0.1733 | 0.1163 |
| fund_score | neutral | 0.0299 | 0.2086 | 0.1431 | 0.1106 | 0.0795 |
| fund_gross_margin | neutral | 0.0249 | 0.1827 | 0.1365 | 0.1106 | 0.0758 |
| fund_profit_growth | neutral | 0.0247 | 0.1977 | 0.1248 | 0.1065 | 0.0690 |
| low_downside | neutral | 0.0245 | 0.2255 | 0.1089 | 0.1232 | 0.0611 |
| low_downside | bull | 0.1133 | 0.1969 | 0.5756 | 0.4470 | 0.4164 |
| volatility | bull | 0.1181 | 0.2096 | 0.5637 | 0.4545 | 0.4100 |
| fund_pb | bull | 0.0988 | 0.1866 | 0.5298 | 0.3333 | 0.3532 |
| fund_pe | bull | 0.0942 | 0.2267 | 0.4154 | 0.3864 | 0.2880 |
| momentum_reversal | bull | 0.0852 | 0.2142 | 0.3977 | 0.2803 | 0.2546 |
| mom_x_lowvol_20_20 | bull | 0.0843 | 0.2171 | 0.3882 | 0.2955 | 0.2515 |
| trend_lowvol | bull | 0.0814 | 0.2156 | 0.3774 | 0.1894 | 0.2244 |
| fund_profit_growth | bull | 0.0659 | 0.1892 | 0.3483 | 0.2576 | 0.2190 |
| fund_score | bull | 0.0613 | 0.1898 | 0.3230 | 0.2348 | 0.1994 |
| turnover_stability | bull | 0.0511 | 0.1814 | 0.2816 | 0.2273 | 0.1728 |
| rsi_vol_combo | bull | 0.0524 | 0.2180 | 0.2402 | 0.1742 | 0.1410 |
| fund_roe | bull | 0.0528 | 0.2156 | 0.2451 | 0.1061 | 0.1356 |
| fund_pb | bear | 0.1132 | 0.2156 | 0.5252 | 0.3973 | 0.3669 |
| mom_x_lowvol_20_20 | bear | 0.1249 | 0.2390 | 0.5227 | 0.3973 | 0.3651 |
| momentum_reversal | bear | 0.1070 | 0.2473 | 0.4327 | 0.2603 | 0.2727 |
| fund_pe | bear | 0.1021 | 0.2413 | 0.4230 | 0.2877 | 0.2723 |
| rsi_vol_combo | bear | 0.0780 | 0.2531 | 0.3080 | 0.2055 | 0.1857 |
| trend_lowvol | bear | 0.0564 | 0.2328 | 0.2421 | 0.2055 | 0.1459 |

### 影视概念

- **Neutral**: ['trend_lowvol', 'fund_pb', 'volatility'] (单因子IC=0.0971, 组合IC=0.1187)
  - weights: [0.3623, 0.3409, 0.2968]
- **Bull**: ['trend_lowvol', 'fund_pb'] (单因子IC=0.0926, 组合IC=0.1132)
  - bull_weights: [0.5332, 0.4668]
- **Bear**: ['momentum_reversal'] (单因子IC=0.1663, 组合IC=0.1663)
  - bear_weights: [1.0]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| trend_lowvol | neutral | 0.1161 | 0.2408 | 0.4821 | 0.3591 | 0.3276 |
| fund_pb | neutral | 0.0773 | 0.1728 | 0.4474 | 0.3779 | 0.3082 |
| volatility | neutral | 0.0978 | 0.2401 | 0.4074 | 0.3173 | 0.2683 |
| momentum_reversal | neutral | 0.0922 | 0.2285 | 0.4035 | 0.3111 | 0.2645 |
| mom_x_lowvol_20_20 | neutral | 0.0855 | 0.2342 | 0.3649 | 0.2923 | 0.2358 |
| rsi_vol_combo | neutral | 0.0740 | 0.2066 | 0.3582 | 0.3111 | 0.2348 |
| low_downside | neutral | 0.0680 | 0.2406 | 0.2827 | 0.2129 | 0.1715 |
| fund_pe | neutral | 0.0563 | 0.2111 | 0.2668 | 0.1879 | 0.1585 |
| turnover_stability | neutral | 0.0317 | 0.1595 | 0.1984 | 0.2276 | 0.1218 |
| trend_lowvol | bull | 0.1040 | 0.1946 | 0.5344 | 0.4242 | 0.3806 |
| fund_pb | bull | 0.0812 | 0.1726 | 0.4703 | 0.4167 | 0.3331 |
| volatility | bull | 0.0868 | 0.1903 | 0.4563 | 0.3636 | 0.3111 |
| mom_x_lowvol_20_20 | bull | 0.0812 | 0.1979 | 0.4105 | 0.4394 | 0.2954 |
| low_downside | bull | 0.0834 | 0.1853 | 0.4502 | 0.3106 | 0.2950 |
| momentum_reversal | bull | 0.0789 | 0.1990 | 0.3966 | 0.3636 | 0.2704 |
| fund_pe | bull | 0.0426 | 0.2121 | 0.2009 | 0.1591 | 0.1164 |
| momentum_reversal | bear | 0.1663 | 0.1962 | 0.8479 | 0.6712 | 0.7085 |
| rsi_vol_combo | bear | 0.1108 | 0.1561 | 0.7098 | 0.4795 | 0.5251 |
| mom_x_lowvol_20_20 | bear | 0.1355 | 0.2294 | 0.5908 | 0.5068 | 0.4451 |
| trend_lowvol | bear | 0.1288 | 0.2315 | 0.5566 | 0.5068 | 0.4194 |
| bb_width_20 | bear | 0.0772 | 0.2191 | 0.3524 | 0.4795 | 0.2607 |

### 微利股

- **Neutral**: ['mom_x_lowvol_20_20', 'momentum_reversal', 'trend_lowvol'] (单因子IC=0.0772, 组合IC=0.0848)
  - weights: [0.3635, 0.3459, 0.2906]
- **Bull**: ['trend_lowvol', 'low_downside'] (单因子IC=0.1008, 组合IC=0.1155)
  - bull_weights: [0.5656, 0.4344]
- **Bear**: ['mom_x_lowvol_20_20'] (单因子IC=0.1077, 组合IC=0.1077)
  - bear_weights: [1.0]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| mom_x_lowvol_20_20 | neutral | 0.0800 | 0.1744 | 0.4584 | 0.3737 | 0.3149 |
| momentum_reversal | neutral | 0.0786 | 0.1767 | 0.4449 | 0.3466 | 0.2996 |
| trend_lowvol | neutral | 0.0731 | 0.1837 | 0.3978 | 0.2651 | 0.2517 |
| rsi_vol_combo | neutral | 0.0551 | 0.1633 | 0.3378 | 0.2109 | 0.2045 |
| fund_pb | neutral | 0.0607 | 0.1843 | 0.3291 | 0.2401 | 0.2040 |
| volatility | neutral | 0.0621 | 0.1984 | 0.3133 | 0.2296 | 0.1926 |
| low_downside | neutral | 0.0422 | 0.1794 | 0.2354 | 0.2046 | 0.1418 |
| fund_profit_growth | neutral | 0.0271 | 0.1308 | 0.2070 | 0.1482 | 0.1188 |
| fund_pe | neutral | 0.0312 | 0.1531 | 0.2036 | 0.1524 | 0.1173 |
| turnover_stability | neutral | 0.0215 | 0.1248 | 0.1720 | 0.1503 | 0.0989 |
| fund_score | neutral | 0.0176 | 0.1478 | 0.1188 | 0.1002 | 0.0654 |
| trend_lowvol | bull | 0.1132 | 0.1663 | 0.6808 | 0.4318 | 0.4874 |
| low_downside | bull | 0.0884 | 0.1646 | 0.5371 | 0.3939 | 0.3743 |
| volatility | bull | 0.0873 | 0.1846 | 0.4727 | 0.3939 | 0.3294 |
| fund_pb | bull | 0.0805 | 0.1750 | 0.4602 | 0.4015 | 0.3225 |
| momentum_reversal | bull | 0.0682 | 0.1516 | 0.4497 | 0.3712 | 0.3083 |
| mom_x_lowvol_20_20 | bull | 0.0523 | 0.1533 | 0.3413 | 0.2386 | 0.2114 |
| turnover_stability | bull | 0.0301 | 0.1107 | 0.2719 | 0.2045 | 0.1638 |
| fund_pe | bull | 0.0406 | 0.1711 | 0.2374 | 0.2576 | 0.1492 |
| rsi_vol_combo | bull | 0.0403 | 0.1675 | 0.2404 | 0.1894 | 0.1430 |
| stroke_phase | bull | 0.0264 | 0.1233 | 0.2144 | 0.1591 | 0.1243 |
| mom_x_lowvol_20_20 | bear | 0.1077 | 0.2198 | 0.4900 | 0.4247 | 0.3490 |
| momentum_reversal | bear | 0.0900 | 0.2249 | 0.4003 | 0.2603 | 0.2523 |
| rsi_vol_combo | bear | 0.0496 | 0.1698 | 0.2919 | 0.3151 | 0.1920 |
| fund_gross_margin | bear | 0.0352 | 0.1474 | 0.2390 | 0.3151 | 0.1571 |
| trend_lowvol | bear | 0.0541 | 0.2408 | 0.2245 | 0.2055 | 0.1353 |

### 微盘精选

- **Neutral**: ['mom_x_lowvol_20_20', 'momentum_reversal', 'volatility'] (单因子IC=0.105, 组合IC=0.1206)
  - weights: [0.3717, 0.3431, 0.2852]
- **Bull**: ['volatility'] (单因子IC=0.1543, 组合IC=0.1543)
  - bull_weights: [1.0]
- **Bear**: ['mom_x_lowvol_20_20'] (单因子IC=0.2062, 组合IC=0.2062)
  - bear_weights: [1.0]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| mom_x_lowvol_20_20 | neutral | 0.1141 | 0.1933 | 0.5900 | 0.4614 | 0.4311 |
| momentum_reversal | neutral | 0.1082 | 0.1950 | 0.5548 | 0.4342 | 0.3978 |
| volatility | neutral | 0.0928 | 0.1971 | 0.4708 | 0.4050 | 0.3307 |
| rsi_vol_combo | neutral | 0.0802 | 0.1884 | 0.4256 | 0.3601 | 0.2894 |
| trend_lowvol | neutral | 0.0841 | 0.2212 | 0.3801 | 0.3132 | 0.2496 |
| fund_pb | neutral | 0.0522 | 0.1501 | 0.3475 | 0.2599 | 0.2189 |
| fund_profit_growth | neutral | 0.0550 | 0.1607 | 0.3424 | 0.2317 | 0.2109 |
| turnover_stability | neutral | 0.0515 | 0.1664 | 0.3093 | 0.2401 | 0.1918 |
| low_downside | neutral | 0.0426 | 0.1946 | 0.2187 | 0.1482 | 0.1256 |
| fund_score | neutral | 0.0293 | 0.1772 | 0.1654 | 0.1148 | 0.0922 |
| volatility | bull | 0.1543 | 0.1972 | 0.7825 | 0.5833 | 0.6195 |
| low_downside | bull | 0.1101 | 0.1842 | 0.5978 | 0.5379 | 0.4596 |
| trend_lowvol | bull | 0.1162 | 0.2093 | 0.5554 | 0.4470 | 0.4018 |
| mom_x_lowvol_20_20 | bull | 0.1034 | 0.1922 | 0.5383 | 0.3939 | 0.3752 |
| momentum_reversal | bull | 0.0923 | 0.1806 | 0.5112 | 0.4545 | 0.3717 |
| fund_pb | bull | 0.0870 | 0.1702 | 0.5113 | 0.3182 | 0.3370 |
| turnover_stability | bull | 0.0713 | 0.1808 | 0.3940 | 0.2727 | 0.2508 |
| rsi_vol_combo | bull | 0.0502 | 0.1889 | 0.2656 | 0.1742 | 0.1559 |
| stroke_phase | bull | 0.0228 | 0.1710 | 0.1334 | 0.1212 | 0.0748 |
| mom_x_lowvol_20_20 | bear | 0.2062 | 0.1999 | 1.0317 | 0.6712 | 0.8621 |
| momentum_reversal | bear | 0.1937 | 0.2106 | 0.9196 | 0.6712 | 0.7684 |
| trend_lowvol | bear | 0.1326 | 0.2061 | 0.6436 | 0.4247 | 0.4584 |
| rsi_vol_combo | bear | 0.1176 | 0.1906 | 0.6171 | 0.4795 | 0.4565 |
| fund_pb | bear | 0.0758 | 0.1289 | 0.5880 | 0.3973 | 0.4108 |
| turnover_stability | bear | 0.0564 | 0.1503 | 0.3755 | 0.1781 | 0.2212 |
| bb_width_20 | bear | 0.0703 | 0.2317 | 0.3032 | 0.1781 | 0.1786 |
| fund_score | bear | 0.0518 | 0.2234 | 0.2319 | 0.2603 | 0.1461 |

### 快手概念

- **Neutral**: ['trend_lowvol', 'momentum_reversal', 'volatility'] (单因子IC=0.1388, 组合IC=0.1725)
  - weights: [0.3734, 0.3212, 0.3055]
- **Bull**: ['volatility', 'fund_pb', 'low_downside'] (单因子IC=0.1246, 组合IC=0.1542)
  - bull_weights: [0.3645, 0.3644, 0.2711]
- **Bear**: ['trend_lowvol'] (单因子IC=0.191, 组合IC=0.191)
  - bear_weights: [1.0]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| trend_lowvol | neutral | 0.1472 | 0.2521 | 0.5838 | 0.4353 | 0.4190 |
| momentum_reversal | neutral | 0.1309 | 0.2525 | 0.5184 | 0.3904 | 0.3604 |
| volatility | neutral | 0.1383 | 0.2775 | 0.4983 | 0.3758 | 0.3428 |
| mom_x_lowvol_20_20 | neutral | 0.1296 | 0.2604 | 0.4976 | 0.3695 | 0.3408 |
| rsi_vol_combo | neutral | 0.0931 | 0.2517 | 0.3696 | 0.2714 | 0.2350 |
| low_downside | neutral | 0.0816 | 0.2508 | 0.3252 | 0.2526 | 0.2037 |
| fund_pb | neutral | 0.0621 | 0.2198 | 0.2827 | 0.2714 | 0.1797 |
| fund_pe | neutral | 0.0560 | 0.2532 | 0.2212 | 0.1649 | 0.1289 |
| turnover_stability | neutral | 0.0430 | 0.1996 | 0.2152 | 0.1451 | 0.1232 |
| fund_profit_growth | neutral | 0.0409 | 0.2090 | 0.1958 | 0.1315 | 0.1108 |
| fund_roe | neutral | 0.0306 | 0.1952 | 0.1568 | 0.1002 | 0.0862 |
| volatility | bull | 0.1420 | 0.2364 | 0.6008 | 0.4848 | 0.4461 |
| fund_pb | bull | 0.1247 | 0.2045 | 0.6100 | 0.4621 | 0.4459 |
| low_downside | bull | 0.1070 | 0.2260 | 0.4734 | 0.4015 | 0.3317 |
| trend_lowvol | bull | 0.1078 | 0.2387 | 0.4516 | 0.3864 | 0.3130 |
| mom_x_lowvol_20_20 | bull | 0.0864 | 0.2181 | 0.3961 | 0.2879 | 0.2551 |
| fund_pe | bull | 0.0709 | 0.2207 | 0.3210 | 0.3182 | 0.2116 |
| momentum_reversal | bull | 0.0651 | 0.2167 | 0.3004 | 0.2045 | 0.1809 |
| fund_score | bull | 0.0598 | 0.2117 | 0.2825 | 0.2576 | 0.1776 |
| fund_roe | bull | 0.0526 | 0.1868 | 0.2815 | 0.2083 | 0.1701 |
| fund_profit_growth | bull | 0.0460 | 0.2074 | 0.2216 | 0.1742 | 0.1301 |
| rsi_vol_combo | bull | 0.0468 | 0.2330 | 0.2007 | 0.2045 | 0.1209 |
| trend_lowvol | bear | 0.1910 | 0.2662 | 0.7175 | 0.5068 | 0.5406 |
| momentum_reversal | bear | 0.1371 | 0.2402 | 0.5709 | 0.5068 | 0.4301 |
| mom_x_lowvol_20_20 | bear | 0.1254 | 0.2659 | 0.4715 | 0.3699 | 0.3230 |
| fund_profit_growth | bear | 0.0764 | 0.1892 | 0.4036 | 0.3973 | 0.2820 |
| turnover_stability | bear | 0.0718 | 0.1713 | 0.4189 | 0.3425 | 0.2812 |
| fund_revenue_growth | bear | 0.0647 | 0.1524 | 0.4243 | 0.2877 | 0.2732 |
| rsi_vol_combo | bear | 0.0779 | 0.2392 | 0.3254 | 0.2877 | 0.2095 |
| fund_score | bear | 0.0637 | 0.2157 | 0.2953 | 0.2055 | 0.1780 |

### 快递概念

- **Neutral**: ['trend_lowvol', 'mom_x_lowvol_20_20'] (单因子IC=0.0876, 组合IC=0.1035)
  - weights: [0.5624, 0.4376]
- **Bull**: ['volatility', 'low_downside'] (单因子IC=0.1097, 组合IC=0.1099)
  - bull_weights: [0.522, 0.478]
- **Bear**: ['mom_x_lowvol_20_20', 'momentum_reversal', 'fund_profit_growth'] (单因子IC=0.1274, 组合IC=0.1706)
  - bear_weights: [0.3944, 0.3845, 0.2211]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| trend_lowvol | neutral | 0.1000 | 0.2406 | 0.4155 | 0.2881 | 0.2676 |
| mom_x_lowvol_20_20 | neutral | 0.0753 | 0.2262 | 0.3331 | 0.2505 | 0.2082 |
| momentum_reversal | neutral | 0.0746 | 0.2283 | 0.3268 | 0.2463 | 0.2036 |
| fund_pb | neutral | 0.0656 | 0.2101 | 0.3123 | 0.2505 | 0.1953 |
| fund_profit_growth | neutral | 0.0489 | 0.1770 | 0.2762 | 0.2088 | 0.1669 |
| turnover_stability | neutral | 0.0469 | 0.1770 | 0.2653 | 0.1785 | 0.1563 |
| volatility | neutral | 0.0562 | 0.2400 | 0.2340 | 0.2025 | 0.1407 |
| fund_pe | neutral | 0.0504 | 0.2578 | 0.1955 | 0.1608 | 0.1135 |
| low_downside | neutral | 0.0428 | 0.2298 | 0.1864 | 0.1795 | 0.1099 |
| fund_score | neutral | 0.0274 | 0.1877 | 0.1462 | 0.1023 | 0.0806 |
| volatility | bull | 0.1088 | 0.2238 | 0.4862 | 0.4318 | 0.3481 |
| low_downside | bull | 0.1106 | 0.2275 | 0.4864 | 0.3106 | 0.3187 |
| turnover_stability | bull | 0.0566 | 0.1463 | 0.3867 | 0.3561 | 0.2622 |
| trend_lowvol | bull | 0.0603 | 0.2233 | 0.2701 | 0.1742 | 0.1586 |
| fund_pb | bull | 0.0516 | 0.2203 | 0.2341 | 0.1970 | 0.1401 |
| stroke_phase | bull | 0.0300 | 0.1935 | 0.1548 | 0.1439 | 0.0885 |
| vol_confirm | bull | 0.0243 | 0.2043 | 0.1191 | 0.2121 | 0.0722 |
| mom_x_lowvol_20_20 | bear | 0.1526 | 0.2331 | 0.6547 | 0.6164 | 0.5291 |
| momentum_reversal | bear | 0.1509 | 0.2324 | 0.6493 | 0.5890 | 0.5159 |
| fund_profit_growth | bear | 0.0788 | 0.1819 | 0.4332 | 0.3699 | 0.2967 |
| trend_lowvol | bear | 0.0906 | 0.1979 | 0.4576 | 0.2603 | 0.2883 |
| rsi_vol_combo | bear | 0.0993 | 0.2523 | 0.3936 | 0.2603 | 0.2480 |
| fund_pe | bear | 0.0967 | 0.2679 | 0.3609 | 0.2603 | 0.2274 |
| fund_score | bear | 0.0560 | 0.1648 | 0.3397 | 0.2877 | 0.2187 |
| fund_pb | bear | 0.0984 | 0.2753 | 0.3573 | 0.2055 | 0.2154 |
| bb_width_20 | bear | 0.0751 | 0.2282 | 0.3292 | 0.1781 | 0.1939 |
| fund_gross_margin | bear | 0.0381 | 0.1952 | 0.1951 | 0.1233 | 0.1096 |

### 成渝特区

- **Neutral**: ['fund_pb', 'fund_profit_growth', 'trend_lowvol'] (单因子IC=0.0602, 组合IC=0.0761)
  - weights: [0.3708, 0.3204, 0.3088]
- **Bull**: ['low_downside', 'turnover_stability'] (单因子IC=0.092, 组合IC=0.1187)
  - bull_weights: [0.5212, 0.4788]
- **Bear**: ['momentum_reversal', 'mom_x_lowvol_20_20'] (单因子IC=0.1281, 组合IC=0.1417)
  - bear_weights: [0.5075, 0.4925]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| fund_pb | neutral | 0.0701 | 0.1900 | 0.3691 | 0.2860 | 0.2373 |
| fund_profit_growth | neutral | 0.0426 | 0.1260 | 0.3381 | 0.2129 | 0.2051 |
| trend_lowvol | neutral | 0.0678 | 0.2118 | 0.3199 | 0.2359 | 0.1977 |
| mom_x_lowvol_20_20 | neutral | 0.0470 | 0.1795 | 0.2619 | 0.1524 | 0.1509 |
| momentum_reversal | neutral | 0.0442 | 0.1937 | 0.2284 | 0.1837 | 0.1352 |
| fund_pe | neutral | 0.0456 | 0.2032 | 0.2243 | 0.1691 | 0.1311 |
| volatility | neutral | 0.0364 | 0.1881 | 0.1935 | 0.1524 | 0.1115 |
| turnover_stability | neutral | 0.0255 | 0.1376 | 0.1851 | 0.1921 | 0.1103 |
| low_downside | neutral | 0.0393 | 0.2118 | 0.1858 | 0.1503 | 0.1068 |
| rsi_vol_combo | neutral | 0.0282 | 0.1887 | 0.1496 | 0.1086 | 0.0829 |
| low_downside | bull | 0.1024 | 0.1530 | 0.6696 | 0.4848 | 0.4971 |
| turnover_stability | bull | 0.0815 | 0.1352 | 0.6028 | 0.5152 | 0.4567 |
| volatility | bull | 0.0791 | 0.1670 | 0.4735 | 0.3485 | 0.3192 |
| trend_lowvol | bull | 0.0770 | 0.1845 | 0.4176 | 0.2727 | 0.2657 |
| momentum_reversal | bull | 0.0694 | 0.1709 | 0.4058 | 0.2955 | 0.2629 |
| stroke_phase | bull | 0.0460 | 0.1274 | 0.3614 | 0.3409 | 0.2423 |
| fund_profit_growth | bull | 0.0429 | 0.1230 | 0.3487 | 0.2273 | 0.2140 |
| fund_pb | bull | 0.0682 | 0.1951 | 0.3496 | 0.2197 | 0.2132 |
| fund_pe | bull | 0.0607 | 0.1990 | 0.3052 | 0.2121 | 0.1850 |
| mom_x_lowvol_20_20 | bull | 0.0401 | 0.1649 | 0.2432 | 0.2273 | 0.1493 |
| rsi_vol_combo | bull | 0.0318 | 0.1633 | 0.1949 | 0.1515 | 0.1122 |
| momentum_reversal | bear | 0.1335 | 0.2349 | 0.5682 | 0.4521 | 0.4125 |
| mom_x_lowvol_20_20 | bear | 0.1227 | 0.2268 | 0.5411 | 0.4795 | 0.4003 |
| fund_revenue_growth | bear | 0.0390 | 0.0946 | 0.4122 | 0.3425 | 0.2767 |
| rsi_vol_combo | bear | 0.0702 | 0.2039 | 0.3444 | 0.2329 | 0.2123 |
| trend_lowvol | bear | 0.0657 | 0.2241 | 0.2933 | 0.2329 | 0.1808 |
| fund_profit_growth | bear | 0.0326 | 0.1146 | 0.2846 | 0.1507 | 0.1638 |
| bb_width_20 | bear | 0.0531 | 0.2255 | 0.2353 | 0.2603 | 0.1483 |

### 户外露营

- **Neutral**: ['fund_pb', 'volatility', 'mom_x_lowvol_20_20'] (单因子IC=0.0833, 组合IC=0.1169)
  - weights: [0.4593, 0.3062, 0.2344]
- **Bull**: ['volatility', 'fund_gross_margin', 'fund_pb'] (单因子IC=0.0854, 组合IC=0.1366)
  - bull_weights: [0.4153, 0.2988, 0.2859]
- **Bear**: ['turnover_stability', 'volatility', 'rsi_vol_combo'] (单因子IC=0.0898, 组合IC=0.1054)
  - bear_weights: [0.4179, 0.3379, 0.2442]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| fund_pb | neutral | 0.1075 | 0.2458 | 0.4374 | 0.3570 | 0.2967 |
| volatility | neutral | 0.0779 | 0.2513 | 0.3102 | 0.2756 | 0.1978 |
| mom_x_lowvol_20_20 | neutral | 0.0646 | 0.2574 | 0.2508 | 0.2077 | 0.1515 |
| low_downside | neutral | 0.0537 | 0.2566 | 0.2095 | 0.2328 | 0.1291 |
| trend_lowvol | neutral | 0.0614 | 0.2780 | 0.2210 | 0.1555 | 0.1277 |
| momentum_reversal | neutral | 0.0549 | 0.2562 | 0.2144 | 0.1775 | 0.1262 |
| fund_profit_growth | neutral | 0.0500 | 0.2410 | 0.2076 | 0.1534 | 0.1198 |
| fund_revenue_growth | neutral | 0.0496 | 0.2419 | 0.2051 | 0.1315 | 0.1160 |
| turnover_stability | neutral | 0.0452 | 0.2330 | 0.1940 | 0.1597 | 0.1125 |
| fund_pe | neutral | 0.0482 | 0.2672 | 0.1802 | 0.1587 | 0.1044 |
| fund_score | neutral | 0.0441 | 0.2659 | 0.1657 | 0.1086 | 0.0919 |
| rsi_vol_combo | neutral | 0.0363 | 0.2444 | 0.1485 | 0.1576 | 0.0859 |
| volatility | bull | 0.1020 | 0.2372 | 0.4302 | 0.3674 | 0.2941 |
| fund_gross_margin | bull | 0.0789 | 0.2535 | 0.3112 | 0.3598 | 0.2116 |
| fund_pb | bull | 0.0752 | 0.2390 | 0.3145 | 0.2879 | 0.2025 |
| low_downside | bull | 0.0674 | 0.2417 | 0.2789 | 0.2500 | 0.1743 |
| fund_revenue_growth | bull | 0.0447 | 0.2086 | 0.2143 | 0.2386 | 0.1327 |
| turnover_stability | bull | 0.0457 | 0.2272 | 0.2012 | 0.2273 | 0.1234 |
| fund_score | bull | 0.0444 | 0.2384 | 0.1864 | 0.1212 | 0.1045 |
| mom_x_lowvol_20_20 | bull | 0.0286 | 0.2403 | 0.1191 | 0.1212 | 0.0668 |
| turnover_stability | bear | 0.1049 | 0.2320 | 0.4524 | 0.3562 | 0.3068 |
| volatility | bear | 0.0932 | 0.2367 | 0.3937 | 0.2603 | 0.2481 |
| rsi_vol_combo | bear | 0.0713 | 0.2396 | 0.2975 | 0.2055 | 0.1793 |
| fund_pb | bear | 0.0657 | 0.2790 | 0.2356 | 0.1781 | 0.1388 |

### 房屋检测

- **Neutral**: ['turnover_stability', 'momentum_reversal', 'mom_x_lowvol_20_20'] (单因子IC=0.0797, 组合IC=0.1035)
  - weights: [0.378, 0.3152, 0.3068]
- **Bull**: ['trend_lowvol'] (单因子IC=0.1393, 组合IC=0.1393)
  - bull_weights: [1.0]
- **Bear**: ['mom_x_lowvol_20_20'] (单因子IC=0.1556, 组合IC=0.1556)
  - bear_weights: [1.0]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| turnover_stability | neutral | 0.0698 | 0.1808 | 0.3860 | 0.3267 | 0.2560 |
| momentum_reversal | neutral | 0.0850 | 0.2532 | 0.3356 | 0.2724 | 0.2135 |
| mom_x_lowvol_20_20 | neutral | 0.0844 | 0.2540 | 0.3324 | 0.2505 | 0.2078 |
| trend_lowvol | neutral | 0.0772 | 0.2577 | 0.2994 | 0.2568 | 0.1882 |
| fund_profit_growth | neutral | 0.0558 | 0.1913 | 0.2917 | 0.2651 | 0.1845 |
| rsi_vol_combo | neutral | 0.0728 | 0.2441 | 0.2981 | 0.2255 | 0.1826 |
| fund_score | neutral | 0.0504 | 0.2048 | 0.2461 | 0.2182 | 0.1499 |
| fund_revenue_growth | neutral | 0.0465 | 0.2054 | 0.2262 | 0.1879 | 0.1344 |
| volatility | neutral | 0.0459 | 0.2567 | 0.1787 | 0.1733 | 0.1049 |
| low_downside | neutral | 0.0418 | 0.2409 | 0.1734 | 0.1587 | 0.1005 |
| fund_pb | neutral | 0.0402 | 0.2431 | 0.1652 | 0.1336 | 0.0936 |
| trend_lowvol | bull | 0.1393 | 0.2370 | 0.5879 | 0.4545 | 0.4275 |
| volatility | bull | 0.1105 | 0.2192 | 0.5041 | 0.3939 | 0.3514 |
| fund_pe | bull | 0.0899 | 0.2065 | 0.4352 | 0.4091 | 0.3066 |
| fund_pb | bull | 0.0860 | 0.2004 | 0.4293 | 0.3258 | 0.2846 |
| low_downside | bull | 0.0969 | 0.2313 | 0.4190 | 0.3030 | 0.2730 |
| stroke_phase | bull | 0.0694 | 0.1730 | 0.4009 | 0.3333 | 0.2672 |
| fund_score | bull | 0.0636 | 0.1727 | 0.3682 | 0.3106 | 0.2413 |
| fund_revenue_growth | bull | 0.0502 | 0.1672 | 0.3003 | 0.2273 | 0.1843 |
| fund_roe | bull | 0.0527 | 0.2227 | 0.2367 | 0.2500 | 0.1479 |
| momentum_reversal | bull | 0.0404 | 0.2094 | 0.1931 | 0.1515 | 0.1112 |
| turnover_stability | bull | 0.0312 | 0.1755 | 0.1775 | 0.1667 | 0.1035 |
| wash_sale_score | bull | 0.0284 | 0.1744 | 0.1629 | 0.1133 | 0.0907 |
| fund_profit_growth | bull | 0.0220 | 0.1660 | 0.1327 | 0.1818 | 0.0784 |
| mom_x_lowvol_20_20 | bear | 0.1556 | 0.2079 | 0.7482 | 0.4795 | 0.5535 |
| momentum_reversal | bear | 0.1501 | 0.2275 | 0.6597 | 0.3973 | 0.4609 |
| rsi_vol_combo | bear | 0.1162 | 0.2399 | 0.4845 | 0.3425 | 0.3252 |
| turnover_stability | bear | 0.1056 | 0.2137 | 0.4943 | 0.2329 | 0.3047 |
| trend_lowvol | bear | 0.1038 | 0.2370 | 0.4381 | 0.3425 | 0.2941 |
| fund_pe | bear | 0.1074 | 0.3058 | 0.3512 | 0.2329 | 0.2165 |

### 抖音概念(字节概念)

- **Neutral**: ['momentum_reversal', 'trend_lowvol'] (单因子IC=0.1171, 组合IC=0.131)
  - weights: [0.5074, 0.4926]
- **Bull**: ['volatility', 'trend_lowvol'] (单因子IC=0.1148, 组合IC=0.1245)
  - bull_weights: [0.5449, 0.4551]
- **Bear**: ['trend_lowvol', 'momentum_reversal', 'mom_x_lowvol_20_20'] (单因子IC=0.1243, 组合IC=0.1452)
  - bear_weights: [0.4085, 0.2983, 0.2932]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| momentum_reversal | neutral | 0.1164 | 0.2063 | 0.5645 | 0.4217 | 0.4013 |
| trend_lowvol | neutral | 0.1178 | 0.2134 | 0.5520 | 0.4113 | 0.3895 |
| mom_x_lowvol_20_20 | neutral | 0.1096 | 0.2069 | 0.5297 | 0.4092 | 0.3732 |
| volatility | neutral | 0.1086 | 0.2255 | 0.4816 | 0.3820 | 0.3328 |
| rsi_vol_combo | neutral | 0.0928 | 0.2005 | 0.4630 | 0.3215 | 0.3059 |
| fund_pb | neutral | 0.0835 | 0.1968 | 0.4246 | 0.3612 | 0.2890 |
| low_downside | neutral | 0.0605 | 0.2072 | 0.2923 | 0.2985 | 0.1898 |
| turnover_stability | neutral | 0.0297 | 0.1325 | 0.2238 | 0.2088 | 0.1353 |
| fund_pe | neutral | 0.0440 | 0.1968 | 0.2238 | 0.1461 | 0.1282 |
| fund_profit_growth | neutral | 0.0333 | 0.1487 | 0.2238 | 0.1211 | 0.1254 |
| volatility | bull | 0.1239 | 0.1951 | 0.6352 | 0.4924 | 0.4740 |
| trend_lowvol | bull | 0.1057 | 0.1993 | 0.5305 | 0.4924 | 0.3959 |
| low_downside | bull | 0.0910 | 0.1707 | 0.5328 | 0.4242 | 0.3794 |
| fund_pb | bull | 0.0875 | 0.1668 | 0.5243 | 0.3333 | 0.3495 |
| fund_pe | bull | 0.0559 | 0.1614 | 0.3464 | 0.2045 | 0.2087 |
| mom_x_lowvol_20_20 | bull | 0.0628 | 0.1918 | 0.3274 | 0.2500 | 0.2046 |
| fund_profit_growth | bull | 0.0501 | 0.1555 | 0.3220 | 0.2576 | 0.2025 |
| fund_score | bull | 0.0535 | 0.1792 | 0.2983 | 0.2652 | 0.1887 |
| momentum_reversal | bull | 0.0555 | 0.1829 | 0.3033 | 0.2273 | 0.1861 |
| fund_roe | bull | 0.0429 | 0.1575 | 0.2723 | 0.3030 | 0.1774 |
| turnover_stability | bull | 0.0261 | 0.1174 | 0.2220 | 0.2045 | 0.1337 |
| rsi_vol_combo | bull | 0.0290 | 0.1812 | 0.1599 | 0.1515 | 0.0921 |
| wash_sale_score | bull | 0.0200 | 0.1311 | 0.1529 | 0.1383 | 0.0871 |
| trend_lowvol | bear | 0.1276 | 0.1837 | 0.6948 | 0.5616 | 0.5425 |
| momentum_reversal | bear | 0.1233 | 0.2216 | 0.5562 | 0.4247 | 0.3962 |
| mom_x_lowvol_20_20 | bear | 0.1220 | 0.2275 | 0.5363 | 0.4521 | 0.3894 |
| fund_profit_growth | bear | 0.0922 | 0.1843 | 0.5002 | 0.3973 | 0.3494 |
| fund_revenue_growth | bear | 0.0652 | 0.1281 | 0.5086 | 0.2603 | 0.3205 |
| fund_score | bear | 0.0680 | 0.1875 | 0.3627 | 0.2877 | 0.2335 |
| rsi_vol_combo | bear | 0.0618 | 0.2018 | 0.3061 | 0.3151 | 0.2012 |
| turnover_stability | bear | 0.0351 | 0.1161 | 0.3022 | 0.3151 | 0.1987 |
| bb_width_20 | bear | 0.0404 | 0.1703 | 0.2371 | 0.3973 | 0.1656 |

### 抽水蓄能

- **Neutral**: ['momentum_reversal', 'fund_profit_growth', 'trend_lowvol'] (单因子IC=0.07, 组合IC=0.0874)
  - weights: [0.3792, 0.3304, 0.2904]
- **Bull**: ['turnover_stability', 'fund_pe', 'fund_pb'] (单因子IC=0.0587, 组合IC=0.0594)
  - bull_weights: [0.3445, 0.3315, 0.324]
- **Bear**: ['mom_x_lowvol_20_20'] (单因子IC=0.1253, 组合IC=0.1253)
  - bear_weights: [1.0]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| momentum_reversal | neutral | 0.0804 | 0.2536 | 0.3170 | 0.2463 | 0.1976 |
| fund_profit_growth | neutral | 0.0574 | 0.2012 | 0.2853 | 0.2067 | 0.1721 |
| trend_lowvol | neutral | 0.0721 | 0.2834 | 0.2543 | 0.1900 | 0.1513 |
| turnover_stability | neutral | 0.0494 | 0.1991 | 0.2479 | 0.1983 | 0.1485 |
| mom_x_lowvol_20_20 | neutral | 0.0586 | 0.2352 | 0.2492 | 0.1775 | 0.1467 |
| rsi_vol_combo | neutral | 0.0535 | 0.2401 | 0.2229 | 0.1733 | 0.1308 |
| volatility | neutral | 0.0550 | 0.2520 | 0.2182 | 0.1743 | 0.1281 |
| fund_pe | neutral | 0.0587 | 0.2662 | 0.2206 | 0.1587 | 0.1278 |
| fund_pb | neutral | 0.0480 | 0.2525 | 0.1900 | 0.1441 | 0.1087 |
| fund_score | neutral | 0.0368 | 0.2292 | 0.1604 | 0.1816 | 0.0948 |
| low_downside | neutral | 0.0250 | 0.2604 | 0.0960 | 0.1013 | 0.0529 |
| turnover_stability | bull | 0.0520 | 0.1870 | 0.2783 | 0.2045 | 0.1676 |
| fund_pe | bull | 0.0615 | 0.2212 | 0.2782 | 0.1591 | 0.1612 |
| fund_pb | bull | 0.0624 | 0.2309 | 0.2701 | 0.1667 | 0.1576 |
| low_downside | bull | 0.0511 | 0.2337 | 0.2186 | 0.2576 | 0.1375 |
| stroke_phase | bull | 0.0296 | 0.1907 | 0.1552 | 0.1212 | 0.0870 |
| mom_x_lowvol_20_20 | bear | 0.1253 | 0.2526 | 0.4961 | 0.5342 | 0.3806 |
| rsi_vol_combo | bear | 0.1043 | 0.2459 | 0.4241 | 0.4247 | 0.3021 |
| momentum_reversal | bear | 0.1179 | 0.2886 | 0.4086 | 0.3699 | 0.2799 |
| fund_profit_growth | bear | 0.0491 | 0.2017 | 0.2435 | 0.2329 | 0.1501 |

### 拼多多概念

- **Neutral**: ['momentum_reversal', 'mom_x_lowvol_20_20', 'trend_lowvol'] (单因子IC=0.1226, 组合IC=0.1413)
  - weights: [0.3809, 0.3403, 0.2789]
- **Bull**: ['volatility', 'fund_pb', 'low_downside'] (单因子IC=0.1187, 组合IC=0.1538)
  - bull_weights: [0.364, 0.3442, 0.2918]
- **Bear**: ['trend_lowvol', 'mom_x_lowvol_20_20'] (单因子IC=0.1194, 组合IC=0.127)
  - bear_weights: [0.5579, 0.4421]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| momentum_reversal | neutral | 0.1341 | 0.2788 | 0.4811 | 0.3716 | 0.3299 |
| mom_x_lowvol_20_20 | neutral | 0.1263 | 0.2877 | 0.4391 | 0.3424 | 0.2947 |
| trend_lowvol | neutral | 0.1073 | 0.2840 | 0.3778 | 0.2787 | 0.2416 |
| volatility | neutral | 0.1078 | 0.2947 | 0.3658 | 0.2589 | 0.2302 |
| rsi_vol_combo | neutral | 0.0993 | 0.2743 | 0.3618 | 0.2693 | 0.2296 |
| fund_pb | neutral | 0.0892 | 0.2868 | 0.3109 | 0.2338 | 0.1918 |
| turnover_stability | neutral | 0.0507 | 0.2151 | 0.2358 | 0.1712 | 0.1381 |
| low_downside | neutral | 0.0498 | 0.2817 | 0.1766 | 0.1253 | 0.0994 |
| fund_profit_growth | neutral | 0.0147 | 0.2290 | 0.0643 | 0.1086 | 0.0356 |
| volatility | bull | 0.1309 | 0.2755 | 0.4750 | 0.4167 | 0.3365 |
| fund_pb | bull | 0.1283 | 0.2718 | 0.4718 | 0.3485 | 0.3181 |
| low_downside | bull | 0.0969 | 0.2462 | 0.3933 | 0.3712 | 0.2697 |
| fund_roe | bull | 0.0519 | 0.1663 | 0.3119 | 0.2197 | 0.1902 |
| mom_x_lowvol_20_20 | bull | 0.0661 | 0.2651 | 0.2493 | 0.2121 | 0.1511 |
| trend_lowvol | bull | 0.0680 | 0.2843 | 0.2390 | 0.1894 | 0.1421 |
| momentum_reversal | bull | 0.0607 | 0.2620 | 0.2315 | 0.1136 | 0.1289 |
| fund_pe | bull | 0.0456 | 0.2503 | 0.1822 | 0.2121 | 0.1104 |
| fund_gross_margin | bull | 0.0404 | 0.2247 | 0.1796 | 0.1667 | 0.1048 |
| fund_score | bull | 0.0288 | 0.2167 | 0.1331 | 0.1439 | 0.0761 |
| trend_lowvol | bear | 0.1165 | 0.1987 | 0.5861 | 0.5068 | 0.4416 |
| mom_x_lowvol_20_20 | bear | 0.1223 | 0.2251 | 0.5435 | 0.2877 | 0.3499 |
| momentum_reversal | bear | 0.1042 | 0.1993 | 0.5230 | 0.2877 | 0.3367 |
| rsi_vol_combo | bear | 0.0845 | 0.2095 | 0.4032 | 0.3425 | 0.2707 |
| turnover_stability | bear | 0.0769 | 0.2073 | 0.3708 | 0.3699 | 0.2540 |

### 换电概念

- **Neutral**: ['momentum_reversal', 'volatility', 'fund_pb'] (单因子IC=0.083, 组合IC=0.1221)
  - weights: [0.3621, 0.3375, 0.3003]
- **Bull**: ['volatility', 'low_downside', 'fund_gross_margin'] (单因子IC=0.0685, 组合IC=0.092)
  - bull_weights: [0.3761, 0.3391, 0.2848]
- **Bear**: ['momentum_reversal', 'rsi_vol_combo', 'trend_lowvol'] (单因子IC=0.1212, 组合IC=0.1303)
  - bear_weights: [0.3857, 0.3311, 0.2832]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| momentum_reversal | neutral | 0.0820 | 0.2144 | 0.3825 | 0.3006 | 0.2488 |
| volatility | neutral | 0.0885 | 0.2410 | 0.3675 | 0.2620 | 0.2319 |
| fund_pb | neutral | 0.0783 | 0.2374 | 0.3299 | 0.2505 | 0.2063 |
| fund_score | neutral | 0.0667 | 0.2022 | 0.3297 | 0.2213 | 0.2013 |
| mom_x_lowvol_20_20 | neutral | 0.0690 | 0.2121 | 0.3254 | 0.2349 | 0.2009 |
| rsi_vol_combo | neutral | 0.0688 | 0.2201 | 0.3126 | 0.2568 | 0.1964 |
| fund_profit_growth | neutral | 0.0534 | 0.1704 | 0.3134 | 0.2505 | 0.1960 |
| trend_lowvol | neutral | 0.0734 | 0.2480 | 0.2959 | 0.1983 | 0.1773 |
| fund_roe | neutral | 0.0426 | 0.2157 | 0.1974 | 0.1284 | 0.1114 |
| low_downside | neutral | 0.0462 | 0.2403 | 0.1922 | 0.1441 | 0.1099 |
| fund_revenue_growth | neutral | 0.0354 | 0.1856 | 0.1908 | 0.1347 | 0.1082 |
| turnover_stability | neutral | 0.0194 | 0.1941 | 0.1000 | 0.1106 | 0.0555 |
| fund_gross_margin | neutral | 0.0180 | 0.1980 | 0.0911 | 0.1253 | 0.0512 |
| volatility | bull | 0.0868 | 0.2115 | 0.4106 | 0.3333 | 0.2737 |
| low_downside | bull | 0.0709 | 0.1893 | 0.3744 | 0.3182 | 0.2467 |
| fund_gross_margin | bull | 0.0479 | 0.1523 | 0.3145 | 0.3182 | 0.2073 |
| fund_pb | bull | 0.0643 | 0.2335 | 0.2755 | 0.1667 | 0.1607 |
| trend_lowvol | bull | 0.0389 | 0.1994 | 0.1951 | 0.1212 | 0.1094 |
| fund_pe | bull | 0.0403 | 0.2412 | 0.1672 | 0.1515 | 0.0962 |
| fund_roe | bull | 0.0347 | 0.2090 | 0.1659 | 0.1212 | 0.0930 |
| fund_revenue_growth | bull | 0.0255 | 0.1737 | 0.1469 | 0.1439 | 0.0840 |
| mom_x_lowvol_20_20 | bull | 0.0246 | 0.1892 | 0.1301 | 0.1061 | 0.0719 |
| stroke_phase | bull | 0.0207 | 0.1640 | 0.1260 | 0.1136 | 0.0702 |
| momentum_reversal | bull | 0.0237 | 0.1927 | 0.1231 | 0.1288 | 0.0695 |
| momentum_reversal | bear | 0.1177 | 0.1967 | 0.5986 | 0.5068 | 0.4510 |
| rsi_vol_combo | bear | 0.1222 | 0.2291 | 0.5332 | 0.4521 | 0.3871 |
| trend_lowvol | bear | 0.1238 | 0.2613 | 0.4739 | 0.3973 | 0.3311 |
| fund_profit_growth | bear | 0.0760 | 0.1683 | 0.4512 | 0.4247 | 0.3214 |
| mom_x_lowvol_20_20 | bear | 0.0932 | 0.2024 | 0.4603 | 0.3425 | 0.3090 |
| fund_score | bear | 0.0660 | 0.2078 | 0.3174 | 0.3425 | 0.2130 |
| fund_revenue_growth | bear | 0.0383 | 0.1470 | 0.2606 | 0.1233 | 0.1464 |
| fund_roe | bear | 0.0428 | 0.1913 | 0.2239 | 0.2192 | 0.1365 |
| volatility | bear | 0.0497 | 0.2578 | 0.1929 | 0.1233 | 0.1083 |

### 数字孪生

- **Neutral**: ['mom_x_lowvol_20_20'] (单因子IC=0.1061, 组合IC=0.1061)
  - weights: [1.0]
- **Bull**: ['volatility', 'low_downside', 'mom_x_lowvol_20_20'] (单因子IC=0.0943, 组合IC=0.1112)
  - bull_weights: [0.3469, 0.3289, 0.3242]
- **Bear**: ['trend_lowvol', 'fund_profit_growth', 'mom_x_lowvol_20_20'] (单因子IC=0.1088, 组合IC=0.1644)
  - bear_weights: [0.372, 0.337, 0.2909]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| mom_x_lowvol_20_20 | neutral | 0.1061 | 0.1803 | 0.5888 | 0.4238 | 0.4192 |
| momentum_reversal | neutral | 0.1019 | 0.1771 | 0.5751 | 0.4280 | 0.4106 |
| rsi_vol_combo | neutral | 0.0813 | 0.1658 | 0.4903 | 0.3716 | 0.3362 |
| trend_lowvol | neutral | 0.0900 | 0.1990 | 0.4525 | 0.3967 | 0.3160 |
| fund_pb | neutral | 0.0830 | 0.1779 | 0.4663 | 0.3340 | 0.3110 |
| volatility | neutral | 0.0617 | 0.2020 | 0.3054 | 0.2589 | 0.1922 |
| fund_profit_growth | neutral | 0.0418 | 0.1526 | 0.2737 | 0.1670 | 0.1597 |
| turnover_stability | neutral | 0.0312 | 0.1407 | 0.2217 | 0.2150 | 0.1347 |
| fund_revenue_growth | neutral | 0.0312 | 0.1382 | 0.2259 | 0.1858 | 0.1339 |
| fund_score | neutral | 0.0399 | 0.1765 | 0.2260 | 0.1524 | 0.1302 |
| fund_pe | neutral | 0.0294 | 0.1802 | 0.1633 | 0.1044 | 0.0902 |
| low_downside | neutral | 0.0255 | 0.1964 | 0.1296 | 0.1649 | 0.0755 |
| volatility | bull | 0.1050 | 0.1815 | 0.5786 | 0.3939 | 0.4033 |
| low_downside | bull | 0.0889 | 0.1586 | 0.5608 | 0.3636 | 0.3823 |
| mom_x_lowvol_20_20 | bull | 0.0888 | 0.1704 | 0.5210 | 0.4470 | 0.3769 |
| momentum_reversal | bull | 0.0850 | 0.1669 | 0.5094 | 0.3939 | 0.3550 |
| fund_pb | bull | 0.0813 | 0.1599 | 0.5085 | 0.2955 | 0.3294 |
| trend_lowvol | bull | 0.0917 | 0.1884 | 0.4868 | 0.3258 | 0.3227 |
| rsi_vol_combo | bull | 0.0686 | 0.1558 | 0.4403 | 0.3030 | 0.2868 |
| fund_score | bull | 0.0465 | 0.1719 | 0.2703 | 0.2045 | 0.1628 |
| fund_profit_growth | bull | 0.0350 | 0.1379 | 0.2541 | 0.1818 | 0.1502 |
| fund_roe | bull | 0.0364 | 0.1737 | 0.2096 | 0.1288 | 0.1183 |
| turnover_stability | bull | 0.0267 | 0.1342 | 0.1987 | 0.1667 | 0.1159 |
| fund_revenue_growth | bull | 0.0311 | 0.1591 | 0.1957 | 0.1439 | 0.1119 |
| stroke_phase | bull | 0.0198 | 0.1371 | 0.1443 | 0.1742 | 0.0847 |
| fund_pe | bull | 0.0190 | 0.1815 | 0.1046 | 0.1818 | 0.0618 |
| trend_lowvol | bear | 0.1266 | 0.1817 | 0.6971 | 0.4795 | 0.5157 |
| fund_profit_growth | bear | 0.1012 | 0.1573 | 0.6434 | 0.4521 | 0.4671 |
| mom_x_lowvol_20_20 | bear | 0.0984 | 0.1739 | 0.5660 | 0.4247 | 0.4032 |
| fund_revenue_growth | bear | 0.0535 | 0.1177 | 0.4547 | 0.4247 | 0.3239 |
| momentum_reversal | bear | 0.0763 | 0.1797 | 0.4247 | 0.3425 | 0.2851 |
| fund_score | bear | 0.0666 | 0.1680 | 0.3968 | 0.2877 | 0.2555 |
| fund_gross_margin | bear | 0.0433 | 0.1483 | 0.2922 | 0.1233 | 0.1641 |

### 数字经济

- **Neutral**: ['momentum_reversal', 'trend_lowvol'] (单因子IC=0.1086, 组合IC=0.1181)
  - weights: [0.5224, 0.4776]
- **Bull**: ['volatility', 'low_downside', 'fund_pb'] (单因子IC=0.129, 组合IC=0.1554)
  - bull_weights: [0.3915, 0.3308, 0.2777]
- **Bear**: ['trend_lowvol', 'momentum_reversal', 'mom_x_lowvol_20_20'] (单因子IC=0.1141, 组合IC=0.1266)
  - bear_weights: [0.405, 0.301, 0.294]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| momentum_reversal | neutral | 0.1054 | 0.1616 | 0.6525 | 0.4509 | 0.4733 |
| trend_lowvol | neutral | 0.1118 | 0.1847 | 0.6051 | 0.4301 | 0.4327 |
| mom_x_lowvol_20_20 | neutral | 0.0974 | 0.1628 | 0.5982 | 0.4134 | 0.4227 |
| rsi_vol_combo | neutral | 0.0851 | 0.1466 | 0.5805 | 0.4363 | 0.4169 |
| volatility | neutral | 0.0857 | 0.1906 | 0.4495 | 0.3299 | 0.2989 |
| fund_profit_growth | neutral | 0.0486 | 0.1098 | 0.4423 | 0.3006 | 0.2876 |
| turnover_stability | neutral | 0.0502 | 0.1172 | 0.4283 | 0.3424 | 0.2875 |
| fund_pb | neutral | 0.0761 | 0.1734 | 0.4389 | 0.3006 | 0.2854 |
| fund_score | neutral | 0.0524 | 0.1418 | 0.3696 | 0.2985 | 0.2400 |
| low_downside | neutral | 0.0444 | 0.1775 | 0.2503 | 0.2777 | 0.1599 |
| fund_roe | neutral | 0.0379 | 0.1563 | 0.2423 | 0.2067 | 0.1462 |
| fund_revenue_growth | neutral | 0.0244 | 0.1080 | 0.2259 | 0.1482 | 0.1297 |
| fund_pe | neutral | 0.0373 | 0.1726 | 0.2161 | 0.1399 | 0.1232 |
| volatility | bull | 0.1480 | 0.1610 | 0.9191 | 0.6364 | 0.7520 |
| low_downside | bull | 0.1185 | 0.1469 | 0.8065 | 0.5758 | 0.6354 |
| fund_pb | bull | 0.1206 | 0.1695 | 0.7112 | 0.5000 | 0.5334 |
| trend_lowvol | bull | 0.1108 | 0.1848 | 0.5995 | 0.4318 | 0.4292 |
| mom_x_lowvol_20_20 | bull | 0.0954 | 0.1716 | 0.5557 | 0.3788 | 0.3831 |
| momentum_reversal | bull | 0.0914 | 0.1663 | 0.5497 | 0.3636 | 0.3748 |
| fund_pe | bull | 0.0886 | 0.1676 | 0.5286 | 0.3864 | 0.3664 |
| turnover_stability | bull | 0.0422 | 0.1046 | 0.4029 | 0.3333 | 0.2686 |
| rsi_vol_combo | bull | 0.0518 | 0.1587 | 0.3265 | 0.2500 | 0.2041 |
| fund_roe | bull | 0.0391 | 0.1505 | 0.2595 | 0.1894 | 0.1543 |
| fund_profit_growth | bull | 0.0277 | 0.1140 | 0.2427 | 0.1364 | 0.1379 |
| fund_score | bull | 0.0320 | 0.1396 | 0.2295 | 0.1818 | 0.1356 |
| fund_revenue_growth | bull | 0.0177 | 0.1016 | 0.1739 | 0.1515 | 0.1001 |
| trend_lowvol | bear | 0.1181 | 0.1383 | 0.8537 | 0.6164 | 0.6900 |
| momentum_reversal | bear | 0.1160 | 0.1735 | 0.6684 | 0.5342 | 0.5128 |
| mom_x_lowvol_20_20 | bear | 0.1082 | 0.1598 | 0.6770 | 0.4795 | 0.5008 |
| fund_profit_growth | bear | 0.0734 | 0.1127 | 0.6511 | 0.5068 | 0.4906 |
| fund_score | bear | 0.0702 | 0.1367 | 0.5135 | 0.4795 | 0.3798 |
| rsi_vol_combo | bear | 0.0772 | 0.1647 | 0.4684 | 0.4247 | 0.3336 |
| turnover_stability | bear | 0.0537 | 0.1077 | 0.4985 | 0.3151 | 0.3278 |
| fund_revenue_growth | bear | 0.0594 | 0.1182 | 0.5024 | 0.2877 | 0.3235 |
| fund_roe | bear | 0.0431 | 0.1645 | 0.2620 | 0.1233 | 0.1471 |

### 数字货币

- **Neutral**: ['trend_lowvol', 'momentum_reversal', 'fund_pb'] (单因子IC=0.0854, 组合IC=0.1159)
  - weights: [0.3589, 0.3273, 0.3138]
- **Bull**: ['volatility', 'low_downside', 'fund_pb'] (单因子IC=0.1232, 组合IC=0.1468)
  - bull_weights: [0.366, 0.3616, 0.2724]
- **Bear**: ['mom_x_lowvol_20_20'] (单因子IC=0.1249, 组合IC=0.1249)
  - bear_weights: [1.0]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| trend_lowvol | neutral | 0.0987 | 0.1931 | 0.5110 | 0.3841 | 0.3537 |
| momentum_reversal | neutral | 0.0896 | 0.1878 | 0.4769 | 0.3528 | 0.3226 |
| fund_pb | neutral | 0.0679 | 0.1462 | 0.4643 | 0.3319 | 0.3092 |
| mom_x_lowvol_20_20 | neutral | 0.0787 | 0.1900 | 0.4144 | 0.3048 | 0.2703 |
| rsi_vol_combo | neutral | 0.0705 | 0.1762 | 0.4004 | 0.3152 | 0.2633 |
| fund_score | neutral | 0.0466 | 0.1542 | 0.3024 | 0.2025 | 0.1818 |
| fund_profit_growth | neutral | 0.0394 | 0.1453 | 0.2714 | 0.2109 | 0.1643 |
| volatility | neutral | 0.0588 | 0.2191 | 0.2686 | 0.2129 | 0.1629 |
| turnover_stability | neutral | 0.0357 | 0.1405 | 0.2541 | 0.2380 | 0.1573 |
| fund_pe | neutral | 0.0467 | 0.1994 | 0.2340 | 0.1649 | 0.1363 |
| fund_revenue_growth | neutral | 0.0251 | 0.1267 | 0.1980 | 0.1482 | 0.1137 |
| low_downside | neutral | 0.0371 | 0.1997 | 0.1858 | 0.1524 | 0.1070 |
| fund_roe | neutral | 0.0273 | 0.1702 | 0.1607 | 0.1378 | 0.0914 |
| volatility | bull | 0.1437 | 0.2018 | 0.7121 | 0.5530 | 0.5530 |
| low_downside | bull | 0.1343 | 0.1938 | 0.6933 | 0.5758 | 0.5463 |
| fund_pb | bull | 0.0916 | 0.1610 | 0.5688 | 0.4470 | 0.4115 |
| trend_lowvol | bull | 0.1125 | 0.2135 | 0.5271 | 0.4318 | 0.3773 |
| fund_pe | bull | 0.0721 | 0.1641 | 0.4394 | 0.3712 | 0.3013 |
| mom_x_lowvol_20_20 | bull | 0.0878 | 0.1958 | 0.4486 | 0.3333 | 0.2991 |
| momentum_reversal | bull | 0.0843 | 0.1943 | 0.4338 | 0.3409 | 0.2908 |
| fund_revenue_growth | bull | 0.0512 | 0.1370 | 0.3740 | 0.3030 | 0.2436 |
| turnover_stability | bull | 0.0412 | 0.1421 | 0.2897 | 0.2273 | 0.1778 |
| rsi_vol_combo | bull | 0.0440 | 0.1934 | 0.2277 | 0.1591 | 0.1320 |
| fund_profit_growth | bull | 0.0278 | 0.1525 | 0.1823 | 0.1212 | 0.1022 |
| fund_score | bull | 0.0266 | 0.1588 | 0.1676 | 0.1288 | 0.0946 |
| mom_x_lowvol_20_20 | bear | 0.1249 | 0.1714 | 0.7288 | 0.5068 | 0.5491 |
| trend_lowvol | bear | 0.0989 | 0.1697 | 0.5830 | 0.5616 | 0.4552 |
| momentum_reversal | bear | 0.1004 | 0.1722 | 0.5830 | 0.3699 | 0.3993 |
| fund_score | bear | 0.0841 | 0.1814 | 0.4634 | 0.3973 | 0.3237 |
| fund_revenue_growth | bear | 0.0648 | 0.1357 | 0.4774 | 0.2877 | 0.3074 |
| fund_profit_growth | bear | 0.0694 | 0.1731 | 0.4008 | 0.2603 | 0.2526 |
| rsi_vol_combo | bear | 0.0462 | 0.1802 | 0.2563 | 0.2055 | 0.1545 |

### 数据中心

- **Neutral**: ['momentum_reversal', 'trend_lowvol'] (单因子IC=0.088, 组合IC=0.0965)
  - weights: [0.5141, 0.4859]
- **Bull**: ['volatility', 'low_downside', 'fund_pb'] (单因子IC=0.093, 组合IC=0.1118)
  - bull_weights: [0.3733, 0.3319, 0.2948]
- **Bear**: ['trend_lowvol', 'mom_x_lowvol_20_20'] (单因子IC=0.1319, 组合IC=0.1467)
  - bear_weights: [0.5061, 0.4939]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| momentum_reversal | neutral | 0.0839 | 0.1670 | 0.5023 | 0.3633 | 0.3424 |
| trend_lowvol | neutral | 0.0922 | 0.1927 | 0.4785 | 0.3528 | 0.3237 |
| mom_x_lowvol_20_20 | neutral | 0.0781 | 0.1631 | 0.4790 | 0.3382 | 0.3205 |
| fund_pb | neutral | 0.0736 | 0.1629 | 0.4517 | 0.3132 | 0.2966 |
| turnover_stability | neutral | 0.0349 | 0.0925 | 0.3769 | 0.3194 | 0.2486 |
| rsi_vol_combo | neutral | 0.0639 | 0.1648 | 0.3878 | 0.2756 | 0.2473 |
| volatility | neutral | 0.0739 | 0.1951 | 0.3788 | 0.2965 | 0.2456 |
| fund_profit_growth | neutral | 0.0443 | 0.1279 | 0.3461 | 0.2589 | 0.2179 |
| fund_pe | neutral | 0.0529 | 0.1595 | 0.3319 | 0.2463 | 0.2068 |
| fund_revenue_growth | neutral | 0.0332 | 0.1340 | 0.2478 | 0.1921 | 0.1477 |
| fund_score | neutral | 0.0443 | 0.1778 | 0.2493 | 0.1670 | 0.1455 |
| low_downside | neutral | 0.0443 | 0.1975 | 0.2241 | 0.2213 | 0.1368 |
| fund_roe | neutral | 0.0326 | 0.1825 | 0.1784 | 0.1336 | 0.1011 |
| volatility | bull | 0.0998 | 0.1566 | 0.6371 | 0.4773 | 0.4706 |
| low_downside | bull | 0.0921 | 0.1634 | 0.5635 | 0.4848 | 0.4184 |
| fund_pb | bull | 0.0872 | 0.1591 | 0.5480 | 0.3561 | 0.3716 |
| fund_pe | bull | 0.0698 | 0.1555 | 0.4492 | 0.2879 | 0.2893 |
| fund_profit_growth | bull | 0.0416 | 0.1050 | 0.3966 | 0.2803 | 0.2539 |
| turnover_stability | bull | 0.0357 | 0.0917 | 0.3888 | 0.2424 | 0.2415 |
| trend_lowvol | bull | 0.0592 | 0.1946 | 0.3040 | 0.2652 | 0.1923 |
| fund_score | bull | 0.0448 | 0.1591 | 0.2818 | 0.2197 | 0.1718 |
| fund_roe | bull | 0.0419 | 0.1600 | 0.2620 | 0.2045 | 0.1578 |
| mom_x_lowvol_20_20 | bull | 0.0363 | 0.1397 | 0.2600 | 0.2045 | 0.1566 |
| momentum_reversal | bull | 0.0297 | 0.1464 | 0.2031 | 0.1439 | 0.1162 |
| stroke_phase | bull | 0.0183 | 0.0935 | 0.1954 | 0.1288 | 0.1103 |
| fund_revenue_growth | bull | 0.0248 | 0.1312 | 0.1892 | 0.1136 | 0.1054 |
| fund_gross_margin | bull | 0.0113 | 0.0804 | 0.1405 | 0.1591 | 0.0814 |
| trend_lowvol | bear | 0.1289 | 0.1570 | 0.8209 | 0.5068 | 0.6185 |
| mom_x_lowvol_20_20 | bear | 0.1350 | 0.1746 | 0.7730 | 0.5616 | 0.6035 |
| momentum_reversal | bear | 0.1220 | 0.1759 | 0.6935 | 0.4795 | 0.5130 |
| rsi_vol_combo | bear | 0.0686 | 0.1422 | 0.4823 | 0.3151 | 0.3171 |

### 数据安全

- **Neutral**: ['momentum_reversal', 'mom_x_lowvol_20_20', 'rsi_vol_combo'] (单因子IC=0.0966, 组合IC=0.1046)
  - weights: [0.3553, 0.3232, 0.3215]
- **Bull**: ['volatility', 'fund_pb', 'mom_x_lowvol_20_20'] (单因子IC=0.1229, 组合IC=0.1587)
  - bull_weights: [0.4019, 0.3289, 0.2691]
- **Bear**: ['mom_x_lowvol_20_20', 'fund_revenue_growth', 'fund_profit_growth'] (单因子IC=0.0804, 组合IC=0.1148)
  - bear_weights: [0.3618, 0.3281, 0.31]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| momentum_reversal | neutral | 0.1023 | 0.1917 | 0.5338 | 0.4217 | 0.3794 |
| mom_x_lowvol_20_20 | neutral | 0.0951 | 0.1907 | 0.4988 | 0.3841 | 0.3452 |
| rsi_vol_combo | neutral | 0.0923 | 0.1869 | 0.4939 | 0.3904 | 0.3434 |
| trend_lowvol | neutral | 0.0949 | 0.2042 | 0.4645 | 0.3528 | 0.3142 |
| fund_pb | neutral | 0.0742 | 0.1680 | 0.4418 | 0.3278 | 0.2933 |
| fund_profit_growth | neutral | 0.0558 | 0.1380 | 0.4044 | 0.3246 | 0.2679 |
| fund_score | neutral | 0.0589 | 0.1659 | 0.3550 | 0.2777 | 0.2268 |
| volatility | neutral | 0.0733 | 0.2119 | 0.3457 | 0.3006 | 0.2248 |
| fund_roe | neutral | 0.0356 | 0.1922 | 0.1851 | 0.1587 | 0.1072 |
| turnover_stability | neutral | 0.0253 | 0.1445 | 0.1751 | 0.1461 | 0.1004 |
| fund_pe | neutral | 0.0312 | 0.1920 | 0.1624 | 0.1127 | 0.0903 |
| low_downside | neutral | 0.0296 | 0.1969 | 0.1503 | 0.1023 | 0.0828 |
| volatility | bull | 0.1502 | 0.1802 | 0.8338 | 0.5227 | 0.6349 |
| fund_pb | bull | 0.1075 | 0.1568 | 0.6858 | 0.5152 | 0.5195 |
| mom_x_lowvol_20_20 | bull | 0.1110 | 0.1870 | 0.5938 | 0.4318 | 0.4251 |
| low_downside | bull | 0.0943 | 0.1650 | 0.5717 | 0.4091 | 0.4028 |
| trend_lowvol | bull | 0.1101 | 0.1974 | 0.5578 | 0.3409 | 0.3740 |
| fund_score | bull | 0.0796 | 0.1585 | 0.5020 | 0.4015 | 0.3518 |
| momentum_reversal | bull | 0.0974 | 0.1898 | 0.5130 | 0.3561 | 0.3478 |
| turnover_stability | bull | 0.0634 | 0.1337 | 0.4742 | 0.3636 | 0.3233 |
| fund_roe | bull | 0.0700 | 0.1664 | 0.4204 | 0.3030 | 0.2739 |
| fund_pe | bull | 0.0847 | 0.2187 | 0.3875 | 0.3258 | 0.2569 |
| fund_profit_growth | bull | 0.0596 | 0.1487 | 0.4010 | 0.2197 | 0.2446 |
| rsi_vol_combo | bull | 0.0583 | 0.1854 | 0.3146 | 0.2576 | 0.1978 |
| fund_revenue_growth | bull | 0.0307 | 0.1574 | 0.1951 | 0.1136 | 0.1086 |
| mom_x_lowvol_20_20 | bear | 0.0911 | 0.1890 | 0.4818 | 0.3425 | 0.3234 |
| fund_revenue_growth | bear | 0.0733 | 0.1608 | 0.4555 | 0.2877 | 0.2933 |
| fund_profit_growth | bear | 0.0767 | 0.1669 | 0.4598 | 0.2055 | 0.2771 |
| momentum_reversal | bear | 0.0603 | 0.1792 | 0.3363 | 0.2603 | 0.2119 |
| trend_lowvol | bear | 0.0531 | 0.1834 | 0.2897 | 0.3151 | 0.1905 |

### 数据确权

- **Neutral**: ['fund_pb', 'momentum_reversal', 'rsi_vol_combo'] (单因子IC=0.1038, 组合IC=0.1548)
  - weights: [0.3804, 0.3254, 0.2942]
- **Bull**: ['fund_pb', 'volatility', 'mom_x_lowvol_20_20'] (单因子IC=0.1672, 组合IC=0.21)
  - bull_weights: [0.4137, 0.3527, 0.2336]
- **Bear**: ['trend_lowvol'] (单因子IC=0.169, 组合IC=0.169)
  - bear_weights: [1.0]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| fund_pb | neutral | 0.1038 | 0.2044 | 0.5080 | 0.4217 | 0.3611 |
| momentum_reversal | neutral | 0.1103 | 0.2390 | 0.4615 | 0.3382 | 0.3088 |
| rsi_vol_combo | neutral | 0.0972 | 0.2358 | 0.4122 | 0.3549 | 0.2792 |
| trend_lowvol | neutral | 0.0983 | 0.2354 | 0.4173 | 0.3006 | 0.2714 |
| mom_x_lowvol_20_20 | neutral | 0.0948 | 0.2348 | 0.4036 | 0.3184 | 0.2660 |
| volatility | neutral | 0.0935 | 0.2529 | 0.3697 | 0.2620 | 0.2333 |
| fund_pe | neutral | 0.0795 | 0.2498 | 0.3181 | 0.2422 | 0.1976 |
| fund_roe | neutral | 0.0546 | 0.2456 | 0.2224 | 0.1566 | 0.1286 |
| low_downside | neutral | 0.0484 | 0.2316 | 0.2091 | 0.1754 | 0.1229 |
| fund_pb | bull | 0.1593 | 0.1765 | 0.9026 | 0.6288 | 0.7351 |
| volatility | bull | 0.1992 | 0.2462 | 0.8091 | 0.5492 | 0.6267 |
| mom_x_lowvol_20_20 | bull | 0.1431 | 0.2455 | 0.5830 | 0.4242 | 0.4151 |
| low_downside | bull | 0.1282 | 0.2228 | 0.5754 | 0.3788 | 0.3967 |
| momentum_reversal | bull | 0.1396 | 0.2533 | 0.5509 | 0.3939 | 0.3840 |
| trend_lowvol | bull | 0.1390 | 0.2639 | 0.5267 | 0.3485 | 0.3551 |
| rsi_vol_combo | bull | 0.0963 | 0.2373 | 0.4058 | 0.2576 | 0.2551 |
| fund_profit_growth | bull | 0.0575 | 0.1971 | 0.2916 | 0.2727 | 0.1856 |
| fund_pe | bull | 0.0650 | 0.2254 | 0.2883 | 0.2197 | 0.1758 |
| fund_score | bull | 0.0440 | 0.2211 | 0.1988 | 0.1364 | 0.1129 |
| fund_roe | bull | 0.0473 | 0.2402 | 0.1969 | 0.1061 | 0.1089 |
| stroke_phase | bull | 0.0345 | 0.2221 | 0.1554 | 0.1250 | 0.0874 |
| fund_revenue_growth | bull | 0.0256 | 0.1860 | 0.1377 | 0.1212 | 0.0772 |
| trend_lowvol | bear | 0.1690 | 0.1795 | 0.9416 | 0.6986 | 0.7997 |
| fund_profit_growth | bear | 0.0837 | 0.1852 | 0.4517 | 0.3151 | 0.2970 |
| momentum_reversal | bear | 0.1352 | 0.3084 | 0.4383 | 0.2877 | 0.2822 |
| rsi_vol_combo | bear | 0.1157 | 0.2769 | 0.4181 | 0.3425 | 0.2806 |
| fund_revenue_growth | bear | 0.0680 | 0.1945 | 0.3499 | 0.2055 | 0.2109 |
| mom_x_lowvol_20_20 | bear | 0.0941 | 0.2983 | 0.3155 | 0.3151 | 0.2074 |
| fund_score | bear | 0.0538 | 0.1813 | 0.2968 | 0.2329 | 0.1830 |
| turnover_stability | bear | 0.0454 | 0.2436 | 0.1865 | 0.1233 | 0.1048 |

### 数据要素

- **Neutral**: ['momentum_reversal', 'mom_x_lowvol_20_20', 'trend_lowvol'] (单因子IC=0.0973, 组合IC=0.1083)
  - weights: [0.3502, 0.327, 0.3229]
- **Bull**: ['fund_pb', 'volatility', 'mom_x_lowvol_20_20'] (单因子IC=0.1138, 组合IC=0.1503)
  - bull_weights: [0.371, 0.3546, 0.2744]
- **Bear**: ['trend_lowvol', 'fund_score', 'momentum_reversal'] (单因子IC=0.0927, 组合IC=0.127)
  - bear_weights: [0.4634, 0.277, 0.2597]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| momentum_reversal | neutral | 0.0983 | 0.1643 | 0.5981 | 0.4739 | 0.4407 |
| mom_x_lowvol_20_20 | neutral | 0.0930 | 0.1621 | 0.5739 | 0.4342 | 0.4115 |
| trend_lowvol | neutral | 0.1007 | 0.1774 | 0.5675 | 0.4322 | 0.4064 |
| rsi_vol_combo | neutral | 0.0764 | 0.1537 | 0.4972 | 0.3758 | 0.3420 |
| fund_pb | neutral | 0.0809 | 0.1614 | 0.5014 | 0.3570 | 0.3402 |
| fund_profit_growth | neutral | 0.0516 | 0.1140 | 0.4522 | 0.3424 | 0.3035 |
| volatility | neutral | 0.0775 | 0.1890 | 0.4103 | 0.3132 | 0.2694 |
| fund_score | neutral | 0.0469 | 0.1437 | 0.3266 | 0.2714 | 0.2076 |
| fund_pe | neutral | 0.0437 | 0.1776 | 0.2461 | 0.1942 | 0.1469 |
| fund_revenue_growth | neutral | 0.0192 | 0.0936 | 0.2056 | 0.1482 | 0.1180 |
| low_downside | neutral | 0.0346 | 0.1758 | 0.1967 | 0.1858 | 0.1166 |
| fund_roe | neutral | 0.0313 | 0.1596 | 0.1962 | 0.1482 | 0.1126 |
| turnover_stability | neutral | 0.0194 | 0.1119 | 0.1732 | 0.2046 | 0.1043 |
| fund_pb | bull | 0.1231 | 0.1439 | 0.8555 | 0.5909 | 0.6805 |
| volatility | bull | 0.1218 | 0.1468 | 0.8296 | 0.5682 | 0.6505 |
| mom_x_lowvol_20_20 | bull | 0.0966 | 0.1432 | 0.6746 | 0.4924 | 0.5034 |
| trend_lowvol | bull | 0.1092 | 0.1632 | 0.6695 | 0.4773 | 0.4945 |
| momentum_reversal | bull | 0.0920 | 0.1454 | 0.6331 | 0.4621 | 0.4628 |
| low_downside | bull | 0.0882 | 0.1423 | 0.6201 | 0.4394 | 0.4463 |
| rsi_vol_combo | bull | 0.0658 | 0.1434 | 0.4587 | 0.2727 | 0.2919 |
| fund_pe | bull | 0.0568 | 0.1580 | 0.3594 | 0.2727 | 0.2287 |
| turnover_stability | bull | 0.0355 | 0.1162 | 0.3057 | 0.2652 | 0.1934 |
| fund_revenue_growth | bull | 0.0299 | 0.1017 | 0.2941 | 0.2197 | 0.1793 |
| fund_score | bull | 0.0301 | 0.1351 | 0.2227 | 0.1742 | 0.1307 |
| fund_profit_growth | bull | 0.0256 | 0.1366 | 0.1874 | 0.1439 | 0.1072 |
| fund_roe | bull | 0.0239 | 0.1340 | 0.1787 | 0.1212 | 0.1002 |
| trend_lowvol | bear | 0.1183 | 0.1513 | 0.7818 | 0.5342 | 0.5997 |
| fund_score | bear | 0.0703 | 0.1423 | 0.4938 | 0.4521 | 0.3585 |
| momentum_reversal | bear | 0.0894 | 0.2005 | 0.4461 | 0.5068 | 0.3361 |
| mom_x_lowvol_20_20 | bear | 0.0889 | 0.1824 | 0.4875 | 0.3699 | 0.3339 |
| fund_revenue_growth | bear | 0.0545 | 0.1095 | 0.4975 | 0.3151 | 0.3271 |
| fund_profit_growth | bear | 0.0474 | 0.1110 | 0.4271 | 0.2329 | 0.2633 |
| wash_sale_score | bear | 0.0342 | 0.1167 | 0.2928 | 0.2759 | 0.1868 |
| rsi_vol_combo | bear | 0.0409 | 0.1901 | 0.2151 | 0.2329 | 0.1326 |
| fund_roe | bear | 0.0328 | 0.1846 | 0.1780 | 0.1233 | 0.0999 |

### 文娱消费

- **Neutral**: ['fund_pb', 'volatility', 'trend_lowvol'] (单因子IC=0.1006, 组合IC=0.1198)
  - weights: [0.4309, 0.2939, 0.2752]
- **Bull**: ['fund_pb', 'trend_lowvol', 'volatility'] (单因子IC=0.0857, 组合IC=0.0979)
  - bull_weights: [0.4444, 0.3051, 0.2505]
- **Bear**: ['momentum_reversal', 'mom_x_lowvol_20_20', 'bb_width_20'] (单因子IC=0.1158, 组合IC=0.1397)
  - bear_weights: [0.4034, 0.4013, 0.1953]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| fund_pb | neutral | 0.0848 | 0.1409 | 0.6022 | 0.4697 | 0.4425 |
| volatility | neutral | 0.1089 | 0.2441 | 0.4462 | 0.3528 | 0.3018 |
| trend_lowvol | neutral | 0.1079 | 0.2491 | 0.4332 | 0.3048 | 0.2826 |
| low_downside | neutral | 0.0830 | 0.2295 | 0.3618 | 0.2735 | 0.2304 |
| rsi_vol_combo | neutral | 0.0739 | 0.2437 | 0.3032 | 0.2610 | 0.1912 |
| momentum_reversal | neutral | 0.0755 | 0.2479 | 0.3044 | 0.2255 | 0.1865 |
| mom_x_lowvol_20_20 | neutral | 0.0692 | 0.2420 | 0.2858 | 0.1858 | 0.1695 |
| turnover_stability | neutral | 0.0395 | 0.1709 | 0.2311 | 0.2171 | 0.1406 |
| fund_pe | neutral | 0.0481 | 0.1997 | 0.2410 | 0.1461 | 0.1381 |
| fund_profit_growth | neutral | 0.0345 | 0.1561 | 0.2211 | 0.1806 | 0.1305 |
| fund_score | neutral | 0.0320 | 0.2110 | 0.1518 | 0.1733 | 0.0891 |
| fund_revenue_growth | neutral | 0.0188 | 0.1843 | 0.1022 | 0.1461 | 0.0586 |
| fund_pb | bull | 0.0734 | 0.1216 | 0.6033 | 0.4621 | 0.4410 |
| trend_lowvol | bull | 0.0970 | 0.2135 | 0.4542 | 0.3333 | 0.3028 |
| volatility | bull | 0.0869 | 0.2158 | 0.4026 | 0.2348 | 0.2486 |
| momentum_reversal | bull | 0.0747 | 0.2071 | 0.3608 | 0.2955 | 0.2337 |
| mom_x_lowvol_20_20 | bull | 0.0673 | 0.2036 | 0.3306 | 0.2803 | 0.2116 |
| fund_pe | bull | 0.0619 | 0.1886 | 0.3282 | 0.2197 | 0.2001 |
| low_downside | bull | 0.0616 | 0.1864 | 0.3308 | 0.1818 | 0.1954 |
| rsi_vol_combo | bull | 0.0593 | 0.1952 | 0.3039 | 0.2576 | 0.1911 |
| turnover_stability | bull | 0.0272 | 0.1532 | 0.1778 | 0.1364 | 0.1010 |
| fund_roe | bull | 0.0185 | 0.1748 | 0.1060 | 0.1061 | 0.0586 |
| momentum_reversal | bear | 0.1371 | 0.2030 | 0.6754 | 0.5068 | 0.5089 |
| mom_x_lowvol_20_20 | bear | 0.1310 | 0.1949 | 0.6718 | 0.5068 | 0.5062 |
| bb_width_20 | bear | 0.0793 | 0.2073 | 0.3827 | 0.2877 | 0.2464 |
| rsi_vol_combo | bear | 0.0694 | 0.1898 | 0.3658 | 0.2603 | 0.2305 |

### 新型城镇化

- **Neutral**: ['mom_x_lowvol_20_20'] (单因子IC=0.1099, 组合IC=0.1099)
  - weights: [1.0]
- **Bull**: ['volatility', 'low_downside', 'trend_lowvol'] (单因子IC=0.1144, 组合IC=0.1237)
  - bull_weights: [0.3487, 0.3265, 0.3249]
- **Bear**: ['mom_x_lowvol_20_20', 'momentum_reversal', 'trend_lowvol'] (单因子IC=0.138, 组合IC=0.1483)
  - bear_weights: [0.3788, 0.3757, 0.2455]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| mom_x_lowvol_20_20 | neutral | 0.1099 | 0.1919 | 0.5726 | 0.4280 | 0.4088 |
| momentum_reversal | neutral | 0.1081 | 0.1916 | 0.5643 | 0.4280 | 0.4029 |
| rsi_vol_combo | neutral | 0.0817 | 0.1787 | 0.4572 | 0.3601 | 0.3110 |
| turnover_stability | neutral | 0.0624 | 0.1403 | 0.4451 | 0.3403 | 0.2983 |
| trend_lowvol | neutral | 0.0904 | 0.2131 | 0.4243 | 0.3424 | 0.2848 |
| fund_profit_growth | neutral | 0.0518 | 0.1325 | 0.3910 | 0.3090 | 0.2559 |
| fund_pb | neutral | 0.0666 | 0.1799 | 0.3703 | 0.2818 | 0.2373 |
| volatility | neutral | 0.0799 | 0.2279 | 0.3508 | 0.3486 | 0.2365 |
| fund_score | neutral | 0.0527 | 0.1558 | 0.3385 | 0.2818 | 0.2170 |
| fund_pe | neutral | 0.0534 | 0.2095 | 0.2550 | 0.1712 | 0.1494 |
| low_downside | neutral | 0.0466 | 0.2122 | 0.2194 | 0.2568 | 0.1379 |
| fund_revenue_growth | neutral | 0.0259 | 0.1254 | 0.2066 | 0.1816 | 0.1221 |
| fund_roe | neutral | 0.0312 | 0.1626 | 0.1916 | 0.1315 | 0.1084 |
| fund_gross_margin | neutral | 0.0272 | 0.1511 | 0.1799 | 0.1503 | 0.1035 |
| volatility | bull | 0.1170 | 0.1777 | 0.6582 | 0.5152 | 0.4986 |
| low_downside | bull | 0.1058 | 0.1700 | 0.6224 | 0.5000 | 0.4668 |
| trend_lowvol | bull | 0.1204 | 0.1876 | 0.6421 | 0.4470 | 0.4645 |
| fund_pb | bull | 0.0965 | 0.1605 | 0.6013 | 0.3939 | 0.4191 |
| turnover_stability | bull | 0.0478 | 0.1074 | 0.4453 | 0.3333 | 0.2969 |
| fund_pe | bull | 0.0691 | 0.1784 | 0.3875 | 0.3485 | 0.2613 |
| mom_x_lowvol_20_20 | bull | 0.0676 | 0.1745 | 0.3871 | 0.2955 | 0.2508 |
| momentum_reversal | bull | 0.0607 | 0.1822 | 0.3333 | 0.2273 | 0.2045 |
| fund_revenue_growth | bull | 0.0368 | 0.1095 | 0.3362 | 0.2121 | 0.2037 |
| fund_score | bull | 0.0429 | 0.1406 | 0.3049 | 0.2500 | 0.1906 |
| wash_sale_score | bull | 0.0349 | 0.1214 | 0.2875 | 0.2610 | 0.1813 |
| fund_roe | bull | 0.0319 | 0.1631 | 0.1957 | 0.1136 | 0.1090 |
| stroke_phase | bull | 0.0240 | 0.1399 | 0.1718 | 0.1364 | 0.0976 |
| mom_x_lowvol_20_20 | bear | 0.1403 | 0.1854 | 0.7567 | 0.6438 | 0.6219 |
| momentum_reversal | bear | 0.1409 | 0.1846 | 0.7632 | 0.6164 | 0.6168 |
| trend_lowvol | bear | 0.1327 | 0.2254 | 0.5885 | 0.3699 | 0.4031 |
| rsi_vol_combo | bear | 0.0956 | 0.1774 | 0.5388 | 0.3973 | 0.3764 |
| volatility | bear | 0.0670 | 0.2313 | 0.2899 | 0.2603 | 0.1827 |
| turnover_stability | bear | 0.0409 | 0.1274 | 0.3213 | 0.1233 | 0.1804 |

### 新型工业化

- **Neutral**: ['mom_x_lowvol_20_20', 'momentum_reversal', 'fund_pb'] (单因子IC=0.0879, 组合IC=0.121)
  - weights: [0.3513, 0.334, 0.3147]
- **Bull**: ['fund_pb', 'low_downside', 'trend_lowvol'] (单因子IC=0.0966, 组合IC=0.1403)
  - bull_weights: [0.506, 0.2567, 0.2372]
- **Bear**: ['fund_revenue_growth', 'rsi_vol_combo', 'momentum_reversal'] (单因子IC=0.0716, 组合IC=0.0731)
  - bear_weights: [0.4232, 0.29, 0.2867]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| mom_x_lowvol_20_20 | neutral | 0.0907 | 0.1810 | 0.5011 | 0.3737 | 0.3442 |
| momentum_reversal | neutral | 0.0893 | 0.1824 | 0.4899 | 0.3361 | 0.3273 |
| fund_pb | neutral | 0.0836 | 0.1817 | 0.4604 | 0.3392 | 0.3083 |
| volatility | neutral | 0.0837 | 0.1996 | 0.4193 | 0.3152 | 0.2757 |
| rsi_vol_combo | neutral | 0.0771 | 0.1864 | 0.4135 | 0.2777 | 0.2642 |
| trend_lowvol | neutral | 0.0810 | 0.2009 | 0.4034 | 0.2547 | 0.2531 |
| fund_profit_growth | neutral | 0.0564 | 0.1482 | 0.3806 | 0.2443 | 0.2368 |
| fund_pe | neutral | 0.0583 | 0.1719 | 0.3392 | 0.2610 | 0.2139 |
| fund_revenue_growth | neutral | 0.0508 | 0.1506 | 0.3374 | 0.2317 | 0.2078 |
| fund_score | neutral | 0.0557 | 0.1779 | 0.3130 | 0.2255 | 0.1918 |
| turnover_stability | neutral | 0.0390 | 0.1327 | 0.2939 | 0.2526 | 0.1841 |
| low_downside | neutral | 0.0508 | 0.1898 | 0.2674 | 0.2328 | 0.1648 |
| fund_pb | bull | 0.1296 | 0.1774 | 0.7308 | 0.5152 | 0.5536 |
| low_downside | bull | 0.0804 | 0.1887 | 0.4262 | 0.3182 | 0.2809 |
| trend_lowvol | bull | 0.0798 | 0.2003 | 0.3984 | 0.3030 | 0.2596 |
| volatility | bull | 0.0770 | 0.2094 | 0.3676 | 0.2652 | 0.2325 |
| fund_revenue_growth | bull | 0.0574 | 0.1574 | 0.3644 | 0.2197 | 0.2222 |
| fund_profit_growth | bull | 0.0544 | 0.1700 | 0.3201 | 0.3106 | 0.2098 |
| momentum_reversal | bull | 0.0447 | 0.1773 | 0.2520 | 0.1894 | 0.1498 |
| mom_x_lowvol_20_20 | bull | 0.0373 | 0.1741 | 0.2140 | 0.2121 | 0.1297 |
| turnover_stability | bull | 0.0289 | 0.1352 | 0.2135 | 0.1818 | 0.1261 |
| fund_score | bull | 0.0448 | 0.2093 | 0.2139 | 0.1364 | 0.1215 |
| rsi_vol_combo | bull | 0.0326 | 0.1742 | 0.1872 | 0.1212 | 0.1049 |
| fund_revenue_growth | bear | 0.0677 | 0.1204 | 0.5622 | 0.3699 | 0.3851 |
| rsi_vol_combo | bear | 0.0700 | 0.1671 | 0.4187 | 0.2603 | 0.2638 |
| momentum_reversal | bear | 0.0772 | 0.1906 | 0.4052 | 0.2877 | 0.2609 |
| trend_lowvol | bear | 0.0649 | 0.1756 | 0.3698 | 0.2603 | 0.2331 |
| mom_x_lowvol_20_20 | bear | 0.0856 | 0.2287 | 0.3742 | 0.2329 | 0.2307 |

### 新材料

- **Neutral**: ['mom_x_lowvol_20_20', 'fund_pb', 'momentum_reversal'] (单因子IC=0.0598, 组合IC=0.0861)
  - weights: [0.342, 0.3293, 0.3287]
- **Bull**: ['low_downside', 'volatility', 'trend_lowvol'] (单因子IC=0.0798, 组合IC=0.0923)
  - bull_weights: [0.3879, 0.3597, 0.2524]
- **Bear**: ['momentum_reversal'] (单因子IC=0.096, 组合IC=0.096)
  - bear_weights: [1.0]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| mom_x_lowvol_20_20 | neutral | 0.0556 | 0.1378 | 0.4036 | 0.3006 | 0.2625 |
| fund_pb | neutral | 0.0660 | 0.1631 | 0.4049 | 0.2484 | 0.2528 |
| momentum_reversal | neutral | 0.0577 | 0.1442 | 0.4002 | 0.2610 | 0.2523 |
| fund_pe | neutral | 0.0554 | 0.1466 | 0.3778 | 0.2296 | 0.2323 |
| trend_lowvol | neutral | 0.0620 | 0.1760 | 0.3525 | 0.2777 | 0.2252 |
| volatility | neutral | 0.0538 | 0.1624 | 0.3309 | 0.2735 | 0.2107 |
| fund_profit_growth | neutral | 0.0419 | 0.1250 | 0.3355 | 0.2380 | 0.2077 |
| turnover_stability | neutral | 0.0297 | 0.0941 | 0.3160 | 0.2881 | 0.2035 |
| rsi_vol_combo | neutral | 0.0417 | 0.1391 | 0.3000 | 0.2463 | 0.1869 |
| fund_score | neutral | 0.0333 | 0.1483 | 0.2246 | 0.1273 | 0.1266 |
| low_downside | neutral | 0.0268 | 0.1639 | 0.1633 | 0.1503 | 0.0939 |
| fund_revenue_growth | neutral | 0.0185 | 0.1101 | 0.1678 | 0.1002 | 0.0923 |
| low_downside | bull | 0.0829 | 0.1319 | 0.6282 | 0.4091 | 0.4426 |
| volatility | bull | 0.0864 | 0.1516 | 0.5702 | 0.4394 | 0.4104 |
| trend_lowvol | bull | 0.0702 | 0.1617 | 0.4345 | 0.3258 | 0.2880 |
| fund_pb | bull | 0.0634 | 0.1616 | 0.3927 | 0.2652 | 0.2484 |
| fund_pe | bull | 0.0489 | 0.1273 | 0.3842 | 0.2727 | 0.2445 |
| turnover_stability | bull | 0.0327 | 0.0882 | 0.3705 | 0.2803 | 0.2372 |
| mom_x_lowvol_20_20 | bull | 0.0469 | 0.1389 | 0.3380 | 0.3485 | 0.2279 |
| momentum_reversal | bull | 0.0454 | 0.1443 | 0.3145 | 0.2955 | 0.2037 |
| rsi_vol_combo | bull | 0.0305 | 0.1281 | 0.2382 | 0.1288 | 0.1344 |
| momentum_reversal | bear | 0.0960 | 0.1732 | 0.5542 | 0.3973 | 0.3872 |
| mom_x_lowvol_20_20 | bear | 0.0970 | 0.1890 | 0.5133 | 0.4247 | 0.3657 |
| rsi_vol_combo | bear | 0.0608 | 0.1220 | 0.4979 | 0.4521 | 0.3615 |
| fund_profit_growth | bear | 0.0575 | 0.1276 | 0.4508 | 0.3699 | 0.3088 |
| turnover_stability | bear | 0.0276 | 0.0783 | 0.3532 | 0.1781 | 0.2081 |
| bb_width_20 | bear | 0.0552 | 0.1737 | 0.3180 | 0.2877 | 0.2047 |
| fund_revenue_growth | bear | 0.0314 | 0.1176 | 0.2675 | 0.2055 | 0.1612 |
| trend_lowvol | bear | 0.0576 | 0.2096 | 0.2750 | 0.1233 | 0.1544 |

### 新消费

- **Neutral**: ['fund_pb', 'trend_lowvol', 'momentum_reversal'] (单因子IC=0.0765, 组合IC=0.1153)
  - weights: [0.3724, 0.3593, 0.2683]
- **Bull**: ['volatility', 'rsi_vol_combo'] (单因子IC=0.0899, 组合IC=0.1082)
  - bull_weights: [0.5055, 0.4945]
- **Bear**: ['momentum_reversal', 'trend_lowvol', 'mom_x_lowvol_20_20'] (单因子IC=0.1298, 组合IC=0.146)
  - bear_weights: [0.3588, 0.339, 0.3022]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| fund_pb | neutral | 0.0790 | 0.1766 | 0.4473 | 0.3027 | 0.2914 |
| trend_lowvol | neutral | 0.0863 | 0.1991 | 0.4337 | 0.2965 | 0.2812 |
| momentum_reversal | neutral | 0.0643 | 0.1913 | 0.3358 | 0.2505 | 0.2100 |
| mom_x_lowvol_20_20 | neutral | 0.0597 | 0.1856 | 0.3214 | 0.2443 | 0.1999 |
| volatility | neutral | 0.0593 | 0.1946 | 0.3050 | 0.2777 | 0.1948 |
| fund_pe | neutral | 0.0447 | 0.1814 | 0.2462 | 0.2129 | 0.1493 |
| rsi_vol_combo | neutral | 0.0438 | 0.1800 | 0.2434 | 0.1900 | 0.1448 |
| low_downside | neutral | 0.0425 | 0.1964 | 0.2162 | 0.2255 | 0.1325 |
| fund_profit_growth | neutral | 0.0200 | 0.1433 | 0.1394 | 0.1294 | 0.0787 |
| volatility | bull | 0.0972 | 0.1784 | 0.5448 | 0.4167 | 0.3859 |
| rsi_vol_combo | bull | 0.0826 | 0.1549 | 0.5329 | 0.4167 | 0.3775 |
| momentum_reversal | bull | 0.0899 | 0.1801 | 0.4994 | 0.4394 | 0.3594 |
| fund_pb | bull | 0.0906 | 0.1796 | 0.5046 | 0.3409 | 0.3383 |
| trend_lowvol | bull | 0.0950 | 0.1930 | 0.4924 | 0.3409 | 0.3301 |
| low_downside | bull | 0.0698 | 0.1631 | 0.4282 | 0.3182 | 0.2822 |
| fund_pe | bull | 0.0616 | 0.1538 | 0.4008 | 0.3106 | 0.2626 |
| mom_x_lowvol_20_20 | bull | 0.0649 | 0.1819 | 0.3570 | 0.3788 | 0.2461 |
| turnover_stability | bull | 0.0312 | 0.1436 | 0.2171 | 0.1818 | 0.1283 |
| momentum_reversal | bear | 0.1295 | 0.1554 | 0.8334 | 0.4795 | 0.6165 |
| trend_lowvol | bear | 0.1319 | 0.1830 | 0.7206 | 0.6164 | 0.5824 |
| mom_x_lowvol_20_20 | bear | 0.1281 | 0.1792 | 0.7151 | 0.4521 | 0.5192 |
| rsi_vol_combo | bear | 0.0888 | 0.1654 | 0.5367 | 0.3699 | 0.3676 |
| fund_pe | bear | 0.0687 | 0.1531 | 0.4487 | 0.3973 | 0.3135 |
| fund_pb | bear | 0.0611 | 0.1822 | 0.3355 | 0.2877 | 0.2160 |
| turnover_stability | bear | 0.0443 | 0.1466 | 0.3021 | 0.2329 | 0.1862 |
| volatility | bear | 0.0550 | 0.2088 | 0.2636 | 0.3151 | 0.1733 |

### 新能源

- **Neutral**: ['fund_profit_growth', 'trend_lowvol', 'volatility'] (单因子IC=0.0681, 组合IC=0.0862)
  - weights: [0.3373, 0.3332, 0.3295]
- **Bull**: ['low_downside', 'volatility', 'fund_pb'] (单因子IC=0.0734, 组合IC=0.0836)
  - bull_weights: [0.3699, 0.3392, 0.2909]
- **Bear**: ['mom_x_lowvol_20_20', 'momentum_reversal', 'trend_lowvol'] (单因子IC=0.1352, 组合IC=0.1573)
  - bear_weights: [0.3471, 0.3338, 0.3192]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| fund_profit_growth | neutral | 0.0493 | 0.1320 | 0.3737 | 0.2756 | 0.2384 |
| trend_lowvol | neutral | 0.0780 | 0.2135 | 0.3651 | 0.2902 | 0.2355 |
| volatility | neutral | 0.0770 | 0.2081 | 0.3700 | 0.2589 | 0.2329 |
| momentum_reversal | neutral | 0.0692 | 0.1863 | 0.3715 | 0.2338 | 0.2292 |
| fund_pb | neutral | 0.0614 | 0.1777 | 0.3455 | 0.2338 | 0.2132 |
| mom_x_lowvol_20_20 | neutral | 0.0617 | 0.1756 | 0.3516 | 0.2088 | 0.2125 |
| fund_pe | neutral | 0.0606 | 0.1902 | 0.3187 | 0.2234 | 0.1949 |
| turnover_stability | neutral | 0.0342 | 0.1116 | 0.3061 | 0.2255 | 0.1876 |
| rsi_vol_combo | neutral | 0.0529 | 0.1797 | 0.2944 | 0.2004 | 0.1767 |
| low_downside | neutral | 0.0514 | 0.1985 | 0.2589 | 0.1983 | 0.1551 |
| fund_score | neutral | 0.0365 | 0.1612 | 0.2265 | 0.1399 | 0.1291 |
| fund_gross_margin | neutral | 0.0138 | 0.1028 | 0.1346 | 0.1023 | 0.0742 |
| low_downside | bull | 0.0745 | 0.1528 | 0.4877 | 0.3864 | 0.3381 |
| volatility | bull | 0.0785 | 0.1725 | 0.4548 | 0.3636 | 0.3101 |
| fund_pb | bull | 0.0673 | 0.1659 | 0.4058 | 0.3106 | 0.2659 |
| turnover_stability | bull | 0.0437 | 0.1097 | 0.3981 | 0.3182 | 0.2624 |
| trend_lowvol | bull | 0.0582 | 0.1730 | 0.3364 | 0.2121 | 0.2039 |
| fund_pe | bull | 0.0462 | 0.1692 | 0.2732 | 0.2121 | 0.1656 |
| fund_gross_margin | bull | 0.0209 | 0.1129 | 0.1856 | 0.2045 | 0.1118 |
| momentum_reversal | bull | 0.0239 | 0.1564 | 0.1528 | 0.1136 | 0.0851 |
| mom_x_lowvol_20_20 | bull | 0.0187 | 0.1504 | 0.1246 | 0.1515 | 0.0717 |
| mom_x_lowvol_20_20 | bear | 0.1380 | 0.2167 | 0.6368 | 0.4795 | 0.4710 |
| momentum_reversal | bear | 0.1411 | 0.2219 | 0.6359 | 0.4247 | 0.4530 |
| trend_lowvol | bear | 0.1266 | 0.2041 | 0.6201 | 0.3973 | 0.4332 |
| rsi_vol_combo | bear | 0.0986 | 0.1734 | 0.5690 | 0.3699 | 0.3897 |
| fund_profit_growth | bear | 0.0401 | 0.1379 | 0.2909 | 0.2329 | 0.1793 |

### 新能源车

- **Neutral**: ['mom_x_lowvol_20_20', 'momentum_reversal', 'fund_pb'] (单因子IC=0.0733, 组合IC=0.1021)
  - weights: [0.3486, 0.3322, 0.3192]
- **Bull**: ['low_downside', 'fund_pb'] (单因子IC=0.0746, 组合IC=0.0914)
  - bull_weights: [0.5522, 0.4478]
- **Bear**: ['mom_x_lowvol_20_20'] (单因子IC=0.0865, 组合IC=0.0865)
  - bear_weights: [1.0]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| mom_x_lowvol_20_20 | neutral | 0.0740 | 0.1378 | 0.5368 | 0.4113 | 0.3788 |
| momentum_reversal | neutral | 0.0744 | 0.1413 | 0.5263 | 0.3716 | 0.3610 |
| fund_pb | neutral | 0.0714 | 0.1416 | 0.5043 | 0.3758 | 0.3469 |
| fund_profit_growth | neutral | 0.0512 | 0.1103 | 0.4635 | 0.3758 | 0.3189 |
| fund_pe | neutral | 0.0611 | 0.1331 | 0.4593 | 0.3486 | 0.3097 |
| trend_lowvol | neutral | 0.0791 | 0.1719 | 0.4599 | 0.3006 | 0.2991 |
| rsi_vol_combo | neutral | 0.0536 | 0.1314 | 0.4076 | 0.3361 | 0.2723 |
| volatility | neutral | 0.0587 | 0.1767 | 0.3323 | 0.2693 | 0.2109 |
| fund_score | neutral | 0.0471 | 0.1392 | 0.3384 | 0.2317 | 0.2084 |
| turnover_stability | neutral | 0.0238 | 0.0816 | 0.2916 | 0.2463 | 0.1817 |
| fund_revenue_growth | neutral | 0.0283 | 0.1015 | 0.2793 | 0.2359 | 0.1726 |
| low_downside | neutral | 0.0325 | 0.1617 | 0.2009 | 0.1399 | 0.1145 |
| fund_roe | neutral | 0.0289 | 0.1480 | 0.1950 | 0.1127 | 0.1085 |
| low_downside | bull | 0.0724 | 0.1211 | 0.5978 | 0.4242 | 0.4257 |
| fund_pb | bull | 0.0768 | 0.1550 | 0.4953 | 0.3939 | 0.3452 |
| volatility | bull | 0.0748 | 0.1514 | 0.4942 | 0.3712 | 0.3388 |
| trend_lowvol | bull | 0.0629 | 0.1496 | 0.4205 | 0.2727 | 0.2676 |
| turnover_stability | bull | 0.0297 | 0.0782 | 0.3803 | 0.3636 | 0.2593 |
| fund_pe | bull | 0.0398 | 0.1362 | 0.2922 | 0.2803 | 0.1871 |
| momentum_reversal | bull | 0.0356 | 0.1382 | 0.2574 | 0.1667 | 0.1502 |
| mom_x_lowvol_20_20 | bull | 0.0338 | 0.1384 | 0.2440 | 0.1818 | 0.1442 |
| fund_gross_margin | bull | 0.0210 | 0.0993 | 0.2110 | 0.1742 | 0.1239 |
| mom_x_lowvol_20_20 | bear | 0.0865 | 0.1921 | 0.4503 | 0.3151 | 0.2961 |
| rsi_vol_combo | bear | 0.0577 | 0.1411 | 0.4091 | 0.3699 | 0.2802 |
| momentum_reversal | bear | 0.0880 | 0.1844 | 0.4775 | 0.1507 | 0.2747 |
| fund_revenue_growth | bear | 0.0403 | 0.1213 | 0.3324 | 0.2603 | 0.2095 |
| fund_pb | bear | 0.0554 | 0.1734 | 0.3197 | 0.1233 | 0.1796 |
| fund_profit_growth | bear | 0.0373 | 0.1275 | 0.2923 | 0.1507 | 0.1682 |
| fund_pe | bear | 0.0433 | 0.1524 | 0.2841 | 0.1233 | 0.1596 |
| fund_gross_margin | bear | 0.0226 | 0.1091 | 0.2071 | 0.2055 | 0.1248 |

### 新零售

- **Neutral**: ['fund_pb', 'trend_lowvol', 'mom_x_lowvol_20_20'] (单因子IC=0.0745, 组合IC=0.1083)
  - weights: [0.4384, 0.2905, 0.2711]
- **Bull**: ['low_downside', 'fund_pb'] (单因子IC=0.0714, 组合IC=0.091)
  - bull_weights: [0.5825, 0.4175]
- **Bear**: ['momentum_reversal', 'rsi_vol_combo', 'fund_pb'] (单因子IC=0.0928, 组合IC=0.1366)
  - bear_weights: [0.3804, 0.3251, 0.2944]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| fund_pb | neutral | 0.0777 | 0.1332 | 0.5836 | 0.4280 | 0.4167 |
| trend_lowvol | neutral | 0.0807 | 0.1921 | 0.4198 | 0.3152 | 0.2761 |
| mom_x_lowvol_20_20 | neutral | 0.0651 | 0.1647 | 0.3949 | 0.3048 | 0.2576 |
| volatility | neutral | 0.0644 | 0.1745 | 0.3693 | 0.2985 | 0.2397 |
| low_downside | neutral | 0.0573 | 0.1643 | 0.3489 | 0.3194 | 0.2302 |
| momentum_reversal | neutral | 0.0626 | 0.1765 | 0.3547 | 0.2422 | 0.2203 |
| fund_pe | neutral | 0.0545 | 0.1570 | 0.3470 | 0.2234 | 0.2123 |
| fund_profit_growth | neutral | 0.0367 | 0.1218 | 0.3016 | 0.2025 | 0.1813 |
| rsi_vol_combo | neutral | 0.0444 | 0.1716 | 0.2588 | 0.1712 | 0.1515 |
| turnover_stability | neutral | 0.0234 | 0.1173 | 0.1998 | 0.1733 | 0.1172 |
| fund_score | neutral | 0.0249 | 0.1583 | 0.1570 | 0.1378 | 0.0893 |
| low_downside | bull | 0.0829 | 0.1417 | 0.5849 | 0.5152 | 0.4431 |
| fund_pb | bull | 0.0598 | 0.1235 | 0.4846 | 0.3106 | 0.3176 |
| volatility | bull | 0.0742 | 0.1591 | 0.4661 | 0.3485 | 0.3143 |
| turnover_stability | bull | 0.0322 | 0.1086 | 0.2963 | 0.2652 | 0.1874 |
| fund_pe | bull | 0.0469 | 0.1512 | 0.3100 | 0.1364 | 0.1761 |
| momentum_reversal | bull | 0.0432 | 0.1554 | 0.2780 | 0.1894 | 0.1653 |
| trend_lowvol | bull | 0.0444 | 0.1787 | 0.2486 | 0.2727 | 0.1582 |
| rsi_vol_combo | bull | 0.0383 | 0.1569 | 0.2443 | 0.2045 | 0.1471 |
| mom_x_lowvol_20_20 | bull | 0.0336 | 0.1521 | 0.2212 | 0.1288 | 0.1248 |
| fund_gross_margin | bull | 0.0194 | 0.1026 | 0.1889 | 0.1667 | 0.1102 |
| momentum_reversal | bear | 0.1226 | 0.2234 | 0.5488 | 0.2877 | 0.3533 |
| rsi_vol_combo | bear | 0.0959 | 0.2001 | 0.4792 | 0.2603 | 0.3020 |
| fund_pb | bear | 0.0600 | 0.1353 | 0.4436 | 0.2329 | 0.2735 |
| turnover_stability | bear | 0.0425 | 0.1035 | 0.4105 | 0.2055 | 0.2474 |
| trend_lowvol | bear | 0.0781 | 0.1926 | 0.4052 | 0.2055 | 0.2442 |
| mom_x_lowvol_20_20 | bear | 0.0756 | 0.2169 | 0.3485 | 0.2329 | 0.2148 |

### 旅游概念

- **Neutral**: ['trend_lowvol', 'mom_x_lowvol_20_20'] (单因子IC=0.0925, 组合IC=0.1146)
  - weights: [0.6143, 0.3857]
- **Bull**: ['momentum_reversal', 'trend_lowvol', 'mom_x_lowvol_20_20'] (单因子IC=0.1069, 组合IC=0.1272)
  - bull_weights: [0.3758, 0.3673, 0.2569]
- **Bear**: ['momentum_reversal', 'mom_x_lowvol_20_20'] (单因子IC=0.1833, 组合IC=0.1935)
  - bear_weights: [0.5289, 0.4711]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| trend_lowvol | neutral | 0.1101 | 0.2606 | 0.4225 | 0.3424 | 0.2836 |
| mom_x_lowvol_20_20 | neutral | 0.0749 | 0.2586 | 0.2896 | 0.2296 | 0.1780 |
| volatility | neutral | 0.0626 | 0.2179 | 0.2870 | 0.2380 | 0.1777 |
| low_downside | neutral | 0.0680 | 0.2351 | 0.2891 | 0.1962 | 0.1729 |
| fund_profit_growth | neutral | 0.0465 | 0.1638 | 0.2841 | 0.1942 | 0.1696 |
| momentum_reversal | neutral | 0.0764 | 0.2782 | 0.2745 | 0.1827 | 0.1623 |
| fund_pb | neutral | 0.0419 | 0.1876 | 0.2234 | 0.1294 | 0.1262 |
| rsi_vol_combo | neutral | 0.0541 | 0.2633 | 0.2056 | 0.1566 | 0.1189 |
| fund_score | neutral | 0.0379 | 0.2114 | 0.1794 | 0.1921 | 0.1070 |
| turnover_stability | neutral | 0.0237 | 0.1803 | 0.1312 | 0.1002 | 0.0722 |
| momentum_reversal | bull | 0.1200 | 0.2185 | 0.5492 | 0.4091 | 0.3869 |
| trend_lowvol | bull | 0.1114 | 0.2075 | 0.5368 | 0.4091 | 0.3782 |
| mom_x_lowvol_20_20 | bull | 0.0895 | 0.2281 | 0.3923 | 0.3485 | 0.2645 |
| volatility | bull | 0.0747 | 0.1881 | 0.3972 | 0.2727 | 0.2528 |
| rsi_vol_combo | bull | 0.0766 | 0.1982 | 0.3862 | 0.2727 | 0.2458 |
| low_downside | bull | 0.0688 | 0.1807 | 0.3807 | 0.2576 | 0.2394 |
| turnover_stability | bull | 0.0412 | 0.1630 | 0.2525 | 0.1970 | 0.1511 |
| fund_pb | bull | 0.0348 | 0.1967 | 0.1769 | 0.2273 | 0.1086 |
| fund_revenue_growth | bull | 0.0264 | 0.1684 | 0.1568 | 0.1136 | 0.0873 |
| fund_pe | bull | 0.0370 | 0.2648 | 0.1397 | 0.1591 | 0.0810 |
| momentum_reversal | bear | 0.1889 | 0.2290 | 0.8251 | 0.6712 | 0.6895 |
| mom_x_lowvol_20_20 | bear | 0.1777 | 0.2220 | 0.8006 | 0.5342 | 0.6141 |
| trend_lowvol | bear | 0.1062 | 0.2507 | 0.4238 | 0.3699 | 0.2903 |
| rsi_vol_combo | bear | 0.1017 | 0.2238 | 0.4546 | 0.2603 | 0.2864 |
| bb_width_20 | bear | 0.0747 | 0.2102 | 0.3552 | 0.2603 | 0.2238 |

### 旅游酒店

- **Neutral**: ['mom_x_lowvol_20_20', 'momentum_reversal', 'trend_lowvol'] (单因子IC=0.0796, 组合IC=0.0854)
  - weights: [0.3712, 0.316, 0.3128]
- **Bull**: ['momentum_reversal', 'mom_x_lowvol_20_20', 'trend_lowvol'] (单因子IC=0.0672, 组合IC=0.0854)
  - bull_weights: [0.4252, 0.3447, 0.2301]
- **Bear**: ['mom_x_lowvol_20_20'] (单因子IC=0.1371, 组合IC=0.1371)
  - bear_weights: [1.0]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| mom_x_lowvol_20_20 | neutral | 0.0842 | 0.2656 | 0.3170 | 0.2537 | 0.1987 |
| momentum_reversal | neutral | 0.0762 | 0.2726 | 0.2794 | 0.2109 | 0.1692 |
| trend_lowvol | neutral | 0.0786 | 0.2825 | 0.2782 | 0.2035 | 0.1674 |
| fund_pb | neutral | 0.0596 | 0.2269 | 0.2627 | 0.1879 | 0.1560 |
| rsi_vol_combo | neutral | 0.0479 | 0.2441 | 0.1964 | 0.1618 | 0.1141 |
| wash_sale_score | neutral | 0.0272 | 0.1926 | 0.1411 | 0.1478 | 0.0810 |
| momentum_reversal | bull | 0.0829 | 0.2405 | 0.3445 | 0.2727 | 0.2193 |
| mom_x_lowvol_20_20 | bull | 0.0697 | 0.2540 | 0.2745 | 0.2955 | 0.1778 |
| trend_lowvol | bull | 0.0491 | 0.2419 | 0.2028 | 0.1705 | 0.1187 |
| volatility | bull | 0.0401 | 0.2459 | 0.1629 | 0.1970 | 0.0975 |
| rsi_vol_combo | bull | 0.0297 | 0.2113 | 0.1404 | 0.1894 | 0.0835 |
| mom_x_lowvol_20_20 | bear | 0.1371 | 0.2480 | 0.5529 | 0.5068 | 0.4165 |
| momentum_reversal | bear | 0.1089 | 0.2280 | 0.4775 | 0.4247 | 0.3402 |
| fund_pb | bear | 0.0583 | 0.2078 | 0.2805 | 0.3973 | 0.1960 |
| rsi_vol_combo | bear | 0.0509 | 0.2026 | 0.2512 | 0.2877 | 0.1617 |

### 无人机

- **Neutral**: ['fund_pb', 'trend_lowvol', 'mom_x_lowvol_20_20'] (单因子IC=0.0793, 组合IC=0.1044)
  - weights: [0.3569, 0.3359, 0.3073]
- **Bull**: ['low_downside', 'fund_pb', 'volatility'] (单因子IC=0.1169, 组合IC=0.1411)
  - bull_weights: [0.3417, 0.3318, 0.3266]
- **Bear**: ['rsi_vol_combo', 'momentum_reversal'] (单因子IC=0.0672, 组合IC=0.0692)
  - bear_weights: [0.6047, 0.3953]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| fund_pb | neutral | 0.0730 | 0.1379 | 0.5291 | 0.4384 | 0.3805 |
| trend_lowvol | neutral | 0.0858 | 0.1631 | 0.5263 | 0.3612 | 0.3582 |
| mom_x_lowvol_20_20 | neutral | 0.0793 | 0.1616 | 0.4905 | 0.3361 | 0.3277 |
| momentum_reversal | neutral | 0.0800 | 0.1676 | 0.4772 | 0.3507 | 0.3223 |
| volatility | neutral | 0.0663 | 0.1625 | 0.4077 | 0.3257 | 0.2703 |
| rsi_vol_combo | neutral | 0.0653 | 0.1617 | 0.4037 | 0.2818 | 0.2588 |
| fund_pe | neutral | 0.0524 | 0.1428 | 0.3670 | 0.3215 | 0.2425 |
| fund_profit_growth | neutral | 0.0473 | 0.1397 | 0.3387 | 0.2526 | 0.2122 |
| turnover_stability | neutral | 0.0304 | 0.1042 | 0.2921 | 0.2672 | 0.1851 |
| fund_score | neutral | 0.0415 | 0.1566 | 0.2651 | 0.1858 | 0.1572 |
| fund_roe | neutral | 0.0263 | 0.1547 | 0.1697 | 0.1169 | 0.0948 |
| low_downside | neutral | 0.0263 | 0.1650 | 0.1593 | 0.1357 | 0.0905 |
| fund_revenue_growth | neutral | 0.0188 | 0.1196 | 0.1571 | 0.1023 | 0.0866 |
| low_downside | bull | 0.1144 | 0.1361 | 0.8409 | 0.5985 | 0.6721 |
| fund_pb | bull | 0.1101 | 0.1374 | 0.8014 | 0.6288 | 0.6526 |
| volatility | bull | 0.1261 | 0.1547 | 0.8153 | 0.5758 | 0.6424 |
| trend_lowvol | bull | 0.1197 | 0.1613 | 0.7420 | 0.5227 | 0.5649 |
| mom_x_lowvol_20_20 | bull | 0.0737 | 0.1461 | 0.5047 | 0.3864 | 0.3499 |
| momentum_reversal | bull | 0.0728 | 0.1513 | 0.4811 | 0.3939 | 0.3353 |
| fund_pe | bull | 0.0717 | 0.1628 | 0.4404 | 0.3333 | 0.2936 |
| turnover_stability | bull | 0.0292 | 0.0949 | 0.3082 | 0.2273 | 0.1891 |
| rsi_vol_combo | bull | 0.0421 | 0.1528 | 0.2755 | 0.1591 | 0.1597 |
| stroke_phase | bull | 0.0164 | 0.1070 | 0.1535 | 0.1136 | 0.0855 |
| fund_score | bull | 0.0126 | 0.1306 | 0.0963 | 0.1136 | 0.0536 |
| rsi_vol_combo | bear | 0.0668 | 0.1318 | 0.5071 | 0.4521 | 0.3682 |
| momentum_reversal | bear | 0.0676 | 0.1653 | 0.4087 | 0.1781 | 0.2407 |

### 无人驾驶

- **Neutral**: ['momentum_reversal', 'trend_lowvol'] (单因子IC=0.085, 组合IC=0.0921)
  - weights: [0.5005, 0.4995]
- **Bull**: ['volatility', 'fund_pb', 'trend_lowvol'] (单因子IC=0.0789, 组合IC=0.0953)
  - bull_weights: [0.3583, 0.3396, 0.3021]
- **Bear**: ['fund_profit_growth', 'fund_revenue_growth', 'mom_x_lowvol_20_20'] (单因子IC=0.0644, 组合IC=0.0878)
  - bear_weights: [0.4436, 0.3028, 0.2536]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| momentum_reversal | neutral | 0.0804 | 0.1582 | 0.5082 | 0.3633 | 0.3464 |
| trend_lowvol | neutral | 0.0896 | 0.1767 | 0.5073 | 0.3633 | 0.3458 |
| mom_x_lowvol_20_20 | neutral | 0.0789 | 0.1568 | 0.5030 | 0.3486 | 0.3392 |
| fund_pb | neutral | 0.0662 | 0.1582 | 0.4186 | 0.2443 | 0.2604 |
| rsi_vol_combo | neutral | 0.0597 | 0.1512 | 0.3949 | 0.3027 | 0.2572 |
| fund_profit_growth | neutral | 0.0478 | 0.1285 | 0.3717 | 0.3006 | 0.2417 |
| volatility | neutral | 0.0689 | 0.1871 | 0.3683 | 0.2714 | 0.2341 |
| fund_score | neutral | 0.0501 | 0.1581 | 0.3171 | 0.2171 | 0.1930 |
| fund_pe | neutral | 0.0503 | 0.1604 | 0.3136 | 0.1942 | 0.1872 |
| fund_revenue_growth | neutral | 0.0344 | 0.1194 | 0.2885 | 0.1962 | 0.1726 |
| fund_roe | neutral | 0.0375 | 0.1542 | 0.2433 | 0.1461 | 0.1394 |
| turnover_stability | neutral | 0.0207 | 0.0993 | 0.2084 | 0.1983 | 0.1249 |
| low_downside | neutral | 0.0367 | 0.1790 | 0.2052 | 0.1608 | 0.1191 |
| volatility | bull | 0.0861 | 0.1864 | 0.4618 | 0.3788 | 0.3184 |
| fund_pb | bull | 0.0759 | 0.1716 | 0.4425 | 0.3636 | 0.3017 |
| trend_lowvol | bull | 0.0748 | 0.1795 | 0.4168 | 0.2879 | 0.2684 |
| turnover_stability | bull | 0.0374 | 0.0929 | 0.4024 | 0.2652 | 0.2546 |
| low_downside | bull | 0.0586 | 0.1466 | 0.3997 | 0.2197 | 0.2437 |
| mom_x_lowvol_20_20 | bull | 0.0509 | 0.1606 | 0.3167 | 0.2803 | 0.2027 |
| momentum_reversal | bull | 0.0430 | 0.1551 | 0.2775 | 0.2197 | 0.1692 |
| fund_pe | bull | 0.0426 | 0.1608 | 0.2649 | 0.2348 | 0.1636 |
| stroke_phase | bull | 0.0217 | 0.0986 | 0.2201 | 0.1970 | 0.1317 |
| fund_gross_margin | bull | 0.0250 | 0.1200 | 0.2087 | 0.1515 | 0.1201 |
| rsi_vol_combo | bull | 0.0268 | 0.1393 | 0.1926 | 0.1818 | 0.1138 |
| fund_profit_growth | bear | 0.0776 | 0.1479 | 0.5250 | 0.4521 | 0.3812 |
| fund_revenue_growth | bear | 0.0548 | 0.1299 | 0.4221 | 0.2329 | 0.2602 |
| mom_x_lowvol_20_20 | bear | 0.0608 | 0.1796 | 0.3385 | 0.2877 | 0.2179 |
| momentum_reversal | bear | 0.0549 | 0.1701 | 0.3230 | 0.2329 | 0.1991 |
| trend_lowvol | bear | 0.0631 | 0.1932 | 0.3268 | 0.1781 | 0.1925 |
| fund_score | bear | 0.0379 | 0.1892 | 0.2003 | 0.1507 | 0.1152 |
| turnover_stability | bear | 0.0191 | 0.0976 | 0.1959 | 0.1507 | 0.1127 |
| fund_pe | bear | 0.0275 | 0.1546 | 0.1778 | 0.2055 | 0.1072 |

### 无线充电

- **Neutral**: ['fund_pb', 'trend_lowvol', 'momentum_reversal'] (单因子IC=0.0789, 组合IC=0.1111)
  - weights: [0.3842, 0.3581, 0.2577]
- **Bull**: ['fund_pb', 'low_downside', 'turnover_stability'] (单因子IC=0.0558, 组合IC=0.0753)
  - bull_weights: [0.3956, 0.3114, 0.293]
- **Bear**: ['fund_pb', 'rsi_vol_combo', 'wash_sale_score'] (单因子IC=0.1042, 组合IC=0.1578)
  - bear_weights: [0.4315, 0.3251, 0.2434]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| fund_pb | neutral | 0.0706 | 0.1720 | 0.4106 | 0.3382 | 0.2747 |
| trend_lowvol | neutral | 0.0955 | 0.2402 | 0.3976 | 0.2881 | 0.2561 |
| momentum_reversal | neutral | 0.0705 | 0.2304 | 0.3059 | 0.2046 | 0.1842 |
| mom_x_lowvol_20_20 | neutral | 0.0641 | 0.2330 | 0.2750 | 0.2234 | 0.1682 |
| fund_pe | neutral | 0.0570 | 0.2100 | 0.2715 | 0.2317 | 0.1672 |
| rsi_vol_combo | neutral | 0.0503 | 0.2229 | 0.2256 | 0.1806 | 0.1332 |
| volatility | neutral | 0.0489 | 0.2545 | 0.1920 | 0.1106 | 0.1066 |
| fund_profit_growth | neutral | 0.0392 | 0.2167 | 0.1808 | 0.1284 | 0.1020 |
| low_downside | neutral | 0.0328 | 0.2292 | 0.1433 | 0.1336 | 0.0812 |
| fund_score | neutral | 0.0344 | 0.2377 | 0.1449 | 0.1159 | 0.0809 |
| fund_revenue_growth | neutral | 0.0283 | 0.2001 | 0.1413 | 0.1399 | 0.0805 |
| fund_pb | bull | 0.0693 | 0.2092 | 0.3315 | 0.2348 | 0.2047 |
| low_downside | bull | 0.0526 | 0.2002 | 0.2626 | 0.2273 | 0.1611 |
| turnover_stability | bull | 0.0455 | 0.1772 | 0.2565 | 0.1818 | 0.1516 |
| momentum_reversal | bull | 0.0559 | 0.2205 | 0.2537 | 0.1894 | 0.1509 |
| fund_pe | bull | 0.0449 | 0.1918 | 0.2341 | 0.1439 | 0.1339 |
| rsi_vol_combo | bull | 0.0399 | 0.2094 | 0.1905 | 0.1136 | 0.1061 |
| trend_lowvol | bull | 0.0348 | 0.2365 | 0.1473 | 0.1136 | 0.0820 |
| mom_x_lowvol_20_20 | bull | 0.0289 | 0.2140 | 0.1351 | 0.1742 | 0.0793 |
| volatility | bull | 0.0286 | 0.2143 | 0.1334 | 0.1439 | 0.0763 |
| fund_gross_margin | bull | 0.0184 | 0.1781 | 0.1035 | 0.1515 | 0.0596 |
| fund_pb | bear | 0.1285 | 0.1904 | 0.6746 | 0.5068 | 0.5082 |
| rsi_vol_combo | bear | 0.1135 | 0.2192 | 0.5175 | 0.4795 | 0.3828 |
| wash_sale_score | bear | 0.0706 | 0.1812 | 0.3894 | 0.4722 | 0.2867 |
| momentum_reversal | bear | 0.1014 | 0.2825 | 0.3587 | 0.2603 | 0.2261 |
| mom_x_lowvol_20_20 | bear | 0.0877 | 0.2771 | 0.3164 | 0.2603 | 0.1994 |

### 无线耳机

- **Neutral**: ['fund_pb', 'mom_x_lowvol_20_20', 'momentum_reversal'] (单因子IC=0.0804, 组合IC=0.1129)
  - weights: [0.3846, 0.3226, 0.2928]
- **Bull**: ['fund_pb', 'turnover_stability', 'volatility'] (单因子IC=0.085, 组合IC=0.1416)
  - bull_weights: [0.4779, 0.2683, 0.2537]
- **Bear**: ['fund_pb', 'mom_x_lowvol_20_20', 'momentum_reversal'] (单因子IC=0.0722, 组合IC=0.1095)
  - bear_weights: [0.4245, 0.3645, 0.2109]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| fund_pb | neutral | 0.0800 | 0.1656 | 0.4834 | 0.3528 | 0.3270 |
| mom_x_lowvol_20_20 | neutral | 0.0818 | 0.1951 | 0.4191 | 0.3090 | 0.2743 |
| momentum_reversal | neutral | 0.0794 | 0.2062 | 0.3852 | 0.2923 | 0.2489 |
| volatility | neutral | 0.0746 | 0.1956 | 0.3812 | 0.2714 | 0.2423 |
| fund_pe | neutral | 0.0586 | 0.1623 | 0.3607 | 0.3090 | 0.2361 |
| trend_lowvol | neutral | 0.0735 | 0.2197 | 0.3344 | 0.2818 | 0.2143 |
| fund_profit_growth | neutral | 0.0561 | 0.1680 | 0.3339 | 0.2338 | 0.2060 |
| rsi_vol_combo | neutral | 0.0591 | 0.1943 | 0.3042 | 0.2255 | 0.1864 |
| low_downside | neutral | 0.0441 | 0.1802 | 0.2447 | 0.2067 | 0.1476 |
| fund_score | neutral | 0.0418 | 0.2018 | 0.2073 | 0.1712 | 0.1214 |
| turnover_stability | neutral | 0.0294 | 0.1559 | 0.1882 | 0.1127 | 0.1047 |
| fund_roe | neutral | 0.0319 | 0.1930 | 0.1655 | 0.1190 | 0.0926 |
| fund_revenue_growth | neutral | 0.0159 | 0.1693 | 0.0941 | 0.1106 | 0.0523 |
| fund_pb | bull | 0.1223 | 0.1709 | 0.7154 | 0.5000 | 0.5365 |
| turnover_stability | bull | 0.0597 | 0.1351 | 0.4418 | 0.3636 | 0.3012 |
| volatility | bull | 0.0730 | 0.1806 | 0.4043 | 0.4091 | 0.2848 |
| trend_lowvol | bull | 0.0705 | 0.1863 | 0.3786 | 0.2576 | 0.2381 |
| mom_x_lowvol_20_20 | bull | 0.0568 | 0.1728 | 0.3289 | 0.3030 | 0.2143 |
| low_downside | bull | 0.0602 | 0.1751 | 0.3440 | 0.2424 | 0.2137 |
| fund_pe | bull | 0.0557 | 0.1779 | 0.3131 | 0.2424 | 0.1945 |
| momentum_reversal | bull | 0.0529 | 0.1767 | 0.2995 | 0.2121 | 0.1815 |
| rsi_vol_combo | bull | 0.0347 | 0.1769 | 0.1963 | 0.1061 | 0.1086 |
| fund_pb | bear | 0.0773 | 0.1524 | 0.5072 | 0.2877 | 0.3266 |
| mom_x_lowvol_20_20 | bear | 0.0801 | 0.1996 | 0.4014 | 0.3973 | 0.2804 |
| momentum_reversal | bear | 0.0591 | 0.2197 | 0.2692 | 0.2055 | 0.1623 |
| rsi_vol_combo | bear | 0.0367 | 0.1685 | 0.2181 | 0.1507 | 0.1255 |
| fund_pe | bear | 0.0295 | 0.1506 | 0.1960 | 0.2055 | 0.1181 |
| vol_confirm | bear | 0.0279 | 0.1440 | 0.1941 | 0.1233 | 0.1090 |

### 时空大数据

- **Neutral**: ['trend_lowvol', 'momentum_reversal'] (单因子IC=0.1077, 组合IC=0.123)
  - weights: [0.5335, 0.4665]
- **Bull**: ['fund_pb', 'low_downside'] (单因子IC=0.1224, 组合IC=0.1405)
  - bull_weights: [0.5058, 0.4942]
- **Bear**: ['turnover_stability', 'fund_revenue_growth', 'trend_lowvol'] (单因子IC=0.1615, 组合IC=0.199)
  - bear_weights: [0.4051, 0.3209, 0.274]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| trend_lowvol | neutral | 0.1122 | 0.2651 | 0.4233 | 0.3340 | 0.2823 |
| momentum_reversal | neutral | 0.1032 | 0.2700 | 0.3821 | 0.2923 | 0.2469 |
| mom_x_lowvol_20_20 | neutral | 0.1034 | 0.2773 | 0.3728 | 0.3079 | 0.2438 |
| rsi_vol_combo | neutral | 0.0737 | 0.2642 | 0.2788 | 0.2484 | 0.1740 |
| volatility | neutral | 0.0657 | 0.2862 | 0.2296 | 0.2317 | 0.1414 |
| turnover_stability | neutral | 0.0430 | 0.2369 | 0.1816 | 0.1628 | 0.1056 |
| fund_pb | neutral | 0.0383 | 0.2391 | 0.1602 | 0.1482 | 0.0919 |
| fund_pe | neutral | 0.0426 | 0.3003 | 0.1419 | 0.1345 | 0.0805 |
| fund_profit_growth | neutral | 0.0356 | 0.2472 | 0.1441 | 0.1148 | 0.0803 |
| low_downside | neutral | 0.0374 | 0.2795 | 0.1337 | 0.1388 | 0.0761 |
| fund_score | neutral | 0.0350 | 0.2645 | 0.1325 | 0.1148 | 0.0739 |
| fund_revenue_growth | neutral | 0.0260 | 0.2508 | 0.1037 | 0.1127 | 0.0577 |
| fund_gross_margin | neutral | 0.0274 | 0.2652 | 0.1033 | 0.1054 | 0.0571 |
| fund_pb | bull | 0.1201 | 0.2350 | 0.5111 | 0.4129 | 0.3610 |
| low_downside | bull | 0.1246 | 0.2435 | 0.5117 | 0.3788 | 0.3528 |
| volatility | bull | 0.1117 | 0.2579 | 0.4330 | 0.3333 | 0.2887 |
| trend_lowvol | bull | 0.0974 | 0.2407 | 0.4045 | 0.3030 | 0.2635 |
| mom_x_lowvol_20_20 | bull | 0.0782 | 0.2575 | 0.3037 | 0.2614 | 0.1915 |
| momentum_reversal | bull | 0.0688 | 0.2525 | 0.2727 | 0.2273 | 0.1673 |
| stroke_phase | bull | 0.0625 | 0.2331 | 0.2679 | 0.1970 | 0.1603 |
| fund_pe | bull | 0.0690 | 0.3131 | 0.2204 | 0.1712 | 0.1291 |
| rsi_vol_combo | bull | 0.0432 | 0.2247 | 0.1922 | 0.2197 | 0.1172 |
| turnover_stability | bear | 0.1657 | 0.2185 | 0.7585 | 0.4521 | 0.5507 |
| fund_revenue_growth | bear | 0.1593 | 0.2601 | 0.6124 | 0.4247 | 0.4363 |
| trend_lowvol | bear | 0.1594 | 0.2932 | 0.5438 | 0.3699 | 0.3724 |
| fund_score | bear | 0.1020 | 0.2435 | 0.4190 | 0.4247 | 0.2985 |
| mom_x_lowvol_20_20 | bear | 0.1272 | 0.3107 | 0.4095 | 0.3151 | 0.2692 |
| rsi_vol_combo | bear | 0.1178 | 0.3282 | 0.3589 | 0.2877 | 0.2311 |
| momentum_reversal | bear | 0.1209 | 0.3514 | 0.3441 | 0.2877 | 0.2215 |
| fund_profit_growth | bear | 0.0823 | 0.2641 | 0.3116 | 0.1644 | 0.1814 |
| volatility | bear | 0.0902 | 0.3168 | 0.2846 | 0.2329 | 0.1754 |
| fund_roe | bear | 0.0648 | 0.2798 | 0.2316 | 0.1507 | 0.1332 |
| fund_pe | bear | 0.0623 | 0.3198 | 0.1948 | 0.1268 | 0.1097 |

### 昨日涨停

- **Neutral**: ['trend_lowvol', 'mom_x_lowvol_20_20'] (单因子IC=0.0833, 组合IC=0.0984)
  - weights: [0.5027, 0.4973]
- **Bull**: ['low_downside'] (单因子IC=0.1083, 组合IC=0.1083)
  - bull_weights: [1.0]
- **Bear**: ['turnover_stability', 'trend_lowvol', 'mom_x_lowvol_20_20'] (单因子IC=0.08, 组合IC=0.0928)
  - bear_weights: [0.3644, 0.3309, 0.3047]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| trend_lowvol | neutral | 0.0886 | 0.1906 | 0.4648 | 0.3507 | 0.3139 |
| mom_x_lowvol_20_20 | neutral | 0.0780 | 0.1687 | 0.4627 | 0.3424 | 0.3105 |
| momentum_reversal | neutral | 0.0788 | 0.1734 | 0.4542 | 0.3278 | 0.3015 |
| volatility | neutral | 0.0692 | 0.1694 | 0.4088 | 0.3236 | 0.2705 |
| fund_pb | neutral | 0.0567 | 0.1562 | 0.3631 | 0.2150 | 0.2206 |
| rsi_vol_combo | neutral | 0.0580 | 0.1703 | 0.3409 | 0.2317 | 0.2099 |
| turnover_stability | neutral | 0.0366 | 0.1127 | 0.3245 | 0.2735 | 0.2066 |
| fund_pe | neutral | 0.0521 | 0.1725 | 0.3018 | 0.2296 | 0.1856 |
| fund_profit_growth | neutral | 0.0376 | 0.1383 | 0.2721 | 0.1983 | 0.1631 |
| low_downside | neutral | 0.0446 | 0.1738 | 0.2568 | 0.2276 | 0.1576 |
| fund_revenue_growth | neutral | 0.0290 | 0.1494 | 0.1938 | 0.1691 | 0.1133 |
| fund_score | neutral | 0.0332 | 0.1704 | 0.1951 | 0.1378 | 0.1110 |
| low_downside | bull | 0.1083 | 0.1417 | 0.7648 | 0.5985 | 0.6112 |
| volatility | bull | 0.0958 | 0.1561 | 0.6139 | 0.4242 | 0.4372 |
| fund_pb | bull | 0.0742 | 0.1438 | 0.5161 | 0.4773 | 0.3812 |
| turnover_stability | bull | 0.0359 | 0.1193 | 0.3013 | 0.2197 | 0.1837 |
| trend_lowvol | bull | 0.0557 | 0.1994 | 0.2795 | 0.2197 | 0.1705 |
| fund_score | bull | 0.0369 | 0.1797 | 0.2054 | 0.2727 | 0.1307 |
| fund_pe | bull | 0.0376 | 0.1755 | 0.2144 | 0.2045 | 0.1291 |
| fund_profit_growth | bull | 0.0300 | 0.1294 | 0.2316 | 0.1061 | 0.1281 |
| fund_revenue_growth | bull | 0.0294 | 0.1411 | 0.2087 | 0.1970 | 0.1249 |
| momentum_reversal | bull | 0.0317 | 0.1696 | 0.1867 | 0.1288 | 0.1054 |
| fund_gross_margin | bull | 0.0182 | 0.1281 | 0.1421 | 0.1364 | 0.0807 |
| stroke_phase | bull | 0.0141 | 0.1113 | 0.1265 | 0.1212 | 0.0709 |
| rsi_vol_combo | bull | 0.0197 | 0.1587 | 0.1240 | 0.1364 | 0.0705 |
| fund_roe | bull | 0.0194 | 0.1794 | 0.1082 | 0.1515 | 0.0623 |
| turnover_stability | bear | 0.0471 | 0.0893 | 0.5271 | 0.4247 | 0.3755 |
| trend_lowvol | bear | 0.0957 | 0.1962 | 0.4881 | 0.3973 | 0.3410 |
| mom_x_lowvol_20_20 | bear | 0.0972 | 0.2121 | 0.4584 | 0.3699 | 0.3139 |
| rsi_vol_combo | bear | 0.0752 | 0.1639 | 0.4588 | 0.3425 | 0.3080 |
| momentum_reversal | bear | 0.0883 | 0.1999 | 0.4416 | 0.3425 | 0.2964 |
| fund_pb | bear | 0.0697 | 0.1586 | 0.4393 | 0.2329 | 0.2708 |
| fund_revenue_growth | bear | 0.0365 | 0.1457 | 0.2507 | 0.2603 | 0.1580 |

### 昨日涨停_含一字

- **Neutral**: ['trend_lowvol', 'mom_x_lowvol_20_20'] (单因子IC=0.0822, 组合IC=0.0974)
  - weights: [0.5015, 0.4985]
- **Bull**: ['low_downside'] (单因子IC=0.1064, 组合IC=0.1064)
  - bull_weights: [1.0]
- **Bear**: ['turnover_stability', 'trend_lowvol', 'mom_x_lowvol_20_20'] (单因子IC=0.0779, 组合IC=0.0924)
  - bear_weights: [0.3508, 0.3409, 0.3084]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| trend_lowvol | neutral | 0.0871 | 0.1890 | 0.4607 | 0.3612 | 0.3135 |
| mom_x_lowvol_20_20 | neutral | 0.0774 | 0.1666 | 0.4643 | 0.3424 | 0.3117 |
| momentum_reversal | neutral | 0.0781 | 0.1712 | 0.4564 | 0.3132 | 0.2996 |
| volatility | neutral | 0.0690 | 0.1684 | 0.4097 | 0.3194 | 0.2703 |
| fund_pb | neutral | 0.0555 | 0.1543 | 0.3600 | 0.2129 | 0.2183 |
| rsi_vol_combo | neutral | 0.0571 | 0.1680 | 0.3400 | 0.2422 | 0.2112 |
| turnover_stability | neutral | 0.0361 | 0.1122 | 0.3218 | 0.2756 | 0.2052 |
| fund_pe | neutral | 0.0522 | 0.1721 | 0.3031 | 0.2380 | 0.1876 |
| fund_profit_growth | neutral | 0.0364 | 0.1386 | 0.2626 | 0.1942 | 0.1568 |
| low_downside | neutral | 0.0439 | 0.1727 | 0.2544 | 0.2046 | 0.1532 |
| fund_score | neutral | 0.0317 | 0.1703 | 0.1862 | 0.1211 | 0.1044 |
| fund_revenue_growth | neutral | 0.0264 | 0.1491 | 0.1769 | 0.1545 | 0.1021 |
| low_downside | bull | 0.1064 | 0.1402 | 0.7586 | 0.5758 | 0.5977 |
| volatility | bull | 0.0935 | 0.1550 | 0.6031 | 0.4394 | 0.4340 |
| fund_pb | bull | 0.0721 | 0.1445 | 0.4987 | 0.4545 | 0.3627 |
| trend_lowvol | bull | 0.0559 | 0.1999 | 0.2797 | 0.2348 | 0.1727 |
| turnover_stability | bull | 0.0334 | 0.1194 | 0.2800 | 0.2121 | 0.1697 |
| fund_profit_growth | bull | 0.0294 | 0.1276 | 0.2303 | 0.1288 | 0.1300 |
| fund_revenue_growth | bull | 0.0302 | 0.1401 | 0.2157 | 0.2045 | 0.1299 |
| fund_score | bull | 0.0350 | 0.1781 | 0.1968 | 0.2727 | 0.1252 |
| fund_pe | bull | 0.0369 | 0.1766 | 0.2090 | 0.1894 | 0.1243 |
| momentum_reversal | bull | 0.0326 | 0.1671 | 0.1951 | 0.1439 | 0.1116 |
| rsi_vol_combo | bull | 0.0211 | 0.1561 | 0.1354 | 0.1364 | 0.0769 |
| fund_gross_margin | bull | 0.0166 | 0.1284 | 0.1291 | 0.1364 | 0.0734 |
| stroke_phase | bull | 0.0143 | 0.1111 | 0.1287 | 0.1212 | 0.0721 |
| fund_roe | bull | 0.0179 | 0.1804 | 0.0994 | 0.1364 | 0.0565 |
| turnover_stability | bear | 0.0454 | 0.0896 | 0.5074 | 0.3699 | 0.3475 |
| trend_lowvol | bear | 0.0949 | 0.1924 | 0.4931 | 0.3699 | 0.3378 |
| mom_x_lowvol_20_20 | bear | 0.0935 | 0.2096 | 0.4461 | 0.3699 | 0.3055 |
| rsi_vol_combo | bear | 0.0724 | 0.1625 | 0.4456 | 0.3151 | 0.2930 |
| fund_pb | bear | 0.0691 | 0.1551 | 0.4456 | 0.2603 | 0.2808 |
| momentum_reversal | bear | 0.0850 | 0.1966 | 0.4323 | 0.2603 | 0.2724 |
| fund_revenue_growth | bear | 0.0344 | 0.1430 | 0.2406 | 0.2603 | 0.1516 |

### 昨日炸板

- **Neutral**: ['volatility', 'trend_lowvol', 'momentum_reversal'] (单因子IC=0.074, 组合IC=0.0855)
  - weights: [0.3496, 0.3493, 0.3011]
- **Bull**: ['trend_lowvol', 'fund_pb', 'volatility'] (单因子IC=0.0801, 组合IC=0.0992)
  - bull_weights: [0.3611, 0.3529, 0.2861]
- **Bear**: ['fund_roe'] (单因子IC=0.1775, 组合IC=0.1775)
  - bear_weights: [1.0]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| volatility | neutral | 0.0753 | 0.2453 | 0.3067 | 0.3038 | 0.2000 |
| trend_lowvol | neutral | 0.0780 | 0.2465 | 0.3164 | 0.2630 | 0.1998 |
| momentum_reversal | neutral | 0.0688 | 0.2458 | 0.2802 | 0.2296 | 0.1722 |
| rsi_vol_combo | neutral | 0.0623 | 0.2303 | 0.2706 | 0.2015 | 0.1625 |
| fund_pb | neutral | 0.0490 | 0.1853 | 0.2645 | 0.2140 | 0.1605 |
| mom_x_lowvol_20_20 | neutral | 0.0644 | 0.2443 | 0.2635 | 0.2067 | 0.1590 |
| fund_pe | neutral | 0.0594 | 0.2256 | 0.2634 | 0.1973 | 0.1577 |
| turnover_stability | neutral | 0.0496 | 0.1937 | 0.2560 | 0.1994 | 0.1535 |
| fund_profit_growth | neutral | 0.0412 | 0.2246 | 0.1834 | 0.1190 | 0.1026 |
| fund_score | neutral | 0.0420 | 0.2483 | 0.1692 | 0.1273 | 0.0954 |
| fund_roe | neutral | 0.0342 | 0.2624 | 0.1305 | 0.1096 | 0.0724 |
| trend_lowvol | bull | 0.0854 | 0.1934 | 0.4414 | 0.3106 | 0.2893 |
| fund_pb | bull | 0.0732 | 0.1678 | 0.4365 | 0.2955 | 0.2827 |
| volatility | bull | 0.0817 | 0.2336 | 0.3497 | 0.3106 | 0.2292 |
| mom_x_lowvol_20_20 | bull | 0.0508 | 0.2113 | 0.2406 | 0.2045 | 0.1449 |
| rsi_vol_combo | bull | 0.0436 | 0.2058 | 0.2121 | 0.1667 | 0.1237 |
| low_downside | bull | 0.0377 | 0.1850 | 0.2038 | 0.2045 | 0.1227 |
| momentum_reversal | bull | 0.0423 | 0.2139 | 0.1976 | 0.1894 | 0.1175 |
| fund_pe | bull | 0.0315 | 0.2507 | 0.1255 | 0.1515 | 0.0723 |
| wash_sale_score | bull | 0.0261 | 0.2061 | 0.1267 | 0.1349 | 0.0719 |
| turnover_stability | bull | 0.0207 | 0.1789 | 0.1159 | 0.1061 | 0.0641 |
| fund_roe | bear | 0.1775 | 0.2583 | 0.6873 | 0.5068 | 0.5178 |
| fund_score | bear | 0.1423 | 0.2446 | 0.5819 | 0.5068 | 0.4384 |
| fund_revenue_growth | bear | 0.1244 | 0.1993 | 0.6244 | 0.3151 | 0.4106 |
| momentum_reversal | bear | 0.1283 | 0.2393 | 0.5359 | 0.2603 | 0.3377 |
| trend_lowvol | bear | 0.0984 | 0.2235 | 0.4404 | 0.2740 | 0.2805 |
| mom_x_lowvol_20_20 | bear | 0.1057 | 0.2372 | 0.4456 | 0.2329 | 0.2747 |
| rsi_vol_combo | bear | 0.0813 | 0.2059 | 0.3950 | 0.3151 | 0.2597 |
| fund_gross_margin | bear | 0.0756 | 0.1967 | 0.3845 | 0.2329 | 0.2370 |
| bb_width_20 | bear | 0.0653 | 0.2519 | 0.2594 | 0.2329 | 0.1599 |
| volatility | bear | 0.0325 | 0.1792 | 0.1814 | 0.1233 | 0.1019 |

### 昨日触板

- **Neutral**: ['trend_lowvol', 'volatility', 'turnover_stability'] (单因子IC=0.0719, 组合IC=0.0974)
  - weights: [0.371, 0.3356, 0.2934]
- **Bull**: ['fund_pb', 'trend_lowvol', 'volatility'] (单因子IC=0.0781, 组合IC=0.0917)
  - bull_weights: [0.3726, 0.3428, 0.2846]
- **Bear**: ['fund_revenue_growth', 'trend_lowvol', 'momentum_reversal'] (单因子IC=0.1333, 组合IC=0.1616)
  - bear_weights: [0.3784, 0.3335, 0.2881]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| trend_lowvol | neutral | 0.0841 | 0.2359 | 0.3565 | 0.2944 | 0.2307 |
| volatility | neutral | 0.0772 | 0.2363 | 0.3267 | 0.2777 | 0.2087 |
| turnover_stability | neutral | 0.0545 | 0.1814 | 0.3004 | 0.2150 | 0.1825 |
| momentum_reversal | neutral | 0.0702 | 0.2420 | 0.2900 | 0.2255 | 0.1777 |
| mom_x_lowvol_20_20 | neutral | 0.0678 | 0.2407 | 0.2818 | 0.2411 | 0.1749 |
| fund_pb | neutral | 0.0492 | 0.1738 | 0.2830 | 0.2296 | 0.1740 |
| fund_pe | neutral | 0.0570 | 0.2182 | 0.2612 | 0.1921 | 0.1557 |
| rsi_vol_combo | neutral | 0.0551 | 0.2252 | 0.2446 | 0.2171 | 0.1488 |
| fund_profit_growth | neutral | 0.0450 | 0.2012 | 0.2237 | 0.1649 | 0.1303 |
| fund_score | neutral | 0.0482 | 0.2245 | 0.2147 | 0.1681 | 0.1254 |
| low_downside | neutral | 0.0432 | 0.2201 | 0.1961 | 0.1127 | 0.1091 |
| fund_roe | neutral | 0.0377 | 0.2294 | 0.1643 | 0.1649 | 0.0957 |
| fund_pb | bull | 0.0739 | 0.1631 | 0.4527 | 0.3258 | 0.3001 |
| trend_lowvol | bull | 0.0819 | 0.1922 | 0.4263 | 0.2955 | 0.2761 |
| volatility | bull | 0.0784 | 0.2178 | 0.3602 | 0.2727 | 0.2292 |
| wash_sale_score | bull | 0.0476 | 0.1826 | 0.2609 | 0.1726 | 0.1530 |
| mom_x_lowvol_20_20 | bull | 0.0489 | 0.2000 | 0.2443 | 0.1970 | 0.1462 |
| low_downside | bull | 0.0423 | 0.1798 | 0.2354 | 0.1591 | 0.1364 |
| turnover_stability | bull | 0.0333 | 0.1601 | 0.2077 | 0.1515 | 0.1196 |
| momentum_reversal | bull | 0.0369 | 0.2086 | 0.1767 | 0.1515 | 0.1017 |
| rsi_vol_combo | bull | 0.0369 | 0.2067 | 0.1785 | 0.1288 | 0.1007 |
| fund_revenue_growth | bear | 0.1261 | 0.1796 | 0.7022 | 0.5068 | 0.5291 |
| trend_lowvol | bear | 0.1306 | 0.2111 | 0.6189 | 0.5068 | 0.4663 |
| momentum_reversal | bear | 0.1431 | 0.2336 | 0.6127 | 0.3151 | 0.4029 |
| mom_x_lowvol_20_20 | bear | 0.1287 | 0.2349 | 0.5478 | 0.3973 | 0.3827 |
| fund_roe | bear | 0.1317 | 0.2604 | 0.5055 | 0.2740 | 0.3220 |
| rsi_vol_combo | bear | 0.0819 | 0.1939 | 0.4223 | 0.3973 | 0.2950 |
| fund_score | bear | 0.1094 | 0.2459 | 0.4450 | 0.2329 | 0.2743 |
| fund_gross_margin | bear | 0.0578 | 0.1826 | 0.3163 | 0.2329 | 0.1950 |
| fund_pb | bear | 0.0376 | 0.1731 | 0.2173 | 0.2329 | 0.1340 |
| bb_width_20 | bear | 0.0551 | 0.2433 | 0.2266 | 0.1233 | 0.1273 |

### 昨日首板

- **Neutral**: ['trend_lowvol', 'mom_x_lowvol_20_20'] (单因子IC=0.0854, 组合IC=0.1011)
  - weights: [0.5067, 0.4933]
- **Bull**: ['low_downside', 'volatility', 'fund_pb'] (单因子IC=0.0953, 组合IC=0.1067)
  - bull_weights: [0.3504, 0.327, 0.3226]
- **Bear**: ['trend_lowvol', 'rsi_vol_combo'] (单因子IC=0.092, 组合IC=0.1076)
  - bear_weights: [0.559, 0.441]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| trend_lowvol | neutral | 0.0899 | 0.1985 | 0.4529 | 0.3674 | 0.3096 |
| mom_x_lowvol_20_20 | neutral | 0.0810 | 0.1786 | 0.4533 | 0.3299 | 0.3014 |
| momentum_reversal | neutral | 0.0807 | 0.1800 | 0.4484 | 0.3215 | 0.2963 |
| volatility | neutral | 0.0640 | 0.1769 | 0.3617 | 0.2923 | 0.2337 |
| rsi_vol_combo | neutral | 0.0600 | 0.1765 | 0.3396 | 0.2651 | 0.2148 |
| fund_pe | neutral | 0.0641 | 0.1910 | 0.3357 | 0.2401 | 0.2082 |
| fund_pb | neutral | 0.0555 | 0.1694 | 0.3277 | 0.1983 | 0.1964 |
| fund_profit_growth | neutral | 0.0429 | 0.1454 | 0.2952 | 0.2610 | 0.1861 |
| turnover_stability | neutral | 0.0316 | 0.1199 | 0.2638 | 0.2401 | 0.1635 |
| low_downside | neutral | 0.0419 | 0.1833 | 0.2283 | 0.2317 | 0.1406 |
| fund_score | neutral | 0.0359 | 0.1682 | 0.2135 | 0.1816 | 0.1261 |
| fund_revenue_growth | neutral | 0.0288 | 0.1506 | 0.1912 | 0.2025 | 0.1149 |
| fund_roe | neutral | 0.0297 | 0.1676 | 0.1773 | 0.1106 | 0.0985 |
| low_downside | bull | 0.1003 | 0.1613 | 0.6216 | 0.4015 | 0.4356 |
| volatility | bull | 0.1034 | 0.1754 | 0.5896 | 0.3788 | 0.4064 |
| fund_pb | bull | 0.0820 | 0.1503 | 0.5457 | 0.4697 | 0.4010 |
| trend_lowvol | bull | 0.0699 | 0.2030 | 0.3442 | 0.2652 | 0.2177 |
| momentum_reversal | bull | 0.0422 | 0.1625 | 0.2593 | 0.1364 | 0.1474 |
| turnover_stability | bull | 0.0308 | 0.1257 | 0.2451 | 0.1667 | 0.1430 |
| mom_x_lowvol_20_20 | bull | 0.0335 | 0.1563 | 0.2141 | 0.1364 | 0.1217 |
| fund_pe | bull | 0.0379 | 0.1792 | 0.2114 | 0.1439 | 0.1209 |
| rsi_vol_combo | bull | 0.0301 | 0.1612 | 0.1868 | 0.1591 | 0.1083 |
| stroke_phase | bull | 0.0212 | 0.1187 | 0.1786 | 0.1742 | 0.1049 |
| fund_score | bull | 0.0223 | 0.1726 | 0.1292 | 0.1515 | 0.0744 |
| fund_revenue_growth | bull | 0.0140 | 0.1430 | 0.0981 | 0.1591 | 0.0568 |
| trend_lowvol | bear | 0.1066 | 0.2296 | 0.4641 | 0.3425 | 0.3115 |
| rsi_vol_combo | bear | 0.0775 | 0.1901 | 0.4077 | 0.2055 | 0.2458 |
| momentum_reversal | bear | 0.0879 | 0.2231 | 0.3940 | 0.2329 | 0.2429 |
| mom_x_lowvol_20_20 | bear | 0.0817 | 0.2343 | 0.3486 | 0.2329 | 0.2149 |
| fund_pb | bear | 0.0581 | 0.1643 | 0.3540 | 0.1233 | 0.1988 |
| turnover_stability | bear | 0.0319 | 0.1098 | 0.2909 | 0.3151 | 0.1912 |

### 显示技术

- **Neutral**: ['momentum_reversal', 'mom_x_lowvol_20_20', 'trend_lowvol'] (单因子IC=0.0771, 组合IC=0.0854)
  - weights: [0.3524, 0.3376, 0.31]
- **Bull**: ['volatility'] (单因子IC=0.0912, 组合IC=0.0912)
  - bull_weights: [1.0]
- **Bear**: ['fund_profit_growth', 'fund_gross_margin', 'trend_lowvol'] (单因子IC=0.0964, 组合IC=0.1179)
  - bear_weights: [0.3645, 0.3556, 0.2799]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| momentum_reversal | neutral | 0.0776 | 0.1994 | 0.3894 | 0.3090 | 0.2548 |
| mom_x_lowvol_20_20 | neutral | 0.0735 | 0.1948 | 0.3772 | 0.2944 | 0.2441 |
| trend_lowvol | neutral | 0.0802 | 0.2256 | 0.3556 | 0.2610 | 0.2242 |
| rsi_vol_combo | neutral | 0.0656 | 0.1990 | 0.3294 | 0.2797 | 0.2108 |
| fund_score | neutral | 0.0578 | 0.1915 | 0.3020 | 0.2401 | 0.1872 |
| fund_pb | neutral | 0.0519 | 0.1753 | 0.2958 | 0.2150 | 0.1797 |
| fund_roe | neutral | 0.0478 | 0.1930 | 0.2476 | 0.1754 | 0.1455 |
| fund_profit_growth | neutral | 0.0426 | 0.1793 | 0.2373 | 0.2098 | 0.1435 |
| volatility | neutral | 0.0567 | 0.2384 | 0.2379 | 0.1639 | 0.1385 |
| fund_pe | neutral | 0.0418 | 0.1799 | 0.2326 | 0.1649 | 0.1355 |
| fund_revenue_growth | neutral | 0.0327 | 0.1679 | 0.1946 | 0.1378 | 0.1107 |
| turnover_stability | neutral | 0.0334 | 0.1824 | 0.1832 | 0.1795 | 0.1080 |
| fund_gross_margin | neutral | 0.0339 | 0.1900 | 0.1786 | 0.1837 | 0.1057 |
| volatility | bull | 0.0912 | 0.2416 | 0.3775 | 0.2652 | 0.2388 |
| trend_lowvol | bull | 0.0788 | 0.2309 | 0.3411 | 0.2803 | 0.2183 |
| fund_pe | bull | 0.0609 | 0.1961 | 0.3107 | 0.2576 | 0.1954 |
| mom_x_lowvol_20_20 | bull | 0.0676 | 0.2101 | 0.3218 | 0.2121 | 0.1951 |
| fund_pb | bull | 0.0644 | 0.2061 | 0.3124 | 0.2424 | 0.1941 |
| momentum_reversal | bull | 0.0604 | 0.2087 | 0.2894 | 0.2273 | 0.1776 |
| low_downside | bull | 0.0550 | 0.2083 | 0.2642 | 0.2424 | 0.1641 |
| rsi_vol_combo | bull | 0.0557 | 0.1973 | 0.2825 | 0.1515 | 0.1626 |
| turnover_stability | bull | 0.0309 | 0.1726 | 0.1787 | 0.1212 | 0.1002 |
| stroke_phase | bull | 0.0231 | 0.1670 | 0.1386 | 0.1742 | 0.0814 |
| fund_revenue_growth | bull | 0.0239 | 0.1755 | 0.1361 | 0.1439 | 0.0778 |
| fund_profit_growth | bear | 0.0903 | 0.1915 | 0.4714 | 0.4521 | 0.3423 |
| fund_gross_margin | bear | 0.0927 | 0.1902 | 0.4875 | 0.3699 | 0.3339 |
| trend_lowvol | bear | 0.1061 | 0.2766 | 0.3837 | 0.3699 | 0.2628 |
| momentum_reversal | bear | 0.1125 | 0.2659 | 0.4231 | 0.2329 | 0.2608 |
| turnover_stability | bear | 0.0578 | 0.1771 | 0.3262 | 0.3425 | 0.2190 |
| rsi_vol_combo | bear | 0.0566 | 0.1962 | 0.2884 | 0.1507 | 0.1659 |
| mom_x_lowvol_20_20 | bear | 0.0737 | 0.3165 | 0.2330 | 0.1233 | 0.1309 |

### 智慧城市

- **Neutral**: ['trend_lowvol', 'mom_x_lowvol_20_20'] (单因子IC=0.0903, 组合IC=0.1029)
  - weights: [0.5019, 0.4981]
- **Bull**: ['volatility', 'fund_pb', 'low_downside'] (单因子IC=0.1126, 组合IC=0.1353)
  - bull_weights: [0.3659, 0.3337, 0.3004]
- **Bear**: ['mom_x_lowvol_20_20'] (单因子IC=0.112, 组合IC=0.112)
  - bear_weights: [1.0]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| trend_lowvol | neutral | 0.0941 | 0.1779 | 0.5287 | 0.4447 | 0.3819 |
| mom_x_lowvol_20_20 | neutral | 0.0866 | 0.1593 | 0.5436 | 0.3946 | 0.3790 |
| momentum_reversal | neutral | 0.0859 | 0.1610 | 0.5337 | 0.3904 | 0.3710 |
| rsi_vol_combo | neutral | 0.0674 | 0.1508 | 0.4468 | 0.3299 | 0.2971 |
| volatility | neutral | 0.0757 | 0.1755 | 0.4312 | 0.3633 | 0.2939 |
| fund_pb | neutral | 0.0709 | 0.1556 | 0.4556 | 0.2818 | 0.2920 |
| fund_profit_growth | neutral | 0.0409 | 0.1133 | 0.3610 | 0.2276 | 0.2215 |
| turnover_stability | neutral | 0.0324 | 0.0984 | 0.3289 | 0.2610 | 0.2074 |
| low_downside | neutral | 0.0509 | 0.1771 | 0.2875 | 0.2902 | 0.1854 |
| fund_score | neutral | 0.0367 | 0.1500 | 0.2444 | 0.1649 | 0.1424 |
| fund_revenue_growth | neutral | 0.0235 | 0.0987 | 0.2382 | 0.1816 | 0.1407 |
| fund_pe | neutral | 0.0404 | 0.1675 | 0.2411 | 0.1608 | 0.1399 |
| volatility | bull | 0.1250 | 0.1653 | 0.7561 | 0.5833 | 0.5986 |
| fund_pb | bull | 0.1158 | 0.1559 | 0.7429 | 0.4697 | 0.5459 |
| low_downside | bull | 0.0970 | 0.1473 | 0.6586 | 0.4924 | 0.4915 |
| trend_lowvol | bull | 0.1027 | 0.1551 | 0.6621 | 0.4773 | 0.4891 |
| turnover_stability | bull | 0.0560 | 0.1094 | 0.5114 | 0.4773 | 0.3777 |
| mom_x_lowvol_20_20 | bull | 0.0758 | 0.1483 | 0.5112 | 0.4091 | 0.3602 |
| momentum_reversal | bull | 0.0717 | 0.1477 | 0.4855 | 0.3258 | 0.3219 |
| fund_pe | bull | 0.0676 | 0.1628 | 0.4151 | 0.3030 | 0.2704 |
| fund_revenue_growth | bull | 0.0433 | 0.1138 | 0.3800 | 0.1818 | 0.2246 |
| fund_profit_growth | bull | 0.0351 | 0.0944 | 0.3721 | 0.2045 | 0.2241 |
| fund_score | bull | 0.0493 | 0.1533 | 0.3214 | 0.1364 | 0.1826 |
| stroke_phase | bull | 0.0312 | 0.1098 | 0.2840 | 0.2500 | 0.1775 |
| rsi_vol_combo | bull | 0.0400 | 0.1457 | 0.2742 | 0.1894 | 0.1631 |
| mom_x_lowvol_20_20 | bear | 0.1120 | 0.1626 | 0.6888 | 0.5342 | 0.5284 |
| trend_lowvol | bear | 0.0831 | 0.1336 | 0.6218 | 0.5068 | 0.4685 |
| momentum_reversal | bear | 0.1020 | 0.1622 | 0.6286 | 0.4521 | 0.4564 |
| fund_profit_growth | bear | 0.0604 | 0.1298 | 0.4651 | 0.3699 | 0.3185 |
| turnover_stability | bear | 0.0436 | 0.1071 | 0.4071 | 0.3425 | 0.2733 |
| rsi_vol_combo | bear | 0.0580 | 0.1389 | 0.4178 | 0.2603 | 0.2633 |
| fund_revenue_growth | bear | 0.0361 | 0.1048 | 0.3443 | 0.1781 | 0.2028 |
| fund_score | bear | 0.0519 | 0.1561 | 0.3326 | 0.1233 | 0.1868 |

### 智慧政务

- **Neutral**: ['momentum_reversal', 'trend_lowvol'] (单因子IC=0.1035, 组合IC=0.1137)
  - weights: [0.5085, 0.4915]
- **Bull**: ['volatility'] (单因子IC=0.1527, 组合IC=0.1527)
  - bull_weights: [1.0]
- **Bear**: ['fund_profit_growth', 'mom_x_lowvol_20_20', 'momentum_reversal'] (单因子IC=0.1295, 组合IC=0.1704)
  - bear_weights: [0.3505, 0.3275, 0.322]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| momentum_reversal | neutral | 0.1035 | 0.1840 | 0.5626 | 0.3925 | 0.3917 |
| trend_lowvol | neutral | 0.1034 | 0.1917 | 0.5397 | 0.4029 | 0.3786 |
| mom_x_lowvol_20_20 | neutral | 0.0967 | 0.1905 | 0.5076 | 0.3779 | 0.3497 |
| volatility | neutral | 0.0937 | 0.2024 | 0.4630 | 0.3779 | 0.3190 |
| fund_pb | neutral | 0.0756 | 0.1632 | 0.4631 | 0.3038 | 0.3019 |
| rsi_vol_combo | neutral | 0.0739 | 0.1710 | 0.4322 | 0.3528 | 0.2924 |
| low_downside | neutral | 0.0582 | 0.1819 | 0.3198 | 0.2568 | 0.2010 |
| fund_profit_growth | neutral | 0.0429 | 0.1375 | 0.3117 | 0.2171 | 0.1897 |
| turnover_stability | neutral | 0.0414 | 0.1402 | 0.2955 | 0.1921 | 0.1761 |
| fund_score | neutral | 0.0478 | 0.1676 | 0.2855 | 0.2192 | 0.1740 |
| fund_pe | neutral | 0.0420 | 0.1930 | 0.2174 | 0.1441 | 0.1244 |
| fund_revenue_growth | neutral | 0.0304 | 0.1469 | 0.2067 | 0.1775 | 0.1217 |
| volatility | bull | 0.1527 | 0.1766 | 0.8647 | 0.5985 | 0.6911 |
| fund_pb | bull | 0.1185 | 0.1413 | 0.8385 | 0.5985 | 0.6701 |
| low_downside | bull | 0.1081 | 0.1561 | 0.6924 | 0.4470 | 0.5009 |
| mom_x_lowvol_20_20 | bull | 0.1185 | 0.1765 | 0.6714 | 0.4848 | 0.4985 |
| momentum_reversal | bull | 0.1101 | 0.1722 | 0.6396 | 0.4545 | 0.4651 |
| trend_lowvol | bull | 0.1161 | 0.1890 | 0.6141 | 0.4848 | 0.4560 |
| fund_pe | bull | 0.1131 | 0.1883 | 0.6006 | 0.3864 | 0.4163 |
| rsi_vol_combo | bull | 0.0689 | 0.1794 | 0.3840 | 0.2879 | 0.2473 |
| turnover_stability | bull | 0.0424 | 0.1390 | 0.3050 | 0.3106 | 0.1998 |
| fund_roe | bull | 0.0224 | 0.1590 | 0.1409 | 0.1061 | 0.0779 |
| fund_profit_growth | bear | 0.1150 | 0.1728 | 0.6652 | 0.5890 | 0.5286 |
| mom_x_lowvol_20_20 | bear | 0.1345 | 0.2052 | 0.6556 | 0.5068 | 0.4939 |
| momentum_reversal | bear | 0.1392 | 0.2081 | 0.6688 | 0.4521 | 0.4856 |
| trend_lowvol | bear | 0.1158 | 0.1813 | 0.6387 | 0.4521 | 0.4637 |
| rsi_vol_combo | bear | 0.0968 | 0.1905 | 0.5080 | 0.3699 | 0.3480 |
| fund_gross_margin | bear | 0.0461 | 0.1524 | 0.3024 | 0.2055 | 0.1822 |
| fund_score | bear | 0.0545 | 0.1848 | 0.2950 | 0.2329 | 0.1819 |

### 智慧灯杆

- **Neutral**: ['trend_lowvol', 'volatility', 'fund_pb'] (单因子IC=0.076, 组合IC=0.0922)
  - weights: [0.3637, 0.3281, 0.3081]
- **Bull**: ['turnover_stability', 'volatility'] (单因子IC=0.0853, 组合IC=0.1018)
  - bull_weights: [0.5038, 0.4962]
- **Bear**: ['fund_revenue_growth', 'fund_score'] (单因子IC=0.1542, 组合IC=0.1818)
  - bear_weights: [0.5924, 0.4076]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| trend_lowvol | neutral | 0.0835 | 0.2523 | 0.3308 | 0.2484 | 0.2065 |
| volatility | neutral | 0.0750 | 0.2471 | 0.3037 | 0.2265 | 0.1862 |
| fund_pb | neutral | 0.0694 | 0.2431 | 0.2855 | 0.2255 | 0.1749 |
| fund_profit_growth | neutral | 0.0543 | 0.1999 | 0.2714 | 0.2432 | 0.1687 |
| mom_x_lowvol_20_20 | neutral | 0.0683 | 0.2460 | 0.2776 | 0.1420 | 0.1585 |
| momentum_reversal | neutral | 0.0639 | 0.2480 | 0.2578 | 0.1576 | 0.1492 |
| rsi_vol_combo | neutral | 0.0601 | 0.2462 | 0.2442 | 0.1660 | 0.1424 |
| fund_revenue_growth | neutral | 0.0458 | 0.2000 | 0.2290 | 0.1795 | 0.1351 |
| low_downside | neutral | 0.0569 | 0.2399 | 0.2371 | 0.1096 | 0.1315 |
| fund_score | neutral | 0.0364 | 0.2117 | 0.1720 | 0.1545 | 0.0993 |
| turnover_stability | neutral | 0.0320 | 0.2231 | 0.1433 | 0.1347 | 0.0813 |
| turnover_stability | bull | 0.0813 | 0.2083 | 0.3903 | 0.3106 | 0.2558 |
| volatility | bull | 0.0893 | 0.2350 | 0.3800 | 0.3258 | 0.2519 |
| trend_lowvol | bull | 0.0800 | 0.2288 | 0.3497 | 0.2197 | 0.2132 |
| low_downside | bull | 0.0700 | 0.2232 | 0.3133 | 0.2273 | 0.1923 |
| fund_pb | bull | 0.0678 | 0.2539 | 0.2672 | 0.2576 | 0.1680 |
| wash_sale_score | bull | 0.0592 | 0.2185 | 0.2711 | 0.2238 | 0.1659 |
| fund_pe | bull | 0.0678 | 0.2741 | 0.2473 | 0.2348 | 0.1527 |
| fund_profit_growth | bull | 0.0475 | 0.1993 | 0.2382 | 0.1515 | 0.1371 |
| rsi_vol_combo | bull | 0.0518 | 0.2346 | 0.2207 | 0.1818 | 0.1304 |
| fund_revenue_growth | bull | 0.0426 | 0.2164 | 0.1970 | 0.2121 | 0.1194 |
| fund_revenue_growth | bear | 0.1639 | 0.2100 | 0.7802 | 0.6712 | 0.6520 |
| fund_score | bear | 0.1446 | 0.2429 | 0.5953 | 0.5068 | 0.4485 |
| fund_profit_growth | bear | 0.1328 | 0.2345 | 0.5663 | 0.5616 | 0.4422 |
| fund_roe | bear | 0.0751 | 0.2225 | 0.3373 | 0.2329 | 0.2079 |

### 智能家居

- **Neutral**: ['trend_lowvol', 'fund_profit_growth', 'fund_pb'] (单因子IC=0.0648, 组合IC=0.0854)
  - weights: [0.3515, 0.3459, 0.3026]
- **Bull**: ['low_downside', 'volatility', 'fund_pb'] (单因子IC=0.0564, 组合IC=0.071)
  - bull_weights: [0.3881, 0.3249, 0.287]
- **Bear**: ['momentum_reversal', 'mom_x_lowvol_20_20', 'trend_lowvol'] (单因子IC=0.1314, 组合IC=0.1431)
  - bear_weights: [0.3659, 0.3187, 0.3153]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| trend_lowvol | neutral | 0.0779 | 0.1988 | 0.3918 | 0.2568 | 0.2462 |
| fund_profit_growth | neutral | 0.0532 | 0.1420 | 0.3744 | 0.2944 | 0.2423 |
| fund_pb | neutral | 0.0635 | 0.1860 | 0.3413 | 0.2422 | 0.2120 |
| mom_x_lowvol_20_20 | neutral | 0.0580 | 0.1758 | 0.3299 | 0.2276 | 0.2025 |
| momentum_reversal | neutral | 0.0555 | 0.1828 | 0.3037 | 0.2088 | 0.1836 |
| fund_pe | neutral | 0.0539 | 0.2156 | 0.2501 | 0.1670 | 0.1459 |
| volatility | neutral | 0.0420 | 0.1872 | 0.2244 | 0.1357 | 0.1274 |
| fund_score | neutral | 0.0372 | 0.1761 | 0.2110 | 0.1545 | 0.1218 |
| rsi_vol_combo | neutral | 0.0334 | 0.1736 | 0.1923 | 0.1587 | 0.1114 |
| low_downside | neutral | 0.0302 | 0.1812 | 0.1669 | 0.1232 | 0.0937 |
| fund_revenue_growth | neutral | 0.0202 | 0.1434 | 0.1410 | 0.1211 | 0.0790 |
| low_downside | bull | 0.0544 | 0.1540 | 0.3532 | 0.3258 | 0.2341 |
| volatility | bull | 0.0550 | 0.1797 | 0.3061 | 0.2803 | 0.1960 |
| fund_pb | bull | 0.0598 | 0.2065 | 0.2893 | 0.1970 | 0.1732 |
| stroke_phase | bull | 0.0302 | 0.1253 | 0.2407 | 0.1742 | 0.1413 |
| trend_lowvol | bull | 0.0399 | 0.1970 | 0.2023 | 0.2045 | 0.1219 |
| momentum_reversal | bull | 0.0220 | 0.1734 | 0.1266 | 0.1742 | 0.0744 |
| fund_profit_growth | bull | 0.0173 | 0.1346 | 0.1287 | 0.1364 | 0.0731 |
| momentum_reversal | bear | 0.1377 | 0.2017 | 0.6825 | 0.3973 | 0.4768 |
| mom_x_lowvol_20_20 | bear | 0.1341 | 0.2212 | 0.6064 | 0.3699 | 0.4153 |
| trend_lowvol | bear | 0.1223 | 0.2039 | 0.5999 | 0.3699 | 0.4109 |
| rsi_vol_combo | bear | 0.0868 | 0.1831 | 0.4738 | 0.3699 | 0.3245 |
| turnover_stability | bear | 0.0482 | 0.1011 | 0.4768 | 0.3425 | 0.3201 |
| fund_revenue_growth | bear | 0.0470 | 0.1602 | 0.2934 | 0.1233 | 0.1648 |
| fund_pb | bear | 0.0513 | 0.2207 | 0.2323 | 0.1233 | 0.1305 |

### 智能电网

- **Neutral**: ['fund_pb', 'momentum_reversal', 'mom_x_lowvol_20_20'] (单因子IC=0.0792, 组合IC=0.1109)
  - weights: [0.3947, 0.3029, 0.3024]
- **Bull**: ['fund_pb', 'volatility', 'low_downside'] (单因子IC=0.0895, 组合IC=0.1101)
  - bull_weights: [0.3888, 0.319, 0.2922]
- **Bear**: ['momentum_reversal', 'mom_x_lowvol_20_20'] (单因子IC=0.1238, 组合IC=0.1256)
  - bear_weights: [0.5233, 0.4767]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| fund_pb | neutral | 0.0731 | 0.1276 | 0.5728 | 0.4113 | 0.4042 |
| momentum_reversal | neutral | 0.0841 | 0.1836 | 0.4579 | 0.3549 | 0.3102 |
| mom_x_lowvol_20_20 | neutral | 0.0804 | 0.1768 | 0.4551 | 0.3612 | 0.3097 |
| volatility | neutral | 0.0769 | 0.1893 | 0.4059 | 0.3674 | 0.2775 |
| rsi_vol_combo | neutral | 0.0702 | 0.1681 | 0.4178 | 0.3215 | 0.2761 |
| fund_profit_growth | neutral | 0.0534 | 0.1255 | 0.4255 | 0.2881 | 0.2741 |
| fund_pe | neutral | 0.0604 | 0.1549 | 0.3898 | 0.3257 | 0.2584 |
| trend_lowvol | neutral | 0.0739 | 0.2018 | 0.3660 | 0.2756 | 0.2334 |
| fund_score | neutral | 0.0523 | 0.1585 | 0.3304 | 0.2463 | 0.2059 |
| fund_revenue_growth | neutral | 0.0328 | 0.1207 | 0.2715 | 0.2276 | 0.1666 |
| low_downside | neutral | 0.0477 | 0.1872 | 0.2549 | 0.2380 | 0.1578 |
| turnover_stability | neutral | 0.0282 | 0.1130 | 0.2492 | 0.1712 | 0.1459 |
| fund_roe | neutral | 0.0314 | 0.1695 | 0.1855 | 0.1399 | 0.1057 |
| fund_pb | bull | 0.0897 | 0.1258 | 0.7128 | 0.5000 | 0.5346 |
| volatility | bull | 0.0984 | 0.1631 | 0.6031 | 0.4545 | 0.4387 |
| low_downside | bull | 0.0805 | 0.1435 | 0.5612 | 0.4318 | 0.4018 |
| fund_pe | bull | 0.0780 | 0.1419 | 0.5495 | 0.4318 | 0.3934 |
| turnover_stability | bull | 0.0357 | 0.1102 | 0.3240 | 0.2424 | 0.2013 |
| fund_score | bull | 0.0412 | 0.1440 | 0.2863 | 0.1061 | 0.1583 |
| fund_profit_growth | bull | 0.0316 | 0.1134 | 0.2789 | 0.1212 | 0.1563 |
| fund_roe | bull | 0.0416 | 0.1658 | 0.2506 | 0.1818 | 0.1481 |
| fund_revenue_growth | bull | 0.0252 | 0.1065 | 0.2371 | 0.1136 | 0.1320 |
| trend_lowvol | bull | 0.0417 | 0.1802 | 0.2311 | 0.1136 | 0.1287 |
| fund_gross_margin | bull | 0.0155 | 0.1120 | 0.1385 | 0.1439 | 0.0792 |
| momentum_reversal | bear | 0.1235 | 0.1850 | 0.6678 | 0.5890 | 0.5306 |
| mom_x_lowvol_20_20 | bear | 0.1240 | 0.1934 | 0.6415 | 0.5068 | 0.4833 |
| rsi_vol_combo | bear | 0.0897 | 0.1539 | 0.5826 | 0.4521 | 0.4230 |
| trend_lowvol | bear | 0.1026 | 0.1889 | 0.5431 | 0.3699 | 0.3720 |
| fund_pb | bear | 0.0702 | 0.1588 | 0.4423 | 0.2603 | 0.2787 |
| fund_profit_growth | bear | 0.0573 | 0.1373 | 0.4174 | 0.2055 | 0.2516 |
| fund_pe | bear | 0.0654 | 0.1734 | 0.3769 | 0.2603 | 0.2375 |
| bb_width_20 | bear | 0.0335 | 0.1651 | 0.2029 | 0.1781 | 0.1195 |

### 智能穿戴

- **Neutral**: ['fund_pb', 'momentum_reversal', 'mom_x_lowvol_20_20'] (单因子IC=0.085, 组合IC=0.1167)
  - weights: [0.4131, 0.2945, 0.2924]
- **Bull**: ['fund_pb', 'turnover_stability', 'volatility'] (单因子IC=0.0569, 组合IC=0.0851)
  - bull_weights: [0.4261, 0.3012, 0.2726]
- **Bear**: ['momentum_reversal'] (单因子IC=0.1108, 组合IC=0.1108)
  - bear_weights: [1.0]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| fund_pb | neutral | 0.0817 | 0.1356 | 0.6020 | 0.4447 | 0.4348 |
| momentum_reversal | neutral | 0.0882 | 0.1856 | 0.4752 | 0.3048 | 0.3100 |
| mom_x_lowvol_20_20 | neutral | 0.0853 | 0.1816 | 0.4696 | 0.3111 | 0.3078 |
| rsi_vol_combo | neutral | 0.0703 | 0.1689 | 0.4159 | 0.2965 | 0.2696 |
| trend_lowvol | neutral | 0.0808 | 0.2038 | 0.3968 | 0.2422 | 0.2464 |
| fund_pe | neutral | 0.0530 | 0.1626 | 0.3262 | 0.2463 | 0.2033 |
| volatility | neutral | 0.0601 | 0.1822 | 0.3298 | 0.2255 | 0.2021 |
| fund_profit_growth | neutral | 0.0381 | 0.1517 | 0.2511 | 0.2150 | 0.1526 |
| fund_score | neutral | 0.0367 | 0.1891 | 0.1943 | 0.1127 | 0.1081 |
| low_downside | neutral | 0.0306 | 0.1717 | 0.1783 | 0.1670 | 0.1040 |
| turnover_stability | neutral | 0.0200 | 0.1215 | 0.1648 | 0.1524 | 0.0950 |
| fund_pb | bull | 0.0719 | 0.1532 | 0.4696 | 0.3333 | 0.3130 |
| turnover_stability | bull | 0.0387 | 0.1134 | 0.3417 | 0.2955 | 0.2213 |
| volatility | bull | 0.0599 | 0.1859 | 0.3224 | 0.2424 | 0.2003 |
| fund_profit_growth | bull | 0.0414 | 0.1382 | 0.2997 | 0.3030 | 0.1953 |
| fund_revenue_growth | bull | 0.0452 | 0.1484 | 0.3046 | 0.2500 | 0.1904 |
| low_downside | bull | 0.0512 | 0.1649 | 0.3105 | 0.2197 | 0.1893 |
| trend_lowvol | bull | 0.0485 | 0.2093 | 0.2316 | 0.2197 | 0.1413 |
| fund_pe | bull | 0.0396 | 0.1677 | 0.2363 | 0.1591 | 0.1370 |
| fund_score | bull | 0.0323 | 0.1769 | 0.1826 | 0.1288 | 0.1031 |
| mom_x_lowvol_20_20 | bull | 0.0280 | 0.1749 | 0.1600 | 0.1818 | 0.0945 |
| momentum_reversal | bull | 0.0249 | 0.1771 | 0.1404 | 0.1667 | 0.0819 |
| momentum_reversal | bear | 0.1108 | 0.1923 | 0.5762 | 0.4247 | 0.4104 |
| mom_x_lowvol_20_20 | bear | 0.1069 | 0.2044 | 0.5233 | 0.3699 | 0.3584 |
| rsi_vol_combo | bear | 0.0602 | 0.1401 | 0.4296 | 0.3699 | 0.2942 |
| bb_width_20 | bear | 0.0545 | 0.1467 | 0.3717 | 0.2329 | 0.2291 |
| turnover_stability | bear | 0.0437 | 0.1145 | 0.3816 | 0.1781 | 0.2248 |
| fund_pb | bear | 0.0403 | 0.1168 | 0.3446 | 0.1781 | 0.2030 |
| trend_lowvol | bear | 0.0660 | 0.2383 | 0.2771 | 0.1233 | 0.1556 |

### 智谱AI概念

- **Neutral**: ['momentum_reversal', 'fund_pb', 'trend_lowvol'] (单因子IC=0.1213, 组合IC=0.1774)
  - weights: [0.3484, 0.3301, 0.3215]
- **Bull**: ['fund_pb', 'trend_lowvol', 'wash_sale_score'] (单因子IC=0.1084, 组合IC=0.1532)
  - bull_weights: [0.3964, 0.329, 0.2745]
- **Bear**: ['fund_score', 'fund_profit_growth', 'trend_lowvol'] (单因子IC=0.1019, 组合IC=0.1613)
  - bear_weights: [0.3763, 0.3564, 0.2673]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| momentum_reversal | neutral | 0.1309 | 0.2452 | 0.5338 | 0.3946 | 0.3722 |
| fund_pb | neutral | 0.1139 | 0.2206 | 0.5161 | 0.3664 | 0.3526 |
| trend_lowvol | neutral | 0.1191 | 0.2399 | 0.4963 | 0.3841 | 0.3435 |
| volatility | neutral | 0.1132 | 0.2330 | 0.4861 | 0.4050 | 0.3415 |
| mom_x_lowvol_20_20 | neutral | 0.1140 | 0.2445 | 0.4662 | 0.3215 | 0.3080 |
| rsi_vol_combo | neutral | 0.0965 | 0.2411 | 0.4004 | 0.2985 | 0.2600 |
| fund_profit_growth | neutral | 0.0610 | 0.1939 | 0.3143 | 0.2610 | 0.1982 |
| fund_score | neutral | 0.0650 | 0.2163 | 0.3007 | 0.2610 | 0.1896 |
| fund_pe | neutral | 0.0741 | 0.2696 | 0.2750 | 0.2390 | 0.1703 |
| low_downside | neutral | 0.0603 | 0.2240 | 0.2690 | 0.2182 | 0.1638 |
| fund_roe | neutral | 0.0508 | 0.2229 | 0.2281 | 0.1962 | 0.1364 |
| fund_pb | bull | 0.1189 | 0.2003 | 0.5935 | 0.3712 | 0.4069 |
| trend_lowvol | bull | 0.1248 | 0.2520 | 0.4953 | 0.3636 | 0.3377 |
| wash_sale_score | bull | 0.0816 | 0.1908 | 0.4275 | 0.3184 | 0.2818 |
| volatility | bull | 0.0803 | 0.2216 | 0.3623 | 0.3636 | 0.2470 |
| fund_profit_growth | bull | 0.0526 | 0.1803 | 0.2918 | 0.1894 | 0.1735 |
| mom_x_lowvol_20_20 | bull | 0.0710 | 0.2397 | 0.2963 | 0.1667 | 0.1728 |
| momentum_reversal | bull | 0.0708 | 0.2601 | 0.2724 | 0.1591 | 0.1579 |
| fund_score | bull | 0.0447 | 0.1827 | 0.2449 | 0.2576 | 0.1540 |
| rsi_vol_combo | bull | 0.0601 | 0.2646 | 0.2270 | 0.1364 | 0.1290 |
| low_downside | bull | 0.0420 | 0.2142 | 0.1960 | 0.1742 | 0.1151 |
| stroke_phase | bull | 0.0324 | 0.2173 | 0.1490 | 0.1288 | 0.0841 |
| fund_score | bear | 0.1249 | 0.2311 | 0.5404 | 0.5068 | 0.4071 |
| fund_profit_growth | bear | 0.0941 | 0.1838 | 0.5118 | 0.5068 | 0.3856 |
| trend_lowvol | bear | 0.0867 | 0.1868 | 0.4640 | 0.2466 | 0.2892 |
| fund_roe | bear | 0.0673 | 0.2555 | 0.2632 | 0.2055 | 0.1586 |
| fund_pe | bear | 0.0668 | 0.2965 | 0.2251 | 0.2222 | 0.1376 |

### 最近多板

- **Neutral**: ['trend_lowvol', 'mom_x_lowvol_20_20'] (单因子IC=0.08, 组合IC=0.0949)
  - weights: [0.5124, 0.4876]
- **Bull**: ['low_downside', 'volatility', 'fund_pb'] (单因子IC=0.0934, 组合IC=0.1098)
  - bull_weights: [0.3802, 0.3377, 0.2821]
- **Bear**: ['turnover_stability', 'trend_lowvol', 'mom_x_lowvol_20_20'] (单因子IC=0.0824, 组合IC=0.0981)
  - bear_weights: [0.3724, 0.3478, 0.2799]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| trend_lowvol | neutral | 0.0868 | 0.1841 | 0.4717 | 0.3820 | 0.3260 |
| mom_x_lowvol_20_20 | neutral | 0.0732 | 0.1588 | 0.4608 | 0.3466 | 0.3102 |
| momentum_reversal | neutral | 0.0740 | 0.1636 | 0.4520 | 0.3299 | 0.3005 |
| volatility | neutral | 0.0699 | 0.1638 | 0.4269 | 0.3194 | 0.2816 |
| fund_pb | neutral | 0.0554 | 0.1482 | 0.3740 | 0.2484 | 0.2335 |
| turnover_stability | neutral | 0.0341 | 0.1001 | 0.3402 | 0.2797 | 0.2177 |
| fund_profit_growth | neutral | 0.0406 | 0.1271 | 0.3196 | 0.2965 | 0.2072 |
| rsi_vol_combo | neutral | 0.0521 | 0.1596 | 0.3268 | 0.2443 | 0.2033 |
| fund_pe | neutral | 0.0512 | 0.1646 | 0.3107 | 0.2443 | 0.1933 |
| low_downside | neutral | 0.0426 | 0.1654 | 0.2574 | 0.2171 | 0.1566 |
| fund_revenue_growth | neutral | 0.0300 | 0.1330 | 0.2252 | 0.1942 | 0.1345 |
| fund_score | neutral | 0.0359 | 0.1592 | 0.2256 | 0.1670 | 0.1316 |
| fund_roe | neutral | 0.0267 | 0.1650 | 0.1617 | 0.1420 | 0.0923 |
| low_downside | bull | 0.1021 | 0.1350 | 0.7567 | 0.5303 | 0.5790 |
| volatility | bull | 0.0986 | 0.1453 | 0.6788 | 0.5152 | 0.5142 |
| fund_pb | bull | 0.0793 | 0.1343 | 0.5906 | 0.4545 | 0.4295 |
| turnover_stability | bull | 0.0407 | 0.1067 | 0.3819 | 0.3182 | 0.2517 |
| trend_lowvol | bull | 0.0562 | 0.1754 | 0.3205 | 0.2576 | 0.2015 |
| mom_x_lowvol_20_20 | bull | 0.0348 | 0.1388 | 0.2504 | 0.2197 | 0.1527 |
| fund_pe | bull | 0.0420 | 0.1684 | 0.2492 | 0.1515 | 0.1435 |
| fund_profit_growth | bull | 0.0292 | 0.1180 | 0.2471 | 0.1591 | 0.1432 |
| momentum_reversal | bull | 0.0344 | 0.1505 | 0.2286 | 0.1894 | 0.1359 |
| fund_score | bull | 0.0321 | 0.1619 | 0.1981 | 0.2348 | 0.1223 |
| stroke_phase | bull | 0.0157 | 0.1023 | 0.1531 | 0.1212 | 0.0858 |
| rsi_vol_combo | bull | 0.0198 | 0.1391 | 0.1421 | 0.1061 | 0.0786 |
| fund_revenue_growth | bull | 0.0176 | 0.1271 | 0.1387 | 0.1136 | 0.0772 |
| fund_gross_margin | bull | 0.0128 | 0.1090 | 0.1176 | 0.1742 | 0.0690 |
| turnover_stability | bear | 0.0498 | 0.0819 | 0.6076 | 0.4795 | 0.4494 |
| trend_lowvol | bear | 0.1014 | 0.1688 | 0.6008 | 0.3973 | 0.4197 |
| mom_x_lowvol_20_20 | bear | 0.0960 | 0.1985 | 0.4835 | 0.3973 | 0.3378 |
| rsi_vol_combo | bear | 0.0699 | 0.1418 | 0.4931 | 0.3425 | 0.3310 |
| momentum_reversal | bear | 0.0938 | 0.1894 | 0.4952 | 0.3151 | 0.3256 |
| fund_pb | bear | 0.0588 | 0.1586 | 0.3705 | 0.1507 | 0.2132 |

### 有机硅概念

- **Neutral**: ['mom_x_lowvol_20_20', 'momentum_reversal', 'trend_lowvol'] (单因子IC=0.0902, 组合IC=0.0965)
  - weights: [0.3971, 0.3173, 0.2856]
- **Bull**: ['low_downside', 'trend_lowvol', 'volatility'] (单因子IC=0.0851, 组合IC=0.0996)
  - bull_weights: [0.3583, 0.3337, 0.308]
- **Bear**: ['mom_x_lowvol_20_20', 'vol_opening_strength'] (单因子IC=0.0835, 组合IC=0.1039)
  - bear_weights: [0.5014, 0.4986]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| mom_x_lowvol_20_20 | neutral | 0.0953 | 0.2351 | 0.4055 | 0.3100 | 0.2656 |
| momentum_reversal | neutral | 0.0831 | 0.2490 | 0.3339 | 0.2714 | 0.2122 |
| trend_lowvol | neutral | 0.0920 | 0.3015 | 0.3051 | 0.2526 | 0.1911 |
| fund_pb | neutral | 0.0619 | 0.2494 | 0.2483 | 0.2015 | 0.1492 |
| fund_score | neutral | 0.0577 | 0.2494 | 0.2315 | 0.2192 | 0.1411 |
| fund_profit_growth | neutral | 0.0504 | 0.2297 | 0.2196 | 0.1691 | 0.1284 |
| fund_pe | neutral | 0.0555 | 0.2685 | 0.2068 | 0.1399 | 0.1179 |
| volatility | neutral | 0.0537 | 0.2707 | 0.1985 | 0.1420 | 0.1133 |
| turnover_stability | neutral | 0.0366 | 0.2096 | 0.1747 | 0.1524 | 0.1006 |
| rsi_vol_combo | neutral | 0.0424 | 0.2410 | 0.1758 | 0.1232 | 0.0987 |
| fund_revenue_growth | neutral | 0.0332 | 0.2383 | 0.1392 | 0.1065 | 0.0770 |
| low_downside | neutral | 0.0342 | 0.2704 | 0.1264 | 0.1441 | 0.0723 |
| low_downside | bull | 0.0810 | 0.2061 | 0.3932 | 0.3258 | 0.2607 |
| trend_lowvol | bull | 0.0882 | 0.2410 | 0.3662 | 0.3258 | 0.2427 |
| volatility | bull | 0.0860 | 0.2399 | 0.3585 | 0.2500 | 0.2240 |
| fund_pb | bull | 0.0810 | 0.2349 | 0.3449 | 0.2576 | 0.2169 |
| rsi_vol_combo | bull | 0.0409 | 0.2179 | 0.1877 | 0.1591 | 0.1088 |
| stroke_phase | bull | 0.0317 | 0.1860 | 0.1704 | 0.1515 | 0.0981 |
| momentum_reversal | bull | 0.0420 | 0.2396 | 0.1753 | 0.1098 | 0.0973 |
| mom_x_lowvol_20_20 | bull | 0.0399 | 0.2554 | 0.1563 | 0.1742 | 0.0918 |
| fund_pe | bull | 0.0354 | 0.2291 | 0.1546 | 0.1439 | 0.0884 |
| mom_x_lowvol_20_20 | bear | 0.0960 | 0.2385 | 0.4027 | 0.3151 | 0.2648 |
| vol_opening_strength | bear | 0.0709 | 0.1884 | 0.3762 | 0.4000 | 0.2633 |
| vol_opening_confirm | bear | 0.0696 | 0.1898 | 0.3665 | 0.4000 | 0.2566 |
| trend_lowvol | bear | 0.0766 | 0.2389 | 0.3206 | 0.1781 | 0.1888 |
| momentum_reversal | bear | 0.0694 | 0.2491 | 0.2787 | 0.1781 | 0.1641 |
| fund_pb | bear | 0.0762 | 0.2994 | 0.2546 | 0.2055 | 0.1535 |

### 机器人执行器

- **Neutral**: ['momentum_reversal', 'mom_x_lowvol_20_20', 'rsi_vol_combo'] (单因子IC=0.0893, 组合IC=0.0981)
  - weights: [0.3626, 0.3282, 0.3092]
- **Bull**: ['volatility', 'fund_pe', 'wash_sale_score'] (单因子IC=0.0718, 组合IC=0.0987)
  - bull_weights: [0.464, 0.2923, 0.2437]
- **Bear**: ['mom_x_lowvol_20_20', 'trend_lowvol', 'fund_roe'] (单因子IC=0.0648, 组合IC=0.0825)
  - bear_weights: [0.4103, 0.3408, 0.2489]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| momentum_reversal | neutral | 0.0974 | 0.2448 | 0.3977 | 0.2589 | 0.2503 |
| mom_x_lowvol_20_20 | neutral | 0.0888 | 0.2478 | 0.3583 | 0.2651 | 0.2266 |
| rsi_vol_combo | neutral | 0.0819 | 0.2386 | 0.3431 | 0.2443 | 0.2135 |
| fund_pe | neutral | 0.0747 | 0.2471 | 0.3022 | 0.2109 | 0.1830 |
| trend_lowvol | neutral | 0.0705 | 0.2717 | 0.2595 | 0.2338 | 0.1601 |
| fund_profit_growth | neutral | 0.0524 | 0.2084 | 0.2516 | 0.1910 | 0.1498 |
| fund_pb | neutral | 0.0709 | 0.2804 | 0.2530 | 0.1441 | 0.1447 |
| volatility | neutral | 0.0634 | 0.2726 | 0.2326 | 0.2098 | 0.1407 |
| fund_score | neutral | 0.0516 | 0.2233 | 0.2312 | 0.1889 | 0.1375 |
| fund_revenue_growth | neutral | 0.0494 | 0.2274 | 0.2172 | 0.1733 | 0.1274 |
| low_downside | neutral | 0.0388 | 0.2538 | 0.1528 | 0.1399 | 0.0871 |
| volatility | bull | 0.0979 | 0.2206 | 0.4436 | 0.3258 | 0.2941 |
| fund_pe | bull | 0.0711 | 0.2399 | 0.2965 | 0.2500 | 0.1853 |
| wash_sale_score | bull | 0.0464 | 0.1806 | 0.2571 | 0.2019 | 0.1545 |
| mom_x_lowvol_20_20 | bull | 0.0561 | 0.2313 | 0.2424 | 0.1439 | 0.1386 |
| fund_pb | bull | 0.0650 | 0.2729 | 0.2381 | 0.1629 | 0.1385 |
| trend_lowvol | bull | 0.0651 | 0.2716 | 0.2398 | 0.1212 | 0.1344 |
| turnover_stability | bull | 0.0501 | 0.2238 | 0.2239 | 0.1818 | 0.1323 |
| rsi_vol_combo | bull | 0.0331 | 0.2594 | 0.1276 | 0.1212 | 0.0715 |
| mom_x_lowvol_20_20 | bear | 0.0757 | 0.2283 | 0.3316 | 0.2877 | 0.2135 |
| trend_lowvol | bear | 0.0749 | 0.2835 | 0.2641 | 0.3425 | 0.1773 |
| fund_roe | bear | 0.0439 | 0.1999 | 0.2198 | 0.1781 | 0.1295 |

### 机器人概念

- **Neutral**: ['momentum_reversal', 'mom_x_lowvol_20_20', 'trend_lowvol'] (单因子IC=0.0815, 组合IC=0.0897)
  - weights: [0.3481, 0.3361, 0.3158]
- **Bull**: ['low_downside', 'volatility', 'fund_pb'] (单因子IC=0.0953, 组合IC=0.1153)
  - bull_weights: [0.3953, 0.3215, 0.2833]
- **Bear**: ['momentum_reversal'] (单因子IC=0.1087, 组合IC=0.1087)
  - bear_weights: [1.0]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| momentum_reversal | neutral | 0.0803 | 0.1482 | 0.5417 | 0.4113 | 0.3823 |
| mom_x_lowvol_20_20 | neutral | 0.0795 | 0.1460 | 0.5448 | 0.3549 | 0.3691 |
| trend_lowvol | neutral | 0.0846 | 0.1660 | 0.5095 | 0.3612 | 0.3468 |
| fund_pb | neutral | 0.0803 | 0.1622 | 0.4951 | 0.3278 | 0.3287 |
| rsi_vol_combo | neutral | 0.0619 | 0.1374 | 0.4509 | 0.3445 | 0.3031 |
| fund_profit_growth | neutral | 0.0473 | 0.1119 | 0.4226 | 0.2881 | 0.2722 |
| fund_pe | neutral | 0.0591 | 0.1457 | 0.4055 | 0.3027 | 0.2641 |
| volatility | neutral | 0.0683 | 0.1807 | 0.3779 | 0.2944 | 0.2446 |
| turnover_stability | neutral | 0.0252 | 0.0796 | 0.3160 | 0.2735 | 0.2012 |
| fund_score | neutral | 0.0397 | 0.1423 | 0.2793 | 0.1399 | 0.1592 |
| low_downside | neutral | 0.0401 | 0.1661 | 0.2414 | 0.2129 | 0.1464 |
| fund_revenue_growth | neutral | 0.0251 | 0.1063 | 0.2359 | 0.1775 | 0.1389 |
| low_downside | bull | 0.0936 | 0.1298 | 0.7211 | 0.4621 | 0.5272 |
| volatility | bull | 0.1003 | 0.1630 | 0.6152 | 0.3939 | 0.4288 |
| fund_pb | bull | 0.0919 | 0.1686 | 0.5451 | 0.3864 | 0.3779 |
| turnover_stability | bull | 0.0396 | 0.0766 | 0.5170 | 0.3712 | 0.3545 |
| trend_lowvol | bull | 0.0735 | 0.1611 | 0.4563 | 0.3182 | 0.3007 |
| fund_pe | bull | 0.0478 | 0.1376 | 0.3471 | 0.3030 | 0.2261 |
| momentum_reversal | bull | 0.0421 | 0.1468 | 0.2869 | 0.2803 | 0.1837 |
| mom_x_lowvol_20_20 | bull | 0.0436 | 0.1494 | 0.2921 | 0.2273 | 0.1793 |
| fund_revenue_growth | bull | 0.0256 | 0.1123 | 0.2280 | 0.1136 | 0.1269 |
| rsi_vol_combo | bull | 0.0203 | 0.1261 | 0.1609 | 0.1136 | 0.0896 |
| stroke_phase | bull | 0.0113 | 0.0828 | 0.1361 | 0.1894 | 0.0810 |
| momentum_reversal | bear | 0.1087 | 0.1686 | 0.6445 | 0.5068 | 0.4856 |
| rsi_vol_combo | bear | 0.0692 | 0.1194 | 0.5795 | 0.4795 | 0.4287 |
| mom_x_lowvol_20_20 | bear | 0.1078 | 0.1818 | 0.5926 | 0.3973 | 0.4140 |
| trend_lowvol | bear | 0.0776 | 0.2140 | 0.3626 | 0.1233 | 0.2037 |
| fund_profit_growth | bear | 0.0419 | 0.1430 | 0.2931 | 0.2055 | 0.1766 |
| fund_pb | bear | 0.0396 | 0.2136 | 0.1855 | 0.1507 | 0.1067 |
| bb_width_20 | bear | 0.0352 | 0.1925 | 0.1828 | 0.1233 | 0.1027 |

### 机器视觉

- **Neutral**: ['mom_x_lowvol_20_20', 'momentum_reversal', 'fund_pb'] (单因子IC=0.0857, 组合IC=0.1163)
  - weights: [0.3443, 0.3418, 0.3139]
- **Bull**: ['trend_lowvol', 'fund_pb', 'volatility'] (单因子IC=0.0966, 组合IC=0.1267)
  - bull_weights: [0.3388, 0.3365, 0.3247]
- **Bear**: ['momentum_reversal', 'rsi_vol_combo', 'mom_x_lowvol_20_20'] (单因子IC=0.108, 组合IC=0.1269)
  - bear_weights: [0.3653, 0.3478, 0.287]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| mom_x_lowvol_20_20 | neutral | 0.0893 | 0.1904 | 0.4687 | 0.3236 | 0.3102 |
| momentum_reversal | neutral | 0.0856 | 0.1855 | 0.4617 | 0.3340 | 0.3080 |
| fund_pb | neutral | 0.0821 | 0.1919 | 0.4280 | 0.3215 | 0.2828 |
| rsi_vol_combo | neutral | 0.0742 | 0.1802 | 0.4117 | 0.3111 | 0.2699 |
| trend_lowvol | neutral | 0.0782 | 0.2118 | 0.3691 | 0.2651 | 0.2335 |
| volatility | neutral | 0.0716 | 0.2236 | 0.3203 | 0.2422 | 0.1990 |
| fund_profit_growth | neutral | 0.0519 | 0.1643 | 0.3161 | 0.2213 | 0.1930 |
| fund_pe | neutral | 0.0422 | 0.1646 | 0.2565 | 0.2255 | 0.1571 |
| fund_score | neutral | 0.0492 | 0.1839 | 0.2677 | 0.1587 | 0.1551 |
| fund_revenue_growth | neutral | 0.0375 | 0.1625 | 0.2305 | 0.1712 | 0.1350 |
| low_downside | neutral | 0.0310 | 0.1944 | 0.1592 | 0.1367 | 0.0905 |
| trend_lowvol | bull | 0.0921 | 0.1847 | 0.4987 | 0.3485 | 0.3362 |
| fund_pb | bull | 0.0960 | 0.1993 | 0.4817 | 0.3864 | 0.3339 |
| volatility | bull | 0.1016 | 0.2113 | 0.4806 | 0.3409 | 0.3222 |
| fund_revenue_growth | bull | 0.0499 | 0.1453 | 0.3434 | 0.2045 | 0.2068 |
| fund_pe | bull | 0.0513 | 0.1691 | 0.3034 | 0.2197 | 0.1850 |
| low_downside | bull | 0.0553 | 0.1736 | 0.3184 | 0.1515 | 0.1833 |
| turnover_stability | bull | 0.0324 | 0.1309 | 0.2479 | 0.2197 | 0.1512 |
| fund_gross_margin | bull | 0.0391 | 0.1499 | 0.2605 | 0.1591 | 0.1510 |
| mom_x_lowvol_20_20 | bull | 0.0493 | 0.1981 | 0.2487 | 0.1818 | 0.1470 |
| fund_profit_growth | bull | 0.0413 | 0.1748 | 0.2364 | 0.1894 | 0.1406 |
| momentum_reversal | bull | 0.0415 | 0.1883 | 0.2207 | 0.2045 | 0.1329 |
| fund_score | bull | 0.0388 | 0.1921 | 0.2020 | 0.1288 | 0.1140 |
| momentum_reversal | bear | 0.1230 | 0.2059 | 0.5974 | 0.4247 | 0.4256 |
| rsi_vol_combo | bear | 0.0942 | 0.1656 | 0.5688 | 0.4247 | 0.4052 |
| mom_x_lowvol_20_20 | bear | 0.1067 | 0.2185 | 0.4882 | 0.3699 | 0.3344 |
| fund_revenue_growth | bear | 0.0733 | 0.1735 | 0.4225 | 0.2877 | 0.2720 |
| trend_lowvol | bear | 0.0830 | 0.2726 | 0.3046 | 0.3151 | 0.2003 |

### 权重股

- **Neutral**: ['fund_pe'] (单因子IC=0.0809, 组合IC=0.0806)
  - weights: [1.0]
- **Bull**: ['relative_strength', 'ema20_slope'] (单因子IC=0.1073, 组合IC=0.1166)
  - bull_weights: [0.5098, 0.4902]
- **Bear**: ['wash_sale_score', 'fund_profit_growth', 'mom_x_lowvol_20_20'] (单因子IC=0.1033, 组合IC=0.1458)
  - bear_weights: [0.3502, 0.3433, 0.3065]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| fund_pe | neutral | 0.0809 | 0.3280 | 0.2465 | 0.1962 | 0.1475 |
| fund_pb | neutral | 0.0702 | 0.3336 | 0.2106 | 0.1858 | 0.1248 |
| trend_lowvol | neutral | 0.0570 | 0.2781 | 0.2050 | 0.1712 | 0.1201 |
| top_fractal_volume | neutral | 0.0278 | 0.1630 | 0.1703 | 0.1070 | 0.0943 |
| turnover_stability | neutral | 0.0216 | 0.2110 | 0.1023 | 0.1106 | 0.0568 |
| relative_strength | neutral | 0.0161 | 0.2751 | 0.0586 | 0.1002 | 0.0322 |
| relative_strength | bull | 0.1076 | 0.2903 | 0.3708 | 0.3258 | 0.2458 |
| ema20_slope | bull | 0.1071 | 0.2986 | 0.3586 | 0.3182 | 0.2363 |
| ma_alignment | bull | 0.1006 | 0.2965 | 0.3393 | 0.3485 | 0.2288 |
| exhaustion_risk | bull | 0.0593 | 0.1946 | 0.3049 | 0.2846 | 0.1959 |
| bb_width_20 | bull | 0.0596 | 0.3052 | 0.1952 | 0.1667 | 0.1138 |
| limit_pullback_score | bull | 0.0433 | 0.2518 | 0.1719 | 0.1873 | 0.1021 |
| vol_confirm | bull | 0.0415 | 0.2644 | 0.1570 | 0.1288 | 0.0886 |
| smart_money_flow | bull | 0.0324 | 0.2703 | 0.1198 | 0.1061 | 0.0662 |
| wash_sale_score | bear | 0.1033 | 0.1850 | 0.5585 | 0.3939 | 0.3893 |
| fund_profit_growth | bear | 0.1030 | 0.2071 | 0.4973 | 0.5342 | 0.3815 |
| mom_x_lowvol_20_20 | bear | 0.1035 | 0.2206 | 0.4692 | 0.4521 | 0.3406 |
| limit_pullback_score | bear | 0.0608 | 0.1835 | 0.3315 | 0.2500 | 0.2072 |
| fund_revenue_growth | bear | 0.0639 | 0.2008 | 0.3184 | 0.2877 | 0.2050 |
| momentum_reversal | bear | 0.0804 | 0.2860 | 0.2810 | 0.2329 | 0.1732 |
| bb_width_20 | bear | 0.0598 | 0.2623 | 0.2278 | 0.2329 | 0.1404 |

### 柔性屏(折叠屏)

- **Neutral**: ['momentum_reversal', 'trend_lowvol', 'fund_pb'] (单因子IC=0.0822, 组合IC=0.1163)
  - weights: [0.3405, 0.3378, 0.3217]
- **Bull**: ['volatility', 'fund_pb', 'low_downside'] (单因子IC=0.0735, 组合IC=0.0963)
  - bull_weights: [0.3554, 0.3522, 0.2924]
- **Bear**: ['fund_pe', 'fund_gross_margin', 'fund_profit_growth'] (单因子IC=0.076, 组合IC=0.0809)
  - bear_weights: [0.3406, 0.3307, 0.3287]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| momentum_reversal | neutral | 0.0848 | 0.1819 | 0.4663 | 0.3883 | 0.3237 |
| trend_lowvol | neutral | 0.0995 | 0.2064 | 0.4822 | 0.3319 | 0.3211 |
| fund_pb | neutral | 0.0623 | 0.1395 | 0.4466 | 0.3695 | 0.3058 |
| mom_x_lowvol_20_20 | neutral | 0.0808 | 0.1810 | 0.4465 | 0.3257 | 0.2959 |
| rsi_vol_combo | neutral | 0.0668 | 0.1691 | 0.3952 | 0.2714 | 0.2512 |
| volatility | neutral | 0.0638 | 0.1961 | 0.3253 | 0.2317 | 0.2003 |
| fund_pe | neutral | 0.0494 | 0.1521 | 0.3245 | 0.2129 | 0.1968 |
| fund_score | neutral | 0.0442 | 0.1759 | 0.2512 | 0.1545 | 0.1450 |
| fund_profit_growth | neutral | 0.0361 | 0.1465 | 0.2467 | 0.1712 | 0.1445 |
| turnover_stability | neutral | 0.0297 | 0.1427 | 0.2081 | 0.1649 | 0.1212 |
| low_downside | neutral | 0.0362 | 0.1861 | 0.1943 | 0.1566 | 0.1124 |
| fund_revenue_growth | neutral | 0.0251 | 0.1462 | 0.1715 | 0.1795 | 0.1011 |
| fund_gross_margin | neutral | 0.0234 | 0.1487 | 0.1573 | 0.1315 | 0.0890 |
| volatility | bull | 0.0821 | 0.2013 | 0.4079 | 0.3712 | 0.2797 |
| fund_pb | bull | 0.0731 | 0.1780 | 0.4110 | 0.3485 | 0.2771 |
| low_downside | bull | 0.0653 | 0.1806 | 0.3615 | 0.2727 | 0.2301 |
| trend_lowvol | bull | 0.0797 | 0.2289 | 0.3481 | 0.2424 | 0.2162 |
| fund_pe | bull | 0.0483 | 0.1529 | 0.3159 | 0.2045 | 0.1902 |
| fund_revenue_growth | bull | 0.0468 | 0.1669 | 0.2806 | 0.2045 | 0.1690 |
| fund_score | bull | 0.0511 | 0.1925 | 0.2654 | 0.2652 | 0.1679 |
| mom_x_lowvol_20_20 | bull | 0.0494 | 0.1908 | 0.2586 | 0.1894 | 0.1538 |
| fund_roe | bull | 0.0399 | 0.1820 | 0.2192 | 0.2955 | 0.1420 |
| momentum_reversal | bull | 0.0450 | 0.1952 | 0.2304 | 0.1591 | 0.1335 |
| rsi_vol_combo | bull | 0.0369 | 0.1791 | 0.2060 | 0.1439 | 0.1178 |
| fund_pe | bear | 0.0644 | 0.1229 | 0.5243 | 0.5068 | 0.3950 |
| fund_gross_margin | bear | 0.0793 | 0.1444 | 0.5489 | 0.3973 | 0.3835 |
| fund_profit_growth | bear | 0.0844 | 0.1577 | 0.5350 | 0.4247 | 0.3811 |
| fund_roe | bear | 0.0524 | 0.1588 | 0.3296 | 0.3151 | 0.2167 |
| momentum_reversal | bear | 0.0631 | 0.2422 | 0.2606 | 0.1507 | 0.1499 |
| fund_revenue_growth | bear | 0.0304 | 0.1416 | 0.2146 | 0.2877 | 0.1381 |
| turnover_stability | bear | 0.0252 | 0.1246 | 0.2022 | 0.1781 | 0.1191 |

### 核污染防治

- **Neutral**: ['mom_x_lowvol_20_20', 'momentum_reversal', 'trend_lowvol'] (单因子IC=0.1051, 组合IC=0.1159)
  - weights: [0.3533, 0.3455, 0.3012]
- **Bull**: ['low_downside', 'fund_pe'] (单因子IC=0.1187, 组合IC=0.1268)
  - bull_weights: [0.507, 0.493]
- **Bear**: ['rsi_vol_combo'] (单因子IC=0.1525, 组合IC=0.1525)
  - bear_weights: [1.0]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| mom_x_lowvol_20_20 | neutral | 0.1089 | 0.2581 | 0.4220 | 0.3048 | 0.2753 |
| momentum_reversal | neutral | 0.1083 | 0.2572 | 0.4211 | 0.2787 | 0.2692 |
| trend_lowvol | neutral | 0.0981 | 0.2645 | 0.3710 | 0.2651 | 0.2347 |
| fund_pb | neutral | 0.0734 | 0.2356 | 0.3115 | 0.2234 | 0.1906 |
| volatility | neutral | 0.0791 | 0.2642 | 0.2995 | 0.2474 | 0.1868 |
| rsi_vol_combo | neutral | 0.0705 | 0.2522 | 0.2795 | 0.1921 | 0.1666 |
| turnover_stability | neutral | 0.0466 | 0.2218 | 0.2100 | 0.1994 | 0.1260 |
| fund_profit_growth | neutral | 0.0380 | 0.2248 | 0.1692 | 0.1545 | 0.0977 |
| fund_score | neutral | 0.0352 | 0.2358 | 0.1492 | 0.1211 | 0.0836 |
| fund_pe | neutral | 0.0250 | 0.2194 | 0.1141 | 0.1211 | 0.0639 |
| low_downside | neutral | 0.0267 | 0.2335 | 0.1144 | 0.1013 | 0.0630 |
| low_downside | bull | 0.1089 | 0.2152 | 0.5059 | 0.4697 | 0.3717 |
| fund_pe | bull | 0.1285 | 0.2423 | 0.5302 | 0.3636 | 0.3615 |
| volatility | bull | 0.1031 | 0.2467 | 0.4177 | 0.3333 | 0.2785 |
| fund_roe | bull | 0.0904 | 0.2348 | 0.3849 | 0.2879 | 0.2478 |
| fund_pb | bull | 0.0806 | 0.2240 | 0.3598 | 0.2424 | 0.2235 |
| fund_score | bull | 0.0745 | 0.2486 | 0.2996 | 0.1894 | 0.1782 |
| trend_lowvol | bull | 0.0618 | 0.2728 | 0.2265 | 0.1591 | 0.1312 |
| fund_revenue_growth | bull | 0.0486 | 0.2154 | 0.2256 | 0.1061 | 0.1248 |
| mom_x_lowvol_20_20 | bull | 0.0462 | 0.2623 | 0.1760 | 0.2348 | 0.1086 |
| fund_profit_growth | bull | 0.0436 | 0.2409 | 0.1808 | 0.1856 | 0.1072 |
| momentum_reversal | bull | 0.0335 | 0.2538 | 0.1319 | 0.2121 | 0.0799 |
| turnover_stability | bull | 0.0265 | 0.2196 | 0.1205 | 0.1515 | 0.0694 |
| rsi_vol_combo | bear | 0.1525 | 0.2483 | 0.6139 | 0.4521 | 0.4457 |
| fund_score | bear | 0.1307 | 0.2286 | 0.5720 | 0.5068 | 0.4310 |
| fund_revenue_growth | bear | 0.1152 | 0.1863 | 0.6185 | 0.3425 | 0.4151 |
| momentum_reversal | bear | 0.1139 | 0.2215 | 0.5143 | 0.4658 | 0.3769 |
| trend_lowvol | bear | 0.0883 | 0.2189 | 0.4034 | 0.3425 | 0.2708 |
| fund_roe | bear | 0.0906 | 0.2426 | 0.3734 | 0.3425 | 0.2507 |
| fund_profit_growth | bear | 0.0767 | 0.2109 | 0.3638 | 0.3151 | 0.2392 |
| mom_x_lowvol_20_20 | bear | 0.0773 | 0.2354 | 0.3284 | 0.2877 | 0.2114 |
| turnover_stability | bear | 0.0718 | 0.2533 | 0.2835 | 0.2466 | 0.1767 |
| volatility | bear | 0.0621 | 0.2354 | 0.2638 | 0.2055 | 0.1590 |
| fund_gross_margin | bear | 0.0572 | 0.2322 | 0.2461 | 0.1507 | 0.1416 |

### 核能核电

- **Neutral**: ['mom_x_lowvol_20_20', 'momentum_reversal', 'trend_lowvol'] (单因子IC=0.0811, 组合IC=0.0885)
  - weights: [0.3665, 0.3314, 0.302]
- **Bull**: ['low_downside', 'fund_pb'] (单因子IC=0.072, 组合IC=0.0832)
  - bull_weights: [0.5784, 0.4216]
- **Bear**: ['mom_x_lowvol_20_20', 'momentum_reversal'] (单因子IC=0.1097, 组合IC=0.1141)
  - bear_weights: [0.5052, 0.4948]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| mom_x_lowvol_20_20 | neutral | 0.0810 | 0.1671 | 0.4848 | 0.3507 | 0.3274 |
| momentum_reversal | neutral | 0.0796 | 0.1752 | 0.4545 | 0.3027 | 0.2960 |
| trend_lowvol | neutral | 0.0825 | 0.1967 | 0.4196 | 0.2860 | 0.2698 |
| fund_profit_growth | neutral | 0.0483 | 0.1224 | 0.3949 | 0.2944 | 0.2555 |
| rsi_vol_combo | neutral | 0.0560 | 0.1609 | 0.3480 | 0.2777 | 0.2223 |
| fund_score | neutral | 0.0458 | 0.1414 | 0.3239 | 0.2589 | 0.2039 |
| volatility | neutral | 0.0624 | 0.2030 | 0.3074 | 0.2714 | 0.1954 |
| fund_pe | neutral | 0.0530 | 0.1830 | 0.2897 | 0.2234 | 0.1772 |
| fund_pb | neutral | 0.0552 | 0.1872 | 0.2947 | 0.1816 | 0.1741 |
| turnover_stability | neutral | 0.0323 | 0.1204 | 0.2681 | 0.1858 | 0.1590 |
| fund_revenue_growth | neutral | 0.0293 | 0.1152 | 0.2542 | 0.1670 | 0.1483 |
| fund_roe | neutral | 0.0281 | 0.1529 | 0.1840 | 0.1441 | 0.1053 |
| low_downside | neutral | 0.0317 | 0.1999 | 0.1587 | 0.1503 | 0.0913 |
| low_downside | bull | 0.0771 | 0.1578 | 0.4885 | 0.3485 | 0.3294 |
| fund_pb | bull | 0.0668 | 0.1782 | 0.3750 | 0.2803 | 0.2401 |
| volatility | bull | 0.0655 | 0.1831 | 0.3576 | 0.2576 | 0.2249 |
| fund_pe | bull | 0.0541 | 0.1711 | 0.3164 | 0.2197 | 0.1929 |
| trend_lowvol | bull | 0.0422 | 0.1812 | 0.2330 | 0.1894 | 0.1386 |
| turnover_stability | bull | 0.0136 | 0.1043 | 0.1304 | 0.1742 | 0.0766 |
| exhaustion_risk | bull | 0.0123 | 0.0946 | 0.1305 | 0.1136 | 0.0727 |
| fund_gross_margin | bull | 0.0143 | 0.1259 | 0.1132 | 0.1061 | 0.0626 |
| relative_strength | bull | 0.0159 | 0.1530 | 0.1037 | 0.1212 | 0.0582 |
| mom_x_lowvol_20_20 | bear | 0.1071 | 0.2182 | 0.4910 | 0.3699 | 0.3363 |
| momentum_reversal | bear | 0.1123 | 0.2242 | 0.5009 | 0.3151 | 0.3294 |
| rsi_vol_combo | bear | 0.0791 | 0.1636 | 0.4834 | 0.3151 | 0.3179 |
| fund_profit_growth | bear | 0.0489 | 0.1140 | 0.4294 | 0.3425 | 0.2882 |
| fund_revenue_growth | bear | 0.0472 | 0.1133 | 0.4162 | 0.2603 | 0.2622 |
| fund_score | bear | 0.0524 | 0.1363 | 0.3841 | 0.2055 | 0.2315 |
| trend_lowvol | bear | 0.0884 | 0.2507 | 0.3527 | 0.1233 | 0.1981 |
| fund_pe | bear | 0.0569 | 0.1943 | 0.2928 | 0.1233 | 0.1644 |
| bb_width_20 | bear | 0.0609 | 0.2436 | 0.2502 | 0.2055 | 0.1508 |

### 毫米波概念

- **Neutral**: ['trend_lowvol', 'mom_x_lowvol_20_20'] (单因子IC=0.0887, 组合IC=0.1089)
  - weights: [0.5365, 0.4635]
- **Bull**: ['fund_pb', 'volatility'] (单因子IC=0.074, 组合IC=0.0853)
  - bull_weights: [0.5526, 0.4474]
- **Bear**: ['momentum_reversal', 'mom_x_lowvol_20_20', 'rsi_vol_combo'] (单因子IC=0.1114, 组合IC=0.1265)
  - bear_weights: [0.3483, 0.3262, 0.3255]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| trend_lowvol | neutral | 0.0941 | 0.2339 | 0.4024 | 0.3152 | 0.2646 |
| mom_x_lowvol_20_20 | neutral | 0.0833 | 0.2313 | 0.3603 | 0.2693 | 0.2287 |
| momentum_reversal | neutral | 0.0769 | 0.2303 | 0.3339 | 0.2797 | 0.2136 |
| volatility | neutral | 0.0808 | 0.2430 | 0.3324 | 0.2766 | 0.2122 |
| fund_pb | neutral | 0.0689 | 0.2118 | 0.3253 | 0.2651 | 0.2058 |
| fund_profit_growth | neutral | 0.0438 | 0.1821 | 0.2406 | 0.2171 | 0.1464 |
| low_downside | neutral | 0.0439 | 0.2180 | 0.2013 | 0.1420 | 0.1150 |
| rsi_vol_combo | neutral | 0.0439 | 0.2218 | 0.1977 | 0.1336 | 0.1121 |
| fund_score | neutral | 0.0375 | 0.2025 | 0.1852 | 0.1357 | 0.1052 |
| fund_pe | neutral | 0.0371 | 0.2074 | 0.1789 | 0.1545 | 0.1033 |
| turnover_stability | neutral | 0.0305 | 0.1710 | 0.1784 | 0.1378 | 0.1015 |
| fund_pb | bull | 0.0732 | 0.1837 | 0.3984 | 0.3182 | 0.2626 |
| volatility | bull | 0.0749 | 0.2268 | 0.3301 | 0.2879 | 0.2126 |
| fund_pe | bull | 0.0616 | 0.2027 | 0.3038 | 0.2538 | 0.1905 |
| fund_revenue_growth | bull | 0.0497 | 0.2064 | 0.2407 | 0.2652 | 0.1523 |
| low_downside | bull | 0.0489 | 0.2100 | 0.2329 | 0.2348 | 0.1438 |
| turnover_stability | bull | 0.0332 | 0.1724 | 0.1927 | 0.1591 | 0.1117 |
| trend_lowvol | bull | 0.0474 | 0.2616 | 0.1813 | 0.2121 | 0.1099 |
| fund_score | bull | 0.0317 | 0.2230 | 0.1422 | 0.1364 | 0.0808 |
| fund_profit_growth | bull | 0.0203 | 0.1807 | 0.1121 | 0.1288 | 0.0633 |
| momentum_reversal | bear | 0.1253 | 0.2467 | 0.5078 | 0.3699 | 0.3478 |
| mom_x_lowvol_20_20 | bear | 0.1156 | 0.2333 | 0.4955 | 0.3151 | 0.3258 |
| rsi_vol_combo | bear | 0.0933 | 0.2004 | 0.4653 | 0.3973 | 0.3251 |
| trend_lowvol | bear | 0.1013 | 0.2514 | 0.4028 | 0.2603 | 0.2538 |
| fund_gross_margin | bear | 0.0439 | 0.1518 | 0.2890 | 0.2877 | 0.1861 |
| fund_pe | bear | 0.0584 | 0.2061 | 0.2832 | 0.1781 | 0.1668 |
| turnover_stability | bear | 0.0424 | 0.1818 | 0.2334 | 0.2055 | 0.1407 |
| fund_pb | bear | 0.0458 | 0.2240 | 0.2045 | 0.1781 | 0.1204 |

### 民爆概念

- **Neutral**: ['mom_x_lowvol_20_20', 'momentum_reversal', 'turnover_stability'] (单因子IC=0.0604, 组合IC=0.0747)
  - weights: [0.3881, 0.3325, 0.2795]
- **Bull**: ['low_downside', 'volatility', 'fund_pb'] (单因子IC=0.1203, 组合IC=0.1503)
  - bull_weights: [0.3359, 0.3335, 0.3306]
- **Bear**: ['vol_confirm', 'fund_roe', 'fund_score'] (单因子IC=0.1054, 组合IC=0.1768)
  - bear_weights: [0.393, 0.3608, 0.2461]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| mom_x_lowvol_20_20 | neutral | 0.0713 | 0.2780 | 0.2563 | 0.2109 | 0.1552 |
| momentum_reversal | neutral | 0.0635 | 0.2817 | 0.2253 | 0.1806 | 0.1330 |
| turnover_stability | neutral | 0.0464 | 0.2375 | 0.1954 | 0.1441 | 0.1118 |
| trend_lowvol | neutral | 0.0555 | 0.2900 | 0.1912 | 0.1399 | 0.1090 |
| fund_revenue_growth | neutral | 0.0355 | 0.2478 | 0.1432 | 0.1503 | 0.0823 |
| rsi_vol_combo | neutral | 0.0404 | 0.2780 | 0.1453 | 0.1211 | 0.0814 |
| fund_score | neutral | 0.0312 | 0.2730 | 0.1145 | 0.1211 | 0.0642 |
| wash_sale_score | neutral | 0.0221 | 0.2351 | 0.0941 | 0.1053 | 0.0520 |
| low_downside | bull | 0.1076 | 0.2439 | 0.4412 | 0.3523 | 0.2983 |
| volatility | bull | 0.1298 | 0.2740 | 0.4739 | 0.2500 | 0.2962 |
| fund_pb | bull | 0.1234 | 0.2674 | 0.4613 | 0.2727 | 0.2936 |
| trend_lowvol | bull | 0.1119 | 0.2902 | 0.3857 | 0.3182 | 0.2542 |
| fund_pe | bull | 0.0682 | 0.2509 | 0.2719 | 0.3182 | 0.1792 |
| momentum_reversal | bull | 0.0446 | 0.2903 | 0.1536 | 0.1439 | 0.0878 |
| mom_x_lowvol_20_20 | bull | 0.0442 | 0.2897 | 0.1526 | 0.1288 | 0.0861 |
| vol_confirm | bear | 0.1474 | 0.3472 | 0.4245 | 0.3425 | 0.2850 |
| fund_roe | bear | 0.0959 | 0.2510 | 0.3820 | 0.3699 | 0.2616 |
| fund_score | bear | 0.0729 | 0.2518 | 0.2895 | 0.2329 | 0.1784 |
| trend_lowvol | bear | 0.0763 | 0.2679 | 0.2847 | 0.1370 | 0.1618 |
| fund_gross_margin | bear | 0.0731 | 0.3037 | 0.2407 | 0.2603 | 0.1517 |
| fund_profit_growth | bear | 0.0673 | 0.2539 | 0.2650 | 0.1233 | 0.1488 |
| fund_revenue_growth | bear | 0.0391 | 0.1934 | 0.2022 | 0.1507 | 0.1164 |
| turnover_stability | bear | 0.0421 | 0.2224 | 0.1892 | 0.1233 | 0.1063 |

### 氟化工概念

- **Neutral**: ['fund_pe', 'fund_pb', 'turnover_stability'] (单因子IC=0.0559, 组合IC=0.0681)
  - weights: [0.3854, 0.3536, 0.2611]
- **Bull**: ['low_downside'] (单因子IC=0.0772, 组合IC=0.0772)
  - bull_weights: [1.0]
- **Bear**: ['fund_profit_growth', 'mom_x_lowvol_20_20'] (单因子IC=0.092, 组合IC=0.1024)
  - bear_weights: [0.5762, 0.4238]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| fund_pe | neutral | 0.0632 | 0.2600 | 0.2432 | 0.1816 | 0.1437 |
| fund_pb | neutral | 0.0668 | 0.2916 | 0.2292 | 0.1503 | 0.1319 |
| turnover_stability | neutral | 0.0377 | 0.2306 | 0.1635 | 0.1910 | 0.0974 |
| volatility | neutral | 0.0399 | 0.2444 | 0.1633 | 0.1712 | 0.0956 |
| rsi_vol_combo | neutral | 0.0340 | 0.2268 | 0.1497 | 0.1013 | 0.0824 |
| fund_revenue_growth | neutral | 0.0173 | 0.2073 | 0.0835 | 0.1127 | 0.0464 |
| low_downside | bull | 0.0772 | 0.2715 | 0.2844 | 0.2727 | 0.1810 |
| volatility | bull | 0.0552 | 0.2294 | 0.2406 | 0.1212 | 0.1349 |
| trend_lowvol | bull | 0.0386 | 0.2470 | 0.1563 | 0.1212 | 0.0876 |
| fund_pb | bull | 0.0456 | 0.3018 | 0.1512 | 0.1212 | 0.0848 |
| relative_strength | bull | 0.0268 | 0.2608 | 0.1026 | 0.1136 | 0.0571 |
| fund_profit_growth | bear | 0.0993 | 0.2046 | 0.4856 | 0.3151 | 0.3193 |
| mom_x_lowvol_20_20 | bear | 0.0847 | 0.2521 | 0.3361 | 0.3973 | 0.2348 |
| momentum_reversal | bear | 0.0824 | 0.2458 | 0.3354 | 0.3151 | 0.2205 |
| rsi_vol_combo | bear | 0.0735 | 0.2108 | 0.3489 | 0.2055 | 0.2103 |
| turnover_stability | bear | 0.0526 | 0.2136 | 0.2463 | 0.1781 | 0.1451 |

### 氢能源

- **Neutral**: ['momentum_reversal', 'mom_x_lowvol_20_20', 'fund_profit_growth'] (单因子IC=0.0693, 组合IC=0.0884)
  - weights: [0.3565, 0.3413, 0.3022]
- **Bull**: ['low_downside', 'fund_pb'] (单因子IC=0.0789, 组合IC=0.0904)
  - bull_weights: [0.5915, 0.4085]
- **Bear**: ['momentum_reversal'] (单因子IC=0.1332, 组合IC=0.1332)
  - bear_weights: [1.0]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| momentum_reversal | neutral | 0.0802 | 0.1651 | 0.4856 | 0.3904 | 0.3376 |
| mom_x_lowvol_20_20 | neutral | 0.0748 | 0.1578 | 0.4741 | 0.3633 | 0.3232 |
| fund_profit_growth | neutral | 0.0530 | 0.1242 | 0.4270 | 0.3403 | 0.2862 |
| trend_lowvol | neutral | 0.0833 | 0.1930 | 0.4318 | 0.3215 | 0.2853 |
| fund_pb | neutral | 0.0691 | 0.1706 | 0.4051 | 0.2985 | 0.2630 |
| turnover_stability | neutral | 0.0434 | 0.1109 | 0.3915 | 0.2985 | 0.2542 |
| volatility | neutral | 0.0750 | 0.2005 | 0.3742 | 0.2923 | 0.2418 |
| rsi_vol_combo | neutral | 0.0580 | 0.1531 | 0.3785 | 0.2589 | 0.2382 |
| fund_pe | neutral | 0.0587 | 0.1616 | 0.3631 | 0.2797 | 0.2324 |
| fund_score | neutral | 0.0464 | 0.1573 | 0.2947 | 0.2192 | 0.1797 |
| low_downside | neutral | 0.0486 | 0.1967 | 0.2472 | 0.1921 | 0.1473 |
| fund_revenue_growth | neutral | 0.0211 | 0.1040 | 0.2026 | 0.1378 | 0.1152 |
| fund_gross_margin | neutral | 0.0181 | 0.1131 | 0.1605 | 0.1420 | 0.0916 |
| low_downside | bull | 0.0839 | 0.1388 | 0.6048 | 0.4924 | 0.4513 |
| fund_pb | bull | 0.0738 | 0.1631 | 0.4521 | 0.3788 | 0.3117 |
| volatility | bull | 0.0741 | 0.1629 | 0.4549 | 0.3636 | 0.3102 |
| fund_pe | bull | 0.0494 | 0.1460 | 0.3383 | 0.2652 | 0.2140 |
| turnover_stability | bull | 0.0297 | 0.0952 | 0.3121 | 0.2273 | 0.1915 |
| trend_lowvol | bull | 0.0402 | 0.1608 | 0.2498 | 0.1212 | 0.1400 |
| fund_gross_margin | bull | 0.0248 | 0.1078 | 0.2299 | 0.1667 | 0.1341 |
| momentum_reversal | bear | 0.1332 | 0.1799 | 0.7402 | 0.5616 | 0.5780 |
| rsi_vol_combo | bear | 0.0906 | 0.1312 | 0.6905 | 0.5616 | 0.5391 |
| mom_x_lowvol_20_20 | bear | 0.1236 | 0.1905 | 0.6486 | 0.5890 | 0.5153 |
| fund_profit_growth | bear | 0.0603 | 0.1248 | 0.4831 | 0.3973 | 0.3375 |
| trend_lowvol | bear | 0.0878 | 0.2122 | 0.4139 | 0.2329 | 0.2552 |
| turnover_stability | bear | 0.0292 | 0.0851 | 0.3430 | 0.3151 | 0.2255 |
| bb_width_20 | bear | 0.0598 | 0.1962 | 0.3048 | 0.2603 | 0.1921 |
| fund_score | bear | 0.0514 | 0.1571 | 0.3269 | 0.1507 | 0.1881 |
| fund_gross_margin | bear | 0.0236 | 0.1089 | 0.2172 | 0.2329 | 0.1339 |

### 氮化镓

- **Neutral**: ['momentum_reversal', 'mom_x_lowvol_20_20', 'rsi_vol_combo'] (单因子IC=0.0978, 组合IC=0.1063)
  - weights: [0.3537, 0.3268, 0.3195]
- **Bull**: ['volatility', 'fund_pe', 'fund_revenue_growth'] (单因子IC=0.094, 组合IC=0.1217)
  - bull_weights: [0.3554, 0.3294, 0.3152]
- **Bear**: ['mom_x_lowvol_20_20', 'momentum_reversal', 'trend_lowvol'] (单因子IC=0.1701, 组合IC=0.1876)
  - bear_weights: [0.3685, 0.3381, 0.2933]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| momentum_reversal | neutral | 0.1028 | 0.2273 | 0.4523 | 0.3486 | 0.3050 |
| mom_x_lowvol_20_20 | neutral | 0.0977 | 0.2298 | 0.4251 | 0.3257 | 0.2818 |
| rsi_vol_combo | neutral | 0.0929 | 0.2233 | 0.4162 | 0.3236 | 0.2754 |
| trend_lowvol | neutral | 0.0977 | 0.2388 | 0.4092 | 0.3069 | 0.2674 |
| volatility | neutral | 0.0760 | 0.2096 | 0.3627 | 0.3069 | 0.2370 |
| fund_pe | neutral | 0.0682 | 0.2080 | 0.3280 | 0.2651 | 0.2075 |
| fund_pb | neutral | 0.0708 | 0.2360 | 0.3000 | 0.2296 | 0.1844 |
| low_downside | neutral | 0.0563 | 0.2081 | 0.2705 | 0.2505 | 0.1692 |
| turnover_stability | neutral | 0.0428 | 0.1858 | 0.2304 | 0.1785 | 0.1357 |
| fund_profit_growth | neutral | 0.0317 | 0.2166 | 0.1463 | 0.1253 | 0.0823 |
| fund_score | neutral | 0.0295 | 0.2250 | 0.1312 | 0.1378 | 0.0746 |
| fund_revenue_growth | neutral | 0.0234 | 0.1948 | 0.1200 | 0.1044 | 0.0663 |
| volatility | bull | 0.1029 | 0.2326 | 0.4422 | 0.2955 | 0.2864 |
| fund_pe | bull | 0.0952 | 0.2172 | 0.4381 | 0.2121 | 0.2655 |
| fund_revenue_growth | bull | 0.0839 | 0.2200 | 0.3811 | 0.3333 | 0.2541 |
| fund_roe | bull | 0.0647 | 0.1907 | 0.3396 | 0.3258 | 0.2251 |
| low_downside | bull | 0.0633 | 0.2076 | 0.3048 | 0.2500 | 0.1905 |
| fund_pb | bull | 0.0739 | 0.2638 | 0.2803 | 0.2121 | 0.1699 |
| fund_score | bull | 0.0591 | 0.2202 | 0.2682 | 0.2652 | 0.1697 |
| trend_lowvol | bull | 0.0571 | 0.2602 | 0.2196 | 0.1970 | 0.1314 |
| fund_profit_growth | bull | 0.0389 | 0.1833 | 0.2124 | 0.1818 | 0.1255 |
| mom_x_lowvol_20_20 | bull | 0.0420 | 0.2359 | 0.1781 | 0.1439 | 0.1019 |
| wash_sale_score | bull | 0.0222 | 0.1731 | 0.1283 | 0.1244 | 0.0721 |
| mom_x_lowvol_20_20 | bear | 0.1772 | 0.2257 | 0.7852 | 0.5616 | 0.6131 |
| momentum_reversal | bear | 0.1641 | 0.2317 | 0.7080 | 0.5890 | 0.5625 |
| trend_lowvol | bear | 0.1690 | 0.2467 | 0.6850 | 0.4247 | 0.4880 |
| rsi_vol_combo | bear | 0.1313 | 0.2159 | 0.6085 | 0.4521 | 0.4418 |
| fund_pe | bear | 0.0920 | 0.2341 | 0.3930 | 0.2877 | 0.2531 |

### 水利建设

- **Neutral**: ['mom_x_lowvol_20_20', 'momentum_reversal', 'turnover_stability'] (单因子IC=0.0787, 组合IC=0.0977)
  - weights: [0.3565, 0.3239, 0.3196]
- **Bull**: ['low_downside', 'volatility', 'turnover_stability'] (单因子IC=0.0761, 组合IC=0.0936)
  - bull_weights: [0.3478, 0.3441, 0.3081]
- **Bear**: ['mom_x_lowvol_20_20', 'momentum_reversal'] (单因子IC=0.0969, 组合IC=0.0962)
  - bear_weights: [0.511, 0.489]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| mom_x_lowvol_20_20 | neutral | 0.0922 | 0.1867 | 0.4939 | 0.3716 | 0.3387 |
| momentum_reversal | neutral | 0.0883 | 0.1922 | 0.4592 | 0.3403 | 0.3077 |
| turnover_stability | neutral | 0.0555 | 0.1229 | 0.4516 | 0.3445 | 0.3036 |
| trend_lowvol | neutral | 0.0849 | 0.2262 | 0.3755 | 0.2735 | 0.2391 |
| volatility | neutral | 0.0711 | 0.1975 | 0.3602 | 0.3111 | 0.2361 |
| rsi_vol_combo | neutral | 0.0607 | 0.1826 | 0.3327 | 0.2672 | 0.2108 |
| fund_profit_growth | neutral | 0.0435 | 0.1455 | 0.2991 | 0.2129 | 0.1814 |
| fund_pb | neutral | 0.0498 | 0.2167 | 0.2299 | 0.1378 | 0.1308 |
| fund_score | neutral | 0.0403 | 0.1769 | 0.2278 | 0.1211 | 0.1277 |
| low_downside | neutral | 0.0438 | 0.2070 | 0.2118 | 0.1900 | 0.1260 |
| fund_pe | neutral | 0.0424 | 0.2297 | 0.1846 | 0.1357 | 0.1048 |
| low_downside | bull | 0.0802 | 0.1537 | 0.5216 | 0.4318 | 0.3734 |
| volatility | bull | 0.0900 | 0.1754 | 0.5133 | 0.4394 | 0.3694 |
| turnover_stability | bull | 0.0581 | 0.1124 | 0.5167 | 0.2803 | 0.3308 |
| trend_lowvol | bull | 0.0866 | 0.1928 | 0.4491 | 0.3485 | 0.3028 |
| momentum_reversal | bull | 0.0641 | 0.1644 | 0.3903 | 0.3030 | 0.2543 |
| mom_x_lowvol_20_20 | bull | 0.0543 | 0.1662 | 0.3267 | 0.2879 | 0.2104 |
| stroke_phase | bull | 0.0396 | 0.1224 | 0.3237 | 0.2803 | 0.2072 |
| fund_pb | bull | 0.0645 | 0.1973 | 0.3269 | 0.2652 | 0.2068 |
| fund_pe | bull | 0.0537 | 0.2033 | 0.2641 | 0.1591 | 0.1531 |
| rsi_vol_combo | bull | 0.0317 | 0.1650 | 0.1923 | 0.1364 | 0.1093 |
| mom_x_lowvol_20_20 | bear | 0.0961 | 0.2263 | 0.4246 | 0.3699 | 0.2908 |
| momentum_reversal | bear | 0.0978 | 0.2263 | 0.4321 | 0.2877 | 0.2782 |
| fund_revenue_growth | bear | 0.0210 | 0.1025 | 0.2045 | 0.3699 | 0.1401 |
| rsi_vol_combo | bear | 0.0426 | 0.1970 | 0.2161 | 0.1507 | 0.1243 |
| trend_lowvol | bear | 0.0489 | 0.2486 | 0.1967 | 0.1781 | 0.1159 |

### 汽车一体化压铸

- **Neutral**: ['trend_lowvol', 'momentum_reversal'] (单因子IC=0.0955, 组合IC=0.1031)
  - weights: [0.5269, 0.4731]
- **Bull**: ['momentum_reversal', 'fund_pe', 'mom_x_lowvol_20_20'] (单因子IC=0.0746, 组合IC=0.1102)
  - bull_weights: [0.3868, 0.3349, 0.2782]
- **Bear**: ['fund_pe', 'fund_roe', 'momentum_reversal'] (单因子IC=0.1253, 组合IC=0.217)
  - bear_weights: [0.496, 0.2672, 0.2368]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| trend_lowvol | neutral | 0.1000 | 0.2547 | 0.3928 | 0.2975 | 0.2548 |
| momentum_reversal | neutral | 0.0910 | 0.2584 | 0.3524 | 0.2985 | 0.2288 |
| mom_x_lowvol_20_20 | neutral | 0.0843 | 0.2617 | 0.3221 | 0.2651 | 0.2037 |
| rsi_vol_combo | neutral | 0.0714 | 0.2587 | 0.2760 | 0.2328 | 0.1701 |
| fund_pe | neutral | 0.0644 | 0.2356 | 0.2732 | 0.2255 | 0.1674 |
| volatility | neutral | 0.0697 | 0.2845 | 0.2451 | 0.1816 | 0.1448 |
| fund_profit_growth | neutral | 0.0477 | 0.2357 | 0.2025 | 0.1670 | 0.1182 |
| fund_score | neutral | 0.0378 | 0.2611 | 0.1446 | 0.1273 | 0.0815 |
| fund_gross_margin | neutral | 0.0324 | 0.2274 | 0.1425 | 0.1315 | 0.0806 |
| low_downside | neutral | 0.0243 | 0.2308 | 0.1052 | 0.1013 | 0.0579 |
| momentum_reversal | bull | 0.0858 | 0.2406 | 0.3564 | 0.3636 | 0.2430 |
| fund_pe | bull | 0.0706 | 0.2238 | 0.3156 | 0.3333 | 0.2104 |
| mom_x_lowvol_20_20 | bull | 0.0673 | 0.2356 | 0.2857 | 0.2235 | 0.1748 |
| fund_pb | bull | 0.0671 | 0.2549 | 0.2634 | 0.1477 | 0.1512 |
| rsi_vol_combo | bull | 0.0564 | 0.2347 | 0.2401 | 0.2197 | 0.1464 |
| volatility | bull | 0.0442 | 0.2817 | 0.1568 | 0.1818 | 0.0926 |
| turnover_stability | bull | 0.0224 | 0.2010 | 0.1116 | 0.2008 | 0.0670 |
| fund_pe | bear | 0.1691 | 0.2539 | 0.6658 | 0.5342 | 0.5108 |
| fund_roe | bear | 0.0954 | 0.2281 | 0.4185 | 0.3151 | 0.2751 |
| momentum_reversal | bear | 0.1115 | 0.2880 | 0.3870 | 0.2603 | 0.2439 |
| trend_lowvol | bear | 0.1166 | 0.3211 | 0.3632 | 0.1233 | 0.2040 |
| rsi_vol_combo | bear | 0.0769 | 0.2346 | 0.3280 | 0.2329 | 0.2022 |
| fund_profit_growth | bear | 0.0593 | 0.1954 | 0.3035 | 0.2055 | 0.1829 |
| fund_pb | bear | 0.0766 | 0.2678 | 0.2859 | 0.2329 | 0.1763 |
| turnover_stability | bear | 0.0541 | 0.2150 | 0.2518 | 0.2877 | 0.1621 |
| mom_x_lowvol_20_20 | bear | 0.0666 | 0.2703 | 0.2463 | 0.2603 | 0.1552 |

### 汽车整车

- **Neutral**: ['volatility', 'fund_pb', 'fund_score'] (单因子IC=0.0678, 组合IC=0.0869)
  - weights: [0.3372, 0.3357, 0.3271]
- **Bull**: ['trend_lowvol', 'fund_pe'] (单因子IC=0.039, 组合IC=0.0421)
  - bull_weights: [0.5528, 0.4472]
- **Bear**: ['fund_pe'] (单因子IC=0.049, 组合IC=0.0498)
  - bear_weights: [1.0]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| volatility | neutral | 0.0693 | 0.2788 | 0.2484 | 0.1524 | 0.1431 |
| fund_pb | neutral | 0.0728 | 0.2918 | 0.2494 | 0.1430 | 0.1425 |
| fund_score | neutral | 0.0613 | 0.2650 | 0.2311 | 0.2015 | 0.1389 |
| low_downside | neutral | 0.0434 | 0.2522 | 0.1719 | 0.1534 | 0.0992 |
| fund_pe | neutral | 0.0532 | 0.3265 | 0.1631 | 0.1180 | 0.0912 |
| fund_roe | neutral | 0.0464 | 0.2893 | 0.1603 | 0.1023 | 0.0884 |
| trend_lowvol | bull | 0.0368 | 0.2336 | 0.1577 | 0.1591 | 0.0914 |
| fund_pe | bull | 0.0411 | 0.3118 | 0.1319 | 0.1212 | 0.0739 |
| stroke_phase | bull | 0.0276 | 0.2220 | 0.1244 | 0.1288 | 0.0702 |
| turnover_stability | bull | 0.0224 | 0.2246 | 0.0997 | 0.1098 | 0.0553 |
| fund_pe | bear | 0.0490 | 0.2717 | 0.1802 | 0.1781 | 0.1061 |

### 汽车热管理

- **Neutral**: ['momentum_reversal', 'mom_x_lowvol_20_20', 'trend_lowvol'] (单因子IC=0.1221, 组合IC=0.1373)
  - weights: [0.3512, 0.3482, 0.3007]
- **Bull**: ['trend_lowvol', 'low_downside'] (单因子IC=0.0752, 组合IC=0.0877)
  - bull_weights: [0.5206, 0.4794]
- **Bear**: ['fund_pe', 'momentum_reversal'] (单因子IC=0.0935, 组合IC=0.1584)
  - bear_weights: [0.5106, 0.4894]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| momentum_reversal | neutral | 0.1263 | 0.1859 | 0.6795 | 0.4885 | 0.5057 |
| mom_x_lowvol_20_20 | neutral | 0.1218 | 0.1833 | 0.6644 | 0.5094 | 0.5014 |
| trend_lowvol | neutral | 0.1182 | 0.2017 | 0.5859 | 0.4781 | 0.4330 |
| rsi_vol_combo | neutral | 0.0904 | 0.1750 | 0.5166 | 0.3653 | 0.3527 |
| fund_pb | neutral | 0.0796 | 0.1712 | 0.4648 | 0.3466 | 0.3129 |
| volatility | neutral | 0.0828 | 0.2101 | 0.3941 | 0.2547 | 0.2472 |
| fund_pe | neutral | 0.0610 | 0.1639 | 0.3721 | 0.2839 | 0.2389 |
| fund_profit_growth | neutral | 0.0527 | 0.1560 | 0.3380 | 0.2777 | 0.2159 |
| turnover_stability | neutral | 0.0431 | 0.1388 | 0.3103 | 0.2004 | 0.1862 |
| low_downside | neutral | 0.0508 | 0.1946 | 0.2611 | 0.1858 | 0.1548 |
| fund_score | neutral | 0.0444 | 0.1840 | 0.2411 | 0.2109 | 0.1460 |
| fund_revenue_growth | neutral | 0.0297 | 0.1521 | 0.1953 | 0.2004 | 0.1172 |
| trend_lowvol | bull | 0.0790 | 0.2140 | 0.3691 | 0.3106 | 0.2419 |
| low_downside | bull | 0.0714 | 0.1980 | 0.3607 | 0.2348 | 0.2227 |
| volatility | bull | 0.0676 | 0.2129 | 0.3176 | 0.2879 | 0.2045 |
| turnover_stability | bull | 0.0456 | 0.1305 | 0.3492 | 0.1591 | 0.2024 |
| momentum_reversal | bull | 0.0310 | 0.1846 | 0.1681 | 0.1439 | 0.0961 |
| fund_pe | bull | 0.0223 | 0.1503 | 0.1481 | 0.2121 | 0.0898 |
| fund_pb | bull | 0.0308 | 0.1976 | 0.1559 | 0.1288 | 0.0880 |
| fund_gross_margin | bull | 0.0207 | 0.1534 | 0.1347 | 0.1288 | 0.0761 |
| fund_pe | bear | 0.0779 | 0.1635 | 0.4763 | 0.4521 | 0.3458 |
| momentum_reversal | bear | 0.1091 | 0.2299 | 0.4745 | 0.3973 | 0.3315 |
| rsi_vol_combo | bear | 0.0779 | 0.1977 | 0.3943 | 0.2603 | 0.2485 |
| mom_x_lowvol_20_20 | bear | 0.0864 | 0.2386 | 0.3622 | 0.3699 | 0.2481 |
| fund_gross_margin | bear | 0.0489 | 0.1672 | 0.2928 | 0.2329 | 0.1805 |
| fund_pb | bear | 0.0596 | 0.1952 | 0.3052 | 0.1233 | 0.1714 |
| trend_lowvol | bear | 0.0644 | 0.2522 | 0.2552 | 0.2055 | 0.1538 |

### 汽车芯片

- **Neutral**: ['momentum_reversal', 'fund_pb'] (单因子IC=0.0721, 组合IC=0.1034)
  - weights: [0.5173, 0.4827]
- **Bull**: ['volatility'] (单因子IC=0.106, 组合IC=0.106)
  - bull_weights: [1.0]
- **Bear**: ['mom_x_lowvol_20_20'] (单因子IC=0.108, 组合IC=0.108)
  - bear_weights: [1.0]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| momentum_reversal | neutral | 0.0731 | 0.2114 | 0.3459 | 0.2380 | 0.2141 |
| fund_pb | neutral | 0.0710 | 0.2285 | 0.3108 | 0.2860 | 0.1998 |
| mom_x_lowvol_20_20 | neutral | 0.0679 | 0.2112 | 0.3217 | 0.2359 | 0.1988 |
| rsi_vol_combo | neutral | 0.0620 | 0.2045 | 0.3033 | 0.2255 | 0.1859 |
| fund_pe | neutral | 0.0559 | 0.1879 | 0.2976 | 0.1837 | 0.1761 |
| trend_lowvol | neutral | 0.0607 | 0.2199 | 0.2761 | 0.1754 | 0.1623 |
| volatility | neutral | 0.0535 | 0.2310 | 0.2318 | 0.2046 | 0.1396 |
| fund_profit_growth | neutral | 0.0426 | 0.1971 | 0.2162 | 0.1649 | 0.1259 |
| turnover_stability | neutral | 0.0309 | 0.1626 | 0.1903 | 0.1712 | 0.1114 |
| fund_score | neutral | 0.0357 | 0.2356 | 0.1514 | 0.1044 | 0.0836 |
| volatility | bull | 0.1060 | 0.2131 | 0.4977 | 0.4545 | 0.3619 |
| low_downside | bull | 0.0833 | 0.1730 | 0.4817 | 0.3712 | 0.3302 |
| turnover_stability | bull | 0.0692 | 0.1539 | 0.4497 | 0.2879 | 0.2896 |
| fund_pb | bull | 0.0885 | 0.2217 | 0.3993 | 0.2424 | 0.2480 |
| trend_lowvol | bull | 0.0793 | 0.2234 | 0.3550 | 0.2841 | 0.2279 |
| mom_x_lowvol_20_20 | bull | 0.0482 | 0.2022 | 0.2386 | 0.2348 | 0.1473 |
| momentum_reversal | bull | 0.0457 | 0.2052 | 0.2228 | 0.2273 | 0.1367 |
| mom_x_lowvol_20_20 | bear | 0.1080 | 0.2106 | 0.5128 | 0.4795 | 0.3794 |
| bb_width_20 | bear | 0.0796 | 0.1904 | 0.4179 | 0.4795 | 0.3091 |
| momentum_reversal | bear | 0.0996 | 0.2229 | 0.4469 | 0.3699 | 0.3061 |
| rsi_vol_combo | bear | 0.0685 | 0.2078 | 0.3298 | 0.2329 | 0.2033 |
| fund_revenue_growth | bear | 0.0379 | 0.2026 | 0.1870 | 0.2055 | 0.1127 |

### 沪企改革

- **Neutral**: ['trend_lowvol'] (单因子IC=0.1274, 组合IC=0.1274)
  - weights: [1.0]
- **Bull**: ['trend_lowvol', 'turnover_stability', 'momentum_reversal'] (单因子IC=0.1155, 组合IC=0.1477)
  - bull_weights: [0.4104, 0.2956, 0.2939]
- **Bear**: ['mom_x_lowvol_20_20', 'momentum_reversal'] (单因子IC=0.1919, 组合IC=0.2015)
  - bear_weights: [0.6475, 0.3525]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| trend_lowvol | neutral | 0.1274 | 0.2277 | 0.5595 | 0.4134 | 0.3954 |
| momentum_reversal | neutral | 0.0983 | 0.2325 | 0.4229 | 0.3017 | 0.2753 |
| mom_x_lowvol_20_20 | neutral | 0.0840 | 0.2089 | 0.4021 | 0.2985 | 0.2611 |
| fund_pb | neutral | 0.0688 | 0.1899 | 0.3623 | 0.2317 | 0.2231 |
| rsi_vol_combo | neutral | 0.0683 | 0.2250 | 0.3033 | 0.2296 | 0.1865 |
| low_downside | neutral | 0.0593 | 0.2307 | 0.2571 | 0.2109 | 0.1556 |
| fund_pe | neutral | 0.0568 | 0.2505 | 0.2269 | 0.1691 | 0.1326 |
| volatility | neutral | 0.0339 | 0.2052 | 0.1654 | 0.1273 | 0.0933 |
| trend_lowvol | bull | 0.1389 | 0.2069 | 0.6713 | 0.5303 | 0.5137 |
| turnover_stability | bull | 0.0963 | 0.1784 | 0.5397 | 0.3712 | 0.3700 |
| momentum_reversal | bull | 0.1114 | 0.2100 | 0.5307 | 0.3864 | 0.3679 |
| volatility | bull | 0.0608 | 0.1667 | 0.3644 | 0.3295 | 0.2423 |
| rsi_vol_combo | bull | 0.0704 | 0.1969 | 0.3577 | 0.3485 | 0.2412 |
| low_downside | bull | 0.0783 | 0.2104 | 0.3723 | 0.2652 | 0.2355 |
| mom_x_lowvol_20_20 | bull | 0.0599 | 0.1717 | 0.3489 | 0.2462 | 0.2174 |
| fund_pb | bull | 0.0565 | 0.2004 | 0.2820 | 0.2576 | 0.1773 |
| fund_profit_growth | bull | 0.0357 | 0.1463 | 0.2437 | 0.1212 | 0.1366 |
| stroke_phase | bull | 0.0275 | 0.1813 | 0.1516 | 0.1439 | 0.0867 |
| fund_pe | bull | 0.0325 | 0.2437 | 0.1335 | 0.2045 | 0.0804 |
| fund_score | bull | 0.0259 | 0.1884 | 0.1374 | 0.1439 | 0.0786 |
| mom_x_lowvol_20_20 | bear | 0.1991 | 0.1917 | 1.0387 | 0.6712 | 0.8679 |
| momentum_reversal | bear | 0.1847 | 0.2784 | 0.6633 | 0.4247 | 0.4725 |
| rsi_vol_combo | bear | 0.1088 | 0.2474 | 0.4399 | 0.3699 | 0.3013 |
| trend_lowvol | bear | 0.1109 | 0.2537 | 0.4372 | 0.2603 | 0.2755 |

### 油气设服

- **Neutral**: ['volatility', 'fund_pb', 'trend_lowvol'] (单因子IC=0.0807, 组合IC=0.1041)
  - weights: [0.3692, 0.3415, 0.2893]
- **Bull**: ['exhaustion_risk', 'ma_alignment', 'ema20_slope'] (单因子IC=0.0594, 组合IC=0.069)
  - bull_weights: [0.3439, 0.3364, 0.3197]
- **Bear**: ['trend_lowvol', 'fund_profit_growth'] (单因子IC=0.1087, 组合IC=0.1395)
  - bear_weights: [0.5284, 0.4716]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| volatility | neutral | 0.0947 | 0.2634 | 0.3596 | 0.3278 | 0.2388 |
| fund_pb | neutral | 0.0727 | 0.2148 | 0.3385 | 0.3048 | 0.2209 |
| trend_lowvol | neutral | 0.0746 | 0.2551 | 0.2924 | 0.2797 | 0.1871 |
| turnover_stability | neutral | 0.0567 | 0.1875 | 0.3025 | 0.2317 | 0.1863 |
| fund_profit_growth | neutral | 0.0421 | 0.1794 | 0.2348 | 0.1858 | 0.1392 |
| momentum_reversal | neutral | 0.0558 | 0.2449 | 0.2278 | 0.1858 | 0.1350 |
| mom_x_lowvol_20_20 | neutral | 0.0559 | 0.2494 | 0.2242 | 0.1816 | 0.1325 |
| fund_revenue_growth | neutral | 0.0418 | 0.1872 | 0.2231 | 0.1712 | 0.1307 |
| low_downside | neutral | 0.0496 | 0.2334 | 0.2125 | 0.2192 | 0.1296 |
| fund_pe | neutral | 0.0484 | 0.2301 | 0.2102 | 0.2109 | 0.1273 |
| fund_score | neutral | 0.0449 | 0.2293 | 0.1957 | 0.1587 | 0.1134 |
| rsi_vol_combo | neutral | 0.0377 | 0.2336 | 0.1615 | 0.1096 | 0.0896 |
| fund_roe | neutral | 0.0286 | 0.2523 | 0.1133 | 0.1253 | 0.0637 |
| exhaustion_risk | bull | 0.0488 | 0.1764 | 0.2766 | 0.2314 | 0.1703 |
| ma_alignment | bull | 0.0632 | 0.2372 | 0.2666 | 0.2500 | 0.1666 |
| ema20_slope | bull | 0.0662 | 0.2455 | 0.2696 | 0.1742 | 0.1583 |
| low_downside | bull | 0.0463 | 0.2154 | 0.2150 | 0.2045 | 0.1295 |
| vol_confirm | bull | 0.0468 | 0.2238 | 0.2093 | 0.2121 | 0.1269 |
| relative_strength | bull | 0.0435 | 0.2230 | 0.1952 | 0.1970 | 0.1168 |
| fund_profit_growth | bull | 0.0328 | 0.1825 | 0.1799 | 0.2273 | 0.1104 |
| fund_pb | bull | 0.0409 | 0.2168 | 0.1885 | 0.1439 | 0.1078 |
| fund_gross_margin | bull | 0.0280 | 0.1721 | 0.1629 | 0.1212 | 0.0913 |
| trend_lowvol | bear | 0.1238 | 0.2113 | 0.5858 | 0.4247 | 0.4173 |
| fund_profit_growth | bear | 0.0936 | 0.1791 | 0.5228 | 0.4247 | 0.3724 |
| rsi_vol_combo | bear | 0.1033 | 0.2033 | 0.5079 | 0.4658 | 0.3723 |
| fund_revenue_growth | bear | 0.1252 | 0.2300 | 0.5445 | 0.3151 | 0.3580 |
| momentum_reversal | bear | 0.1324 | 0.2796 | 0.4735 | 0.3973 | 0.3308 |
| fund_roe | bear | 0.1213 | 0.2547 | 0.4761 | 0.2055 | 0.2870 |
| fund_gross_margin | bear | 0.0741 | 0.1622 | 0.4571 | 0.2329 | 0.2818 |
| fund_score | bear | 0.1104 | 0.2779 | 0.3973 | 0.2603 | 0.2503 |
| mom_x_lowvol_20_20 | bear | 0.1002 | 0.2926 | 0.3423 | 0.3699 | 0.2344 |
| fund_pe | bear | 0.0904 | 0.2597 | 0.3480 | 0.2603 | 0.2193 |
| fund_pb | bear | 0.0817 | 0.2433 | 0.3357 | 0.2603 | 0.2115 |
| volatility | bear | 0.0650 | 0.3067 | 0.2118 | 0.2055 | 0.1277 |

### 油气资源

- **Neutral**: ['mom_x_lowvol_20_20', 'momentum_reversal', 'fund_pb'] (单因子IC=0.0759, 组合IC=0.1076)
  - weights: [0.3509, 0.3354, 0.3137]
- **Bull**: ['fund_profit_growth', 'fund_pb', 'low_downside'] (单因子IC=0.0492, 组合IC=0.0653)
  - bull_weights: [0.3708, 0.3275, 0.3017]
- **Bear**: ['trend_lowvol', 'fund_profit_growth'] (单因子IC=0.1294, 组合IC=0.1586)
  - bear_weights: [0.5351, 0.4649]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| mom_x_lowvol_20_20 | neutral | 0.0784 | 0.1865 | 0.4205 | 0.3549 | 0.2849 |
| momentum_reversal | neutral | 0.0783 | 0.1916 | 0.4088 | 0.3319 | 0.2723 |
| fund_pb | neutral | 0.0710 | 0.1805 | 0.3936 | 0.2944 | 0.2547 |
| trend_lowvol | neutral | 0.0769 | 0.2068 | 0.3721 | 0.2944 | 0.2408 |
| volatility | neutral | 0.0725 | 0.2156 | 0.3361 | 0.3090 | 0.2200 |
| fund_profit_growth | neutral | 0.0403 | 0.1242 | 0.3248 | 0.2881 | 0.2092 |
| fund_score | neutral | 0.0474 | 0.1554 | 0.3049 | 0.2735 | 0.1941 |
| turnover_stability | neutral | 0.0417 | 0.1340 | 0.3112 | 0.2317 | 0.1917 |
| fund_pe | neutral | 0.0624 | 0.2074 | 0.3011 | 0.2547 | 0.1889 |
| rsi_vol_combo | neutral | 0.0543 | 0.1823 | 0.2981 | 0.2150 | 0.1811 |
| fund_revenue_growth | neutral | 0.0242 | 0.1201 | 0.2011 | 0.1169 | 0.1123 |
| low_downside | neutral | 0.0332 | 0.1945 | 0.1705 | 0.1775 | 0.1004 |
| fund_roe | neutral | 0.0325 | 0.1941 | 0.1675 | 0.1253 | 0.0942 |
| fund_profit_growth | bull | 0.0369 | 0.1020 | 0.3617 | 0.3030 | 0.2356 |
| fund_pb | bull | 0.0607 | 0.1800 | 0.3370 | 0.2348 | 0.2081 |
| low_downside | bull | 0.0500 | 0.1611 | 0.3106 | 0.2348 | 0.1918 |
| volatility | bull | 0.0511 | 0.1927 | 0.2652 | 0.2273 | 0.1627 |
| fund_pe | bull | 0.0515 | 0.1997 | 0.2577 | 0.1591 | 0.1493 |
| exhaustion_risk | bull | 0.0192 | 0.1019 | 0.1885 | 0.1136 | 0.1049 |
| ema20_slope | bull | 0.0244 | 0.1741 | 0.1401 | 0.1742 | 0.0822 |
| ma_alignment | bull | 0.0213 | 0.1716 | 0.1241 | 0.1970 | 0.0743 |
| relative_strength | bull | 0.0186 | 0.1520 | 0.1223 | 0.1591 | 0.0709 |
| stroke_phase | bull | 0.0116 | 0.1070 | 0.1089 | 0.1364 | 0.0619 |
| trend_lowvol | bear | 0.1503 | 0.2020 | 0.7444 | 0.5068 | 0.5608 |
| fund_profit_growth | bear | 0.1085 | 0.1647 | 0.6588 | 0.4795 | 0.4873 |
| rsi_vol_combo | bear | 0.1141 | 0.1849 | 0.6169 | 0.5616 | 0.4817 |
| fund_gross_margin | bear | 0.0765 | 0.1147 | 0.6671 | 0.3973 | 0.4660 |
| mom_x_lowvol_20_20 | bear | 0.1402 | 0.2466 | 0.5687 | 0.5616 | 0.4440 |
| momentum_reversal | bear | 0.1461 | 0.2389 | 0.6114 | 0.4521 | 0.4439 |
| fund_score | bear | 0.0892 | 0.1940 | 0.4597 | 0.4521 | 0.3337 |
| fund_revenue_growth | bear | 0.0460 | 0.1197 | 0.3841 | 0.3151 | 0.2525 |
| fund_roe | bear | 0.0641 | 0.2108 | 0.3038 | 0.3425 | 0.2039 |

### 流感

- **Neutral**: ['fund_pb', 'volatility', 'turnover_stability'] (单因子IC=0.0688, 组合IC=0.1004)
  - weights: [0.4619, 0.2713, 0.2668]
- **Bull**: ['fund_pb', 'low_downside', 'turnover_stability'] (单因子IC=0.0701, 组合IC=0.1053)
  - bull_weights: [0.3699, 0.3434, 0.2868]
- **Bear**: ['momentum_reversal', 'trend_lowvol'] (单因子IC=0.1258, 组合IC=0.1352)
  - bear_weights: [0.5013, 0.4987]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| fund_pb | neutral | 0.0853 | 0.1577 | 0.5413 | 0.4092 | 0.3814 |
| volatility | neutral | 0.0750 | 0.2173 | 0.3449 | 0.2985 | 0.2240 |
| turnover_stability | neutral | 0.0461 | 0.1321 | 0.3488 | 0.2630 | 0.2203 |
| momentum_reversal | neutral | 0.0637 | 0.1925 | 0.3309 | 0.2526 | 0.2073 |
| trend_lowvol | neutral | 0.0668 | 0.2124 | 0.3144 | 0.2443 | 0.1956 |
| low_downside | neutral | 0.0666 | 0.2197 | 0.3033 | 0.2672 | 0.1921 |
| mom_x_lowvol_20_20 | neutral | 0.0590 | 0.1916 | 0.3079 | 0.2025 | 0.1851 |
| rsi_vol_combo | neutral | 0.0480 | 0.1787 | 0.2688 | 0.2129 | 0.1630 |
| fund_pe | neutral | 0.0534 | 0.2025 | 0.2637 | 0.2171 | 0.1605 |
| fund_profit_growth | neutral | 0.0352 | 0.1379 | 0.2552 | 0.1921 | 0.1521 |
| fund_revenue_growth | neutral | 0.0265 | 0.1318 | 0.2008 | 0.2067 | 0.1212 |
| fund_score | neutral | 0.0367 | 0.1792 | 0.2050 | 0.1775 | 0.1207 |
| fund_pb | bull | 0.0769 | 0.1475 | 0.5210 | 0.3030 | 0.3394 |
| low_downside | bull | 0.0857 | 0.1865 | 0.4596 | 0.3712 | 0.3151 |
| turnover_stability | bull | 0.0478 | 0.1204 | 0.3970 | 0.3258 | 0.2632 |
| volatility | bull | 0.0647 | 0.1789 | 0.3617 | 0.2955 | 0.2343 |
| rsi_vol_combo | bull | 0.0350 | 0.1589 | 0.2205 | 0.2045 | 0.1328 |
| trend_lowvol | bull | 0.0379 | 0.1776 | 0.2133 | 0.1212 | 0.1196 |
| fund_pe | bull | 0.0345 | 0.2027 | 0.1702 | 0.1212 | 0.0954 |
| momentum_reversal | bull | 0.0210 | 0.1698 | 0.1238 | 0.1515 | 0.0713 |
| momentum_reversal | bear | 0.1298 | 0.2297 | 0.5648 | 0.4247 | 0.4023 |
| trend_lowvol | bear | 0.1219 | 0.2169 | 0.5617 | 0.4247 | 0.4001 |
| mom_x_lowvol_20_20 | bear | 0.0915 | 0.2166 | 0.4224 | 0.3973 | 0.2951 |
| fund_gross_margin | bear | 0.0617 | 0.1572 | 0.3928 | 0.3425 | 0.2637 |
| turnover_stability | bear | 0.0559 | 0.1371 | 0.4076 | 0.1781 | 0.2401 |
| rsi_vol_combo | bear | 0.0840 | 0.2352 | 0.3572 | 0.2877 | 0.2300 |
| fund_pe | bear | 0.0631 | 0.2158 | 0.2922 | 0.2055 | 0.1761 |
| fund_pb | bear | 0.0462 | 0.1582 | 0.2921 | 0.1781 | 0.1721 |
| fund_revenue_growth | bear | 0.0348 | 0.1381 | 0.2520 | 0.2603 | 0.1588 |

### 海南自贸

- **Neutral**: ['volatility', 'mom_x_lowvol_20_20', 'momentum_reversal'] (单因子IC=0.0662, 组合IC=0.0794)
  - weights: [0.3523, 0.3294, 0.3183]
- **Bull**: ['low_downside', 'volatility', 'turnover_stability'] (单因子IC=0.0927, 组合IC=0.1121)
  - bull_weights: [0.4081, 0.3105, 0.2814]
- **Bear**: ['momentum_reversal', 'mom_x_lowvol_20_20', 'trend_lowvol'] (单因子IC=0.1255, 组合IC=0.14)
  - bear_weights: [0.3482, 0.3299, 0.322]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| volatility | neutral | 0.0680 | 0.1819 | 0.3737 | 0.3111 | 0.2450 |
| mom_x_lowvol_20_20 | neutral | 0.0648 | 0.1783 | 0.3633 | 0.2610 | 0.2291 |
| momentum_reversal | neutral | 0.0659 | 0.1835 | 0.3594 | 0.2317 | 0.2213 |
| trend_lowvol | neutral | 0.0686 | 0.2027 | 0.3383 | 0.2735 | 0.2154 |
| rsi_vol_combo | neutral | 0.0564 | 0.1711 | 0.3297 | 0.2088 | 0.1993 |
| turnover_stability | neutral | 0.0402 | 0.1288 | 0.3124 | 0.2150 | 0.1898 |
| fund_pb | neutral | 0.0495 | 0.1601 | 0.3095 | 0.2192 | 0.1887 |
| fund_pe | neutral | 0.0476 | 0.1701 | 0.2800 | 0.2443 | 0.1742 |
| low_downside | neutral | 0.0374 | 0.1738 | 0.2154 | 0.1942 | 0.1286 |
| fund_profit_growth | neutral | 0.0221 | 0.1305 | 0.1696 | 0.1065 | 0.0938 |
| low_downside | bull | 0.1060 | 0.1355 | 0.7821 | 0.5682 | 0.6132 |
| volatility | bull | 0.0995 | 0.1591 | 0.6254 | 0.4924 | 0.4667 |
| turnover_stability | bull | 0.0725 | 0.1221 | 0.5938 | 0.4242 | 0.4229 |
| fund_pb | bull | 0.0829 | 0.1466 | 0.5659 | 0.4545 | 0.4116 |
| fund_pe | bull | 0.0731 | 0.1593 | 0.4586 | 0.3333 | 0.3057 |
| rsi_vol_combo | bull | 0.0483 | 0.1599 | 0.3020 | 0.2500 | 0.1888 |
| mom_x_lowvol_20_20 | bull | 0.0495 | 0.1779 | 0.2782 | 0.2576 | 0.1749 |
| momentum_reversal | bull | 0.0493 | 0.1744 | 0.2825 | 0.2273 | 0.1733 |
| trend_lowvol | bull | 0.0475 | 0.1892 | 0.2509 | 0.2273 | 0.1540 |
| fund_score | bull | 0.0289 | 0.1535 | 0.1886 | 0.1364 | 0.1071 |
| fund_profit_growth | bull | 0.0168 | 0.1168 | 0.1439 | 0.1667 | 0.0840 |
| fund_roe | bull | 0.0234 | 0.1666 | 0.1405 | 0.1364 | 0.0798 |
| momentum_reversal | bear | 0.1275 | 0.1622 | 0.7862 | 0.5068 | 0.5924 |
| mom_x_lowvol_20_20 | bear | 0.1316 | 0.1703 | 0.7730 | 0.4521 | 0.5612 |
| trend_lowvol | bear | 0.1172 | 0.1583 | 0.7405 | 0.4795 | 0.5478 |
| rsi_vol_combo | bear | 0.0902 | 0.1403 | 0.6429 | 0.4247 | 0.4580 |
| bb_width_20 | bear | 0.0698 | 0.1986 | 0.3516 | 0.3699 | 0.2408 |
| fund_gross_margin | bear | 0.0379 | 0.1238 | 0.3061 | 0.3151 | 0.2013 |
| fund_profit_growth | bear | 0.0241 | 0.1009 | 0.2391 | 0.2055 | 0.1441 |
| fund_score | bear | 0.0361 | 0.1913 | 0.1886 | 0.1781 | 0.1111 |

### 海工装备

- **Neutral**: ['volatility', 'trend_lowvol', 'mom_x_lowvol_20_20'] (单因子IC=0.089, 组合IC=0.122)
  - weights: [0.3534, 0.3428, 0.3038]
- **Bull**: ['low_downside', 'volatility'] (单因子IC=0.0982, 组合IC=0.1037)
  - bull_weights: [0.5639, 0.4361]
- **Bear**: ['mom_x_lowvol_20_20', 'fund_gross_margin', 'rsi_vol_combo'] (单因子IC=0.073, 组合IC=0.1047)
  - bear_weights: [0.3545, 0.347, 0.2985]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| volatility | neutral | 0.0876 | 0.1917 | 0.4568 | 0.4113 | 0.3223 |
| trend_lowvol | neutral | 0.0980 | 0.2163 | 0.4532 | 0.3800 | 0.3127 |
| mom_x_lowvol_20_20 | neutral | 0.0814 | 0.1923 | 0.4234 | 0.3090 | 0.2771 |
| momentum_reversal | neutral | 0.0810 | 0.2103 | 0.3852 | 0.2871 | 0.2479 |
| fund_pb | neutral | 0.0721 | 0.1912 | 0.3770 | 0.2505 | 0.2357 |
| turnover_stability | neutral | 0.0471 | 0.1496 | 0.3150 | 0.2735 | 0.2006 |
| rsi_vol_combo | neutral | 0.0638 | 0.1998 | 0.3192 | 0.2255 | 0.1956 |
| fund_revenue_growth | neutral | 0.0478 | 0.1547 | 0.3091 | 0.2046 | 0.1862 |
| low_downside | neutral | 0.0517 | 0.2022 | 0.2560 | 0.2276 | 0.1571 |
| fund_profit_growth | neutral | 0.0385 | 0.1573 | 0.2450 | 0.2025 | 0.1473 |
| fund_score | neutral | 0.0408 | 0.1977 | 0.2063 | 0.1545 | 0.1191 |
| fund_pe | neutral | 0.0370 | 0.1944 | 0.1904 | 0.1649 | 0.1109 |
| low_downside | bull | 0.1018 | 0.1811 | 0.5622 | 0.4470 | 0.4067 |
| volatility | bull | 0.0946 | 0.1936 | 0.4885 | 0.2879 | 0.3146 |
| fund_pb | bull | 0.0813 | 0.1829 | 0.4444 | 0.3485 | 0.2996 |
| fund_pe | bull | 0.0509 | 0.1929 | 0.2638 | 0.2500 | 0.1649 |
| turnover_stability | bull | 0.0340 | 0.1315 | 0.2586 | 0.2500 | 0.1616 |
| fund_score | bull | 0.0463 | 0.1829 | 0.2532 | 0.1591 | 0.1467 |
| fund_profit_growth | bull | 0.0298 | 0.1352 | 0.2205 | 0.1212 | 0.1236 |
| trend_lowvol | bull | 0.0427 | 0.2074 | 0.2058 | 0.1667 | 0.1201 |
| exhaustion_risk | bull | 0.0249 | 0.1518 | 0.1642 | 0.1450 | 0.0940 |
| fund_roe | bull | 0.0326 | 0.1946 | 0.1674 | 0.1061 | 0.0926 |
| fund_revenue_growth | bull | 0.0231 | 0.1628 | 0.1417 | 0.1894 | 0.0843 |
| mom_x_lowvol_20_20 | bear | 0.0851 | 0.2360 | 0.3606 | 0.1781 | 0.2124 |
| fund_gross_margin | bear | 0.0706 | 0.2000 | 0.3531 | 0.1781 | 0.2080 |
| rsi_vol_combo | bear | 0.0634 | 0.2136 | 0.2968 | 0.2055 | 0.1789 |
| fund_pb | bear | 0.0463 | 0.2122 | 0.2182 | 0.1507 | 0.1255 |

### 海洋经济

- **Neutral**: ['trend_lowvol', 'mom_x_lowvol_20_20'] (单因子IC=0.0929, 组合IC=0.1105)
  - weights: [0.5164, 0.4836]
- **Bull**: ['low_downside', 'volatility', 'trend_lowvol'] (单因子IC=0.1003, 组合IC=0.1167)
  - bull_weights: [0.4098, 0.3328, 0.2574]
- **Bear**: ['fund_profit_growth', 'momentum_reversal', 'mom_x_lowvol_20_20'] (单因子IC=0.0664, 组合IC=0.0759)
  - bear_weights: [0.3419, 0.3368, 0.3212]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| trend_lowvol | neutral | 0.1003 | 0.2003 | 0.5008 | 0.3695 | 0.3430 |
| mom_x_lowvol_20_20 | neutral | 0.0855 | 0.1796 | 0.4763 | 0.3486 | 0.3211 |
| momentum_reversal | neutral | 0.0849 | 0.1896 | 0.4479 | 0.3528 | 0.3030 |
| volatility | neutral | 0.0775 | 0.1874 | 0.4137 | 0.3779 | 0.2850 |
| rsi_vol_combo | neutral | 0.0670 | 0.1844 | 0.3633 | 0.2568 | 0.2283 |
| fund_pb | neutral | 0.0676 | 0.1909 | 0.3543 | 0.2213 | 0.2164 |
| turnover_stability | neutral | 0.0419 | 0.1278 | 0.3277 | 0.2610 | 0.2066 |
| low_downside | neutral | 0.0447 | 0.1928 | 0.2320 | 0.2025 | 0.1395 |
| fund_profit_growth | neutral | 0.0298 | 0.1384 | 0.2151 | 0.1608 | 0.1248 |
| fund_score | neutral | 0.0347 | 0.1663 | 0.2088 | 0.1733 | 0.1225 |
| fund_pe | neutral | 0.0331 | 0.2106 | 0.1571 | 0.1190 | 0.0879 |
| low_downside | bull | 0.1105 | 0.1749 | 0.6318 | 0.4015 | 0.4427 |
| volatility | bull | 0.1042 | 0.1953 | 0.5332 | 0.3485 | 0.3595 |
| trend_lowvol | bull | 0.0864 | 0.1918 | 0.4504 | 0.2348 | 0.2781 |
| fund_pb | bull | 0.0605 | 0.1876 | 0.3225 | 0.2500 | 0.2016 |
| momentum_reversal | bull | 0.0438 | 0.1487 | 0.2949 | 0.2197 | 0.1799 |
| mom_x_lowvol_20_20 | bull | 0.0344 | 0.1593 | 0.2158 | 0.1894 | 0.1283 |
| fund_pe | bull | 0.0375 | 0.1943 | 0.1929 | 0.1288 | 0.1089 |
| rsi_vol_combo | bull | 0.0244 | 0.1420 | 0.1716 | 0.1288 | 0.0969 |
| turnover_stability | bull | 0.0188 | 0.1228 | 0.1531 | 0.1061 | 0.0847 |
| fund_profit_growth | bear | 0.0399 | 0.1133 | 0.3520 | 0.3425 | 0.2363 |
| momentum_reversal | bear | 0.0825 | 0.2331 | 0.3540 | 0.3151 | 0.2328 |
| mom_x_lowvol_20_20 | bear | 0.0768 | 0.2368 | 0.3241 | 0.3699 | 0.2220 |
| rsi_vol_combo | bear | 0.0608 | 0.1983 | 0.3066 | 0.1781 | 0.1806 |
| turnover_stability | bear | 0.0412 | 0.1344 | 0.3070 | 0.1507 | 0.1766 |
| bb_width_20 | bear | 0.0601 | 0.2233 | 0.2694 | 0.1781 | 0.1587 |
| trend_lowvol | bear | 0.0530 | 0.2606 | 0.2036 | 0.2603 | 0.1283 |
| fund_pb | bear | 0.0463 | 0.2301 | 0.2012 | 0.1781 | 0.1185 |

### 海绵城市

- **Neutral**: ['mom_x_lowvol_20_20'] (单因子IC=0.093, 组合IC=0.093)
  - weights: [1.0]
- **Bull**: ['trend_lowvol', 'low_downside'] (单因子IC=0.118, 组合IC=0.1449)
  - bull_weights: [0.5125, 0.4875]
- **Bear**: ['trend_lowvol'] (单因子IC=0.1227, 组合IC=0.1227)
  - bear_weights: [1.0]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| mom_x_lowvol_20_20 | neutral | 0.0930 | 0.2364 | 0.3934 | 0.3058 | 0.2569 |
| momentum_reversal | neutral | 0.0788 | 0.2412 | 0.3269 | 0.2630 | 0.2065 |
| fund_profit_growth | neutral | 0.0586 | 0.1848 | 0.3170 | 0.2505 | 0.1982 |
| fund_pb | neutral | 0.0682 | 0.2170 | 0.3144 | 0.1858 | 0.1864 |
| fund_pe | neutral | 0.0681 | 0.2667 | 0.2554 | 0.1545 | 0.1474 |
| fund_score | neutral | 0.0522 | 0.2193 | 0.2382 | 0.1336 | 0.1350 |
| rsi_vol_combo | neutral | 0.0512 | 0.2295 | 0.2232 | 0.1608 | 0.1296 |
| turnover_stability | neutral | 0.0443 | 0.2055 | 0.2154 | 0.1670 | 0.1257 |
| volatility | neutral | 0.0514 | 0.2752 | 0.1867 | 0.1712 | 0.1094 |
| low_downside | neutral | 0.0448 | 0.2502 | 0.1790 | 0.1566 | 0.1035 |
| fund_revenue_growth | neutral | 0.0247 | 0.1902 | 0.1297 | 0.1044 | 0.0716 |
| fund_gross_margin | neutral | 0.0151 | 0.2154 | 0.0699 | 0.1127 | 0.0389 |
| trend_lowvol | bull | 0.1248 | 0.2137 | 0.5837 | 0.4697 | 0.4289 |
| low_downside | bull | 0.1113 | 0.1941 | 0.5731 | 0.4242 | 0.4081 |
| volatility | bull | 0.1138 | 0.2328 | 0.4887 | 0.3258 | 0.3239 |
| fund_pb | bull | 0.0922 | 0.2011 | 0.4583 | 0.3144 | 0.3012 |
| turnover_stability | bull | 0.0689 | 0.1672 | 0.4121 | 0.3333 | 0.2747 |
| fund_pe | bull | 0.1123 | 0.2602 | 0.4314 | 0.2727 | 0.2745 |
| stroke_phase | bull | 0.0472 | 0.1738 | 0.2714 | 0.2348 | 0.1676 |
| momentum_reversal | bull | 0.0543 | 0.1945 | 0.2790 | 0.1742 | 0.1638 |
| fund_roe | bull | 0.0500 | 0.2638 | 0.1895 | 0.1742 | 0.1113 |
| trend_lowvol | bear | 0.1227 | 0.2687 | 0.4565 | 0.2877 | 0.2939 |
| momentum_reversal | bear | 0.0877 | 0.2471 | 0.3550 | 0.2877 | 0.2286 |
| rsi_vol_combo | bear | 0.0670 | 0.2198 | 0.3050 | 0.2055 | 0.1838 |
| mom_x_lowvol_20_20 | bear | 0.0705 | 0.2272 | 0.3105 | 0.1781 | 0.1829 |
| fund_pe | bear | 0.0777 | 0.2839 | 0.2737 | 0.2329 | 0.1687 |
| fund_score | bear | 0.0447 | 0.1772 | 0.2521 | 0.1781 | 0.1485 |
| fund_revenue_growth | bear | 0.0374 | 0.1681 | 0.2225 | 0.2329 | 0.1371 |

### 消费电子概念

- **Neutral**: ['fund_pb', 'mom_x_lowvol_20_20', 'momentum_reversal'] (单因子IC=0.0818, 组合IC=0.114)
  - weights: [0.3969, 0.3049, 0.2982]
- **Bull**: ['fund_pb', 'turnover_stability', 'volatility'] (单因子IC=0.0676, 组合IC=0.1075)
  - bull_weights: [0.4434, 0.3298, 0.2268]
- **Bear**: ['momentum_reversal'] (单因子IC=0.1119, 组合IC=0.1119)
  - bear_weights: [1.0]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| fund_pb | neutral | 0.0788 | 0.1403 | 0.5620 | 0.4781 | 0.4153 |
| mom_x_lowvol_20_20 | neutral | 0.0831 | 0.1737 | 0.4783 | 0.3340 | 0.3191 |
| momentum_reversal | neutral | 0.0836 | 0.1793 | 0.4664 | 0.3382 | 0.3121 |
| trend_lowvol | neutral | 0.0769 | 0.1955 | 0.3934 | 0.2902 | 0.2538 |
| rsi_vol_combo | neutral | 0.0644 | 0.1658 | 0.3883 | 0.2818 | 0.2489 |
| fund_pe | neutral | 0.0497 | 0.1458 | 0.3413 | 0.2589 | 0.2148 |
| volatility | neutral | 0.0593 | 0.1825 | 0.3248 | 0.2213 | 0.1983 |
| fund_profit_growth | neutral | 0.0399 | 0.1448 | 0.2758 | 0.2171 | 0.1678 |
| fund_score | neutral | 0.0335 | 0.1792 | 0.1868 | 0.1545 | 0.1078 |
| low_downside | neutral | 0.0300 | 0.1644 | 0.1826 | 0.1441 | 0.1044 |
| turnover_stability | neutral | 0.0201 | 0.1142 | 0.1759 | 0.1253 | 0.0990 |
| fund_pb | bull | 0.0951 | 0.1515 | 0.6279 | 0.4470 | 0.4543 |
| turnover_stability | bull | 0.0469 | 0.0966 | 0.4849 | 0.3939 | 0.3380 |
| volatility | bull | 0.0610 | 0.1769 | 0.3446 | 0.3485 | 0.2324 |
| low_downside | bull | 0.0524 | 0.1515 | 0.3459 | 0.3182 | 0.2280 |
| mom_x_lowvol_20_20 | bull | 0.0455 | 0.1682 | 0.2706 | 0.2803 | 0.1732 |
| trend_lowvol | bull | 0.0532 | 0.1909 | 0.2786 | 0.2045 | 0.1678 |
| fund_pe | bull | 0.0466 | 0.1548 | 0.3014 | 0.1061 | 0.1667 |
| momentum_reversal | bull | 0.0451 | 0.1711 | 0.2634 | 0.2424 | 0.1636 |
| rsi_vol_combo | bull | 0.0287 | 0.1503 | 0.1907 | 0.1591 | 0.1105 |
| fund_profit_growth | bull | 0.0219 | 0.1287 | 0.1704 | 0.1364 | 0.0968 |
| momentum_reversal | bear | 0.1119 | 0.1775 | 0.6306 | 0.3699 | 0.4319 |
| mom_x_lowvol_20_20 | bear | 0.1064 | 0.1892 | 0.5625 | 0.3973 | 0.3930 |
| rsi_vol_combo | bear | 0.0677 | 0.1381 | 0.4904 | 0.5068 | 0.3695 |
| bb_width_20 | bear | 0.0539 | 0.1588 | 0.3392 | 0.2055 | 0.2044 |
| trend_lowvol | bear | 0.0738 | 0.2124 | 0.3475 | 0.1507 | 0.2000 |
| turnover_stability | bear | 0.0380 | 0.1168 | 0.3256 | 0.1781 | 0.1918 |
| wash_sale_score | bear | 0.0353 | 0.1145 | 0.3084 | 0.2000 | 0.1850 |
| fund_pb | bear | 0.0487 | 0.1734 | 0.2809 | 0.1781 | 0.1655 |

### 液冷概念

- **Neutral**: ['fund_pb', 'fund_pe', 'mom_x_lowvol_20_20'] (单因子IC=0.0761, 组合IC=0.0913)
  - weights: [0.4084, 0.315, 0.2767]
- **Bull**: ['fund_pb'] (单因子IC=0.0848, 组合IC=0.0848)
  - bull_weights: [1.0]
- **Bear**: ['mom_x_lowvol_20_20'] (单因子IC=0.1013, 组合IC=0.1013)
  - bear_weights: [1.0]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| fund_pb | neutral | 0.0853 | 0.1505 | 0.5670 | 0.3862 | 0.3930 |
| fund_pe | neutral | 0.0714 | 0.1573 | 0.4538 | 0.3361 | 0.3031 |
| mom_x_lowvol_20_20 | neutral | 0.0716 | 0.1721 | 0.4161 | 0.2797 | 0.2662 |
| momentum_reversal | neutral | 0.0716 | 0.1842 | 0.3888 | 0.2589 | 0.2447 |
| trend_lowvol | neutral | 0.0691 | 0.2043 | 0.3380 | 0.2296 | 0.2078 |
| fund_profit_growth | neutral | 0.0426 | 0.1413 | 0.3016 | 0.2004 | 0.1811 |
| volatility | neutral | 0.0575 | 0.2037 | 0.2824 | 0.2296 | 0.1737 |
| rsi_vol_combo | neutral | 0.0474 | 0.1749 | 0.2712 | 0.1795 | 0.1600 |
| fund_score | neutral | 0.0404 | 0.1699 | 0.2380 | 0.1524 | 0.1371 |
| fund_roe | neutral | 0.0344 | 0.1795 | 0.1916 | 0.1566 | 0.1108 |
| fund_revenue_growth | neutral | 0.0156 | 0.1338 | 0.1165 | 0.1273 | 0.0657 |
| low_downside | neutral | 0.0195 | 0.1973 | 0.0986 | 0.1190 | 0.0552 |
| fund_pb | bull | 0.0848 | 0.1592 | 0.5323 | 0.3712 | 0.3649 |
| fund_pe | bull | 0.0643 | 0.1532 | 0.4195 | 0.3409 | 0.2813 |
| volatility | bull | 0.0603 | 0.1728 | 0.3487 | 0.3561 | 0.2365 |
| low_downside | bull | 0.0547 | 0.1622 | 0.3372 | 0.3106 | 0.2210 |
| trend_lowvol | bull | 0.0568 | 0.1989 | 0.2856 | 0.1667 | 0.1666 |
| turnover_stability | bull | 0.0262 | 0.1215 | 0.2160 | 0.1894 | 0.1285 |
| fund_gross_margin | bull | 0.0180 | 0.0837 | 0.2147 | 0.1667 | 0.1252 |
| rsi_vol_combo | bull | 0.0298 | 0.1489 | 0.2002 | 0.1288 | 0.1130 |
| fund_score | bull | 0.0205 | 0.1519 | 0.1349 | 0.1591 | 0.0782 |
| fund_profit_growth | bull | 0.0149 | 0.1105 | 0.1348 | 0.1364 | 0.0766 |
| fund_roe | bull | 0.0161 | 0.1570 | 0.1026 | 0.1288 | 0.0579 |
| mom_x_lowvol_20_20 | bear | 0.1013 | 0.1980 | 0.5115 | 0.3151 | 0.3363 |
| rsi_vol_combo | bear | 0.0688 | 0.1484 | 0.4634 | 0.3699 | 0.3174 |
| momentum_reversal | bear | 0.0981 | 0.2104 | 0.4663 | 0.3425 | 0.3130 |
| trend_lowvol | bear | 0.0902 | 0.2126 | 0.4242 | 0.2329 | 0.2615 |
| fund_revenue_growth | bear | 0.0379 | 0.1015 | 0.3733 | 0.3699 | 0.2557 |
| fund_gross_margin | bear | 0.0415 | 0.1139 | 0.3647 | 0.2877 | 0.2348 |
| fund_pb | bear | 0.0544 | 0.1599 | 0.3400 | 0.2877 | 0.2189 |

### 深证100R

- **Neutral**: ['fund_profit_growth', 'top_fractal_volume'] (单因子IC=0.0349, 组合IC=0.0511)
  - weights: [0.5758, 0.4242]
- **Bull**: ['fund_profit_growth', 'ema20_slope', 'relative_strength'] (单因子IC=0.0388, 组合IC=0.0538)
  - bull_weights: [0.3535, 0.3398, 0.3067]
- **Bear**: ['limit_pullback_score', 'bb_width_20', 'momentum_reversal'] (单因子IC=0.1118, 组合IC=0.1774)
  - bear_weights: [0.363, 0.3378, 0.2991]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| fund_profit_growth | neutral | 0.0472 | 0.1782 | 0.2648 | 0.1733 | 0.1553 |
| top_fractal_volume | neutral | 0.0226 | 0.1166 | 0.1939 | 0.1803 | 0.1145 |
| fund_score | neutral | 0.0304 | 0.1751 | 0.1738 | 0.1253 | 0.0978 |
| trend_lowvol | neutral | 0.0398 | 0.2301 | 0.1729 | 0.1253 | 0.0973 |
| bb_width_20 | neutral | 0.0239 | 0.2078 | 0.1149 | 0.1044 | 0.0635 |
| fund_revenue_growth | neutral | 0.0126 | 0.1674 | 0.0753 | 0.1148 | 0.0420 |
| turnover_stability | neutral | 0.0097 | 0.1400 | 0.0695 | 0.1086 | 0.0385 |
| fund_profit_growth | bull | 0.0426 | 0.1959 | 0.2173 | 0.1364 | 0.1235 |
| ema20_slope | bull | 0.0394 | 0.2124 | 0.1854 | 0.2803 | 0.1187 |
| relative_strength | bull | 0.0345 | 0.1950 | 0.1767 | 0.2121 | 0.1071 |
| fund_score | bull | 0.0345 | 0.1825 | 0.1890 | 0.1288 | 0.1067 |
| turnover_stability | bull | 0.0217 | 0.1236 | 0.1756 | 0.1894 | 0.1044 |
| ma_alignment | bull | 0.0324 | 0.2200 | 0.1475 | 0.2803 | 0.0944 |
| low_downside | bull | 0.0243 | 0.2131 | 0.1140 | 0.1061 | 0.0631 |
| limit_pullback_score | bear | 0.0776 | 0.1227 | 0.6323 | 0.5294 | 0.4835 |
| bb_width_20 | bear | 0.1315 | 0.2202 | 0.5973 | 0.5068 | 0.4500 |
| momentum_reversal | bear | 0.1263 | 0.2388 | 0.5288 | 0.5068 | 0.3984 |
| mom_x_lowvol_20_20 | bear | 0.0939 | 0.1913 | 0.4908 | 0.3973 | 0.3429 |
| trend_lowvol | bear | 0.1338 | 0.2800 | 0.4780 | 0.3151 | 0.3143 |
| wash_sale_score | bear | 0.0496 | 0.1207 | 0.4110 | 0.3125 | 0.2697 |
| rsi_vol_combo | bear | 0.0771 | 0.1936 | 0.3981 | 0.3151 | 0.2617 |
| fund_gross_margin | bear | 0.0392 | 0.1144 | 0.3427 | 0.3151 | 0.2253 |
| fund_pb | bear | 0.0432 | 0.1803 | 0.2394 | 0.2603 | 0.1509 |

### 混合现实

- **Neutral**: ['fund_pb', 'momentum_reversal', 'mom_x_lowvol_20_20'] (单因子IC=0.0884, 组合IC=0.1296)
  - weights: [0.3572, 0.3289, 0.3139]
- **Bull**: ['mom_x_lowvol_20_20', 'momentum_reversal', 'rsi_vol_combo'] (单因子IC=0.089, 组合IC=0.0947)
  - bull_weights: [0.3394, 0.3378, 0.3228]
- **Bear**: ['fund_gross_margin', 'fund_pb', 'fund_pe'] (单因子IC=0.0776, 组合IC=0.1294)
  - bear_weights: [0.4317, 0.3773, 0.191]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| fund_pb | neutral | 0.0937 | 0.2495 | 0.3754 | 0.2860 | 0.2414 |
| momentum_reversal | neutral | 0.0870 | 0.2478 | 0.3510 | 0.2662 | 0.2222 |
| mom_x_lowvol_20_20 | neutral | 0.0844 | 0.2477 | 0.3409 | 0.2443 | 0.2121 |
| fund_pe | neutral | 0.0747 | 0.2480 | 0.3014 | 0.2923 | 0.1947 |
| rsi_vol_combo | neutral | 0.0725 | 0.2513 | 0.2886 | 0.2286 | 0.1773 |
| trend_lowvol | neutral | 0.0749 | 0.2880 | 0.2600 | 0.2547 | 0.1631 |
| volatility | neutral | 0.0638 | 0.2841 | 0.2246 | 0.1921 | 0.1339 |
| fund_profit_growth | neutral | 0.0472 | 0.2374 | 0.1990 | 0.1597 | 0.1154 |
| fund_score | neutral | 0.0439 | 0.2558 | 0.1715 | 0.1420 | 0.0979 |
| fund_revenue_growth | neutral | 0.0276 | 0.2348 | 0.1175 | 0.1044 | 0.0649 |
| mom_x_lowvol_20_20 | bull | 0.0871 | 0.2118 | 0.4110 | 0.2652 | 0.2600 |
| momentum_reversal | bull | 0.0889 | 0.2356 | 0.3773 | 0.3712 | 0.2587 |
| rsi_vol_combo | bull | 0.0911 | 0.2456 | 0.3709 | 0.3333 | 0.2472 |
| turnover_stability | bull | 0.0582 | 0.2028 | 0.2870 | 0.1667 | 0.1674 |
| volatility | bull | 0.0590 | 0.2424 | 0.2433 | 0.2348 | 0.1502 |
| trend_lowvol | bull | 0.0681 | 0.2714 | 0.2510 | 0.1667 | 0.1464 |
| fund_pb | bull | 0.0412 | 0.2005 | 0.2055 | 0.1932 | 0.1226 |
| fund_gross_margin | bull | 0.0247 | 0.1924 | 0.1285 | 0.1515 | 0.0740 |
| fund_revenue_growth | bull | 0.0181 | 0.1905 | 0.0949 | 0.1364 | 0.0539 |
| fund_gross_margin | bear | 0.1032 | 0.2351 | 0.4389 | 0.3425 | 0.2946 |
| fund_pb | bear | 0.0911 | 0.2351 | 0.3876 | 0.3288 | 0.2575 |
| fund_pe | bear | 0.0384 | 0.1736 | 0.2213 | 0.1781 | 0.1304 |
| mom_x_lowvol_20_20 | bear | 0.0507 | 0.2376 | 0.2133 | 0.1233 | 0.1198 |
| fund_revenue_growth | bear | 0.0425 | 0.2156 | 0.1969 | 0.2055 | 0.1187 |

### 湖北自贸

- **Neutral**: ['trend_lowvol', 'mom_x_lowvol_20_20'] (单因子IC=0.0775, 组合IC=0.0912)
  - weights: [0.5008, 0.4992]
- **Bull**: ['trend_lowvol', 'low_downside'] (单因子IC=0.091, 组合IC=0.1041)
  - bull_weights: [0.5029, 0.4971]
- **Bear**: ['mom_x_lowvol_20_20', 'momentum_reversal', 'turnover_stability'] (单因子IC=0.0868, 组合IC=0.1138)
  - bear_weights: [0.3596, 0.3312, 0.3092]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| trend_lowvol | neutral | 0.0821 | 0.2709 | 0.3029 | 0.2255 | 0.1856 |
| mom_x_lowvol_20_20 | neutral | 0.0729 | 0.2404 | 0.3035 | 0.2192 | 0.1850 |
| volatility | neutral | 0.0618 | 0.2545 | 0.2427 | 0.2129 | 0.1472 |
| momentum_reversal | neutral | 0.0636 | 0.2568 | 0.2478 | 0.1712 | 0.1451 |
| fund_profit_growth | neutral | 0.0482 | 0.2190 | 0.2200 | 0.1827 | 0.1301 |
| turnover_stability | neutral | 0.0440 | 0.2096 | 0.2097 | 0.1733 | 0.1230 |
| fund_revenue_growth | neutral | 0.0447 | 0.2299 | 0.1944 | 0.2203 | 0.1186 |
| fund_score | neutral | 0.0456 | 0.2541 | 0.1792 | 0.1816 | 0.1059 |
| fund_pe | neutral | 0.0456 | 0.2928 | 0.1559 | 0.1482 | 0.0895 |
| fund_pb | neutral | 0.0470 | 0.3075 | 0.1529 | 0.1127 | 0.0851 |
| rsi_vol_combo | neutral | 0.0339 | 0.2513 | 0.1350 | 0.1106 | 0.0750 |
| trend_lowvol | bull | 0.0977 | 0.2868 | 0.3407 | 0.2576 | 0.2142 |
| low_downside | bull | 0.0843 | 0.2458 | 0.3430 | 0.2348 | 0.2118 |
| volatility | bull | 0.0774 | 0.2854 | 0.2712 | 0.1894 | 0.1613 |
| fund_pb | bull | 0.0657 | 0.2946 | 0.2229 | 0.2576 | 0.1402 |
| turnover_stability | bull | 0.0399 | 0.2027 | 0.1968 | 0.1742 | 0.1156 |
| fund_pe | bull | 0.0567 | 0.3431 | 0.1654 | 0.1515 | 0.0952 |
| stroke_phase | bull | 0.0244 | 0.2290 | 0.1065 | 0.1515 | 0.0613 |
| mom_x_lowvol_20_20 | bear | 0.1062 | 0.2915 | 0.3644 | 0.2877 | 0.2346 |
| momentum_reversal | bear | 0.0957 | 0.2669 | 0.3585 | 0.2055 | 0.2161 |
| turnover_stability | bear | 0.0584 | 0.1785 | 0.3272 | 0.2329 | 0.2017 |
| trend_lowvol | bear | 0.0658 | 0.2715 | 0.2422 | 0.2329 | 0.1493 |

### 滨海新区

- **Neutral**: ['trend_lowvol'] (单因子IC=0.0993, 组合IC=0.0993)
  - weights: [1.0]
- **Bull**: ['low_downside', 'momentum_reversal'] (单因子IC=0.0772, 组合IC=0.0965)
  - bull_weights: [0.5336, 0.4664]
- **Bear**: ['momentum_reversal'] (单因子IC=0.1967, 组合IC=0.1967)
  - bear_weights: [1.0]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| trend_lowvol | neutral | 0.0993 | 0.3119 | 0.3185 | 0.2286 | 0.1956 |
| momentum_reversal | neutral | 0.0806 | 0.2877 | 0.2803 | 0.2182 | 0.1707 |
| mom_x_lowvol_20_20 | neutral | 0.0715 | 0.2896 | 0.2469 | 0.2046 | 0.1487 |
| fund_revenue_growth | neutral | 0.0503 | 0.2186 | 0.2301 | 0.2234 | 0.1408 |
| rsi_vol_combo | neutral | 0.0525 | 0.2933 | 0.1788 | 0.1534 | 0.1031 |
| volatility | neutral | 0.0455 | 0.2879 | 0.1579 | 0.1868 | 0.0937 |
| fund_pe | neutral | 0.0468 | 0.2985 | 0.1569 | 0.1221 | 0.0880 |
| fund_pb | neutral | 0.0330 | 0.2590 | 0.1276 | 0.1065 | 0.0706 |
| turnover_stability | neutral | 0.0259 | 0.2279 | 0.1138 | 0.1273 | 0.0641 |
| low_downside | neutral | 0.0249 | 0.2832 | 0.0881 | 0.1148 | 0.0491 |
| low_downside | bull | 0.0824 | 0.2705 | 0.3048 | 0.2424 | 0.1893 |
| momentum_reversal | bull | 0.0720 | 0.2570 | 0.2801 | 0.1818 | 0.1655 |
| turnover_stability | bull | 0.0692 | 0.2429 | 0.2848 | 0.1477 | 0.1635 |
| rsi_vol_combo | bull | 0.0645 | 0.2636 | 0.2448 | 0.1932 | 0.1460 |
| fund_revenue_growth | bull | 0.0460 | 0.1912 | 0.2406 | 0.2045 | 0.1449 |
| trend_lowvol | bull | 0.0665 | 0.2909 | 0.2286 | 0.1515 | 0.1316 |
| mom_x_lowvol_20_20 | bull | 0.0475 | 0.2818 | 0.1685 | 0.1439 | 0.0964 |
| volatility | bull | 0.0430 | 0.2844 | 0.1512 | 0.1477 | 0.0867 |
| momentum_reversal | bear | 0.1967 | 0.2766 | 0.7110 | 0.5205 | 0.5406 |
| rsi_vol_combo | bear | 0.1411 | 0.2654 | 0.5315 | 0.3699 | 0.3640 |
| mom_x_lowvol_20_20 | bear | 0.1500 | 0.2992 | 0.5015 | 0.3425 | 0.3366 |
| trend_lowvol | bear | 0.1364 | 0.2900 | 0.4704 | 0.3151 | 0.3093 |
| fund_revenue_growth | bear | 0.0867 | 0.1900 | 0.4563 | 0.2877 | 0.2938 |
| fund_pb | bear | 0.0626 | 0.2376 | 0.2633 | 0.2329 | 0.1623 |
| turnover_stability | bear | 0.0471 | 0.2245 | 0.2099 | 0.2329 | 0.1294 |

### 激光雷达

- **Neutral**: ['mom_x_lowvol_20_20', 'momentum_reversal', 'rsi_vol_combo'] (单因子IC=0.0963, 组合IC=0.1039)
  - weights: [0.3596, 0.3508, 0.2896]
- **Bull**: ['fund_profit_growth', 'fund_revenue_growth', 'fund_pb'] (单因子IC=0.0727, 组合IC=0.0967)
  - bull_weights: [0.3441, 0.3356, 0.3203]
- **Bear**: ['fund_revenue_growth', 'trend_lowvol', 'momentum_reversal'] (单因子IC=0.1141, 组合IC=0.1742)
  - bear_weights: [0.3826, 0.3317, 0.2856]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| mom_x_lowvol_20_20 | neutral | 0.1016 | 0.2310 | 0.4399 | 0.3737 | 0.3021 |
| momentum_reversal | neutral | 0.1016 | 0.2324 | 0.4372 | 0.3486 | 0.2948 |
| rsi_vol_combo | neutral | 0.0858 | 0.2322 | 0.3695 | 0.3173 | 0.2434 |
| trend_lowvol | neutral | 0.0804 | 0.2499 | 0.3218 | 0.2192 | 0.1962 |
| fund_pe | neutral | 0.0660 | 0.2327 | 0.2835 | 0.2401 | 0.1758 |
| fund_score | neutral | 0.0590 | 0.2331 | 0.2529 | 0.1858 | 0.1499 |
| fund_pb | neutral | 0.0489 | 0.2085 | 0.2346 | 0.2453 | 0.1461 |
| fund_roe | neutral | 0.0559 | 0.2420 | 0.2310 | 0.1942 | 0.1379 |
| fund_profit_growth | neutral | 0.0509 | 0.2181 | 0.2332 | 0.1357 | 0.1324 |
| volatility | neutral | 0.0555 | 0.2674 | 0.2077 | 0.1670 | 0.1212 |
| fund_revenue_growth | neutral | 0.0242 | 0.1889 | 0.1280 | 0.1420 | 0.0731 |
| fund_profit_growth | bull | 0.0736 | 0.2089 | 0.3525 | 0.2273 | 0.2163 |
| fund_revenue_growth | bull | 0.0722 | 0.2113 | 0.3416 | 0.2348 | 0.2109 |
| fund_pb | bull | 0.0723 | 0.2232 | 0.3240 | 0.2424 | 0.2013 |
| fund_score | bull | 0.0676 | 0.2347 | 0.2879 | 0.1818 | 0.1701 |
| trend_lowvol | bull | 0.0678 | 0.2590 | 0.2619 | 0.1818 | 0.1547 |
| volatility | bull | 0.0552 | 0.2434 | 0.2268 | 0.1136 | 0.1263 |
| fund_roe | bull | 0.0508 | 0.2298 | 0.2211 | 0.1364 | 0.1256 |
| wash_sale_score | bull | 0.0433 | 0.1984 | 0.2182 | 0.1388 | 0.1243 |
| mom_x_lowvol_20_20 | bull | 0.0445 | 0.2365 | 0.1881 | 0.1970 | 0.1126 |
| turnover_stability | bull | 0.0265 | 0.1732 | 0.1533 | 0.2083 | 0.0926 |
| low_downside | bull | 0.0335 | 0.2152 | 0.1557 | 0.1364 | 0.0885 |
| momentum_reversal | bull | 0.0288 | 0.2344 | 0.1231 | 0.1136 | 0.0685 |
| rsi_vol_combo | bull | 0.0205 | 0.2092 | 0.0978 | 0.1742 | 0.0574 |
| fund_revenue_growth | bear | 0.1042 | 0.2038 | 0.5115 | 0.4795 | 0.3783 |
| trend_lowvol | bear | 0.1305 | 0.2508 | 0.5205 | 0.2603 | 0.3280 |
| momentum_reversal | bear | 0.1074 | 0.2345 | 0.4581 | 0.2329 | 0.2824 |
| mom_x_lowvol_20_20 | bear | 0.0970 | 0.2431 | 0.3989 | 0.3425 | 0.2678 |
| fund_roe | bear | 0.0852 | 0.2462 | 0.3460 | 0.2877 | 0.2227 |
| fund_score | bear | 0.0969 | 0.2889 | 0.3353 | 0.2329 | 0.2067 |
| top_fractal_volume | bear | 0.0596 | 0.1865 | 0.3197 | 0.2353 | 0.1975 |
| turnover_stability | bear | 0.0422 | 0.2056 | 0.2054 | 0.1781 | 0.1210 |

### 煤化工概念

- **Neutral**: ['fund_pe', 'fund_pb'] (单因子IC=0.0818, 组合IC=0.0996)
  - weights: [0.5062, 0.4938]
- **Bull**: ['volatility', 'low_downside', 'fund_pb'] (单因子IC=0.0606, 组合IC=0.0675)
  - bull_weights: [0.3714, 0.3573, 0.2713]
- **Bear**: ['bb_width_20', 'fund_profit_growth'] (单因子IC=0.1033, 组合IC=0.1323)
  - bear_weights: [0.5122, 0.4878]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| fund_pe | neutral | 0.0919 | 0.2777 | 0.3308 | 0.3215 | 0.2186 |
| fund_pb | neutral | 0.0718 | 0.2129 | 0.3371 | 0.2651 | 0.2133 |
| volatility | neutral | 0.0586 | 0.2032 | 0.2885 | 0.2380 | 0.1786 |
| fund_profit_growth | neutral | 0.0366 | 0.1783 | 0.2053 | 0.2109 | 0.1243 |
| mom_x_lowvol_20_20 | neutral | 0.0403 | 0.1974 | 0.2041 | 0.1691 | 0.1193 |
| fund_score | neutral | 0.0458 | 0.2224 | 0.2060 | 0.1378 | 0.1172 |
| momentum_reversal | neutral | 0.0437 | 0.2229 | 0.1959 | 0.1441 | 0.1120 |
| fund_gross_margin | neutral | 0.0373 | 0.1946 | 0.1916 | 0.1294 | 0.1082 |
| trend_lowvol | neutral | 0.0466 | 0.2501 | 0.1865 | 0.1503 | 0.1073 |
| turnover_stability | neutral | 0.0287 | 0.1599 | 0.1793 | 0.1357 | 0.1018 |
| fund_roe | neutral | 0.0408 | 0.2484 | 0.1641 | 0.1294 | 0.0927 |
| fund_revenue_growth | neutral | 0.0259 | 0.1663 | 0.1559 | 0.1002 | 0.0857 |
| low_downside | neutral | 0.0219 | 0.1943 | 0.1129 | 0.1420 | 0.0645 |
| volatility | bull | 0.0600 | 0.1808 | 0.3318 | 0.3258 | 0.2199 |
| low_downside | bull | 0.0636 | 0.1912 | 0.3325 | 0.2727 | 0.2116 |
| fund_pb | bull | 0.0582 | 0.2018 | 0.2886 | 0.1136 | 0.1607 |
| fund_pe | bull | 0.0573 | 0.2368 | 0.2422 | 0.2576 | 0.1523 |
| vol_confirm | bull | 0.0272 | 0.1759 | 0.1547 | 0.1212 | 0.0867 |
| fund_revenue_growth | bull | 0.0204 | 0.1753 | 0.1162 | 0.1515 | 0.0669 |
| fund_roe | bull | 0.0266 | 0.2706 | 0.0982 | 0.1818 | 0.0580 |
| bb_width_20 | bear | 0.1108 | 0.2151 | 0.5150 | 0.3699 | 0.3527 |
| fund_profit_growth | bear | 0.0958 | 0.1992 | 0.4809 | 0.3973 | 0.3359 |
| fund_score | bear | 0.0979 | 0.2313 | 0.4234 | 0.3151 | 0.2784 |
| top_fractal_volume | bear | 0.0453 | 0.1335 | 0.3392 | 0.3409 | 0.2274 |
| momentum_reversal | bear | 0.0931 | 0.2600 | 0.3583 | 0.2329 | 0.2208 |
| fund_gross_margin | bear | 0.0595 | 0.1959 | 0.3034 | 0.1781 | 0.1787 |
| turnover_stability | bear | 0.0559 | 0.2062 | 0.2710 | 0.3151 | 0.1782 |
| mom_x_lowvol_20_20 | bear | 0.0704 | 0.2623 | 0.2685 | 0.1644 | 0.1563 |
| fund_revenue_growth | bear | 0.0355 | 0.1440 | 0.2466 | 0.1781 | 0.1452 |
| fund_roe | bear | 0.0649 | 0.2808 | 0.2312 | 0.2055 | 0.1393 |

### 燃料电池概念

- **Neutral**: ['momentum_reversal', 'fund_profit_growth', 'mom_x_lowvol_20_20'] (单因子IC=0.0716, 组合IC=0.095)
  - weights: [0.366, 0.3234, 0.3106]
- **Bull**: ['fund_pb', 'low_downside', 'volatility'] (单因子IC=0.0614, 组合IC=0.0695)
  - bull_weights: [0.3483, 0.3354, 0.3163]
- **Bear**: ['momentum_reversal'] (单因子IC=0.0919, 组合IC=0.0919)
  - bear_weights: [1.0]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| momentum_reversal | neutral | 0.0827 | 0.1591 | 0.5199 | 0.4008 | 0.3641 |
| fund_profit_growth | neutral | 0.0620 | 0.1298 | 0.4778 | 0.3466 | 0.3217 |
| mom_x_lowvol_20_20 | neutral | 0.0700 | 0.1529 | 0.4576 | 0.3507 | 0.3091 |
| rsi_vol_combo | neutral | 0.0662 | 0.1512 | 0.4376 | 0.3340 | 0.2919 |
| trend_lowvol | neutral | 0.0817 | 0.1888 | 0.4325 | 0.3299 | 0.2876 |
| volatility | neutral | 0.0745 | 0.1943 | 0.3834 | 0.3132 | 0.2518 |
| fund_pb | neutral | 0.0597 | 0.1720 | 0.3471 | 0.2359 | 0.2145 |
| fund_score | neutral | 0.0566 | 0.1637 | 0.3455 | 0.2276 | 0.2121 |
| fund_pe | neutral | 0.0589 | 0.1738 | 0.3386 | 0.2463 | 0.2110 |
| fund_revenue_growth | neutral | 0.0351 | 0.1311 | 0.2681 | 0.1962 | 0.1604 |
| fund_roe | neutral | 0.0436 | 0.1753 | 0.2487 | 0.1503 | 0.1431 |
| turnover_stability | neutral | 0.0269 | 0.1165 | 0.2309 | 0.2067 | 0.1393 |
| low_downside | neutral | 0.0359 | 0.1864 | 0.1927 | 0.1148 | 0.1074 |
| fund_pb | bull | 0.0625 | 0.1656 | 0.3776 | 0.3106 | 0.2475 |
| low_downside | bull | 0.0546 | 0.1537 | 0.3554 | 0.3409 | 0.2383 |
| volatility | bull | 0.0671 | 0.1924 | 0.3491 | 0.2879 | 0.2248 |
| turnover_stability | bull | 0.0344 | 0.1069 | 0.3220 | 0.2727 | 0.2049 |
| trend_lowvol | bull | 0.0572 | 0.1899 | 0.3010 | 0.2803 | 0.1927 |
| fund_pe | bull | 0.0451 | 0.1690 | 0.2666 | 0.2197 | 0.1626 |
| fund_profit_growth | bull | 0.0266 | 0.1357 | 0.1962 | 0.2424 | 0.1219 |
| fund_revenue_growth | bull | 0.0225 | 0.1351 | 0.1663 | 0.1818 | 0.0983 |
| fund_score | bull | 0.0299 | 0.1869 | 0.1600 | 0.2045 | 0.0964 |
| fund_roe | bull | 0.0196 | 0.2002 | 0.0977 | 0.1136 | 0.0544 |
| momentum_reversal | bear | 0.0919 | 0.1770 | 0.5191 | 0.3425 | 0.3484 |
| rsi_vol_combo | bear | 0.0781 | 0.1532 | 0.5099 | 0.3151 | 0.3352 |
| mom_x_lowvol_20_20 | bear | 0.0781 | 0.1756 | 0.4445 | 0.4247 | 0.3167 |
| trend_lowvol | bear | 0.0730 | 0.2014 | 0.3626 | 0.2055 | 0.2186 |
| fund_pe | bear | 0.0561 | 0.1838 | 0.3050 | 0.2055 | 0.1838 |
| vol_confirm | bear | 0.0357 | 0.1235 | 0.2890 | 0.2329 | 0.1782 |
| fund_revenue_growth | bear | 0.0397 | 0.1403 | 0.2830 | 0.2329 | 0.1745 |

### 物联网

- **Neutral**: ['mom_x_lowvol_20_20', 'momentum_reversal', 'trend_lowvol'] (单因子IC=0.0826, 组合IC=0.091)
  - weights: [0.3408, 0.3401, 0.3191]
- **Bull**: ['volatility', 'low_downside', 'fund_pb'] (单因子IC=0.0998, 组合IC=0.1206)
  - bull_weights: [0.3755, 0.3276, 0.2969]
- **Bear**: ['mom_x_lowvol_20_20', 'momentum_reversal', 'trend_lowvol'] (单因子IC=0.101, 组合IC=0.1116)
  - bear_weights: [0.3584, 0.3252, 0.3165]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| mom_x_lowvol_20_20 | neutral | 0.0807 | 0.1523 | 0.5299 | 0.3737 | 0.3640 |
| momentum_reversal | neutral | 0.0827 | 0.1567 | 0.5281 | 0.3758 | 0.3633 |
| trend_lowvol | neutral | 0.0845 | 0.1694 | 0.4985 | 0.3674 | 0.3408 |
| fund_pb | neutral | 0.0719 | 0.1424 | 0.5050 | 0.2923 | 0.3263 |
| rsi_vol_combo | neutral | 0.0655 | 0.1455 | 0.4500 | 0.3424 | 0.3021 |
| volatility | neutral | 0.0703 | 0.1723 | 0.4080 | 0.3486 | 0.2751 |
| turnover_stability | neutral | 0.0345 | 0.0861 | 0.4004 | 0.3591 | 0.2721 |
| fund_profit_growth | neutral | 0.0454 | 0.1188 | 0.3819 | 0.2672 | 0.2420 |
| fund_pe | neutral | 0.0450 | 0.1535 | 0.2929 | 0.2067 | 0.1767 |
| fund_score | neutral | 0.0395 | 0.1526 | 0.2588 | 0.1691 | 0.1513 |
| low_downside | neutral | 0.0380 | 0.1652 | 0.2302 | 0.2192 | 0.1404 |
| fund_revenue_growth | neutral | 0.0200 | 0.1126 | 0.1773 | 0.1524 | 0.1022 |
| volatility | bull | 0.1107 | 0.1520 | 0.7278 | 0.5152 | 0.5514 |
| low_downside | bull | 0.0922 | 0.1386 | 0.6648 | 0.4470 | 0.4810 |
| fund_pb | bull | 0.0964 | 0.1576 | 0.6121 | 0.4242 | 0.4359 |
| turnover_stability | bull | 0.0441 | 0.0797 | 0.5527 | 0.5076 | 0.4166 |
| trend_lowvol | bull | 0.0725 | 0.1647 | 0.4402 | 0.3485 | 0.2968 |
| fund_pe | bull | 0.0565 | 0.1476 | 0.3827 | 0.3106 | 0.2508 |
| mom_x_lowvol_20_20 | bull | 0.0487 | 0.1451 | 0.3358 | 0.2652 | 0.2124 |
| momentum_reversal | bull | 0.0421 | 0.1457 | 0.2887 | 0.2045 | 0.1739 |
| fund_profit_growth | bull | 0.0302 | 0.1181 | 0.2559 | 0.2045 | 0.1541 |
| stroke_phase | bull | 0.0239 | 0.0941 | 0.2535 | 0.2121 | 0.1536 |
| mom_x_lowvol_20_20 | bear | 0.1052 | 0.1677 | 0.6273 | 0.4521 | 0.4554 |
| momentum_reversal | bear | 0.1023 | 0.1763 | 0.5802 | 0.4247 | 0.4133 |
| trend_lowvol | bear | 0.0955 | 0.1692 | 0.5646 | 0.4247 | 0.4022 |
| rsi_vol_combo | bear | 0.0659 | 0.1463 | 0.4502 | 0.3973 | 0.3145 |
| fund_revenue_growth | bear | 0.0333 | 0.1306 | 0.2549 | 0.1233 | 0.1432 |

### 特斯拉概念

- **Neutral**: ['momentum_reversal', 'fund_pb'] (单因子IC=0.07, 组合IC=0.0975)
  - weights: [0.5031, 0.4969]
- **Bull**: ['volatility', 'fund_pb', 'trend_lowvol'] (单因子IC=0.0759, 组合IC=0.1001)
  - bull_weights: [0.4011, 0.3146, 0.2843]
- **Bear**: ['momentum_reversal', 'mom_x_lowvol_20_20', 'fund_pe'] (单因子IC=0.0743, 组合IC=0.1155)
  - bear_weights: [0.3596, 0.3313, 0.3091]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| momentum_reversal | neutral | 0.0697 | 0.1646 | 0.4238 | 0.3403 | 0.2840 |
| fund_pb | neutral | 0.0702 | 0.1668 | 0.4212 | 0.3319 | 0.2805 |
| mom_x_lowvol_20_20 | neutral | 0.0648 | 0.1551 | 0.4180 | 0.3048 | 0.2727 |
| trend_lowvol | neutral | 0.0740 | 0.1865 | 0.3969 | 0.2630 | 0.2506 |
| fund_pe | neutral | 0.0575 | 0.1556 | 0.3697 | 0.2735 | 0.2354 |
| fund_profit_growth | neutral | 0.0486 | 0.1322 | 0.3673 | 0.2797 | 0.2350 |
| rsi_vol_combo | neutral | 0.0495 | 0.1600 | 0.3096 | 0.2443 | 0.1926 |
| volatility | neutral | 0.0540 | 0.1856 | 0.2912 | 0.2296 | 0.1790 |
| fund_score | neutral | 0.0457 | 0.1646 | 0.2776 | 0.1983 | 0.1663 |
| fund_revenue_growth | neutral | 0.0255 | 0.1272 | 0.2002 | 0.1628 | 0.1164 |
| turnover_stability | neutral | 0.0223 | 0.1165 | 0.1916 | 0.1942 | 0.1144 |
| low_downside | neutral | 0.0285 | 0.1777 | 0.1603 | 0.1482 | 0.0920 |
| volatility | bull | 0.0792 | 0.1540 | 0.5142 | 0.3864 | 0.3564 |
| fund_pb | bull | 0.0813 | 0.1840 | 0.4419 | 0.2652 | 0.2795 |
| trend_lowvol | bull | 0.0671 | 0.1721 | 0.3901 | 0.2955 | 0.2527 |
| low_downside | bull | 0.0523 | 0.1434 | 0.3643 | 0.3864 | 0.2525 |
| fund_pe | bull | 0.0449 | 0.1351 | 0.3326 | 0.2273 | 0.2041 |
| mom_x_lowvol_20_20 | bull | 0.0373 | 0.1515 | 0.2465 | 0.1439 | 0.1410 |
| momentum_reversal | bull | 0.0355 | 0.1620 | 0.2191 | 0.1667 | 0.1278 |
| turnover_stability | bull | 0.0196 | 0.0954 | 0.2058 | 0.1288 | 0.1162 |
| fund_score | bull | 0.0291 | 0.1723 | 0.1689 | 0.1818 | 0.0998 |
| stroke_phase | bull | 0.0109 | 0.1023 | 0.1069 | 0.1894 | 0.0636 |
| momentum_reversal | bear | 0.0875 | 0.2048 | 0.4274 | 0.3151 | 0.2810 |
| mom_x_lowvol_20_20 | bear | 0.0799 | 0.2156 | 0.3706 | 0.3973 | 0.2589 |
| fund_pe | bear | 0.0555 | 0.1573 | 0.3526 | 0.3699 | 0.2415 |
| fund_profit_growth | bear | 0.0592 | 0.1596 | 0.3712 | 0.1507 | 0.2136 |
| fund_pb | bear | 0.0535 | 0.1753 | 0.3051 | 0.1233 | 0.1714 |
| bb_width_20 | bear | 0.0577 | 0.2272 | 0.2541 | 0.2877 | 0.1636 |
| rsi_vol_combo | bear | 0.0436 | 0.1696 | 0.2569 | 0.1507 | 0.1478 |
| fund_gross_margin | bear | 0.0268 | 0.1075 | 0.2495 | 0.1781 | 0.1469 |
| trend_lowvol | bear | 0.0496 | 0.2439 | 0.2035 | 0.2329 | 0.1254 |

### 特色药

- **Neutral**: ['fund_pe', 'fund_pb'] (单因子IC=0.0808, 组合IC=0.1)
  - weights: [0.5604, 0.4396]
- **Bull**: ['fund_pe', 'low_downside', 'fund_pb'] (单因子IC=0.0687, 组合IC=0.1005)
  - bull_weights: [0.4401, 0.3051, 0.2548]
- **Bear**: ['turnover_stability', 'fund_pe'] (单因子IC=0.1673, 组合IC=0.2324)
  - bear_weights: [0.5296, 0.4704]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| fund_pe | neutral | 0.0952 | 0.2158 | 0.4412 | 0.3737 | 0.3030 |
| fund_pb | neutral | 0.0664 | 0.1840 | 0.3609 | 0.3173 | 0.2377 |
| fund_profit_growth | neutral | 0.0598 | 0.2050 | 0.2917 | 0.2276 | 0.1791 |
| low_downside | neutral | 0.0604 | 0.2316 | 0.2608 | 0.2223 | 0.1594 |
| trend_lowvol | neutral | 0.0666 | 0.2559 | 0.2602 | 0.2213 | 0.1589 |
| volatility | neutral | 0.0449 | 0.2063 | 0.2178 | 0.2004 | 0.1307 |
| fund_score | neutral | 0.0454 | 0.2201 | 0.2063 | 0.2349 | 0.1274 |
| momentum_reversal | neutral | 0.0493 | 0.2356 | 0.2092 | 0.1545 | 0.1207 |
| mom_x_lowvol_20_20 | neutral | 0.0436 | 0.2139 | 0.2040 | 0.1754 | 0.1199 |
| fund_revenue_growth | neutral | 0.0359 | 0.1736 | 0.2069 | 0.1461 | 0.1186 |
| turnover_stability | neutral | 0.0308 | 0.1691 | 0.1819 | 0.1900 | 0.1082 |
| fund_pe | bull | 0.0907 | 0.2183 | 0.4153 | 0.2727 | 0.2643 |
| low_downside | bull | 0.0657 | 0.2132 | 0.3081 | 0.1894 | 0.1832 |
| fund_pb | bull | 0.0498 | 0.1909 | 0.2606 | 0.1742 | 0.1530 |
| fund_profit_growth | bull | 0.0483 | 0.1875 | 0.2576 | 0.1515 | 0.1483 |
| fund_roe | bull | 0.0444 | 0.2485 | 0.1787 | 0.2424 | 0.1110 |
| rsi_vol_combo | bull | 0.0311 | 0.2145 | 0.1448 | 0.1667 | 0.0845 |
| trend_lowvol | bull | 0.0323 | 0.2460 | 0.1314 | 0.1061 | 0.0727 |
| turnover_stability | bull | 0.0184 | 0.1652 | 0.1115 | 0.1742 | 0.0654 |
| fund_gross_margin | bull | 0.0211 | 0.2083 | 0.1013 | 0.2045 | 0.0610 |
| turnover_stability | bear | 0.1311 | 0.1472 | 0.8902 | 0.5890 | 0.7073 |
| fund_pe | bear | 0.2035 | 0.2618 | 0.7774 | 0.6164 | 0.6283 |
| fund_pb | bear | 0.1010 | 0.1545 | 0.6539 | 0.5342 | 0.5016 |
| trend_lowvol | bear | 0.1650 | 0.2693 | 0.6127 | 0.4521 | 0.4448 |
| fund_score | bear | 0.1374 | 0.2372 | 0.5793 | 0.4795 | 0.4285 |
| fund_profit_growth | bear | 0.1191 | 0.2152 | 0.5537 | 0.3699 | 0.3792 |
| fund_roe | bear | 0.1447 | 0.3008 | 0.4809 | 0.3699 | 0.3294 |
| momentum_reversal | bear | 0.1114 | 0.2854 | 0.3905 | 0.2603 | 0.2461 |
| fund_revenue_growth | bear | 0.0519 | 0.1555 | 0.3337 | 0.2055 | 0.2011 |
| mom_x_lowvol_20_20 | bear | 0.0748 | 0.2261 | 0.3309 | 0.2055 | 0.1994 |
| volatility | bear | 0.0609 | 0.2173 | 0.2801 | 0.2329 | 0.1726 |
| rsi_vol_combo | bear | 0.0619 | 0.2743 | 0.2258 | 0.3151 | 0.1485 |
| fund_gross_margin | bear | 0.0329 | 0.1778 | 0.1853 | 0.3699 | 0.1269 |

### 特高压

- **Neutral**: ['mom_x_lowvol_20_20', 'fund_profit_growth', 'momentum_reversal'] (单因子IC=0.0678, 组合IC=0.0855)
  - weights: [0.3382, 0.3326, 0.3292]
- **Bull**: ['volatility', 'low_downside', 'fund_revenue_growth'] (单因子IC=0.0499, 组合IC=0.0646)
  - bull_weights: [0.3766, 0.3384, 0.285]
- **Bear**: ['mom_x_lowvol_20_20', 'momentum_reversal', 'fund_pb'] (单因子IC=0.0827, 组合IC=0.1185)
  - bear_weights: [0.4031, 0.318, 0.2788]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| mom_x_lowvol_20_20 | neutral | 0.0715 | 0.1919 | 0.3727 | 0.3236 | 0.2466 |
| fund_profit_growth | neutral | 0.0550 | 0.1465 | 0.3753 | 0.2923 | 0.2425 |
| momentum_reversal | neutral | 0.0769 | 0.2071 | 0.3715 | 0.2923 | 0.2401 |
| fund_pb | neutral | 0.0554 | 0.1643 | 0.3371 | 0.2401 | 0.2090 |
| rsi_vol_combo | neutral | 0.0603 | 0.1949 | 0.3095 | 0.2756 | 0.1974 |
| trend_lowvol | neutral | 0.0695 | 0.2172 | 0.3198 | 0.2286 | 0.1964 |
| fund_pe | neutral | 0.0553 | 0.1901 | 0.2910 | 0.2276 | 0.1786 |
| volatility | neutral | 0.0600 | 0.2071 | 0.2899 | 0.2213 | 0.1770 |
| fund_score | neutral | 0.0540 | 0.1832 | 0.2947 | 0.1983 | 0.1766 |
| fund_revenue_growth | neutral | 0.0361 | 0.1345 | 0.2684 | 0.2213 | 0.1639 |
| turnover_stability | neutral | 0.0329 | 0.1384 | 0.2375 | 0.2025 | 0.1428 |
| fund_roe | neutral | 0.0347 | 0.2023 | 0.1716 | 0.1357 | 0.0975 |
| low_downside | neutral | 0.0225 | 0.2075 | 0.1082 | 0.1190 | 0.0606 |
| volatility | bull | 0.0612 | 0.1868 | 0.3277 | 0.2879 | 0.2110 |
| low_downside | bull | 0.0508 | 0.1645 | 0.3090 | 0.2273 | 0.1896 |
| fund_revenue_growth | bull | 0.0375 | 0.1495 | 0.2510 | 0.2727 | 0.1597 |
| trend_lowvol | bull | 0.0396 | 0.2154 | 0.1840 | 0.2652 | 0.1164 |
| fund_pb | bull | 0.0229 | 0.1814 | 0.1262 | 0.1742 | 0.0741 |
| mom_x_lowvol_20_20 | bull | 0.0204 | 0.1826 | 0.1118 | 0.1288 | 0.0631 |
| mom_x_lowvol_20_20 | bear | 0.0926 | 0.2103 | 0.4402 | 0.3425 | 0.2955 |
| momentum_reversal | bear | 0.0754 | 0.2083 | 0.3621 | 0.2877 | 0.2331 |
| fund_pb | bear | 0.0802 | 0.2310 | 0.3470 | 0.1781 | 0.2044 |
| rsi_vol_combo | bear | 0.0390 | 0.1674 | 0.2328 | 0.2603 | 0.1467 |
| fund_profit_growth | bear | 0.0453 | 0.1742 | 0.2604 | 0.1233 | 0.1462 |
| fund_pe | bear | 0.0363 | 0.1984 | 0.1830 | 0.1781 | 0.1078 |

### 独家药品

- **Neutral**: ['fund_pe', 'fund_pb', 'fund_profit_growth'] (单因子IC=0.0737, 组合IC=0.0987)
  - weights: [0.3598, 0.3338, 0.3063]
- **Bull**: ['low_downside', 'rsi_vol_combo'] (单因子IC=0.064, 组合IC=0.0996)
  - bull_weights: [0.5856, 0.4144]
- **Bear**: ['fund_profit_growth', 'fund_pe'] (单因子IC=0.1327, 组合IC=0.1915)
  - bear_weights: [0.512, 0.488]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| fund_pe | neutral | 0.0793 | 0.1943 | 0.4081 | 0.3236 | 0.2701 |
| fund_pb | neutral | 0.0740 | 0.1905 | 0.3883 | 0.2902 | 0.2505 |
| fund_profit_growth | neutral | 0.0678 | 0.1881 | 0.3608 | 0.2745 | 0.2299 |
| volatility | neutral | 0.0666 | 0.2062 | 0.3230 | 0.3278 | 0.2144 |
| low_downside | neutral | 0.0750 | 0.2315 | 0.3237 | 0.2704 | 0.2056 |
| fund_revenue_growth | neutral | 0.0425 | 0.1786 | 0.2380 | 0.2234 | 0.1456 |
| fund_score | neutral | 0.0417 | 0.1996 | 0.2090 | 0.1691 | 0.1222 |
| turnover_stability | neutral | 0.0292 | 0.1585 | 0.1844 | 0.1712 | 0.1080 |
| trend_lowvol | neutral | 0.0474 | 0.2605 | 0.1818 | 0.1733 | 0.1067 |
| low_downside | bull | 0.0777 | 0.2058 | 0.3773 | 0.2500 | 0.2358 |
| rsi_vol_combo | bull | 0.0503 | 0.1928 | 0.2607 | 0.2803 | 0.1669 |
| fund_pb | bull | 0.0552 | 0.2060 | 0.2678 | 0.2348 | 0.1653 |
| fund_pe | bull | 0.0602 | 0.2227 | 0.2702 | 0.2121 | 0.1638 |
| turnover_stability | bull | 0.0414 | 0.1591 | 0.2600 | 0.1742 | 0.1527 |
| fund_profit_growth | bull | 0.0490 | 0.2133 | 0.2296 | 0.1136 | 0.1278 |
| momentum_reversal | bull | 0.0422 | 0.2040 | 0.2070 | 0.1439 | 0.1184 |
| trend_lowvol | bull | 0.0493 | 0.2467 | 0.1997 | 0.1591 | 0.1157 |
| volatility | bull | 0.0327 | 0.1837 | 0.1778 | 0.1894 | 0.1058 |
| fund_roe | bull | 0.0365 | 0.2130 | 0.1714 | 0.2273 | 0.1052 |
| fund_profit_growth | bear | 0.1285 | 0.1858 | 0.6915 | 0.4247 | 0.4926 |
| fund_pe | bear | 0.1370 | 0.2279 | 0.6012 | 0.5616 | 0.4694 |
| fund_score | bear | 0.1146 | 0.2111 | 0.5430 | 0.5068 | 0.4091 |
| fund_gross_margin | bear | 0.0925 | 0.1810 | 0.5111 | 0.4521 | 0.3711 |
| turnover_stability | bear | 0.0692 | 0.1421 | 0.4873 | 0.4521 | 0.3538 |
| fund_roe | bear | 0.1035 | 0.2392 | 0.4327 | 0.3973 | 0.3023 |
| fund_revenue_growth | bear | 0.0625 | 0.1498 | 0.4174 | 0.3425 | 0.2802 |
| trend_lowvol | bear | 0.1044 | 0.2689 | 0.3882 | 0.3973 | 0.2712 |
| bb_width_20 | bear | 0.1090 | 0.2960 | 0.3682 | 0.2603 | 0.2320 |
| momentum_reversal | bear | 0.0919 | 0.2744 | 0.3349 | 0.2329 | 0.2064 |
| fund_pb | bear | 0.0419 | 0.1656 | 0.2533 | 0.3151 | 0.1666 |
| mom_x_lowvol_20_20 | bear | 0.0493 | 0.2322 | 0.2124 | 0.2329 | 0.1309 |

### 独角兽

- **Neutral**: ['trend_lowvol', 'fund_profit_growth', 'fund_pb'] (单因子IC=0.0574, 组合IC=0.078)
  - weights: [0.3432, 0.3413, 0.3155]
- **Bull**: ['low_downside', 'turnover_stability', 'fund_pb'] (单因子IC=0.0596, 组合IC=0.0979)
  - bull_weights: [0.3598, 0.3553, 0.2849]
- **Bear**: ['mom_x_lowvol_20_20', 'momentum_reversal', 'trend_lowvol'] (单因子IC=0.0893, 组合IC=0.0988)
  - bear_weights: [0.4256, 0.307, 0.2674]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| trend_lowvol | neutral | 0.0692 | 0.2106 | 0.3287 | 0.2255 | 0.2014 |
| fund_profit_growth | neutral | 0.0470 | 0.1477 | 0.3181 | 0.2589 | 0.2003 |
| fund_pb | neutral | 0.0561 | 0.1846 | 0.3037 | 0.2192 | 0.1851 |
| mom_x_lowvol_20_20 | neutral | 0.0539 | 0.1820 | 0.2961 | 0.2359 | 0.1830 |
| momentum_reversal | neutral | 0.0552 | 0.1952 | 0.2830 | 0.2317 | 0.1743 |
| fund_pe | neutral | 0.0492 | 0.1884 | 0.2609 | 0.1942 | 0.1558 |
| turnover_stability | neutral | 0.0294 | 0.1181 | 0.2490 | 0.1378 | 0.1417 |
| rsi_vol_combo | neutral | 0.0420 | 0.1778 | 0.2363 | 0.1733 | 0.1386 |
| volatility | neutral | 0.0446 | 0.1950 | 0.2286 | 0.1900 | 0.1360 |
| fund_score | neutral | 0.0427 | 0.1864 | 0.2289 | 0.1858 | 0.1357 |
| low_downside | neutral | 0.0371 | 0.2018 | 0.1837 | 0.1545 | 0.1060 |
| fund_revenue_growth | neutral | 0.0145 | 0.1511 | 0.0961 | 0.1002 | 0.0529 |
| low_downside | bull | 0.0707 | 0.1767 | 0.4004 | 0.2652 | 0.2533 |
| turnover_stability | bull | 0.0432 | 0.1126 | 0.3839 | 0.3030 | 0.2501 |
| fund_pb | bull | 0.0647 | 0.2005 | 0.3228 | 0.2424 | 0.2006 |
| momentum_reversal | bull | 0.0518 | 0.1689 | 0.3066 | 0.2576 | 0.1928 |
| trend_lowvol | bull | 0.0628 | 0.1952 | 0.3220 | 0.1667 | 0.1878 |
| volatility | bull | 0.0551 | 0.1836 | 0.3002 | 0.1515 | 0.1728 |
| fund_profit_growth | bull | 0.0350 | 0.1467 | 0.2387 | 0.2424 | 0.1483 |
| fund_pe | bull | 0.0395 | 0.1777 | 0.2223 | 0.1970 | 0.1330 |
| stroke_phase | bull | 0.0247 | 0.1202 | 0.2056 | 0.1818 | 0.1215 |
| fund_revenue_growth | bull | 0.0263 | 0.1351 | 0.1944 | 0.1515 | 0.1119 |
| rsi_vol_combo | bull | 0.0281 | 0.1543 | 0.1822 | 0.1288 | 0.1029 |
| fund_score | bull | 0.0274 | 0.1694 | 0.1618 | 0.1136 | 0.0901 |
| mom_x_lowvol_20_20 | bull | 0.0223 | 0.1692 | 0.1319 | 0.1439 | 0.0754 |
| mom_x_lowvol_20_20 | bear | 0.0979 | 0.2082 | 0.4703 | 0.3699 | 0.3221 |
| momentum_reversal | bear | 0.0828 | 0.2148 | 0.3856 | 0.2055 | 0.2324 |
| trend_lowvol | bear | 0.0870 | 0.2532 | 0.3437 | 0.1781 | 0.2024 |
| bb_width_20 | bear | 0.0517 | 0.2521 | 0.2052 | 0.3973 | 0.1433 |
| fund_revenue_growth | bear | 0.0317 | 0.1513 | 0.2093 | 0.1233 | 0.1175 |

### 猪肉概念

- **Neutral**: ['fund_pb', 'trend_lowvol'] (单因子IC=0.0504, 组合IC=0.061)
  - weights: [0.5856, 0.4144]
- **Bull**: ['fund_pb', 'turnover_stability'] (单因子IC=0.0617, 组合IC=0.0851)
  - bull_weights: [0.5762, 0.4238]
- **Bear**: ['fund_pe', 'fund_pb'] (单因子IC=0.1229, 组合IC=0.2185)
  - bear_weights: [0.5032, 0.4968]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| fund_pb | neutral | 0.0515 | 0.2089 | 0.2465 | 0.1743 | 0.1448 |
| trend_lowvol | neutral | 0.0493 | 0.2822 | 0.1748 | 0.1722 | 0.1024 |
| volatility | neutral | 0.0404 | 0.2313 | 0.1747 | 0.1430 | 0.0998 |
| mom_x_lowvol_20_20 | neutral | 0.0398 | 0.2331 | 0.1709 | 0.1190 | 0.0956 |
| fund_pe | neutral | 0.0369 | 0.2605 | 0.1418 | 0.1472 | 0.0813 |
| turnover_stability | neutral | 0.0264 | 0.2085 | 0.1267 | 0.1566 | 0.0733 |
| fund_pb | bull | 0.0697 | 0.2192 | 0.3181 | 0.3144 | 0.2090 |
| turnover_stability | bull | 0.0537 | 0.2132 | 0.2521 | 0.2197 | 0.1538 |
| volatility | bull | 0.0491 | 0.2191 | 0.2243 | 0.1818 | 0.1325 |
| mom_x_lowvol_20_20 | bull | 0.0382 | 0.2317 | 0.1649 | 0.1288 | 0.0930 |
| momentum_reversal | bull | 0.0378 | 0.2477 | 0.1524 | 0.1364 | 0.0866 |
| rsi_vol_combo | bull | 0.0293 | 0.2271 | 0.1291 | 0.1288 | 0.0729 |
| trend_lowvol | bull | 0.0295 | 0.2495 | 0.1184 | 0.1212 | 0.0664 |
| fund_pe | bear | 0.1140 | 0.2546 | 0.4476 | 0.3699 | 0.3066 |
| fund_pb | bear | 0.1319 | 0.2865 | 0.4604 | 0.3151 | 0.3027 |
| fund_gross_margin | bear | 0.0834 | 0.2907 | 0.2870 | 0.1507 | 0.1651 |
| fund_profit_growth | bear | 0.0523 | 0.2059 | 0.2538 | 0.2603 | 0.1599 |
| momentum_reversal | bear | 0.0731 | 0.2909 | 0.2514 | 0.2329 | 0.1550 |
| mom_x_lowvol_20_20 | bear | 0.0463 | 0.2557 | 0.1811 | 0.1507 | 0.1042 |

### 环氧丙烷

- **Neutral**: ['fund_pe', 'fund_pb'] (单因子IC=0.0962, 组合IC=0.1038)
  - weights: [0.5004, 0.4996]
- **Bull**: ['fund_profit_growth', 'fund_pb'] (单因子IC=0.1042, 组合IC=0.1464)
  - bull_weights: [0.5469, 0.4531]
- **Bear**: ['mom_x_lowvol_20_20'] (单因子IC=0.1023, 组合IC=0.1023)
  - bear_weights: [1.0]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| fund_pe | neutral | 0.0943 | 0.3026 | 0.3116 | 0.3100 | 0.2041 |
| fund_pb | neutral | 0.0982 | 0.2997 | 0.3276 | 0.2443 | 0.2038 |
| volatility | neutral | 0.0651 | 0.2813 | 0.2315 | 0.1816 | 0.1367 |
| fund_profit_growth | neutral | 0.0538 | 0.2513 | 0.2141 | 0.2056 | 0.1291 |
| fund_score | neutral | 0.0433 | 0.2791 | 0.1553 | 0.1503 | 0.0893 |
| low_downside | neutral | 0.0336 | 0.3128 | 0.1073 | 0.1691 | 0.0627 |
| fund_roe | neutral | 0.0303 | 0.2769 | 0.1093 | 0.1096 | 0.0607 |
| fund_profit_growth | bull | 0.1110 | 0.2561 | 0.4333 | 0.3030 | 0.2823 |
| fund_pb | bull | 0.0975 | 0.2628 | 0.3708 | 0.2614 | 0.2339 |
| volatility | bull | 0.0748 | 0.2455 | 0.3048 | 0.1629 | 0.1772 |
| fund_pe | bull | 0.0864 | 0.2917 | 0.2964 | 0.1477 | 0.1701 |
| fund_score | bull | 0.0722 | 0.2540 | 0.2841 | 0.1856 | 0.1684 |
| low_downside | bull | 0.0672 | 0.2605 | 0.2582 | 0.1439 | 0.1477 |
| rsi_vol_combo | bull | 0.0701 | 0.2722 | 0.2576 | 0.1326 | 0.1459 |
| fund_roe | bull | 0.0573 | 0.2684 | 0.2133 | 0.1629 | 0.1240 |
| momentum_reversal | bull | 0.0572 | 0.2842 | 0.2013 | 0.1061 | 0.1113 |
| mom_x_lowvol_20_20 | bear | 0.1023 | 0.2531 | 0.4040 | 0.4521 | 0.2933 |
| momentum_reversal | bear | 0.0676 | 0.2598 | 0.2603 | 0.3151 | 0.1712 |
| rsi_vol_combo | bear | 0.0605 | 0.2659 | 0.2274 | 0.2603 | 0.1433 |

### 玻璃基板

- **Neutral**: ['fund_pb', 'momentum_reversal'] (单因子IC=0.0803, 组合IC=0.1085)
  - weights: [0.5007, 0.4993]
- **Bull**: ['trend_lowvol', 'low_downside'] (单因子IC=0.0838, 组合IC=0.1019)
  - bull_weights: [0.5338, 0.4662]
- **Bear**: ['momentum_reversal'] (单因子IC=0.1151, 组合IC=0.1151)
  - bear_weights: [1.0]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| fund_pb | neutral | 0.0805 | 0.2445 | 0.3291 | 0.3017 | 0.2142 |
| momentum_reversal | neutral | 0.0802 | 0.2424 | 0.3308 | 0.2912 | 0.2136 |
| mom_x_lowvol_20_20 | neutral | 0.0781 | 0.2373 | 0.3293 | 0.2599 | 0.2074 |
| trend_lowvol | neutral | 0.0747 | 0.2409 | 0.3101 | 0.2651 | 0.1961 |
| volatility | neutral | 0.0683 | 0.2485 | 0.2747 | 0.1910 | 0.1636 |
| rsi_vol_combo | neutral | 0.0636 | 0.2369 | 0.2685 | 0.2119 | 0.1627 |
| turnover_stability | neutral | 0.0317 | 0.1864 | 0.1699 | 0.1472 | 0.0975 |
| fund_pe | neutral | 0.0376 | 0.2231 | 0.1684 | 0.1315 | 0.0953 |
| low_downside | neutral | 0.0383 | 0.2422 | 0.1583 | 0.1357 | 0.0899 |
| fund_profit_growth | neutral | 0.0125 | 0.2211 | 0.0565 | 0.1106 | 0.0314 |
| trend_lowvol | bull | 0.0898 | 0.2321 | 0.3868 | 0.3636 | 0.2637 |
| low_downside | bull | 0.0778 | 0.2110 | 0.3685 | 0.2500 | 0.2303 |
| volatility | bull | 0.0696 | 0.2493 | 0.2793 | 0.2273 | 0.1714 |
| fund_pb | bull | 0.0579 | 0.2273 | 0.2549 | 0.3106 | 0.1670 |
| mom_x_lowvol_20_20 | bull | 0.0537 | 0.2341 | 0.2294 | 0.1818 | 0.1355 |
| momentum_reversal | bull | 0.0510 | 0.2462 | 0.2073 | 0.1439 | 0.1186 |
| rsi_vol_combo | bull | 0.0448 | 0.2226 | 0.2012 | 0.1212 | 0.1128 |
| fund_pe | bull | 0.0272 | 0.2007 | 0.1356 | 0.1818 | 0.0801 |
| exhaustion_risk | bull | 0.0274 | 0.2113 | 0.1299 | 0.1715 | 0.0761 |
| momentum_reversal | bear | 0.1151 | 0.2863 | 0.4021 | 0.3699 | 0.2754 |
| vol_confirm | bear | 0.0605 | 0.1792 | 0.3377 | 0.2329 | 0.2082 |
| rsi_vol_combo | bear | 0.0759 | 0.2614 | 0.2902 | 0.3699 | 0.1988 |
| bb_width_20 | bear | 0.0773 | 0.2393 | 0.3232 | 0.1781 | 0.1904 |
| fund_profit_growth | bear | 0.0575 | 0.1995 | 0.2881 | 0.2877 | 0.1855 |
| mom_x_lowvol_20_20 | bear | 0.0669 | 0.2653 | 0.2523 | 0.2877 | 0.1625 |
| fund_revenue_growth | bear | 0.0583 | 0.2269 | 0.2569 | 0.2329 | 0.1584 |
| trend_lowvol | bear | 0.0613 | 0.2694 | 0.2276 | 0.1781 | 0.1341 |
| fund_gross_margin | bear | 0.0482 | 0.2398 | 0.2010 | 0.1781 | 0.1184 |
| turnover_stability | bear | 0.0366 | 0.1934 | 0.1891 | 0.1644 | 0.1101 |

### 生态农业

- **Neutral**: ['volatility', 'fund_pb', 'turnover_stability'] (单因子IC=0.0738, 组合IC=0.0926)
  - weights: [0.3701, 0.325, 0.3048]
- **Bull**: ['momentum_reversal', 'rsi_vol_combo'] (单因子IC=0.0958, 组合IC=0.106)
  - bull_weights: [0.501, 0.499]
- **Bear**: ['fund_pb', 'mom_x_lowvol_20_20', 'rsi_vol_combo'] (单因子IC=0.1021, 组合IC=0.1493)
  - bear_weights: [0.5601, 0.2258, 0.2141]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| volatility | neutral | 0.0896 | 0.2231 | 0.4017 | 0.3591 | 0.2730 |
| fund_pb | neutral | 0.0685 | 0.1830 | 0.3746 | 0.2797 | 0.2397 |
| turnover_stability | neutral | 0.0631 | 0.1798 | 0.3511 | 0.2808 | 0.2248 |
| momentum_reversal | neutral | 0.0737 | 0.2319 | 0.3179 | 0.2860 | 0.2044 |
| mom_x_lowvol_20_20 | neutral | 0.0702 | 0.2303 | 0.3048 | 0.2526 | 0.1909 |
| rsi_vol_combo | neutral | 0.0612 | 0.2176 | 0.2814 | 0.2568 | 0.1768 |
| trend_lowvol | neutral | 0.0665 | 0.2397 | 0.2776 | 0.1983 | 0.1663 |
| fund_profit_growth | neutral | 0.0472 | 0.2105 | 0.2244 | 0.1743 | 0.1318 |
| fund_score | neutral | 0.0490 | 0.2192 | 0.2233 | 0.1681 | 0.1304 |
| fund_pe | neutral | 0.0538 | 0.2608 | 0.2061 | 0.1461 | 0.1181 |
| low_downside | neutral | 0.0431 | 0.2255 | 0.1913 | 0.2213 | 0.1168 |
| fund_roe | neutral | 0.0366 | 0.2292 | 0.1596 | 0.1169 | 0.0892 |
| momentum_reversal | bull | 0.0971 | 0.2332 | 0.4165 | 0.3485 | 0.2808 |
| rsi_vol_combo | bull | 0.0945 | 0.2241 | 0.4219 | 0.3258 | 0.2797 |
| volatility | bull | 0.0870 | 0.2202 | 0.3950 | 0.3106 | 0.2588 |
| mom_x_lowvol_20_20 | bull | 0.0863 | 0.2363 | 0.3651 | 0.3182 | 0.2406 |
| trend_lowvol | bull | 0.0797 | 0.2248 | 0.3545 | 0.2879 | 0.2283 |
| low_downside | bull | 0.0699 | 0.2203 | 0.3171 | 0.2955 | 0.2054 |
| fund_pb | bull | 0.0668 | 0.2108 | 0.3169 | 0.2197 | 0.1933 |
| fund_pe | bull | 0.0593 | 0.2251 | 0.2634 | 0.2424 | 0.1636 |
| turnover_stability | bull | 0.0488 | 0.1810 | 0.2697 | 0.1894 | 0.1604 |
| fund_gross_margin | bull | 0.0352 | 0.1845 | 0.1906 | 0.1061 | 0.1054 |
| stroke_phase | bull | 0.0231 | 0.1856 | 0.1246 | 0.1667 | 0.0727 |
| fund_pb | bear | 0.1281 | 0.1723 | 0.7438 | 0.6164 | 0.6011 |
| mom_x_lowvol_20_20 | bear | 0.1005 | 0.2557 | 0.3931 | 0.2329 | 0.2424 |
| rsi_vol_combo | bear | 0.0777 | 0.2109 | 0.3686 | 0.2466 | 0.2298 |
| fund_score | bear | 0.0678 | 0.1924 | 0.3524 | 0.2055 | 0.2124 |
| momentum_reversal | bear | 0.0932 | 0.2672 | 0.3489 | 0.1918 | 0.2079 |
| fund_pe | bear | 0.0817 | 0.2504 | 0.3262 | 0.2055 | 0.1966 |
| fund_roe | bear | 0.0738 | 0.2367 | 0.3117 | 0.2055 | 0.1879 |
| bb_width_20 | bear | 0.0519 | 0.2272 | 0.2286 | 0.2055 | 0.1378 |
| trend_lowvol | bear | 0.0486 | 0.2290 | 0.2124 | 0.1507 | 0.1222 |

### 生物疫苗

- **Neutral**: ['fund_pb', 'volatility', 'low_downside'] (单因子IC=0.0951, 组合IC=0.1146)
  - weights: [0.4228, 0.3228, 0.2545]
- **Bull**: ['fund_pb', 'low_downside'] (单因子IC=0.0947, 组合IC=0.1128)
  - bull_weights: [0.5236, 0.4764]
- **Bear**: ['trend_lowvol', 'mom_x_lowvol_20_20'] (单因子IC=0.1266, 组合IC=0.1449)
  - bear_weights: [0.5699, 0.4301]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| fund_pb | neutral | 0.1121 | 0.2092 | 0.5357 | 0.4259 | 0.3819 |
| volatility | neutral | 0.0957 | 0.2186 | 0.4378 | 0.3319 | 0.2915 |
| low_downside | neutral | 0.0776 | 0.2164 | 0.3586 | 0.2818 | 0.2299 |
| fund_pe | neutral | 0.0643 | 0.2092 | 0.3075 | 0.2109 | 0.1862 |
| turnover_stability | neutral | 0.0398 | 0.1594 | 0.2496 | 0.2067 | 0.1506 |
| trend_lowvol | neutral | 0.0572 | 0.2266 | 0.2524 | 0.1566 | 0.1460 |
| rsi_vol_combo | neutral | 0.0357 | 0.1924 | 0.1856 | 0.1409 | 0.1059 |
| fund_pb | bull | 0.1030 | 0.1850 | 0.5564 | 0.4242 | 0.3962 |
| low_downside | bull | 0.0865 | 0.1718 | 0.5035 | 0.4318 | 0.3605 |
| volatility | bull | 0.1014 | 0.2083 | 0.4870 | 0.4697 | 0.3579 |
| turnover_stability | bull | 0.0521 | 0.1363 | 0.3819 | 0.2727 | 0.2430 |
| trend_lowvol | bull | 0.0654 | 0.1775 | 0.3687 | 0.2879 | 0.2374 |
| fund_pe | bull | 0.0571 | 0.2121 | 0.2692 | 0.2424 | 0.1672 |
| rsi_vol_combo | bull | 0.0380 | 0.1710 | 0.2220 | 0.1515 | 0.1278 |
| momentum_reversal | bull | 0.0362 | 0.1817 | 0.1993 | 0.2197 | 0.1215 |
| fund_revenue_growth | bull | 0.0325 | 0.1740 | 0.1869 | 0.1212 | 0.1048 |
| fund_score | bull | 0.0340 | 0.2007 | 0.1692 | 0.1288 | 0.0955 |
| mom_x_lowvol_20_20 | bull | 0.0201 | 0.1929 | 0.1042 | 0.1970 | 0.0624 |
| trend_lowvol | bear | 0.1277 | 0.1712 | 0.7460 | 0.5068 | 0.5620 |
| mom_x_lowvol_20_20 | bear | 0.1256 | 0.2150 | 0.5843 | 0.4521 | 0.4242 |
| momentum_reversal | bear | 0.1000 | 0.2059 | 0.4854 | 0.3425 | 0.3258 |
| rsi_vol_combo | bear | 0.0728 | 0.1772 | 0.4108 | 0.3151 | 0.2701 |
| fund_pb | bear | 0.0552 | 0.1716 | 0.3219 | 0.2329 | 0.1984 |
| fund_pe | bear | 0.0450 | 0.1849 | 0.2432 | 0.2055 | 0.1466 |
| fund_gross_margin | bear | 0.0541 | 0.2407 | 0.2246 | 0.1507 | 0.1292 |
| bb_width_20 | bear | 0.0451 | 0.2001 | 0.2254 | 0.1233 | 0.1266 |

### 生物识别

- **Neutral**: ['fund_pb', 'trend_lowvol', 'momentum_reversal'] (单因子IC=0.0967, 组合IC=0.1315)
  - weights: [0.3997, 0.3088, 0.2915]
- **Bull**: ['volatility', 'trend_lowvol', 'fund_pb'] (单因子IC=0.0944, 组合IC=0.1247)
  - bull_weights: [0.4192, 0.2951, 0.2857]
- **Bear**: ['turnover_stability', 'mom_x_lowvol_20_20', 'fund_pb'] (单因子IC=0.1042, 组合IC=0.177)
  - bear_weights: [0.5089, 0.2729, 0.2182]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| fund_pb | neutral | 0.0907 | 0.1800 | 0.5041 | 0.4092 | 0.3552 |
| trend_lowvol | neutral | 0.1060 | 0.2523 | 0.4200 | 0.3069 | 0.2745 |
| momentum_reversal | neutral | 0.0933 | 0.2338 | 0.3990 | 0.2985 | 0.2590 |
| mom_x_lowvol_20_20 | neutral | 0.0796 | 0.2290 | 0.3478 | 0.2484 | 0.2171 |
| rsi_vol_combo | neutral | 0.0782 | 0.2298 | 0.3404 | 0.2693 | 0.2161 |
| volatility | neutral | 0.0799 | 0.2507 | 0.3188 | 0.2380 | 0.1973 |
| low_downside | neutral | 0.0649 | 0.2463 | 0.2633 | 0.2046 | 0.1586 |
| fund_pe | neutral | 0.0416 | 0.1933 | 0.2153 | 0.1879 | 0.1278 |
| fund_roe | neutral | 0.0287 | 0.2284 | 0.1256 | 0.1169 | 0.0702 |
| volatility | bull | 0.1155 | 0.2164 | 0.5340 | 0.4394 | 0.3843 |
| trend_lowvol | bull | 0.0957 | 0.2426 | 0.3946 | 0.3712 | 0.2705 |
| fund_pb | bull | 0.0719 | 0.1872 | 0.3842 | 0.3636 | 0.2620 |
| turnover_stability | bull | 0.0702 | 0.1743 | 0.4028 | 0.2652 | 0.2548 |
| low_downside | bull | 0.0783 | 0.1985 | 0.3944 | 0.2652 | 0.2495 |
| fund_revenue_growth | bull | 0.0518 | 0.2074 | 0.2496 | 0.2197 | 0.1522 |
| mom_x_lowvol_20_20 | bull | 0.0464 | 0.2227 | 0.2085 | 0.2121 | 0.1264 |
| momentum_reversal | bull | 0.0465 | 0.2316 | 0.2009 | 0.1894 | 0.1195 |
| fund_profit_growth | bull | 0.0388 | 0.1936 | 0.2006 | 0.1667 | 0.1170 |
| fund_pe | bull | 0.0352 | 0.1909 | 0.1842 | 0.2197 | 0.1123 |
| fund_score | bull | 0.0332 | 0.2166 | 0.1532 | 0.1515 | 0.0882 |
| turnover_stability | bear | 0.1219 | 0.1625 | 0.7501 | 0.5616 | 0.5857 |
| mom_x_lowvol_20_20 | bear | 0.1164 | 0.2487 | 0.4680 | 0.3425 | 0.3141 |
| fund_pb | bear | 0.0743 | 0.1825 | 0.4073 | 0.2329 | 0.2511 |
| momentum_reversal | bear | 0.0996 | 0.2564 | 0.3884 | 0.2055 | 0.2341 |
| fund_revenue_growth | bear | 0.0653 | 0.2262 | 0.2885 | 0.2877 | 0.1858 |
| fund_profit_growth | bear | 0.0686 | 0.2219 | 0.3094 | 0.1781 | 0.1822 |
| wash_sale_score | bear | 0.0503 | 0.1825 | 0.2759 | 0.2941 | 0.1785 |
| trend_lowvol | bear | 0.0675 | 0.2402 | 0.2811 | 0.2329 | 0.1733 |
| fund_score | bear | 0.0585 | 0.2361 | 0.2476 | 0.2329 | 0.1526 |

### 生物质能发电

- **Neutral**: ['fund_pb', 'turnover_stability', 'rsi_vol_combo'] (单因子IC=0.0786, 组合IC=0.1092)
  - weights: [0.4314, 0.3053, 0.2632]
- **Bull**: ['turnover_stability', 'fund_pb', 'fund_pe'] (单因子IC=0.087, 组合IC=0.1352)
  - bull_weights: [0.4014, 0.3065, 0.2921]
- **Bear**: ['fund_pb', 'trend_lowvol'] (单因子IC=0.1042, 组合IC=0.1387)
  - bear_weights: [0.6431, 0.3569]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| fund_pb | neutral | 0.0936 | 0.2202 | 0.4250 | 0.3466 | 0.2861 |
| turnover_stability | neutral | 0.0712 | 0.2150 | 0.3311 | 0.2234 | 0.2025 |
| rsi_vol_combo | neutral | 0.0709 | 0.2499 | 0.2837 | 0.2307 | 0.1746 |
| momentum_reversal | neutral | 0.0754 | 0.2621 | 0.2877 | 0.2088 | 0.1739 |
| fund_profit_growth | neutral | 0.0557 | 0.2111 | 0.2636 | 0.2004 | 0.1582 |
| mom_x_lowvol_20_20 | neutral | 0.0626 | 0.2472 | 0.2532 | 0.1931 | 0.1510 |
| volatility | neutral | 0.0589 | 0.2372 | 0.2482 | 0.2150 | 0.1508 |
| fund_score | neutral | 0.0530 | 0.2239 | 0.2368 | 0.1879 | 0.1407 |
| trend_lowvol | neutral | 0.0545 | 0.2697 | 0.2022 | 0.1743 | 0.1187 |
| turnover_stability | bull | 0.0891 | 0.1954 | 0.4561 | 0.3523 | 0.3084 |
| fund_pb | bull | 0.0808 | 0.2139 | 0.3779 | 0.2462 | 0.2355 |
| fund_pe | bull | 0.0909 | 0.2548 | 0.3568 | 0.2576 | 0.2244 |
| wash_sale_score | bull | 0.0678 | 0.2147 | 0.3159 | 0.2239 | 0.1933 |
| volatility | bull | 0.0519 | 0.2195 | 0.2366 | 0.2879 | 0.1523 |
| low_downside | bull | 0.0598 | 0.2327 | 0.2567 | 0.1742 | 0.1507 |
| momentum_reversal | bull | 0.0501 | 0.2393 | 0.2092 | 0.2045 | 0.1260 |
| rsi_vol_combo | bull | 0.0470 | 0.2358 | 0.1992 | 0.1591 | 0.1155 |
| trend_lowvol | bull | 0.0447 | 0.2658 | 0.1682 | 0.1818 | 0.0994 |
| mom_x_lowvol_20_20 | bull | 0.0398 | 0.2373 | 0.1676 | 0.1136 | 0.0933 |
| fund_pb | bear | 0.1190 | 0.1860 | 0.6395 | 0.5068 | 0.4818 |
| trend_lowvol | bear | 0.0893 | 0.2243 | 0.3983 | 0.3425 | 0.2674 |
| top_fractal_volume | bear | 0.0691 | 0.1861 | 0.3714 | 0.1304 | 0.2099 |
| momentum_reversal | bear | 0.0880 | 0.2883 | 0.3054 | 0.3014 | 0.1987 |
| mom_x_lowvol_20_20 | bear | 0.0606 | 0.2916 | 0.2077 | 0.2055 | 0.1252 |

### 电商概念

- **Neutral**: ['fund_pb', 'turnover_stability', 'trend_lowvol'] (单因子IC=0.0702, 组合IC=0.1157)
  - weights: [0.4147, 0.2946, 0.2907]
- **Bull**: ['low_downside', 'fund_pb'] (单因子IC=0.0799, 组合IC=0.0992)
  - bull_weights: [0.5316, 0.4684]
- **Bear**: ['turnover_stability', 'mom_x_lowvol_20_20', 'momentum_reversal'] (单因子IC=0.0914, 组合IC=0.1251)
  - bear_weights: [0.3513, 0.3372, 0.3116]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| fund_pb | neutral | 0.0880 | 0.1330 | 0.6613 | 0.4572 | 0.4818 |
| turnover_stability | neutral | 0.0405 | 0.0814 | 0.4976 | 0.3758 | 0.3423 |
| trend_lowvol | neutral | 0.0822 | 0.1674 | 0.4910 | 0.3758 | 0.3377 |
| mom_x_lowvol_20_20 | neutral | 0.0718 | 0.1481 | 0.4850 | 0.3758 | 0.3336 |
| volatility | neutral | 0.0717 | 0.1625 | 0.4411 | 0.3883 | 0.3062 |
| momentum_reversal | neutral | 0.0697 | 0.1567 | 0.4447 | 0.2985 | 0.2888 |
| fund_pe | neutral | 0.0617 | 0.1492 | 0.4132 | 0.2484 | 0.2579 |
| low_downside | neutral | 0.0523 | 0.1577 | 0.3318 | 0.3820 | 0.2293 |
| fund_profit_growth | neutral | 0.0355 | 0.1066 | 0.3327 | 0.3006 | 0.2163 |
| rsi_vol_combo | neutral | 0.0525 | 0.1521 | 0.3455 | 0.2463 | 0.2153 |
| fund_score | neutral | 0.0244 | 0.1449 | 0.1686 | 0.1273 | 0.0950 |
| low_downside | bull | 0.0774 | 0.1122 | 0.6903 | 0.5606 | 0.5386 |
| fund_pb | bull | 0.0824 | 0.1283 | 0.6425 | 0.4773 | 0.4745 |
| volatility | bull | 0.0704 | 0.1285 | 0.5478 | 0.5227 | 0.4171 |
| fund_pe | bull | 0.0618 | 0.1493 | 0.4142 | 0.3258 | 0.2746 |
| momentum_reversal | bull | 0.0505 | 0.1355 | 0.3724 | 0.3864 | 0.2581 |
| turnover_stability | bull | 0.0292 | 0.0829 | 0.3523 | 0.2727 | 0.2242 |
| rsi_vol_combo | bull | 0.0431 | 0.1311 | 0.3289 | 0.3030 | 0.2143 |
| trend_lowvol | bull | 0.0454 | 0.1528 | 0.2970 | 0.2576 | 0.1867 |
| mom_x_lowvol_20_20 | bull | 0.0322 | 0.1296 | 0.2485 | 0.3182 | 0.1638 |
| fund_profit_growth | bull | 0.0213 | 0.0932 | 0.2289 | 0.1061 | 0.1266 |
| fund_gross_margin | bull | 0.0198 | 0.1058 | 0.1868 | 0.1136 | 0.1040 |
| turnover_stability | bear | 0.0396 | 0.0644 | 0.6150 | 0.4247 | 0.4381 |
| mom_x_lowvol_20_20 | bear | 0.1163 | 0.1970 | 0.5903 | 0.4247 | 0.4205 |
| momentum_reversal | bear | 0.1182 | 0.2000 | 0.5910 | 0.3151 | 0.3886 |
| trend_lowvol | bear | 0.0839 | 0.1903 | 0.4411 | 0.3973 | 0.3081 |
| rsi_vol_combo | bear | 0.0829 | 0.1795 | 0.4620 | 0.2329 | 0.2848 |
| fund_profit_growth | bear | 0.0454 | 0.1209 | 0.3760 | 0.2055 | 0.2266 |
| fund_pb | bear | 0.0412 | 0.1371 | 0.3003 | 0.1781 | 0.1769 |
| fund_pe | bear | 0.0416 | 0.1472 | 0.2827 | 0.1781 | 0.1665 |
| fund_score | bear | 0.0456 | 0.1609 | 0.2833 | 0.1507 | 0.1630 |
| fund_revenue_growth | bear | 0.0305 | 0.1181 | 0.2584 | 0.1233 | 0.1451 |
| bb_width_20 | bear | 0.0423 | 0.2106 | 0.2007 | 0.2603 | 0.1264 |

### 电子后视镜

- **Neutral**: ['momentum_reversal', 'rsi_vol_combo'] (单因子IC=0.092, 组合IC=0.1007)
  - weights: [0.5109, 0.4891]
- **Bull**: ['mom_x_lowvol_20_20', 'momentum_reversal', 'trend_lowvol'] (单因子IC=0.1354, 组合IC=0.1478)
  - bull_weights: [0.3422, 0.3345, 0.3233]
- **Bear**: ['fund_pb', 'fund_gross_margin', 'mom_x_lowvol_20_20'] (单因子IC=0.1266, 组合IC=0.235)
  - bear_weights: [0.3986, 0.3379, 0.2634]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| momentum_reversal | neutral | 0.0934 | 0.2636 | 0.3543 | 0.2599 | 0.2232 |
| rsi_vol_combo | neutral | 0.0905 | 0.2641 | 0.3428 | 0.2463 | 0.2136 |
| mom_x_lowvol_20_20 | neutral | 0.0906 | 0.2657 | 0.3411 | 0.2516 | 0.2134 |
| volatility | neutral | 0.0869 | 0.3028 | 0.2869 | 0.2766 | 0.1831 |
| fund_pb | neutral | 0.0623 | 0.2321 | 0.2686 | 0.2411 | 0.1667 |
| fund_profit_growth | neutral | 0.0538 | 0.2390 | 0.2249 | 0.1848 | 0.1332 |
| fund_score | neutral | 0.0512 | 0.2529 | 0.2026 | 0.1576 | 0.1173 |
| trend_lowvol | neutral | 0.0540 | 0.2817 | 0.1918 | 0.1190 | 0.1073 |
| low_downside | neutral | 0.0468 | 0.2911 | 0.1607 | 0.1587 | 0.0931 |
| fund_pe | neutral | 0.0358 | 0.2694 | 0.1330 | 0.1503 | 0.0765 |
| fund_revenue_growth | neutral | 0.0309 | 0.2330 | 0.1325 | 0.1169 | 0.0740 |
| mom_x_lowvol_20_20 | bull | 0.1457 | 0.2919 | 0.4992 | 0.3788 | 0.3441 |
| momentum_reversal | bull | 0.1363 | 0.2894 | 0.4711 | 0.4280 | 0.3364 |
| trend_lowvol | bull | 0.1241 | 0.2617 | 0.4742 | 0.3712 | 0.3251 |
| fund_pb | bull | 0.1046 | 0.2210 | 0.4734 | 0.3371 | 0.3165 |
| volatility | bull | 0.1032 | 0.2659 | 0.3882 | 0.2879 | 0.2500 |
| fund_revenue_growth | bull | 0.0677 | 0.2171 | 0.3117 | 0.2273 | 0.1913 |
| rsi_vol_combo | bull | 0.0759 | 0.2788 | 0.2722 | 0.1894 | 0.1619 |
| turnover_stability | bull | 0.0534 | 0.2121 | 0.2519 | 0.1136 | 0.1403 |
| low_downside | bull | 0.0541 | 0.2526 | 0.2142 | 0.1629 | 0.1245 |
| fund_pe | bull | 0.0603 | 0.2903 | 0.2078 | 0.1515 | 0.1196 |
| fund_pb | bear | 0.1627 | 0.3085 | 0.5273 | 0.4795 | 0.3901 |
| fund_gross_margin | bear | 0.1050 | 0.2174 | 0.4828 | 0.3699 | 0.3307 |
| mom_x_lowvol_20_20 | bear | 0.1122 | 0.2803 | 0.4004 | 0.2877 | 0.2578 |
| fund_pe | bear | 0.1264 | 0.3328 | 0.3799 | 0.2329 | 0.2342 |
| momentum_reversal | bear | 0.0943 | 0.2781 | 0.3391 | 0.2055 | 0.2044 |
| low_downside | bear | 0.0718 | 0.2699 | 0.2659 | 0.1507 | 0.1530 |
| fund_roe | bear | 0.0585 | 0.2254 | 0.2594 | 0.1507 | 0.1493 |
| fund_profit_growth | bear | 0.0726 | 0.2836 | 0.2561 | 0.1233 | 0.1438 |
| volatility | bear | 0.0595 | 0.3323 | 0.1789 | 0.1507 | 0.1029 |

### 电子烟

- **Neutral**: ['trend_lowvol'] (单因子IC=0.0948, 组合IC=0.0948)
  - weights: [1.0]
- **Bull**: ['fund_pe', 'volatility', 'turnover_stability'] (单因子IC=0.0922, 组合IC=0.1238)
  - bull_weights: [0.3501, 0.3279, 0.322]
- **Bear**: ['mom_x_lowvol_20_20'] (单因子IC=0.1236, 组合IC=0.1236)
  - bear_weights: [1.0]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| trend_lowvol | neutral | 0.0948 | 0.2234 | 0.4243 | 0.3570 | 0.2879 |
| momentum_reversal | neutral | 0.0781 | 0.2095 | 0.3728 | 0.3048 | 0.2432 |
| mom_x_lowvol_20_20 | neutral | 0.0762 | 0.2074 | 0.3673 | 0.2860 | 0.2362 |
| fund_pb | neutral | 0.0828 | 0.2253 | 0.3677 | 0.2630 | 0.2322 |
| volatility | neutral | 0.0729 | 0.2160 | 0.3373 | 0.2150 | 0.2049 |
| fund_profit_growth | neutral | 0.0496 | 0.1717 | 0.2890 | 0.2735 | 0.1840 |
| fund_pe | neutral | 0.0528 | 0.1876 | 0.2812 | 0.2296 | 0.1729 |
| rsi_vol_combo | neutral | 0.0486 | 0.2017 | 0.2409 | 0.1983 | 0.1443 |
| fund_score | neutral | 0.0390 | 0.1933 | 0.2020 | 0.1775 | 0.1189 |
| low_downside | neutral | 0.0394 | 0.2025 | 0.1943 | 0.1952 | 0.1161 |
| turnover_stability | neutral | 0.0220 | 0.1636 | 0.1348 | 0.1075 | 0.0746 |
| fund_roe | neutral | 0.0247 | 0.2017 | 0.1223 | 0.1023 | 0.0674 |
| fund_pe | bull | 0.0882 | 0.1625 | 0.5428 | 0.3561 | 0.3681 |
| volatility | bull | 0.1027 | 0.2145 | 0.4790 | 0.4394 | 0.3447 |
| turnover_stability | bull | 0.0857 | 0.1803 | 0.4753 | 0.4242 | 0.3385 |
| trend_lowvol | bull | 0.0988 | 0.2149 | 0.4596 | 0.3409 | 0.3081 |
| low_downside | bull | 0.0763 | 0.1728 | 0.4419 | 0.3030 | 0.2879 |
| fund_pb | bull | 0.0842 | 0.2207 | 0.3817 | 0.2500 | 0.2386 |
| fund_gross_margin | bull | 0.0510 | 0.1434 | 0.3554 | 0.2576 | 0.2235 |
| rsi_vol_combo | bull | 0.0559 | 0.1891 | 0.2957 | 0.1591 | 0.1713 |
| mom_x_lowvol_20_20 | bull | 0.0577 | 0.2151 | 0.2683 | 0.1742 | 0.1575 |
| momentum_reversal | bull | 0.0532 | 0.2184 | 0.2435 | 0.1818 | 0.1439 |
| mom_x_lowvol_20_20 | bear | 0.1236 | 0.2104 | 0.5876 | 0.3973 | 0.4105 |
| momentum_reversal | bear | 0.1061 | 0.2041 | 0.5200 | 0.4247 | 0.3704 |
| turnover_stability | bear | 0.0677 | 0.1647 | 0.4111 | 0.3699 | 0.2816 |
| fund_profit_growth | bear | 0.0603 | 0.1552 | 0.3882 | 0.3973 | 0.2712 |
| rsi_vol_combo | bear | 0.0753 | 0.1862 | 0.4044 | 0.2877 | 0.2603 |
| trend_lowvol | bear | 0.0859 | 0.2290 | 0.3751 | 0.2055 | 0.2261 |
| fund_pe | bear | 0.0468 | 0.2114 | 0.2215 | 0.2055 | 0.1335 |
| fund_pb | bear | 0.0563 | 0.2551 | 0.2208 | 0.1507 | 0.1270 |

### 电子竞技

- **Neutral**: ['trend_lowvol', 'fund_pb', 'momentum_reversal'] (单因子IC=0.0786, 组合IC=0.1138)
  - weights: [0.3581, 0.3498, 0.292]
- **Bull**: ['fund_pb', 'fund_pe', 'volatility'] (单因子IC=0.0564, 组合IC=0.0639)
  - bull_weights: [0.4038, 0.3203, 0.276]
- **Bear**: ['turnover_stability', 'momentum_reversal'] (单因子IC=0.094, 组合IC=0.1228)
  - bear_weights: [0.5189, 0.4811]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| trend_lowvol | neutral | 0.0917 | 0.2779 | 0.3299 | 0.2756 | 0.2104 |
| fund_pb | neutral | 0.0678 | 0.2061 | 0.3292 | 0.2484 | 0.2055 |
| momentum_reversal | neutral | 0.0764 | 0.2706 | 0.2824 | 0.2150 | 0.1715 |
| mom_x_lowvol_20_20 | neutral | 0.0729 | 0.2632 | 0.2771 | 0.2317 | 0.1707 |
| fund_pe | neutral | 0.0676 | 0.2566 | 0.2633 | 0.1983 | 0.1578 |
| rsi_vol_combo | neutral | 0.0646 | 0.2603 | 0.2482 | 0.2109 | 0.1502 |
| volatility | neutral | 0.0721 | 0.3144 | 0.2292 | 0.1587 | 0.1328 |
| turnover_stability | neutral | 0.0388 | 0.2068 | 0.1875 | 0.1858 | 0.1111 |
| fund_profit_growth | neutral | 0.0379 | 0.1939 | 0.1954 | 0.1169 | 0.1091 |
| fund_score | neutral | 0.0408 | 0.2284 | 0.1786 | 0.1441 | 0.1022 |
| fund_revenue_growth | neutral | 0.0361 | 0.2185 | 0.1653 | 0.1482 | 0.0949 |
| fund_pb | bull | 0.0551 | 0.1852 | 0.2976 | 0.2462 | 0.1854 |
| fund_pe | bull | 0.0634 | 0.2563 | 0.2473 | 0.1894 | 0.1471 |
| volatility | bull | 0.0508 | 0.2428 | 0.2091 | 0.2121 | 0.1267 |
| turnover_stability | bull | 0.0362 | 0.2110 | 0.1718 | 0.1098 | 0.0953 |
| fund_revenue_growth | bull | 0.0365 | 0.2203 | 0.1659 | 0.1136 | 0.0924 |
| momentum_reversal | bull | 0.0319 | 0.2560 | 0.1247 | 0.1515 | 0.0718 |
| turnover_stability | bear | 0.0781 | 0.1878 | 0.4157 | 0.4521 | 0.3018 |
| momentum_reversal | bear | 0.1099 | 0.2689 | 0.4086 | 0.3699 | 0.2799 |
| mom_x_lowvol_20_20 | bear | 0.0838 | 0.2425 | 0.3455 | 0.2329 | 0.2130 |
| fund_profit_growth | bear | 0.0682 | 0.2241 | 0.3041 | 0.3425 | 0.2041 |
| trend_lowvol | bear | 0.0892 | 0.2567 | 0.3475 | 0.1507 | 0.1999 |
| wash_sale_score | bear | 0.0741 | 0.2074 | 0.3571 | 0.1176 | 0.1996 |
| fund_revenue_growth | bear | 0.0534 | 0.1855 | 0.2877 | 0.2055 | 0.1734 |

### 电子身份证

- **Neutral**: ['fund_pb', 'momentum_reversal', 'trend_lowvol'] (单因子IC=0.1101, 组合IC=0.1561)
  - weights: [0.4232, 0.3001, 0.2768]
- **Bull**: ['volatility', 'trend_lowvol', 'mom_x_lowvol_20_20'] (单因子IC=0.1748, 组合IC=0.2053)
  - bull_weights: [0.4066, 0.2994, 0.294]
- **Bear**: ['momentum_reversal', 'mom_x_lowvol_20_20', 'fund_pb'] (单因子IC=0.097, 组合IC=0.1222)
  - bear_weights: [0.37, 0.341, 0.289]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| fund_pb | neutral | 0.1136 | 0.1841 | 0.6173 | 0.4739 | 0.4549 |
| momentum_reversal | neutral | 0.1102 | 0.2337 | 0.4718 | 0.3674 | 0.3226 |
| trend_lowvol | neutral | 0.1064 | 0.2368 | 0.4493 | 0.3246 | 0.2976 |
| rsi_vol_combo | neutral | 0.0990 | 0.2243 | 0.4415 | 0.3184 | 0.2910 |
| mom_x_lowvol_20_20 | neutral | 0.1015 | 0.2369 | 0.4285 | 0.3424 | 0.2876 |
| volatility | neutral | 0.0853 | 0.2414 | 0.3534 | 0.2933 | 0.2285 |
| fund_score | neutral | 0.0529 | 0.2107 | 0.2512 | 0.1754 | 0.1477 |
| fund_roe | neutral | 0.0510 | 0.2170 | 0.2349 | 0.1441 | 0.1344 |
| fund_pe | neutral | 0.0451 | 0.2736 | 0.1650 | 0.1367 | 0.0938 |
| turnover_stability | neutral | 0.0242 | 0.1928 | 0.1254 | 0.1482 | 0.0720 |
| fund_revenue_growth | neutral | 0.0225 | 0.1806 | 0.1248 | 0.1096 | 0.0693 |
| low_downside | neutral | 0.0257 | 0.2311 | 0.1113 | 0.1002 | 0.0612 |
| volatility | bull | 0.1923 | 0.2434 | 0.7900 | 0.6061 | 0.6344 |
| trend_lowvol | bull | 0.1683 | 0.2621 | 0.6422 | 0.4545 | 0.4671 |
| mom_x_lowvol_20_20 | bull | 0.1638 | 0.2624 | 0.6243 | 0.4697 | 0.4587 |
| fund_pb | bull | 0.0987 | 0.1577 | 0.6259 | 0.4394 | 0.4504 |
| momentum_reversal | bull | 0.1504 | 0.2542 | 0.5916 | 0.4697 | 0.4348 |
| low_downside | bull | 0.1110 | 0.2437 | 0.4554 | 0.3864 | 0.3157 |
| rsi_vol_combo | bull | 0.0811 | 0.2602 | 0.3116 | 0.2386 | 0.1930 |
| turnover_stability | bull | 0.0619 | 0.2110 | 0.2934 | 0.2235 | 0.1795 |
| fund_pe | bull | 0.0677 | 0.2519 | 0.2686 | 0.2045 | 0.1618 |
| fund_profit_growth | bull | 0.0222 | 0.1780 | 0.1250 | 0.1061 | 0.0691 |
| momentum_reversal | bear | 0.1189 | 0.2748 | 0.4327 | 0.3425 | 0.2904 |
| mom_x_lowvol_20_20 | bear | 0.1127 | 0.2710 | 0.4158 | 0.2877 | 0.2677 |
| fund_pb | bear | 0.0594 | 0.1686 | 0.3524 | 0.2877 | 0.2269 |
| fund_profit_growth | bear | 0.0548 | 0.1877 | 0.2920 | 0.2329 | 0.1800 |
| rsi_vol_combo | bear | 0.0563 | 0.2698 | 0.2085 | 0.1781 | 0.1228 |
| fund_score | bear | 0.0419 | 0.2197 | 0.1909 | 0.1781 | 0.1124 |
| fund_pe | bear | 0.0451 | 0.2395 | 0.1884 | 0.1549 | 0.1088 |

### 电池技术

- **Neutral**: ['fund_pb', 'momentum_reversal', 'mom_x_lowvol_20_20'] (单因子IC=0.0639, 组合IC=0.095)
  - weights: [0.3555, 0.3292, 0.3153]
- **Bull**: ['low_downside', 'volatility', 'fund_pb'] (单因子IC=0.0845, 组合IC=0.1025)
  - bull_weights: [0.4017, 0.3149, 0.2834]
- **Bear**: ['rsi_vol_combo', 'momentum_reversal', 'mom_x_lowvol_20_20'] (单因子IC=0.1034, 组合IC=0.1135)
  - bear_weights: [0.3836, 0.3213, 0.2951]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| fund_pb | neutral | 0.0709 | 0.1574 | 0.4504 | 0.3486 | 0.3037 |
| momentum_reversal | neutral | 0.0629 | 0.1468 | 0.4284 | 0.3132 | 0.2813 |
| mom_x_lowvol_20_20 | neutral | 0.0581 | 0.1400 | 0.4149 | 0.2985 | 0.2694 |
| volatility | neutral | 0.0663 | 0.1717 | 0.3862 | 0.2881 | 0.2487 |
| fund_pe | neutral | 0.0569 | 0.1520 | 0.3740 | 0.2651 | 0.2366 |
| fund_profit_growth | neutral | 0.0444 | 0.1201 | 0.3697 | 0.2777 | 0.2362 |
| trend_lowvol | neutral | 0.0651 | 0.1775 | 0.3669 | 0.2651 | 0.2321 |
| rsi_vol_combo | neutral | 0.0469 | 0.1379 | 0.3402 | 0.2422 | 0.2113 |
| fund_score | neutral | 0.0367 | 0.1495 | 0.2456 | 0.1106 | 0.1364 |
| turnover_stability | neutral | 0.0218 | 0.0942 | 0.2317 | 0.1775 | 0.1364 |
| fund_revenue_growth | neutral | 0.0210 | 0.1063 | 0.1974 | 0.1712 | 0.1156 |
| low_downside | neutral | 0.0329 | 0.1752 | 0.1876 | 0.1315 | 0.1061 |
| low_downside | bull | 0.0886 | 0.1316 | 0.6735 | 0.5303 | 0.5153 |
| volatility | bull | 0.0861 | 0.1551 | 0.5555 | 0.4545 | 0.4040 |
| fund_pb | bull | 0.0788 | 0.1469 | 0.5363 | 0.3561 | 0.3636 |
| turnover_stability | bull | 0.0322 | 0.0843 | 0.3820 | 0.3485 | 0.2575 |
| trend_lowvol | bull | 0.0532 | 0.1552 | 0.3428 | 0.2121 | 0.2077 |
| fund_pe | bull | 0.0487 | 0.1457 | 0.3340 | 0.1742 | 0.1961 |
| fund_profit_growth | bull | 0.0226 | 0.1196 | 0.1886 | 0.2652 | 0.1193 |
| fund_gross_margin | bull | 0.0165 | 0.0937 | 0.1760 | 0.1667 | 0.1027 |
| fund_score | bull | 0.0198 | 0.1547 | 0.1281 | 0.1439 | 0.0733 |
| rsi_vol_combo | bear | 0.0886 | 0.1190 | 0.7452 | 0.6438 | 0.6125 |
| momentum_reversal | bear | 0.1160 | 0.1672 | 0.6935 | 0.4795 | 0.5130 |
| mom_x_lowvol_20_20 | bear | 0.1055 | 0.1717 | 0.6144 | 0.5342 | 0.4713 |
| trend_lowvol | bear | 0.0715 | 0.1866 | 0.3833 | 0.1233 | 0.2153 |
| bb_width_20 | bear | 0.0581 | 0.1841 | 0.3153 | 0.2055 | 0.1901 |
| fund_pb | bear | 0.0339 | 0.1924 | 0.1761 | 0.1507 | 0.1013 |

### 电网概念

- **Neutral**: ['fund_pb', 'fund_profit_growth', 'mom_x_lowvol_20_20'] (单因子IC=0.0679, 组合IC=0.0959)
  - weights: [0.4178, 0.2931, 0.2891]
- **Bull**: ['volatility', 'low_downside', 'fund_pb'] (单因子IC=0.0916, 组合IC=0.1145)
  - bull_weights: [0.3499, 0.3294, 0.3207]
- **Bear**: ['momentum_reversal', 'mom_x_lowvol_20_20'] (单因子IC=0.1127, 组合IC=0.1172)
  - bear_weights: [0.5129, 0.4871]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| fund_pb | neutral | 0.0755 | 0.1318 | 0.5728 | 0.4697 | 0.4209 |
| fund_profit_growth | neutral | 0.0529 | 0.1193 | 0.4434 | 0.3319 | 0.2953 |
| mom_x_lowvol_20_20 | neutral | 0.0752 | 0.1719 | 0.4373 | 0.3319 | 0.2912 |
| momentum_reversal | neutral | 0.0779 | 0.1800 | 0.4325 | 0.3361 | 0.2890 |
| fund_pe | neutral | 0.0633 | 0.1517 | 0.4171 | 0.3445 | 0.2804 |
| volatility | neutral | 0.0742 | 0.1893 | 0.3919 | 0.3361 | 0.2618 |
| rsi_vol_combo | neutral | 0.0640 | 0.1653 | 0.3869 | 0.3299 | 0.2573 |
| trend_lowvol | neutral | 0.0712 | 0.1988 | 0.3583 | 0.2839 | 0.2300 |
| fund_score | neutral | 0.0528 | 0.1492 | 0.3538 | 0.2672 | 0.2242 |
| turnover_stability | neutral | 0.0297 | 0.1058 | 0.2811 | 0.2046 | 0.1693 |
| fund_revenue_growth | neutral | 0.0317 | 0.1169 | 0.2713 | 0.2296 | 0.1668 |
| low_downside | neutral | 0.0458 | 0.1866 | 0.2456 | 0.2422 | 0.1526 |
| fund_roe | neutral | 0.0337 | 0.1614 | 0.2087 | 0.1608 | 0.1211 |
| volatility | bull | 0.1058 | 0.1613 | 0.6560 | 0.4697 | 0.4821 |
| low_downside | bull | 0.0866 | 0.1402 | 0.6175 | 0.4697 | 0.4538 |
| fund_pb | bull | 0.0824 | 0.1343 | 0.6138 | 0.4394 | 0.4417 |
| fund_pe | bull | 0.0689 | 0.1490 | 0.4626 | 0.3485 | 0.3119 |
| turnover_stability | bull | 0.0346 | 0.1077 | 0.3212 | 0.2652 | 0.2032 |
| trend_lowvol | bull | 0.0574 | 0.1832 | 0.3136 | 0.2348 | 0.1936 |
| fund_score | bull | 0.0421 | 0.1401 | 0.3008 | 0.1212 | 0.1686 |
| fund_revenue_growth | bull | 0.0263 | 0.1073 | 0.2451 | 0.1591 | 0.1420 |
| fund_profit_growth | bull | 0.0285 | 0.1150 | 0.2480 | 0.1288 | 0.1400 |
| fund_roe | bull | 0.0361 | 0.1570 | 0.2299 | 0.1591 | 0.1332 |
| fund_gross_margin | bull | 0.0207 | 0.1142 | 0.1811 | 0.1818 | 0.1070 |
| momentum_reversal | bear | 0.1160 | 0.1850 | 0.6271 | 0.4521 | 0.4553 |
| mom_x_lowvol_20_20 | bear | 0.1095 | 0.1873 | 0.5845 | 0.4795 | 0.4324 |
| rsi_vol_combo | bear | 0.0814 | 0.1394 | 0.5837 | 0.4795 | 0.4318 |
| trend_lowvol | bear | 0.0922 | 0.1972 | 0.4675 | 0.2603 | 0.2946 |
| fund_profit_growth | bear | 0.0572 | 0.1431 | 0.3999 | 0.3699 | 0.2739 |
| fund_pb | bear | 0.0738 | 0.1814 | 0.4071 | 0.1507 | 0.2342 |
| fund_pe | bear | 0.0617 | 0.1846 | 0.3341 | 0.1781 | 0.1968 |
| bb_width_20 | bear | 0.0385 | 0.1650 | 0.2336 | 0.1781 | 0.1376 |

### 病原体防治

- **Neutral**: ['fund_pb', 'volatility', 'low_downside'] (单因子IC=0.0884, 组合IC=0.1075)
  - weights: [0.3861, 0.3588, 0.2552]
- **Bull**: ['low_downside', 'volatility', 'fund_pb'] (单因子IC=0.0881, 组合IC=0.1005)
  - bull_weights: [0.3751, 0.3174, 0.3074]
- **Bear**: ['momentum_reversal', 'mom_x_lowvol_20_20', 'fund_pb'] (单因子IC=0.1185, 组合IC=0.1525)
  - bear_weights: [0.4024, 0.3369, 0.2606]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| fund_pb | neutral | 0.0859 | 0.1533 | 0.5603 | 0.4134 | 0.3960 |
| volatility | neutral | 0.1037 | 0.1979 | 0.5238 | 0.4050 | 0.3680 |
| low_downside | neutral | 0.0757 | 0.1929 | 0.3924 | 0.3340 | 0.2617 |
| turnover_stability | neutral | 0.0398 | 0.1156 | 0.3440 | 0.2610 | 0.2169 |
| momentum_reversal | neutral | 0.0591 | 0.1835 | 0.3222 | 0.2025 | 0.1937 |
| trend_lowvol | neutral | 0.0557 | 0.1966 | 0.2831 | 0.1921 | 0.1687 |
| fund_profit_growth | neutral | 0.0376 | 0.1419 | 0.2646 | 0.2380 | 0.1638 |
| mom_x_lowvol_20_20 | neutral | 0.0480 | 0.1775 | 0.2703 | 0.1900 | 0.1608 |
| rsi_vol_combo | neutral | 0.0424 | 0.1728 | 0.2453 | 0.1545 | 0.1416 |
| fund_pe | neutral | 0.0417 | 0.1733 | 0.2405 | 0.1691 | 0.1406 |
| fund_revenue_growth | neutral | 0.0249 | 0.1311 | 0.1898 | 0.1858 | 0.1125 |
| fund_score | neutral | 0.0281 | 0.1732 | 0.1622 | 0.1754 | 0.0953 |
| low_downside | bull | 0.0915 | 0.1490 | 0.6138 | 0.4697 | 0.4511 |
| volatility | bull | 0.0884 | 0.1659 | 0.5332 | 0.4318 | 0.3817 |
| fund_pb | bull | 0.0845 | 0.1610 | 0.5247 | 0.4091 | 0.3697 |
| turnover_stability | bull | 0.0480 | 0.1215 | 0.3952 | 0.3258 | 0.2620 |
| fund_pe | bull | 0.0443 | 0.1693 | 0.2618 | 0.1136 | 0.1458 |
| rsi_vol_combo | bull | 0.0279 | 0.1267 | 0.2203 | 0.1364 | 0.1252 |
| momentum_reversal | bull | 0.0270 | 0.1397 | 0.1931 | 0.1591 | 0.1119 |
| momentum_reversal | bear | 0.1499 | 0.1769 | 0.8478 | 0.6164 | 0.6852 |
| mom_x_lowvol_20_20 | bear | 0.1299 | 0.1737 | 0.7478 | 0.5342 | 0.5737 |
| fund_pb | bear | 0.0757 | 0.1331 | 0.5683 | 0.5616 | 0.4438 |
| trend_lowvol | bear | 0.1097 | 0.1918 | 0.5721 | 0.3973 | 0.3997 |
| rsi_vol_combo | bear | 0.1042 | 0.1909 | 0.5461 | 0.4247 | 0.3890 |
| fund_revenue_growth | bear | 0.0598 | 0.1131 | 0.5288 | 0.3973 | 0.3694 |
| fund_score | bear | 0.0492 | 0.1662 | 0.2963 | 0.3151 | 0.1948 |
| turnover_stability | bear | 0.0332 | 0.1342 | 0.2475 | 0.2055 | 0.1492 |
| fund_profit_growth | bear | 0.0353 | 0.1401 | 0.2517 | 0.1781 | 0.1483 |

### 病毒防治

- **Neutral**: ['fund_pb', 'volatility', 'low_downside'] (单因子IC=0.088, 组合IC=0.1094)
  - weights: [0.4231, 0.3039, 0.273]
- **Bull**: ['low_downside', 'fund_pb'] (单因子IC=0.0995, 组合IC=0.1178)
  - bull_weights: [0.5233, 0.4767]
- **Bear**: ['mom_x_lowvol_20_20'] (单因子IC=0.157, 组合IC=0.157)
  - bear_weights: [1.0]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| fund_pb | neutral | 0.0979 | 0.1743 | 0.5616 | 0.3987 | 0.3927 |
| volatility | neutral | 0.0892 | 0.2176 | 0.4100 | 0.3758 | 0.2821 |
| low_downside | neutral | 0.0768 | 0.2057 | 0.3734 | 0.3570 | 0.2534 |
| turnover_stability | neutral | 0.0419 | 0.1266 | 0.3312 | 0.2714 | 0.2105 |
| fund_pe | neutral | 0.0575 | 0.1726 | 0.3329 | 0.2505 | 0.2082 |
| trend_lowvol | neutral | 0.0602 | 0.2144 | 0.2807 | 0.1775 | 0.1653 |
| fund_profit_growth | neutral | 0.0417 | 0.1598 | 0.2608 | 0.2484 | 0.1628 |
| momentum_reversal | neutral | 0.0496 | 0.1936 | 0.2560 | 0.1712 | 0.1499 |
| mom_x_lowvol_20_20 | neutral | 0.0389 | 0.1824 | 0.2135 | 0.1461 | 0.1223 |
| fund_revenue_growth | neutral | 0.0270 | 0.1499 | 0.1801 | 0.1441 | 0.1030 |
| rsi_vol_combo | neutral | 0.0312 | 0.1864 | 0.1673 | 0.1086 | 0.0927 |
| fund_score | neutral | 0.0284 | 0.1872 | 0.1518 | 0.1253 | 0.0854 |
| low_downside | bull | 0.0958 | 0.1642 | 0.5833 | 0.4470 | 0.4220 |
| fund_pb | bull | 0.1031 | 0.1870 | 0.5515 | 0.3939 | 0.3844 |
| volatility | bull | 0.0990 | 0.1852 | 0.5345 | 0.4015 | 0.3745 |
| turnover_stability | bull | 0.0629 | 0.1219 | 0.5157 | 0.4167 | 0.3653 |
| fund_pe | bull | 0.0605 | 0.1717 | 0.3522 | 0.2045 | 0.2121 |
| momentum_reversal | bull | 0.0401 | 0.1735 | 0.2309 | 0.1364 | 0.1312 |
| trend_lowvol | bull | 0.0452 | 0.2007 | 0.2255 | 0.1364 | 0.1281 |
| mom_x_lowvol_20_20 | bull | 0.0315 | 0.1663 | 0.1896 | 0.1818 | 0.1120 |
| mom_x_lowvol_20_20 | bear | 0.1570 | 0.1967 | 0.7984 | 0.7534 | 0.7000 |
| momentum_reversal | bear | 0.1408 | 0.1888 | 0.7460 | 0.5616 | 0.5825 |
| fund_pb | bear | 0.0844 | 0.1491 | 0.5657 | 0.4521 | 0.4107 |
| trend_lowvol | bear | 0.1097 | 0.1976 | 0.5551 | 0.3425 | 0.3726 |
| rsi_vol_combo | bear | 0.0814 | 0.1842 | 0.4421 | 0.3973 | 0.3088 |
| fund_gross_margin | bear | 0.0471 | 0.1530 | 0.3080 | 0.1507 | 0.1772 |
| fund_pe | bear | 0.0403 | 0.1843 | 0.2188 | 0.1781 | 0.1289 |

### 白酒

- **Neutral**: ['trend_lowvol', 'mom_x_lowvol_20_20', 'turnover_stability'] (单因子IC=0.0588, 组合IC=0.0896)
  - weights: [0.3804, 0.3399, 0.2798]
- **Bull**: ['trend_lowvol'] (单因子IC=0.0828, 组合IC=0.0828)
  - bull_weights: [1.0]
- **Bear**: ['mom_x_lowvol_20_20', 'fund_pb', 'momentum_reversal'] (单因子IC=0.1681, 组合IC=0.2386)
  - bear_weights: [0.404, 0.3043, 0.2918]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| trend_lowvol | neutral | 0.0760 | 0.2549 | 0.2981 | 0.2286 | 0.1831 |
| mom_x_lowvol_20_20 | neutral | 0.0564 | 0.2067 | 0.2729 | 0.1994 | 0.1636 |
| turnover_stability | neutral | 0.0439 | 0.1988 | 0.2210 | 0.2192 | 0.1347 |
| low_downside | neutral | 0.0488 | 0.2317 | 0.2107 | 0.2307 | 0.1297 |
| fund_pb | neutral | 0.0646 | 0.3176 | 0.2034 | 0.1649 | 0.1185 |
| momentum_reversal | neutral | 0.0446 | 0.2283 | 0.1952 | 0.1420 | 0.1114 |
| rsi_vol_combo | neutral | 0.0393 | 0.2253 | 0.1746 | 0.1858 | 0.1035 |
| fund_profit_growth | neutral | 0.0430 | 0.2341 | 0.1837 | 0.1169 | 0.1026 |
| volatility | neutral | 0.0326 | 0.1963 | 0.1663 | 0.1566 | 0.0961 |
| fund_pe | neutral | 0.0381 | 0.2354 | 0.1620 | 0.1555 | 0.0936 |
| fund_revenue_growth | neutral | 0.0280 | 0.2145 | 0.1306 | 0.1106 | 0.0725 |
| trend_lowvol | bull | 0.0828 | 0.2565 | 0.3226 | 0.2500 | 0.2016 |
| volatility | bull | 0.0470 | 0.1885 | 0.2491 | 0.2121 | 0.1510 |
| low_downside | bull | 0.0399 | 0.2106 | 0.1895 | 0.1894 | 0.1127 |
| fund_pb | bull | 0.0393 | 0.2700 | 0.1455 | 0.1364 | 0.0827 |
| fund_pe | bull | 0.0279 | 0.2518 | 0.1109 | 0.1061 | 0.0613 |
| mom_x_lowvol_20_20 | bear | 0.1560 | 0.2182 | 0.7149 | 0.4795 | 0.5288 |
| fund_pb | bear | 0.1910 | 0.3415 | 0.5592 | 0.4247 | 0.3983 |
| momentum_reversal | bear | 0.1574 | 0.2935 | 0.5362 | 0.4247 | 0.3819 |
| rsi_vol_combo | bear | 0.1125 | 0.2760 | 0.4075 | 0.1781 | 0.2400 |
| trend_lowvol | bear | 0.0857 | 0.2898 | 0.2958 | 0.3151 | 0.1945 |
| fund_pe | bear | 0.0695 | 0.2843 | 0.2445 | 0.2055 | 0.1474 |
| bb_width_20 | bear | 0.0570 | 0.2857 | 0.1995 | 0.2055 | 0.1203 |

### 百度概念

- **Neutral**: ['momentum_reversal', 'fund_pb', 'trend_lowvol'] (单因子IC=0.0941, 组合IC=0.1278)
  - weights: [0.3489, 0.3411, 0.31]
- **Bull**: ['volatility', 'trend_lowvol', 'fund_pb'] (单因子IC=0.0953, 组合IC=0.1122)
  - bull_weights: [0.3779, 0.34, 0.2821]
- **Bear**: ['trend_lowvol', 'mom_x_lowvol_20_20'] (单因子IC=0.107, 组合IC=0.1203)
  - bear_weights: [0.5945, 0.4055]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| momentum_reversal | neutral | 0.1013 | 0.1765 | 0.5739 | 0.3946 | 0.4002 |
| fund_pb | neutral | 0.0788 | 0.1408 | 0.5594 | 0.3987 | 0.3912 |
| trend_lowvol | neutral | 0.1023 | 0.1995 | 0.5130 | 0.3862 | 0.3556 |
| mom_x_lowvol_20_20 | neutral | 0.0917 | 0.1751 | 0.5239 | 0.3528 | 0.3544 |
| volatility | neutral | 0.0920 | 0.2012 | 0.4574 | 0.4029 | 0.3209 |
| rsi_vol_combo | neutral | 0.0814 | 0.1710 | 0.4759 | 0.3278 | 0.3159 |
| turnover_stability | neutral | 0.0387 | 0.1177 | 0.3291 | 0.2610 | 0.2075 |
| low_downside | neutral | 0.0565 | 0.1907 | 0.2964 | 0.2881 | 0.1909 |
| fund_profit_growth | neutral | 0.0374 | 0.1316 | 0.2838 | 0.1921 | 0.1691 |
| fund_pe | neutral | 0.0414 | 0.1539 | 0.2691 | 0.2255 | 0.1649 |
| fund_score | neutral | 0.0403 | 0.1531 | 0.2630 | 0.1545 | 0.1518 |
| fund_revenue_growth | neutral | 0.0252 | 0.1265 | 0.1993 | 0.1357 | 0.1132 |
| fund_roe | neutral | 0.0207 | 0.1573 | 0.1315 | 0.1002 | 0.0724 |
| volatility | bull | 0.1042 | 0.1707 | 0.6103 | 0.5833 | 0.4831 |
| trend_lowvol | bull | 0.1003 | 0.1705 | 0.5885 | 0.4773 | 0.4347 |
| fund_pb | bull | 0.0815 | 0.1566 | 0.5203 | 0.3864 | 0.3607 |
| low_downside | bull | 0.0759 | 0.1584 | 0.4789 | 0.3864 | 0.3320 |
| mom_x_lowvol_20_20 | bull | 0.0699 | 0.1637 | 0.4272 | 0.3712 | 0.2929 |
| momentum_reversal | bull | 0.0647 | 0.1603 | 0.4037 | 0.3182 | 0.2661 |
| fund_profit_growth | bull | 0.0376 | 0.1182 | 0.3183 | 0.2576 | 0.2001 |
| fund_pe | bull | 0.0491 | 0.1615 | 0.3042 | 0.2879 | 0.1959 |
| fund_score | bull | 0.0472 | 0.1513 | 0.3120 | 0.2348 | 0.1926 |
| turnover_stability | bull | 0.0278 | 0.1004 | 0.2767 | 0.2803 | 0.1771 |
| fund_revenue_growth | bull | 0.0312 | 0.1101 | 0.2838 | 0.2197 | 0.1731 |
| fund_roe | bull | 0.0311 | 0.1481 | 0.2101 | 0.1742 | 0.1234 |
| rsi_vol_combo | bull | 0.0292 | 0.1556 | 0.1876 | 0.1515 | 0.1080 |
| wash_sale_score | bull | 0.0139 | 0.0926 | 0.1501 | 0.1692 | 0.0878 |
| trend_lowvol | bear | 0.1118 | 0.1665 | 0.6713 | 0.4795 | 0.4966 |
| mom_x_lowvol_20_20 | bear | 0.1022 | 0.1983 | 0.5152 | 0.3151 | 0.3387 |
| momentum_reversal | bear | 0.0989 | 0.2060 | 0.4799 | 0.3425 | 0.3221 |
| fund_revenue_growth | bear | 0.0563 | 0.1419 | 0.3966 | 0.1781 | 0.2336 |
| turnover_stability | bear | 0.0434 | 0.1212 | 0.3578 | 0.2603 | 0.2255 |
| fund_profit_growth | bear | 0.0420 | 0.1200 | 0.3496 | 0.2329 | 0.2155 |
| rsi_vol_combo | bear | 0.0463 | 0.1840 | 0.2515 | 0.2877 | 0.1619 |

### 百日新高

- **Neutral**: ['trend_lowvol'] (单因子IC=0.091, 组合IC=0.091)
  - weights: [1.0]
- **Bull**: ['volatility', 'low_downside', 'fund_pb'] (单因子IC=0.0637, 组合IC=0.0773)
  - bull_weights: [0.3605, 0.3512, 0.2883]
- **Bear**: ['rsi_vol_combo', 'momentum_reversal', 'mom_x_lowvol_20_20'] (单因子IC=0.0886, 组合IC=0.0937)
  - bear_weights: [0.3494, 0.3254, 0.3252]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| trend_lowvol | neutral | 0.0910 | 0.1872 | 0.4861 | 0.3633 | 0.3313 |
| momentum_reversal | neutral | 0.0738 | 0.1800 | 0.4099 | 0.3090 | 0.2683 |
| mom_x_lowvol_20_20 | neutral | 0.0716 | 0.1755 | 0.4083 | 0.2923 | 0.2638 |
| volatility | neutral | 0.0558 | 0.1778 | 0.3138 | 0.2777 | 0.2005 |
| fund_pb | neutral | 0.0543 | 0.1691 | 0.3211 | 0.2422 | 0.1995 |
| fund_profit_growth | neutral | 0.0474 | 0.1598 | 0.2967 | 0.2046 | 0.1787 |
| rsi_vol_combo | neutral | 0.0500 | 0.1741 | 0.2873 | 0.2088 | 0.1737 |
| fund_pe | neutral | 0.0426 | 0.1598 | 0.2665 | 0.2651 | 0.1686 |
| turnover_stability | neutral | 0.0295 | 0.1230 | 0.2397 | 0.2547 | 0.1504 |
| fund_revenue_growth | neutral | 0.0325 | 0.1544 | 0.2107 | 0.1608 | 0.1223 |
| low_downside | neutral | 0.0289 | 0.1904 | 0.1520 | 0.1503 | 0.0874 |
| volatility | bull | 0.0727 | 0.1794 | 0.4053 | 0.2727 | 0.2579 |
| low_downside | bull | 0.0604 | 0.1520 | 0.3973 | 0.2652 | 0.2513 |
| fund_pb | bull | 0.0579 | 0.1819 | 0.3185 | 0.2955 | 0.2063 |
| fund_pe | bull | 0.0411 | 0.1622 | 0.2537 | 0.2652 | 0.1605 |
| mom_x_lowvol_20_20 | bull | 0.0408 | 0.1612 | 0.2532 | 0.1439 | 0.1448 |
| trend_lowvol | bull | 0.0416 | 0.1904 | 0.2184 | 0.1515 | 0.1258 |
| turnover_stability | bull | 0.0246 | 0.1190 | 0.2068 | 0.1818 | 0.1222 |
| fund_gross_margin | bull | 0.0266 | 0.1318 | 0.2021 | 0.1970 | 0.1210 |
| rsi_vol_combo | bull | 0.0326 | 0.1603 | 0.2037 | 0.1742 | 0.1196 |
| fund_profit_growth | bull | 0.0311 | 0.1634 | 0.1906 | 0.1364 | 0.1083 |
| rsi_vol_combo | bear | 0.0765 | 0.1655 | 0.4619 | 0.4521 | 0.3353 |
| momentum_reversal | bear | 0.0922 | 0.1982 | 0.4652 | 0.3425 | 0.3122 |
| mom_x_lowvol_20_20 | bear | 0.0972 | 0.2048 | 0.4746 | 0.3151 | 0.3121 |
| trend_lowvol | bear | 0.0875 | 0.2031 | 0.4306 | 0.2877 | 0.2772 |
| turnover_stability | bear | 0.0420 | 0.0980 | 0.4291 | 0.2877 | 0.2763 |
| fund_profit_growth | bear | 0.0519 | 0.1502 | 0.3454 | 0.2603 | 0.2177 |

### 知识产权

- **Neutral**: ['fund_pb', 'volatility', 'trend_lowvol'] (单因子IC=0.1001, 组合IC=0.1247)
  - weights: [0.3872, 0.3132, 0.2996]
- **Bull**: ['trend_lowvol', 'fund_pb', 'volatility'] (单因子IC=0.1157, 组合IC=0.1408)
  - bull_weights: [0.3662, 0.3197, 0.3141]
- **Bear**: ['fund_revenue_growth', 'momentum_reversal'] (单因子IC=0.132, 组合IC=0.1252)
  - bear_weights: [0.5258, 0.4742]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| fund_pb | neutral | 0.0966 | 0.1986 | 0.4863 | 0.3549 | 0.3294 |
| volatility | neutral | 0.1065 | 0.2525 | 0.4219 | 0.2630 | 0.2664 |
| trend_lowvol | neutral | 0.0973 | 0.2507 | 0.3881 | 0.3132 | 0.2548 |
| momentum_reversal | neutral | 0.0956 | 0.2422 | 0.3948 | 0.2829 | 0.2532 |
| mom_x_lowvol_20_20 | neutral | 0.0855 | 0.2317 | 0.3687 | 0.2818 | 0.2363 |
| rsi_vol_combo | neutral | 0.0830 | 0.2419 | 0.3430 | 0.2526 | 0.2148 |
| low_downside | neutral | 0.0713 | 0.2193 | 0.3250 | 0.2474 | 0.2027 |
| fund_score | neutral | 0.0508 | 0.1979 | 0.2565 | 0.2192 | 0.1564 |
| fund_pe | neutral | 0.0555 | 0.2213 | 0.2506 | 0.2276 | 0.1538 |
| fund_roe | neutral | 0.0488 | 0.1978 | 0.2465 | 0.1649 | 0.1436 |
| fund_profit_growth | neutral | 0.0374 | 0.1868 | 0.1999 | 0.1211 | 0.1121 |
| fund_revenue_growth | neutral | 0.0254 | 0.1972 | 0.1287 | 0.1023 | 0.0709 |
| trend_lowvol | bull | 0.1273 | 0.2153 | 0.5914 | 0.4848 | 0.4391 |
| fund_pb | bull | 0.1070 | 0.1935 | 0.5531 | 0.3864 | 0.3834 |
| volatility | bull | 0.1127 | 0.2232 | 0.5047 | 0.4924 | 0.3766 |
| low_downside | bull | 0.0844 | 0.1954 | 0.4321 | 0.3182 | 0.2848 |
| momentum_reversal | bull | 0.0905 | 0.2112 | 0.4284 | 0.3258 | 0.2840 |
| mom_x_lowvol_20_20 | bull | 0.0882 | 0.2079 | 0.4241 | 0.3333 | 0.2827 |
| fund_pe | bull | 0.0691 | 0.2210 | 0.3128 | 0.2727 | 0.1991 |
| rsi_vol_combo | bull | 0.0638 | 0.2195 | 0.2907 | 0.1742 | 0.1707 |
| fund_profit_growth | bull | 0.0365 | 0.1504 | 0.2426 | 0.1818 | 0.1434 |
| turnover_stability | bull | 0.0304 | 0.1676 | 0.1816 | 0.1288 | 0.1025 |
| stroke_phase | bull | 0.0279 | 0.2049 | 0.1362 | 0.1212 | 0.0764 |
| wash_sale_score | bull | 0.0246 | 0.1828 | 0.1345 | 0.1218 | 0.0754 |
| fund_revenue_growth | bear | 0.1187 | 0.1979 | 0.5996 | 0.4795 | 0.4436 |
| momentum_reversal | bear | 0.1453 | 0.2488 | 0.5842 | 0.3699 | 0.4001 |
| mom_x_lowvol_20_20 | bear | 0.1236 | 0.2384 | 0.5185 | 0.3699 | 0.3551 |
| trend_lowvol | bear | 0.1265 | 0.2440 | 0.5186 | 0.3151 | 0.3410 |
| rsi_vol_combo | bear | 0.1224 | 0.2611 | 0.4688 | 0.3151 | 0.3083 |
| bb_width_20 | bear | 0.0722 | 0.2594 | 0.2782 | 0.2877 | 0.1791 |

### 短剧互动游戏

- **Neutral**: ['volatility', 'fund_pb', 'trend_lowvol'] (单因子IC=0.1152, 组合IC=0.1378)
  - weights: [0.3489, 0.3409, 0.3102]
- **Bull**: ['trend_lowvol', 'volatility'] (单因子IC=0.1137, 组合IC=0.123)
  - bull_weights: [0.5611, 0.4389]
- **Bear**: ['trend_lowvol', 'momentum_reversal'] (单因子IC=0.1622, 组合IC=0.1787)
  - bear_weights: [0.5508, 0.4492]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| volatility | neutral | 0.1318 | 0.2737 | 0.4817 | 0.3445 | 0.3238 |
| fund_pb | neutral | 0.0997 | 0.2125 | 0.4693 | 0.3486 | 0.3164 |
| trend_lowvol | neutral | 0.1139 | 0.2622 | 0.4345 | 0.3257 | 0.2880 |
| momentum_reversal | neutral | 0.0983 | 0.2520 | 0.3902 | 0.2839 | 0.2505 |
| rsi_vol_combo | neutral | 0.0882 | 0.2316 | 0.3806 | 0.3027 | 0.2479 |
| low_downside | neutral | 0.0903 | 0.2592 | 0.3484 | 0.3058 | 0.2275 |
| mom_x_lowvol_20_20 | neutral | 0.0894 | 0.2528 | 0.3536 | 0.2630 | 0.2233 |
| fund_pe | neutral | 0.0633 | 0.2474 | 0.2560 | 0.2192 | 0.1560 |
| turnover_stability | neutral | 0.0538 | 0.2173 | 0.2477 | 0.2119 | 0.1501 |
| fund_profit_growth | neutral | 0.0460 | 0.2082 | 0.2208 | 0.1649 | 0.1286 |
| fund_score | neutral | 0.0423 | 0.2316 | 0.1827 | 0.1921 | 0.1089 |
| trend_lowvol | bull | 0.1215 | 0.2324 | 0.5226 | 0.3712 | 0.3583 |
| volatility | bull | 0.1059 | 0.2390 | 0.4431 | 0.2652 | 0.2803 |
| fund_pb | bull | 0.0789 | 0.2093 | 0.3770 | 0.2955 | 0.2442 |
| low_downside | bull | 0.0913 | 0.2324 | 0.3928 | 0.2424 | 0.2440 |
| fund_roe | bull | 0.0522 | 0.1808 | 0.2887 | 0.1742 | 0.1695 |
| mom_x_lowvol_20_20 | bull | 0.0638 | 0.2390 | 0.2668 | 0.1970 | 0.1597 |
| fund_pe | bull | 0.0558 | 0.2265 | 0.2465 | 0.1818 | 0.1457 |
| fund_revenue_growth | bull | 0.0515 | 0.2065 | 0.2495 | 0.1591 | 0.1446 |
| momentum_reversal | bull | 0.0633 | 0.2498 | 0.2536 | 0.1364 | 0.1441 |
| fund_score | bull | 0.0464 | 0.1953 | 0.2374 | 0.1742 | 0.1394 |
| turnover_stability | bull | 0.0460 | 0.1956 | 0.2353 | 0.1288 | 0.1328 |
| rsi_vol_combo | bull | 0.0344 | 0.2243 | 0.1532 | 0.1288 | 0.0865 |
| trend_lowvol | bear | 0.1595 | 0.2305 | 0.6920 | 0.6438 | 0.5687 |
| momentum_reversal | bear | 0.1648 | 0.2629 | 0.6269 | 0.4795 | 0.4638 |
| mom_x_lowvol_20_20 | bear | 0.1539 | 0.2886 | 0.5335 | 0.3151 | 0.3508 |
| fund_revenue_growth | bear | 0.0993 | 0.2473 | 0.4013 | 0.3151 | 0.2639 |
| rsi_vol_combo | bear | 0.0990 | 0.2526 | 0.3920 | 0.2877 | 0.2524 |
| turnover_stability | bear | 0.0545 | 0.1972 | 0.2763 | 0.1781 | 0.1628 |

### 石墨烯

- **Neutral**: ['trend_lowvol', 'momentum_reversal', 'volatility'] (单因子IC=0.0763, 组合IC=0.0937)
  - weights: [0.3558, 0.3224, 0.3218]
- **Bull**: ['low_downside', 'volatility', 'turnover_stability'] (单因子IC=0.0773, 组合IC=0.1021)
  - bull_weights: [0.3729, 0.3638, 0.2633]
- **Bear**: ['momentum_reversal'] (单因子IC=0.129, 组合IC=0.129)
  - bear_weights: [1.0]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| trend_lowvol | neutral | 0.0869 | 0.2276 | 0.3818 | 0.2630 | 0.2411 |
| momentum_reversal | neutral | 0.0678 | 0.2000 | 0.3393 | 0.2881 | 0.2185 |
| volatility | neutral | 0.0741 | 0.2214 | 0.3348 | 0.3027 | 0.2181 |
| mom_x_lowvol_20_20 | neutral | 0.0659 | 0.1984 | 0.3319 | 0.3027 | 0.2162 |
| fund_profit_growth | neutral | 0.0639 | 0.1812 | 0.3527 | 0.2109 | 0.2135 |
| fund_pe | neutral | 0.0579 | 0.1771 | 0.3271 | 0.2547 | 0.2052 |
| fund_pb | neutral | 0.0573 | 0.2048 | 0.2799 | 0.1994 | 0.1679 |
| rsi_vol_combo | neutral | 0.0487 | 0.1860 | 0.2618 | 0.1879 | 0.1555 |
| fund_score | neutral | 0.0487 | 0.2122 | 0.2296 | 0.1608 | 0.1333 |
| turnover_stability | neutral | 0.0315 | 0.1477 | 0.2133 | 0.2109 | 0.1291 |
| low_downside | neutral | 0.0315 | 0.2024 | 0.1559 | 0.1430 | 0.0891 |
| low_downside | bull | 0.0821 | 0.1687 | 0.4863 | 0.4545 | 0.3536 |
| volatility | bull | 0.0942 | 0.1902 | 0.4950 | 0.3939 | 0.3450 |
| turnover_stability | bull | 0.0558 | 0.1421 | 0.3925 | 0.2727 | 0.2497 |
| trend_lowvol | bull | 0.0742 | 0.1932 | 0.3840 | 0.1970 | 0.2298 |
| fund_pb | bull | 0.0677 | 0.2113 | 0.3204 | 0.2727 | 0.2039 |
| fund_pe | bull | 0.0453 | 0.1976 | 0.2294 | 0.1667 | 0.1338 |
| mom_x_lowvol_20_20 | bull | 0.0411 | 0.1965 | 0.2089 | 0.1742 | 0.1227 |
| rsi_vol_combo | bull | 0.0373 | 0.1819 | 0.2049 | 0.1515 | 0.1180 |
| momentum_reversal | bull | 0.0366 | 0.1944 | 0.1884 | 0.1591 | 0.1092 |
| fund_gross_margin | bull | 0.0178 | 0.1478 | 0.1203 | 0.1970 | 0.0720 |
| momentum_reversal | bear | 0.1290 | 0.1775 | 0.7268 | 0.5068 | 0.5476 |
| mom_x_lowvol_20_20 | bear | 0.1261 | 0.1961 | 0.6427 | 0.4247 | 0.4578 |
| rsi_vol_combo | bear | 0.0816 | 0.1597 | 0.5113 | 0.4521 | 0.3712 |
| fund_pe | bear | 0.0602 | 0.1481 | 0.4063 | 0.2603 | 0.2560 |
| fund_revenue_growth | bear | 0.0608 | 0.1888 | 0.3220 | 0.1507 | 0.1852 |
| turnover_stability | bear | 0.0546 | 0.1712 | 0.3190 | 0.1507 | 0.1835 |
| trend_lowvol | bear | 0.0643 | 0.2226 | 0.2889 | 0.2603 | 0.1821 |
| bb_width_20 | bear | 0.0477 | 0.2055 | 0.2322 | 0.1233 | 0.1304 |

### 碳交易

- **Neutral**: ['momentum_reversal', 'mom_x_lowvol_20_20', 'rsi_vol_combo'] (单因子IC=0.0839, 组合IC=0.0933)
  - weights: [0.364, 0.3286, 0.3075]
- **Bull**: ['low_downside', 'turnover_stability'] (单因子IC=0.0662, 组合IC=0.0861)
  - bull_weights: [0.5482, 0.4518]
- **Bear**: ['trend_lowvol', 'momentum_reversal'] (单因子IC=0.1045, 组合IC=0.1223)
  - bear_weights: [0.5511, 0.4489]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| momentum_reversal | neutral | 0.0920 | 0.1779 | 0.5171 | 0.3925 | 0.3600 |
| mom_x_lowvol_20_20 | neutral | 0.0830 | 0.1745 | 0.4753 | 0.3674 | 0.3250 |
| rsi_vol_combo | neutral | 0.0766 | 0.1693 | 0.4524 | 0.3445 | 0.3041 |
| fund_profit_growth | neutral | 0.0467 | 0.1217 | 0.3835 | 0.3006 | 0.2494 |
| trend_lowvol | neutral | 0.0797 | 0.2079 | 0.3835 | 0.2777 | 0.2450 |
| turnover_stability | neutral | 0.0423 | 0.1171 | 0.3616 | 0.3048 | 0.2359 |
| volatility | neutral | 0.0696 | 0.2104 | 0.3310 | 0.3111 | 0.2170 |
| fund_pb | neutral | 0.0570 | 0.1669 | 0.3413 | 0.2505 | 0.2134 |
| fund_score | neutral | 0.0406 | 0.1573 | 0.2584 | 0.1608 | 0.1500 |
| fund_pe | neutral | 0.0492 | 0.1928 | 0.2554 | 0.1566 | 0.1477 |
| low_downside | neutral | 0.0286 | 0.2040 | 0.1400 | 0.1441 | 0.0801 |
| fund_revenue_growth | neutral | 0.0156 | 0.1132 | 0.1376 | 0.1023 | 0.0758 |
| low_downside | bull | 0.0853 | 0.1583 | 0.5391 | 0.3864 | 0.3737 |
| turnover_stability | bull | 0.0472 | 0.1050 | 0.4492 | 0.3712 | 0.3080 |
| volatility | bull | 0.0680 | 0.1635 | 0.4156 | 0.3409 | 0.2787 |
| fund_pb | bull | 0.0672 | 0.1765 | 0.3809 | 0.2500 | 0.2380 |
| momentum_reversal | bull | 0.0396 | 0.1576 | 0.2511 | 0.1970 | 0.1503 |
| trend_lowvol | bull | 0.0444 | 0.1743 | 0.2549 | 0.1136 | 0.1419 |
| fund_revenue_growth | bull | 0.0181 | 0.0942 | 0.1917 | 0.1970 | 0.1147 |
| mom_x_lowvol_20_20 | bull | 0.0242 | 0.1447 | 0.1675 | 0.1970 | 0.1002 |
| rsi_vol_combo | bull | 0.0232 | 0.1482 | 0.1566 | 0.1515 | 0.0902 |
| fund_score | bull | 0.0202 | 0.1380 | 0.1463 | 0.1364 | 0.0831 |
| fund_profit_growth | bull | 0.0113 | 0.1117 | 0.1015 | 0.1288 | 0.0573 |
| trend_lowvol | bear | 0.1067 | 0.1783 | 0.5981 | 0.4247 | 0.4261 |
| momentum_reversal | bear | 0.1024 | 0.2102 | 0.4872 | 0.4247 | 0.3470 |
| mom_x_lowvol_20_20 | bear | 0.0962 | 0.2120 | 0.4538 | 0.3699 | 0.3109 |
| rsi_vol_combo | bear | 0.0714 | 0.1707 | 0.4184 | 0.3151 | 0.2751 |
| bb_width_20 | bear | 0.0576 | 0.1776 | 0.3243 | 0.3151 | 0.2132 |
| fund_revenue_growth | bear | 0.0381 | 0.1126 | 0.3381 | 0.2055 | 0.2038 |

### 碳化硅

- **Neutral**: ['trend_lowvol', 'momentum_reversal'] (单因子IC=0.0647, 组合IC=0.0754)
  - weights: [0.5221, 0.4779]
- **Bull**: ['volatility', 'low_downside'] (单因子IC=0.0638, 组合IC=0.0685)
  - bull_weights: [0.5101, 0.4899]
- **Bear**: ['trend_lowvol', 'mom_x_lowvol_20_20', 'momentum_reversal'] (单因子IC=0.1571, 组合IC=0.161)
  - bear_weights: [0.3519, 0.3511, 0.297]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| trend_lowvol | neutral | 0.0684 | 0.2150 | 0.3182 | 0.2505 | 0.1989 |
| momentum_reversal | neutral | 0.0610 | 0.2067 | 0.2952 | 0.2338 | 0.1821 |
| mom_x_lowvol_20_20 | neutral | 0.0574 | 0.2031 | 0.2825 | 0.2578 | 0.1777 |
| rsi_vol_combo | neutral | 0.0550 | 0.2053 | 0.2678 | 0.2255 | 0.1641 |
| fund_pb | neutral | 0.0584 | 0.2337 | 0.2498 | 0.2505 | 0.1562 |
| volatility | neutral | 0.0511 | 0.2081 | 0.2455 | 0.1879 | 0.1458 |
| fund_pe | neutral | 0.0366 | 0.2152 | 0.1699 | 0.1096 | 0.0942 |
| low_downside | neutral | 0.0332 | 0.2130 | 0.1560 | 0.1482 | 0.0896 |
| volatility | bull | 0.0632 | 0.1835 | 0.3448 | 0.3106 | 0.2259 |
| low_downside | bull | 0.0644 | 0.1844 | 0.3493 | 0.2424 | 0.2170 |
| trend_lowvol | bull | 0.0573 | 0.2269 | 0.2526 | 0.1894 | 0.1502 |
| momentum_reversal | bull | 0.0467 | 0.2065 | 0.2259 | 0.1515 | 0.1301 |
| wash_sale_score | bull | 0.0370 | 0.1675 | 0.2208 | 0.1660 | 0.1287 |
| mom_x_lowvol_20_20 | bull | 0.0410 | 0.1974 | 0.2077 | 0.1970 | 0.1243 |
| fund_pb | bull | 0.0390 | 0.2210 | 0.1766 | 0.1326 | 0.1000 |
| rsi_vol_combo | bull | 0.0357 | 0.2351 | 0.1517 | 0.1667 | 0.0885 |
| trend_lowvol | bear | 0.1530 | 0.2032 | 0.7528 | 0.5068 | 0.5672 |
| mom_x_lowvol_20_20 | bear | 0.1681 | 0.2319 | 0.7247 | 0.5616 | 0.5658 |
| momentum_reversal | bear | 0.1504 | 0.2367 | 0.6352 | 0.5068 | 0.4786 |
| rsi_vol_combo | bear | 0.1143 | 0.2001 | 0.5710 | 0.4521 | 0.4146 |
| fund_profit_growth | bear | 0.0833 | 0.2210 | 0.3772 | 0.3151 | 0.2480 |
| bb_width_20 | bear | 0.0666 | 0.1959 | 0.3398 | 0.2877 | 0.2188 |
| fund_pe | bear | 0.0754 | 0.2318 | 0.3253 | 0.2603 | 0.2050 |

### 碳纤维

- **Neutral**: ['fund_pb', 'momentum_reversal', 'mom_x_lowvol_20_20'] (单因子IC=0.0796, 组合IC=0.1301)
  - weights: [0.372, 0.3299, 0.2981]
- **Bull**: ['low_downside', 'fund_pb', 'volatility'] (单因子IC=0.1366, 组合IC=0.1824)
  - bull_weights: [0.3833, 0.3605, 0.2562]
- **Bear**: ['momentum_reversal', 'turnover_stability', 'trend_lowvol'] (单因子IC=0.1032, 组合IC=0.152)
  - bear_weights: [0.3922, 0.3536, 0.2542]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| fund_pb | neutral | 0.0947 | 0.2720 | 0.3483 | 0.2589 | 0.2192 |
| momentum_reversal | neutral | 0.0776 | 0.2415 | 0.3211 | 0.2109 | 0.1944 |
| mom_x_lowvol_20_20 | neutral | 0.0665 | 0.2341 | 0.2838 | 0.2380 | 0.1757 |
| trend_lowvol | neutral | 0.0709 | 0.2542 | 0.2790 | 0.2004 | 0.1675 |
| rsi_vol_combo | neutral | 0.0561 | 0.2274 | 0.2465 | 0.1722 | 0.1445 |
| fund_profit_growth | neutral | 0.0471 | 0.2177 | 0.2165 | 0.2182 | 0.1319 |
| fund_score | neutral | 0.0407 | 0.2120 | 0.1920 | 0.2317 | 0.1182 |
| volatility | neutral | 0.0490 | 0.2513 | 0.1950 | 0.1795 | 0.1150 |
| fund_pe | neutral | 0.0335 | 0.2114 | 0.1584 | 0.1775 | 0.0933 |
| fund_revenue_growth | neutral | 0.0273 | 0.1916 | 0.1428 | 0.1211 | 0.0800 |
| turnover_stability | neutral | 0.0209 | 0.1925 | 0.1088 | 0.1086 | 0.0603 |
| low_downside | neutral | 0.0180 | 0.2455 | 0.0733 | 0.1127 | 0.0408 |
| low_downside | bull | 0.1432 | 0.2000 | 0.7159 | 0.5985 | 0.5722 |
| fund_pb | bull | 0.1527 | 0.2257 | 0.6765 | 0.5909 | 0.5381 |
| volatility | bull | 0.1138 | 0.2187 | 0.5203 | 0.4697 | 0.3823 |
| trend_lowvol | bull | 0.0960 | 0.2436 | 0.3939 | 0.3333 | 0.2626 |
| fund_pe | bull | 0.0626 | 0.2012 | 0.3109 | 0.2045 | 0.1873 |
| momentum_reversal | bull | 0.0656 | 0.2230 | 0.2943 | 0.2576 | 0.1851 |
| mom_x_lowvol_20_20 | bull | 0.0587 | 0.2239 | 0.2622 | 0.2424 | 0.1629 |
| stroke_phase | bull | 0.0222 | 0.1762 | 0.1260 | 0.1553 | 0.0728 |
| momentum_reversal | bear | 0.1216 | 0.2435 | 0.4995 | 0.4521 | 0.3626 |
| turnover_stability | bear | 0.0747 | 0.1660 | 0.4502 | 0.4521 | 0.3269 |
| trend_lowvol | bear | 0.1132 | 0.2971 | 0.3812 | 0.2329 | 0.2350 |
| mom_x_lowvol_20_20 | bear | 0.0937 | 0.2614 | 0.3583 | 0.2603 | 0.2258 |
| fund_revenue_growth | bear | 0.0728 | 0.2167 | 0.3358 | 0.1233 | 0.1886 |
| fund_pe | bear | 0.0639 | 0.2308 | 0.2769 | 0.3151 | 0.1820 |
| bb_width_20 | bear | 0.0730 | 0.2393 | 0.3052 | 0.1781 | 0.1797 |
| rsi_vol_combo | bear | 0.0546 | 0.2011 | 0.2717 | 0.2055 | 0.1638 |
| fund_profit_growth | bear | 0.0470 | 0.2097 | 0.2242 | 0.1781 | 0.1321 |

### 磁悬浮概念

- **Neutral**: ['mom_x_lowvol_20_20', 'trend_lowvol'] (单因子IC=0.0809, 组合IC=0.1026)
  - weights: [0.5084, 0.4916]
- **Bull**: ['volatility', 'fund_pe', 'low_downside'] (单因子IC=0.1081, 组合IC=0.1393)
  - bull_weights: [0.3765, 0.3179, 0.3056]
- **Bear**: ['mom_x_lowvol_20_20', 'momentum_reversal', 'fund_pb'] (单因子IC=0.0977, 组合IC=0.1607)
  - bear_weights: [0.5084, 0.2888, 0.2028]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| mom_x_lowvol_20_20 | neutral | 0.0781 | 0.2742 | 0.2847 | 0.2140 | 0.1728 |
| trend_lowvol | neutral | 0.0837 | 0.3045 | 0.2748 | 0.2161 | 0.1671 |
| momentum_reversal | neutral | 0.0730 | 0.2719 | 0.2683 | 0.2046 | 0.1616 |
| fund_profit_growth | neutral | 0.0470 | 0.2194 | 0.2141 | 0.1722 | 0.1255 |
| fund_revenue_growth | neutral | 0.0373 | 0.2078 | 0.1796 | 0.1378 | 0.1022 |
| wash_sale_score | neutral | 0.0310 | 0.2126 | 0.1458 | 0.1461 | 0.0835 |
| fund_score | neutral | 0.0339 | 0.2467 | 0.1373 | 0.1065 | 0.0759 |
| turnover_stability | neutral | 0.0257 | 0.2310 | 0.1114 | 0.1086 | 0.0618 |
| volatility | bull | 0.1194 | 0.2677 | 0.4462 | 0.3371 | 0.2983 |
| fund_pe | bull | 0.1081 | 0.2796 | 0.3866 | 0.3030 | 0.2519 |
| low_downside | bull | 0.0966 | 0.2592 | 0.3728 | 0.2992 | 0.2422 |
| momentum_reversal | bull | 0.0833 | 0.2431 | 0.3425 | 0.2727 | 0.2180 |
| mom_x_lowvol_20_20 | bull | 0.0831 | 0.2452 | 0.3387 | 0.2652 | 0.2142 |
| fund_pb | bull | 0.0751 | 0.2335 | 0.3219 | 0.2576 | 0.2024 |
| trend_lowvol | bull | 0.0766 | 0.2715 | 0.2823 | 0.1970 | 0.1690 |
| rsi_vol_combo | bull | 0.0567 | 0.2498 | 0.2268 | 0.2197 | 0.1383 |
| fund_profit_growth | bull | 0.0365 | 0.2135 | 0.1709 | 0.1970 | 0.1023 |
| turnover_stability | bull | 0.0263 | 0.2522 | 0.1042 | 0.1212 | 0.0584 |
| mom_x_lowvol_20_20 | bear | 0.1357 | 0.3178 | 0.4271 | 0.4521 | 0.3101 |
| momentum_reversal | bear | 0.0835 | 0.2986 | 0.2796 | 0.2603 | 0.1762 |
| fund_pb | bear | 0.0738 | 0.3351 | 0.2203 | 0.1233 | 0.1237 |

### 磷化工

- **Neutral**: ['fund_pb', 'mom_x_lowvol_20_20', 'fund_profit_growth'] (单因子IC=0.0613, 组合IC=0.1037)
  - weights: [0.3525, 0.3377, 0.3099]
- **Bull**: ['fund_pb'] (单因子IC=0.0847, 组合IC=0.0846)
  - bull_weights: [1.0]
- **Bear**: ['fund_profit_growth', 'mom_x_lowvol_20_20'] (单因子IC=0.1379, 组合IC=0.1858)
  - bear_weights: [0.5323, 0.4677]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| fund_pb | neutral | 0.0633 | 0.2009 | 0.3150 | 0.2568 | 0.1980 |
| mom_x_lowvol_20_20 | neutral | 0.0646 | 0.2152 | 0.3003 | 0.2630 | 0.1896 |
| fund_profit_growth | neutral | 0.0560 | 0.1973 | 0.2838 | 0.2265 | 0.1740 |
| trend_lowvol | neutral | 0.0751 | 0.2656 | 0.2827 | 0.2067 | 0.1706 |
| fund_pe | neutral | 0.0682 | 0.2418 | 0.2821 | 0.1775 | 0.1661 |
| volatility | neutral | 0.0655 | 0.2415 | 0.2712 | 0.2234 | 0.1659 |
| momentum_reversal | neutral | 0.0551 | 0.2297 | 0.2400 | 0.1879 | 0.1426 |
| fund_score | neutral | 0.0437 | 0.2310 | 0.1889 | 0.1461 | 0.1083 |
| rsi_vol_combo | neutral | 0.0399 | 0.2068 | 0.1928 | 0.1221 | 0.1082 |
| low_downside | neutral | 0.0432 | 0.2413 | 0.1789 | 0.1180 | 0.1000 |
| fund_roe | neutral | 0.0396 | 0.2516 | 0.1573 | 0.1075 | 0.0871 |
| fund_pb | bull | 0.0847 | 0.2085 | 0.4062 | 0.2576 | 0.2554 |
| volatility | bull | 0.0630 | 0.2158 | 0.2919 | 0.2652 | 0.1846 |
| fund_pe | bull | 0.0681 | 0.2411 | 0.2823 | 0.2727 | 0.1797 |
| low_downside | bull | 0.0624 | 0.2348 | 0.2659 | 0.1515 | 0.1531 |
| rsi_vol_combo | bull | 0.0376 | 0.1883 | 0.1999 | 0.1439 | 0.1144 |
| mom_x_lowvol_20_20 | bull | 0.0236 | 0.1940 | 0.1217 | 0.1212 | 0.0683 |
| fund_profit_growth | bear | 0.1269 | 0.1939 | 0.6544 | 0.4247 | 0.4662 |
| mom_x_lowvol_20_20 | bear | 0.1489 | 0.2640 | 0.5641 | 0.4521 | 0.4095 |
| fund_score | bear | 0.1050 | 0.2147 | 0.4889 | 0.3699 | 0.3349 |
| momentum_reversal | bear | 0.1101 | 0.2646 | 0.4161 | 0.2877 | 0.2679 |
| fund_gross_margin | bear | 0.0588 | 0.1530 | 0.3841 | 0.2603 | 0.2420 |
| bb_width_20 | bear | 0.0835 | 0.2563 | 0.3257 | 0.2329 | 0.2007 |
| rsi_vol_combo | bear | 0.0519 | 0.2190 | 0.2368 | 0.1233 | 0.1330 |
| fund_pb | bear | 0.0293 | 0.1661 | 0.1763 | 0.1507 | 0.1014 |

### 社区团购

- **Neutral**: ['fund_pb', 'mom_x_lowvol_20_20', 'momentum_reversal'] (单因子IC=0.0971, 组合IC=0.1482)
  - weights: [0.3747, 0.3271, 0.2983]
- **Bull**: ['fund_pb', 'fund_pe', 'low_downside'] (单因子IC=0.0772, 组合IC=0.131)
  - bull_weights: [0.356, 0.3296, 0.3144]
- **Bear**: ['mom_x_lowvol_20_20'] (单因子IC=0.1457, 组合IC=0.1457)
  - bear_weights: [1.0]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| fund_pb | neutral | 0.1027 | 0.2345 | 0.4378 | 0.3486 | 0.2952 |
| mom_x_lowvol_20_20 | neutral | 0.0978 | 0.2482 | 0.3941 | 0.3079 | 0.2577 |
| momentum_reversal | neutral | 0.0909 | 0.2504 | 0.3631 | 0.2944 | 0.2350 |
| trend_lowvol | neutral | 0.0796 | 0.2526 | 0.3153 | 0.2338 | 0.1945 |
| rsi_vol_combo | neutral | 0.0701 | 0.2367 | 0.2959 | 0.2004 | 0.1776 |
| fund_pe | neutral | 0.0529 | 0.2020 | 0.2617 | 0.1712 | 0.1533 |
| volatility | neutral | 0.0525 | 0.2404 | 0.2186 | 0.2203 | 0.1334 |
| fund_profit_growth | neutral | 0.0252 | 0.1823 | 0.1381 | 0.1409 | 0.0788 |
| fund_pb | bull | 0.0884 | 0.2554 | 0.3462 | 0.4167 | 0.2452 |
| fund_pe | bull | 0.0701 | 0.2025 | 0.3464 | 0.3106 | 0.2270 |
| low_downside | bull | 0.0730 | 0.2082 | 0.3507 | 0.2348 | 0.2165 |
| turnover_stability | bull | 0.0467 | 0.1649 | 0.2830 | 0.3636 | 0.1929 |
| volatility | bull | 0.0524 | 0.2276 | 0.2303 | 0.1742 | 0.1352 |
| rsi_vol_combo | bull | 0.0367 | 0.2322 | 0.1581 | 0.1515 | 0.0910 |
| fund_revenue_growth | bull | 0.0255 | 0.1634 | 0.1562 | 0.1439 | 0.0893 |
| fund_score | bull | 0.0252 | 0.1833 | 0.1376 | 0.1212 | 0.0771 |
| momentum_reversal | bull | 0.0242 | 0.2494 | 0.0970 | 0.1136 | 0.0540 |
| mom_x_lowvol_20_20 | bear | 0.1457 | 0.2891 | 0.5040 | 0.4795 | 0.3728 |
| momentum_reversal | bear | 0.1254 | 0.2846 | 0.4406 | 0.3699 | 0.3018 |
| fund_pe | bear | 0.0627 | 0.1962 | 0.3193 | 0.1233 | 0.1794 |
| fund_profit_growth | bear | 0.0456 | 0.1657 | 0.2750 | 0.2603 | 0.1733 |

### 科创板做市股

- **Neutral**: ['fund_pb', 'mom_x_lowvol_20_20', 'fund_pe'] (单因子IC=0.0429, 组合IC=0.0465)
  - weights: [0.3374, 0.3345, 0.328]
- **Bull**: ['momentum_reversal', 'trend_lowvol'] (单因子IC=0.0624, 组合IC=0.0658)
  - bull_weights: [0.5087, 0.4913]
- **Bear**: ['fund_pe', 'momentum_reversal', 'fund_pb'] (单因子IC=0.0522, 组合IC=0.0825)
  - bear_weights: [0.4881, 0.2605, 0.2514]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| fund_pb | neutral | 0.0423 | 0.2279 | 0.1858 | 0.1889 | 0.1105 |
| mom_x_lowvol_20_20 | neutral | 0.0466 | 0.2445 | 0.1904 | 0.1503 | 0.1095 |
| fund_pe | neutral | 0.0397 | 0.2167 | 0.1834 | 0.1712 | 0.1074 |
| momentum_reversal | neutral | 0.0443 | 0.2435 | 0.1818 | 0.1409 | 0.1037 |
| fund_revenue_growth | neutral | 0.0285 | 0.2115 | 0.1346 | 0.1117 | 0.0748 |
| momentum_reversal | bull | 0.0617 | 0.2101 | 0.2935 | 0.2008 | 0.1762 |
| trend_lowvol | bull | 0.0632 | 0.2265 | 0.2791 | 0.2197 | 0.1702 |
| mom_x_lowvol_20_20 | bull | 0.0564 | 0.2115 | 0.2669 | 0.1667 | 0.1557 |
| fund_profit_growth | bull | 0.0447 | 0.2287 | 0.1955 | 0.1818 | 0.1155 |
| fund_pe | bull | 0.0384 | 0.2205 | 0.1740 | 0.2348 | 0.1074 |
| volatility | bull | 0.0399 | 0.2180 | 0.1832 | 0.1439 | 0.1048 |
| rsi_vol_combo | bull | 0.0293 | 0.2029 | 0.1445 | 0.1326 | 0.0818 |
| fund_revenue_growth | bull | 0.0298 | 0.2190 | 0.1359 | 0.1439 | 0.0777 |
| fund_pb | bull | 0.0274 | 0.2296 | 0.1192 | 0.1136 | 0.0664 |
| fund_pe | bear | 0.0545 | 0.1515 | 0.3597 | 0.3151 | 0.2365 |
| momentum_reversal | bear | 0.0587 | 0.2610 | 0.2247 | 0.1233 | 0.1262 |
| fund_pb | bear | 0.0434 | 0.2195 | 0.1976 | 0.2329 | 0.1218 |

### 移动支付

- **Neutral**: ['trend_lowvol', 'momentum_reversal', 'fund_pb'] (单因子IC=0.0925, 组合IC=0.1279)
  - weights: [0.3845, 0.332, 0.2835]
- **Bull**: ['volatility', 'fund_pb', 'low_downside'] (单因子IC=0.1259, 组合IC=0.1653)
  - bull_weights: [0.4127, 0.2966, 0.2906]
- **Bear**: ['trend_lowvol', 'mom_x_lowvol_20_20'] (单因子IC=0.1476, 组合IC=0.1729)
  - bear_weights: [0.517, 0.483]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| trend_lowvol | neutral | 0.1080 | 0.2063 | 0.5232 | 0.3674 | 0.3577 |
| momentum_reversal | neutral | 0.0966 | 0.2083 | 0.4638 | 0.3319 | 0.3089 |
| fund_pb | neutral | 0.0730 | 0.1784 | 0.4089 | 0.2902 | 0.2638 |
| mom_x_lowvol_20_20 | neutral | 0.0858 | 0.2081 | 0.4125 | 0.2610 | 0.2601 |
| rsi_vol_combo | neutral | 0.0708 | 0.1870 | 0.3788 | 0.2526 | 0.2372 |
| volatility | neutral | 0.0646 | 0.2117 | 0.3052 | 0.2610 | 0.1924 |
| turnover_stability | neutral | 0.0407 | 0.1441 | 0.2823 | 0.2359 | 0.1745 |
| fund_score | neutral | 0.0499 | 0.1983 | 0.2514 | 0.2286 | 0.1544 |
| fund_profit_growth | neutral | 0.0395 | 0.1654 | 0.2390 | 0.1754 | 0.1405 |
| fund_roe | neutral | 0.0441 | 0.1974 | 0.2231 | 0.1712 | 0.1307 |
| fund_pe | neutral | 0.0429 | 0.1982 | 0.2166 | 0.1587 | 0.1255 |
| low_downside | neutral | 0.0425 | 0.2148 | 0.1980 | 0.2317 | 0.1219 |
| fund_revenue_growth | neutral | 0.0348 | 0.1683 | 0.2066 | 0.1670 | 0.1205 |
| volatility | bull | 0.1495 | 0.2142 | 0.6981 | 0.5303 | 0.5342 |
| fund_pb | bull | 0.1116 | 0.1993 | 0.5599 | 0.3712 | 0.3839 |
| low_downside | bull | 0.1166 | 0.2242 | 0.5199 | 0.4470 | 0.3762 |
| trend_lowvol | bull | 0.1046 | 0.2378 | 0.4398 | 0.3333 | 0.2932 |
| fund_pe | bull | 0.0908 | 0.2060 | 0.4407 | 0.2803 | 0.2821 |
| turnover_stability | bull | 0.0445 | 0.1356 | 0.3286 | 0.2803 | 0.2104 |
| fund_score | bull | 0.0687 | 0.2230 | 0.3083 | 0.3182 | 0.2032 |
| fund_profit_growth | bull | 0.0532 | 0.1763 | 0.3018 | 0.1667 | 0.1761 |
| fund_roe | bull | 0.0568 | 0.2089 | 0.2720 | 0.1894 | 0.1618 |
| fund_revenue_growth | bull | 0.0448 | 0.1786 | 0.2509 | 0.2197 | 0.1530 |
| mom_x_lowvol_20_20 | bull | 0.0484 | 0.2295 | 0.2111 | 0.1364 | 0.1200 |
| momentum_reversal | bull | 0.0430 | 0.2285 | 0.1883 | 0.1288 | 0.1063 |
| trend_lowvol | bear | 0.1475 | 0.2038 | 0.7240 | 0.5616 | 0.5653 |
| mom_x_lowvol_20_20 | bear | 0.1477 | 0.2030 | 0.7275 | 0.4521 | 0.5282 |
| momentum_reversal | bear | 0.1386 | 0.1994 | 0.6950 | 0.4521 | 0.5046 |
| rsi_vol_combo | bear | 0.0806 | 0.2025 | 0.3981 | 0.2877 | 0.2563 |
| fund_revenue_growth | bear | 0.0482 | 0.1753 | 0.2750 | 0.2055 | 0.1657 |

### 稀土永磁

- **Neutral**: ['fund_pb', 'fund_pe', 'trend_lowvol'] (单因子IC=0.0655, 组合IC=0.08)
  - weights: [0.3431, 0.3321, 0.3248]
- **Bull**: ['volatility', 'fund_pb', 'low_downside'] (单因子IC=0.0934, 组合IC=0.1112)
  - bull_weights: [0.4013, 0.3219, 0.2767]
- **Bear**: ['mom_x_lowvol_20_20'] (单因子IC=0.1482, 组合IC=0.1482)
  - bear_weights: [1.0]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| fund_pb | neutral | 0.0677 | 0.2208 | 0.3066 | 0.2276 | 0.1882 |
| fund_pe | neutral | 0.0593 | 0.2060 | 0.2879 | 0.2651 | 0.1821 |
| trend_lowvol | neutral | 0.0696 | 0.2285 | 0.3047 | 0.1691 | 0.1781 |
| volatility | neutral | 0.0613 | 0.2092 | 0.2932 | 0.1816 | 0.1732 |
| momentum_reversal | neutral | 0.0613 | 0.2155 | 0.2842 | 0.1910 | 0.1693 |
| mom_x_lowvol_20_20 | neutral | 0.0553 | 0.1975 | 0.2801 | 0.1962 | 0.1675 |
| rsi_vol_combo | neutral | 0.0577 | 0.2117 | 0.2726 | 0.1587 | 0.1579 |
| low_downside | neutral | 0.0273 | 0.2167 | 0.1260 | 0.1086 | 0.0698 |
| fund_revenue_growth | neutral | 0.0159 | 0.1724 | 0.0923 | 0.1253 | 0.0520 |
| volatility | bull | 0.1041 | 0.2019 | 0.5155 | 0.3712 | 0.3534 |
| fund_pb | bull | 0.0911 | 0.2080 | 0.4377 | 0.2955 | 0.2835 |
| low_downside | bull | 0.0850 | 0.2074 | 0.4098 | 0.1894 | 0.2437 |
| trend_lowvol | bull | 0.0844 | 0.2273 | 0.3713 | 0.2424 | 0.2307 |
| mom_x_lowvol_20_20 | bull | 0.0613 | 0.1884 | 0.3253 | 0.2576 | 0.2046 |
| momentum_reversal | bull | 0.0693 | 0.2117 | 0.3272 | 0.2197 | 0.1995 |
| rsi_vol_combo | bull | 0.0626 | 0.1899 | 0.3296 | 0.2045 | 0.1985 |
| fund_pe | bull | 0.0400 | 0.2151 | 0.1860 | 0.1061 | 0.1029 |
| mom_x_lowvol_20_20 | bear | 0.1482 | 0.2126 | 0.6970 | 0.5342 | 0.5347 |
| bb_width_20 | bear | 0.0942 | 0.1821 | 0.5175 | 0.3151 | 0.3403 |
| momentum_reversal | bear | 0.1014 | 0.1988 | 0.5100 | 0.2055 | 0.3074 |
| wash_sale_score | bear | 0.0662 | 0.1539 | 0.4303 | 0.3143 | 0.2828 |
| rsi_vol_combo | bear | 0.0509 | 0.2269 | 0.2242 | 0.1781 | 0.1321 |

### 稀缺资源

- **Neutral**: ['fund_pe', 'fund_score'] (单因子IC=0.0696, 组合IC=0.0934)
  - weights: [0.5143, 0.4857]
- **Bull**: ['volatility', 'low_downside', 'fund_pe'] (单因子IC=0.0743, 组合IC=0.0945)
  - bull_weights: [0.3423, 0.3374, 0.3204]
- **Bear**: ['fund_profit_growth', 'bb_width_20', 'turnover_stability'] (单因子IC=0.0842, 组合IC=0.1293)
  - bear_weights: [0.3732, 0.3415, 0.2854]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| fund_pe | neutral | 0.0795 | 0.3024 | 0.2630 | 0.2276 | 0.1614 |
| fund_score | neutral | 0.0597 | 0.2314 | 0.2580 | 0.1816 | 0.1524 |
| fund_profit_growth | neutral | 0.0472 | 0.1874 | 0.2519 | 0.1670 | 0.1470 |
| fund_roe | neutral | 0.0537 | 0.2575 | 0.2085 | 0.1900 | 0.1241 |
| mom_x_lowvol_20_20 | neutral | 0.0382 | 0.1912 | 0.1999 | 0.1399 | 0.1139 |
| volatility | neutral | 0.0360 | 0.1830 | 0.1966 | 0.1461 | 0.1127 |
| fund_pb | neutral | 0.0557 | 0.2826 | 0.1972 | 0.1399 | 0.1124 |
| turnover_stability | neutral | 0.0266 | 0.1670 | 0.1594 | 0.1795 | 0.0940 |
| fund_revenue_growth | neutral | 0.0254 | 0.1957 | 0.1300 | 0.1086 | 0.0721 |
| volatility | bull | 0.0630 | 0.1868 | 0.3373 | 0.3636 | 0.2300 |
| low_downside | bull | 0.0646 | 0.1868 | 0.3459 | 0.3106 | 0.2267 |
| fund_pe | bull | 0.0954 | 0.2905 | 0.3285 | 0.3106 | 0.2153 |
| turnover_stability | bull | 0.0533 | 0.1855 | 0.2876 | 0.3030 | 0.1874 |
| fund_score | bull | 0.0688 | 0.2677 | 0.2569 | 0.1515 | 0.1479 |
| fund_pb | bull | 0.0574 | 0.3004 | 0.1911 | 0.1061 | 0.1057 |
| fund_roe | bull | 0.0341 | 0.2687 | 0.1268 | 0.1515 | 0.0730 |
| momentum_reversal | bull | 0.0221 | 0.2300 | 0.0960 | 0.1136 | 0.0535 |
| fund_profit_growth | bear | 0.0817 | 0.1677 | 0.4873 | 0.3425 | 0.3271 |
| bb_width_20 | bear | 0.1005 | 0.2301 | 0.4369 | 0.3699 | 0.2993 |
| turnover_stability | bear | 0.0703 | 0.1695 | 0.4149 | 0.2055 | 0.2501 |
| fund_score | bear | 0.0798 | 0.2407 | 0.3315 | 0.3151 | 0.2180 |
| fund_roe | bear | 0.0750 | 0.2802 | 0.2679 | 0.2877 | 0.1725 |
| mom_x_lowvol_20_20 | bear | 0.0621 | 0.2420 | 0.2566 | 0.2329 | 0.1582 |
| vol_confirm | bear | 0.0393 | 0.1723 | 0.2279 | 0.1233 | 0.1280 |

### 空气能热泵

- **Neutral**: ['momentum_reversal', 'mom_x_lowvol_20_20', 'trend_lowvol'] (单因子IC=0.0824, 组合IC=0.0975)
  - weights: [0.3604, 0.3371, 0.3026]
- **Bull**: ['low_downside'] (单因子IC=0.0727, 组合IC=0.0727)
  - bull_weights: [1.0]
- **Bear**: ['trend_lowvol', 'rsi_vol_combo'] (单因子IC=0.1677, 组合IC=0.2053)
  - bear_weights: [0.5663, 0.4337]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| momentum_reversal | neutral | 0.0875 | 0.2460 | 0.3558 | 0.2860 | 0.2288 |
| mom_x_lowvol_20_20 | neutral | 0.0720 | 0.2113 | 0.3406 | 0.2568 | 0.2140 |
| trend_lowvol | neutral | 0.0877 | 0.2821 | 0.3108 | 0.2359 | 0.1921 |
| fund_pb | neutral | 0.0694 | 0.2716 | 0.2557 | 0.1775 | 0.1505 |
| rsi_vol_combo | neutral | 0.0492 | 0.2295 | 0.2144 | 0.1691 | 0.1253 |
| fund_profit_growth | neutral | 0.0407 | 0.2171 | 0.1875 | 0.1461 | 0.1074 |
| fund_pe | neutral | 0.0510 | 0.2777 | 0.1835 | 0.1044 | 0.1013 |
| volatility | neutral | 0.0354 | 0.2504 | 0.1413 | 0.1221 | 0.0793 |
| turnover_stability | neutral | 0.0228 | 0.1937 | 0.1176 | 0.1086 | 0.0652 |
| low_downside | neutral | 0.0270 | 0.2581 | 0.1044 | 0.1086 | 0.0579 |
| low_downside | bull | 0.0727 | 0.2322 | 0.3128 | 0.2614 | 0.1973 |
| fund_profit_growth | bull | 0.0547 | 0.1876 | 0.2915 | 0.2348 | 0.1800 |
| fund_revenue_growth | bull | 0.0434 | 0.1868 | 0.2325 | 0.1136 | 0.1294 |
| stroke_phase | bull | 0.0440 | 0.1997 | 0.2204 | 0.1515 | 0.1269 |
| fund_pb | bull | 0.0614 | 0.3129 | 0.1963 | 0.2197 | 0.1197 |
| trend_lowvol | bear | 0.2000 | 0.3094 | 0.6462 | 0.2877 | 0.4161 |
| rsi_vol_combo | bear | 0.1353 | 0.2909 | 0.4653 | 0.3699 | 0.3187 |
| momentum_reversal | bear | 0.1345 | 0.3217 | 0.4179 | 0.2877 | 0.2691 |
| fund_pb | bear | 0.0884 | 0.2066 | 0.4279 | 0.2329 | 0.2638 |
| fund_revenue_growth | bear | 0.0821 | 0.2219 | 0.3700 | 0.2055 | 0.2230 |
| vol_confirm | bear | 0.0679 | 0.2413 | 0.2815 | 0.2055 | 0.1697 |
| mom_x_lowvol_20_20 | bear | 0.0650 | 0.2907 | 0.2235 | 0.2329 | 0.1378 |
| turnover_stability | bear | 0.0363 | 0.1830 | 0.1984 | 0.1507 | 0.1141 |
| volatility | bear | 0.0417 | 0.2339 | 0.1784 | 0.1233 | 0.1002 |

### 空间站概念

- **Neutral**: ['fund_pb', 'momentum_reversal', 'mom_x_lowvol_20_20'] (单因子IC=0.0878, 组合IC=0.1076)
  - weights: [0.3428, 0.3299, 0.3274]
- **Bull**: ['volatility', 'fund_pb', 'mom_x_lowvol_20_20'] (单因子IC=0.1184, 组合IC=0.1577)
  - bull_weights: [0.4325, 0.2957, 0.2718]
- **Bear**: ['mom_x_lowvol_20_20', 'momentum_reversal', 'turnover_stability'] (单因子IC=0.2666, 组合IC=0.3016)
  - bear_weights: [0.3554, 0.3342, 0.3104]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| fund_pb | neutral | 0.0795 | 0.2455 | 0.3237 | 0.2850 | 0.2080 |
| momentum_reversal | neutral | 0.0932 | 0.2918 | 0.3193 | 0.2537 | 0.2001 |
| mom_x_lowvol_20_20 | neutral | 0.0907 | 0.2853 | 0.3179 | 0.2495 | 0.1986 |
| trend_lowvol | neutral | 0.0757 | 0.3176 | 0.2384 | 0.1628 | 0.1386 |
| fund_profit_growth | neutral | 0.0587 | 0.2629 | 0.2232 | 0.1921 | 0.1330 |
| rsi_vol_combo | neutral | 0.0595 | 0.2800 | 0.2127 | 0.2192 | 0.1296 |
| volatility | neutral | 0.0582 | 0.2946 | 0.1975 | 0.1232 | 0.1109 |
| fund_score | neutral | 0.0342 | 0.2861 | 0.1195 | 0.1409 | 0.0681 |
| volatility | bull | 0.1474 | 0.2532 | 0.5823 | 0.3788 | 0.4014 |
| fund_pb | bull | 0.0997 | 0.2407 | 0.4141 | 0.3258 | 0.2745 |
| mom_x_lowvol_20_20 | bull | 0.1080 | 0.2790 | 0.3872 | 0.3030 | 0.2522 |
| low_downside | bull | 0.0961 | 0.2628 | 0.3659 | 0.2917 | 0.2363 |
| turnover_stability | bull | 0.0823 | 0.2453 | 0.3355 | 0.2879 | 0.2161 |
| momentum_reversal | bull | 0.0977 | 0.2893 | 0.3377 | 0.1818 | 0.1996 |
| trend_lowvol | bull | 0.0970 | 0.3065 | 0.3166 | 0.2121 | 0.1919 |
| rsi_vol_combo | bull | 0.0789 | 0.2623 | 0.3009 | 0.2197 | 0.1835 |
| fund_pe | bull | 0.0497 | 0.3199 | 0.1555 | 0.1136 | 0.0866 |
| mom_x_lowvol_20_20 | bear | 0.2968 | 0.2465 | 1.2042 | 0.7260 | 1.0392 |
| momentum_reversal | bear | 0.2905 | 0.2483 | 1.1698 | 0.6712 | 0.9775 |
| turnover_stability | bear | 0.2125 | 0.2068 | 1.0275 | 0.7671 | 0.9078 |
| trend_lowvol | bear | 0.3063 | 0.3146 | 0.9737 | 0.6438 | 0.8003 |
| fund_pe | bear | 0.1759 | 0.2377 | 0.7398 | 0.5616 | 0.5777 |
| rsi_vol_combo | bear | 0.1621 | 0.2563 | 0.6323 | 0.4247 | 0.4504 |
| fund_score | bear | 0.1558 | 0.2548 | 0.6114 | 0.4521 | 0.4439 |
| fund_roe | bear | 0.1801 | 0.3124 | 0.5764 | 0.5068 | 0.4343 |
| fund_gross_margin | bear | 0.1455 | 0.2430 | 0.5987 | 0.3973 | 0.4183 |
| fund_profit_growth | bear | 0.0897 | 0.2265 | 0.3960 | 0.3014 | 0.2577 |
| fund_revenue_growth | bear | 0.1107 | 0.2737 | 0.4045 | 0.2055 | 0.2438 |

### 空间计算

- **Neutral**: ['trend_lowvol', 'volatility', 'mom_x_lowvol_20_20'] (单因子IC=0.1037, 组合IC=0.1423)
  - weights: [0.3772, 0.3322, 0.2906]
- **Bull**: ['wash_sale_score', 'fund_revenue_growth', 'volatility'] (单因子IC=0.0722, 组合IC=0.095)
  - bull_weights: [0.351, 0.3342, 0.3148]
- **Bear**: ['volatility', 'momentum_reversal', 'turnover_stability'] (单因子IC=0.0938, 组合IC=0.1361)
  - bear_weights: [0.398, 0.3021, 0.2999]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| trend_lowvol | neutral | 0.1138 | 0.3000 | 0.3795 | 0.2401 | 0.2353 |
| volatility | neutral | 0.0997 | 0.3086 | 0.3230 | 0.2829 | 0.2072 |
| mom_x_lowvol_20_20 | neutral | 0.0975 | 0.3249 | 0.2999 | 0.2088 | 0.1813 |
| momentum_reversal | neutral | 0.0956 | 0.3263 | 0.2930 | 0.2088 | 0.1771 |
| rsi_vol_combo | neutral | 0.0672 | 0.3093 | 0.2171 | 0.2046 | 0.1308 |
| fund_pb | neutral | 0.0697 | 0.3473 | 0.2008 | 0.1775 | 0.1182 |
| low_downside | neutral | 0.0575 | 0.3030 | 0.1898 | 0.1795 | 0.1120 |
| turnover_stability | neutral | 0.0345 | 0.2729 | 0.1262 | 0.1597 | 0.0732 |
| fund_profit_growth | neutral | 0.0298 | 0.2549 | 0.1170 | 0.1106 | 0.0650 |
| wash_sale_score | bull | 0.0720 | 0.2360 | 0.3050 | 0.2447 | 0.1898 |
| fund_revenue_growth | bull | 0.0740 | 0.2499 | 0.2963 | 0.2197 | 0.1807 |
| volatility | bull | 0.0707 | 0.2500 | 0.2827 | 0.2045 | 0.1702 |
| trend_lowvol | bull | 0.0787 | 0.3025 | 0.2601 | 0.2879 | 0.1675 |
| fund_profit_growth | bull | 0.0387 | 0.2153 | 0.1797 | 0.1553 | 0.1038 |
| mom_x_lowvol_20_20 | bull | 0.0446 | 0.2981 | 0.1497 | 0.1515 | 0.0862 |
| fund_pb | bull | 0.0423 | 0.2945 | 0.1437 | 0.1515 | 0.0827 |
| stroke_phase | bull | 0.0340 | 0.2441 | 0.1392 | 0.1667 | 0.0812 |
| low_downside | bull | 0.0374 | 0.2824 | 0.1323 | 0.1629 | 0.0769 |
| momentum_reversal | bull | 0.0344 | 0.2855 | 0.1204 | 0.1212 | 0.0675 |
| volatility | bear | 0.1143 | 0.3459 | 0.3305 | 0.2055 | 0.1992 |
| momentum_reversal | bear | 0.0822 | 0.3353 | 0.2453 | 0.2329 | 0.1512 |
| turnover_stability | bear | 0.0848 | 0.3327 | 0.2548 | 0.1781 | 0.1501 |
| rsi_vol_combo | bear | 0.0732 | 0.3252 | 0.2252 | 0.2055 | 0.1357 |
| trend_lowvol | bear | 0.0699 | 0.3337 | 0.2094 | 0.1233 | 0.1176 |

### 第三代半导体

- **Neutral**: ['mom_x_lowvol_20_20', 'momentum_reversal', 'volatility'] (单因子IC=0.0804, 组合IC=0.0945)
  - weights: [0.3611, 0.344, 0.2949]
- **Bull**: ['volatility', 'low_downside', 'mom_x_lowvol_20_20'] (单因子IC=0.0781, 组合IC=0.0941)
  - bull_weights: [0.3951, 0.3059, 0.299]
- **Bear**: ['momentum_reversal', 'mom_x_lowvol_20_20', 'trend_lowvol'] (单因子IC=0.1582, 组合IC=0.1796)
  - bear_weights: [0.3901, 0.3604, 0.2494]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| mom_x_lowvol_20_20 | neutral | 0.0855 | 0.1775 | 0.4820 | 0.3570 | 0.3270 |
| momentum_reversal | neutral | 0.0859 | 0.1851 | 0.4642 | 0.3424 | 0.3116 |
| volatility | neutral | 0.0697 | 0.1722 | 0.4049 | 0.3194 | 0.2671 |
| rsi_vol_combo | neutral | 0.0739 | 0.1867 | 0.3958 | 0.3194 | 0.2611 |
| trend_lowvol | neutral | 0.0686 | 0.1927 | 0.3560 | 0.2547 | 0.2233 |
| fund_pb | neutral | 0.0530 | 0.1895 | 0.2797 | 0.2171 | 0.1702 |
| turnover_stability | neutral | 0.0361 | 0.1302 | 0.2775 | 0.1962 | 0.1660 |
| fund_pe | neutral | 0.0412 | 0.1578 | 0.2610 | 0.1670 | 0.1523 |
| low_downside | neutral | 0.0392 | 0.1642 | 0.2391 | 0.2380 | 0.1480 |
| fund_revenue_growth | neutral | 0.0240 | 0.1622 | 0.1482 | 0.1628 | 0.0862 |
| volatility | bull | 0.0903 | 0.1643 | 0.5499 | 0.4318 | 0.3937 |
| low_downside | bull | 0.0715 | 0.1546 | 0.4623 | 0.3182 | 0.3047 |
| mom_x_lowvol_20_20 | bull | 0.0724 | 0.1684 | 0.4298 | 0.3864 | 0.2979 |
| fund_pe | bull | 0.0556 | 0.1446 | 0.3844 | 0.3030 | 0.2504 |
| trend_lowvol | bull | 0.0704 | 0.1917 | 0.3670 | 0.3030 | 0.2391 |
| momentum_reversal | bull | 0.0619 | 0.1756 | 0.3524 | 0.3258 | 0.2336 |
| fund_pb | bull | 0.0654 | 0.1959 | 0.3340 | 0.2727 | 0.2126 |
| fund_revenue_growth | bull | 0.0438 | 0.1758 | 0.2491 | 0.1970 | 0.1491 |
| turnover_stability | bull | 0.0290 | 0.1185 | 0.2445 | 0.1742 | 0.1435 |
| rsi_vol_combo | bull | 0.0394 | 0.1781 | 0.2214 | 0.2803 | 0.1417 |
| fund_roe | bull | 0.0380 | 0.1702 | 0.2233 | 0.1667 | 0.1303 |
| fund_score | bull | 0.0346 | 0.1902 | 0.1821 | 0.2121 | 0.1104 |
| momentum_reversal | bear | 0.1667 | 0.1849 | 0.9013 | 0.7260 | 0.7778 |
| mom_x_lowvol_20_20 | bear | 0.1672 | 0.1880 | 0.8891 | 0.6164 | 0.7186 |
| trend_lowvol | bear | 0.1409 | 0.2018 | 0.6981 | 0.4247 | 0.4973 |
| rsi_vol_combo | bear | 0.1063 | 0.1851 | 0.5741 | 0.5068 | 0.4325 |
| fund_pe | bear | 0.0579 | 0.1740 | 0.3330 | 0.1781 | 0.1962 |
| fund_pb | bear | 0.0737 | 0.2168 | 0.3397 | 0.1507 | 0.1954 |
| bb_width_20 | bear | 0.0385 | 0.1726 | 0.2228 | 0.2877 | 0.1435 |

### 算力概念

- **Neutral**: ['mom_x_lowvol_20_20', 'momentum_reversal', 'volatility'] (单因子IC=0.0716, 组合IC=0.0832)
  - weights: [0.3478, 0.3471, 0.305]
- **Bull**: ['volatility'] (单因子IC=0.1015, 组合IC=0.1015)
  - bull_weights: [1.0]
- **Bear**: ['trend_lowvol', 'momentum_reversal', 'mom_x_lowvol_20_20'] (单因子IC=0.1324, 组合IC=0.154)
  - bear_weights: [0.4795, 0.2659, 0.2546]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| mom_x_lowvol_20_20 | neutral | 0.0723 | 0.1585 | 0.4563 | 0.3027 | 0.2972 |
| momentum_reversal | neutral | 0.0729 | 0.1598 | 0.4562 | 0.3006 | 0.2967 |
| volatility | neutral | 0.0696 | 0.1817 | 0.3830 | 0.3612 | 0.2607 |
| trend_lowvol | neutral | 0.0702 | 0.1847 | 0.3800 | 0.3236 | 0.2515 |
| fund_pb | neutral | 0.0633 | 0.1629 | 0.3888 | 0.2443 | 0.2419 |
| rsi_vol_combo | neutral | 0.0551 | 0.1483 | 0.3716 | 0.2714 | 0.2363 |
| fund_pe | neutral | 0.0505 | 0.1491 | 0.3390 | 0.2839 | 0.2176 |
| fund_profit_growth | neutral | 0.0411 | 0.1277 | 0.3216 | 0.2651 | 0.2034 |
| fund_score | neutral | 0.0491 | 0.1607 | 0.3056 | 0.2338 | 0.1885 |
| fund_roe | neutral | 0.0452 | 0.1588 | 0.2845 | 0.2004 | 0.1708 |
| low_downside | neutral | 0.0412 | 0.1849 | 0.2229 | 0.1733 | 0.1308 |
| turnover_stability | neutral | 0.0208 | 0.1068 | 0.1952 | 0.1399 | 0.1113 |
| fund_revenue_growth | neutral | 0.0209 | 0.1104 | 0.1893 | 0.1587 | 0.1097 |
| volatility | bull | 0.1015 | 0.1332 | 0.7619 | 0.5682 | 0.5974 |
| fund_pe | bull | 0.0738 | 0.1299 | 0.5682 | 0.4470 | 0.4111 |
| low_downside | bull | 0.0832 | 0.1492 | 0.5574 | 0.4318 | 0.3990 |
| fund_pb | bull | 0.0725 | 0.1450 | 0.5001 | 0.3182 | 0.3296 |
| mom_x_lowvol_20_20 | bull | 0.0601 | 0.1289 | 0.4664 | 0.4091 | 0.3286 |
| turnover_stability | bull | 0.0472 | 0.1012 | 0.4662 | 0.3712 | 0.3196 |
| trend_lowvol | bull | 0.0727 | 0.1542 | 0.4718 | 0.3030 | 0.3074 |
| momentum_reversal | bull | 0.0524 | 0.1340 | 0.3908 | 0.3636 | 0.2665 |
| fund_profit_growth | bull | 0.0383 | 0.1195 | 0.3204 | 0.2273 | 0.1966 |
| rsi_vol_combo | bull | 0.0392 | 0.1360 | 0.2878 | 0.2121 | 0.1745 |
| fund_roe | bull | 0.0399 | 0.1464 | 0.2724 | 0.2045 | 0.1640 |
| fund_score | bull | 0.0310 | 0.1635 | 0.1899 | 0.1061 | 0.1050 |
| fund_gross_margin | bull | 0.0156 | 0.1012 | 0.1545 | 0.1364 | 0.0878 |
| stroke_phase | bull | 0.0143 | 0.1100 | 0.1304 | 0.1212 | 0.0731 |
| trend_lowvol | bear | 0.1494 | 0.1408 | 1.0613 | 0.7260 | 0.9159 |
| momentum_reversal | bear | 0.1241 | 0.1774 | 0.6996 | 0.4521 | 0.5079 |
| mom_x_lowvol_20_20 | bear | 0.1238 | 0.1812 | 0.6829 | 0.4247 | 0.4864 |
| rsi_vol_combo | bear | 0.0828 | 0.1472 | 0.5627 | 0.5342 | 0.4316 |
| fund_profit_growth | bear | 0.0525 | 0.1243 | 0.4223 | 0.2329 | 0.2603 |
| turnover_stability | bear | 0.0276 | 0.0981 | 0.2818 | 0.2877 | 0.1814 |
| fund_revenue_growth | bear | 0.0295 | 0.1371 | 0.2153 | 0.1507 | 0.1239 |

### 粤港自贸

- **Neutral**: ['trend_lowvol', 'mom_x_lowvol_20_20'] (单因子IC=0.0915, 组合IC=0.1045)
  - weights: [0.5216, 0.4784]
- **Bull**: ['fund_pb', 'turnover_stability', 'momentum_reversal'] (单因子IC=0.0688, 组合IC=0.1114)
  - bull_weights: [0.3864, 0.3219, 0.2917]
- **Bear**: ['momentum_reversal', 'mom_x_lowvol_20_20', 'bb_width_20'] (单因子IC=0.105, 组合IC=0.1352)
  - bear_weights: [0.3975, 0.342, 0.2605]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| trend_lowvol | neutral | 0.0997 | 0.2285 | 0.4364 | 0.3027 | 0.2843 |
| mom_x_lowvol_20_20 | neutral | 0.0832 | 0.2082 | 0.3996 | 0.3048 | 0.2607 |
| momentum_reversal | neutral | 0.0749 | 0.2230 | 0.3357 | 0.2255 | 0.2057 |
| turnover_stability | neutral | 0.0505 | 0.1527 | 0.3307 | 0.2370 | 0.2045 |
| fund_pb | neutral | 0.0599 | 0.1960 | 0.3057 | 0.2443 | 0.1902 |
| fund_profit_growth | neutral | 0.0411 | 0.1478 | 0.2780 | 0.2088 | 0.1680 |
| fund_pe | neutral | 0.0542 | 0.2152 | 0.2516 | 0.2610 | 0.1587 |
| low_downside | neutral | 0.0519 | 0.2107 | 0.2462 | 0.2787 | 0.1574 |
| fund_score | neutral | 0.0436 | 0.1703 | 0.2563 | 0.1889 | 0.1523 |
| volatility | neutral | 0.0501 | 0.2168 | 0.2310 | 0.1962 | 0.1381 |
| rsi_vol_combo | neutral | 0.0481 | 0.2085 | 0.2309 | 0.1524 | 0.1331 |
| fund_roe | neutral | 0.0337 | 0.1873 | 0.1800 | 0.1461 | 0.1031 |
| fund_revenue_growth | neutral | 0.0206 | 0.1405 | 0.1468 | 0.1357 | 0.0833 |
| fund_gross_margin | neutral | 0.0181 | 0.1286 | 0.1409 | 0.1441 | 0.0806 |
| fund_pb | bull | 0.0889 | 0.2056 | 0.4326 | 0.3788 | 0.2982 |
| turnover_stability | bull | 0.0504 | 0.1331 | 0.3790 | 0.3106 | 0.2484 |
| momentum_reversal | bull | 0.0670 | 0.1791 | 0.3738 | 0.2045 | 0.2251 |
| volatility | bull | 0.0745 | 0.2179 | 0.3419 | 0.2879 | 0.2202 |
| rsi_vol_combo | bull | 0.0588 | 0.1729 | 0.3404 | 0.1364 | 0.1934 |
| low_downside | bull | 0.0599 | 0.1970 | 0.3041 | 0.2652 | 0.1924 |
| trend_lowvol | bull | 0.0608 | 0.2076 | 0.2931 | 0.1742 | 0.1721 |
| mom_x_lowvol_20_20 | bull | 0.0413 | 0.1630 | 0.2534 | 0.1136 | 0.1411 |
| fund_pe | bull | 0.0379 | 0.2219 | 0.1710 | 0.1439 | 0.0978 |
| fund_profit_growth | bull | 0.0144 | 0.1484 | 0.0969 | 0.1061 | 0.0536 |
| momentum_reversal | bear | 0.1295 | 0.2696 | 0.4803 | 0.3151 | 0.3158 |
| mom_x_lowvol_20_20 | bear | 0.1071 | 0.2483 | 0.4312 | 0.2603 | 0.2717 |
| bb_width_20 | bear | 0.0785 | 0.2390 | 0.3285 | 0.2603 | 0.2070 |
| trend_lowvol | bear | 0.0772 | 0.2559 | 0.3016 | 0.1507 | 0.1735 |
| rsi_vol_combo | bear | 0.0706 | 0.2389 | 0.2953 | 0.1507 | 0.1699 |
| fund_pe | bear | 0.0433 | 0.2402 | 0.1804 | 0.3151 | 0.1186 |

### 粮食概念

- **Neutral**: ['momentum_reversal', 'rsi_vol_combo'] (单因子IC=0.0969, 组合IC=0.1056)
  - weights: [0.5393, 0.4607]
- **Bull**: ['fund_pb'] (单因子IC=0.1299, 组合IC=0.1299)
  - bull_weights: [1.0]
- **Bear**: ['fund_gross_margin', 'fund_score', 'fund_profit_growth'] (单因子IC=0.0632, 组合IC=0.103)
  - bear_weights: [0.4247, 0.3099, 0.2655]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| momentum_reversal | neutral | 0.1049 | 0.2826 | 0.3711 | 0.3173 | 0.2444 |
| rsi_vol_combo | neutral | 0.0889 | 0.2641 | 0.3365 | 0.2411 | 0.2088 |
| mom_x_lowvol_20_20 | neutral | 0.0819 | 0.2655 | 0.3084 | 0.2516 | 0.1930 |
| trend_lowvol | neutral | 0.0776 | 0.2863 | 0.2711 | 0.2109 | 0.1641 |
| fund_pe | neutral | 0.0676 | 0.2720 | 0.2485 | 0.2192 | 0.1515 |
| volatility | neutral | 0.0641 | 0.2609 | 0.2455 | 0.2088 | 0.1484 |
| fund_pb | neutral | 0.0686 | 0.2982 | 0.2300 | 0.1931 | 0.1372 |
| low_downside | neutral | 0.0559 | 0.2753 | 0.2031 | 0.1754 | 0.1193 |
| turnover_stability | neutral | 0.0479 | 0.2445 | 0.1960 | 0.1378 | 0.1115 |
| wash_sale_score | neutral | 0.0201 | 0.2281 | 0.0879 | 0.1101 | 0.0488 |
| fund_pb | bull | 0.1299 | 0.2889 | 0.4496 | 0.3712 | 0.3083 |
| trend_lowvol | bull | 0.0822 | 0.2409 | 0.3414 | 0.2955 | 0.2211 |
| fund_pe | bull | 0.0867 | 0.2619 | 0.3311 | 0.2083 | 0.2000 |
| rsi_vol_combo | bull | 0.0439 | 0.2365 | 0.1855 | 0.2727 | 0.1181 |
| volatility | bull | 0.0476 | 0.2457 | 0.1938 | 0.2121 | 0.1175 |
| fund_profit_growth | bull | 0.0418 | 0.2098 | 0.1992 | 0.1591 | 0.1155 |
| low_downside | bull | 0.0441 | 0.2321 | 0.1901 | 0.1970 | 0.1138 |
| stroke_phase | bull | 0.0369 | 0.1986 | 0.1859 | 0.1591 | 0.1078 |
| limit_pullback_score | bull | 0.0402 | 0.2471 | 0.1626 | 0.1639 | 0.0946 |
| mom_x_lowvol_20_20 | bull | 0.0395 | 0.2391 | 0.1651 | 0.1364 | 0.0938 |
| momentum_reversal | bull | 0.0371 | 0.2539 | 0.1460 | 0.1667 | 0.0852 |
| fund_score | bull | 0.0297 | 0.2321 | 0.1280 | 0.1439 | 0.0732 |
| turnover_stability | bull | 0.0254 | 0.2457 | 0.1033 | 0.1667 | 0.0603 |
| fund_gross_margin | bear | 0.0636 | 0.1586 | 0.4014 | 0.2603 | 0.2529 |
| fund_score | bear | 0.0662 | 0.2259 | 0.2929 | 0.2603 | 0.1846 |
| fund_profit_growth | bear | 0.0599 | 0.2337 | 0.2565 | 0.2329 | 0.1581 |
| fund_roe | bear | 0.0604 | 0.2422 | 0.2493 | 0.2329 | 0.1537 |
| bb_width_20 | bear | 0.0673 | 0.2582 | 0.2607 | 0.1781 | 0.1535 |

### 精准医疗

- **Neutral**: ['fund_profit_growth', 'volatility', 'trend_lowvol'] (单因子IC=0.07, 组合IC=0.1033)
  - weights: [0.3583, 0.3271, 0.3146]
- **Bull**: ['volatility', 'low_downside', 'turnover_stability'] (单因子IC=0.0896, 组合IC=0.1031)
  - bull_weights: [0.3717, 0.323, 0.3053]
- **Bear**: ['mom_x_lowvol_20_20'] (单因子IC=0.164, 组合IC=0.164)
  - bear_weights: [1.0]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| fund_profit_growth | neutral | 0.0695 | 0.2061 | 0.3373 | 0.3132 | 0.2214 |
| volatility | neutral | 0.0672 | 0.2081 | 0.3231 | 0.2516 | 0.2022 |
| trend_lowvol | neutral | 0.0731 | 0.2317 | 0.3157 | 0.2317 | 0.1944 |
| momentum_reversal | neutral | 0.0687 | 0.2288 | 0.3002 | 0.1816 | 0.1773 |
| mom_x_lowvol_20_20 | neutral | 0.0608 | 0.2206 | 0.2755 | 0.1837 | 0.1631 |
| fund_pb | neutral | 0.0546 | 0.2126 | 0.2566 | 0.2234 | 0.1570 |
| rsi_vol_combo | neutral | 0.0506 | 0.2198 | 0.2301 | 0.1973 | 0.1377 |
| low_downside | neutral | 0.0433 | 0.2144 | 0.2020 | 0.1712 | 0.1183 |
| fund_score | neutral | 0.0440 | 0.2435 | 0.1807 | 0.1733 | 0.1060 |
| fund_pe | neutral | 0.0374 | 0.2549 | 0.1467 | 0.1232 | 0.0824 |
| volatility | bull | 0.1026 | 0.2091 | 0.4907 | 0.3788 | 0.3383 |
| low_downside | bull | 0.0850 | 0.1916 | 0.4434 | 0.3258 | 0.2939 |
| turnover_stability | bull | 0.0812 | 0.1971 | 0.4121 | 0.3485 | 0.2779 |
| fund_profit_growth | bull | 0.0690 | 0.1864 | 0.3703 | 0.3333 | 0.2469 |
| rsi_vol_combo | bull | 0.0616 | 0.1976 | 0.3117 | 0.2727 | 0.1983 |
| fund_pb | bull | 0.0681 | 0.2303 | 0.2959 | 0.2045 | 0.1782 |
| momentum_reversal | bull | 0.0665 | 0.2239 | 0.2971 | 0.1894 | 0.1767 |
| fund_score | bull | 0.0498 | 0.2019 | 0.2469 | 0.1667 | 0.1440 |
| mom_x_lowvol_20_20 | bull | 0.0457 | 0.2056 | 0.2221 | 0.1667 | 0.1295 |
| fund_revenue_growth | bull | 0.0363 | 0.1837 | 0.1976 | 0.2273 | 0.1213 |
| fund_roe | bull | 0.0371 | 0.2078 | 0.1788 | 0.1212 | 0.1002 |
| fund_pe | bull | 0.0315 | 0.2445 | 0.1287 | 0.1515 | 0.0741 |
| mom_x_lowvol_20_20 | bear | 0.1640 | 0.2365 | 0.6932 | 0.5616 | 0.5413 |
| momentum_reversal | bear | 0.1366 | 0.2412 | 0.5662 | 0.4247 | 0.4033 |
| trend_lowvol | bear | 0.1315 | 0.2462 | 0.5342 | 0.4247 | 0.3805 |
| fund_gross_margin | bear | 0.0630 | 0.1866 | 0.3373 | 0.2877 | 0.2172 |
| rsi_vol_combo | bear | 0.0749 | 0.2408 | 0.3108 | 0.2603 | 0.1959 |
| fund_revenue_growth | bear | 0.0689 | 0.2286 | 0.3013 | 0.2329 | 0.1857 |

### 精准诊断

- **Neutral**: ['volatility', 'momentum_reversal', 'turnover_stability'] (单因子IC=0.062, 组合IC=0.0849)
  - weights: [0.4089, 0.2958, 0.2953]
- **Bull**: ['volatility', 'fund_profit_growth'] (单因子IC=0.0592, 组合IC=0.0777)
  - bull_weights: [0.5278, 0.4722]
- **Bear**: ['fund_gross_margin', 'fund_profit_growth'] (单因子IC=0.0777, 组合IC=0.0904)
  - bear_weights: [0.5161, 0.4839]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| volatility | neutral | 0.0786 | 0.2064 | 0.3807 | 0.3048 | 0.2483 |
| momentum_reversal | neutral | 0.0627 | 0.2096 | 0.2993 | 0.2004 | 0.1796 |
| turnover_stability | neutral | 0.0447 | 0.1523 | 0.2937 | 0.2213 | 0.1793 |
| fund_pb | neutral | 0.0632 | 0.2219 | 0.2849 | 0.2443 | 0.1772 |
| mom_x_lowvol_20_20 | neutral | 0.0593 | 0.2090 | 0.2837 | 0.1545 | 0.1638 |
| trend_lowvol | neutral | 0.0623 | 0.2291 | 0.2718 | 0.1733 | 0.1594 |
| rsi_vol_combo | neutral | 0.0426 | 0.1890 | 0.2255 | 0.1482 | 0.1294 |
| fund_profit_growth | neutral | 0.0358 | 0.1772 | 0.2021 | 0.1441 | 0.1156 |
| low_downside | neutral | 0.0369 | 0.2144 | 0.1723 | 0.1378 | 0.0980 |
| volatility | bull | 0.0727 | 0.2025 | 0.3589 | 0.3106 | 0.2352 |
| fund_profit_growth | bull | 0.0458 | 0.1369 | 0.3347 | 0.2576 | 0.2104 |
| low_downside | bull | 0.0562 | 0.1804 | 0.3114 | 0.1818 | 0.1840 |
| fund_revenue_growth | bull | 0.0408 | 0.1503 | 0.2717 | 0.2273 | 0.1667 |
| turnover_stability | bull | 0.0389 | 0.1630 | 0.2386 | 0.2273 | 0.1464 |
| fund_pb | bull | 0.0510 | 0.2201 | 0.2319 | 0.2500 | 0.1449 |
| fund_score | bull | 0.0425 | 0.1744 | 0.2438 | 0.1667 | 0.1422 |
| fund_gross_margin | bull | 0.0300 | 0.1476 | 0.2032 | 0.1515 | 0.1170 |
| rsi_vol_combo | bull | 0.0333 | 0.1741 | 0.1915 | 0.1591 | 0.1110 |
| momentum_reversal | bull | 0.0238 | 0.1730 | 0.1378 | 0.1818 | 0.0814 |
| mom_x_lowvol_20_20 | bull | 0.0183 | 0.1721 | 0.1062 | 0.1061 | 0.0587 |
| fund_gross_margin | bear | 0.0810 | 0.2147 | 0.3774 | 0.2603 | 0.2378 |
| fund_profit_growth | bear | 0.0743 | 0.2145 | 0.3464 | 0.2877 | 0.2230 |
| fund_score | bear | 0.0605 | 0.2478 | 0.2441 | 0.2055 | 0.1471 |
| mom_x_lowvol_20_20 | bear | 0.0616 | 0.2503 | 0.2462 | 0.1507 | 0.1417 |
| fund_pb | bear | 0.0356 | 0.1599 | 0.2230 | 0.2055 | 0.1344 |
| turnover_stability | bear | 0.0334 | 0.1668 | 0.2000 | 0.2055 | 0.1205 |
| bb_width_20 | bear | 0.0382 | 0.1953 | 0.1957 | 0.1233 | 0.1099 |

### 红利股

- **Neutral**: ['fund_pe'] (单因子IC=0.1094, 组合IC=0.1093)
  - weights: [1.0]
- **Bull**: ['vol_confirm', 'relative_strength', 'ma_alignment'] (单因子IC=0.0501, 组合IC=0.0832)
  - bull_weights: [0.4404, 0.3433, 0.2163]
- **Bear**: ['bb_width_20', 'mom_x_lowvol_20_20', 'fund_pb'] (单因子IC=0.0712, 组合IC=0.1441)
  - bear_weights: [0.3993, 0.3245, 0.2761]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| fund_pe | neutral | 0.1094 | 0.2053 | 0.5328 | 0.4301 | 0.3810 |
| fund_pb | neutral | 0.0743 | 0.1798 | 0.4131 | 0.2923 | 0.2669 |
| fund_profit_growth | neutral | 0.0569 | 0.1947 | 0.2922 | 0.2296 | 0.1797 |
| fund_score | neutral | 0.0473 | 0.1817 | 0.2602 | 0.1921 | 0.1551 |
| fund_revenue_growth | neutral | 0.0343 | 0.1762 | 0.1946 | 0.2171 | 0.1184 |
| turnover_stability | neutral | 0.0260 | 0.1435 | 0.1808 | 0.1399 | 0.1031 |
| fund_roe | neutral | 0.0342 | 0.1914 | 0.1788 | 0.1399 | 0.1019 |
| fund_gross_margin | neutral | 0.0186 | 0.1567 | 0.1184 | 0.1002 | 0.0651 |
| low_downside | neutral | 0.0248 | 0.2274 | 0.1092 | 0.1065 | 0.0604 |
| vol_confirm | bull | 0.0617 | 0.2019 | 0.3058 | 0.2348 | 0.1888 |
| relative_strength | bull | 0.0504 | 0.2048 | 0.2459 | 0.1970 | 0.1472 |
| ma_alignment | bull | 0.0383 | 0.2300 | 0.1666 | 0.1136 | 0.0928 |
| stroke_phase | bull | 0.0234 | 0.1466 | 0.1594 | 0.1364 | 0.0905 |
| ema20_slope | bull | 0.0335 | 0.2254 | 0.1488 | 0.1023 | 0.0820 |
| bb_width_20 | bear | 0.0866 | 0.2087 | 0.4149 | 0.4247 | 0.2955 |
| mom_x_lowvol_20_20 | bear | 0.0669 | 0.1756 | 0.3811 | 0.2603 | 0.2402 |
| fund_pb | bear | 0.0601 | 0.1813 | 0.3315 | 0.2329 | 0.2043 |
| momentum_reversal | bear | 0.0867 | 0.2964 | 0.2925 | 0.2877 | 0.1883 |
| fund_pe | bear | 0.0642 | 0.2130 | 0.3015 | 0.1507 | 0.1735 |
| wash_sale_score | bear | 0.0253 | 0.1000 | 0.2527 | 0.1515 | 0.1455 |

### 统一大市场

- **Neutral**: ['mom_x_lowvol_20_20', 'momentum_reversal', 'trend_lowvol'] (单因子IC=0.092, 组合IC=0.0994)
  - weights: [0.3565, 0.3357, 0.3078]
- **Bull**: ['low_downside', 'turnover_stability'] (单因子IC=0.0992, 组合IC=0.1279)
  - bull_weights: [0.6259, 0.3741]
- **Bear**: ['fund_pe', 'mom_x_lowvol_20_20', 'momentum_reversal'] (单因子IC=0.1639, 组合IC=0.2207)
  - bear_weights: [0.4487, 0.2791, 0.2722]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| mom_x_lowvol_20_20 | neutral | 0.0919 | 0.2122 | 0.4332 | 0.3507 | 0.2926 |
| momentum_reversal | neutral | 0.0924 | 0.2233 | 0.4138 | 0.3319 | 0.2756 |
| trend_lowvol | neutral | 0.0915 | 0.2378 | 0.3849 | 0.3132 | 0.2527 |
| fund_pe | neutral | 0.0832 | 0.2261 | 0.3680 | 0.2756 | 0.2347 |
| fund_pb | neutral | 0.0687 | 0.2047 | 0.3354 | 0.3340 | 0.2237 |
| volatility | neutral | 0.0708 | 0.2223 | 0.3182 | 0.2766 | 0.2031 |
| rsi_vol_combo | neutral | 0.0658 | 0.2200 | 0.2992 | 0.2192 | 0.1824 |
| turnover_stability | neutral | 0.0541 | 0.1879 | 0.2881 | 0.2463 | 0.1796 |
| low_downside | neutral | 0.0427 | 0.2114 | 0.2018 | 0.1795 | 0.1190 |
| fund_profit_growth | neutral | 0.0354 | 0.1869 | 0.1893 | 0.1566 | 0.1095 |
| fund_roe | neutral | 0.0324 | 0.2091 | 0.1552 | 0.1754 | 0.0912 |
| fund_score | neutral | 0.0280 | 0.2058 | 0.1359 | 0.1065 | 0.0752 |
| low_downside | bull | 0.1231 | 0.1843 | 0.6679 | 0.4621 | 0.4883 |
| turnover_stability | bull | 0.0753 | 0.1661 | 0.4532 | 0.2879 | 0.2918 |
| volatility | bull | 0.0784 | 0.1872 | 0.4188 | 0.3030 | 0.2729 |
| fund_pb | bull | 0.0847 | 0.2116 | 0.4000 | 0.3030 | 0.2606 |
| fund_pe | bull | 0.0836 | 0.2230 | 0.3747 | 0.2576 | 0.2356 |
| momentum_reversal | bull | 0.0724 | 0.1960 | 0.3695 | 0.2197 | 0.2254 |
| fund_revenue_growth | bull | 0.0513 | 0.1622 | 0.3163 | 0.3333 | 0.2109 |
| mom_x_lowvol_20_20 | bull | 0.0614 | 0.1931 | 0.3178 | 0.2348 | 0.1962 |
| rsi_vol_combo | bull | 0.0422 | 0.1994 | 0.2117 | 0.1288 | 0.1195 |
| fund_score | bull | 0.0317 | 0.2123 | 0.1494 | 0.1136 | 0.0832 |
| fund_roe | bull | 0.0228 | 0.2156 | 0.1056 | 0.1515 | 0.0608 |
| fund_pe | bear | 0.1839 | 0.2016 | 0.9118 | 0.6986 | 0.7744 |
| mom_x_lowvol_20_20 | bear | 0.1489 | 0.2201 | 0.6762 | 0.4247 | 0.4817 |
| momentum_reversal | bear | 0.1589 | 0.2478 | 0.6411 | 0.4658 | 0.4699 |
| rsi_vol_combo | bear | 0.1368 | 0.2063 | 0.6631 | 0.3973 | 0.4632 |
| fund_pb | bear | 0.1202 | 0.2532 | 0.4746 | 0.3151 | 0.3121 |
| trend_lowvol | bear | 0.0830 | 0.2128 | 0.3900 | 0.3973 | 0.2724 |
| fund_roe | bear | 0.0784 | 0.1904 | 0.4119 | 0.2877 | 0.2652 |
| turnover_stability | bear | 0.0499 | 0.1642 | 0.3039 | 0.1781 | 0.1790 |
| fund_revenue_growth | bear | 0.0458 | 0.1824 | 0.2510 | 0.2877 | 0.1616 |
| fund_score | bear | 0.0472 | 0.1855 | 0.2545 | 0.2329 | 0.1569 |

### 维生素

- **Neutral**: ['fund_pb', 'fund_pe', 'volatility'] (单因子IC=0.079, 组合IC=0.1027)
  - weights: [0.4649, 0.2936, 0.2415]
- **Bull**: ['fund_pb', 'trend_lowvol', 'momentum_reversal'] (单因子IC=0.0795, 组合IC=0.115)
  - bull_weights: [0.3849, 0.3493, 0.2659]
- **Bear**: ['fund_score', 'fund_profit_growth', 'fund_revenue_growth'] (单因子IC=0.158, 组合IC=0.1969)
  - bear_weights: [0.3853, 0.3326, 0.282]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| fund_pb | neutral | 0.1014 | 0.2336 | 0.4341 | 0.3111 | 0.2845 |
| fund_pe | neutral | 0.0732 | 0.2412 | 0.3037 | 0.1837 | 0.1797 |
| volatility | neutral | 0.0623 | 0.2506 | 0.2486 | 0.1889 | 0.1478 |
| trend_lowvol | neutral | 0.0556 | 0.2376 | 0.2340 | 0.1722 | 0.1371 |
| fund_profit_growth | neutral | 0.0480 | 0.2246 | 0.2139 | 0.1211 | 0.1199 |
| low_downside | neutral | 0.0415 | 0.2366 | 0.1754 | 0.1587 | 0.1016 |
| fund_revenue_growth | neutral | 0.0248 | 0.2280 | 0.1087 | 0.1096 | 0.0603 |
| fund_pb | bull | 0.0926 | 0.2063 | 0.4491 | 0.3030 | 0.2926 |
| trend_lowvol | bull | 0.0747 | 0.1856 | 0.4028 | 0.3182 | 0.2655 |
| momentum_reversal | bull | 0.0712 | 0.2216 | 0.3214 | 0.2576 | 0.2021 |
| rsi_vol_combo | bull | 0.0658 | 0.2256 | 0.2914 | 0.1970 | 0.1744 |
| low_downside | bull | 0.0520 | 0.2153 | 0.2414 | 0.1667 | 0.1408 |
| turnover_stability | bull | 0.0360 | 0.2054 | 0.1755 | 0.1705 | 0.1027 |
| fund_pe | bull | 0.0401 | 0.2443 | 0.1643 | 0.1364 | 0.0933 |
| fund_profit_growth | bull | 0.0280 | 0.1909 | 0.1466 | 0.1061 | 0.0811 |
| fund_score | bear | 0.1882 | 0.2460 | 0.7651 | 0.6164 | 0.6184 |
| fund_profit_growth | bear | 0.1713 | 0.2418 | 0.7085 | 0.5068 | 0.5338 |
| fund_revenue_growth | bear | 0.1145 | 0.1906 | 0.6007 | 0.5068 | 0.4526 |
| stroke_phase | bear | 0.1341 | 0.2720 | 0.4928 | 0.3151 | 0.3241 |
| turnover_stability | bear | 0.1074 | 0.2529 | 0.4248 | 0.3151 | 0.2793 |
| fund_roe | bear | 0.0872 | 0.2142 | 0.4070 | 0.3699 | 0.2788 |
| fund_gross_margin | bear | 0.0896 | 0.2528 | 0.3543 | 0.1507 | 0.2038 |
| trend_lowvol | bear | 0.0578 | 0.2709 | 0.2135 | 0.1507 | 0.1228 |
| mom_x_lowvol_20_20 | bear | 0.0497 | 0.2536 | 0.1961 | 0.2329 | 0.1209 |

### 绿色电力

- **Neutral**: ['fund_pb', 'momentum_reversal'] (单因子IC=0.0761, 组合IC=0.1074)
  - weights: [0.5585, 0.4415]
- **Bull**: ['turnover_stability', 'fund_pb'] (单因子IC=0.0564, 组合IC=0.0812)
  - bull_weights: [0.5058, 0.4942]
- **Bear**: ['mom_x_lowvol_20_20', 'momentum_reversal'] (单因子IC=0.1314, 组合IC=0.1337)
  - bear_weights: [0.5336, 0.4664]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| fund_pb | neutral | 0.0723 | 0.1524 | 0.4742 | 0.2902 | 0.3059 |
| momentum_reversal | neutral | 0.0799 | 0.2087 | 0.3830 | 0.2630 | 0.2419 |
| volatility | neutral | 0.0691 | 0.2004 | 0.3449 | 0.3069 | 0.2254 |
| mom_x_lowvol_20_20 | neutral | 0.0656 | 0.1839 | 0.3569 | 0.2589 | 0.2247 |
| fund_pe | neutral | 0.0734 | 0.2016 | 0.3641 | 0.2276 | 0.2235 |
| fund_profit_growth | neutral | 0.0427 | 0.1287 | 0.3319 | 0.3027 | 0.2162 |
| trend_lowvol | neutral | 0.0795 | 0.2375 | 0.3347 | 0.2317 | 0.2061 |
| turnover_stability | neutral | 0.0394 | 0.1219 | 0.3231 | 0.2422 | 0.2007 |
| fund_score | neutral | 0.0442 | 0.1419 | 0.3114 | 0.2484 | 0.1944 |
| rsi_vol_combo | neutral | 0.0622 | 0.1983 | 0.3139 | 0.2192 | 0.1913 |
| low_downside | neutral | 0.0462 | 0.2210 | 0.2092 | 0.2004 | 0.1256 |
| fund_roe | neutral | 0.0340 | 0.1793 | 0.1895 | 0.1357 | 0.1076 |
| fund_revenue_growth | neutral | 0.0163 | 0.1070 | 0.1525 | 0.1190 | 0.0853 |
| turnover_stability | bull | 0.0493 | 0.1218 | 0.4047 | 0.3409 | 0.2713 |
| fund_pb | bull | 0.0636 | 0.1535 | 0.4142 | 0.2803 | 0.2651 |
| volatility | bull | 0.0565 | 0.1753 | 0.3221 | 0.1742 | 0.1891 |
| low_downside | bull | 0.0578 | 0.1825 | 0.3166 | 0.1667 | 0.1847 |
| fund_pe | bull | 0.0477 | 0.2006 | 0.2379 | 0.1742 | 0.1397 |
| stroke_phase | bull | 0.0235 | 0.1128 | 0.2084 | 0.2121 | 0.1263 |
| trend_lowvol | bull | 0.0349 | 0.1812 | 0.1927 | 0.1061 | 0.1066 |
| fund_gross_margin | bull | 0.0150 | 0.1194 | 0.1261 | 0.1136 | 0.0702 |
| mom_x_lowvol_20_20 | bear | 0.1336 | 0.2404 | 0.5556 | 0.5068 | 0.4186 |
| momentum_reversal | bear | 0.1292 | 0.2467 | 0.5236 | 0.3973 | 0.3658 |
| rsi_vol_combo | bear | 0.0845 | 0.1934 | 0.4367 | 0.4247 | 0.3111 |
| fund_pb | bear | 0.0880 | 0.1986 | 0.4430 | 0.1507 | 0.2549 |
| trend_lowvol | bear | 0.0773 | 0.2380 | 0.3248 | 0.1507 | 0.1868 |

### 网红经济

- **Neutral**: ['fund_pb', 'trend_lowvol', 'volatility'] (单因子IC=0.0936, 组合IC=0.1239)
  - weights: [0.4479, 0.2784, 0.2737]
- **Bull**: ['fund_pb', 'volatility', 'low_downside'] (单因子IC=0.0954, 组合IC=0.1217)
  - bull_weights: [0.4117, 0.3125, 0.2757]
- **Bear**: ['mom_x_lowvol_20_20'] (单因子IC=0.1542, 组合IC=0.1542)
  - bear_weights: [1.0]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| fund_pb | neutral | 0.0989 | 0.1596 | 0.6199 | 0.4384 | 0.4458 |
| trend_lowvol | neutral | 0.0894 | 0.2112 | 0.4235 | 0.3090 | 0.2771 |
| volatility | neutral | 0.0925 | 0.2229 | 0.4150 | 0.3132 | 0.2725 |
| momentum_reversal | neutral | 0.0822 | 0.1994 | 0.4125 | 0.2944 | 0.2670 |
| mom_x_lowvol_20_20 | neutral | 0.0813 | 0.1954 | 0.4160 | 0.2651 | 0.2631 |
| rsi_vol_combo | neutral | 0.0646 | 0.1889 | 0.3419 | 0.2610 | 0.2156 |
| fund_pe | neutral | 0.0599 | 0.1836 | 0.3263 | 0.2443 | 0.2030 |
| low_downside | neutral | 0.0572 | 0.1992 | 0.2871 | 0.2234 | 0.1756 |
| fund_profit_growth | neutral | 0.0338 | 0.1256 | 0.2688 | 0.2380 | 0.1664 |
| turnover_stability | neutral | 0.0303 | 0.1192 | 0.2544 | 0.1691 | 0.1487 |
| fund_pb | bull | 0.1090 | 0.1548 | 0.7043 | 0.5530 | 0.5469 |
| volatility | bull | 0.0996 | 0.1745 | 0.5708 | 0.4545 | 0.4152 |
| low_downside | bull | 0.0775 | 0.1474 | 0.5255 | 0.3939 | 0.3663 |
| momentum_reversal | bull | 0.0574 | 0.1482 | 0.3871 | 0.2803 | 0.2478 |
| rsi_vol_combo | bull | 0.0547 | 0.1498 | 0.3652 | 0.3030 | 0.2380 |
| mom_x_lowvol_20_20 | bull | 0.0567 | 0.1508 | 0.3757 | 0.2652 | 0.2376 |
| fund_pe | bull | 0.0574 | 0.1776 | 0.3231 | 0.1667 | 0.1885 |
| trend_lowvol | bull | 0.0522 | 0.1934 | 0.2697 | 0.2121 | 0.1634 |
| fund_gross_margin | bull | 0.0302 | 0.1499 | 0.2015 | 0.1364 | 0.1145 |
| turnover_stability | bull | 0.0246 | 0.1297 | 0.1894 | 0.1364 | 0.1076 |
| mom_x_lowvol_20_20 | bear | 0.1542 | 0.2082 | 0.7405 | 0.5890 | 0.5884 |
| momentum_reversal | bear | 0.1441 | 0.2186 | 0.6592 | 0.5068 | 0.4967 |
| trend_lowvol | bear | 0.0913 | 0.1677 | 0.5448 | 0.3973 | 0.3806 |
| fund_revenue_growth | bear | 0.0513 | 0.1065 | 0.4814 | 0.4247 | 0.3429 |
| rsi_vol_combo | bear | 0.0853 | 0.2006 | 0.4252 | 0.3425 | 0.2854 |
| fund_profit_growth | bear | 0.0478 | 0.1246 | 0.3841 | 0.3151 | 0.2526 |
| turnover_stability | bear | 0.0392 | 0.1321 | 0.2965 | 0.1781 | 0.1747 |
| fund_score | bear | 0.0405 | 0.1515 | 0.2674 | 0.1507 | 0.1538 |
| bb_width_20 | bear | 0.0538 | 0.2309 | 0.2331 | 0.2877 | 0.1501 |

### 网络安全

- **Neutral**: ['fund_pb', 'momentum_reversal', 'mom_x_lowvol_20_20'] (单因子IC=0.0902, 组合IC=0.1265)
  - weights: [0.4049, 0.3153, 0.2798]
- **Bull**: ['fund_pb', 'volatility', 'low_downside'] (单因子IC=0.108, 组合IC=0.1329)
  - bull_weights: [0.3853, 0.3623, 0.2523]
- **Bear**: ['trend_lowvol', 'fund_profit_growth'] (单因子IC=0.0845, 组合IC=0.1131)
  - bear_weights: [0.5288, 0.4712]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| fund_pb | neutral | 0.0923 | 0.1319 | 0.6996 | 0.4990 | 0.5243 |
| momentum_reversal | neutral | 0.0940 | 0.1628 | 0.5771 | 0.4154 | 0.4084 |
| mom_x_lowvol_20_20 | neutral | 0.0845 | 0.1608 | 0.5256 | 0.3789 | 0.3623 |
| trend_lowvol | neutral | 0.0935 | 0.1974 | 0.4737 | 0.4008 | 0.3318 |
| rsi_vol_combo | neutral | 0.0786 | 0.1661 | 0.4734 | 0.3967 | 0.3306 |
| volatility | neutral | 0.0815 | 0.1843 | 0.4421 | 0.3674 | 0.3023 |
| fund_pe | neutral | 0.0697 | 0.1767 | 0.3942 | 0.2735 | 0.2510 |
| fund_profit_growth | neutral | 0.0485 | 0.1258 | 0.3854 | 0.2589 | 0.2426 |
| fund_score | neutral | 0.0513 | 0.1766 | 0.2905 | 0.2129 | 0.1762 |
| turnover_stability | neutral | 0.0295 | 0.1193 | 0.2471 | 0.2088 | 0.1494 |
| fund_roe | neutral | 0.0448 | 0.1868 | 0.2399 | 0.2150 | 0.1457 |
| low_downside | neutral | 0.0424 | 0.1838 | 0.2308 | 0.1962 | 0.1380 |
| fund_revenue_growth | neutral | 0.0174 | 0.1181 | 0.1469 | 0.1044 | 0.0811 |
| fund_pb | bull | 0.1065 | 0.1388 | 0.7669 | 0.5833 | 0.6071 |
| volatility | bull | 0.1223 | 0.1640 | 0.7461 | 0.5303 | 0.5709 |
| low_downside | bull | 0.0951 | 0.1704 | 0.5583 | 0.4242 | 0.3976 |
| mom_x_lowvol_20_20 | bull | 0.0835 | 0.1498 | 0.5576 | 0.3788 | 0.3844 |
| momentum_reversal | bull | 0.0717 | 0.1482 | 0.4835 | 0.3864 | 0.3352 |
| trend_lowvol | bull | 0.0865 | 0.1796 | 0.4817 | 0.3864 | 0.3339 |
| fund_pe | bull | 0.0723 | 0.1612 | 0.4483 | 0.3258 | 0.2972 |
| fund_score | bull | 0.0603 | 0.1601 | 0.3763 | 0.1970 | 0.2252 |
| turnover_stability | bull | 0.0413 | 0.1248 | 0.3307 | 0.3030 | 0.2154 |
| rsi_vol_combo | bull | 0.0436 | 0.1408 | 0.3095 | 0.2727 | 0.1970 |
| fund_roe | bull | 0.0530 | 0.1706 | 0.3109 | 0.1742 | 0.1826 |
| fund_revenue_growth | bull | 0.0352 | 0.1164 | 0.3020 | 0.1667 | 0.1761 |
| trend_lowvol | bear | 0.1085 | 0.1867 | 0.5811 | 0.3425 | 0.3901 |
| fund_profit_growth | bear | 0.0605 | 0.1240 | 0.4880 | 0.4247 | 0.3476 |
| fund_revenue_growth | bear | 0.0503 | 0.1151 | 0.4372 | 0.3699 | 0.2994 |
| mom_x_lowvol_20_20 | bear | 0.0862 | 0.1947 | 0.4430 | 0.3151 | 0.2913 |
| momentum_reversal | bear | 0.0903 | 0.2271 | 0.3975 | 0.2877 | 0.2560 |
| fund_score | bear | 0.0560 | 0.1727 | 0.3239 | 0.1781 | 0.1908 |
| rsi_vol_combo | bear | 0.0502 | 0.2037 | 0.2466 | 0.2329 | 0.1520 |

### 网络游戏

- **Neutral**: ['fund_pb', 'momentum_reversal', 'mom_x_lowvol_20_20'] (单因子IC=0.0822, 组合IC=0.1139)
  - weights: [0.3888, 0.3089, 0.3023]
- **Bull**: ['low_downside', 'fund_profit_growth'] (单因子IC=0.0652, 组合IC=0.0859)
  - bull_weights: [0.5951, 0.4049]
- **Bear**: ['trend_lowvol', 'mom_x_lowvol_20_20'] (单因子IC=0.1204, 组合IC=0.1443)
  - bear_weights: [0.5433, 0.4567]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| fund_pb | neutral | 0.0903 | 0.1911 | 0.4724 | 0.2965 | 0.3062 |
| momentum_reversal | neutral | 0.0790 | 0.2102 | 0.3759 | 0.2944 | 0.2432 |
| mom_x_lowvol_20_20 | neutral | 0.0774 | 0.2125 | 0.3644 | 0.3069 | 0.2381 |
| trend_lowvol | neutral | 0.0797 | 0.2415 | 0.3300 | 0.3132 | 0.2167 |
| volatility | neutral | 0.0844 | 0.2430 | 0.3474 | 0.2422 | 0.2158 |
| rsi_vol_combo | neutral | 0.0647 | 0.1939 | 0.3338 | 0.2526 | 0.2090 |
| fund_profit_growth | neutral | 0.0523 | 0.1528 | 0.3423 | 0.2067 | 0.2065 |
| fund_score | neutral | 0.0610 | 0.1967 | 0.3102 | 0.2547 | 0.1946 |
| fund_revenue_growth | neutral | 0.0447 | 0.1476 | 0.3031 | 0.2296 | 0.1863 |
| fund_pe | neutral | 0.0587 | 0.2161 | 0.2717 | 0.2067 | 0.1639 |
| low_downside | neutral | 0.0502 | 0.2238 | 0.2244 | 0.1587 | 0.1300 |
| turnover_stability | neutral | 0.0233 | 0.1548 | 0.1506 | 0.1086 | 0.0835 |
| low_downside | bull | 0.0817 | 0.1717 | 0.4759 | 0.4091 | 0.3353 |
| fund_profit_growth | bull | 0.0487 | 0.1390 | 0.3501 | 0.3030 | 0.2281 |
| fund_revenue_growth | bull | 0.0465 | 0.1374 | 0.3386 | 0.2424 | 0.2104 |
| volatility | bull | 0.0667 | 0.2052 | 0.3250 | 0.2727 | 0.2068 |
| trend_lowvol | bull | 0.0626 | 0.2190 | 0.2860 | 0.2348 | 0.1766 |
| fund_pb | bull | 0.0438 | 0.1835 | 0.2387 | 0.1894 | 0.1419 |
| turnover_stability | bull | 0.0358 | 0.1529 | 0.2342 | 0.1288 | 0.1322 |
| mom_x_lowvol_20_20 | bull | 0.0363 | 0.1984 | 0.1832 | 0.1667 | 0.1068 |
| momentum_reversal | bull | 0.0379 | 0.2058 | 0.1843 | 0.1515 | 0.1061 |
| fund_pe | bull | 0.0245 | 0.2169 | 0.1130 | 0.1136 | 0.0629 |
| trend_lowvol | bear | 0.1179 | 0.1851 | 0.6372 | 0.5616 | 0.4975 |
| mom_x_lowvol_20_20 | bear | 0.1229 | 0.2174 | 0.5654 | 0.4795 | 0.4182 |
| momentum_reversal | bear | 0.1187 | 0.2149 | 0.5524 | 0.4247 | 0.3935 |
| fund_revenue_growth | bear | 0.0731 | 0.1486 | 0.4921 | 0.3699 | 0.3370 |
| rsi_vol_combo | bear | 0.0548 | 0.1886 | 0.2907 | 0.2603 | 0.1832 |
| bb_width_20 | bear | 0.0533 | 0.2007 | 0.2657 | 0.1507 | 0.1528 |
| turnover_stability | bear | 0.0331 | 0.1430 | 0.2311 | 0.2055 | 0.1393 |

### 职业教育

- **Neutral**: ['fund_pb', 'trend_lowvol', 'volatility'] (单因子IC=0.0977, 组合IC=0.1289)
  - weights: [0.3817, 0.312, 0.3063]
- **Bull**: ['fund_pb', 'low_downside', 'volatility'] (单因子IC=0.1157, 组合IC=0.1436)
  - bull_weights: [0.3697, 0.3393, 0.291]
- **Bear**: ['trend_lowvol', 'mom_x_lowvol_20_20'] (单因子IC=0.1697, 组合IC=0.1842)
  - bear_weights: [0.5143, 0.4857]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| fund_pb | neutral | 0.0878 | 0.1499 | 0.5858 | 0.4029 | 0.4109 |
| trend_lowvol | neutral | 0.0995 | 0.2062 | 0.4824 | 0.3925 | 0.3358 |
| volatility | neutral | 0.1057 | 0.2262 | 0.4673 | 0.4113 | 0.3298 |
| momentum_reversal | neutral | 0.0916 | 0.2136 | 0.4291 | 0.3361 | 0.2867 |
| mom_x_lowvol_20_20 | neutral | 0.0897 | 0.2134 | 0.4203 | 0.3069 | 0.2746 |
| rsi_vol_combo | neutral | 0.0731 | 0.1954 | 0.3739 | 0.3048 | 0.2440 |
| low_downside | neutral | 0.0577 | 0.2143 | 0.2692 | 0.2714 | 0.1712 |
| turnover_stability | neutral | 0.0402 | 0.1436 | 0.2800 | 0.2192 | 0.1707 |
| fund_pe | neutral | 0.0564 | 0.2113 | 0.2671 | 0.2338 | 0.1648 |
| fund_revenue_growth | neutral | 0.0305 | 0.1429 | 0.2133 | 0.2067 | 0.1287 |
| fund_score | neutral | 0.0313 | 0.1651 | 0.1896 | 0.1086 | 0.1051 |
| fund_roe | neutral | 0.0228 | 0.1746 | 0.1307 | 0.1148 | 0.0729 |
| fund_pb | bull | 0.1031 | 0.1371 | 0.7516 | 0.5909 | 0.5979 |
| low_downside | bull | 0.1160 | 0.1673 | 0.6931 | 0.5833 | 0.5487 |
| volatility | bull | 0.1280 | 0.2060 | 0.6213 | 0.5152 | 0.4707 |
| trend_lowvol | bull | 0.1059 | 0.2310 | 0.4582 | 0.3864 | 0.3176 |
| mom_x_lowvol_20_20 | bull | 0.0877 | 0.2073 | 0.4232 | 0.4015 | 0.2965 |
| momentum_reversal | bull | 0.0724 | 0.2082 | 0.3478 | 0.3333 | 0.2319 |
| turnover_stability | bull | 0.0505 | 0.1521 | 0.3322 | 0.3409 | 0.2227 |
| stroke_phase | bull | 0.0350 | 0.1727 | 0.2024 | 0.1591 | 0.1173 |
| rsi_vol_combo | bull | 0.0284 | 0.1854 | 0.1532 | 0.2121 | 0.0929 |
| trend_lowvol | bear | 0.1633 | 0.1780 | 0.9173 | 0.6712 | 0.7665 |
| mom_x_lowvol_20_20 | bear | 0.1762 | 0.2034 | 0.8663 | 0.6712 | 0.7239 |
| momentum_reversal | bear | 0.1682 | 0.1941 | 0.8661 | 0.6438 | 0.7119 |
| rsi_vol_combo | bear | 0.1196 | 0.1742 | 0.6865 | 0.4247 | 0.4890 |
| turnover_stability | bear | 0.0721 | 0.1315 | 0.5480 | 0.3151 | 0.3603 |
| fund_gross_margin | bear | 0.0411 | 0.2009 | 0.2044 | 0.2603 | 0.1288 |

### 肝炎概念

- **Neutral**: ['volatility'] (单因子IC=0.1049, 组合IC=0.1049)
  - weights: [1.0]
- **Bull**: ['low_downside', 'volatility', 'fund_pb'] (单因子IC=0.0611, 组合IC=0.068)
  - bull_weights: [0.3695, 0.3158, 0.3147]
- **Bear**: ['mom_x_lowvol_20_20', 'momentum_reversal'] (单因子IC=0.1337, 组合IC=0.1362)
  - bear_weights: [0.5187, 0.4813]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| volatility | neutral | 0.1049 | 0.2248 | 0.4667 | 0.3967 | 0.3259 |
| fund_pe | neutral | 0.0668 | 0.1892 | 0.3531 | 0.2944 | 0.2285 |
| fund_pb | neutral | 0.0737 | 0.2071 | 0.3561 | 0.2171 | 0.2167 |
| fund_profit_growth | neutral | 0.0562 | 0.1753 | 0.3205 | 0.2975 | 0.2080 |
| momentum_reversal | neutral | 0.0691 | 0.2276 | 0.3034 | 0.1962 | 0.1815 |
| fund_revenue_growth | neutral | 0.0498 | 0.1894 | 0.2631 | 0.2276 | 0.1615 |
| trend_lowvol | neutral | 0.0634 | 0.2399 | 0.2642 | 0.2046 | 0.1591 |
| turnover_stability | neutral | 0.0420 | 0.1630 | 0.2576 | 0.1754 | 0.1514 |
| low_downside | neutral | 0.0533 | 0.2138 | 0.2494 | 0.1733 | 0.1463 |
| rsi_vol_combo | neutral | 0.0513 | 0.2117 | 0.2421 | 0.1336 | 0.1372 |
| mom_x_lowvol_20_20 | neutral | 0.0517 | 0.2253 | 0.2295 | 0.1555 | 0.1326 |
| fund_score | neutral | 0.0394 | 0.2121 | 0.1858 | 0.1942 | 0.1110 |
| low_downside | bull | 0.0629 | 0.1886 | 0.3336 | 0.2727 | 0.2123 |
| volatility | bull | 0.0565 | 0.1889 | 0.2994 | 0.2121 | 0.1814 |
| fund_pb | bull | 0.0640 | 0.2024 | 0.3161 | 0.1439 | 0.1808 |
| turnover_stability | bull | 0.0232 | 0.1620 | 0.1433 | 0.1439 | 0.0820 |
| stroke_phase | bull | 0.0161 | 0.1600 | 0.1008 | 0.1364 | 0.0573 |
| mom_x_lowvol_20_20 | bear | 0.1327 | 0.2182 | 0.6079 | 0.5342 | 0.4663 |
| momentum_reversal | bear | 0.1346 | 0.2344 | 0.5743 | 0.5068 | 0.4327 |
| rsi_vol_combo | bear | 0.1210 | 0.2183 | 0.5543 | 0.3973 | 0.3873 |
| trend_lowvol | bear | 0.1288 | 0.2242 | 0.5743 | 0.3425 | 0.3855 |
| fund_profit_growth | bear | 0.0523 | 0.1496 | 0.3495 | 0.4247 | 0.2489 |
| fund_pb | bear | 0.0567 | 0.1510 | 0.3753 | 0.3151 | 0.2467 |
| fund_score | bear | 0.0454 | 0.1948 | 0.2333 | 0.3151 | 0.1534 |
| fund_revenue_growth | bear | 0.0312 | 0.1394 | 0.2242 | 0.1781 | 0.1320 |

### 股权激励

- **Neutral**: ['fund_pb', 'momentum_reversal', 'trend_lowvol'] (单因子IC=0.0624, 组合IC=0.0876)
  - weights: [0.3651, 0.3182, 0.3167]
- **Bull**: ['low_downside', 'fund_pb'] (单因子IC=0.0677, 组合IC=0.0819)
  - bull_weights: [0.5376, 0.4624]
- **Bear**: ['trend_lowvol', 'mom_x_lowvol_20_20'] (单因子IC=0.0818, 组合IC=0.0946)
  - bear_weights: [0.5232, 0.4768]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| fund_pb | neutral | 0.0674 | 0.1746 | 0.3859 | 0.2547 | 0.2421 |
| momentum_reversal | neutral | 0.0565 | 0.1669 | 0.3386 | 0.2463 | 0.2110 |
| trend_lowvol | neutral | 0.0633 | 0.1876 | 0.3375 | 0.2443 | 0.2100 |
| fund_profit_growth | neutral | 0.0436 | 0.1285 | 0.3389 | 0.2255 | 0.2077 |
| mom_x_lowvol_20_20 | neutral | 0.0541 | 0.1615 | 0.3352 | 0.2213 | 0.2047 |
| volatility | neutral | 0.0582 | 0.1886 | 0.3084 | 0.2839 | 0.1980 |
| fund_pe | neutral | 0.0517 | 0.1810 | 0.2858 | 0.1942 | 0.1707 |
| rsi_vol_combo | neutral | 0.0394 | 0.1579 | 0.2492 | 0.1837 | 0.1475 |
| turnover_stability | neutral | 0.0205 | 0.0924 | 0.2217 | 0.2443 | 0.1379 |
| fund_revenue_growth | neutral | 0.0284 | 0.1273 | 0.2229 | 0.1524 | 0.1284 |
| fund_score | neutral | 0.0340 | 0.1664 | 0.2046 | 0.1253 | 0.1151 |
| low_downside | neutral | 0.0282 | 0.1780 | 0.1584 | 0.1608 | 0.0919 |
| low_downside | bull | 0.0628 | 0.1408 | 0.4460 | 0.3636 | 0.3041 |
| fund_pb | bull | 0.0727 | 0.1841 | 0.3947 | 0.3258 | 0.2616 |
| volatility | bull | 0.0611 | 0.1570 | 0.3893 | 0.2955 | 0.2521 |
| turnover_stability | bull | 0.0263 | 0.0904 | 0.2906 | 0.2576 | 0.1827 |
| trend_lowvol | bull | 0.0500 | 0.1866 | 0.2679 | 0.1515 | 0.1543 |
| momentum_reversal | bull | 0.0338 | 0.1591 | 0.2128 | 0.1364 | 0.1209 |
| wash_sale_score | bull | 0.0157 | 0.0835 | 0.1882 | 0.1527 | 0.1084 |
| mom_x_lowvol_20_20 | bull | 0.0289 | 0.1602 | 0.1803 | 0.1061 | 0.0997 |
| fund_revenue_growth | bull | 0.0201 | 0.1241 | 0.1617 | 0.1061 | 0.0894 |
| stroke_phase | bull | 0.0105 | 0.1028 | 0.1018 | 0.1136 | 0.0567 |
| trend_lowvol | bear | 0.0853 | 0.1790 | 0.4766 | 0.2603 | 0.3003 |
| mom_x_lowvol_20_20 | bear | 0.0782 | 0.1879 | 0.4162 | 0.3151 | 0.2737 |
| momentum_reversal | bear | 0.0762 | 0.1845 | 0.4132 | 0.3151 | 0.2717 |
| fund_revenue_growth | bear | 0.0475 | 0.1222 | 0.3883 | 0.2877 | 0.2500 |
| rsi_vol_combo | bear | 0.0484 | 0.1588 | 0.3045 | 0.2055 | 0.1835 |
| fund_profit_growth | bear | 0.0303 | 0.1373 | 0.2207 | 0.1781 | 0.1300 |

### 股权转让

- **Neutral**: ['trend_lowvol'] (单因子IC=0.1094, 组合IC=0.1094)
  - weights: [1.0]
- **Bull**: ['trend_lowvol'] (单因子IC=0.1619, 组合IC=0.1619)
  - bull_weights: [1.0]
- **Bear**: ['mom_x_lowvol_20_20'] (单因子IC=0.1532, 组合IC=0.1532)
  - bear_weights: [1.0]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| trend_lowvol | neutral | 0.1094 | 0.2273 | 0.4811 | 0.3559 | 0.3262 |
| momentum_reversal | neutral | 0.0918 | 0.2197 | 0.4178 | 0.3038 | 0.2724 |
| mom_x_lowvol_20_20 | neutral | 0.0829 | 0.2230 | 0.3718 | 0.3184 | 0.2451 |
| rsi_vol_combo | neutral | 0.0684 | 0.2087 | 0.3276 | 0.2203 | 0.1999 |
| fund_pb | neutral | 0.0562 | 0.1759 | 0.3193 | 0.2370 | 0.1975 |
| volatility | neutral | 0.0638 | 0.2392 | 0.2669 | 0.2276 | 0.1638 |
| fund_profit_growth | neutral | 0.0480 | 0.1797 | 0.2669 | 0.2025 | 0.1605 |
| fund_score | neutral | 0.0398 | 0.2034 | 0.1957 | 0.1618 | 0.1137 |
| low_downside | neutral | 0.0420 | 0.2298 | 0.1827 | 0.1482 | 0.1049 |
| turnover_stability | neutral | 0.0306 | 0.1835 | 0.1669 | 0.1701 | 0.0976 |
| fund_pe | neutral | 0.0337 | 0.2309 | 0.1459 | 0.1273 | 0.0822 |
| fund_roe | neutral | 0.0281 | 0.2106 | 0.1332 | 0.1211 | 0.0747 |
| trend_lowvol | bull | 0.1619 | 0.2428 | 0.6669 | 0.4773 | 0.4926 |
| momentum_reversal | bull | 0.1203 | 0.2257 | 0.5332 | 0.4318 | 0.3817 |
| mom_x_lowvol_20_20 | bull | 0.1104 | 0.2237 | 0.4935 | 0.3864 | 0.3421 |
| low_downside | bull | 0.1228 | 0.2396 | 0.5124 | 0.3333 | 0.3416 |
| volatility | bull | 0.1250 | 0.2740 | 0.4560 | 0.3485 | 0.3075 |
| fund_pb | bull | 0.0835 | 0.1993 | 0.4189 | 0.3182 | 0.2761 |
| turnover_stability | bull | 0.0882 | 0.2113 | 0.4177 | 0.3030 | 0.2721 |
| rsi_vol_combo | bull | 0.0528 | 0.2055 | 0.2568 | 0.2197 | 0.1566 |
| fund_revenue_growth | bull | 0.0325 | 0.2101 | 0.1546 | 0.1591 | 0.0896 |
| fund_score | bull | 0.0218 | 0.1991 | 0.1094 | 0.1742 | 0.0642 |
| mom_x_lowvol_20_20 | bear | 0.1532 | 0.1810 | 0.8464 | 0.6164 | 0.6841 |
| momentum_reversal | bear | 0.1503 | 0.1719 | 0.8745 | 0.5068 | 0.6589 |
| rsi_vol_combo | bear | 0.1148 | 0.1723 | 0.6660 | 0.4521 | 0.4836 |
| trend_lowvol | bear | 0.1281 | 0.2084 | 0.6149 | 0.4247 | 0.4380 |
| turnover_stability | bear | 0.0656 | 0.1867 | 0.3513 | 0.3973 | 0.2454 |
| fund_score | bear | 0.0652 | 0.2382 | 0.2740 | 0.2877 | 0.1764 |
| fund_gross_margin | bear | 0.0600 | 0.2260 | 0.2655 | 0.2877 | 0.1709 |
| fund_profit_growth | bear | 0.0568 | 0.2273 | 0.2498 | 0.1507 | 0.1437 |
| fund_revenue_growth | bear | 0.0517 | 0.2494 | 0.2072 | 0.1233 | 0.1164 |

### 胎压监测

- **Neutral**: ['trend_lowvol', 'mom_x_lowvol_20_20'] (单因子IC=0.1062, 组合IC=0.1266)
  - weights: [0.5872, 0.4128]
- **Bull**: ['trend_lowvol', 'mom_x_lowvol_20_20'] (单因子IC=0.1145, 组合IC=0.1335)
  - bull_weights: [0.5834, 0.4166]
- **Bear**: ['fund_profit_growth'] (单因子IC=0.0447, 组合IC=0.0447)
  - bear_weights: [1.0]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| trend_lowvol | neutral | 0.1207 | 0.2382 | 0.5067 | 0.3810 | 0.3499 |
| mom_x_lowvol_20_20 | neutral | 0.0916 | 0.2453 | 0.3735 | 0.3173 | 0.2460 |
| momentum_reversal | neutral | 0.0844 | 0.2461 | 0.3430 | 0.2797 | 0.2195 |
| volatility | neutral | 0.0757 | 0.2512 | 0.3015 | 0.2839 | 0.1935 |
| fund_pb | neutral | 0.0730 | 0.2475 | 0.2951 | 0.1921 | 0.1759 |
| rsi_vol_combo | neutral | 0.0632 | 0.2509 | 0.2517 | 0.1775 | 0.1482 |
| fund_pe | neutral | 0.0620 | 0.2527 | 0.2454 | 0.1691 | 0.1434 |
| fund_profit_growth | neutral | 0.0535 | 0.2322 | 0.2304 | 0.1649 | 0.1342 |
| low_downside | neutral | 0.0545 | 0.2431 | 0.2240 | 0.1816 | 0.1324 |
| fund_revenue_growth | neutral | 0.0375 | 0.2280 | 0.1646 | 0.1242 | 0.0925 |
| fund_score | neutral | 0.0364 | 0.2542 | 0.1433 | 0.1472 | 0.0822 |
| turnover_stability | neutral | 0.0294 | 0.2067 | 0.1423 | 0.1409 | 0.0812 |
| trend_lowvol | bull | 0.1210 | 0.2078 | 0.5821 | 0.4697 | 0.4278 |
| mom_x_lowvol_20_20 | bull | 0.1080 | 0.2397 | 0.4506 | 0.3561 | 0.3055 |
| volatility | bull | 0.1140 | 0.2523 | 0.4518 | 0.2955 | 0.2927 |
| momentum_reversal | bull | 0.0995 | 0.2388 | 0.4168 | 0.3409 | 0.2795 |
| fund_pb | bull | 0.0987 | 0.2692 | 0.3666 | 0.2803 | 0.2347 |
| rsi_vol_combo | bull | 0.0745 | 0.2152 | 0.3463 | 0.2652 | 0.2191 |
| turnover_stability | bull | 0.0617 | 0.1948 | 0.3165 | 0.2727 | 0.2014 |
| low_downside | bull | 0.0715 | 0.2205 | 0.3241 | 0.1970 | 0.1940 |
| fund_pe | bull | 0.0651 | 0.2676 | 0.2431 | 0.2652 | 0.1538 |
| fund_profit_growth | bear | 0.0447 | 0.2074 | 0.2156 | 0.2603 | 0.1358 |

### 腾讯云

- **Neutral**: ['momentum_reversal', 'trend_lowvol', 'fund_profit_growth'] (单因子IC=0.096, 组合IC=0.1148)
  - weights: [0.3661, 0.3317, 0.3021]
- **Bull**: ['volatility', 'trend_lowvol'] (单因子IC=0.0949, 组合IC=0.1024)
  - bull_weights: [0.5434, 0.4566]
- **Bear**: ['fund_pb', 'momentum_reversal', 'vol_opening_strength'] (单因子IC=0.1186, 组合IC=0.1822)
  - bear_weights: [0.5133, 0.2859, 0.2008]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| momentum_reversal | neutral | 0.1044 | 0.2868 | 0.3639 | 0.2714 | 0.2313 |
| trend_lowvol | neutral | 0.1018 | 0.3103 | 0.3281 | 0.2777 | 0.2096 |
| fund_profit_growth | neutral | 0.0819 | 0.2747 | 0.2980 | 0.2808 | 0.1909 |
| fund_pe | neutral | 0.0883 | 0.2966 | 0.2978 | 0.2728 | 0.1895 |
| mom_x_lowvol_20_20 | neutral | 0.0837 | 0.2795 | 0.2996 | 0.2370 | 0.1853 |
| fund_score | neutral | 0.0785 | 0.2939 | 0.2671 | 0.2234 | 0.1634 |
| fund_roe | neutral | 0.0782 | 0.3218 | 0.2430 | 0.2067 | 0.1466 |
| rsi_vol_combo | neutral | 0.0615 | 0.2796 | 0.2201 | 0.1472 | 0.1263 |
| volatility | neutral | 0.0583 | 0.2900 | 0.2009 | 0.1587 | 0.1164 |
| turnover_stability | neutral | 0.0416 | 0.2652 | 0.1568 | 0.1305 | 0.0886 |
| fund_pb | neutral | 0.0402 | 0.2656 | 0.1514 | 0.1294 | 0.0855 |
| volatility | bull | 0.0945 | 0.2621 | 0.3606 | 0.3333 | 0.2404 |
| trend_lowvol | bull | 0.0952 | 0.2965 | 0.3212 | 0.2576 | 0.2020 |
| fund_pe | bull | 0.0740 | 0.2878 | 0.2572 | 0.2374 | 0.1591 |
| fund_pb | bull | 0.0615 | 0.2389 | 0.2574 | 0.1250 | 0.1448 |
| fund_roe | bull | 0.0768 | 0.3142 | 0.2443 | 0.1439 | 0.1397 |
| stroke_phase | bull | 0.0371 | 0.2610 | 0.1423 | 0.1667 | 0.0830 |
| momentum_reversal | bull | 0.0341 | 0.2878 | 0.1185 | 0.1402 | 0.0676 |
| fund_pb | bear | 0.1538 | 0.2144 | 0.7173 | 0.5342 | 0.5503 |
| momentum_reversal | bear | 0.1120 | 0.2253 | 0.4971 | 0.2329 | 0.3064 |
| vol_opening_strength | bear | 0.0899 | 0.2761 | 0.3256 | 0.3226 | 0.2153 |
| vol_opening_confirm | bear | 0.0889 | 0.2719 | 0.3269 | 0.2581 | 0.2056 |
| mom_x_lowvol_20_20 | bear | 0.0807 | 0.2958 | 0.2730 | 0.1370 | 0.1552 |
| rsi_vol_combo | bear | 0.0621 | 0.2437 | 0.2548 | 0.2055 | 0.1536 |
| trend_lowvol | bear | 0.0611 | 0.2686 | 0.2275 | 0.1644 | 0.1324 |

### 航天航空

- **Neutral**: ['fund_pb', 'momentum_reversal'] (单因子IC=0.0733, 组合IC=0.0969)
  - weights: [0.5033, 0.4967]
- **Bull**: ['fund_pb', 'low_downside', 'trend_lowvol'] (单因子IC=0.0998, 组合IC=0.1321)
  - bull_weights: [0.3772, 0.3764, 0.2464]
- **Bear**: ['fund_profit_growth', 'fund_gross_margin'] (单因子IC=0.1139, 组合IC=0.1428)
  - bear_weights: [0.5058, 0.4942]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| fund_pb | neutral | 0.0680 | 0.1882 | 0.3613 | 0.2526 | 0.2263 |
| momentum_reversal | neutral | 0.0787 | 0.2293 | 0.3431 | 0.3017 | 0.2233 |
| rsi_vol_combo | neutral | 0.0677 | 0.2242 | 0.3019 | 0.2338 | 0.1862 |
| mom_x_lowvol_20_20 | neutral | 0.0627 | 0.2359 | 0.2656 | 0.2025 | 0.1597 |
| trend_lowvol | neutral | 0.0570 | 0.2483 | 0.2295 | 0.1420 | 0.1310 |
| volatility | neutral | 0.0500 | 0.2487 | 0.2011 | 0.1879 | 0.1195 |
| fund_profit_growth | neutral | 0.0372 | 0.2059 | 0.1805 | 0.1868 | 0.1071 |
| turnover_stability | neutral | 0.0311 | 0.1970 | 0.1580 | 0.1587 | 0.0915 |
| fund_score | neutral | 0.0375 | 0.2301 | 0.1631 | 0.1106 | 0.0906 |
| fund_pb | bull | 0.1023 | 0.1922 | 0.5323 | 0.3826 | 0.3679 |
| low_downside | bull | 0.1073 | 0.2003 | 0.5356 | 0.3712 | 0.3672 |
| trend_lowvol | bull | 0.0899 | 0.2436 | 0.3690 | 0.3030 | 0.2404 |
| momentum_reversal | bull | 0.0674 | 0.2308 | 0.2923 | 0.2197 | 0.1783 |
| volatility | bull | 0.0683 | 0.2352 | 0.2904 | 0.1970 | 0.1738 |
| rsi_vol_combo | bull | 0.0462 | 0.2179 | 0.2122 | 0.2273 | 0.1302 |
| mom_x_lowvol_20_20 | bull | 0.0396 | 0.2204 | 0.1799 | 0.1515 | 0.1036 |
| fund_profit_growth | bear | 0.1136 | 0.2217 | 0.5124 | 0.3973 | 0.3580 |
| fund_gross_margin | bear | 0.1141 | 0.2280 | 0.5006 | 0.3973 | 0.3497 |
| fund_score | bear | 0.0888 | 0.2409 | 0.3685 | 0.3151 | 0.2423 |
| trend_lowvol | bear | 0.0929 | 0.2609 | 0.3558 | 0.2603 | 0.2242 |
| mom_x_lowvol_20_20 | bear | 0.0897 | 0.2834 | 0.3166 | 0.3151 | 0.2082 |
| fund_pe | bear | 0.0620 | 0.1927 | 0.3219 | 0.2877 | 0.2073 |
| fund_revenue_growth | bear | 0.0704 | 0.2017 | 0.3493 | 0.1233 | 0.1962 |
| fund_roe | bear | 0.0612 | 0.2122 | 0.2884 | 0.3425 | 0.1936 |
| momentum_reversal | bear | 0.0716 | 0.2527 | 0.2832 | 0.3151 | 0.1862 |
| bb_width_20 | bear | 0.0654 | 0.2412 | 0.2711 | 0.2329 | 0.1671 |
| rsi_vol_combo | bear | 0.0523 | 0.2354 | 0.2224 | 0.1233 | 0.1249 |

### 航母概念

- **Neutral**: ['trend_lowvol', 'fund_pb', 'momentum_reversal'] (单因子IC=0.0726, 组合IC=0.0973)
  - weights: [0.4546, 0.276, 0.2694]
- **Bull**: ['low_downside', 'fund_pb', 'rsi_vol_combo'] (单因子IC=0.0622, 组合IC=0.1016)
  - bull_weights: [0.4367, 0.3519, 0.2114]
- **Bear**: ['momentum_reversal', 'rsi_vol_combo', 'bb_width_20'] (单因子IC=0.1474, 组合IC=0.1823)
  - bear_weights: [0.3706, 0.3704, 0.259]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| trend_lowvol | neutral | 0.0954 | 0.2687 | 0.3550 | 0.2954 | 0.2300 |
| fund_pb | neutral | 0.0609 | 0.2551 | 0.2389 | 0.1691 | 0.1396 |
| momentum_reversal | neutral | 0.0613 | 0.2655 | 0.2310 | 0.1795 | 0.1363 |
| rsi_vol_combo | neutral | 0.0539 | 0.2495 | 0.2158 | 0.1983 | 0.1293 |
| mom_x_lowvol_20_20 | neutral | 0.0420 | 0.2525 | 0.1662 | 0.1357 | 0.0944 |
| fund_pe | neutral | 0.0369 | 0.2378 | 0.1550 | 0.1503 | 0.0891 |
| volatility | neutral | 0.0368 | 0.2730 | 0.1350 | 0.1451 | 0.0773 |
| fund_profit_growth | neutral | 0.0244 | 0.2349 | 0.1041 | 0.1211 | 0.0583 |
| fund_roe | neutral | 0.0233 | 0.2405 | 0.0968 | 0.1263 | 0.0545 |
| low_downside | bull | 0.0772 | 0.2421 | 0.3190 | 0.2727 | 0.2030 |
| fund_pb | bull | 0.0668 | 0.2399 | 0.2786 | 0.1742 | 0.1636 |
| rsi_vol_combo | bull | 0.0425 | 0.2394 | 0.1777 | 0.1061 | 0.0983 |
| fund_pe | bull | 0.0470 | 0.2918 | 0.1609 | 0.1061 | 0.0890 |
| momentum_reversal | bear | 0.1490 | 0.2127 | 0.7003 | 0.5342 | 0.5372 |
| rsi_vol_combo | bear | 0.1571 | 0.2204 | 0.7126 | 0.5068 | 0.5369 |
| bb_width_20 | bear | 0.1363 | 0.2586 | 0.5270 | 0.4247 | 0.3754 |
| fund_profit_growth | bear | 0.1020 | 0.2844 | 0.3587 | 0.3699 | 0.2457 |
| fund_gross_margin | bear | 0.0827 | 0.2322 | 0.3562 | 0.2603 | 0.2245 |
| fund_roe | bear | 0.0825 | 0.2429 | 0.3397 | 0.2329 | 0.2094 |
| mom_x_lowvol_20_20 | bear | 0.0782 | 0.2568 | 0.3047 | 0.1507 | 0.1753 |
| trend_lowvol | bear | 0.0599 | 0.2736 | 0.2191 | 0.1233 | 0.1231 |
| fund_pe | bear | 0.0453 | 0.2412 | 0.1877 | 0.1781 | 0.1105 |
| turnover_stability | bear | 0.0397 | 0.2019 | 0.1964 | 0.1233 | 0.1103 |

### 节能环保

- **Neutral**: ['turnover_stability', 'momentum_reversal', 'mom_x_lowvol_20_20'] (单因子IC=0.0673, 组合IC=0.088)
  - weights: [0.3375, 0.3318, 0.3307]
- **Bull**: ['turnover_stability', 'low_downside', 'volatility'] (单因子IC=0.0808, 组合IC=0.1035)
  - bull_weights: [0.3588, 0.3263, 0.3149]
- **Bear**: ['momentum_reversal'] (单因子IC=0.1184, 组合IC=0.1184)
  - bear_weights: [1.0]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| turnover_stability | neutral | 0.0433 | 0.0854 | 0.5070 | 0.3946 | 0.3536 |
| momentum_reversal | neutral | 0.0799 | 0.1581 | 0.5052 | 0.3758 | 0.3475 |
| mom_x_lowvol_20_20 | neutral | 0.0788 | 0.1564 | 0.5036 | 0.3758 | 0.3464 |
| fund_profit_growth | neutral | 0.0458 | 0.0964 | 0.4750 | 0.3069 | 0.3104 |
| trend_lowvol | neutral | 0.0713 | 0.1747 | 0.4084 | 0.3090 | 0.2673 |
| fund_pb | neutral | 0.0656 | 0.1595 | 0.4114 | 0.2547 | 0.2581 |
| rsi_vol_combo | neutral | 0.0561 | 0.1430 | 0.3921 | 0.2756 | 0.2501 |
| volatility | neutral | 0.0670 | 0.1798 | 0.3728 | 0.3111 | 0.2444 |
| fund_pe | neutral | 0.0583 | 0.1545 | 0.3775 | 0.2610 | 0.2380 |
| fund_score | neutral | 0.0398 | 0.1319 | 0.3020 | 0.1983 | 0.1810 |
| fund_revenue_growth | neutral | 0.0197 | 0.0914 | 0.2153 | 0.1566 | 0.1245 |
| low_downside | neutral | 0.0340 | 0.1676 | 0.2029 | 0.1670 | 0.1184 |
| fund_roe | neutral | 0.0293 | 0.1508 | 0.1943 | 0.1148 | 0.1083 |
| turnover_stability | bull | 0.0539 | 0.0761 | 0.7086 | 0.5303 | 0.5422 |
| low_downside | bull | 0.0888 | 0.1329 | 0.6677 | 0.4773 | 0.4932 |
| volatility | bull | 0.0999 | 0.1542 | 0.6476 | 0.4697 | 0.4759 |
| fund_pb | bull | 0.0812 | 0.1536 | 0.5287 | 0.3939 | 0.3685 |
| fund_profit_growth | bull | 0.0345 | 0.0971 | 0.3555 | 0.3106 | 0.2330 |
| fund_pe | bull | 0.0568 | 0.1581 | 0.3592 | 0.2803 | 0.2300 |
| momentum_reversal | bull | 0.0413 | 0.1388 | 0.2976 | 0.2121 | 0.1804 |
| trend_lowvol | bull | 0.0504 | 0.1753 | 0.2877 | 0.2348 | 0.1776 |
| mom_x_lowvol_20_20 | bull | 0.0376 | 0.1402 | 0.2684 | 0.1894 | 0.1596 |
| fund_score | bull | 0.0283 | 0.1318 | 0.2148 | 0.1364 | 0.1220 |
| fund_revenue_growth | bull | 0.0157 | 0.0883 | 0.1783 | 0.1818 | 0.1054 |
| rsi_vol_combo | bull | 0.0213 | 0.1222 | 0.1746 | 0.1212 | 0.0979 |
| fund_gross_margin | bull | 0.0114 | 0.0765 | 0.1490 | 0.2045 | 0.0897 |
| stroke_phase | bull | 0.0122 | 0.0887 | 0.1375 | 0.1439 | 0.0786 |
| momentum_reversal | bear | 0.1184 | 0.1947 | 0.6080 | 0.5616 | 0.4748 |
| mom_x_lowvol_20_20 | bear | 0.1126 | 0.2030 | 0.5546 | 0.5616 | 0.4331 |
| rsi_vol_combo | bear | 0.0715 | 0.1492 | 0.4790 | 0.4521 | 0.3478 |
| trend_lowvol | bear | 0.0857 | 0.1957 | 0.4380 | 0.1507 | 0.2520 |
| turnover_stability | bear | 0.0240 | 0.0709 | 0.3387 | 0.1781 | 0.1995 |
| fund_pb | bear | 0.0551 | 0.2033 | 0.2710 | 0.1507 | 0.1559 |
| fund_revenue_growth | bear | 0.0241 | 0.0922 | 0.2613 | 0.1507 | 0.1504 |
| fund_profit_growth | bear | 0.0320 | 0.1282 | 0.2498 | 0.1233 | 0.1403 |
| bb_width_20 | bear | 0.0369 | 0.2051 | 0.1799 | 0.2329 | 0.1109 |

### 英伟达概念

- **Neutral**: ['fund_pb', 'momentum_reversal', 'mom_x_lowvol_20_20'] (单因子IC=0.0666, 组合IC=0.0886)
  - weights: [0.3862, 0.3338, 0.2801]
- **Bull**: ['volatility', 'low_downside', 'fund_pb'] (单因子IC=0.0688, 组合IC=0.0878)
  - bull_weights: [0.4266, 0.3241, 0.2493]
- **Bear**: ['fund_pb', 'vol_opening_strength', 'vol_opening_confirm'] (单因子IC=0.0717, 组合IC=0.114)
  - bear_weights: [0.4858, 0.2713, 0.2428]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| fund_pb | neutral | 0.0650 | 0.1658 | 0.3919 | 0.3424 | 0.2630 |
| momentum_reversal | neutral | 0.0725 | 0.2074 | 0.3496 | 0.3006 | 0.2273 |
| mom_x_lowvol_20_20 | neutral | 0.0624 | 0.2057 | 0.3036 | 0.2568 | 0.1908 |
| rsi_vol_combo | neutral | 0.0590 | 0.1953 | 0.3021 | 0.2276 | 0.1854 |
| volatility | neutral | 0.0583 | 0.2018 | 0.2891 | 0.1942 | 0.1726 |
| fund_pe | neutral | 0.0489 | 0.1795 | 0.2724 | 0.1983 | 0.1632 |
| trend_lowvol | neutral | 0.0627 | 0.2370 | 0.2645 | 0.2234 | 0.1618 |
| fund_score | neutral | 0.0552 | 0.2089 | 0.2642 | 0.1858 | 0.1566 |
| fund_profit_growth | neutral | 0.0397 | 0.1860 | 0.2133 | 0.1388 | 0.1214 |
| fund_roe | neutral | 0.0424 | 0.2008 | 0.2110 | 0.1420 | 0.1205 |
| fund_revenue_growth | neutral | 0.0320 | 0.1597 | 0.2005 | 0.1441 | 0.1147 |
| low_downside | neutral | 0.0215 | 0.1786 | 0.1204 | 0.1399 | 0.0686 |
| volatility | bull | 0.0878 | 0.1859 | 0.4720 | 0.3561 | 0.3200 |
| low_downside | bull | 0.0679 | 0.1789 | 0.3798 | 0.2803 | 0.2432 |
| fund_pb | bull | 0.0508 | 0.1749 | 0.2904 | 0.2879 | 0.1870 |
| fund_gross_margin | bull | 0.0497 | 0.1692 | 0.2935 | 0.2424 | 0.1823 |
| turnover_stability | bull | 0.0469 | 0.1618 | 0.2898 | 0.1591 | 0.1679 |
| trend_lowvol | bull | 0.0541 | 0.2190 | 0.2468 | 0.1894 | 0.1468 |
| mom_x_lowvol_20_20 | bull | 0.0412 | 0.1755 | 0.2348 | 0.1439 | 0.1343 |
| momentum_reversal | bull | 0.0383 | 0.1815 | 0.2111 | 0.1288 | 0.1191 |
| rsi_vol_combo | bull | 0.0371 | 0.1850 | 0.2007 | 0.1439 | 0.1148 |
| fund_pb | bear | 0.1021 | 0.1778 | 0.5743 | 0.3425 | 0.3855 |
| vol_opening_strength | bear | 0.0578 | 0.1736 | 0.3327 | 0.2941 | 0.2153 |
| vol_opening_confirm | bear | 0.0552 | 0.1741 | 0.3170 | 0.2157 | 0.1927 |
| stroke_phase | bear | 0.0479 | 0.1795 | 0.2668 | 0.3699 | 0.1828 |
| momentum_reversal | bear | 0.0616 | 0.2427 | 0.2540 | 0.1507 | 0.1461 |
| trend_lowvol | bear | 0.0603 | 0.2488 | 0.2423 | 0.1233 | 0.1361 |

### 苹果概念

- **Neutral**: ['mom_x_lowvol_20_20', 'momentum_reversal', 'trend_lowvol'] (单因子IC=0.0792, 组合IC=0.0894)
  - weights: [0.3542, 0.3471, 0.2987]
- **Bull**: ['fund_pb', 'turnover_stability', 'trend_lowvol'] (单因子IC=0.048, 组合IC=0.0681)
  - bull_weights: [0.3451, 0.3403, 0.3146]
- **Bear**: ['mom_x_lowvol_20_20', 'fund_pb', 'momentum_reversal'] (单因子IC=0.0853, 组合IC=0.1173)
  - bear_weights: [0.4385, 0.3375, 0.2241]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| mom_x_lowvol_20_20 | neutral | 0.0767 | 0.1682 | 0.4562 | 0.3299 | 0.3033 |
| momentum_reversal | neutral | 0.0807 | 0.1748 | 0.4615 | 0.2881 | 0.2972 |
| trend_lowvol | neutral | 0.0802 | 0.2003 | 0.4003 | 0.2777 | 0.2557 |
| fund_pb | neutral | 0.0598 | 0.1612 | 0.3710 | 0.3027 | 0.2417 |
| rsi_vol_combo | neutral | 0.0658 | 0.1716 | 0.3833 | 0.2610 | 0.2416 |
| volatility | neutral | 0.0648 | 0.1818 | 0.3565 | 0.2839 | 0.2289 |
| fund_pe | neutral | 0.0506 | 0.1560 | 0.3242 | 0.2422 | 0.2014 |
| fund_profit_growth | neutral | 0.0482 | 0.1488 | 0.3237 | 0.2004 | 0.1943 |
| low_downside | neutral | 0.0340 | 0.1781 | 0.1909 | 0.1691 | 0.1116 |
| fund_score | neutral | 0.0349 | 0.1814 | 0.1926 | 0.1169 | 0.1076 |
| fund_revenue_growth | neutral | 0.0262 | 0.1515 | 0.1728 | 0.1649 | 0.1006 |
| fund_pb | bull | 0.0500 | 0.1596 | 0.3135 | 0.2576 | 0.1971 |
| turnover_stability | bull | 0.0372 | 0.1204 | 0.3091 | 0.2576 | 0.1944 |
| trend_lowvol | bull | 0.0567 | 0.1923 | 0.2946 | 0.2197 | 0.1797 |
| fund_gross_margin | bull | 0.0429 | 0.1579 | 0.2716 | 0.2424 | 0.1687 |
| volatility | bull | 0.0443 | 0.1668 | 0.2657 | 0.2576 | 0.1670 |
| rsi_vol_combo | bull | 0.0406 | 0.1568 | 0.2587 | 0.2652 | 0.1636 |
| low_downside | bull | 0.0413 | 0.1686 | 0.2452 | 0.1742 | 0.1440 |
| momentum_reversal | bull | 0.0380 | 0.1762 | 0.2155 | 0.2424 | 0.1339 |
| mom_x_lowvol_20_20 | bull | 0.0311 | 0.1632 | 0.1904 | 0.1818 | 0.1125 |
| fund_profit_growth | bull | 0.0235 | 0.1532 | 0.1536 | 0.1136 | 0.0855 |
| mom_x_lowvol_20_20 | bear | 0.1074 | 0.1767 | 0.6079 | 0.4521 | 0.4413 |
| fund_pb | bear | 0.0780 | 0.1541 | 0.5060 | 0.3425 | 0.3397 |
| momentum_reversal | bear | 0.0704 | 0.1926 | 0.3658 | 0.2329 | 0.2255 |
| rsi_vol_combo | bear | 0.0546 | 0.1451 | 0.3766 | 0.1781 | 0.2218 |
| fund_gross_margin | bear | 0.0413 | 0.1872 | 0.2204 | 0.1781 | 0.1298 |
| fund_profit_growth | bear | 0.0292 | 0.1447 | 0.2016 | 0.2603 | 0.1270 |
| fund_pe | bear | 0.0273 | 0.1390 | 0.1965 | 0.1507 | 0.1130 |

### 茅指数

- **Neutral**: ['fund_profit_growth', 'fund_pb', 'fund_pe'] (单因子IC=0.0514, 组合IC=0.0788)
  - weights: [0.3563, 0.3547, 0.289]
- **Bull**: ['stroke_phase', 'fund_pb', 'ema20_slope'] (单因子IC=0.036, 组合IC=0.0562)
  - bull_weights: [0.5761, 0.229, 0.1949]
- **Bear**: ['bb_width_20', 'trend_lowvol'] (单因子IC=0.1536, 组合IC=0.1792)
  - bear_weights: [0.5147, 0.4853]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| fund_profit_growth | neutral | 0.0490 | 0.1935 | 0.2532 | 0.1315 | 0.1432 |
| fund_pb | neutral | 0.0535 | 0.2221 | 0.2409 | 0.1837 | 0.1426 |
| fund_pe | neutral | 0.0516 | 0.2515 | 0.2053 | 0.1315 | 0.1162 |
| fund_score | neutral | 0.0333 | 0.1790 | 0.1862 | 0.1399 | 0.1061 |
| top_fractal_volume | neutral | 0.0198 | 0.1672 | 0.1185 | 0.1311 | 0.0670 |
| stroke_phase | bull | 0.0515 | 0.1768 | 0.2915 | 0.2348 | 0.1800 |
| fund_pb | bull | 0.0291 | 0.2279 | 0.1276 | 0.1212 | 0.0715 |
| ema20_slope | bull | 0.0275 | 0.2580 | 0.1064 | 0.1439 | 0.0609 |
| bb_width_20 | bear | 0.1390 | 0.1821 | 0.7632 | 0.5890 | 0.6064 |
| trend_lowvol | bear | 0.1682 | 0.2378 | 0.7074 | 0.6164 | 0.5718 |
| momentum_reversal | bear | 0.1384 | 0.2402 | 0.5760 | 0.4521 | 0.4182 |
| wash_sale_score | bear | 0.0868 | 0.2137 | 0.4064 | 0.3000 | 0.2641 |
| rsi_vol_combo | bear | 0.0976 | 0.2659 | 0.3673 | 0.3151 | 0.2415 |
| mom_x_lowvol_20_20 | bear | 0.0767 | 0.2266 | 0.3386 | 0.2329 | 0.2087 |
| fund_pb | bear | 0.0458 | 0.1928 | 0.2377 | 0.1781 | 0.1400 |

### 荣耀概念

- **Neutral**: ['fund_pb', 'low_downside', 'volatility'] (单因子IC=0.0663, 组合IC=0.0862)
  - weights: [0.4048, 0.3041, 0.2911]
- **Bull**: ['fund_pb', 'trend_lowvol', 'turnover_stability'] (单因子IC=0.1009, 组合IC=0.1546)
  - bull_weights: [0.3619, 0.326, 0.3121]
- **Bear**: ['fund_revenue_growth', 'fund_roe'] (单因子IC=0.1367, 组合IC=0.1845)
  - bear_weights: [0.5445, 0.4555]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| fund_pb | neutral | 0.0752 | 0.1996 | 0.3768 | 0.2693 | 0.2391 |
| low_downside | neutral | 0.0590 | 0.2082 | 0.2835 | 0.2672 | 0.1796 |
| volatility | neutral | 0.0648 | 0.2304 | 0.2811 | 0.2234 | 0.1720 |
| momentum_reversal | neutral | 0.0610 | 0.2296 | 0.2657 | 0.2067 | 0.1603 |
| mom_x_lowvol_20_20 | neutral | 0.0562 | 0.2256 | 0.2491 | 0.1806 | 0.1471 |
| trend_lowvol | neutral | 0.0581 | 0.2372 | 0.2448 | 0.1399 | 0.1395 |
| rsi_vol_combo | neutral | 0.0481 | 0.2136 | 0.2251 | 0.1722 | 0.1319 |
| fund_pb | bull | 0.1110 | 0.1905 | 0.5828 | 0.4470 | 0.4216 |
| trend_lowvol | bull | 0.1108 | 0.2089 | 0.5305 | 0.4318 | 0.3798 |
| turnover_stability | bull | 0.0809 | 0.1559 | 0.5189 | 0.4015 | 0.3636 |
| mom_x_lowvol_20_20 | bull | 0.0962 | 0.1931 | 0.4983 | 0.4015 | 0.3492 |
| volatility | bull | 0.1105 | 0.2206 | 0.5008 | 0.3712 | 0.3434 |
| low_downside | bull | 0.0884 | 0.2128 | 0.4154 | 0.2803 | 0.2659 |
| momentum_reversal | bull | 0.0816 | 0.2053 | 0.3974 | 0.3182 | 0.2619 |
| rsi_vol_combo | bull | 0.0655 | 0.1808 | 0.3621 | 0.2424 | 0.2250 |
| stroke_phase | bull | 0.0207 | 0.1507 | 0.1371 | 0.1061 | 0.0758 |
| fund_revenue_growth | bear | 0.1405 | 0.1632 | 0.8611 | 0.5616 | 0.6723 |
| fund_roe | bear | 0.1328 | 0.1941 | 0.6844 | 0.6438 | 0.5625 |
| fund_score | bear | 0.0945 | 0.2392 | 0.3950 | 0.2877 | 0.2543 |
| wash_sale_score | bear | 0.0629 | 0.1700 | 0.3702 | 0.2917 | 0.2391 |
| fund_pe | bear | 0.0768 | 0.2112 | 0.3636 | 0.2603 | 0.2291 |
| turnover_stability | bear | 0.0441 | 0.1248 | 0.3535 | 0.2329 | 0.2179 |
| relative_strength | bear | 0.0464 | 0.2224 | 0.2088 | 0.2329 | 0.1287 |

### 虚拟数字人

- **Neutral**: ['trend_lowvol', 'volatility'] (单因子IC=0.1291, 组合IC=0.1502)
  - weights: [0.5436, 0.4564]
- **Bull**: ['trend_lowvol', 'fund_pb', 'volatility'] (单因子IC=0.1134, 组合IC=0.1353)
  - bull_weights: [0.3483, 0.3356, 0.3161]
- **Bear**: ['trend_lowvol', 'mom_x_lowvol_20_20', 'fund_profit_growth'] (单因子IC=0.1041, 组合IC=0.1407)
  - bear_weights: [0.4397, 0.2856, 0.2747]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| trend_lowvol | neutral | 0.1335 | 0.2086 | 0.6397 | 0.4823 | 0.4741 |
| volatility | neutral | 0.1247 | 0.2182 | 0.5718 | 0.3925 | 0.3981 |
| fund_pb | neutral | 0.0951 | 0.1835 | 0.5183 | 0.4008 | 0.3630 |
| momentum_reversal | neutral | 0.1031 | 0.2066 | 0.4992 | 0.3674 | 0.3413 |
| mom_x_lowvol_20_20 | neutral | 0.0936 | 0.2083 | 0.4496 | 0.3299 | 0.2989 |
| rsi_vol_combo | neutral | 0.0799 | 0.1809 | 0.4417 | 0.3006 | 0.2872 |
| low_downside | neutral | 0.0797 | 0.2012 | 0.3959 | 0.3278 | 0.2629 |
| fund_score | neutral | 0.0488 | 0.1794 | 0.2722 | 0.1795 | 0.1605 |
| turnover_stability | neutral | 0.0393 | 0.1447 | 0.2713 | 0.1545 | 0.1566 |
| fund_profit_growth | neutral | 0.0368 | 0.1512 | 0.2433 | 0.1837 | 0.1440 |
| fund_roe | neutral | 0.0424 | 0.1797 | 0.2359 | 0.1775 | 0.1389 |
| fund_pe | neutral | 0.0439 | 0.1994 | 0.2203 | 0.2088 | 0.1331 |
| trend_lowvol | bull | 0.1202 | 0.1871 | 0.6424 | 0.5076 | 0.4843 |
| fund_pb | bull | 0.1049 | 0.1669 | 0.6285 | 0.4848 | 0.4666 |
| volatility | bull | 0.1150 | 0.1893 | 0.6073 | 0.4470 | 0.4394 |
| mom_x_lowvol_20_20 | bull | 0.0884 | 0.1648 | 0.5366 | 0.3939 | 0.3740 |
| low_downside | bull | 0.0844 | 0.1768 | 0.4770 | 0.3485 | 0.3216 |
| fund_profit_growth | bull | 0.0545 | 0.1252 | 0.4351 | 0.3788 | 0.3000 |
| momentum_reversal | bull | 0.0722 | 0.1629 | 0.4430 | 0.3485 | 0.2987 |
| fund_pe | bull | 0.0716 | 0.1846 | 0.3879 | 0.2803 | 0.2483 |
| fund_score | bull | 0.0391 | 0.1410 | 0.2775 | 0.2197 | 0.1692 |
| rsi_vol_combo | bull | 0.0513 | 0.1777 | 0.2889 | 0.1439 | 0.1652 |
| turnover_stability | bull | 0.0356 | 0.1271 | 0.2799 | 0.1364 | 0.1590 |
| fund_revenue_growth | bull | 0.0309 | 0.1265 | 0.2444 | 0.2045 | 0.1472 |
| trend_lowvol | bear | 0.1212 | 0.1719 | 0.7048 | 0.4795 | 0.5214 |
| mom_x_lowvol_20_20 | bear | 0.1012 | 0.2128 | 0.4755 | 0.4247 | 0.3387 |
| fund_profit_growth | bear | 0.0901 | 0.1856 | 0.4853 | 0.3425 | 0.3258 |
| momentum_reversal | bear | 0.0995 | 0.2103 | 0.4733 | 0.3425 | 0.3177 |
| turnover_stability | bear | 0.0608 | 0.1504 | 0.4042 | 0.3699 | 0.2768 |
| fund_revenue_growth | bear | 0.0501 | 0.1682 | 0.2976 | 0.1781 | 0.1753 |
| rsi_vol_combo | bear | 0.0526 | 0.2085 | 0.2523 | 0.2329 | 0.1555 |

### 虚拟现实

- **Neutral**: ['momentum_reversal', 'mom_x_lowvol_20_20', 'trend_lowvol'] (单因子IC=0.1019, 组合IC=0.1128)
  - weights: [0.3476, 0.3382, 0.3142]
- **Bull**: ['turnover_stability', 'volatility', 'fund_pb'] (单因子IC=0.0841, 组合IC=0.1135)
  - bull_weights: [0.3464, 0.3312, 0.3224]
- **Bear**: ['mom_x_lowvol_20_20', 'momentum_reversal', 'fund_profit_growth'] (单因子IC=0.0945, 组合IC=0.1139)
  - bear_weights: [0.4215, 0.3246, 0.254]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| momentum_reversal | neutral | 0.1016 | 0.1664 | 0.6103 | 0.4426 | 0.4402 |
| mom_x_lowvol_20_20 | neutral | 0.0982 | 0.1619 | 0.6062 | 0.4134 | 0.4284 |
| trend_lowvol | neutral | 0.1059 | 0.1895 | 0.5591 | 0.4238 | 0.3980 |
| fund_pb | neutral | 0.0726 | 0.1322 | 0.5494 | 0.3549 | 0.3722 |
| rsi_vol_combo | neutral | 0.0822 | 0.1582 | 0.5199 | 0.3904 | 0.3614 |
| volatility | neutral | 0.0847 | 0.1836 | 0.4611 | 0.3486 | 0.3109 |
| fund_pe | neutral | 0.0534 | 0.1361 | 0.3924 | 0.3257 | 0.2601 |
| turnover_stability | neutral | 0.0360 | 0.1070 | 0.3370 | 0.2902 | 0.2174 |
| fund_profit_growth | neutral | 0.0429 | 0.1293 | 0.3319 | 0.2683 | 0.2105 |
| low_downside | neutral | 0.0412 | 0.1598 | 0.2576 | 0.2338 | 0.1589 |
| fund_score | neutral | 0.0427 | 0.1635 | 0.2612 | 0.1691 | 0.1527 |
| fund_revenue_growth | neutral | 0.0293 | 0.1272 | 0.2305 | 0.1566 | 0.1333 |
| turnover_stability | bull | 0.0632 | 0.0980 | 0.6452 | 0.5000 | 0.4839 |
| volatility | bull | 0.1064 | 0.1708 | 0.6232 | 0.4848 | 0.4627 |
| fund_pb | bull | 0.0827 | 0.1336 | 0.6193 | 0.4545 | 0.4504 |
| low_downside | bull | 0.0791 | 0.1434 | 0.5512 | 0.4015 | 0.3862 |
| fund_pe | bull | 0.0615 | 0.1461 | 0.4211 | 0.3788 | 0.2903 |
| trend_lowvol | bull | 0.0813 | 0.1886 | 0.4311 | 0.3258 | 0.2858 |
| mom_x_lowvol_20_20 | bull | 0.0637 | 0.1554 | 0.4100 | 0.3030 | 0.2671 |
| momentum_reversal | bull | 0.0519 | 0.1535 | 0.3379 | 0.2348 | 0.2086 |
| fund_profit_growth | bull | 0.0385 | 0.1303 | 0.2956 | 0.3182 | 0.1948 |
| fund_gross_margin | bull | 0.0305 | 0.1143 | 0.2665 | 0.1894 | 0.1585 |
| fund_score | bull | 0.0396 | 0.1629 | 0.2433 | 0.1288 | 0.1373 |
| fund_revenue_growth | bull | 0.0276 | 0.1303 | 0.2122 | 0.1818 | 0.1254 |
| fund_roe | bull | 0.0281 | 0.1580 | 0.1779 | 0.1364 | 0.1011 |
| mom_x_lowvol_20_20 | bear | 0.1131 | 0.1703 | 0.6640 | 0.5068 | 0.5002 |
| momentum_reversal | bear | 0.1054 | 0.1874 | 0.5624 | 0.3699 | 0.3852 |
| fund_profit_growth | bear | 0.0650 | 0.1330 | 0.4891 | 0.2329 | 0.3015 |
| trend_lowvol | bear | 0.0772 | 0.1721 | 0.4487 | 0.3425 | 0.3012 |
| rsi_vol_combo | bear | 0.0515 | 0.1680 | 0.3067 | 0.2877 | 0.1974 |
| wash_sale_score | bear | 0.0223 | 0.0987 | 0.2255 | 0.2459 | 0.1405 |
| fund_revenue_growth | bear | 0.0321 | 0.1517 | 0.2114 | 0.1781 | 0.1245 |
| fund_gross_margin | bear | 0.0237 | 0.1320 | 0.1792 | 0.3425 | 0.1203 |
| turnover_stability | bear | 0.0166 | 0.0859 | 0.1937 | 0.1233 | 0.1088 |

### 虚拟电厂

- **Neutral**: ['fund_pb', 'momentum_reversal'] (单因子IC=0.0772, 组合IC=0.112)
  - weights: [0.5471, 0.4529]
- **Bull**: ['low_downside', 'volatility', 'fund_pb'] (单因子IC=0.0842, 组合IC=0.1011)
  - bull_weights: [0.3967, 0.3457, 0.2576]
- **Bear**: ['rsi_vol_combo', 'momentum_reversal', 'mom_x_lowvol_20_20'] (单因子IC=0.1436, 组合IC=0.1579)
  - bear_weights: [0.3649, 0.3523, 0.2828]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| fund_pb | neutral | 0.0790 | 0.1877 | 0.4209 | 0.3236 | 0.2786 |
| momentum_reversal | neutral | 0.0753 | 0.2085 | 0.3613 | 0.2766 | 0.2306 |
| fund_profit_growth | neutral | 0.0547 | 0.1610 | 0.3399 | 0.2651 | 0.2150 |
| mom_x_lowvol_20_20 | neutral | 0.0670 | 0.2058 | 0.3256 | 0.2735 | 0.2073 |
| volatility | neutral | 0.0784 | 0.2449 | 0.3200 | 0.2672 | 0.2028 |
| trend_lowvol | neutral | 0.0739 | 0.2374 | 0.3112 | 0.2651 | 0.1969 |
| rsi_vol_combo | neutral | 0.0630 | 0.1983 | 0.3175 | 0.2317 | 0.1956 |
| fund_score | neutral | 0.0464 | 0.1748 | 0.2653 | 0.1921 | 0.1581 |
| fund_pe | neutral | 0.0543 | 0.2128 | 0.2553 | 0.1942 | 0.1524 |
| low_downside | neutral | 0.0529 | 0.2347 | 0.2254 | 0.1441 | 0.1289 |
| turnover_stability | neutral | 0.0205 | 0.1411 | 0.1455 | 0.1023 | 0.0802 |
| fund_revenue_growth | neutral | 0.0218 | 0.1537 | 0.1419 | 0.1138 | 0.0790 |
| low_downside | bull | 0.0976 | 0.1986 | 0.4912 | 0.3712 | 0.3368 |
| volatility | bull | 0.0931 | 0.2030 | 0.4585 | 0.2803 | 0.2935 |
| fund_pb | bull | 0.0620 | 0.1761 | 0.3520 | 0.2424 | 0.2187 |
| fund_profit_growth | bull | 0.0478 | 0.1499 | 0.3190 | 0.2576 | 0.2006 |
| fund_pe | bull | 0.0633 | 0.2035 | 0.3111 | 0.1818 | 0.1838 |
| fund_revenue_growth | bull | 0.0378 | 0.1462 | 0.2589 | 0.2197 | 0.1579 |
| stroke_phase | bull | 0.0388 | 0.1518 | 0.2557 | 0.1288 | 0.1443 |
| trend_lowvol | bull | 0.0533 | 0.2099 | 0.2539 | 0.1288 | 0.1433 |
| fund_roe | bull | 0.0456 | 0.1944 | 0.2345 | 0.1894 | 0.1395 |
| fund_score | bull | 0.0406 | 0.1718 | 0.2363 | 0.1742 | 0.1387 |
| turnover_stability | bull | 0.0340 | 0.1529 | 0.2224 | 0.2273 | 0.1365 |
| fund_gross_margin | bull | 0.0322 | 0.1725 | 0.1865 | 0.2121 | 0.1130 |
| rsi_vol_combo | bear | 0.1298 | 0.1850 | 0.7017 | 0.5616 | 0.5479 |
| momentum_reversal | bear | 0.1551 | 0.2289 | 0.6774 | 0.5616 | 0.5289 |
| mom_x_lowvol_20_20 | bear | 0.1460 | 0.2496 | 0.5850 | 0.4521 | 0.4247 |
| trend_lowvol | bear | 0.1240 | 0.2564 | 0.4836 | 0.3973 | 0.3378 |
| top_fractal_volume | bear | 0.0489 | 0.1518 | 0.3218 | 0.2432 | 0.2000 |
| fund_profit_growth | bear | 0.0510 | 0.2129 | 0.2397 | 0.1507 | 0.1379 |

### 蚂蚁概念

- **Neutral**: ['trend_lowvol'] (单因子IC=0.0879, 组合IC=0.0879)
  - weights: [1.0]
- **Bull**: ['volatility', 'momentum_reversal', 'fund_pb'] (单因子IC=0.1118, 组合IC=0.156)
  - bull_weights: [0.3664, 0.3395, 0.2941]
- **Bear**: ['trend_lowvol', 'mom_x_lowvol_20_20'] (单因子IC=0.2132, 组合IC=0.2426)
  - bear_weights: [0.5678, 0.4322]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| trend_lowvol | neutral | 0.0879 | 0.2647 | 0.3321 | 0.2777 | 0.2122 |
| fund_profit_growth | neutral | 0.0435 | 0.1661 | 0.2620 | 0.2025 | 0.1575 |
| fund_score | neutral | 0.0513 | 0.2012 | 0.2549 | 0.2150 | 0.1548 |
| fund_pb | neutral | 0.0651 | 0.2977 | 0.2188 | 0.1294 | 0.1236 |
| momentum_reversal | neutral | 0.0492 | 0.2520 | 0.1952 | 0.1273 | 0.1100 |
| low_downside | neutral | 0.0486 | 0.2886 | 0.1683 | 0.1169 | 0.0940 |
| fund_roe | neutral | 0.0419 | 0.2460 | 0.1704 | 0.1002 | 0.0937 |
| volatility | neutral | 0.0469 | 0.2830 | 0.1658 | 0.1169 | 0.0926 |
| mom_x_lowvol_20_20 | neutral | 0.0403 | 0.2486 | 0.1621 | 0.1002 | 0.0892 |
| volatility | bull | 0.1202 | 0.2313 | 0.5195 | 0.3409 | 0.3483 |
| momentum_reversal | bull | 0.0958 | 0.2013 | 0.4760 | 0.3561 | 0.3227 |
| fund_pb | bull | 0.1194 | 0.2702 | 0.4419 | 0.2652 | 0.2795 |
| low_downside | bull | 0.0999 | 0.2419 | 0.4128 | 0.2652 | 0.2611 |
| mom_x_lowvol_20_20 | bull | 0.0770 | 0.2009 | 0.3832 | 0.2803 | 0.2453 |
| rsi_vol_combo | bull | 0.0663 | 0.1824 | 0.3633 | 0.3258 | 0.2408 |
| trend_lowvol | bull | 0.0823 | 0.2498 | 0.3294 | 0.2273 | 0.2021 |
| turnover_stability | bull | 0.0333 | 0.1787 | 0.1862 | 0.1742 | 0.1093 |
| fund_profit_growth | bull | 0.0278 | 0.1571 | 0.1771 | 0.2121 | 0.1074 |
| fund_pe | bull | 0.0638 | 0.3465 | 0.1842 | 0.1439 | 0.1054 |
| trend_lowvol | bear | 0.2214 | 0.2652 | 0.8349 | 0.6986 | 0.7091 |
| mom_x_lowvol_20_20 | bear | 0.2050 | 0.2965 | 0.6914 | 0.5616 | 0.5398 |
| momentum_reversal | bear | 0.1981 | 0.3216 | 0.6159 | 0.5068 | 0.4640 |
| bb_width_20 | bear | 0.1802 | 0.3144 | 0.5731 | 0.3425 | 0.3847 |
| rsi_vol_combo | bear | 0.1095 | 0.2708 | 0.4044 | 0.3699 | 0.2770 |
| fund_gross_margin | bear | 0.0385 | 0.1469 | 0.2619 | 0.2603 | 0.1651 |

### 行业龙头

- **Neutral**: ['trend_lowvol', 'fund_pb'] (单因子IC=0.0521, 组合IC=0.0637)
  - weights: [0.5145, 0.4855]
- **Bull**: ['exhaustion_risk', 'relative_strength', 'ema20_slope'] (单因子IC=0.0581, 组合IC=0.0624)
  - bull_weights: [0.3414, 0.3312, 0.3273]
- **Bear**: ['bb_width_20', 'momentum_reversal'] (单因子IC=0.1585, 组合IC=0.179)
  - bear_weights: [0.5178, 0.4822]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| trend_lowvol | neutral | 0.0522 | 0.2306 | 0.2265 | 0.1900 | 0.1347 |
| fund_pb | neutral | 0.0519 | 0.2330 | 0.2227 | 0.1420 | 0.1272 |
| fund_pe | neutral | 0.0433 | 0.2309 | 0.1873 | 0.1211 | 0.1050 |
| top_fractal_volume | neutral | 0.0216 | 0.1165 | 0.1854 | 0.1182 | 0.1037 |
| fund_profit_growth | neutral | 0.0272 | 0.1551 | 0.1757 | 0.1086 | 0.0974 |
| exhaustion_risk | bull | 0.0406 | 0.1243 | 0.3267 | 0.3144 | 0.2147 |
| relative_strength | bull | 0.0630 | 0.2050 | 0.3072 | 0.3561 | 0.2083 |
| ema20_slope | bull | 0.0708 | 0.2255 | 0.3141 | 0.3106 | 0.2058 |
| ma_alignment | bull | 0.0616 | 0.2266 | 0.2720 | 0.3106 | 0.1783 |
| top_fractal_volume | bull | 0.0219 | 0.1307 | 0.1676 | 0.2095 | 0.1014 |
| low_downside | bull | 0.0358 | 0.2076 | 0.1726 | 0.1742 | 0.1013 |
| bb_width_20 | bear | 0.1547 | 0.2183 | 0.7087 | 0.4247 | 0.5048 |
| momentum_reversal | bear | 0.1623 | 0.2507 | 0.6475 | 0.4521 | 0.4701 |
| rsi_vol_combo | bear | 0.1099 | 0.2171 | 0.5064 | 0.3699 | 0.3469 |
| wash_sale_score | bear | 0.0591 | 0.1261 | 0.4687 | 0.3538 | 0.3173 |
| mom_x_lowvol_20_20 | bear | 0.0930 | 0.2186 | 0.4254 | 0.4247 | 0.3030 |
| vol_confirm | bear | 0.0609 | 0.1581 | 0.3851 | 0.2877 | 0.2479 |
| trend_lowvol | bear | 0.1115 | 0.2870 | 0.3883 | 0.1507 | 0.2234 |
| fund_profit_growth | bear | 0.0446 | 0.1303 | 0.3421 | 0.2329 | 0.2109 |
| fund_pb | bear | 0.0580 | 0.1890 | 0.3071 | 0.2877 | 0.1977 |
| limit_pullback_score | bear | 0.0264 | 0.1286 | 0.2052 | 0.1507 | 0.1181 |

### 被动元件概念

- **Neutral**: ['mom_x_lowvol_20_20', 'fund_pb', 'momentum_reversal'] (单因子IC=0.0605, 组合IC=0.0751)
  - weights: [0.3674, 0.3215, 0.3111]
- **Bull**: ['low_downside', 'vol_confirm'] (单因子IC=0.1207, 组合IC=0.1742)
  - bull_weights: [0.6247, 0.3753]
- **Bear**: ['rsi_vol_combo', 'bb_width_20'] (单因子IC=0.1618, 组合IC=0.2014)
  - bear_weights: [0.54, 0.46]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| mom_x_lowvol_20_20 | neutral | 0.0655 | 0.2618 | 0.2502 | 0.1691 | 0.1463 |
| fund_pb | neutral | 0.0539 | 0.2483 | 0.2172 | 0.1785 | 0.1280 |
| momentum_reversal | neutral | 0.0621 | 0.2892 | 0.2146 | 0.1545 | 0.1239 |
| rsi_vol_combo | neutral | 0.0529 | 0.2706 | 0.1955 | 0.1524 | 0.1127 |
| trend_lowvol | neutral | 0.0521 | 0.3001 | 0.1736 | 0.1253 | 0.0976 |
| wash_sale_score | neutral | 0.0275 | 0.2200 | 0.1251 | 0.1069 | 0.0693 |
| low_downside | bull | 0.1336 | 0.2364 | 0.5651 | 0.3902 | 0.3928 |
| vol_confirm | bull | 0.1079 | 0.2910 | 0.3708 | 0.2727 | 0.2360 |
| fund_pe | bull | 0.0762 | 0.2737 | 0.2785 | 0.1553 | 0.1609 |
| fund_pb | bull | 0.0688 | 0.2573 | 0.2673 | 0.1667 | 0.1559 |
| volatility | bull | 0.0599 | 0.2461 | 0.2434 | 0.1970 | 0.1457 |
| fund_revenue_growth | bull | 0.0325 | 0.2463 | 0.1321 | 0.2731 | 0.0841 |
| fund_roe | bull | 0.0352 | 0.2537 | 0.1387 | 0.1288 | 0.0783 |
| exhaustion_risk | bull | 0.0308 | 0.2281 | 0.1351 | 0.1485 | 0.0776 |
| rsi_vol_combo | bear | 0.1557 | 0.2382 | 0.6535 | 0.5616 | 0.5103 |
| bb_width_20 | bear | 0.1680 | 0.2700 | 0.6222 | 0.3973 | 0.4347 |
| momentum_reversal | bear | 0.1576 | 0.2625 | 0.6006 | 0.3699 | 0.4114 |
| trend_lowvol | bear | 0.1294 | 0.2569 | 0.5035 | 0.3425 | 0.3380 |
| mom_x_lowvol_20_20 | bear | 0.1027 | 0.2722 | 0.3771 | 0.2603 | 0.2376 |
| fund_profit_growth | bear | 0.0802 | 0.2596 | 0.3088 | 0.1507 | 0.1776 |
| fund_revenue_growth | bear | 0.0640 | 0.2558 | 0.2503 | 0.2877 | 0.1612 |
| fund_pb | bear | 0.0465 | 0.2297 | 0.2024 | 0.1781 | 0.1192 |
| fund_pe | bear | 0.0402 | 0.2275 | 0.1765 | 0.2329 | 0.1088 |

### 装配建筑

- **Neutral**: ['mom_x_lowvol_20_20'] (单因子IC=0.0924, 组合IC=0.0924)
  - weights: [1.0]
- **Bull**: ['trend_lowvol'] (单因子IC=0.1216, 组合IC=0.1216)
  - bull_weights: [1.0]
- **Bear**: ['trend_lowvol', 'mom_x_lowvol_20_20'] (单因子IC=0.1539, 组合IC=0.1815)
  - bear_weights: [0.5531, 0.4469]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| mom_x_lowvol_20_20 | neutral | 0.0924 | 0.2153 | 0.4291 | 0.3006 | 0.2791 |
| fund_profit_growth | neutral | 0.0516 | 0.1408 | 0.3662 | 0.2505 | 0.2290 |
| momentum_reversal | neutral | 0.0774 | 0.2240 | 0.3455 | 0.2443 | 0.2149 |
| trend_lowvol | neutral | 0.0843 | 0.2456 | 0.3431 | 0.2422 | 0.2131 |
| turnover_stability | neutral | 0.0412 | 0.1447 | 0.2846 | 0.2359 | 0.1759 |
| rsi_vol_combo | neutral | 0.0530 | 0.2098 | 0.2528 | 0.2213 | 0.1544 |
| fund_pb | neutral | 0.0433 | 0.2127 | 0.2034 | 0.1023 | 0.1121 |
| low_downside | neutral | 0.0376 | 0.2179 | 0.1725 | 0.2505 | 0.1079 |
| volatility | neutral | 0.0405 | 0.2269 | 0.1784 | 0.1524 | 0.1028 |
| fund_score | neutral | 0.0287 | 0.1777 | 0.1616 | 0.1754 | 0.0949 |
| trend_lowvol | bull | 0.1216 | 0.1823 | 0.6671 | 0.4848 | 0.4953 |
| momentum_reversal | bull | 0.0890 | 0.1748 | 0.5089 | 0.3409 | 0.3412 |
| volatility | bull | 0.0734 | 0.1818 | 0.4038 | 0.3030 | 0.2631 |
| low_downside | bull | 0.0692 | 0.1657 | 0.4174 | 0.2576 | 0.2625 |
| turnover_stability | bull | 0.0437 | 0.1173 | 0.3720 | 0.3106 | 0.2438 |
| mom_x_lowvol_20_20 | bull | 0.0613 | 0.1696 | 0.3617 | 0.2576 | 0.2274 |
| fund_pb | bull | 0.0581 | 0.1932 | 0.3009 | 0.1439 | 0.1721 |
| stroke_phase | bull | 0.0379 | 0.1512 | 0.2507 | 0.2727 | 0.1595 |
| rsi_vol_combo | bull | 0.0433 | 0.1676 | 0.2585 | 0.2121 | 0.1566 |
| fund_profit_growth | bull | 0.0324 | 0.1403 | 0.2311 | 0.2045 | 0.1392 |
| trend_lowvol | bear | 0.1530 | 0.1999 | 0.7652 | 0.6164 | 0.6184 |
| mom_x_lowvol_20_20 | bear | 0.1548 | 0.2547 | 0.6080 | 0.6438 | 0.4998 |
| rsi_vol_combo | bear | 0.1238 | 0.1938 | 0.6389 | 0.5616 | 0.4988 |
| momentum_reversal | bear | 0.1513 | 0.2493 | 0.6071 | 0.4795 | 0.4491 |

### 裸眼3D

- **Neutral**: ['trend_lowvol', 'momentum_reversal'] (单因子IC=0.1119, 组合IC=0.1268)
  - weights: [0.523, 0.477]
- **Bull**: ['low_downside', 'trend_lowvol'] (单因子IC=0.0765, 组合IC=0.0804)
  - bull_weights: [0.5339, 0.4661]
- **Bear**: ['fund_profit_growth', 'mom_x_lowvol_20_20', 'momentum_reversal'] (单因子IC=0.1617, 组合IC=0.1984)
  - bear_weights: [0.3633, 0.3395, 0.2972]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| trend_lowvol | neutral | 0.1189 | 0.2813 | 0.4225 | 0.2923 | 0.2730 |
| momentum_reversal | neutral | 0.1049 | 0.2718 | 0.3860 | 0.2902 | 0.2490 |
| mom_x_lowvol_20_20 | neutral | 0.1011 | 0.2766 | 0.3654 | 0.2630 | 0.2308 |
| rsi_vol_combo | neutral | 0.0915 | 0.2541 | 0.3600 | 0.2630 | 0.2273 |
| volatility | neutral | 0.0945 | 0.2755 | 0.3428 | 0.2443 | 0.2133 |
| low_downside | neutral | 0.0651 | 0.2629 | 0.2475 | 0.1795 | 0.1460 |
| fund_pb | neutral | 0.0593 | 0.2600 | 0.2279 | 0.1587 | 0.1320 |
| fund_score | neutral | 0.0414 | 0.2547 | 0.1625 | 0.1388 | 0.0925 |
| fund_roe | neutral | 0.0385 | 0.2610 | 0.1474 | 0.1044 | 0.0814 |
| low_downside | bull | 0.0701 | 0.2525 | 0.2778 | 0.2311 | 0.1710 |
| trend_lowvol | bull | 0.0828 | 0.3258 | 0.2542 | 0.1742 | 0.1493 |
| volatility | bull | 0.0568 | 0.2709 | 0.2097 | 0.1818 | 0.1239 |
| fund_revenue_growth | bull | 0.0355 | 0.2597 | 0.1367 | 0.1667 | 0.0797 |
| fund_pb | bull | 0.0278 | 0.2474 | 0.1125 | 0.1136 | 0.0626 |
| fund_profit_growth | bear | 0.1288 | 0.1675 | 0.7691 | 0.5068 | 0.5795 |
| mom_x_lowvol_20_20 | bear | 0.1793 | 0.2449 | 0.7320 | 0.4795 | 0.5415 |
| momentum_reversal | bear | 0.1772 | 0.2714 | 0.6528 | 0.4521 | 0.4740 |
| fund_gross_margin | bear | 0.1013 | 0.2772 | 0.3655 | 0.3699 | 0.2503 |
| fund_score | bear | 0.0893 | 0.2398 | 0.3724 | 0.3151 | 0.2449 |
| rsi_vol_combo | bear | 0.0889 | 0.2638 | 0.3369 | 0.2055 | 0.2030 |
| fund_pe | bear | 0.0723 | 0.2581 | 0.2802 | 0.2778 | 0.1790 |
| fund_roe | bear | 0.0633 | 0.2837 | 0.2233 | 0.1781 | 0.1315 |

### 证金持股

- **Neutral**: ['fund_pe', 'fund_pb', 'trend_lowvol'] (单因子IC=0.0725, 组合IC=0.097)
  - weights: [0.3946, 0.32, 0.2854]
- **Bull**: ['turnover_stability', 'low_downside', 'fund_pb'] (单因子IC=0.0514, 组合IC=0.0757)
  - bull_weights: [0.3777, 0.318, 0.3043]
- **Bear**: ['mom_x_lowvol_20_20', 'momentum_reversal', 'bb_width_20'] (单因子IC=0.0814, 组合IC=0.1055)
  - bear_weights: [0.3793, 0.3106, 0.3101]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| fund_pe | neutral | 0.0889 | 0.2242 | 0.3968 | 0.2192 | 0.2419 |
| fund_pb | neutral | 0.0674 | 0.2124 | 0.3174 | 0.2359 | 0.1962 |
| trend_lowvol | neutral | 0.0612 | 0.2161 | 0.2832 | 0.2359 | 0.1750 |
| fund_profit_growth | neutral | 0.0340 | 0.1364 | 0.2494 | 0.2088 | 0.1508 |
| low_downside | neutral | 0.0446 | 0.1865 | 0.2393 | 0.1608 | 0.1389 |
| turnover_stability | neutral | 0.0274 | 0.1246 | 0.2203 | 0.2067 | 0.1329 |
| volatility | neutral | 0.0207 | 0.1060 | 0.1950 | 0.1649 | 0.1136 |
| fund_score | neutral | 0.0295 | 0.1517 | 0.1945 | 0.1023 | 0.1072 |
| fund_roe | neutral | 0.0294 | 0.1878 | 0.1564 | 0.1336 | 0.0887 |
| turnover_stability | bull | 0.0443 | 0.1256 | 0.3527 | 0.3333 | 0.2351 |
| low_downside | bull | 0.0535 | 0.1659 | 0.3225 | 0.2273 | 0.1979 |
| fund_pb | bull | 0.0565 | 0.1795 | 0.3145 | 0.2045 | 0.1894 |
| stroke_phase | bull | 0.0282 | 0.1237 | 0.2278 | 0.2121 | 0.1381 |
| volatility | bull | 0.0256 | 0.1269 | 0.2020 | 0.2197 | 0.1232 |
| fund_pe | bull | 0.0404 | 0.2312 | 0.1747 | 0.1212 | 0.0979 |
| mom_x_lowvol_20_20 | bear | 0.0665 | 0.1641 | 0.4050 | 0.3699 | 0.2774 |
| momentum_reversal | bear | 0.0917 | 0.2599 | 0.3528 | 0.2877 | 0.2271 |
| bb_width_20 | bear | 0.0861 | 0.2495 | 0.3449 | 0.3151 | 0.2268 |
| trend_lowvol | bear | 0.0774 | 0.2865 | 0.2700 | 0.2877 | 0.1738 |
| top_fractal_volume | bear | 0.0213 | 0.0913 | 0.2338 | 0.2000 | 0.1403 |
| fund_profit_growth | bear | 0.0336 | 0.1352 | 0.2487 | 0.1233 | 0.1397 |

### 调味品概念

- **Neutral**: ['fund_pb', 'fund_pe', 'momentum_reversal'] (单因子IC=0.0815, 组合IC=0.1241)
  - weights: [0.4324, 0.3336, 0.234]
- **Bull**: ['rsi_vol_combo', 'momentum_reversal', 'fund_pb'] (单因子IC=0.0646, 组合IC=0.0932)
  - bull_weights: [0.3912, 0.3264, 0.2823]
- **Bear**: ['fund_profit_growth', 'fund_score', 'fund_pb'] (单因子IC=0.1408, 组合IC=0.1796)
  - bear_weights: [0.467, 0.2862, 0.2468]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| fund_pb | neutral | 0.1089 | 0.3017 | 0.3610 | 0.2839 | 0.2317 |
| fund_pe | neutral | 0.0815 | 0.2813 | 0.2899 | 0.2338 | 0.1788 |
| momentum_reversal | neutral | 0.0540 | 0.2486 | 0.2172 | 0.1545 | 0.1254 |
| mom_x_lowvol_20_20 | neutral | 0.0458 | 0.2273 | 0.2014 | 0.1399 | 0.1148 |
| rsi_vol_combo | neutral | 0.0429 | 0.2394 | 0.1791 | 0.1503 | 0.1030 |
| turnover_stability | neutral | 0.0358 | 0.2046 | 0.1752 | 0.1180 | 0.0979 |
| trend_lowvol | neutral | 0.0379 | 0.2988 | 0.1268 | 0.1399 | 0.0723 |
| fund_score | neutral | 0.0299 | 0.2368 | 0.1265 | 0.1033 | 0.0698 |
| rsi_vol_combo | bull | 0.0744 | 0.2746 | 0.2710 | 0.2576 | 0.1704 |
| momentum_reversal | bull | 0.0633 | 0.2835 | 0.2234 | 0.2727 | 0.1422 |
| fund_pb | bull | 0.0561 | 0.2619 | 0.2143 | 0.1477 | 0.1230 |
| mom_x_lowvol_20_20 | bull | 0.0450 | 0.2412 | 0.1864 | 0.2045 | 0.1123 |
| fund_gross_margin | bull | 0.0497 | 0.2640 | 0.1883 | 0.1894 | 0.1120 |
| volatility | bull | 0.0370 | 0.2058 | 0.1800 | 0.1136 | 0.1002 |
| fund_pe | bull | 0.0439 | 0.2583 | 0.1701 | 0.1439 | 0.0973 |
| fund_profit_growth | bear | 0.1753 | 0.2512 | 0.6977 | 0.3973 | 0.4874 |
| fund_score | bear | 0.1118 | 0.2564 | 0.4361 | 0.3699 | 0.2987 |
| fund_pb | bear | 0.1353 | 0.3167 | 0.4273 | 0.2055 | 0.2576 |
| fund_pe | bear | 0.1296 | 0.3453 | 0.3752 | 0.3699 | 0.2570 |
| fund_revenue_growth | bear | 0.0788 | 0.2073 | 0.3800 | 0.2603 | 0.2394 |
| bb_width_20 | bear | 0.0714 | 0.3079 | 0.2320 | 0.2329 | 0.1430 |
| mom_x_lowvol_20_20 | bear | 0.0681 | 0.2877 | 0.2368 | 0.1781 | 0.1395 |
| momentum_reversal | bear | 0.0635 | 0.3321 | 0.1912 | 0.1507 | 0.1100 |
| fund_roe | bear | 0.0491 | 0.2752 | 0.1784 | 0.1781 | 0.1051 |

### 谷子经济

- **Neutral**: ['momentum_reversal', 'trend_lowvol', 'fund_pb'] (单因子IC=0.108, 组合IC=0.1505)
  - weights: [0.3418, 0.3348, 0.3234]
- **Bull**: ['low_downside', 'volatility', 'fund_pb'] (单因子IC=0.0827, 组合IC=0.0933)
  - bull_weights: [0.372, 0.3472, 0.2808]
- **Bear**: ['rsi_vol_combo', 'momentum_reversal', 'trend_lowvol'] (单因子IC=0.1233, 组合IC=0.1489)
  - bear_weights: [0.3531, 0.3462, 0.3007]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| momentum_reversal | neutral | 0.1170 | 0.2128 | 0.5501 | 0.4134 | 0.3887 |
| trend_lowvol | neutral | 0.1204 | 0.2262 | 0.5326 | 0.4301 | 0.3808 |
| fund_pb | neutral | 0.0865 | 0.1613 | 0.5364 | 0.3716 | 0.3679 |
| mom_x_lowvol_20_20 | neutral | 0.1075 | 0.2119 | 0.5073 | 0.3925 | 0.3532 |
| volatility | neutral | 0.1162 | 0.2318 | 0.5015 | 0.3695 | 0.3434 |
| rsi_vol_combo | neutral | 0.1015 | 0.2086 | 0.4865 | 0.3549 | 0.3296 |
| low_downside | neutral | 0.0720 | 0.2050 | 0.3511 | 0.2777 | 0.2243 |
| fund_pe | neutral | 0.0493 | 0.1742 | 0.2829 | 0.1962 | 0.1692 |
| turnover_stability | neutral | 0.0407 | 0.1703 | 0.2389 | 0.1461 | 0.1369 |
| fund_revenue_growth | neutral | 0.0306 | 0.1446 | 0.2115 | 0.2296 | 0.1300 |
| fund_profit_growth | neutral | 0.0218 | 0.1486 | 0.1470 | 0.1044 | 0.0812 |
| fund_score | neutral | 0.0208 | 0.1690 | 0.1231 | 0.1190 | 0.0689 |
| low_downside | bull | 0.0903 | 0.1799 | 0.5020 | 0.3636 | 0.3423 |
| volatility | bull | 0.0943 | 0.1979 | 0.4765 | 0.3409 | 0.3195 |
| fund_pb | bull | 0.0634 | 0.1663 | 0.3811 | 0.3561 | 0.2584 |
| trend_lowvol | bull | 0.0695 | 0.1873 | 0.3709 | 0.3144 | 0.2437 |
| fund_pe | bull | 0.0454 | 0.1572 | 0.2886 | 0.2121 | 0.1749 |
| rsi_vol_combo | bull | 0.0401 | 0.1737 | 0.2307 | 0.1364 | 0.1311 |
| fund_revenue_growth | bull | 0.0294 | 0.1375 | 0.2140 | 0.1667 | 0.1249 |
| turnover_stability | bull | 0.0258 | 0.1500 | 0.1719 | 0.1212 | 0.0964 |
| fund_score | bull | 0.0237 | 0.1526 | 0.1551 | 0.1061 | 0.0858 |
| fund_roe | bull | 0.0179 | 0.1707 | 0.1047 | 0.1212 | 0.0587 |
| wash_sale_score | bull | 0.0135 | 0.1359 | 0.0994 | 0.1008 | 0.0547 |
| rsi_vol_combo | bear | 0.1260 | 0.1826 | 0.6904 | 0.5068 | 0.5202 |
| momentum_reversal | bear | 0.1337 | 0.2118 | 0.6311 | 0.6164 | 0.5101 |
| trend_lowvol | bear | 0.1100 | 0.1838 | 0.5988 | 0.4795 | 0.4430 |
| mom_x_lowvol_20_20 | bear | 0.1100 | 0.2254 | 0.4879 | 0.5890 | 0.3877 |
| turnover_stability | bear | 0.0383 | 0.1443 | 0.2654 | 0.2877 | 0.1709 |
| volatility | bear | 0.0574 | 0.2192 | 0.2616 | 0.2603 | 0.1648 |
| fund_revenue_growth | bear | 0.0307 | 0.1280 | 0.2399 | 0.2055 | 0.1446 |

### 财税数字化

- **Neutral**: ['momentum_reversal', 'trend_lowvol', 'fund_pb'] (单因子IC=0.0935, 组合IC=0.141)
  - weights: [0.354, 0.3279, 0.3182]
- **Bull**: ['fund_pb', 'volatility', 'trend_lowvol'] (单因子IC=0.1188, 组合IC=0.1514)
  - bull_weights: [0.3825, 0.3258, 0.2916]
- **Bear**: ['mom_x_lowvol_20_20'] (单因子IC=0.1436, 组合IC=0.1436)
  - bear_weights: [1.0]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| momentum_reversal | neutral | 0.0962 | 0.2122 | 0.4533 | 0.3424 | 0.3043 |
| trend_lowvol | neutral | 0.1020 | 0.2354 | 0.4334 | 0.3006 | 0.2818 |
| fund_pb | neutral | 0.0824 | 0.1946 | 0.4236 | 0.2912 | 0.2735 |
| mom_x_lowvol_20_20 | neutral | 0.0866 | 0.2198 | 0.3938 | 0.3006 | 0.2561 |
| rsi_vol_combo | neutral | 0.0733 | 0.2023 | 0.3625 | 0.2933 | 0.2344 |
| volatility | neutral | 0.0813 | 0.2396 | 0.3391 | 0.2818 | 0.2173 |
| low_downside | neutral | 0.0679 | 0.2164 | 0.3136 | 0.2902 | 0.2023 |
| fund_score | neutral | 0.0558 | 0.2050 | 0.2723 | 0.1670 | 0.1589 |
| fund_pe | neutral | 0.0617 | 0.2552 | 0.2420 | 0.2035 | 0.1456 |
| fund_profit_growth | neutral | 0.0432 | 0.1877 | 0.2303 | 0.1775 | 0.1356 |
| turnover_stability | neutral | 0.0384 | 0.1832 | 0.2096 | 0.1639 | 0.1220 |
| fund_roe | neutral | 0.0395 | 0.1937 | 0.2041 | 0.1388 | 0.1162 |
| fund_revenue_growth | neutral | 0.0322 | 0.1904 | 0.1690 | 0.1200 | 0.0946 |
| fund_pb | bull | 0.1166 | 0.1989 | 0.5861 | 0.4773 | 0.4329 |
| volatility | bull | 0.1190 | 0.2285 | 0.5205 | 0.4167 | 0.3687 |
| trend_lowvol | bull | 0.1209 | 0.2387 | 0.5066 | 0.3030 | 0.3300 |
| low_downside | bull | 0.0891 | 0.2013 | 0.4428 | 0.2500 | 0.2768 |
| fund_pe | bull | 0.1040 | 0.2902 | 0.3584 | 0.3030 | 0.2335 |
| momentum_reversal | bull | 0.0770 | 0.2350 | 0.3278 | 0.2576 | 0.2061 |
| mom_x_lowvol_20_20 | bull | 0.0759 | 0.2400 | 0.3163 | 0.2008 | 0.1899 |
| rsi_vol_combo | bull | 0.0461 | 0.2125 | 0.2169 | 0.1477 | 0.1245 |
| mom_x_lowvol_20_20 | bear | 0.1436 | 0.1942 | 0.7395 | 0.4795 | 0.5470 |
| momentum_reversal | bear | 0.1090 | 0.2046 | 0.5327 | 0.3151 | 0.3503 |
| trend_lowvol | bear | 0.0921 | 0.1840 | 0.5009 | 0.3699 | 0.3431 |
| fund_revenue_growth | bear | 0.0649 | 0.1846 | 0.3517 | 0.3151 | 0.2313 |
| bb_width_20 | bear | 0.0486 | 0.2034 | 0.2388 | 0.3151 | 0.1570 |
| rsi_vol_combo | bear | 0.0580 | 0.2122 | 0.2734 | 0.1233 | 0.1535 |

### 贬值受益

- **Neutral**: ['fund_pb', 'trend_lowvol'] (单因子IC=0.0846, 组合IC=0.1036)
  - weights: [0.5721, 0.4279]
- **Bull**: ['volatility', 'low_downside', 'turnover_stability'] (单因子IC=0.0747, 组合IC=0.098)
  - bull_weights: [0.3824, 0.3247, 0.2929]
- **Bear**: ['mom_x_lowvol_20_20'] (单因子IC=0.1081, 组合IC=0.1081)
  - bear_weights: [1.0]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| fund_pb | neutral | 0.0829 | 0.1351 | 0.6131 | 0.4175 | 0.4345 |
| trend_lowvol | neutral | 0.0864 | 0.1785 | 0.4842 | 0.3424 | 0.3250 |
| volatility | neutral | 0.0693 | 0.1534 | 0.4515 | 0.3486 | 0.3045 |
| mom_x_lowvol_20_20 | neutral | 0.0704 | 0.1596 | 0.4410 | 0.2818 | 0.2826 |
| fund_pe | neutral | 0.0659 | 0.1566 | 0.4204 | 0.3132 | 0.2760 |
| momentum_reversal | neutral | 0.0645 | 0.1668 | 0.3870 | 0.2547 | 0.2428 |
| low_downside | neutral | 0.0395 | 0.1561 | 0.2530 | 0.2568 | 0.1590 |
| rsi_vol_combo | neutral | 0.0409 | 0.1615 | 0.2531 | 0.1775 | 0.1490 |
| fund_profit_growth | neutral | 0.0301 | 0.1375 | 0.2188 | 0.1399 | 0.1247 |
| turnover_stability | neutral | 0.0213 | 0.1112 | 0.1911 | 0.1378 | 0.1087 |
| volatility | bull | 0.0972 | 0.1440 | 0.6748 | 0.4470 | 0.4882 |
| low_downside | bull | 0.0749 | 0.1314 | 0.5700 | 0.4545 | 0.4145 |
| turnover_stability | bull | 0.0521 | 0.0944 | 0.5515 | 0.3561 | 0.3740 |
| fund_pb | bull | 0.0649 | 0.1341 | 0.4838 | 0.3409 | 0.3243 |
| fund_pe | bull | 0.0540 | 0.1506 | 0.3586 | 0.2576 | 0.2255 |
| trend_lowvol | bull | 0.0452 | 0.1597 | 0.2829 | 0.2727 | 0.1800 |
| mom_x_lowvol_20_20 | bull | 0.0264 | 0.1477 | 0.1787 | 0.2273 | 0.1096 |
| momentum_reversal | bull | 0.0245 | 0.1410 | 0.1741 | 0.2045 | 0.1049 |
| mom_x_lowvol_20_20 | bear | 0.1081 | 0.2032 | 0.5317 | 0.3151 | 0.3496 |
| momentum_reversal | bear | 0.1041 | 0.2259 | 0.4608 | 0.3151 | 0.3030 |
| rsi_vol_combo | bear | 0.0785 | 0.1768 | 0.4439 | 0.2603 | 0.2797 |
| fund_pb | bear | 0.0521 | 0.1325 | 0.3932 | 0.2329 | 0.2424 |
| fund_profit_growth | bear | 0.0457 | 0.1317 | 0.3474 | 0.3425 | 0.2332 |
| trend_lowvol | bear | 0.0577 | 0.2684 | 0.2148 | 0.2329 | 0.1324 |
| bb_width_20 | bear | 0.0374 | 0.1978 | 0.1892 | 0.1507 | 0.1089 |

### 资源开采概念

- **Neutral**: ['trend_lowvol', 'volatility'] (单因子IC=0.0702, 组合IC=0.0801)
  - weights: [0.507, 0.493]
- **Bull**: ['ema20_slope', 'stroke_phase'] (单因子IC=0.0573, 组合IC=0.0666)
  - bull_weights: [0.5323, 0.4677]
- **Bear**: ['fund_pe', 'fund_roe', 'wash_sale_score'] (单因子IC=0.1345, 组合IC=0.1665)
  - bear_weights: [0.3683, 0.3231, 0.3086]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| trend_lowvol | neutral | 0.0720 | 0.3183 | 0.2262 | 0.1649 | 0.1318 |
| volatility | neutral | 0.0684 | 0.3216 | 0.2125 | 0.2056 | 0.1281 |
| fund_pe | neutral | 0.0709 | 0.3204 | 0.2213 | 0.1534 | 0.1276 |
| fund_pb | neutral | 0.0528 | 0.2644 | 0.1999 | 0.1576 | 0.1157 |
| momentum_reversal | neutral | 0.0435 | 0.3387 | 0.1283 | 0.1013 | 0.0706 |
| ema20_slope | bull | 0.0639 | 0.2963 | 0.2155 | 0.1667 | 0.1257 |
| stroke_phase | bull | 0.0507 | 0.2729 | 0.1858 | 0.1894 | 0.1105 |
| ma_alignment | bull | 0.0546 | 0.2958 | 0.1845 | 0.1439 | 0.1055 |
| fund_pb | bull | 0.0447 | 0.2580 | 0.1731 | 0.1212 | 0.0970 |
| vol_confirm | bull | 0.0360 | 0.2788 | 0.1291 | 0.1742 | 0.0758 |
| fund_pe | bear | 0.1500 | 0.2771 | 0.5415 | 0.4521 | 0.3931 |
| fund_roe | bear | 0.1524 | 0.2784 | 0.5473 | 0.2603 | 0.3449 |
| wash_sale_score | bear | 0.1012 | 0.2304 | 0.4392 | 0.5000 | 0.3294 |
| fund_gross_margin | bear | 0.0980 | 0.2219 | 0.4419 | 0.3151 | 0.2905 |
| trend_lowvol | bear | 0.1348 | 0.3335 | 0.4042 | 0.3699 | 0.2769 |
| momentum_reversal | bear | 0.1390 | 0.3906 | 0.3557 | 0.2877 | 0.2290 |
| mom_x_lowvol_20_20 | bear | 0.1113 | 0.3860 | 0.2884 | 0.1507 | 0.1659 |
| rsi_vol_combo | bear | 0.0976 | 0.3545 | 0.2755 | 0.1781 | 0.1623 |
| volatility | bear | 0.1138 | 0.4237 | 0.2686 | 0.2055 | 0.1619 |
| fund_score | bear | 0.0734 | 0.3019 | 0.2430 | 0.2329 | 0.1498 |
| turnover_stability | bear | 0.0542 | 0.2660 | 0.2039 | 0.2055 | 0.1229 |
| low_downside | bear | 0.0755 | 0.3769 | 0.2004 | 0.1507 | 0.1153 |

### 超清视频

- **Neutral**: ['momentum_reversal', 'mom_x_lowvol_20_20', 'trend_lowvol'] (单因子IC=0.0739, 组合IC=0.081)
  - weights: [0.3475, 0.3456, 0.3069]
- **Bull**: ['volatility', 'fund_pb', 'trend_lowvol'] (单因子IC=0.0805, 组合IC=0.0964)
  - bull_weights: [0.3925, 0.3425, 0.2649]
- **Bear**: ['momentum_reversal', 'mom_x_lowvol_20_20'] (单因子IC=0.1316, 组合IC=0.1357)
  - bear_weights: [0.5294, 0.4706]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| momentum_reversal | neutral | 0.0753 | 0.1875 | 0.4015 | 0.3006 | 0.2611 |
| mom_x_lowvol_20_20 | neutral | 0.0707 | 0.1787 | 0.3956 | 0.3132 | 0.2597 |
| trend_lowvol | neutral | 0.0759 | 0.2078 | 0.3652 | 0.2630 | 0.2306 |
| volatility | neutral | 0.0622 | 0.1802 | 0.3450 | 0.2944 | 0.2233 |
| rsi_vol_combo | neutral | 0.0632 | 0.1821 | 0.3469 | 0.2714 | 0.2205 |
| fund_pb | neutral | 0.0528 | 0.1888 | 0.2794 | 0.2234 | 0.1709 |
| fund_pe | neutral | 0.0467 | 0.1637 | 0.2851 | 0.1942 | 0.1702 |
| turnover_stability | neutral | 0.0355 | 0.1325 | 0.2677 | 0.2088 | 0.1618 |
| fund_profit_growth | neutral | 0.0399 | 0.1641 | 0.2434 | 0.1795 | 0.1435 |
| fund_score | neutral | 0.0421 | 0.2072 | 0.2030 | 0.1649 | 0.1183 |
| low_downside | neutral | 0.0354 | 0.1807 | 0.1959 | 0.1816 | 0.1158 |
| fund_revenue_growth | neutral | 0.0254 | 0.1560 | 0.1627 | 0.1127 | 0.0905 |
| volatility | bull | 0.0851 | 0.1700 | 0.5002 | 0.3788 | 0.3449 |
| fund_pb | bull | 0.0817 | 0.1799 | 0.4540 | 0.3258 | 0.3009 |
| trend_lowvol | bull | 0.0749 | 0.2120 | 0.3531 | 0.3182 | 0.2328 |
| turnover_stability | bull | 0.0448 | 0.1293 | 0.3468 | 0.2955 | 0.2246 |
| low_downside | bull | 0.0504 | 0.1610 | 0.3127 | 0.2652 | 0.1978 |
| fund_pe | bull | 0.0530 | 0.1791 | 0.2958 | 0.1818 | 0.1748 |
| mom_x_lowvol_20_20 | bull | 0.0397 | 0.1660 | 0.2392 | 0.2348 | 0.1477 |
| momentum_reversal | bull | 0.0366 | 0.1666 | 0.2199 | 0.1970 | 0.1316 |
| fund_profit_growth | bull | 0.0238 | 0.1327 | 0.1795 | 0.1970 | 0.1074 |
| fund_revenue_growth | bull | 0.0315 | 0.1785 | 0.1763 | 0.1212 | 0.0988 |
| wash_sale_score | bull | 0.0172 | 0.1164 | 0.1473 | 0.1190 | 0.0824 |
| fund_gross_margin | bull | 0.0214 | 0.1692 | 0.1264 | 0.1667 | 0.0737 |
| momentum_reversal | bear | 0.1354 | 0.2054 | 0.6595 | 0.4521 | 0.4788 |
| mom_x_lowvol_20_20 | bear | 0.1278 | 0.2139 | 0.5975 | 0.4247 | 0.4256 |
| rsi_vol_combo | bear | 0.0730 | 0.1513 | 0.4825 | 0.3425 | 0.3239 |
| trend_lowvol | bear | 0.0973 | 0.2318 | 0.4198 | 0.3151 | 0.2761 |
| fund_profit_growth | bear | 0.0536 | 0.1821 | 0.2941 | 0.1233 | 0.1652 |
| fund_gross_margin | bear | 0.0481 | 0.1678 | 0.2868 | 0.1507 | 0.1650 |
| turnover_stability | bear | 0.0309 | 0.1173 | 0.2633 | 0.2329 | 0.1623 |
| wash_sale_score | bear | 0.0279 | 0.1189 | 0.2347 | 0.2000 | 0.1408 |
| bb_width_20 | bear | 0.0465 | 0.2256 | 0.2063 | 0.1233 | 0.1159 |

### 超级品牌

- **Neutral**: ['fund_profit_growth', 'fund_score', 'fund_pb'] (单因子IC=0.0669, 组合IC=0.1003)
  - weights: [0.3559, 0.3551, 0.289]
- **Bull**: ['fund_profit_growth', 'momentum_reversal', 'fund_gross_margin'] (单因子IC=0.0461, 组合IC=0.0679)
  - bull_weights: [0.3973, 0.3064, 0.2962]
- **Bear**: ['trend_lowvol'] (单因子IC=0.2235, 组合IC=0.2235)
  - bear_weights: [1.0]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| fund_profit_growth | neutral | 0.0631 | 0.2092 | 0.3016 | 0.2777 | 0.1927 |
| fund_score | neutral | 0.0752 | 0.2419 | 0.3110 | 0.2359 | 0.1922 |
| fund_pb | neutral | 0.0623 | 0.2385 | 0.2614 | 0.1973 | 0.1565 |
| fund_revenue_growth | neutral | 0.0502 | 0.2213 | 0.2270 | 0.1159 | 0.1266 |
| trend_lowvol | neutral | 0.0428 | 0.2629 | 0.1626 | 0.1305 | 0.0919 |
| fund_profit_growth | bull | 0.0548 | 0.2374 | 0.2310 | 0.2197 | 0.1409 |
| momentum_reversal | bull | 0.0417 | 0.2328 | 0.1792 | 0.2121 | 0.1086 |
| fund_gross_margin | bull | 0.0419 | 0.2219 | 0.1886 | 0.1136 | 0.1050 |
| fund_pb | bull | 0.0349 | 0.2224 | 0.1568 | 0.1667 | 0.0915 |
| top_fractal_volume | bull | 0.0283 | 0.2024 | 0.1396 | 0.1242 | 0.0785 |
| trend_lowvol | bull | 0.0343 | 0.2743 | 0.1252 | 0.1212 | 0.0702 |
| low_downside | bull | 0.0208 | 0.2215 | 0.0939 | 0.1439 | 0.0537 |
| trend_lowvol | bear | 0.2235 | 0.2634 | 0.8485 | 0.5342 | 0.6509 |
| bb_width_20 | bear | 0.1222 | 0.1970 | 0.6205 | 0.5068 | 0.4675 |
| momentum_reversal | bear | 0.1745 | 0.3022 | 0.5775 | 0.3973 | 0.4035 |
| fund_pb | bear | 0.1022 | 0.2349 | 0.4351 | 0.3425 | 0.2920 |
| mom_x_lowvol_20_20 | bear | 0.1165 | 0.3027 | 0.3849 | 0.2329 | 0.2373 |
| rsi_vol_combo | bear | 0.1037 | 0.3110 | 0.3335 | 0.2603 | 0.2101 |
| top_fractal_volume | bear | 0.0639 | 0.1998 | 0.3199 | 0.2500 | 0.1999 |

### 超级电容

- **Neutral**: ['momentum_reversal', 'fund_pe', 'fund_profit_growth'] (单因子IC=0.061, 组合IC=0.0986)
  - weights: [0.3474, 0.3377, 0.3149]
- **Bull**: ['volatility', 'mom_x_lowvol_20_20', 'trend_lowvol'] (单因子IC=0.0836, 组合IC=0.104)
  - bull_weights: [0.449, 0.2802, 0.2708]
- **Bear**: ['trend_lowvol', 'momentum_reversal', 'fund_pe'] (单因子IC=0.0913, 组合IC=0.1541)
  - bear_weights: [0.355, 0.3437, 0.3012]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| momentum_reversal | neutral | 0.0706 | 0.2549 | 0.2769 | 0.2401 | 0.1717 |
| fund_pe | neutral | 0.0560 | 0.2012 | 0.2783 | 0.1994 | 0.1669 |
| fund_profit_growth | neutral | 0.0564 | 0.2223 | 0.2537 | 0.2265 | 0.1556 |
| rsi_vol_combo | neutral | 0.0609 | 0.2516 | 0.2419 | 0.2255 | 0.1482 |
| mom_x_lowvol_20_20 | neutral | 0.0535 | 0.2365 | 0.2262 | 0.1597 | 0.1312 |
| trend_lowvol | neutral | 0.0547 | 0.2680 | 0.2042 | 0.2296 | 0.1256 |
| fund_pb | neutral | 0.0522 | 0.2450 | 0.2130 | 0.1681 | 0.1244 |
| fund_revenue_growth | neutral | 0.0432 | 0.2110 | 0.2047 | 0.1190 | 0.1146 |
| fund_score | neutral | 0.0407 | 0.2454 | 0.1659 | 0.1200 | 0.0929 |
| turnover_stability | neutral | 0.0266 | 0.1993 | 0.1333 | 0.1273 | 0.0751 |
| volatility | bull | 0.0988 | 0.1878 | 0.5259 | 0.3636 | 0.3586 |
| mom_x_lowvol_20_20 | bull | 0.0761 | 0.2176 | 0.3496 | 0.2803 | 0.2238 |
| trend_lowvol | bull | 0.0759 | 0.2114 | 0.3590 | 0.2045 | 0.2162 |
| fund_pe | bull | 0.0677 | 0.2214 | 0.3056 | 0.2273 | 0.1875 |
| momentum_reversal | bull | 0.0673 | 0.2245 | 0.2998 | 0.2462 | 0.1868 |
| fund_pb | bull | 0.0628 | 0.2274 | 0.2760 | 0.3030 | 0.1798 |
| fund_score | bull | 0.0710 | 0.2583 | 0.2751 | 0.1970 | 0.1646 |
| turnover_stability | bull | 0.0559 | 0.2033 | 0.2748 | 0.1970 | 0.1644 |
| fund_revenue_growth | bull | 0.0556 | 0.2078 | 0.2676 | 0.2121 | 0.1622 |
| low_downside | bull | 0.0577 | 0.2266 | 0.2544 | 0.1970 | 0.1523 |
| fund_roe | bull | 0.0607 | 0.2721 | 0.2230 | 0.2197 | 0.1360 |
| top_fractal_volume | bull | 0.0376 | 0.2131 | 0.1766 | 0.1795 | 0.1041 |
| trend_lowvol | bear | 0.1070 | 0.2283 | 0.4685 | 0.3151 | 0.3081 |
| momentum_reversal | bear | 0.0978 | 0.2155 | 0.4536 | 0.3151 | 0.2983 |
| fund_pe | bear | 0.0692 | 0.1778 | 0.3894 | 0.3425 | 0.2614 |
| rsi_vol_combo | bear | 0.0843 | 0.2173 | 0.3880 | 0.2055 | 0.2338 |
| fund_profit_growth | bear | 0.0573 | 0.1946 | 0.2944 | 0.3699 | 0.2016 |
| fund_revenue_growth | bear | 0.0537 | 0.1831 | 0.2935 | 0.2055 | 0.1769 |
| fund_score | bear | 0.0534 | 0.2140 | 0.2494 | 0.2603 | 0.1572 |

### 超超临界发电

- **Neutral**: ['fund_pb', 'momentum_reversal', 'mom_x_lowvol_20_20'] (单因子IC=0.0759, 组合IC=0.1021)
  - weights: [0.3822, 0.3114, 0.3064]
- **Bull**: ['fund_pb'] (单因子IC=0.1047, 组合IC=0.1047)
  - bull_weights: [1.0]
- **Bear**: ['fund_profit_growth', 'mom_x_lowvol_20_20'] (单因子IC=0.1188, 组合IC=0.1549)
  - bear_weights: [0.5458, 0.4542]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| fund_pb | neutral | 0.0804 | 0.2211 | 0.3636 | 0.3006 | 0.2365 |
| momentum_reversal | neutral | 0.0791 | 0.2490 | 0.3176 | 0.2129 | 0.1926 |
| mom_x_lowvol_20_20 | neutral | 0.0683 | 0.2165 | 0.3155 | 0.2015 | 0.1895 |
| fund_profit_growth | neutral | 0.0587 | 0.1979 | 0.2964 | 0.2192 | 0.1807 |
| fund_score | neutral | 0.0510 | 0.1969 | 0.2589 | 0.1795 | 0.1527 |
| trend_lowvol | neutral | 0.0669 | 0.2650 | 0.2524 | 0.1566 | 0.1459 |
| fund_pe | neutral | 0.0597 | 0.2523 | 0.2367 | 0.1942 | 0.1413 |
| rsi_vol_combo | neutral | 0.0492 | 0.2361 | 0.2086 | 0.1315 | 0.1180 |
| fund_roe | neutral | 0.0468 | 0.2384 | 0.1964 | 0.1712 | 0.1150 |
| volatility | neutral | 0.0390 | 0.2404 | 0.1624 | 0.1660 | 0.0947 |
| turnover_stability | neutral | 0.0223 | 0.1900 | 0.1174 | 0.1106 | 0.0652 |
| fund_gross_margin | neutral | 0.0256 | 0.2234 | 0.1147 | 0.1044 | 0.0634 |
| fund_pb | bull | 0.1047 | 0.2132 | 0.4913 | 0.3939 | 0.3424 |
| low_downside | bull | 0.0673 | 0.2032 | 0.3311 | 0.2424 | 0.2057 |
| volatility | bull | 0.0602 | 0.2369 | 0.2543 | 0.1742 | 0.1493 |
| momentum_reversal | bull | 0.0498 | 0.2284 | 0.2180 | 0.2121 | 0.1321 |
| trend_lowvol | bull | 0.0490 | 0.2629 | 0.1862 | 0.1667 | 0.1086 |
| stroke_phase | bull | 0.0291 | 0.1738 | 0.1672 | 0.1250 | 0.0941 |
| mom_x_lowvol_20_20 | bull | 0.0321 | 0.2123 | 0.1512 | 0.1364 | 0.0859 |
| fund_revenue_growth | bull | 0.0226 | 0.1661 | 0.1361 | 0.1288 | 0.0768 |
| fund_profit_growth | bear | 0.1171 | 0.2080 | 0.5628 | 0.3425 | 0.3778 |
| mom_x_lowvol_20_20 | bear | 0.1204 | 0.2625 | 0.4589 | 0.3699 | 0.3143 |
| fund_score | bear | 0.0948 | 0.2102 | 0.4507 | 0.3151 | 0.2963 |
| fund_pe | bear | 0.1194 | 0.2764 | 0.4319 | 0.2603 | 0.2722 |
| momentum_reversal | bear | 0.0996 | 0.2623 | 0.3796 | 0.3425 | 0.2548 |
| bb_width_20 | bear | 0.0911 | 0.2503 | 0.3639 | 0.2055 | 0.2194 |
| fund_roe | bear | 0.0715 | 0.2424 | 0.2952 | 0.1507 | 0.1698 |
| rsi_vol_combo | bear | 0.0697 | 0.2591 | 0.2689 | 0.2603 | 0.1694 |
| fund_pb | bear | 0.0561 | 0.2427 | 0.2310 | 0.2055 | 0.1392 |
| fund_gross_margin | bear | 0.0515 | 0.2456 | 0.2096 | 0.1781 | 0.1235 |
| trend_lowvol | bear | 0.0388 | 0.2144 | 0.1810 | 0.1233 | 0.1017 |

### 超跌股

- **Neutral**: ['mom_x_lowvol_20_20', 'momentum_reversal', 'volatility'] (单因子IC=0.0901, 组合IC=0.1041)
  - weights: [0.3515, 0.3397, 0.3088]
- **Bull**: ['fund_pb', 'volatility', 'momentum_reversal'] (单因子IC=0.1225, 组合IC=0.1731)
  - bull_weights: [0.4008, 0.3787, 0.2205]
- **Bear**: ['trend_lowvol', 'wash_sale_score'] (单因子IC=0.0868, 组合IC=0.1079)
  - bear_weights: [0.5621, 0.4379]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| mom_x_lowvol_20_20 | neutral | 0.0954 | 0.2544 | 0.3752 | 0.2630 | 0.2369 |
| momentum_reversal | neutral | 0.0909 | 0.2541 | 0.3578 | 0.2797 | 0.2289 |
| volatility | neutral | 0.0841 | 0.2553 | 0.3294 | 0.2641 | 0.2082 |
| rsi_vol_combo | neutral | 0.0676 | 0.2502 | 0.2703 | 0.2077 | 0.1632 |
| trend_lowvol | neutral | 0.0700 | 0.2650 | 0.2643 | 0.2088 | 0.1597 |
| fund_pb | neutral | 0.0503 | 0.2084 | 0.2414 | 0.2203 | 0.1473 |
| fund_score | neutral | 0.0539 | 0.2256 | 0.2390 | 0.2317 | 0.1472 |
| turnover_stability | neutral | 0.0371 | 0.2061 | 0.1800 | 0.1597 | 0.1044 |
| low_downside | neutral | 0.0481 | 0.2646 | 0.1818 | 0.1336 | 0.1031 |
| fund_roe | neutral | 0.0413 | 0.2457 | 0.1683 | 0.1461 | 0.0964 |
| fund_profit_growth | neutral | 0.0323 | 0.1939 | 0.1665 | 0.1555 | 0.0962 |
| fund_revenue_growth | neutral | 0.0336 | 0.2135 | 0.1575 | 0.1232 | 0.0885 |
| fund_pb | bull | 0.1289 | 0.1881 | 0.6852 | 0.5606 | 0.5346 |
| volatility | bull | 0.1385 | 0.2077 | 0.6668 | 0.5152 | 0.5051 |
| momentum_reversal | bull | 0.1001 | 0.2314 | 0.4326 | 0.3598 | 0.2942 |
| low_downside | bull | 0.0940 | 0.2094 | 0.4489 | 0.2576 | 0.2823 |
| mom_x_lowvol_20_20 | bull | 0.0937 | 0.2260 | 0.4147 | 0.3182 | 0.2733 |
| fund_pe | bull | 0.0867 | 0.2248 | 0.3855 | 0.3106 | 0.2526 |
| trend_lowvol | bull | 0.0846 | 0.2161 | 0.3916 | 0.2765 | 0.2500 |
| rsi_vol_combo | bull | 0.0786 | 0.2365 | 0.3322 | 0.3182 | 0.2189 |
| fund_score | bull | 0.0496 | 0.2312 | 0.2146 | 0.1477 | 0.1231 |
| fund_profit_growth | bull | 0.0331 | 0.1825 | 0.1812 | 0.1023 | 0.0999 |
| fund_revenue_growth | bull | 0.0255 | 0.2148 | 0.1189 | 0.1212 | 0.0667 |
| fund_roe | bull | 0.0231 | 0.2262 | 0.1021 | 0.1970 | 0.0611 |
| trend_lowvol | bear | 0.1066 | 0.2457 | 0.4338 | 0.2055 | 0.2615 |
| wash_sale_score | bear | 0.0671 | 0.1950 | 0.3441 | 0.1837 | 0.2037 |
| momentum_reversal | bear | 0.0624 | 0.2458 | 0.2539 | 0.1781 | 0.1496 |
| mom_x_lowvol_20_20 | bear | 0.0582 | 0.2284 | 0.2548 | 0.1507 | 0.1466 |
| exhaustion_risk | bear | 0.0482 | 0.2249 | 0.2141 | 0.1731 | 0.1256 |

### 跨境支付

- **Neutral**: ['trend_lowvol', 'fund_pb', 'turnover_stability'] (单因子IC=0.073, 组合IC=0.1209)
  - weights: [0.4675, 0.2696, 0.2628]
- **Bull**: ['low_downside', 'volatility', 'fund_pb'] (单因子IC=0.1436, 组合IC=0.1593)
  - bull_weights: [0.3822, 0.3467, 0.2711]
- **Bear**: ['trend_lowvol', 'mom_x_lowvol_20_20', 'bb_width_20'] (单因子IC=0.1494, 组合IC=0.1799)
  - bear_weights: [0.4318, 0.2845, 0.2836]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| trend_lowvol | neutral | 0.1015 | 0.2781 | 0.3650 | 0.2944 | 0.2362 |
| fund_pb | neutral | 0.0770 | 0.3282 | 0.2347 | 0.1608 | 0.1362 |
| turnover_stability | neutral | 0.0405 | 0.1851 | 0.2190 | 0.2129 | 0.1328 |
| fund_roe | neutral | 0.0572 | 0.2529 | 0.2262 | 0.1545 | 0.1306 |
| volatility | neutral | 0.0665 | 0.3249 | 0.2048 | 0.1524 | 0.1180 |
| momentum_reversal | neutral | 0.0537 | 0.2587 | 0.2077 | 0.1044 | 0.1147 |
| fund_score | neutral | 0.0417 | 0.2088 | 0.1996 | 0.1482 | 0.1146 |
| low_downside | neutral | 0.0613 | 0.3072 | 0.1994 | 0.1378 | 0.1135 |
| mom_x_lowvol_20_20 | neutral | 0.0500 | 0.2500 | 0.2002 | 0.1075 | 0.1108 |
| fund_pe | neutral | 0.0673 | 0.3788 | 0.1777 | 0.1420 | 0.1014 |
| low_downside | bull | 0.1489 | 0.2878 | 0.5173 | 0.4167 | 0.3664 |
| volatility | bull | 0.1489 | 0.2970 | 0.5015 | 0.3258 | 0.3324 |
| fund_pb | bull | 0.1329 | 0.3292 | 0.4036 | 0.2879 | 0.2599 |
| mom_x_lowvol_20_20 | bull | 0.1037 | 0.2520 | 0.4114 | 0.2273 | 0.2524 |
| trend_lowvol | bull | 0.0960 | 0.3057 | 0.3139 | 0.1667 | 0.1831 |
| momentum_reversal | bull | 0.0736 | 0.2570 | 0.2863 | 0.1742 | 0.1681 |
| turnover_stability | bull | 0.0557 | 0.2142 | 0.2602 | 0.1894 | 0.1547 |
| fund_pe | bull | 0.0891 | 0.4053 | 0.2199 | 0.1742 | 0.1291 |
| fund_score | bull | 0.0481 | 0.2159 | 0.2229 | 0.1288 | 0.1258 |
| fund_roe | bull | 0.0452 | 0.2744 | 0.1646 | 0.1061 | 0.0910 |
| rsi_vol_combo | bull | 0.0324 | 0.2367 | 0.1367 | 0.1212 | 0.0766 |
| trend_lowvol | bear | 0.1734 | 0.2875 | 0.6029 | 0.4795 | 0.4460 |
| mom_x_lowvol_20_20 | bear | 0.1365 | 0.2864 | 0.4767 | 0.2329 | 0.2938 |
| bb_width_20 | bear | 0.1384 | 0.3106 | 0.4455 | 0.3151 | 0.2929 |
| momentum_reversal | bear | 0.1222 | 0.2963 | 0.4125 | 0.2603 | 0.2599 |
| fund_profit_growth | bear | 0.0443 | 0.1737 | 0.2551 | 0.1507 | 0.1468 |

### 跨境电商

- **Neutral**: ['fund_pb', 'mom_x_lowvol_20_20', 'momentum_reversal'] (单因子IC=0.0846, 组合IC=0.1234)
  - weights: [0.3683, 0.3239, 0.3078]
- **Bull**: ['volatility', 'low_downside', 'fund_pb'] (单因子IC=0.1006, 组合IC=0.1193)
  - bull_weights: [0.3513, 0.3375, 0.3112]
- **Bear**: ['rsi_vol_combo', 'momentum_reversal', 'turnover_stability'] (单因子IC=0.1041, 组合IC=0.1436)
  - bear_weights: [0.369, 0.3304, 0.3006]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| fund_pb | neutral | 0.0866 | 0.1388 | 0.6237 | 0.3862 | 0.4323 |
| mom_x_lowvol_20_20 | neutral | 0.0837 | 0.1563 | 0.5355 | 0.4196 | 0.3801 |
| momentum_reversal | neutral | 0.0834 | 0.1608 | 0.5188 | 0.3925 | 0.3612 |
| volatility | neutral | 0.0895 | 0.1738 | 0.5148 | 0.3967 | 0.3595 |
| trend_lowvol | neutral | 0.0888 | 0.1825 | 0.4866 | 0.4050 | 0.3418 |
| fund_pe | neutral | 0.0579 | 0.1444 | 0.4008 | 0.3382 | 0.2682 |
| turnover_stability | neutral | 0.0425 | 0.1048 | 0.4054 | 0.3006 | 0.2636 |
| rsi_vol_combo | neutral | 0.0609 | 0.1649 | 0.3694 | 0.2317 | 0.2275 |
| low_downside | neutral | 0.0525 | 0.1621 | 0.3237 | 0.3006 | 0.2105 |
| fund_profit_growth | neutral | 0.0221 | 0.1262 | 0.1751 | 0.1712 | 0.1026 |
| volatility | bull | 0.1092 | 0.1473 | 0.7417 | 0.5606 | 0.5787 |
| low_downside | bull | 0.0964 | 0.1321 | 0.7303 | 0.5227 | 0.5560 |
| fund_pb | bull | 0.0960 | 0.1397 | 0.6871 | 0.4924 | 0.5127 |
| turnover_stability | bull | 0.0470 | 0.0990 | 0.4745 | 0.4242 | 0.3379 |
| fund_pe | bull | 0.0574 | 0.1451 | 0.3953 | 0.3258 | 0.2621 |
| momentum_reversal | bull | 0.0478 | 0.1351 | 0.3534 | 0.3788 | 0.2436 |
| mom_x_lowvol_20_20 | bull | 0.0446 | 0.1407 | 0.3173 | 0.3864 | 0.2199 |
| trend_lowvol | bull | 0.0518 | 0.1536 | 0.3371 | 0.2803 | 0.2158 |
| rsi_vol_combo | bull | 0.0387 | 0.1250 | 0.3092 | 0.2727 | 0.1968 |
| rsi_vol_combo | bear | 0.1218 | 0.1528 | 0.7972 | 0.5616 | 0.6225 |
| momentum_reversal | bear | 0.1394 | 0.1815 | 0.7676 | 0.4521 | 0.5573 |
| turnover_stability | bear | 0.0511 | 0.0758 | 0.6732 | 0.5068 | 0.5072 |
| mom_x_lowvol_20_20 | bear | 0.1250 | 0.1852 | 0.6749 | 0.4521 | 0.4900 |
| trend_lowvol | bear | 0.0843 | 0.1725 | 0.4884 | 0.3425 | 0.3278 |
| fund_pe | bear | 0.0327 | 0.1436 | 0.2275 | 0.1507 | 0.1309 |

### 车联网(车路云)

- **Neutral**: ['trend_lowvol'] (单因子IC=0.0993, 组合IC=0.0993)
  - weights: [1.0]
- **Bull**: ['trend_lowvol', 'low_downside'] (单因子IC=0.0979, 组合IC=0.1138)
  - bull_weights: [0.5195, 0.4805]
- **Bear**: ['fund_revenue_growth', 'fund_profit_growth', 'mom_x_lowvol_20_20'] (单因子IC=0.0504, 组合IC=0.0635)
  - bear_weights: [0.4329, 0.3015, 0.2656]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| trend_lowvol | neutral | 0.0993 | 0.1853 | 0.5362 | 0.4134 | 0.3789 |
| momentum_reversal | neutral | 0.0830 | 0.1580 | 0.5251 | 0.3841 | 0.3634 |
| mom_x_lowvol_20_20 | neutral | 0.0796 | 0.1595 | 0.4993 | 0.3820 | 0.3450 |
| volatility | neutral | 0.0781 | 0.1925 | 0.4055 | 0.3591 | 0.2756 |
| fund_pb | neutral | 0.0676 | 0.1644 | 0.4114 | 0.2818 | 0.2636 |
| rsi_vol_combo | neutral | 0.0601 | 0.1559 | 0.3856 | 0.2818 | 0.2472 |
| fund_profit_growth | neutral | 0.0465 | 0.1314 | 0.3540 | 0.3069 | 0.2313 |
| fund_score | neutral | 0.0516 | 0.1646 | 0.3138 | 0.2380 | 0.1943 |
| turnover_stability | neutral | 0.0267 | 0.1019 | 0.2620 | 0.2985 | 0.1701 |
| low_downside | neutral | 0.0479 | 0.1804 | 0.2657 | 0.2255 | 0.1628 |
| fund_pe | neutral | 0.0444 | 0.1707 | 0.2602 | 0.1983 | 0.1559 |
| fund_revenue_growth | neutral | 0.0290 | 0.1218 | 0.2378 | 0.1587 | 0.1378 |
| fund_roe | neutral | 0.0381 | 0.1709 | 0.2230 | 0.1482 | 0.1280 |
| trend_lowvol | bull | 0.1049 | 0.1675 | 0.6261 | 0.4545 | 0.4554 |
| low_downside | bull | 0.0909 | 0.1521 | 0.5979 | 0.4091 | 0.4212 |
| volatility | bull | 0.1039 | 0.1886 | 0.5506 | 0.4318 | 0.3942 |
| turnover_stability | bull | 0.0459 | 0.0896 | 0.5123 | 0.4242 | 0.3648 |
| fund_pb | bull | 0.0722 | 0.1663 | 0.4342 | 0.2955 | 0.2812 |
| mom_x_lowvol_20_20 | bull | 0.0661 | 0.1622 | 0.4076 | 0.3258 | 0.2702 |
| momentum_reversal | bull | 0.0625 | 0.1605 | 0.3893 | 0.3030 | 0.2536 |
| fund_profit_growth | bull | 0.0284 | 0.1209 | 0.2349 | 0.1439 | 0.1343 |
| fund_revenue_growth | bull | 0.0278 | 0.1249 | 0.2223 | 0.1136 | 0.1238 |
| fund_pe | bull | 0.0344 | 0.1720 | 0.1999 | 0.1894 | 0.1189 |
| fund_revenue_growth | bear | 0.0488 | 0.1148 | 0.4252 | 0.4247 | 0.3029 |
| fund_profit_growth | bear | 0.0503 | 0.1534 | 0.3276 | 0.2877 | 0.2109 |
| mom_x_lowvol_20_20 | bear | 0.0521 | 0.1921 | 0.2713 | 0.3699 | 0.1858 |
| trend_lowvol | bear | 0.0502 | 0.1702 | 0.2946 | 0.1233 | 0.1654 |
| turnover_stability | bear | 0.0244 | 0.1007 | 0.2418 | 0.1781 | 0.1424 |
| momentum_reversal | bear | 0.0351 | 0.1972 | 0.1777 | 0.2603 | 0.1120 |

### 辅助生殖

- **Neutral**: ['fund_pb', 'volatility', 'fund_profit_growth'] (单因子IC=0.072, 组合IC=0.1017)
  - weights: [0.3459, 0.3431, 0.3109]
- **Bull**: ['low_downside', 'turnover_stability', 'fund_pb'] (单因子IC=0.0777, 组合IC=0.108)
  - bull_weights: [0.3642, 0.3335, 0.3023]
- **Bear**: ['trend_lowvol', 'mom_x_lowvol_20_20'] (单因子IC=0.1551, 组合IC=0.1828)
  - bear_weights: [0.5385, 0.4615]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| fund_pb | neutral | 0.0736 | 0.1990 | 0.3698 | 0.2693 | 0.2347 |
| volatility | neutral | 0.0858 | 0.2410 | 0.3563 | 0.3069 | 0.2328 |
| fund_profit_growth | neutral | 0.0566 | 0.1665 | 0.3402 | 0.2401 | 0.2109 |
| trend_lowvol | neutral | 0.0746 | 0.2317 | 0.3219 | 0.2317 | 0.1982 |
| momentum_reversal | neutral | 0.0650 | 0.2198 | 0.2957 | 0.1754 | 0.1738 |
| low_downside | neutral | 0.0613 | 0.2346 | 0.2614 | 0.2526 | 0.1637 |
| fund_revenue_growth | neutral | 0.0471 | 0.1785 | 0.2637 | 0.2025 | 0.1585 |
| fund_score | neutral | 0.0529 | 0.1976 | 0.2679 | 0.1628 | 0.1557 |
| mom_x_lowvol_20_20 | neutral | 0.0525 | 0.2139 | 0.2452 | 0.1336 | 0.1390 |
| turnover_stability | neutral | 0.0402 | 0.1742 | 0.2310 | 0.1649 | 0.1346 |
| rsi_vol_combo | neutral | 0.0463 | 0.2038 | 0.2273 | 0.1357 | 0.1291 |
| fund_pe | neutral | 0.0395 | 0.2131 | 0.1855 | 0.1388 | 0.1056 |
| low_downside | bull | 0.0818 | 0.1792 | 0.4564 | 0.3636 | 0.3112 |
| turnover_stability | bull | 0.0816 | 0.1856 | 0.4400 | 0.2955 | 0.2850 |
| fund_pb | bull | 0.0696 | 0.1765 | 0.3942 | 0.3106 | 0.2583 |
| volatility | bull | 0.0798 | 0.2042 | 0.3910 | 0.3030 | 0.2548 |
| fund_profit_growth | bull | 0.0472 | 0.1480 | 0.3189 | 0.2197 | 0.1945 |
| rsi_vol_combo | bull | 0.0472 | 0.1863 | 0.2533 | 0.1970 | 0.1516 |
| momentum_reversal | bull | 0.0564 | 0.2218 | 0.2541 | 0.1212 | 0.1424 |
| fund_revenue_growth | bull | 0.0340 | 0.1566 | 0.2171 | 0.1818 | 0.1283 |
| fund_score | bull | 0.0416 | 0.1978 | 0.2104 | 0.1818 | 0.1243 |
| trend_lowvol | bull | 0.0521 | 0.2391 | 0.2181 | 0.1212 | 0.1223 |
| fund_pe | bull | 0.0203 | 0.2181 | 0.0933 | 0.1212 | 0.0523 |
| trend_lowvol | bear | 0.1581 | 0.2049 | 0.7718 | 0.4795 | 0.5709 |
| mom_x_lowvol_20_20 | bear | 0.1521 | 0.2300 | 0.6615 | 0.4795 | 0.4893 |
| momentum_reversal | bear | 0.1493 | 0.2265 | 0.6593 | 0.3973 | 0.4606 |
| rsi_vol_combo | bear | 0.1308 | 0.2272 | 0.5757 | 0.4521 | 0.4180 |
| turnover_stability | bear | 0.0679 | 0.1454 | 0.4668 | 0.3973 | 0.3262 |
| fund_revenue_growth | bear | 0.0733 | 0.1880 | 0.3898 | 0.3151 | 0.2563 |
| fund_gross_margin | bear | 0.0624 | 0.1962 | 0.3181 | 0.1781 | 0.1874 |
| fund_score | bear | 0.0412 | 0.2231 | 0.1848 | 0.2877 | 0.1190 |

### 边缘计算

- **Neutral**: ['momentum_reversal', 'trend_lowvol'] (单因子IC=0.0886, 组合IC=0.0992)
  - weights: [0.5111, 0.4889]
- **Bull**: ['volatility', 'trend_lowvol', 'low_downside'] (单因子IC=0.0979, 组合IC=0.1118)
  - bull_weights: [0.3862, 0.3162, 0.2976]
- **Bear**: ['trend_lowvol'] (单因子IC=0.1436, 组合IC=0.1436)
  - bear_weights: [1.0]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| momentum_reversal | neutral | 0.0870 | 0.1610 | 0.5403 | 0.3570 | 0.3666 |
| trend_lowvol | neutral | 0.0901 | 0.1790 | 0.5036 | 0.3925 | 0.3506 |
| rsi_vol_combo | neutral | 0.0761 | 0.1556 | 0.4891 | 0.3737 | 0.3359 |
| mom_x_lowvol_20_20 | neutral | 0.0787 | 0.1615 | 0.4872 | 0.3570 | 0.3305 |
| fund_pb | neutral | 0.0654 | 0.1562 | 0.4185 | 0.3257 | 0.2774 |
| volatility | neutral | 0.0765 | 0.1853 | 0.4128 | 0.3152 | 0.2715 |
| fund_profit_growth | neutral | 0.0497 | 0.1432 | 0.3473 | 0.2171 | 0.2113 |
| fund_pe | neutral | 0.0497 | 0.1657 | 0.3000 | 0.2651 | 0.1898 |
| fund_score | neutral | 0.0501 | 0.1736 | 0.2884 | 0.1775 | 0.1698 |
| low_downside | neutral | 0.0440 | 0.1765 | 0.2493 | 0.2276 | 0.1530 |
| fund_roe | neutral | 0.0406 | 0.1852 | 0.2192 | 0.1482 | 0.1259 |
| turnover_stability | neutral | 0.0239 | 0.1242 | 0.1929 | 0.1962 | 0.1154 |
| fund_revenue_growth | neutral | 0.0253 | 0.1387 | 0.1827 | 0.1545 | 0.1055 |
| volatility | bull | 0.1078 | 0.1648 | 0.6540 | 0.4697 | 0.4806 |
| trend_lowvol | bull | 0.0998 | 0.1893 | 0.5274 | 0.4924 | 0.3936 |
| low_downside | bull | 0.0860 | 0.1591 | 0.5403 | 0.3712 | 0.3704 |
| fund_pb | bull | 0.0768 | 0.1582 | 0.4857 | 0.3712 | 0.3330 |
| fund_pe | bull | 0.0564 | 0.1467 | 0.3841 | 0.3409 | 0.2575 |
| turnover_stability | bull | 0.0400 | 0.1070 | 0.3742 | 0.2652 | 0.2367 |
| momentum_reversal | bull | 0.0500 | 0.1509 | 0.3313 | 0.2727 | 0.2108 |
| fund_profit_growth | bull | 0.0441 | 0.1354 | 0.3258 | 0.2500 | 0.2037 |
| mom_x_lowvol_20_20 | bull | 0.0424 | 0.1541 | 0.2752 | 0.2197 | 0.1678 |
| fund_revenue_growth | bull | 0.0426 | 0.1612 | 0.2642 | 0.1212 | 0.1481 |
| rsi_vol_combo | bull | 0.0344 | 0.1413 | 0.2436 | 0.1742 | 0.1430 |
| fund_score | bull | 0.0392 | 0.1779 | 0.2205 | 0.1515 | 0.1269 |
| trend_lowvol | bear | 0.1436 | 0.1475 | 0.9735 | 0.7260 | 0.8402 |
| momentum_reversal | bear | 0.1138 | 0.1551 | 0.7341 | 0.5068 | 0.5531 |
| mom_x_lowvol_20_20 | bear | 0.0910 | 0.1554 | 0.5859 | 0.4795 | 0.4334 |
| rsi_vol_combo | bear | 0.0841 | 0.1434 | 0.5861 | 0.4521 | 0.4255 |
| fund_profit_growth | bear | 0.0665 | 0.1571 | 0.4232 | 0.2603 | 0.2667 |
| fund_score | bear | 0.0624 | 0.1883 | 0.3316 | 0.1507 | 0.1908 |
| fund_pb | bear | 0.0524 | 0.1759 | 0.2979 | 0.1233 | 0.1673 |

### 近期新高

- **Neutral**: ['trend_lowvol'] (单因子IC=0.0883, 组合IC=0.0883)
  - weights: [1.0]
- **Bull**: ['volatility', 'low_downside', 'fund_pb'] (单因子IC=0.0594, 组合IC=0.0731)
  - bull_weights: [0.3446, 0.3397, 0.3157]
- **Bear**: ['rsi_vol_combo', 'momentum_reversal', 'trend_lowvol'] (单因子IC=0.0945, 组合IC=0.1048)
  - bear_weights: [0.3635, 0.3204, 0.3161]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| trend_lowvol | neutral | 0.0883 | 0.1870 | 0.4722 | 0.3779 | 0.3253 |
| momentum_reversal | neutral | 0.0743 | 0.1804 | 0.4120 | 0.3236 | 0.2727 |
| mom_x_lowvol_20_20 | neutral | 0.0725 | 0.1761 | 0.4116 | 0.2944 | 0.2664 |
| fund_pb | neutral | 0.0557 | 0.1693 | 0.3289 | 0.2359 | 0.2032 |
| volatility | neutral | 0.0566 | 0.1811 | 0.3125 | 0.2547 | 0.1961 |
| rsi_vol_combo | neutral | 0.0509 | 0.1743 | 0.2920 | 0.2443 | 0.1816 |
| fund_pe | neutral | 0.0444 | 0.1602 | 0.2772 | 0.2714 | 0.1762 |
| fund_profit_growth | neutral | 0.0463 | 0.1585 | 0.2923 | 0.1921 | 0.1742 |
| turnover_stability | neutral | 0.0286 | 0.1216 | 0.2353 | 0.2380 | 0.1457 |
| fund_revenue_growth | neutral | 0.0306 | 0.1518 | 0.2015 | 0.1649 | 0.1174 |
| low_downside | neutral | 0.0307 | 0.1941 | 0.1579 | 0.1482 | 0.0907 |
| volatility | bull | 0.0650 | 0.1831 | 0.3547 | 0.2197 | 0.2163 |
| low_downside | bull | 0.0555 | 0.1588 | 0.3496 | 0.2197 | 0.2132 |
| fund_pb | bull | 0.0576 | 0.1839 | 0.3133 | 0.2652 | 0.1982 |
| fund_pe | bull | 0.0402 | 0.1605 | 0.2502 | 0.2121 | 0.1516 |
| rsi_vol_combo | bull | 0.0310 | 0.1602 | 0.1935 | 0.2045 | 0.1166 |
| turnover_stability | bull | 0.0213 | 0.1156 | 0.1843 | 0.1212 | 0.1033 |
| fund_profit_growth | bull | 0.0285 | 0.1636 | 0.1744 | 0.1061 | 0.0965 |
| fund_gross_margin | bull | 0.0221 | 0.1348 | 0.1636 | 0.1515 | 0.0942 |
| rsi_vol_combo | bear | 0.0838 | 0.1622 | 0.5164 | 0.5342 | 0.3962 |
| momentum_reversal | bear | 0.1010 | 0.1981 | 0.5098 | 0.3699 | 0.3492 |
| trend_lowvol | bear | 0.0986 | 0.1922 | 0.5133 | 0.3425 | 0.3445 |
| mom_x_lowvol_20_20 | bear | 0.1034 | 0.2042 | 0.5067 | 0.3151 | 0.3331 |
| fund_profit_growth | bear | 0.0591 | 0.1505 | 0.3930 | 0.2603 | 0.2476 |
| fund_pb | bear | 0.0691 | 0.1971 | 0.3504 | 0.1781 | 0.2064 |
| turnover_stability | bear | 0.0311 | 0.0907 | 0.3426 | 0.1781 | 0.2018 |

### 通信技术

- **Neutral**: ['mom_x_lowvol_20_20', 'momentum_reversal', 'fund_pb'] (单因子IC=0.0821, 组合IC=0.11)
  - weights: [0.3448, 0.33, 0.3253]
- **Bull**: ['fund_pb', 'turnover_stability'] (单因子IC=0.0603, 组合IC=0.0933)
  - bull_weights: [0.511, 0.489]
- **Bear**: ['mom_x_lowvol_20_20'] (单因子IC=0.1003, 组合IC=0.1003)
  - bear_weights: [1.0]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| mom_x_lowvol_20_20 | neutral | 0.0864 | 0.1518 | 0.5691 | 0.4154 | 0.4028 |
| momentum_reversal | neutral | 0.0859 | 0.1569 | 0.5480 | 0.4071 | 0.3855 |
| fund_pb | neutral | 0.0741 | 0.1369 | 0.5409 | 0.4050 | 0.3800 |
| trend_lowvol | neutral | 0.0877 | 0.1719 | 0.5099 | 0.3779 | 0.3513 |
| rsi_vol_combo | neutral | 0.0683 | 0.1545 | 0.4423 | 0.3382 | 0.2960 |
| fund_pe | neutral | 0.0534 | 0.1346 | 0.3966 | 0.3236 | 0.2625 |
| volatility | neutral | 0.0644 | 0.1739 | 0.3702 | 0.3173 | 0.2439 |
| fund_profit_growth | neutral | 0.0474 | 0.1265 | 0.3745 | 0.2547 | 0.2349 |
| turnover_stability | neutral | 0.0337 | 0.0948 | 0.3561 | 0.3132 | 0.2338 |
| fund_score | neutral | 0.0445 | 0.1670 | 0.2663 | 0.1942 | 0.1590 |
| fund_revenue_growth | neutral | 0.0295 | 0.1297 | 0.2275 | 0.1691 | 0.1330 |
| fund_roe | neutral | 0.0305 | 0.1787 | 0.1705 | 0.1294 | 0.0963 |
| low_downside | neutral | 0.0245 | 0.1641 | 0.1496 | 0.1044 | 0.0826 |
| fund_pb | bull | 0.0810 | 0.1473 | 0.5501 | 0.3939 | 0.3834 |
| turnover_stability | bull | 0.0395 | 0.0747 | 0.5294 | 0.3864 | 0.3670 |
| fund_pe | bull | 0.0659 | 0.1321 | 0.4991 | 0.3788 | 0.3440 |
| volatility | bull | 0.0780 | 0.1648 | 0.4733 | 0.4091 | 0.3334 |
| low_downside | bull | 0.0694 | 0.1557 | 0.4454 | 0.3561 | 0.3020 |
| fund_profit_growth | bull | 0.0520 | 0.1305 | 0.3985 | 0.3182 | 0.2627 |
| fund_score | bull | 0.0569 | 0.1732 | 0.3286 | 0.2424 | 0.2041 |
| fund_revenue_growth | bull | 0.0441 | 0.1457 | 0.3024 | 0.1894 | 0.1799 |
| trend_lowvol | bull | 0.0465 | 0.1841 | 0.2526 | 0.2424 | 0.1569 |
| mom_x_lowvol_20_20 | bull | 0.0349 | 0.1401 | 0.2493 | 0.2045 | 0.1501 |
| momentum_reversal | bull | 0.0249 | 0.1465 | 0.1698 | 0.1515 | 0.0977 |
| mom_x_lowvol_20_20 | bear | 0.1003 | 0.1570 | 0.6393 | 0.4521 | 0.4641 |
| momentum_reversal | bear | 0.0925 | 0.1589 | 0.5820 | 0.3699 | 0.3986 |
| rsi_vol_combo | bear | 0.0682 | 0.1330 | 0.5126 | 0.4521 | 0.3722 |
| trend_lowvol | bear | 0.0899 | 0.1819 | 0.4940 | 0.1507 | 0.2842 |
| fund_revenue_growth | bear | 0.0257 | 0.1358 | 0.1894 | 0.1233 | 0.1064 |

### 通用航空

- **Neutral**: ['trend_lowvol'] (单因子IC=0.1054, 组合IC=0.1054)
  - weights: [1.0]
- **Bull**: ['low_downside', 'volatility', 'fund_pb'] (单因子IC=0.121, 组合IC=0.1537)
  - bull_weights: [0.3671, 0.3202, 0.3127]
- **Bear**: ['rsi_vol_combo', 'momentum_reversal'] (单因子IC=0.1357, 组合IC=0.1435)
  - bear_weights: [0.514, 0.486]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| trend_lowvol | neutral | 0.1054 | 0.2601 | 0.4053 | 0.2797 | 0.2593 |
| momentum_reversal | neutral | 0.0754 | 0.2575 | 0.2927 | 0.2098 | 0.1771 |
| rsi_vol_combo | neutral | 0.0625 | 0.2492 | 0.2509 | 0.1587 | 0.1453 |
| mom_x_lowvol_20_20 | neutral | 0.0524 | 0.2401 | 0.2180 | 0.1691 | 0.1274 |
| volatility | neutral | 0.0505 | 0.2358 | 0.2141 | 0.1722 | 0.1255 |
| fund_pb | neutral | 0.0391 | 0.2503 | 0.1563 | 0.1524 | 0.0901 |
| fund_profit_growth | neutral | 0.0316 | 0.2089 | 0.1511 | 0.1002 | 0.0831 |
| fund_score | neutral | 0.0315 | 0.2604 | 0.1211 | 0.1096 | 0.0672 |
| fund_revenue_growth | neutral | 0.0242 | 0.2326 | 0.1041 | 0.1033 | 0.0574 |
| low_downside | bull | 0.1419 | 0.2353 | 0.6030 | 0.4242 | 0.4294 |
| volatility | bull | 0.1134 | 0.2202 | 0.5150 | 0.4545 | 0.3746 |
| fund_pb | bull | 0.1077 | 0.2097 | 0.5137 | 0.4242 | 0.3658 |
| trend_lowvol | bull | 0.1422 | 0.3084 | 0.4611 | 0.4394 | 0.3319 |
| mom_x_lowvol_20_20 | bull | 0.0863 | 0.2511 | 0.3436 | 0.2955 | 0.2225 |
| momentum_reversal | bull | 0.0941 | 0.2761 | 0.3406 | 0.3030 | 0.2219 |
| rsi_vol_combo | bull | 0.0620 | 0.2494 | 0.2488 | 0.1288 | 0.1404 |
| turnover_stability | bull | 0.0434 | 0.2294 | 0.1890 | 0.1667 | 0.1102 |
| rsi_vol_combo | bear | 0.1297 | 0.2213 | 0.5863 | 0.5068 | 0.4417 |
| momentum_reversal | bear | 0.1416 | 0.2508 | 0.5646 | 0.4795 | 0.4176 |
| mom_x_lowvol_20_20 | bear | 0.1079 | 0.2270 | 0.4754 | 0.3973 | 0.3321 |
| trend_lowvol | bear | 0.1161 | 0.2906 | 0.3994 | 0.2877 | 0.2571 |
| fund_gross_margin | bear | 0.0935 | 0.2534 | 0.3689 | 0.2055 | 0.2224 |
| fund_roe | bear | 0.1014 | 0.3371 | 0.3008 | 0.1233 | 0.1690 |
| turnover_stability | bear | 0.0428 | 0.1714 | 0.2495 | 0.3425 | 0.1675 |
| fund_pb | bear | 0.0678 | 0.2650 | 0.2559 | 0.1233 | 0.1437 |
| fund_score | bear | 0.0757 | 0.3107 | 0.2436 | 0.1233 | 0.1368 |
| vol_confirm | bear | 0.0459 | 0.2113 | 0.2170 | 0.2329 | 0.1338 |

### 造纸印刷

- **Neutral**: ['fund_pb'] (单因子IC=0.0721, 组合IC=0.0721)
  - weights: [1.0]
- **Bull**: ['volatility', 'fund_pb', 'fund_pe'] (单因子IC=0.105, 组合IC=0.1339)
  - bull_weights: [0.3639, 0.3223, 0.3138]
- **Bear**: ['fund_profit_growth', 'mom_x_lowvol_20_20'] (单因子IC=0.0776, 组合IC=0.1148)
  - bear_weights: [0.5325, 0.4675]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| fund_pb | neutral | 0.0721 | 0.2371 | 0.3041 | 0.2547 | 0.1908 |
| volatility | neutral | 0.0572 | 0.2159 | 0.2652 | 0.1952 | 0.1585 |
| fund_profit_growth | neutral | 0.0420 | 0.1910 | 0.2198 | 0.1701 | 0.1286 |
| mom_x_lowvol_20_20 | neutral | 0.0522 | 0.2340 | 0.2232 | 0.1514 | 0.1285 |
| trend_lowvol | neutral | 0.0509 | 0.2396 | 0.2124 | 0.1127 | 0.1182 |
| low_downside | neutral | 0.0443 | 0.2291 | 0.1933 | 0.1628 | 0.1124 |
| momentum_reversal | neutral | 0.0460 | 0.2313 | 0.1991 | 0.1263 | 0.1121 |
| turnover_stability | neutral | 0.0330 | 0.1946 | 0.1694 | 0.1054 | 0.0936 |
| fund_pe | neutral | 0.0329 | 0.2316 | 0.1422 | 0.1336 | 0.0806 |
| volatility | bull | 0.1080 | 0.2148 | 0.5030 | 0.2879 | 0.3239 |
| fund_pb | bull | 0.1042 | 0.2422 | 0.4302 | 0.3333 | 0.2868 |
| fund_pe | bull | 0.1026 | 0.2520 | 0.4073 | 0.3712 | 0.2793 |
| rsi_vol_combo | bull | 0.0586 | 0.2215 | 0.2645 | 0.2348 | 0.1633 |
| mom_x_lowvol_20_20 | bull | 0.0556 | 0.2097 | 0.2650 | 0.2273 | 0.1626 |
| low_downside | bull | 0.0543 | 0.2070 | 0.2622 | 0.1515 | 0.1510 |
| momentum_reversal | bull | 0.0401 | 0.2226 | 0.1799 | 0.1894 | 0.1070 |
| fund_profit_growth | bear | 0.0608 | 0.1584 | 0.3836 | 0.3425 | 0.2575 |
| mom_x_lowvol_20_20 | bear | 0.0944 | 0.2517 | 0.3750 | 0.2055 | 0.2260 |
| momentum_reversal | bear | 0.0802 | 0.2731 | 0.2936 | 0.1507 | 0.1689 |
| trend_lowvol | bear | 0.0572 | 0.2291 | 0.2496 | 0.2740 | 0.1590 |
| rsi_vol_combo | bear | 0.0480 | 0.1812 | 0.2651 | 0.1507 | 0.1525 |
| fund_score | bear | 0.0380 | 0.1798 | 0.2115 | 0.2877 | 0.1362 |

### 酿酒概念

- **Neutral**: ['fund_pb', 'fund_profit_growth', 'turnover_stability'] (单因子IC=0.0624, 组合IC=0.0836)
  - weights: [0.34, 0.3379, 0.3221]
- **Bull**: ['stroke_phase'] (单因子IC=0.0463, 组合IC=0.0463)
  - bull_weights: [1.0]
- **Bear**: ['fund_pb', 'mom_x_lowvol_20_20', 'trend_lowvol'] (单因子IC=0.1331, 组合IC=0.2118)
  - bear_weights: [0.4089, 0.3211, 0.27]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| fund_pb | neutral | 0.0724 | 0.2793 | 0.2593 | 0.2119 | 0.1571 |
| fund_profit_growth | neutral | 0.0618 | 0.2361 | 0.2616 | 0.1942 | 0.1562 |
| turnover_stability | neutral | 0.0531 | 0.2257 | 0.2351 | 0.2662 | 0.1489 |
| trend_lowvol | neutral | 0.0616 | 0.2932 | 0.2101 | 0.1555 | 0.1214 |
| low_downside | neutral | 0.0436 | 0.2449 | 0.1781 | 0.1670 | 0.1039 |
| mom_x_lowvol_20_20 | neutral | 0.0393 | 0.2317 | 0.1697 | 0.1357 | 0.0964 |
| fund_revenue_growth | neutral | 0.0370 | 0.2443 | 0.1514 | 0.1086 | 0.0839 |
| momentum_reversal | neutral | 0.0277 | 0.2663 | 0.1042 | 0.1002 | 0.0573 |
| stroke_phase | bull | 0.0463 | 0.2162 | 0.2142 | 0.1894 | 0.1274 |
| fund_pb | bear | 0.1770 | 0.3143 | 0.5630 | 0.4247 | 0.4010 |
| mom_x_lowvol_20_20 | bear | 0.1023 | 0.2224 | 0.4597 | 0.3699 | 0.3149 |
| trend_lowvol | bear | 0.1201 | 0.2734 | 0.4393 | 0.2055 | 0.2648 |
| bb_width_20 | bear | 0.1178 | 0.3050 | 0.3862 | 0.3425 | 0.2592 |
| momentum_reversal | bear | 0.1306 | 0.3041 | 0.4293 | 0.2055 | 0.2588 |

### 量子科技

- **Neutral**: ['fund_pb', 'momentum_reversal'] (单因子IC=0.0811, 组合IC=0.1091)
  - weights: [0.5061, 0.4939]
- **Bull**: ['volatility', 'fund_profit_growth', 'trend_lowvol'] (单因子IC=0.0825, 组合IC=0.1071)
  - bull_weights: [0.3359, 0.3332, 0.331]
- **Bear**: ['trend_lowvol'] (单因子IC=0.1143, 组合IC=0.1143)
  - bear_weights: [1.0]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| fund_pb | neutral | 0.0789 | 0.1992 | 0.3962 | 0.2839 | 0.2544 |
| momentum_reversal | neutral | 0.0833 | 0.2201 | 0.3784 | 0.3121 | 0.2483 |
| mom_x_lowvol_20_20 | neutral | 0.0702 | 0.2163 | 0.3246 | 0.2693 | 0.2060 |
| trend_lowvol | neutral | 0.0750 | 0.2359 | 0.3180 | 0.2213 | 0.1942 |
| rsi_vol_combo | neutral | 0.0623 | 0.2084 | 0.2988 | 0.2213 | 0.1825 |
| fund_pe | neutral | 0.0675 | 0.2298 | 0.2936 | 0.2088 | 0.1775 |
| fund_profit_growth | neutral | 0.0478 | 0.1765 | 0.2710 | 0.2276 | 0.1663 |
| fund_score | neutral | 0.0593 | 0.2264 | 0.2622 | 0.2150 | 0.1593 |
| volatility | neutral | 0.0586 | 0.2407 | 0.2435 | 0.1962 | 0.1457 |
| fund_roe | neutral | 0.0554 | 0.2404 | 0.2304 | 0.1441 | 0.1318 |
| turnover_stability | neutral | 0.0345 | 0.1746 | 0.1977 | 0.1190 | 0.1106 |
| fund_revenue_growth | neutral | 0.0271 | 0.1732 | 0.1566 | 0.1983 | 0.0938 |
| low_downside | neutral | 0.0339 | 0.2370 | 0.1432 | 0.1273 | 0.0807 |
| volatility | bull | 0.0912 | 0.2343 | 0.3891 | 0.2424 | 0.2417 |
| fund_profit_growth | bull | 0.0761 | 0.2043 | 0.3723 | 0.2879 | 0.2398 |
| trend_lowvol | bull | 0.0802 | 0.2219 | 0.3614 | 0.3182 | 0.2382 |
| turnover_stability | bull | 0.0618 | 0.1681 | 0.3676 | 0.2462 | 0.2290 |
| fund_revenue_growth | bull | 0.0777 | 0.2187 | 0.3552 | 0.2273 | 0.2180 |
| fund_pe | bull | 0.0671 | 0.2092 | 0.3208 | 0.3030 | 0.2090 |
| fund_score | bull | 0.0863 | 0.2580 | 0.3345 | 0.1818 | 0.1976 |
| fund_roe | bull | 0.0681 | 0.2432 | 0.2802 | 0.1894 | 0.1667 |
| mom_x_lowvol_20_20 | bull | 0.0523 | 0.2068 | 0.2528 | 0.2879 | 0.1628 |
| low_downside | bull | 0.0564 | 0.2166 | 0.2603 | 0.1970 | 0.1558 |
| momentum_reversal | bull | 0.0433 | 0.2077 | 0.2085 | 0.2045 | 0.1256 |
| rsi_vol_combo | bull | 0.0298 | 0.1898 | 0.1573 | 0.1288 | 0.0888 |
| stroke_phase | bull | 0.0199 | 0.1571 | 0.1265 | 0.1212 | 0.0709 |
| trend_lowvol | bear | 0.1143 | 0.2510 | 0.4554 | 0.3973 | 0.3181 |
| rsi_vol_combo | bear | 0.0662 | 0.1844 | 0.3593 | 0.2877 | 0.2313 |
| momentum_reversal | bear | 0.0812 | 0.2373 | 0.3422 | 0.2877 | 0.2203 |
| bb_width_20 | bear | 0.0676 | 0.1894 | 0.3568 | 0.1781 | 0.2102 |
| top_fractal_volume | bear | 0.0541 | 0.1760 | 0.3077 | 0.3158 | 0.2024 |
| mom_x_lowvol_20_20 | bear | 0.0672 | 0.2461 | 0.2731 | 0.1507 | 0.1571 |
| fund_gross_margin | bear | 0.0426 | 0.2110 | 0.2020 | 0.1781 | 0.1190 |

### 钒电池

- **Neutral**: ['fund_pe', 'fund_pb', 'trend_lowvol'] (单因子IC=0.0886, 组合IC=0.128)
  - weights: [0.4221, 0.3458, 0.2321]
- **Bull**: ['low_downside', 'turnover_stability'] (单因子IC=0.1082, 组合IC=0.1464)
  - bull_weights: [0.5868, 0.4132]
- **Bear**: ['fund_revenue_growth', 'fund_score', 'momentum_reversal'] (单因子IC=0.1352, 组合IC=0.1757)
  - bear_weights: [0.4272, 0.2993, 0.2735]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| fund_pe | neutral | 0.1058 | 0.2703 | 0.3915 | 0.2860 | 0.2518 |
| fund_pb | neutral | 0.0897 | 0.2761 | 0.3250 | 0.2693 | 0.2063 |
| trend_lowvol | neutral | 0.0703 | 0.2976 | 0.2362 | 0.1722 | 0.1384 |
| volatility | neutral | 0.0461 | 0.2668 | 0.1726 | 0.1534 | 0.0996 |
| fund_profit_growth | neutral | 0.0428 | 0.2484 | 0.1721 | 0.1451 | 0.0986 |
| momentum_reversal | neutral | 0.0486 | 0.2985 | 0.1628 | 0.1169 | 0.0909 |
| mom_x_lowvol_20_20 | neutral | 0.0454 | 0.2874 | 0.1581 | 0.1461 | 0.0906 |
| wash_sale_score | neutral | 0.0282 | 0.2433 | 0.1160 | 0.1124 | 0.0645 |
| fund_roe | neutral | 0.0270 | 0.2673 | 0.1011 | 0.1075 | 0.0560 |
| fund_gross_margin | neutral | 0.0192 | 0.2073 | 0.0926 | 0.1367 | 0.0526 |
| low_downside | bull | 0.1230 | 0.2299 | 0.5352 | 0.4091 | 0.3771 |
| turnover_stability | bull | 0.0933 | 0.2290 | 0.4075 | 0.3030 | 0.2655 |
| volatility | bull | 0.0925 | 0.2794 | 0.3311 | 0.1970 | 0.1982 |
| fund_pb | bull | 0.0607 | 0.2512 | 0.2418 | 0.2576 | 0.1520 |
| fund_pe | bull | 0.0608 | 0.3085 | 0.1970 | 0.1288 | 0.1112 |
| fund_profit_growth | bull | 0.0506 | 0.2547 | 0.1988 | 0.1174 | 0.1111 |
| momentum_reversal | bull | 0.0329 | 0.2580 | 0.1277 | 0.1288 | 0.0721 |
| fund_revenue_growth | bear | 0.1351 | 0.2016 | 0.6698 | 0.5616 | 0.5230 |
| fund_score | bear | 0.1222 | 0.2354 | 0.5193 | 0.4110 | 0.3663 |
| momentum_reversal | bear | 0.1482 | 0.3335 | 0.4444 | 0.5068 | 0.3349 |
| fund_profit_growth | bear | 0.1027 | 0.2135 | 0.4810 | 0.3151 | 0.3163 |
| trend_lowvol | bear | 0.1312 | 0.2994 | 0.4383 | 0.3973 | 0.3062 |
| rsi_vol_combo | bear | 0.1197 | 0.2872 | 0.4167 | 0.3973 | 0.2911 |
| mom_x_lowvol_20_20 | bear | 0.1221 | 0.3559 | 0.3431 | 0.4521 | 0.2491 |
| bb_width_20 | bear | 0.0799 | 0.2848 | 0.2807 | 0.3151 | 0.1846 |
| fund_pe | bear | 0.0551 | 0.2984 | 0.1848 | 0.1507 | 0.1063 |

### 钙钛矿电池

- **Neutral**: ['trend_lowvol', 'mom_x_lowvol_20_20'] (单因子IC=0.0625, 组合IC=0.07)
  - weights: [0.5211, 0.4789]
- **Bull**: ['volatility', 'momentum_reversal', 'trend_lowvol'] (单因子IC=0.0989, 组合IC=0.1187)
  - bull_weights: [0.3979, 0.3049, 0.2972]
- **Bear**: ['mom_x_lowvol_20_20'] (单因子IC=0.1715, 组合IC=0.1715)
  - bear_weights: [1.0]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| trend_lowvol | neutral | 0.0673 | 0.2547 | 0.2643 | 0.2213 | 0.1614 |
| mom_x_lowvol_20_20 | neutral | 0.0577 | 0.2312 | 0.2498 | 0.1879 | 0.1483 |
| momentum_reversal | neutral | 0.0564 | 0.2412 | 0.2338 | 0.1587 | 0.1354 |
| fund_pb | neutral | 0.0493 | 0.2097 | 0.2349 | 0.1420 | 0.1341 |
| rsi_vol_combo | neutral | 0.0400 | 0.2312 | 0.1730 | 0.1461 | 0.0992 |
| turnover_stability | neutral | 0.0213 | 0.2186 | 0.0973 | 0.1211 | 0.0545 |
| fund_pe | neutral | 0.0251 | 0.2605 | 0.0962 | 0.1127 | 0.0535 |
| top_fractal_volume | neutral | 0.0119 | 0.1798 | 0.0664 | 0.1007 | 0.0365 |
| volatility | bull | 0.1128 | 0.2139 | 0.5273 | 0.4318 | 0.3775 |
| momentum_reversal | bull | 0.0902 | 0.2007 | 0.4492 | 0.2879 | 0.2893 |
| trend_lowvol | bull | 0.0938 | 0.2256 | 0.4158 | 0.3561 | 0.2819 |
| low_downside | bull | 0.0756 | 0.1875 | 0.4034 | 0.3182 | 0.2659 |
| rsi_vol_combo | bull | 0.0782 | 0.1919 | 0.4075 | 0.2955 | 0.2640 |
| mom_x_lowvol_20_20 | bull | 0.0800 | 0.1972 | 0.4057 | 0.2727 | 0.2582 |
| turnover_stability | bull | 0.0716 | 0.1968 | 0.3640 | 0.3295 | 0.2419 |
| fund_gross_margin | bull | 0.0871 | 0.2560 | 0.3400 | 0.3030 | 0.2215 |
| fund_pb | bull | 0.0591 | 0.1970 | 0.3001 | 0.2955 | 0.1944 |
| fund_score | bull | 0.0636 | 0.2610 | 0.2439 | 0.2273 | 0.1497 |
| fund_roe | bull | 0.0486 | 0.2359 | 0.2059 | 0.1364 | 0.1170 |
| fund_profit_growth | bull | 0.0369 | 0.1888 | 0.1954 | 0.1591 | 0.1133 |
| fund_pe | bull | 0.0368 | 0.2436 | 0.1509 | 0.1364 | 0.0857 |
| mom_x_lowvol_20_20 | bear | 0.1715 | 0.2730 | 0.6281 | 0.5890 | 0.4991 |
| trend_lowvol | bear | 0.1279 | 0.2356 | 0.5429 | 0.4247 | 0.3867 |
| momentum_reversal | bear | 0.1313 | 0.2606 | 0.5039 | 0.4247 | 0.3589 |
| fund_profit_growth | bear | 0.0896 | 0.1985 | 0.4515 | 0.2877 | 0.2907 |
| fund_gross_margin | bear | 0.0817 | 0.2381 | 0.3430 | 0.2603 | 0.2161 |
| fund_score | bear | 0.0618 | 0.2300 | 0.2687 | 0.2055 | 0.1619 |
| fund_pe | bear | 0.0563 | 0.2493 | 0.2259 | 0.3425 | 0.1516 |
| wash_sale_score | bear | 0.0456 | 0.2092 | 0.2180 | 0.1600 | 0.1264 |
| turnover_stability | bear | 0.0315 | 0.1700 | 0.1853 | 0.2329 | 0.1142 |

### 钠离子电池

- **Neutral**: ['fund_pb', 'volatility'] (单因子IC=0.0768, 组合IC=0.0914)
  - weights: [0.5059, 0.4941]
- **Bull**: ['low_downside'] (单因子IC=0.1017, 组合IC=0.1017)
  - bull_weights: [1.0]
- **Bear**: ['momentum_reversal'] (单因子IC=0.136, 组合IC=0.136)
  - bear_weights: [1.0]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| fund_pb | neutral | 0.0792 | 0.2299 | 0.3445 | 0.2923 | 0.2226 |
| volatility | neutral | 0.0744 | 0.2224 | 0.3343 | 0.3006 | 0.2174 |
| fund_pe | neutral | 0.0618 | 0.2409 | 0.2567 | 0.2129 | 0.1557 |
| trend_lowvol | neutral | 0.0557 | 0.2355 | 0.2363 | 0.1691 | 0.1381 |
| fund_profit_growth | neutral | 0.0399 | 0.1776 | 0.2248 | 0.1827 | 0.1329 |
| mom_x_lowvol_20_20 | neutral | 0.0439 | 0.2028 | 0.2166 | 0.1994 | 0.1299 |
| momentum_reversal | neutral | 0.0455 | 0.2140 | 0.2125 | 0.1649 | 0.1238 |
| fund_revenue_growth | neutral | 0.0324 | 0.1694 | 0.1911 | 0.1597 | 0.1108 |
| fund_score | neutral | 0.0374 | 0.2091 | 0.1790 | 0.1127 | 0.0996 |
| rsi_vol_combo | neutral | 0.0321 | 0.1960 | 0.1636 | 0.1023 | 0.0902 |
| low_downside | bull | 0.1017 | 0.2020 | 0.5035 | 0.4773 | 0.3719 |
| volatility | bull | 0.0844 | 0.2034 | 0.4148 | 0.3409 | 0.2781 |
| fund_pb | bull | 0.0644 | 0.1870 | 0.3443 | 0.2576 | 0.2165 |
| turnover_stability | bull | 0.0364 | 0.1395 | 0.2609 | 0.2955 | 0.1690 |
| fund_profit_growth | bull | 0.0457 | 0.1991 | 0.2294 | 0.3258 | 0.1521 |
| fund_score | bull | 0.0239 | 0.2219 | 0.1077 | 0.1136 | 0.0600 |
| momentum_reversal | bear | 0.1360 | 0.1793 | 0.7588 | 0.5890 | 0.6029 |
| rsi_vol_combo | bear | 0.1131 | 0.1482 | 0.7635 | 0.4521 | 0.5543 |
| mom_x_lowvol_20_20 | bear | 0.1202 | 0.1954 | 0.6151 | 0.3973 | 0.4298 |
| trend_lowvol | bear | 0.0841 | 0.1994 | 0.4218 | 0.2329 | 0.2600 |
| fund_revenue_growth | bear | 0.0550 | 0.1850 | 0.2972 | 0.2603 | 0.1873 |
| fund_profit_growth | bear | 0.0393 | 0.1597 | 0.2463 | 0.1233 | 0.1383 |
| bb_width_20 | bear | 0.0461 | 0.2287 | 0.2013 | 0.1781 | 0.1186 |

### 铁路基建

- **Neutral**: ['mom_x_lowvol_20_20', 'momentum_reversal', 'trend_lowvol'] (单因子IC=0.0746, 组合IC=0.0817)
  - weights: [0.3811, 0.3183, 0.3006]
- **Bull**: ['low_downside', 'trend_lowvol'] (单因子IC=0.0913, 组合IC=0.1068)
  - bull_weights: [0.5448, 0.4552]
- **Bear**: ['momentum_reversal', 'mom_x_lowvol_20_20'] (单因子IC=0.1153, 组合IC=0.1202)
  - bear_weights: [0.5132, 0.4868]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| mom_x_lowvol_20_20 | neutral | 0.0779 | 0.1956 | 0.3984 | 0.2871 | 0.2564 |
| momentum_reversal | neutral | 0.0716 | 0.2052 | 0.3488 | 0.2276 | 0.2141 |
| trend_lowvol | neutral | 0.0743 | 0.2248 | 0.3305 | 0.2234 | 0.2022 |
| fund_profit_growth | neutral | 0.0452 | 0.1397 | 0.3237 | 0.2380 | 0.2004 |
| turnover_stability | neutral | 0.0395 | 0.1210 | 0.3263 | 0.1921 | 0.1945 |
| fund_score | neutral | 0.0433 | 0.1505 | 0.2877 | 0.2025 | 0.1730 |
| rsi_vol_combo | neutral | 0.0502 | 0.1917 | 0.2620 | 0.1879 | 0.1556 |
| fund_pb | neutral | 0.0553 | 0.2180 | 0.2536 | 0.1190 | 0.1419 |
| volatility | neutral | 0.0489 | 0.2192 | 0.2232 | 0.1858 | 0.1323 |
| low_downside | neutral | 0.0320 | 0.2143 | 0.1495 | 0.1441 | 0.0855 |
| fund_revenue_growth | neutral | 0.0199 | 0.1323 | 0.1503 | 0.1211 | 0.0843 |
| low_downside | bull | 0.0959 | 0.1942 | 0.4941 | 0.3409 | 0.3313 |
| trend_lowvol | bull | 0.0866 | 0.2039 | 0.4249 | 0.3030 | 0.2768 |
| volatility | bull | 0.0824 | 0.2110 | 0.3903 | 0.2348 | 0.2410 |
| fund_pb | bull | 0.0738 | 0.2028 | 0.3638 | 0.3106 | 0.2384 |
| mom_x_lowvol_20_20 | bull | 0.0513 | 0.1838 | 0.2793 | 0.2121 | 0.1693 |
| turnover_stability | bull | 0.0362 | 0.1400 | 0.2584 | 0.2121 | 0.1566 |
| stroke_phase | bull | 0.0291 | 0.1266 | 0.2301 | 0.3030 | 0.1499 |
| momentum_reversal | bull | 0.0468 | 0.1926 | 0.2428 | 0.1742 | 0.1425 |
| fund_pe | bull | 0.0521 | 0.2038 | 0.2558 | 0.1061 | 0.1415 |
| fund_profit_growth | bull | 0.0207 | 0.1172 | 0.1764 | 0.1970 | 0.1056 |
| momentum_reversal | bear | 0.1167 | 0.2390 | 0.4885 | 0.3425 | 0.3279 |
| mom_x_lowvol_20_20 | bear | 0.1139 | 0.2608 | 0.4367 | 0.4247 | 0.3111 |
| fund_score | bear | 0.0541 | 0.1450 | 0.3731 | 0.2877 | 0.2402 |
| fund_revenue_growth | bear | 0.0448 | 0.1229 | 0.3646 | 0.2329 | 0.2247 |
| fund_gross_margin | bear | 0.0666 | 0.1957 | 0.3404 | 0.2877 | 0.2192 |
| trend_lowvol | bear | 0.0857 | 0.2512 | 0.3411 | 0.2329 | 0.2103 |
| fund_profit_growth | bear | 0.0329 | 0.1020 | 0.3223 | 0.2329 | 0.1987 |
| rsi_vol_combo | bear | 0.0472 | 0.1656 | 0.2852 | 0.2055 | 0.1719 |
| fund_roe | bear | 0.0366 | 0.1810 | 0.2024 | 0.1507 | 0.1164 |

### 铜缆高速连接

- **Neutral**: ['fund_pe', 'momentum_reversal'] (单因子IC=0.0822, 组合IC=0.0963)
  - weights: [0.502, 0.498]
- **Bull**: ['trend_lowvol', 'volatility', 'fund_pb'] (单因子IC=0.0965, 组合IC=0.1159)
  - bull_weights: [0.3797, 0.3399, 0.2804]
- **Bear**: ['mom_x_lowvol_20_20', 'momentum_reversal', 'fund_gross_margin'] (单因子IC=0.1345, 组合IC=0.1827)
  - bear_weights: [0.4473, 0.2929, 0.2598]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| fund_pe | neutral | 0.0797 | 0.2355 | 0.3386 | 0.2401 | 0.2099 |
| momentum_reversal | neutral | 0.0847 | 0.2618 | 0.3236 | 0.2871 | 0.2082 |
| mom_x_lowvol_20_20 | neutral | 0.0800 | 0.2487 | 0.3218 | 0.2902 | 0.2076 |
| fund_pb | neutral | 0.0764 | 0.2564 | 0.2980 | 0.2276 | 0.1829 |
| trend_lowvol | neutral | 0.0792 | 0.2686 | 0.2950 | 0.2046 | 0.1777 |
| rsi_vol_combo | neutral | 0.0589 | 0.2425 | 0.2428 | 0.2025 | 0.1460 |
| fund_profit_growth | neutral | 0.0473 | 0.2002 | 0.2361 | 0.2307 | 0.1453 |
| volatility | neutral | 0.0580 | 0.2792 | 0.2076 | 0.1514 | 0.1195 |
| fund_score | neutral | 0.0390 | 0.2052 | 0.1900 | 0.1524 | 0.1095 |
| fund_revenue_growth | neutral | 0.0340 | 0.2013 | 0.1690 | 0.1200 | 0.0946 |
| trend_lowvol | bull | 0.1085 | 0.2696 | 0.4023 | 0.3106 | 0.2636 |
| volatility | bull | 0.0986 | 0.2650 | 0.3720 | 0.2689 | 0.2360 |
| fund_pb | bull | 0.0826 | 0.2649 | 0.3116 | 0.2500 | 0.1947 |
| mom_x_lowvol_20_20 | bull | 0.0743 | 0.2322 | 0.3200 | 0.2121 | 0.1939 |
| low_downside | bull | 0.0511 | 0.2448 | 0.2086 | 0.1515 | 0.1201 |
| fund_revenue_growth | bull | 0.0348 | 0.1920 | 0.1815 | 0.1894 | 0.1079 |
| fund_score | bull | 0.0340 | 0.2122 | 0.1602 | 0.1629 | 0.0931 |
| fund_pe | bull | 0.0329 | 0.2248 | 0.1462 | 0.2045 | 0.0880 |
| fund_gross_margin | bull | 0.0358 | 0.2356 | 0.1520 | 0.1364 | 0.0864 |
| mom_x_lowvol_20_20 | bear | 0.1736 | 0.2196 | 0.7906 | 0.6712 | 0.6606 |
| momentum_reversal | bear | 0.1201 | 0.2016 | 0.5957 | 0.4521 | 0.4325 |
| fund_gross_margin | bear | 0.1098 | 0.2118 | 0.5187 | 0.4795 | 0.3837 |
| trend_lowvol | bear | 0.1361 | 0.2986 | 0.4558 | 0.2603 | 0.2872 |
| rsi_vol_combo | bear | 0.0867 | 0.2012 | 0.4308 | 0.2603 | 0.2715 |
| fund_profit_growth | bear | 0.0770 | 0.1868 | 0.4122 | 0.1781 | 0.2428 |
| fund_pe | bear | 0.0478 | 0.2299 | 0.2078 | 0.1233 | 0.1167 |

### 锂电池概念

- **Neutral**: ['fund_pb', 'mom_x_lowvol_20_20'] (单因子IC=0.0615, 组合IC=0.0851)
  - weights: [0.574, 0.426]
- **Bull**: ['low_downside', 'volatility', 'fund_pb'] (单因子IC=0.0933, 组合IC=0.1148)
  - bull_weights: [0.3722, 0.3184, 0.3095]
- **Bear**: ['rsi_vol_combo', 'momentum_reversal', 'mom_x_lowvol_20_20'] (单因子IC=0.1082, 组合IC=0.1211)
  - bear_weights: [0.3666, 0.3437, 0.2898]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| fund_pb | neutral | 0.0704 | 0.1583 | 0.4445 | 0.3549 | 0.3011 |
| mom_x_lowvol_20_20 | neutral | 0.0526 | 0.1480 | 0.3556 | 0.2568 | 0.2235 |
| volatility | neutral | 0.0616 | 0.1741 | 0.3541 | 0.2526 | 0.2218 |
| momentum_reversal | neutral | 0.0551 | 0.1548 | 0.3555 | 0.2443 | 0.2212 |
| fund_pe | neutral | 0.0503 | 0.1448 | 0.3475 | 0.2359 | 0.2147 |
| fund_profit_growth | neutral | 0.0422 | 0.1274 | 0.3315 | 0.2359 | 0.2048 |
| trend_lowvol | neutral | 0.0567 | 0.1882 | 0.3010 | 0.2296 | 0.1851 |
| rsi_vol_combo | neutral | 0.0404 | 0.1445 | 0.2797 | 0.2046 | 0.1685 |
| turnover_stability | neutral | 0.0209 | 0.1089 | 0.1920 | 0.2109 | 0.1162 |
| fund_score | neutral | 0.0330 | 0.1597 | 0.2067 | 0.1148 | 0.1152 |
| low_downside | neutral | 0.0307 | 0.1814 | 0.1691 | 0.1169 | 0.0944 |
| fund_revenue_growth | neutral | 0.0162 | 0.1129 | 0.1432 | 0.1482 | 0.0822 |
| low_downside | bull | 0.0957 | 0.1370 | 0.6985 | 0.5455 | 0.5397 |
| volatility | bull | 0.0949 | 0.1519 | 0.6251 | 0.4773 | 0.4617 |
| fund_pb | bull | 0.0894 | 0.1410 | 0.6337 | 0.4167 | 0.4488 |
| turnover_stability | bull | 0.0330 | 0.0962 | 0.3431 | 0.3561 | 0.2326 |
| trend_lowvol | bull | 0.0560 | 0.1563 | 0.3581 | 0.1894 | 0.2129 |
| fund_pe | bull | 0.0505 | 0.1494 | 0.3379 | 0.1894 | 0.2010 |
| fund_profit_growth | bull | 0.0284 | 0.1201 | 0.2368 | 0.2197 | 0.1444 |
| fund_score | bull | 0.0244 | 0.1550 | 0.1577 | 0.1364 | 0.0896 |
| rsi_vol_combo | bear | 0.0909 | 0.1289 | 0.7048 | 0.5616 | 0.5503 |
| momentum_reversal | bear | 0.1229 | 0.1763 | 0.6974 | 0.4795 | 0.5159 |
| mom_x_lowvol_20_20 | bear | 0.1107 | 0.1812 | 0.6106 | 0.4247 | 0.4350 |
| bb_width_20 | bear | 0.0731 | 0.1895 | 0.3856 | 0.2329 | 0.2377 |
| fund_pb | bear | 0.0462 | 0.1941 | 0.2381 | 0.1233 | 0.1337 |
| vol_confirm | bear | 0.0209 | 0.1170 | 0.1791 | 0.1233 | 0.1006 |

### 锂矿概念

- **Neutral**: ['fund_pe', 'fund_pb', 'fund_gross_margin'] (单因子IC=0.0381, 组合IC=0.0546)
  - weights: [0.4403, 0.3188, 0.2409]
- **Bull**: ['low_downside', 'volatility'] (单因子IC=0.1333, 组合IC=0.1512)
  - bull_weights: [0.556, 0.444]
- **Bear**: ['rsi_vol_combo', 'momentum_reversal', 'trend_lowvol'] (单因子IC=0.1064, 组合IC=0.12)
  - bear_weights: [0.3667, 0.3261, 0.3072]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| fund_pe | neutral | 0.0553 | 0.2553 | 0.2167 | 0.1628 | 0.1260 |
| fund_pb | neutral | 0.0343 | 0.2128 | 0.1609 | 0.1336 | 0.0912 |
| fund_gross_margin | neutral | 0.0248 | 0.2022 | 0.1225 | 0.1253 | 0.0689 |
| low_downside | bull | 0.1452 | 0.1955 | 0.7430 | 0.5227 | 0.5657 |
| volatility | bull | 0.1213 | 0.1923 | 0.6309 | 0.4318 | 0.4517 |
| fund_pe | bull | 0.0878 | 0.2191 | 0.4009 | 0.3295 | 0.2665 |
| fund_pb | bull | 0.0674 | 0.2057 | 0.3275 | 0.1288 | 0.1849 |
| mom_x_lowvol_20_20 | bull | 0.0659 | 0.2500 | 0.2635 | 0.1439 | 0.1507 |
| turnover_stability | bull | 0.0486 | 0.2070 | 0.2347 | 0.1818 | 0.1387 |
| fund_gross_margin | bull | 0.0302 | 0.1848 | 0.1632 | 0.1818 | 0.0964 |
| fund_roe | bull | 0.0355 | 0.2659 | 0.1334 | 0.1212 | 0.0748 |
| fund_score | bull | 0.0347 | 0.2764 | 0.1256 | 0.1553 | 0.0726 |
| fund_profit_growth | bull | 0.0234 | 0.2190 | 0.1070 | 0.1136 | 0.0596 |
| rsi_vol_combo | bear | 0.1115 | 0.2333 | 0.4781 | 0.3973 | 0.3340 |
| momentum_reversal | bear | 0.1122 | 0.2485 | 0.4517 | 0.3151 | 0.2970 |
| trend_lowvol | bear | 0.0953 | 0.2287 | 0.4168 | 0.3425 | 0.2798 |
| mom_x_lowvol_20_20 | bear | 0.0647 | 0.2216 | 0.2922 | 0.2329 | 0.1801 |
| vol_confirm | bear | 0.0507 | 0.2487 | 0.2039 | 0.1781 | 0.1201 |

### 长期破净

- **Neutral**: ['fund_pb', 'fund_pe'] (单因子IC=0.0964, 组合IC=0.1124)
  - weights: [0.5218, 0.4782]
- **Bull**: ['low_downside'] (单因子IC=0.085, 组合IC=0.085)
  - bull_weights: [1.0]
- **Bear**: ['momentum_reversal', 'bb_width_20', 'trend_lowvol'] (单因子IC=0.1336, 组合IC=0.1642)
  - bear_weights: [0.4061, 0.3147, 0.2792]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| fund_pb | neutral | 0.0846 | 0.1952 | 0.4335 | 0.3152 | 0.2851 |
| fund_pe | neutral | 0.1081 | 0.2717 | 0.3980 | 0.3132 | 0.2613 |
| turnover_stability | neutral | 0.0549 | 0.1390 | 0.3948 | 0.3173 | 0.2601 |
| fund_profit_growth | neutral | 0.0549 | 0.1562 | 0.3516 | 0.2568 | 0.2209 |
| fund_score | neutral | 0.0778 | 0.2267 | 0.3432 | 0.2401 | 0.2128 |
| fund_roe | neutral | 0.0785 | 0.2635 | 0.2980 | 0.2213 | 0.1820 |
| fund_revenue_growth | neutral | 0.0422 | 0.1441 | 0.2928 | 0.1900 | 0.1742 |
| low_downside | neutral | 0.0630 | 0.2252 | 0.2796 | 0.2401 | 0.1733 |
| volatility | neutral | 0.0420 | 0.1641 | 0.2558 | 0.2484 | 0.1597 |
| fund_gross_margin | neutral | 0.0355 | 0.1724 | 0.2060 | 0.1566 | 0.1191 |
| mom_x_lowvol_20_20 | neutral | 0.0334 | 0.1628 | 0.2053 | 0.1273 | 0.1157 |
| trend_lowvol | neutral | 0.0441 | 0.2515 | 0.1752 | 0.1795 | 0.1033 |
| momentum_reversal | neutral | 0.0272 | 0.2112 | 0.1289 | 0.1065 | 0.0713 |
| low_downside | bull | 0.0850 | 0.2043 | 0.4158 | 0.3712 | 0.2851 |
| fund_pb | bull | 0.0658 | 0.1833 | 0.3589 | 0.2803 | 0.2297 |
| volatility | bull | 0.0445 | 0.1496 | 0.2976 | 0.2500 | 0.1860 |
| trend_lowvol | bull | 0.0468 | 0.2420 | 0.1935 | 0.1591 | 0.1121 |
| fund_pe | bull | 0.0374 | 0.3154 | 0.1187 | 0.1136 | 0.0661 |
| momentum_reversal | bear | 0.1587 | 0.2550 | 0.6223 | 0.5342 | 0.4774 |
| bb_width_20 | bear | 0.1145 | 0.2290 | 0.5001 | 0.4795 | 0.3699 |
| trend_lowvol | bear | 0.1276 | 0.2609 | 0.4889 | 0.3425 | 0.3282 |
| mom_x_lowvol_20_20 | bear | 0.0957 | 0.2087 | 0.4588 | 0.3973 | 0.3205 |
| rsi_vol_combo | bear | 0.0899 | 0.2179 | 0.4123 | 0.3151 | 0.2711 |

### 阿兹海默

- **Neutral**: ['fund_pb', 'trend_lowvol'] (单因子IC=0.0929, 组合IC=0.1254)
  - weights: [0.5063, 0.4937]
- **Bull**: ['volatility', 'low_downside', 'fund_pb'] (单因子IC=0.1166, 组合IC=0.1487)
  - bull_weights: [0.4101, 0.3991, 0.1908]
- **Bear**: ['trend_lowvol', 'fund_gross_margin', 'rsi_vol_combo'] (单因子IC=0.0812, 组合IC=0.1464)
  - bear_weights: [0.4042, 0.3607, 0.2352]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| fund_pb | neutral | 0.0940 | 0.2637 | 0.3564 | 0.2547 | 0.2236 |
| trend_lowvol | neutral | 0.0919 | 0.2694 | 0.3411 | 0.2787 | 0.2181 |
| volatility | neutral | 0.0869 | 0.2575 | 0.3373 | 0.2808 | 0.2160 |
| fund_pe | neutral | 0.0876 | 0.2821 | 0.3105 | 0.1827 | 0.1836 |
| turnover_stability | neutral | 0.0589 | 0.2369 | 0.2484 | 0.2046 | 0.1496 |
| low_downside | neutral | 0.0738 | 0.3014 | 0.2448 | 0.1983 | 0.1467 |
| fund_profit_growth | neutral | 0.0595 | 0.2625 | 0.2266 | 0.2025 | 0.1363 |
| mom_x_lowvol_20_20 | neutral | 0.0571 | 0.2711 | 0.2105 | 0.1712 | 0.1232 |
| momentum_reversal | neutral | 0.0577 | 0.2887 | 0.1998 | 0.1399 | 0.1139 |
| volatility | bull | 0.1371 | 0.2502 | 0.5480 | 0.4545 | 0.3986 |
| low_downside | bull | 0.1355 | 0.2521 | 0.5375 | 0.4432 | 0.3879 |
| fund_pb | bull | 0.0773 | 0.2526 | 0.3059 | 0.2121 | 0.1854 |
| stroke_phase | bull | 0.0519 | 0.2384 | 0.2178 | 0.1970 | 0.1303 |
| fund_pe | bull | 0.0490 | 0.2317 | 0.2117 | 0.2197 | 0.1291 |
| turnover_stability | bull | 0.0451 | 0.2454 | 0.1840 | 0.1439 | 0.1052 |
| trend_lowvol | bull | 0.0471 | 0.2936 | 0.1603 | 0.1439 | 0.0917 |
| relative_strength | bull | 0.0363 | 0.2417 | 0.1502 | 0.1250 | 0.0845 |
| trend_lowvol | bear | 0.0999 | 0.3201 | 0.3120 | 0.1507 | 0.1795 |
| fund_gross_margin | bear | 0.0797 | 0.2862 | 0.2784 | 0.1507 | 0.1602 |
| rsi_vol_combo | bear | 0.0639 | 0.3523 | 0.1815 | 0.1507 | 0.1044 |

### 阿里概念

- **Neutral**: ['trend_lowvol', 'momentum_reversal'] (单因子IC=0.0859, 组合IC=0.095)
  - weights: [0.5675, 0.4325]
- **Bull**: ['trend_lowvol', 'low_downside'] (单因子IC=0.0773, 组合IC=0.0938)
  - bull_weights: [0.5132, 0.4868]
- **Bear**: ['trend_lowvol', 'mom_x_lowvol_20_20', 'momentum_reversal'] (单因子IC=0.1396, 组合IC=0.1622)
  - bear_weights: [0.3733, 0.3144, 0.3123]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| trend_lowvol | neutral | 0.0936 | 0.1862 | 0.5027 | 0.4092 | 0.3542 |
| momentum_reversal | neutral | 0.0782 | 0.1830 | 0.4274 | 0.2630 | 0.2699 |
| mom_x_lowvol_20_20 | neutral | 0.0718 | 0.1847 | 0.3890 | 0.2683 | 0.2467 |
| fund_pb | neutral | 0.0626 | 0.1684 | 0.3715 | 0.2610 | 0.2342 |
| rsi_vol_combo | neutral | 0.0587 | 0.1661 | 0.3531 | 0.2630 | 0.2230 |
| volatility | neutral | 0.0678 | 0.2177 | 0.3115 | 0.2276 | 0.1912 |
| fund_profit_growth | neutral | 0.0318 | 0.1250 | 0.2548 | 0.1900 | 0.1516 |
| turnover_stability | neutral | 0.0253 | 0.1022 | 0.2478 | 0.1816 | 0.1464 |
| fund_score | neutral | 0.0342 | 0.1532 | 0.2232 | 0.1503 | 0.1284 |
| low_downside | neutral | 0.0422 | 0.2097 | 0.2014 | 0.1900 | 0.1198 |
| fund_pe | neutral | 0.0406 | 0.2118 | 0.1916 | 0.1086 | 0.1062 |
| fund_revenue_growth | neutral | 0.0177 | 0.1171 | 0.1512 | 0.1002 | 0.0832 |
| trend_lowvol | bull | 0.0825 | 0.1789 | 0.4609 | 0.3409 | 0.3090 |
| low_downside | bull | 0.0722 | 0.1652 | 0.4371 | 0.3409 | 0.2931 |
| volatility | bull | 0.0821 | 0.1823 | 0.4503 | 0.2500 | 0.2814 |
| fund_pb | bull | 0.0682 | 0.1654 | 0.4122 | 0.2955 | 0.2670 |
| turnover_stability | bull | 0.0330 | 0.1034 | 0.3188 | 0.1894 | 0.1896 |
| fund_score | bull | 0.0403 | 0.1357 | 0.2966 | 0.2500 | 0.1854 |
| momentum_reversal | bull | 0.0476 | 0.1582 | 0.3007 | 0.2197 | 0.1834 |
| fund_roe | bull | 0.0412 | 0.1434 | 0.2875 | 0.2500 | 0.1797 |
| mom_x_lowvol_20_20 | bull | 0.0448 | 0.1518 | 0.2953 | 0.1894 | 0.1756 |
| fund_pe | bull | 0.0483 | 0.1957 | 0.2467 | 0.1970 | 0.1476 |
| fund_profit_growth | bull | 0.0249 | 0.1109 | 0.2248 | 0.2348 | 0.1388 |
| trend_lowvol | bear | 0.1255 | 0.1589 | 0.7895 | 0.6164 | 0.6381 |
| mom_x_lowvol_20_20 | bear | 0.1484 | 0.2119 | 0.7004 | 0.5342 | 0.5373 |
| momentum_reversal | bear | 0.1449 | 0.1971 | 0.7353 | 0.4521 | 0.5338 |
| fund_profit_growth | bear | 0.0713 | 0.1312 | 0.5431 | 0.5342 | 0.4166 |
| bb_width_20 | bear | 0.0792 | 0.1934 | 0.4098 | 0.3973 | 0.2863 |
| rsi_vol_combo | bear | 0.0672 | 0.1643 | 0.4089 | 0.2877 | 0.2632 |
| fund_score | bear | 0.0474 | 0.1632 | 0.2907 | 0.2877 | 0.1872 |

### 降解塑料

- **Neutral**: ['momentum_reversal', 'trend_lowvol', 'fund_pb'] (单因子IC=0.0607, 组合IC=0.0973)
  - weights: [0.3532, 0.3266, 0.3203]
- **Bull**: ['fund_pb', 'volatility', 'low_downside'] (单因子IC=0.0849, 组合IC=0.101)
  - bull_weights: [0.3751, 0.3452, 0.2797]
- **Bear**: ['turnover_stability', 'fund_score', 'momentum_reversal'] (单因子IC=0.0968, 组合IC=0.1219)
  - bear_weights: [0.4095, 0.3281, 0.2625]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| momentum_reversal | neutral | 0.0601 | 0.1960 | 0.3065 | 0.2171 | 0.1865 |
| trend_lowvol | neutral | 0.0632 | 0.2226 | 0.2839 | 0.2150 | 0.1725 |
| fund_pb | neutral | 0.0589 | 0.2135 | 0.2756 | 0.2276 | 0.1692 |
| mom_x_lowvol_20_20 | neutral | 0.0539 | 0.1960 | 0.2752 | 0.2035 | 0.1656 |
| fund_profit_growth | neutral | 0.0475 | 0.1681 | 0.2823 | 0.1733 | 0.1656 |
| rsi_vol_combo | neutral | 0.0431 | 0.1872 | 0.2303 | 0.1701 | 0.1347 |
| volatility | neutral | 0.0501 | 0.2219 | 0.2256 | 0.1816 | 0.1333 |
| turnover_stability | neutral | 0.0228 | 0.1556 | 0.1463 | 0.1503 | 0.0841 |
| low_downside | neutral | 0.0226 | 0.2034 | 0.1111 | 0.1106 | 0.0617 |
| fund_pb | bull | 0.0980 | 0.1828 | 0.5364 | 0.4015 | 0.3759 |
| volatility | bull | 0.0848 | 0.1755 | 0.4831 | 0.4318 | 0.3459 |
| low_downside | bull | 0.0719 | 0.1788 | 0.4020 | 0.3939 | 0.2802 |
| trend_lowvol | bull | 0.0751 | 0.1921 | 0.3908 | 0.2045 | 0.2354 |
| fund_pe | bull | 0.0557 | 0.2051 | 0.2717 | 0.1894 | 0.1616 |
| momentum_reversal | bull | 0.0438 | 0.2025 | 0.2164 | 0.2197 | 0.1320 |
| turnover_stability | bull | 0.0357 | 0.1535 | 0.2323 | 0.1288 | 0.1311 |
| rsi_vol_combo | bull | 0.0421 | 0.1938 | 0.2173 | 0.1894 | 0.1292 |
| mom_x_lowvol_20_20 | bull | 0.0245 | 0.1942 | 0.1263 | 0.1818 | 0.0746 |
| fund_revenue_growth | bull | 0.0196 | 0.1550 | 0.1266 | 0.1439 | 0.0724 |
| fund_profit_growth | bull | 0.0183 | 0.1569 | 0.1164 | 0.1288 | 0.0657 |
| turnover_stability | bear | 0.0856 | 0.1221 | 0.7010 | 0.5342 | 0.5378 |
| fund_score | bear | 0.0836 | 0.1408 | 0.5935 | 0.4521 | 0.4309 |
| momentum_reversal | bear | 0.1214 | 0.2364 | 0.5135 | 0.3425 | 0.3447 |
| trend_lowvol | bear | 0.1183 | 0.2538 | 0.4662 | 0.4247 | 0.3321 |
| rsi_vol_combo | bear | 0.0834 | 0.1745 | 0.4782 | 0.3151 | 0.3144 |
| fund_gross_margin | bear | 0.0425 | 0.1194 | 0.3562 | 0.3973 | 0.2488 |
| fund_roe | bear | 0.0578 | 0.1494 | 0.3868 | 0.2603 | 0.2437 |
| mom_x_lowvol_20_20 | bear | 0.0976 | 0.2646 | 0.3686 | 0.3151 | 0.2424 |
| fund_profit_growth | bear | 0.0545 | 0.1452 | 0.3757 | 0.2877 | 0.2419 |
| vol_confirm | bear | 0.0389 | 0.1958 | 0.1984 | 0.1507 | 0.1142 |

### 雄安新区

- **Neutral**: ['mom_x_lowvol_20_20', 'momentum_reversal', 'trend_lowvol'] (单因子IC=0.0912, 组合IC=0.0991)
  - weights: [0.3599, 0.3223, 0.3178]
- **Bull**: ['low_downside', 'trend_lowvol'] (单因子IC=0.1027, 组合IC=0.1152)
  - bull_weights: [0.5445, 0.4555]
- **Bear**: ['trend_lowvol', 'mom_x_lowvol_20_20'] (单因子IC=0.1413, 组合IC=0.1585)
  - bear_weights: [0.5348, 0.4652]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| mom_x_lowvol_20_20 | neutral | 0.0930 | 0.1867 | 0.4981 | 0.3674 | 0.3405 |
| momentum_reversal | neutral | 0.0856 | 0.1882 | 0.4550 | 0.3403 | 0.3049 |
| trend_lowvol | neutral | 0.0950 | 0.2076 | 0.4579 | 0.3132 | 0.3007 |
| fund_profit_growth | neutral | 0.0475 | 0.1122 | 0.4231 | 0.3299 | 0.2813 |
| turnover_stability | neutral | 0.0451 | 0.1123 | 0.4018 | 0.3549 | 0.2722 |
| rsi_vol_combo | neutral | 0.0529 | 0.1779 | 0.2973 | 0.2109 | 0.1800 |
| fund_score | neutral | 0.0445 | 0.1569 | 0.2833 | 0.2276 | 0.1739 |
| volatility | neutral | 0.0564 | 0.2233 | 0.2524 | 0.2526 | 0.1581 |
| fund_pe | neutral | 0.0460 | 0.2000 | 0.2299 | 0.1399 | 0.1310 |
| fund_revenue_growth | neutral | 0.0262 | 0.1243 | 0.2108 | 0.1420 | 0.1204 |
| low_downside | neutral | 0.0385 | 0.2060 | 0.1869 | 0.2088 | 0.1130 |
| fund_pb | neutral | 0.0411 | 0.2194 | 0.1873 | 0.1294 | 0.1057 |
| low_downside | bull | 0.1012 | 0.1507 | 0.6720 | 0.5076 | 0.5066 |
| trend_lowvol | bull | 0.1042 | 0.1797 | 0.5796 | 0.4621 | 0.4237 |
| volatility | bull | 0.0847 | 0.1833 | 0.4618 | 0.3561 | 0.3131 |
| momentum_reversal | bull | 0.0761 | 0.1797 | 0.4236 | 0.2045 | 0.2551 |
| fund_pb | bull | 0.0667 | 0.1944 | 0.3433 | 0.2955 | 0.2223 |
| turnover_stability | bull | 0.0385 | 0.1150 | 0.3350 | 0.2803 | 0.2144 |
| mom_x_lowvol_20_20 | bull | 0.0538 | 0.1645 | 0.3270 | 0.2500 | 0.2044 |
| fund_profit_growth | bull | 0.0297 | 0.1216 | 0.2441 | 0.2576 | 0.1535 |
| rsi_vol_combo | bull | 0.0434 | 0.1621 | 0.2675 | 0.1439 | 0.1530 |
| stroke_phase | bull | 0.0208 | 0.1162 | 0.1792 | 0.1818 | 0.1059 |
| fund_score | bull | 0.0236 | 0.1503 | 0.1571 | 0.1061 | 0.0869 |
| trend_lowvol | bear | 0.1361 | 0.1925 | 0.7068 | 0.4795 | 0.5229 |
| mom_x_lowvol_20_20 | bear | 0.1465 | 0.2294 | 0.6386 | 0.4247 | 0.4549 |
| momentum_reversal | bear | 0.1229 | 0.2172 | 0.5657 | 0.3699 | 0.3875 |
| fund_profit_growth | bear | 0.0547 | 0.1259 | 0.4343 | 0.3699 | 0.2975 |
| fund_revenue_growth | bear | 0.0355 | 0.0962 | 0.3696 | 0.3425 | 0.2481 |
| fund_score | bear | 0.0559 | 0.1505 | 0.3712 | 0.2603 | 0.2339 |
| rsi_vol_combo | bear | 0.0658 | 0.1795 | 0.3667 | 0.2329 | 0.2260 |
| fund_gross_margin | bear | 0.0432 | 0.1524 | 0.2834 | 0.2603 | 0.1786 |
| turnover_stability | bear | 0.0316 | 0.1124 | 0.2809 | 0.2603 | 0.1770 |

### 雅下水电概念

- **Neutral**: ['mom_x_lowvol_20_20', 'fund_profit_growth', 'momentum_reversal'] (单因子IC=0.0532, 组合IC=0.0617)
  - weights: [0.35, 0.3269, 0.323]
- **Bull**: ['fund_pb', 'volatility', 'low_downside'] (单因子IC=0.0927, 组合IC=0.1238)
  - bull_weights: [0.4419, 0.2912, 0.2669]
- **Bear**: ['fund_revenue_growth'] (单因子IC=0.0872, 组合IC=0.0872)
  - bear_weights: [1.0]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| mom_x_lowvol_20_20 | neutral | 0.0569 | 0.1989 | 0.2861 | 0.1795 | 0.1687 |
| fund_profit_growth | neutral | 0.0458 | 0.1777 | 0.2577 | 0.2234 | 0.1576 |
| momentum_reversal | neutral | 0.0571 | 0.2202 | 0.2590 | 0.2025 | 0.1557 |
| trend_lowvol | neutral | 0.0579 | 0.2467 | 0.2345 | 0.1691 | 0.1371 |
| rsi_vol_combo | neutral | 0.0458 | 0.2066 | 0.2217 | 0.1232 | 0.1245 |
| volatility | neutral | 0.0429 | 0.2259 | 0.1901 | 0.1795 | 0.1121 |
| fund_revenue_growth | neutral | 0.0275 | 0.1715 | 0.1602 | 0.1138 | 0.0892 |
| fund_pb | bull | 0.1123 | 0.2123 | 0.5291 | 0.4697 | 0.3888 |
| volatility | bull | 0.0952 | 0.2464 | 0.3865 | 0.3258 | 0.2562 |
| low_downside | bull | 0.0705 | 0.1899 | 0.3713 | 0.2652 | 0.2349 |
| fund_pe | bull | 0.0763 | 0.2297 | 0.3321 | 0.3030 | 0.2164 |
| trend_lowvol | bull | 0.0691 | 0.2196 | 0.3146 | 0.1364 | 0.1787 |
| mom_x_lowvol_20_20 | bull | 0.0438 | 0.2109 | 0.2076 | 0.1212 | 0.1164 |
| fund_profit_growth | bull | 0.0261 | 0.1402 | 0.1861 | 0.1288 | 0.1051 |
| rsi_vol_combo | bull | 0.0337 | 0.1888 | 0.1786 | 0.1364 | 0.1015 |
| fund_roe | bull | 0.0210 | 0.1882 | 0.1118 | 0.1061 | 0.0618 |
| fund_revenue_growth | bear | 0.0872 | 0.1606 | 0.5427 | 0.4795 | 0.4014 |
| fund_gross_margin | bear | 0.0770 | 0.1753 | 0.4392 | 0.3699 | 0.3008 |
| top_fractal_volume | bear | 0.0592 | 0.1553 | 0.3811 | 0.4146 | 0.2696 |
| fund_roe | bear | 0.0684 | 0.1686 | 0.4054 | 0.3151 | 0.2666 |
| fund_profit_growth | bear | 0.0734 | 0.2032 | 0.3609 | 0.3151 | 0.2373 |
| momentum_reversal | bear | 0.1017 | 0.2866 | 0.3547 | 0.1781 | 0.2089 |
| mom_x_lowvol_20_20 | bear | 0.0774 | 0.2510 | 0.3084 | 0.1781 | 0.1817 |
| fund_score | bear | 0.0591 | 0.1828 | 0.3231 | 0.1233 | 0.1815 |
| rsi_vol_combo | bear | 0.0652 | 0.2492 | 0.2614 | 0.1233 | 0.1468 |

### 零售概念

- **Neutral**: ['fund_pb', 'fund_pe', 'trend_lowvol'] (单因子IC=0.0749, 组合IC=0.1123)
  - weights: [0.4263, 0.3562, 0.2175]
- **Bull**: ['low_downside', 'momentum_reversal', 'fund_pb'] (单因子IC=0.0758, 组合IC=0.1347)
  - bull_weights: [0.3589, 0.3498, 0.2914]
- **Bear**: ['fund_pe', 'fund_pb', 'bb_width_20'] (单因子IC=0.0762, 组合IC=0.1222)
  - bear_weights: [0.3827, 0.357, 0.2603]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| fund_pb | neutral | 0.0969 | 0.2123 | 0.4565 | 0.3257 | 0.3026 |
| fund_pe | neutral | 0.0704 | 0.1802 | 0.3906 | 0.2944 | 0.2528 |
| trend_lowvol | neutral | 0.0574 | 0.2273 | 0.2524 | 0.2234 | 0.1544 |
| volatility | neutral | 0.0497 | 0.2095 | 0.2371 | 0.2192 | 0.1445 |
| low_downside | neutral | 0.0488 | 0.2100 | 0.2322 | 0.2255 | 0.1423 |
| fund_profit_growth | neutral | 0.0356 | 0.1591 | 0.2239 | 0.1691 | 0.1309 |
| mom_x_lowvol_20_20 | neutral | 0.0375 | 0.2036 | 0.1843 | 0.1148 | 0.1027 |
| turnover_stability | neutral | 0.0234 | 0.1486 | 0.1574 | 0.1232 | 0.0884 |
| fund_revenue_growth | neutral | 0.0284 | 0.1795 | 0.1582 | 0.1086 | 0.0877 |
| wash_sale_score | neutral | 0.0172 | 0.1579 | 0.1089 | 0.1019 | 0.0600 |
| low_downside | bull | 0.0711 | 0.1623 | 0.4378 | 0.3485 | 0.2952 |
| momentum_reversal | bull | 0.0777 | 0.1853 | 0.4196 | 0.3712 | 0.2877 |
| fund_pb | bull | 0.0785 | 0.2084 | 0.3766 | 0.2727 | 0.2397 |
| rsi_vol_combo | bull | 0.0601 | 0.1802 | 0.3333 | 0.2348 | 0.2058 |
| mom_x_lowvol_20_20 | bull | 0.0512 | 0.1640 | 0.3124 | 0.2727 | 0.1988 |
| fund_pe | bull | 0.0607 | 0.1845 | 0.3290 | 0.1667 | 0.1919 |
| volatility | bull | 0.0501 | 0.1713 | 0.2928 | 0.1288 | 0.1652 |
| fund_revenue_growth | bull | 0.0290 | 0.1545 | 0.1877 | 0.1742 | 0.1102 |
| trend_lowvol | bull | 0.0388 | 0.2161 | 0.1795 | 0.1591 | 0.1040 |
| fund_pe | bear | 0.0705 | 0.1733 | 0.4071 | 0.4795 | 0.3012 |
| fund_pb | bear | 0.0766 | 0.1831 | 0.4185 | 0.3425 | 0.2809 |
| bb_width_20 | bear | 0.0815 | 0.2454 | 0.3323 | 0.2329 | 0.2048 |
| momentum_reversal | bear | 0.0840 | 0.2734 | 0.3074 | 0.1507 | 0.1769 |
| mom_x_lowvol_20_20 | bear | 0.0768 | 0.2560 | 0.3001 | 0.1781 | 0.1767 |
| rsi_vol_combo | bear | 0.0720 | 0.2621 | 0.2746 | 0.1781 | 0.1617 |
| trend_lowvol | bear | 0.0723 | 0.2716 | 0.2662 | 0.1507 | 0.1531 |

### 页岩气

- **Neutral**: ['trend_lowvol'] (单因子IC=0.0902, 组合IC=0.0902)
  - weights: [1.0]
- **Bull**: ['low_downside', 'fund_pe', 'exhaustion_risk'] (单因子IC=0.0621, 组合IC=0.097)
  - bull_weights: [0.3687, 0.3589, 0.2725]
- **Bear**: ['trend_lowvol', 'momentum_reversal', 'rsi_vol_combo'] (单因子IC=0.1423, 组合IC=0.1638)
  - bear_weights: [0.3933, 0.3189, 0.2878]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| trend_lowvol | neutral | 0.0902 | 0.2755 | 0.3273 | 0.2683 | 0.2076 |
| volatility | neutral | 0.0701 | 0.2544 | 0.2755 | 0.2630 | 0.1740 |
| fund_pb | neutral | 0.0560 | 0.2487 | 0.2254 | 0.1649 | 0.1313 |
| fund_score | neutral | 0.0547 | 0.2672 | 0.2047 | 0.1127 | 0.1139 |
| momentum_reversal | neutral | 0.0509 | 0.2647 | 0.1923 | 0.1493 | 0.1105 |
| mom_x_lowvol_20_20 | neutral | 0.0502 | 0.2550 | 0.1967 | 0.1190 | 0.1101 |
| fund_roe | neutral | 0.0555 | 0.2838 | 0.1955 | 0.1180 | 0.1093 |
| fund_gross_margin | neutral | 0.0385 | 0.2166 | 0.1777 | 0.1691 | 0.1039 |
| fund_pe | neutral | 0.0417 | 0.2727 | 0.1530 | 0.1253 | 0.0861 |
| low_downside | neutral | 0.0315 | 0.2505 | 0.1257 | 0.1566 | 0.0727 |
| low_downside | bull | 0.0636 | 0.2257 | 0.2816 | 0.2045 | 0.1696 |
| fund_pe | bull | 0.0778 | 0.2694 | 0.2887 | 0.1439 | 0.1651 |
| exhaustion_risk | bull | 0.0451 | 0.2131 | 0.2117 | 0.1847 | 0.1254 |
| fund_profit_growth | bull | 0.0376 | 0.2083 | 0.1806 | 0.1705 | 0.1057 |
| fund_gross_margin | bull | 0.0378 | 0.2199 | 0.1721 | 0.1136 | 0.0959 |
| turnover_stability | bull | 0.0247 | 0.2234 | 0.1107 | 0.1136 | 0.0616 |
| trend_lowvol | bear | 0.1365 | 0.2217 | 0.6156 | 0.4795 | 0.4554 |
| momentum_reversal | bear | 0.1611 | 0.3108 | 0.5182 | 0.4247 | 0.3691 |
| rsi_vol_combo | bear | 0.1293 | 0.2712 | 0.4770 | 0.3973 | 0.3332 |
| fund_revenue_growth | bear | 0.1026 | 0.2241 | 0.4577 | 0.3151 | 0.3010 |
| fund_profit_growth | bear | 0.1019 | 0.2309 | 0.4410 | 0.3425 | 0.2960 |
| mom_x_lowvol_20_20 | bear | 0.1258 | 0.3037 | 0.4142 | 0.2329 | 0.2553 |
| fund_score | bear | 0.1036 | 0.3323 | 0.3118 | 0.3425 | 0.2093 |
| fund_roe | bear | 0.1008 | 0.3174 | 0.3175 | 0.2329 | 0.1957 |
| fund_gross_margin | bear | 0.0805 | 0.2446 | 0.3292 | 0.1781 | 0.1939 |
| fund_pe | bear | 0.0699 | 0.2954 | 0.2366 | 0.1781 | 0.1394 |

### 预制菜概念

- **Neutral**: ['fund_pb', 'momentum_reversal', 'mom_x_lowvol_20_20'] (单因子IC=0.0969, 组合IC=0.1338)
  - weights: [0.3497, 0.3385, 0.3118]
- **Bull**: ['turnover_stability', 'fund_pb', 'volatility'] (单因子IC=0.0858, 组合IC=0.1165)
  - bull_weights: [0.3798, 0.3314, 0.2888]
- **Bear**: ['fund_pb', 'mom_x_lowvol_20_20', 'momentum_reversal'] (单因子IC=0.1367, 组合IC=0.1675)
  - bear_weights: [0.3687, 0.3274, 0.3039]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| fund_pb | neutral | 0.0865 | 0.1732 | 0.4995 | 0.3643 | 0.3408 |
| momentum_reversal | neutral | 0.1072 | 0.2225 | 0.4818 | 0.3695 | 0.3299 |
| mom_x_lowvol_20_20 | neutral | 0.0969 | 0.2145 | 0.4520 | 0.3445 | 0.3039 |
| trend_lowvol | neutral | 0.0991 | 0.2307 | 0.4297 | 0.3184 | 0.2832 |
| volatility | neutral | 0.0789 | 0.2160 | 0.3651 | 0.2923 | 0.2359 |
| rsi_vol_combo | neutral | 0.0795 | 0.2145 | 0.3705 | 0.2484 | 0.2313 |
| low_downside | neutral | 0.0447 | 0.2011 | 0.2223 | 0.1681 | 0.1298 |
| turnover_stability | neutral | 0.0365 | 0.1722 | 0.2120 | 0.1775 | 0.1248 |
| fund_profit_growth | neutral | 0.0328 | 0.1764 | 0.1857 | 0.1106 | 0.1031 |
| fund_score | neutral | 0.0259 | 0.1926 | 0.1346 | 0.1273 | 0.0758 |
| turnover_stability | bull | 0.0803 | 0.1432 | 0.5603 | 0.4697 | 0.4117 |
| fund_pb | bull | 0.0875 | 0.1697 | 0.5154 | 0.3939 | 0.3592 |
| volatility | bull | 0.0896 | 0.1918 | 0.4669 | 0.3409 | 0.3130 |
| trend_lowvol | bull | 0.0871 | 0.2159 | 0.4035 | 0.3409 | 0.2705 |
| low_downside | bull | 0.0692 | 0.1776 | 0.3897 | 0.3409 | 0.2613 |
| momentum_reversal | bull | 0.0676 | 0.2010 | 0.3361 | 0.2273 | 0.2063 |
| fund_pe | bull | 0.0561 | 0.1755 | 0.3194 | 0.2803 | 0.2045 |
| rsi_vol_combo | bull | 0.0621 | 0.1844 | 0.3368 | 0.2121 | 0.2041 |
| mom_x_lowvol_20_20 | bull | 0.0590 | 0.2031 | 0.2906 | 0.2500 | 0.1816 |
| fund_pb | bear | 0.0981 | 0.1348 | 0.7279 | 0.5890 | 0.5783 |
| mom_x_lowvol_20_20 | bear | 0.1510 | 0.2175 | 0.6943 | 0.4795 | 0.5136 |
| momentum_reversal | bear | 0.1610 | 0.2406 | 0.6694 | 0.4247 | 0.4768 |
| rsi_vol_combo | bear | 0.1488 | 0.2520 | 0.5907 | 0.4521 | 0.4288 |
| fund_pe | bear | 0.1029 | 0.1831 | 0.5618 | 0.3699 | 0.3848 |
| fund_revenue_growth | bear | 0.0530 | 0.1143 | 0.4636 | 0.4247 | 0.3303 |
| trend_lowvol | bear | 0.0863 | 0.2477 | 0.3483 | 0.3151 | 0.2290 |

### 风能

- **Neutral**: ['volatility', 'fund_pb', 'mom_x_lowvol_20_20'] (单因子IC=0.068, 组合IC=0.0889)
  - weights: [0.3416, 0.3364, 0.322]
- **Bull**: ['fund_pb', 'low_downside', 'volatility'] (单因子IC=0.0768, 组合IC=0.0928)
  - bull_weights: [0.3693, 0.326, 0.3047]
- **Bear**: ['mom_x_lowvol_20_20'] (单因子IC=0.1397, 组合IC=0.1397)
  - bear_weights: [1.0]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| volatility | neutral | 0.0736 | 0.2173 | 0.3386 | 0.2672 | 0.2145 |
| fund_pb | neutral | 0.0691 | 0.1991 | 0.3472 | 0.2171 | 0.2113 |
| mom_x_lowvol_20_20 | neutral | 0.0614 | 0.1890 | 0.3245 | 0.2463 | 0.2022 |
| fund_profit_growth | neutral | 0.0445 | 0.1367 | 0.3255 | 0.2401 | 0.2018 |
| momentum_reversal | neutral | 0.0635 | 0.1962 | 0.3236 | 0.2150 | 0.1966 |
| trend_lowvol | neutral | 0.0691 | 0.2222 | 0.3109 | 0.2150 | 0.1889 |
| fund_pe | neutral | 0.0588 | 0.1991 | 0.2952 | 0.2171 | 0.1796 |
| rsi_vol_combo | neutral | 0.0492 | 0.1828 | 0.2690 | 0.1879 | 0.1598 |
| fund_score | neutral | 0.0346 | 0.1550 | 0.2231 | 0.1482 | 0.1281 |
| low_downside | neutral | 0.0420 | 0.2082 | 0.2018 | 0.1378 | 0.1148 |
| turnover_stability | neutral | 0.0187 | 0.1152 | 0.1626 | 0.1065 | 0.0900 |
| fund_pb | bull | 0.0860 | 0.1670 | 0.5150 | 0.3258 | 0.3414 |
| low_downside | bull | 0.0666 | 0.1530 | 0.4349 | 0.3864 | 0.3014 |
| volatility | bull | 0.0778 | 0.1821 | 0.4274 | 0.3182 | 0.2817 |
| fund_pe | bull | 0.0632 | 0.1719 | 0.3680 | 0.2879 | 0.2370 |
| fund_profit_growth | bull | 0.0289 | 0.1245 | 0.2324 | 0.2727 | 0.1479 |
| fund_score | bull | 0.0293 | 0.1509 | 0.1940 | 0.1288 | 0.1095 |
| top_fractal_volume | bull | 0.0099 | 0.0884 | 0.1118 | 0.1085 | 0.0620 |
| turnover_stability | bull | 0.0109 | 0.1104 | 0.0989 | 0.1894 | 0.0588 |
| mom_x_lowvol_20_20 | bear | 0.1397 | 0.2568 | 0.5441 | 0.4521 | 0.3950 |
| momentum_reversal | bear | 0.1325 | 0.2500 | 0.5302 | 0.4795 | 0.3922 |
| rsi_vol_combo | bear | 0.0910 | 0.1713 | 0.5312 | 0.3699 | 0.3639 |
| trend_lowvol | bear | 0.1010 | 0.2394 | 0.4216 | 0.2055 | 0.2541 |
| fund_profit_growth | bear | 0.0413 | 0.1321 | 0.3129 | 0.3151 | 0.2057 |
| fund_score | bear | 0.0267 | 0.1430 | 0.1868 | 0.1507 | 0.1075 |

### 飞行汽车(eVTOL)

- **Neutral**: ['mom_x_lowvol_20_20', 'momentum_reversal', 'rsi_vol_combo'] (单因子IC=0.0878, 组合IC=0.0956)
  - weights: [0.3504, 0.3403, 0.3093]
- **Bull**: ['fund_pb', 'volatility'] (单因子IC=0.0784, 组合IC=0.0878)
  - bull_weights: [0.5183, 0.4817]
- **Bear**: ['vol_confirm'] (单因子IC=0.0741, 组合IC=0.0743)
  - bear_weights: [1.0]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| mom_x_lowvol_20_20 | neutral | 0.0894 | 0.1851 | 0.4832 | 0.3445 | 0.3248 |
| momentum_reversal | neutral | 0.0930 | 0.1947 | 0.4774 | 0.3215 | 0.3155 |
| rsi_vol_combo | neutral | 0.0810 | 0.1869 | 0.4332 | 0.3236 | 0.2867 |
| trend_lowvol | neutral | 0.0812 | 0.2051 | 0.3960 | 0.2923 | 0.2559 |
| fund_pb | neutral | 0.0721 | 0.1814 | 0.3971 | 0.2651 | 0.2512 |
| fund_profit_growth | neutral | 0.0471 | 0.1574 | 0.2996 | 0.2380 | 0.1854 |
| fund_pe | neutral | 0.0523 | 0.1776 | 0.2947 | 0.2046 | 0.1775 |
| volatility | neutral | 0.0596 | 0.2030 | 0.2936 | 0.2088 | 0.1774 |
| fund_score | neutral | 0.0440 | 0.1772 | 0.2482 | 0.2171 | 0.1510 |
| turnover_stability | neutral | 0.0299 | 0.1354 | 0.2207 | 0.1889 | 0.1312 |
| fund_revenue_growth | neutral | 0.0331 | 0.1657 | 0.1997 | 0.1566 | 0.1155 |
| fund_pb | bull | 0.0775 | 0.1804 | 0.4294 | 0.3030 | 0.2798 |
| volatility | bull | 0.0794 | 0.1954 | 0.4061 | 0.2803 | 0.2600 |
| low_downside | bull | 0.0671 | 0.1815 | 0.3696 | 0.2803 | 0.2366 |
| trend_lowvol | bull | 0.0678 | 0.2080 | 0.3257 | 0.2121 | 0.1974 |
| momentum_reversal | bull | 0.0526 | 0.2005 | 0.2621 | 0.2121 | 0.1588 |
| mom_x_lowvol_20_20 | bull | 0.0474 | 0.1918 | 0.2474 | 0.2273 | 0.1518 |
| fund_gross_margin | bull | 0.0287 | 0.1335 | 0.2148 | 0.1439 | 0.1228 |
| fund_pe | bull | 0.0325 | 0.1645 | 0.1976 | 0.1439 | 0.1130 |
| rsi_vol_combo | bull | 0.0306 | 0.1854 | 0.1650 | 0.1591 | 0.0957 |
| vol_confirm | bear | 0.0741 | 0.1152 | 0.6427 | 0.5068 | 0.4842 |
| fund_pe | bear | 0.0549 | 0.1491 | 0.3682 | 0.4247 | 0.2623 |
| rsi_vol_combo | bear | 0.0636 | 0.1980 | 0.3212 | 0.1233 | 0.1804 |
| momentum_reversal | bear | 0.0696 | 0.2425 | 0.2868 | 0.1233 | 0.1611 |
| mom_x_lowvol_20_20 | bear | 0.0612 | 0.2231 | 0.2744 | 0.1507 | 0.1579 |
| fund_pb | bear | 0.0454 | 0.2004 | 0.2267 | 0.1507 | 0.1304 |
| fund_gross_margin | bear | 0.0245 | 0.1377 | 0.1779 | 0.2877 | 0.1145 |

### 食品安全

- **Neutral**: ['fund_pb', 'turnover_stability', 'momentum_reversal'] (单因子IC=0.0678, 组合IC=0.1148)
  - weights: [0.3931, 0.3133, 0.2936]
- **Bull**: ['momentum_reversal', 'volatility', 'rsi_vol_combo'] (单因子IC=0.0725, 组合IC=0.0909)
  - bull_weights: [0.3545, 0.3305, 0.315]
- **Bear**: ['fund_pb', 'trend_lowvol', 'rsi_vol_combo'] (单因子IC=0.1242, 组合IC=0.1556)
  - bear_weights: [0.3718, 0.3382, 0.29]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| fund_pb | neutral | 0.0749 | 0.1788 | 0.4187 | 0.2944 | 0.2710 |
| turnover_stability | neutral | 0.0556 | 0.1612 | 0.3448 | 0.2526 | 0.2160 |
| momentum_reversal | neutral | 0.0729 | 0.2279 | 0.3200 | 0.2651 | 0.2024 |
| trend_lowvol | neutral | 0.0757 | 0.2397 | 0.3158 | 0.2422 | 0.1961 |
| mom_x_lowvol_20_20 | neutral | 0.0686 | 0.2311 | 0.2966 | 0.2171 | 0.1805 |
| fund_pe | neutral | 0.0608 | 0.2233 | 0.2721 | 0.2422 | 0.1690 |
| volatility | neutral | 0.0595 | 0.2296 | 0.2590 | 0.2004 | 0.1554 |
| rsi_vol_combo | neutral | 0.0518 | 0.2229 | 0.2323 | 0.1837 | 0.1375 |
| fund_revenue_growth | neutral | 0.0340 | 0.1764 | 0.1928 | 0.1545 | 0.1113 |
| fund_profit_growth | neutral | 0.0316 | 0.1755 | 0.1802 | 0.1524 | 0.1038 |
| momentum_reversal | bull | 0.0728 | 0.1884 | 0.3866 | 0.3561 | 0.2621 |
| volatility | bull | 0.0791 | 0.2084 | 0.3795 | 0.2879 | 0.2444 |
| rsi_vol_combo | bull | 0.0657 | 0.1784 | 0.3683 | 0.2652 | 0.2330 |
| low_downside | bull | 0.0609 | 0.1757 | 0.3467 | 0.2879 | 0.2233 |
| turnover_stability | bull | 0.0502 | 0.1516 | 0.3312 | 0.3182 | 0.2183 |
| trend_lowvol | bull | 0.0691 | 0.2160 | 0.3201 | 0.2576 | 0.2013 |
| mom_x_lowvol_20_20 | bull | 0.0593 | 0.1953 | 0.3037 | 0.2727 | 0.1932 |
| fund_pb | bull | 0.0543 | 0.1853 | 0.2932 | 0.2121 | 0.1777 |
| fund_pe | bull | 0.0400 | 0.2186 | 0.1831 | 0.1667 | 0.1068 |
| fund_pb | bear | 0.1042 | 0.1613 | 0.6461 | 0.5068 | 0.4868 |
| trend_lowvol | bear | 0.1415 | 0.2233 | 0.6337 | 0.3973 | 0.4427 |
| rsi_vol_combo | bear | 0.1269 | 0.2151 | 0.5898 | 0.2877 | 0.3797 |
| momentum_reversal | bear | 0.1313 | 0.2344 | 0.5599 | 0.2877 | 0.3605 |
| mom_x_lowvol_20_20 | bear | 0.1182 | 0.2327 | 0.5078 | 0.3425 | 0.3409 |
| turnover_stability | bear | 0.0539 | 0.1381 | 0.3899 | 0.2055 | 0.2350 |
| volatility | bear | 0.0827 | 0.2445 | 0.3384 | 0.2329 | 0.2086 |

### 首发经济

- **Neutral**: ['trend_lowvol', 'mom_x_lowvol_20_20'] (单因子IC=0.1126, 组合IC=0.1328)
  - weights: [0.5397, 0.4603]
- **Bull**: ['trend_lowvol', 'low_downside', 'rsi_vol_combo'] (单因子IC=0.0855, 组合IC=0.134)
  - bull_weights: [0.347, 0.3411, 0.312]
- **Bear**: ['momentum_reversal', 'rsi_vol_combo', 'trend_lowvol'] (单因子IC=0.1783, 组合IC=0.2098)
  - bear_weights: [0.3516, 0.3439, 0.3045]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| trend_lowvol | neutral | 0.1213 | 0.2365 | 0.5130 | 0.3695 | 0.3513 |
| mom_x_lowvol_20_20 | neutral | 0.1040 | 0.2319 | 0.4484 | 0.3361 | 0.2996 |
| momentum_reversal | neutral | 0.1107 | 0.2500 | 0.4427 | 0.3194 | 0.2920 |
| volatility | neutral | 0.1043 | 0.2457 | 0.4246 | 0.3361 | 0.2836 |
| rsi_vol_combo | neutral | 0.0793 | 0.2403 | 0.3300 | 0.2505 | 0.2063 |
| fund_pb | neutral | 0.0638 | 0.2123 | 0.3007 | 0.1879 | 0.1786 |
| low_downside | neutral | 0.0643 | 0.2248 | 0.2861 | 0.2443 | 0.1780 |
| fund_pe | neutral | 0.0510 | 0.2173 | 0.2347 | 0.2025 | 0.1411 |
| fund_score | neutral | 0.0363 | 0.1965 | 0.1846 | 0.1086 | 0.1023 |
| fund_profit_growth | neutral | 0.0324 | 0.1875 | 0.1731 | 0.1086 | 0.0959 |
| fund_revenue_growth | neutral | 0.0309 | 0.1804 | 0.1712 | 0.1148 | 0.0954 |
| turnover_stability | neutral | 0.0311 | 0.1835 | 0.1697 | 0.1148 | 0.0946 |
| trend_lowvol | bull | 0.0874 | 0.1980 | 0.4414 | 0.3788 | 0.3043 |
| low_downside | bull | 0.0867 | 0.1986 | 0.4363 | 0.3712 | 0.2991 |
| rsi_vol_combo | bull | 0.0825 | 0.1919 | 0.4299 | 0.2727 | 0.2736 |
| momentum_reversal | bull | 0.0747 | 0.1962 | 0.3810 | 0.3636 | 0.2597 |
| volatility | bull | 0.0770 | 0.2170 | 0.3551 | 0.3030 | 0.2313 |
| turnover_stability | bull | 0.0641 | 0.1790 | 0.3580 | 0.2879 | 0.2306 |
| mom_x_lowvol_20_20 | bull | 0.0668 | 0.2075 | 0.3221 | 0.3106 | 0.2110 |
| fund_gross_margin | bull | 0.0579 | 0.1844 | 0.3139 | 0.2879 | 0.2022 |
| fund_pb | bull | 0.0489 | 0.1809 | 0.2702 | 0.2500 | 0.1689 |
| fund_pe | bull | 0.0410 | 0.2439 | 0.1682 | 0.1136 | 0.0936 |
| momentum_reversal | bear | 0.2037 | 0.2105 | 0.9676 | 0.6986 | 0.8218 |
| rsi_vol_combo | bear | 0.1527 | 0.1665 | 0.9170 | 0.7534 | 0.8040 |
| trend_lowvol | bear | 0.1787 | 0.1994 | 0.8959 | 0.5890 | 0.7118 |
| mom_x_lowvol_20_20 | bear | 0.1588 | 0.2025 | 0.7842 | 0.5890 | 0.6230 |
| fund_pb | bear | 0.1228 | 0.2710 | 0.4533 | 0.2055 | 0.2732 |
| turnover_stability | bear | 0.0782 | 0.2024 | 0.3865 | 0.3151 | 0.2541 |
| fund_roe | bear | 0.0407 | 0.1858 | 0.2188 | 0.2055 | 0.1319 |
| fund_profit_growth | bear | 0.0469 | 0.2047 | 0.2290 | 0.1233 | 0.1286 |

### 高压快充

- **Neutral**: ['fund_pb', 'volatility', 'momentum_reversal'] (单因子IC=0.0917, 组合IC=0.1357)
  - weights: [0.3547, 0.325, 0.3203]
- **Bull**: ['low_downside', 'fund_pe'] (单因子IC=0.0825, 组合IC=0.0972)
  - bull_weights: [0.5491, 0.4509]
- **Bear**: ['rsi_vol_combo', 'fund_profit_growth'] (单因子IC=0.0892, 组合IC=0.1096)
  - bear_weights: [0.5017, 0.4983]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| fund_pb | neutral | 0.0851 | 0.1757 | 0.4844 | 0.3340 | 0.3231 |
| volatility | neutral | 0.0954 | 0.2172 | 0.4391 | 0.3486 | 0.2961 |
| momentum_reversal | neutral | 0.0945 | 0.2113 | 0.4473 | 0.3048 | 0.2918 |
| mom_x_lowvol_20_20 | neutral | 0.0916 | 0.2105 | 0.4353 | 0.3006 | 0.2831 |
| rsi_vol_combo | neutral | 0.0804 | 0.2060 | 0.3902 | 0.3132 | 0.2562 |
| trend_lowvol | neutral | 0.0820 | 0.2274 | 0.3607 | 0.2651 | 0.2282 |
| fund_pe | neutral | 0.0638 | 0.1873 | 0.3409 | 0.2276 | 0.2092 |
| low_downside | neutral | 0.0581 | 0.2140 | 0.2715 | 0.2150 | 0.1649 |
| fund_profit_growth | neutral | 0.0505 | 0.1899 | 0.2659 | 0.1795 | 0.1568 |
| turnover_stability | neutral | 0.0396 | 0.1645 | 0.2409 | 0.1691 | 0.1408 |
| low_downside | bull | 0.0875 | 0.1930 | 0.4536 | 0.3939 | 0.3162 |
| fund_pe | bull | 0.0774 | 0.1898 | 0.4079 | 0.2727 | 0.2596 |
| volatility | bull | 0.0783 | 0.1952 | 0.4012 | 0.2879 | 0.2583 |
| fund_pb | bull | 0.0587 | 0.2049 | 0.2864 | 0.2500 | 0.1790 |
| trend_lowvol | bull | 0.0451 | 0.2043 | 0.2209 | 0.1515 | 0.1272 |
| fund_revenue_growth | bull | 0.0389 | 0.1935 | 0.2012 | 0.1667 | 0.1174 |
| stroke_phase | bull | 0.0286 | 0.1629 | 0.1757 | 0.2121 | 0.1065 |
| rsi_vol_combo | bear | 0.1054 | 0.2114 | 0.4986 | 0.3425 | 0.3347 |
| fund_profit_growth | bear | 0.0731 | 0.1596 | 0.4578 | 0.4521 | 0.3324 |
| momentum_reversal | bear | 0.0976 | 0.2394 | 0.4078 | 0.3425 | 0.2737 |
| mom_x_lowvol_20_20 | bear | 0.1040 | 0.2652 | 0.3921 | 0.3288 | 0.2605 |
| fund_pb | bear | 0.0688 | 0.2158 | 0.3187 | 0.1507 | 0.1833 |
| fund_gross_margin | bear | 0.0264 | 0.1211 | 0.2179 | 0.1781 | 0.1283 |
| trend_lowvol | bear | 0.0521 | 0.2641 | 0.1971 | 0.2055 | 0.1188 |

### 高带宽内存

- **Neutral**: ['momentum_reversal', 'mom_x_lowvol_20_20', 'rsi_vol_combo'] (单因子IC=0.0955, 组合IC=0.1028)
  - weights: [0.3441, 0.3297, 0.3262]
- **Bull**: ['turnover_stability', 'rsi_vol_combo', 'mom_x_lowvol_20_20'] (单因子IC=0.074, 组合IC=0.1101)
  - bull_weights: [0.4687, 0.2937, 0.2376]
- **Bear**: ['mom_x_lowvol_20_20', 'rsi_vol_combo', 'momentum_reversal'] (单因子IC=0.1986, 组合IC=0.204)
  - bear_weights: [0.3482, 0.3312, 0.3206]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| momentum_reversal | neutral | 0.0991 | 0.2971 | 0.3336 | 0.2902 | 0.2152 |
| mom_x_lowvol_20_20 | neutral | 0.0956 | 0.2931 | 0.3260 | 0.2651 | 0.2062 |
| rsi_vol_combo | neutral | 0.0917 | 0.2861 | 0.3204 | 0.2735 | 0.2040 |
| trend_lowvol | neutral | 0.0835 | 0.3039 | 0.2750 | 0.1994 | 0.1649 |
| fund_pe | neutral | 0.0709 | 0.2957 | 0.2396 | 0.1775 | 0.1411 |
| fund_score | neutral | 0.0685 | 0.3056 | 0.2241 | 0.1733 | 0.1315 |
| volatility | neutral | 0.0570 | 0.2717 | 0.2097 | 0.1336 | 0.1189 |
| fund_roe | neutral | 0.0545 | 0.2692 | 0.2024 | 0.1347 | 0.1148 |
| fund_profit_growth | neutral | 0.0517 | 0.2961 | 0.1746 | 0.1785 | 0.1029 |
| turnover_stability | neutral | 0.0206 | 0.2653 | 0.0775 | 0.1044 | 0.0428 |
| wash_sale_score | neutral | 0.0151 | 0.2353 | 0.0642 | 0.1297 | 0.0363 |
| turnover_stability | bull | 0.0928 | 0.2197 | 0.4226 | 0.3106 | 0.2769 |
| rsi_vol_combo | bull | 0.0735 | 0.2469 | 0.2975 | 0.1667 | 0.1735 |
| mom_x_lowvol_20_20 | bull | 0.0556 | 0.2423 | 0.2295 | 0.2235 | 0.1404 |
| momentum_reversal | bull | 0.0565 | 0.2651 | 0.2130 | 0.1591 | 0.1234 |
| volatility | bull | 0.0444 | 0.2142 | 0.2073 | 0.1894 | 0.1233 |
| trend_lowvol | bull | 0.0434 | 0.2538 | 0.1710 | 0.1742 | 0.1004 |
| low_downside | bull | 0.0273 | 0.2337 | 0.1169 | 0.1061 | 0.0646 |
| mom_x_lowvol_20_20 | bear | 0.2029 | 0.2936 | 0.6912 | 0.4795 | 0.5113 |
| rsi_vol_combo | bear | 0.1916 | 0.2834 | 0.6762 | 0.4384 | 0.4863 |
| momentum_reversal | bear | 0.2014 | 0.2901 | 0.6941 | 0.3562 | 0.4707 |
| trend_lowvol | bear | 0.1405 | 0.2849 | 0.4930 | 0.3699 | 0.3377 |
| bb_width_20 | bear | 0.1086 | 0.2886 | 0.3764 | 0.2877 | 0.2424 |
| fund_pb | bear | 0.0961 | 0.2536 | 0.3789 | 0.1781 | 0.2232 |
| vol_opening_strength | bear | 0.1050 | 0.2911 | 0.3607 | 0.1724 | 0.2114 |
| vol_opening_confirm | bear | 0.0990 | 0.2952 | 0.3352 | 0.1724 | 0.1965 |
| fund_pe | bear | 0.0887 | 0.2947 | 0.3010 | 0.2740 | 0.1917 |

### 鸡肉概念

- **Neutral**: ['trend_lowvol'] (单因子IC=0.0546, 组合IC=0.0546)
  - weights: [1.0]
- **Bull**: ['trend_lowvol', 'low_downside', 'momentum_reversal'] (单因子IC=0.0925, 组合IC=0.1196)
  - bull_weights: [0.3928, 0.35, 0.2572]
- **Bear**: ['fund_score', 'fund_roe', 'fund_gross_margin'] (单因子IC=0.1441, 组合IC=0.1664)
  - bear_weights: [0.4065, 0.3038, 0.2897]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| trend_lowvol | neutral | 0.0546 | 0.3595 | 0.1520 | 0.1138 | 0.0847 |
| trend_lowvol | bull | 0.1060 | 0.2771 | 0.3826 | 0.2765 | 0.2442 |
| low_downside | bull | 0.0871 | 0.2562 | 0.3400 | 0.2803 | 0.2176 |
| momentum_reversal | bull | 0.0844 | 0.3238 | 0.2606 | 0.2273 | 0.1599 |
| mom_x_lowvol_20_20 | bull | 0.0587 | 0.3167 | 0.1854 | 0.1629 | 0.1078 |
| turnover_stability | bull | 0.0415 | 0.2315 | 0.1793 | 0.1667 | 0.1046 |
| rsi_vol_combo | bull | 0.0506 | 0.2959 | 0.1712 | 0.1136 | 0.0953 |
| fund_score | bear | 0.1439 | 0.2347 | 0.6130 | 0.6164 | 0.4954 |
| fund_roe | bear | 0.1446 | 0.2513 | 0.5752 | 0.2877 | 0.3703 |
| fund_gross_margin | bear | 0.1439 | 0.2848 | 0.5054 | 0.3973 | 0.3531 |
| fund_pb | bear | 0.1468 | 0.3314 | 0.4429 | 0.1507 | 0.2548 |
| fund_profit_growth | bear | 0.0995 | 0.2818 | 0.3531 | 0.3425 | 0.2370 |
| fund_pe | bear | 0.1535 | 0.4049 | 0.3792 | 0.2174 | 0.2308 |
| mom_x_lowvol_20_20 | bear | 0.1047 | 0.3244 | 0.3229 | 0.4247 | 0.2300 |
| momentum_reversal | bear | 0.1225 | 0.3530 | 0.3471 | 0.2877 | 0.2235 |
| fund_revenue_growth | bear | 0.0964 | 0.2912 | 0.3310 | 0.2329 | 0.2041 |
| rsi_vol_combo | bear | 0.0631 | 0.3327 | 0.1897 | 0.2329 | 0.1169 |
| bb_width_20 | bear | 0.0532 | 0.2827 | 0.1882 | 0.2329 | 0.1160 |

### 鸿蒙概念

- **Neutral**: ['trend_lowvol', 'momentum_reversal', 'fund_pb'] (单因子IC=0.0866, 组合IC=0.1181)
  - weights: [0.3931, 0.3083, 0.2986]
- **Bull**: ['volatility', 'low_downside', 'fund_pb'] (单因子IC=0.1379, 组合IC=0.1678)
  - bull_weights: [0.346, 0.3454, 0.3086]
- **Bear**: ['trend_lowvol', 'mom_x_lowvol_20_20'] (单因子IC=0.1242, 组合IC=0.1501)
  - bear_weights: [0.5557, 0.4443]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| trend_lowvol | neutral | 0.1029 | 0.1864 | 0.5524 | 0.4092 | 0.3892 |
| momentum_reversal | neutral | 0.0839 | 0.1807 | 0.4642 | 0.3152 | 0.3053 |
| fund_pb | neutral | 0.0730 | 0.1629 | 0.4482 | 0.3194 | 0.2957 |
| mom_x_lowvol_20_20 | neutral | 0.0787 | 0.1783 | 0.4417 | 0.3382 | 0.2955 |
| volatility | neutral | 0.0777 | 0.1875 | 0.4146 | 0.3466 | 0.2792 |
| rsi_vol_combo | neutral | 0.0630 | 0.1730 | 0.3644 | 0.2797 | 0.2332 |
| fund_pe | neutral | 0.0539 | 0.1517 | 0.3554 | 0.2797 | 0.2274 |
| low_downside | neutral | 0.0452 | 0.1699 | 0.2659 | 0.2756 | 0.1696 |
| fund_profit_growth | neutral | 0.0316 | 0.1214 | 0.2603 | 0.1764 | 0.1531 |
| turnover_stability | neutral | 0.0227 | 0.1232 | 0.1842 | 0.1190 | 0.1031 |
| volatility | bull | 0.1510 | 0.1783 | 0.8467 | 0.5909 | 0.6735 |
| low_downside | bull | 0.1273 | 0.1520 | 0.8372 | 0.6061 | 0.6723 |
| fund_pb | bull | 0.1355 | 0.1760 | 0.7700 | 0.5606 | 0.6008 |
| fund_pe | bull | 0.0833 | 0.1358 | 0.6136 | 0.4242 | 0.4370 |
| trend_lowvol | bull | 0.1170 | 0.2132 | 0.5490 | 0.4924 | 0.4097 |
| mom_x_lowvol_20_20 | bull | 0.1036 | 0.1947 | 0.5321 | 0.4015 | 0.3728 |
| momentum_reversal | bull | 0.0858 | 0.1914 | 0.4485 | 0.3030 | 0.2922 |
| turnover_stability | bull | 0.0352 | 0.1360 | 0.2591 | 0.1061 | 0.1433 |
| fund_roe | bull | 0.0308 | 0.1440 | 0.2139 | 0.1364 | 0.1215 |
| rsi_vol_combo | bull | 0.0326 | 0.1740 | 0.1872 | 0.1061 | 0.1035 |
| trend_lowvol | bear | 0.1393 | 0.1894 | 0.7355 | 0.5890 | 0.5843 |
| mom_x_lowvol_20_20 | bear | 0.1090 | 0.1695 | 0.6434 | 0.4521 | 0.4671 |
| momentum_reversal | bear | 0.1181 | 0.1886 | 0.6261 | 0.4521 | 0.4546 |
| rsi_vol_combo | bear | 0.0660 | 0.1714 | 0.3850 | 0.4247 | 0.2742 |
| fund_revenue_growth | bear | 0.0401 | 0.1081 | 0.3704 | 0.2603 | 0.2334 |
| fund_profit_growth | bear | 0.0368 | 0.1387 | 0.2656 | 0.2329 | 0.1637 |

### 黄金概念

- **Neutral**: ['fund_pe', 'turnover_stability', 'mom_x_lowvol_20_20'] (单因子IC=0.0539, 组合IC=0.0754)
  - weights: [0.3881, 0.3211, 0.2907]
- **Bull**: ['fund_pe', 'volatility', 'low_downside'] (单因子IC=0.0818, 组合IC=0.1146)
  - bull_weights: [0.4318, 0.3025, 0.2657]
- **Bear**: ['turnover_stability', 'mom_x_lowvol_20_20', 'momentum_reversal'] (单因子IC=0.0848, 组合IC=0.0883)
  - bear_weights: [0.4779, 0.2719, 0.2501]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| fund_pe | neutral | 0.0614 | 0.1879 | 0.3265 | 0.2463 | 0.2035 |
| turnover_stability | neutral | 0.0456 | 0.1681 | 0.2715 | 0.2401 | 0.1683 |
| mom_x_lowvol_20_20 | neutral | 0.0547 | 0.2192 | 0.2496 | 0.2213 | 0.1524 |
| fund_profit_growth | neutral | 0.0437 | 0.1874 | 0.2334 | 0.2422 | 0.1449 |
| fund_score | neutral | 0.0488 | 0.2090 | 0.2335 | 0.2213 | 0.1426 |
| fund_roe | neutral | 0.0453 | 0.2263 | 0.2000 | 0.2192 | 0.1219 |
| momentum_reversal | neutral | 0.0473 | 0.2506 | 0.1887 | 0.1378 | 0.1074 |
| trend_lowvol | neutral | 0.0446 | 0.2439 | 0.1828 | 0.1378 | 0.1040 |
| fund_revenue_growth | neutral | 0.0294 | 0.1890 | 0.1553 | 0.1608 | 0.0901 |
| fund_pb | neutral | 0.0302 | 0.2150 | 0.1405 | 0.1211 | 0.0788 |
| fund_pe | bull | 0.0858 | 0.1655 | 0.5188 | 0.4545 | 0.3773 |
| volatility | bull | 0.0757 | 0.1812 | 0.4178 | 0.2652 | 0.2643 |
| low_downside | bull | 0.0838 | 0.2270 | 0.3691 | 0.2576 | 0.2321 |
| fund_pb | bull | 0.0711 | 0.2526 | 0.2816 | 0.1742 | 0.1653 |
| turnover_stability | bull | 0.0245 | 0.1590 | 0.1539 | 0.1439 | 0.0881 |
| fund_gross_margin | bull | 0.0120 | 0.1148 | 0.1046 | 0.1061 | 0.0579 |
| turnover_stability | bear | 0.0634 | 0.1111 | 0.5703 | 0.5068 | 0.4297 |
| mom_x_lowvol_20_20 | bear | 0.0958 | 0.2470 | 0.3880 | 0.2603 | 0.2445 |
| momentum_reversal | bear | 0.0952 | 0.2667 | 0.3569 | 0.2603 | 0.2249 |
| rsi_vol_combo | bear | 0.0676 | 0.2298 | 0.2941 | 0.2603 | 0.1853 |
| fund_pb | bear | 0.0342 | 0.1629 | 0.2099 | 0.1507 | 0.1208 |

