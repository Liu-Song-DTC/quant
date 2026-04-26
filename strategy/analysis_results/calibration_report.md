# 离线标定报告

生成时间: 2026-04-26 19:30

## 汇总

- 行业数: 14
- 总标定组合: 377
- IC>5%的组合: 226 (59.9%)

## 详细结果

### 互联网/软件

- **Neutral**: ['trend_lowvol', 'mom_x_lowvol_20_10', 'mom_x_lowvol_20_20'] (单因子IC=0.0826, 组合IC=0.11)
  - weights: [0.3879, 0.3072, 0.3049]
- **Bull**: ['volatility', 'trend_lowvol', 'mom_x_lowvol_20_20'] (单因子IC=0.1086, 组合IC=0.136)
  - bull_weights: [0.3838, 0.3268, 0.2894]
- **Bear**: ['momentum_reversal'] (单因子IC=0.1154, 组合IC=0.1154)
  - bear_weights: [1.0]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| trend_lowvol | neutral | 0.0993 | 0.3017 | 0.3290 | 0.2806 | 0.2106 |
| mom_x_lowvol_20_10 | neutral | 0.0741 | 0.2747 | 0.2696 | 0.2374 | 0.1668 |
| mom_x_lowvol_20_20 | neutral | 0.0744 | 0.2776 | 0.2680 | 0.2356 | 0.1656 |
| momentum_reversal | neutral | 0.0732 | 0.2799 | 0.2615 | 0.2464 | 0.1629 |
| fund_profit_growth | neutral | 0.0702 | 0.2655 | 0.2642 | 0.2086 | 0.1597 |
| volatility | neutral | 0.0721 | 0.2872 | 0.2509 | 0.2113 | 0.1520 |
| mom_diff_5_20 | neutral | 0.0549 | 0.2749 | 0.1997 | 0.1538 | 0.1152 |
| fund_score | neutral | 0.0532 | 0.2715 | 0.1959 | 0.1763 | 0.1152 |
| fund_cf_to_profit | neutral | 0.0448 | 0.2455 | 0.1826 | 0.1295 | 0.1031 |
| mom_x_lowvol_10_20 | neutral | 0.0496 | 0.2845 | 0.1743 | 0.1367 | 0.0991 |
| mom_x_lowvol_10_10 | neutral | 0.0487 | 0.2840 | 0.1715 | 0.1385 | 0.0976 |
| bb_rsi_combo | neutral | 0.0464 | 0.2859 | 0.1621 | 0.1718 | 0.0950 |
| rsi_vol_combo | neutral | 0.0446 | 0.2854 | 0.1564 | 0.1556 | 0.0903 |
| volatility | bull | 0.1195 | 0.2621 | 0.4558 | 0.3834 | 0.3153 |
| trend_lowvol | bull | 0.1138 | 0.2799 | 0.4065 | 0.3207 | 0.2684 |
| mom_x_lowvol_20_20 | bull | 0.0926 | 0.2532 | 0.3657 | 0.3003 | 0.2377 |
| mom_x_lowvol_20_10 | bull | 0.0864 | 0.2513 | 0.3436 | 0.2595 | 0.2164 |
| momentum_reversal | bull | 0.0850 | 0.2605 | 0.3262 | 0.2755 | 0.2080 |
| mom_diff_5_20 | bull | 0.0678 | 0.2552 | 0.2656 | 0.2216 | 0.1622 |
| fund_profit_growth | bull | 0.0599 | 0.2443 | 0.2452 | 0.1443 | 0.1403 |
| bb_rsi_combo | bull | 0.0457 | 0.2615 | 0.1746 | 0.1283 | 0.0985 |
| mom_diff_10_20 | bull | 0.0464 | 0.2652 | 0.1750 | 0.1108 | 0.0972 |
| momentum_acceleration | bull | 0.0464 | 0.2652 | 0.1750 | 0.1108 | 0.0972 |
| mom_x_lowvol_10_20 | bull | 0.0442 | 0.2604 | 0.1698 | 0.1181 | 0.0949 |
| mom_x_lowvol_10_10 | bull | 0.0424 | 0.2650 | 0.1601 | 0.1239 | 0.0900 |
| momentum_reversal | bear | 0.1154 | 0.2463 | 0.4685 | 0.3312 | 0.3118 |
| mom_x_lowvol_20_20 | bear | 0.1149 | 0.2492 | 0.4609 | 0.3441 | 0.3097 |
| mom_x_lowvol_20_10 | bear | 0.1118 | 0.2496 | 0.4478 | 0.3344 | 0.2988 |
| bb_rsi_combo | bear | 0.0899 | 0.2507 | 0.3585 | 0.2785 | 0.2292 |
| fund_roe | bear | 0.0992 | 0.2819 | 0.3519 | 0.2742 | 0.2242 |
| rsi_vol_combo | bear | 0.0879 | 0.2505 | 0.3508 | 0.2667 | 0.2222 |
| mom_x_lowvol_10_10 | bear | 0.0919 | 0.2684 | 0.3424 | 0.2581 | 0.2154 |
| mom_x_lowvol_10_20 | bear | 0.0888 | 0.2672 | 0.3324 | 0.2677 | 0.2107 |
| fund_score | bear | 0.0945 | 0.2927 | 0.3230 | 0.3032 | 0.2105 |
| mom_diff_5_20 | bear | 0.0815 | 0.2539 | 0.3211 | 0.2656 | 0.2032 |
| trend_lowvol | bear | 0.0799 | 0.2645 | 0.3020 | 0.1935 | 0.1802 |
| fund_profit_growth | bear | 0.0718 | 0.2826 | 0.2542 | 0.2516 | 0.1591 |
| mom_diff_10_20 | bear | 0.0592 | 0.2533 | 0.2336 | 0.1613 | 0.1356 |
| momentum_acceleration | bear | 0.0592 | 0.2533 | 0.2336 | 0.1613 | 0.1356 |
| bb_width_20 | bear | 0.0220 | 0.2691 | 0.0818 | 0.1151 | 0.0456 |

### 交运

- **Neutral**: ['trend_lowvol'] (单因子IC=0.0941, 组合IC=0.0941)
  - weights: [1.0]
- **Bull**: ['volatility'] (单因子IC=0.0732, 组合IC=0.0732)
  - bull_weights: [1.0]
- **Bear**: ['trend_lowvol', 'bb_width_20'] (单因子IC=0.1081, 组合IC=0.1276)
  - bear_weights: [0.529, 0.471]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| trend_lowvol | neutral | 0.0941 | 0.3871 | 0.2431 | 0.1457 | 0.1393 |
| mom_x_lowvol_20_10 | neutral | 0.0572 | 0.3704 | 0.1543 | 0.1574 | 0.0893 |
| momentum_reversal | neutral | 0.0521 | 0.3667 | 0.1420 | 0.1403 | 0.0810 |
| mom_x_lowvol_20_20 | neutral | 0.0521 | 0.3721 | 0.1401 | 0.1412 | 0.0799 |
| fund_score | neutral | 0.0405 | 0.3503 | 0.1156 | 0.1241 | 0.0650 |
| fund_profit_growth | neutral | 0.0401 | 0.3505 | 0.1144 | 0.1061 | 0.0633 |
| volatility | bull | 0.0732 | 0.3344 | 0.2188 | 0.1720 | 0.1282 |
| fund_cf_to_profit | bull | 0.0368 | 0.3777 | 0.0976 | 0.1647 | 0.0568 |
| trend_lowvol | bear | 0.1198 | 0.3905 | 0.3066 | 0.2333 | 0.1891 |
| bb_width_20 | bear | 0.0965 | 0.3495 | 0.2761 | 0.2194 | 0.1684 |
| mom_x_lowvol_20_20 | bear | 0.1032 | 0.3933 | 0.2624 | 0.2333 | 0.1618 |
| bb_rsi_combo | bear | 0.0969 | 0.3749 | 0.2585 | 0.2366 | 0.1599 |
| rsi_vol_combo | bear | 0.0931 | 0.3663 | 0.2542 | 0.2366 | 0.1572 |
| mom_x_lowvol_20_10 | bear | 0.1003 | 0.3941 | 0.2544 | 0.2344 | 0.1570 |
| momentum_reversal | bear | 0.0960 | 0.3942 | 0.2435 | 0.2032 | 0.1465 |
| mom_x_lowvol_10_10 | bear | 0.0817 | 0.3821 | 0.2138 | 0.2290 | 0.1314 |
| mom_x_lowvol_10_20 | bear | 0.0760 | 0.3854 | 0.1972 | 0.1957 | 0.1179 |
| mom_diff_5_20 | bear | 0.0762 | 0.3981 | 0.1914 | 0.1269 | 0.1079 |
| fund_gross_margin | bear | 0.0429 | 0.3474 | 0.1236 | 0.1237 | 0.0694 |
| mom_diff_10_20 | bear | 0.0433 | 0.3859 | 0.1122 | 0.1151 | 0.0626 |
| momentum_acceleration | bear | 0.0433 | 0.3859 | 0.1122 | 0.1151 | 0.0626 |

### 军工

- **Neutral**: ['fund_profit_growth', 'fund_revenue_growth', 'fund_roe'] (单因子IC=0.0665, 组合IC=0.0748)
  - weights: [0.3825, 0.3202, 0.2973]
- **Bull**: ['volatility'] (单因子IC=0.1229, 组合IC=0.1229)
  - bull_weights: [1.0]
- **Bear**: ['mom_x_lowvol_20_20'] (单因子IC=0.1193, 组合IC=0.1193)
  - bear_weights: [1.0]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| fund_profit_growth | neutral | 0.0706 | 0.3527 | 0.2000 | 0.1574 | 0.1158 |
| fund_revenue_growth | neutral | 0.0613 | 0.3669 | 0.1671 | 0.1601 | 0.0969 |
| fund_roe | neutral | 0.0677 | 0.4221 | 0.1605 | 0.1214 | 0.0900 |
| fund_score | neutral | 0.0564 | 0.4172 | 0.1352 | 0.1268 | 0.0762 |
| volatility | bull | 0.1229 | 0.3586 | 0.3426 | 0.2551 | 0.2150 |
| mom_x_lowvol_20_20 | bear | 0.1193 | 0.3829 | 0.3117 | 0.2774 | 0.1991 |
| momentum_reversal | bear | 0.1024 | 0.3793 | 0.2699 | 0.2194 | 0.1646 |
| mom_x_lowvol_20_10 | bear | 0.1033 | 0.3866 | 0.2672 | 0.2172 | 0.1626 |
| rsi_vol_combo | bear | 0.0846 | 0.3605 | 0.2347 | 0.1785 | 0.1383 |
| mom_x_lowvol_10_20 | bear | 0.0924 | 0.4009 | 0.2306 | 0.1624 | 0.1340 |
| bb_rsi_combo | bear | 0.0845 | 0.3664 | 0.2306 | 0.1516 | 0.1328 |
| mom_x_lowvol_10_10 | bear | 0.0849 | 0.4048 | 0.2097 | 0.1505 | 0.1206 |
| trend_lowvol | bear | 0.0733 | 0.3865 | 0.1897 | 0.1677 | 0.1108 |
| mom_diff_5_20 | bear | 0.0692 | 0.3741 | 0.1850 | 0.1667 | 0.1079 |

### 化工

- **Neutral**: ['volatility'] (单因子IC=0.0526, 组合IC=0.0526)
  - weights: [1.0]
- **Bear**: ['momentum_reversal'] (单因子IC=0.0775, 组合IC=0.0775)
  - bear_weights: [1.0]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| volatility | neutral | 0.0526 | 0.2690 | 0.1956 | 0.1124 | 0.1088 |
| fund_cf_to_profit | neutral | 0.0315 | 0.2084 | 0.1512 | 0.1547 | 0.0873 |
| momentum_reversal | bear | 0.0775 | 0.2295 | 0.3379 | 0.2473 | 0.2107 |
| mom_x_lowvol_20_20 | bear | 0.0772 | 0.2389 | 0.3231 | 0.2409 | 0.2005 |
| mom_x_lowvol_20_10 | bear | 0.0732 | 0.2376 | 0.3080 | 0.2215 | 0.1881 |
| bb_rsi_combo | bear | 0.0607 | 0.2233 | 0.2718 | 0.1796 | 0.1603 |
| rsi_vol_combo | bear | 0.0571 | 0.2216 | 0.2579 | 0.1978 | 0.1544 |
| mom_x_lowvol_10_10 | bear | 0.0605 | 0.2504 | 0.2415 | 0.1269 | 0.1361 |
| mom_x_lowvol_10_20 | bear | 0.0598 | 0.2538 | 0.2355 | 0.1505 | 0.1355 |
| trend_lowvol | bear | 0.0560 | 0.2457 | 0.2280 | 0.1688 | 0.1332 |
| fund_revenue_growth | bear | 0.0453 | 0.2104 | 0.2152 | 0.1892 | 0.1280 |
| mom_diff_5_20 | bear | 0.0492 | 0.2351 | 0.2091 | 0.1570 | 0.1210 |
| mom_diff_10_20 | bear | 0.0459 | 0.2278 | 0.2013 | 0.1914 | 0.1199 |
| momentum_acceleration | bear | 0.0459 | 0.2278 | 0.2013 | 0.1914 | 0.1199 |
| fund_profit_growth | bear | 0.0482 | 0.2314 | 0.2083 | 0.1419 | 0.1189 |
| fund_score | bear | 0.0470 | 0.2351 | 0.2001 | 0.1785 | 0.1179 |
| bb_width_20 | bear | 0.0495 | 0.2662 | 0.1858 | 0.1215 | 0.1042 |
| fund_cf_to_profit | bear | 0.0310 | 0.2126 | 0.1459 | 0.1505 | 0.0839 |
| fund_roe | bear | 0.0267 | 0.2456 | 0.1086 | 0.1032 | 0.0599 |

### 半导体/光伏

- **Neutral**: ['fund_gross_margin'] (单因子IC=0.0541, 组合IC=0.0541)
  - weights: [1.0]
- **Bull**: ['volatility', 'mom_x_lowvol_20_20', 'momentum_reversal'] (单因子IC=0.0696, 组合IC=0.0972)
  - bull_weights: [0.3793, 0.312, 0.3087]
- **Bear**: ['mom_x_lowvol_20_20', 'momentum_reversal'] (单因子IC=0.1246, 组合IC=0.1252)
  - bear_weights: [0.5058, 0.4942]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| fund_gross_margin | neutral | 0.0541 | 0.2639 | 0.2049 | 0.1484 | 0.1177 |
| fund_revenue_growth | neutral | 0.0379 | 0.2644 | 0.1432 | 0.1385 | 0.0815 |
| volatility | bull | 0.0784 | 0.2921 | 0.2682 | 0.2245 | 0.1642 |
| mom_x_lowvol_20_20 | bull | 0.0659 | 0.2857 | 0.2305 | 0.1720 | 0.1351 |
| momentum_reversal | bull | 0.0645 | 0.2809 | 0.2295 | 0.1647 | 0.1337 |
| fund_score | bull | 0.0605 | 0.2826 | 0.2140 | 0.1895 | 0.1273 |
| fund_revenue_growth | bull | 0.0580 | 0.2591 | 0.2238 | 0.1166 | 0.1249 |
| mom_x_lowvol_20_10 | bull | 0.0616 | 0.2869 | 0.2148 | 0.1531 | 0.1238 |
| fund_roe | bull | 0.0585 | 0.2814 | 0.2078 | 0.1749 | 0.1221 |
| fund_cf_to_profit | bull | 0.0405 | 0.2111 | 0.1919 | 0.2682 | 0.1217 |
| mom_diff_5_20 | bull | 0.0558 | 0.2829 | 0.1973 | 0.1312 | 0.1116 |
| fund_profit_growth | bull | 0.0382 | 0.2522 | 0.1516 | 0.1808 | 0.0895 |
| mom_x_lowvol_10_10 | bull | 0.0368 | 0.2696 | 0.1365 | 0.1312 | 0.0772 |
| mom_x_lowvol_10_20 | bull | 0.0363 | 0.2687 | 0.1351 | 0.1429 | 0.0772 |
| mom_diff_10_20 | bull | 0.0369 | 0.2772 | 0.1331 | 0.1370 | 0.0757 |
| momentum_acceleration | bull | 0.0369 | 0.2772 | 0.1331 | 0.1370 | 0.0757 |
| trend_lowvol | bull | 0.0218 | 0.3018 | 0.0723 | 0.1195 | 0.0405 |
| mom_x_lowvol_20_20 | bear | 0.1241 | 0.2719 | 0.4566 | 0.3785 | 0.3147 |
| momentum_reversal | bear | 0.1252 | 0.2772 | 0.4514 | 0.3624 | 0.3075 |
| mom_x_lowvol_20_10 | bear | 0.1161 | 0.2704 | 0.4293 | 0.3441 | 0.2885 |
| bb_rsi_combo | bear | 0.0963 | 0.2668 | 0.3608 | 0.3441 | 0.2425 |
| rsi_vol_combo | bear | 0.0951 | 0.2702 | 0.3521 | 0.3419 | 0.2362 |
| mom_diff_5_20 | bear | 0.0994 | 0.2758 | 0.3604 | 0.2914 | 0.2327 |
| mom_x_lowvol_10_20 | bear | 0.0932 | 0.2712 | 0.3435 | 0.3140 | 0.2257 |
| mom_x_lowvol_10_10 | bear | 0.0922 | 0.2703 | 0.3412 | 0.3075 | 0.2231 |
| bb_width_20 | bear | 0.0848 | 0.2764 | 0.3067 | 0.2000 | 0.1840 |
| mom_diff_10_20 | bear | 0.0802 | 0.2793 | 0.2873 | 0.2333 | 0.1771 |
| momentum_acceleration | bear | 0.0802 | 0.2793 | 0.2873 | 0.2333 | 0.1771 |
| trend_lowvol | bear | 0.0845 | 0.2930 | 0.2884 | 0.1968 | 0.1726 |

### 基建/地产/石油石化

- **Neutral**: ['fund_roe', 'fund_score', 'fund_profit_growth'] (单因子IC=0.081, 组合IC=0.0877)
  - weights: [0.3497, 0.3439, 0.3065]
- **Bull**: ['fund_profit_growth', 'volatility'] (单因子IC=0.0758, 组合IC=0.0936)
  - bull_weights: [0.6769, 0.3231]
- **Bear**: ['mom_x_lowvol_10_20', 'mom_x_lowvol_10_10', 'momentum_reversal'] (单因子IC=0.0651, 组合IC=0.0656)
  - bear_weights: [0.3411, 0.3411, 0.3178]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| fund_roe | neutral | 0.0877 | 0.2833 | 0.3095 | 0.2698 | 0.1965 |
| fund_score | neutral | 0.0879 | 0.2862 | 0.3070 | 0.2590 | 0.1933 |
| fund_profit_growth | neutral | 0.0674 | 0.2456 | 0.2744 | 0.2554 | 0.1722 |
| fund_revenue_growth | neutral | 0.0448 | 0.2473 | 0.1813 | 0.1376 | 0.1031 |
| volatility | neutral | 0.0457 | 0.2714 | 0.1684 | 0.1655 | 0.0981 |
| tech_fund_combo | neutral | 0.0460 | 0.2827 | 0.1628 | 0.1241 | 0.0915 |
| fund_cf_to_profit | neutral | 0.0257 | 0.3075 | 0.0836 | 0.1277 | 0.0471 |
| fund_profit_growth | bull | 0.0902 | 0.2137 | 0.4219 | 0.3688 | 0.2888 |
| volatility | bull | 0.0614 | 0.2599 | 0.2364 | 0.1662 | 0.1378 |
| fund_score | bull | 0.0433 | 0.2539 | 0.1706 | 0.1472 | 0.0979 |
| fund_roe | bull | 0.0463 | 0.2657 | 0.1743 | 0.1210 | 0.0977 |
| tech_fund_combo | bull | 0.0230 | 0.2763 | 0.0831 | 0.1283 | 0.0469 |
| ret_vol_ratio_10 | bull | 0.0227 | 0.2740 | 0.0829 | 0.1210 | 0.0465 |
| fund_cf_to_profit | bull | 0.0209 | 0.2969 | 0.0705 | 0.1108 | 0.0392 |
| ret_vol_ratio_20 | bull | 0.0172 | 0.2812 | 0.0613 | 0.1356 | 0.0348 |
| mom_x_lowvol_10_20 | bear | 0.0645 | 0.2808 | 0.2299 | 0.1946 | 0.1373 |
| mom_x_lowvol_10_10 | bear | 0.0653 | 0.2833 | 0.2305 | 0.1914 | 0.1373 |
| momentum_reversal | bear | 0.0654 | 0.3056 | 0.2139 | 0.1957 | 0.1279 |
| fund_roe | bear | 0.0598 | 0.2785 | 0.2148 | 0.1731 | 0.1260 |
| mom_x_lowvol_20_20 | bear | 0.0655 | 0.3059 | 0.2140 | 0.1699 | 0.1252 |
| mom_x_lowvol_20_10 | bear | 0.0605 | 0.3059 | 0.1976 | 0.1398 | 0.1126 |
| bb_width_20 | bear | 0.0544 | 0.2947 | 0.1847 | 0.1247 | 0.1039 |
| rsi_vol_combo | bear | 0.0446 | 0.2812 | 0.1585 | 0.1591 | 0.0919 |
| bb_rsi_combo | bear | 0.0456 | 0.2835 | 0.1607 | 0.1183 | 0.0898 |
| mom_diff_5_20 | bear | 0.0454 | 0.2855 | 0.1590 | 0.1194 | 0.0890 |
| fund_profit_growth | bear | 0.0381 | 0.2542 | 0.1500 | 0.1237 | 0.0843 |
| trend_lowvol | bear | 0.0343 | 0.3034 | 0.1129 | 0.1355 | 0.0641 |
| mom_diff_10_20 | bear | 0.0304 | 0.2830 | 0.1075 | 0.1011 | 0.0592 |
| momentum_acceleration | bear | 0.0304 | 0.2830 | 0.1075 | 0.1011 | 0.0592 |
| fund_cf_to_profit | bear | 0.0332 | 0.3259 | 0.1018 | 0.1204 | 0.0570 |

### 新能源车/风电

- **Neutral**: ['fund_profit_growth', 'fund_score'] (单因子IC=0.0808, 组合IC=0.0822)
  - weights: [0.5159, 0.4841]
- **Bull**: ['volatility', 'fund_profit_growth'] (单因子IC=0.0723, 组合IC=0.0903)
  - bull_weights: [0.5472, 0.4528]
- **Bear**: ['fund_revenue_growth', 'mom_x_lowvol_20_20'] (单因子IC=0.0798, 组合IC=0.1276)
  - bear_weights: [0.5521, 0.4479]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| fund_profit_growth | neutral | 0.0804 | 0.2166 | 0.3712 | 0.2824 | 0.2380 |
| fund_score | neutral | 0.0813 | 0.2331 | 0.3488 | 0.2806 | 0.2233 |
| fund_roe | neutral | 0.0613 | 0.2366 | 0.2592 | 0.2266 | 0.1590 |
| fund_revenue_growth | neutral | 0.0569 | 0.2204 | 0.2580 | 0.2059 | 0.1556 |
| tech_fund_combo | neutral | 0.0613 | 0.2532 | 0.2422 | 0.1637 | 0.1409 |
| fund_gross_margin | neutral | 0.0393 | 0.2263 | 0.1736 | 0.1385 | 0.0988 |
| fund_cf_to_profit | neutral | 0.0231 | 0.2035 | 0.1135 | 0.1133 | 0.0632 |
| volatility | bull | 0.0831 | 0.2660 | 0.3124 | 0.2420 | 0.1940 |
| fund_profit_growth | bull | 0.0614 | 0.2181 | 0.2816 | 0.1399 | 0.1605 |
| fund_score | bull | 0.0491 | 0.2492 | 0.1968 | 0.1749 | 0.1156 |
| fund_revenue_growth | bull | 0.0420 | 0.2374 | 0.1769 | 0.1603 | 0.1026 |
| mom_x_lowvol_10_20 | bull | 0.0354 | 0.2180 | 0.1623 | 0.1181 | 0.0907 |
| mom_x_lowvol_20_20 | bull | 0.0327 | 0.2361 | 0.1386 | 0.1560 | 0.0801 |
| mom_x_lowvol_20_10 | bull | 0.0288 | 0.2362 | 0.1218 | 0.1516 | 0.0701 |
| momentum_reversal | bull | 0.0266 | 0.2325 | 0.1143 | 0.1283 | 0.0645 |
| fund_revenue_growth | bear | 0.0772 | 0.2029 | 0.3807 | 0.2935 | 0.2462 |
| mom_x_lowvol_20_20 | bear | 0.0824 | 0.2600 | 0.3170 | 0.2602 | 0.1998 |
| mom_x_lowvol_20_10 | bear | 0.0805 | 0.2597 | 0.3101 | 0.2441 | 0.1929 |
| fund_profit_growth | bear | 0.0673 | 0.2165 | 0.3110 | 0.2247 | 0.1905 |
| momentum_reversal | bear | 0.0744 | 0.2608 | 0.2853 | 0.2828 | 0.1830 |
| bb_rsi_combo | bear | 0.0608 | 0.2346 | 0.2592 | 0.2430 | 0.1611 |
| fund_score | bear | 0.0615 | 0.2222 | 0.2769 | 0.1602 | 0.1606 |
| mom_x_lowvol_10_10 | bear | 0.0610 | 0.2411 | 0.2532 | 0.2495 | 0.1582 |
| mom_x_lowvol_10_20 | bear | 0.0606 | 0.2381 | 0.2544 | 0.2172 | 0.1548 |
| rsi_vol_combo | bear | 0.0584 | 0.2342 | 0.2496 | 0.2333 | 0.1539 |
| mom_diff_5_20 | bear | 0.0486 | 0.2482 | 0.1958 | 0.2022 | 0.1177 |
| bb_width_20 | bear | 0.0501 | 0.2530 | 0.1979 | 0.1828 | 0.1170 |
| fund_roe | bear | 0.0310 | 0.2264 | 0.1368 | 0.1677 | 0.0799 |
| mom_diff_10_20 | bear | 0.0278 | 0.2380 | 0.1167 | 0.1312 | 0.0660 |
| momentum_acceleration | bear | 0.0278 | 0.2380 | 0.1167 | 0.1312 | 0.0660 |

### 有色/钢铁/煤炭/建材

- **Neutral**: ['fund_profit_growth'] (单因子IC=0.0623, 组合IC=0.0623)
  - weights: [1.0]
- **Bull**: ['volatility', 'fund_profit_growth'] (单因子IC=0.0854, 组合IC=0.1186)
  - bull_weights: [0.6538, 0.3462]
- **Bear**: ['fund_profit_growth'] (单因子IC=0.0489, 组合IC=0.0489)
  - bear_weights: [1.0]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| fund_profit_growth | neutral | 0.0623 | 0.1743 | 0.3573 | 0.3040 | 0.2329 |
| fund_score | neutral | 0.0509 | 0.1881 | 0.2706 | 0.1996 | 0.1623 |
| fund_revenue_growth | neutral | 0.0387 | 0.1603 | 0.2412 | 0.1888 | 0.1434 |
| tech_fund_combo | neutral | 0.0374 | 0.1901 | 0.1965 | 0.1511 | 0.1131 |
| fund_roe | neutral | 0.0362 | 0.2001 | 0.1810 | 0.1745 | 0.1063 |
| volatility | neutral | 0.0456 | 0.2531 | 0.1801 | 0.1241 | 0.1012 |
| fund_cf_to_profit | neutral | 0.0290 | 0.1755 | 0.1650 | 0.1493 | 0.0948 |
| trend_lowvol | neutral | 0.0329 | 0.2239 | 0.1471 | 0.1727 | 0.0863 |
| rsi_factor | neutral | 0.0171 | 0.2347 | 0.0728 | 0.1133 | 0.0405 |
| volatility | bull | 0.1118 | 0.2231 | 0.5012 | 0.3819 | 0.3463 |
| fund_profit_growth | bull | 0.0589 | 0.1893 | 0.3114 | 0.1778 | 0.1834 |
| fund_revenue_growth | bull | 0.0497 | 0.1811 | 0.2741 | 0.1749 | 0.1610 |
| fund_score | bull | 0.0547 | 0.2033 | 0.2691 | 0.1778 | 0.1585 |
| mom_x_lowvol_20_20 | bull | 0.0435 | 0.2123 | 0.2049 | 0.1487 | 0.1177 |
| mom_x_lowvol_20_10 | bull | 0.0429 | 0.2121 | 0.2022 | 0.1399 | 0.1152 |
| momentum_reversal | bull | 0.0324 | 0.2082 | 0.1556 | 0.1254 | 0.0875 |
| mom_diff_5_20 | bull | 0.0281 | 0.2053 | 0.1366 | 0.1108 | 0.0759 |
| fund_profit_growth | bear | 0.0489 | 0.1789 | 0.2734 | 0.1978 | 0.1637 |
| fund_score | bear | 0.0492 | 0.2209 | 0.2228 | 0.1720 | 0.1306 |
| mom_x_lowvol_10_10 | bear | 0.0514 | 0.2453 | 0.2097 | 0.1591 | 0.1215 |
| fund_roe | bear | 0.0496 | 0.2336 | 0.2125 | 0.1419 | 0.1213 |
| mom_x_lowvol_20_20 | bear | 0.0509 | 0.2502 | 0.2034 | 0.1591 | 0.1179 |
| mom_x_lowvol_10_20 | bear | 0.0509 | 0.2478 | 0.2052 | 0.1441 | 0.1174 |
| trend_lowvol | bear | 0.0496 | 0.2420 | 0.2050 | 0.1398 | 0.1168 |
| momentum_reversal | bear | 0.0501 | 0.2565 | 0.1952 | 0.1312 | 0.1104 |
| mom_x_lowvol_20_10 | bear | 0.0471 | 0.2486 | 0.1893 | 0.1355 | 0.1075 |
| rsi_vol_combo | bear | 0.0423 | 0.2423 | 0.1745 | 0.1011 | 0.0960 |
| fund_cf_to_profit | bear | 0.0253 | 0.1822 | 0.1388 | 0.1269 | 0.0782 |

### 消费/传媒/农业/环保/医药

- **Neutral**: ['fund_score', 'tech_fund_combo', 'bb_width_20'] (单因子IC=0.0495, 组合IC=0.0666)
  - weights: [0.4399, 0.3436, 0.2165]
- **Bull**: ['fund_revenue_growth', 'bb_width_20'] (单因子IC=0.0195, 组合IC=0.0338)
  - bull_weights: [0.6469, 0.3531]
- **Bear**: ['bb_width_20'] (单因子IC=0.0721, 组合IC=0.0721)
  - bear_weights: [1.0]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| fund_score | neutral | 0.0661 | 0.2768 | 0.2390 | 0.1763 | 0.1405 |
| tech_fund_combo | neutral | 0.0496 | 0.2601 | 0.1907 | 0.1511 | 0.1098 |
| bb_width_20 | neutral | 0.0329 | 0.2725 | 0.1208 | 0.1457 | 0.0692 |
| fund_profit_growth | neutral | 0.0251 | 0.2198 | 0.1143 | 0.1520 | 0.0658 |
| fund_revenue_growth | bull | 0.0215 | 0.1799 | 0.1195 | 0.1487 | 0.0686 |
| bb_width_20 | bull | 0.0175 | 0.2586 | 0.0676 | 0.1079 | 0.0374 |
| bb_width_20 | bear | 0.0721 | 0.2817 | 0.2560 | 0.2161 | 0.1557 |
| rsi_vol_combo | bear | 0.0519 | 0.2328 | 0.2228 | 0.1677 | 0.1301 |
| bb_rsi_combo | bear | 0.0501 | 0.2344 | 0.2137 | 0.1505 | 0.1229 |
| fund_score | bear | 0.0642 | 0.3055 | 0.2103 | 0.1591 | 0.1219 |
| momentum_reversal | bear | 0.0509 | 0.2695 | 0.1888 | 0.1247 | 0.1062 |
| trend_lowvol | bear | 0.0525 | 0.2928 | 0.1794 | 0.1828 | 0.1061 |
| mom_x_lowvol_20_20 | bear | 0.0507 | 0.2771 | 0.1829 | 0.1065 | 0.1012 |
| fund_revenue_growth | bear | 0.0351 | 0.2039 | 0.1722 | 0.1634 | 0.1001 |
| mom_diff_5_20 | bear | 0.0452 | 0.2681 | 0.1685 | 0.1000 | 0.0927 |
| mom_x_lowvol_20_10 | bear | 0.0460 | 0.2777 | 0.1658 | 0.1075 | 0.0918 |
| fund_profit_growth | bear | 0.0359 | 0.2285 | 0.1572 | 0.1172 | 0.0878 |
| mom_x_lowvol_10_20 | bear | 0.0366 | 0.2562 | 0.1427 | 0.1011 | 0.0786 |

### 电力设备

- **Neutral**: ['fund_score'] (单因子IC=0.0806, 组合IC=0.0806)
  - weights: [1.0]
- **Bull**: ['fund_score', 'fund_profit_growth'] (单因子IC=0.074, 组合IC=0.08)
  - bull_weights: [0.5163, 0.4837]
- **Bear**: ['momentum_reversal', 'mom_x_lowvol_20_20'] (单因子IC=0.107, 组合IC=0.1098)
  - bear_weights: [0.5143, 0.4857]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| fund_score | neutral | 0.0806 | 0.2069 | 0.3896 | 0.2482 | 0.2431 |
| fund_roe | neutral | 0.0634 | 0.1994 | 0.3179 | 0.2680 | 0.2015 |
| fund_profit_growth | neutral | 0.0603 | 0.1895 | 0.3180 | 0.2581 | 0.2000 |
| mom_x_lowvol_20_20 | neutral | 0.0548 | 0.2509 | 0.2183 | 0.1592 | 0.1265 |
| mom_x_lowvol_20_10 | neutral | 0.0527 | 0.2502 | 0.2107 | 0.1529 | 0.1214 |
| fund_revenue_growth | neutral | 0.0316 | 0.1751 | 0.1805 | 0.2185 | 0.1100 |
| momentum_reversal | neutral | 0.0490 | 0.2510 | 0.1953 | 0.1223 | 0.1096 |
| trend_lowvol | neutral | 0.0429 | 0.2431 | 0.1765 | 0.1547 | 0.1019 |
| volatility | neutral | 0.0441 | 0.2534 | 0.1739 | 0.1637 | 0.1012 |
| fund_score | bull | 0.0780 | 0.1951 | 0.3997 | 0.2974 | 0.2593 |
| fund_profit_growth | bull | 0.0700 | 0.1873 | 0.3737 | 0.3003 | 0.2430 |
| fund_roe | bull | 0.0644 | 0.2020 | 0.3188 | 0.2872 | 0.2052 |
| volatility | bull | 0.0676 | 0.2161 | 0.3129 | 0.2711 | 0.1989 |
| fund_revenue_growth | bull | 0.0525 | 0.1717 | 0.3059 | 0.2770 | 0.1953 |
| trend_lowvol | bull | 0.0387 | 0.2356 | 0.1644 | 0.1866 | 0.0975 |
| mom_x_lowvol_20_20 | bull | 0.0281 | 0.2222 | 0.1264 | 0.1283 | 0.0713 |
| mom_x_lowvol_20_10 | bull | 0.0256 | 0.2234 | 0.1146 | 0.1195 | 0.0641 |
| momentum_reversal | bull | 0.0217 | 0.2225 | 0.0977 | 0.1137 | 0.0544 |
| momentum_reversal | bear | 0.1091 | 0.2502 | 0.4360 | 0.3290 | 0.2898 |
| mom_x_lowvol_20_20 | bear | 0.1050 | 0.2537 | 0.4138 | 0.3226 | 0.2736 |
| mom_x_lowvol_20_10 | bear | 0.1006 | 0.2491 | 0.4040 | 0.3226 | 0.2671 |
| mom_x_lowvol_10_10 | bear | 0.0898 | 0.2464 | 0.3643 | 0.3054 | 0.2378 |
| mom_x_lowvol_10_20 | bear | 0.0902 | 0.2515 | 0.3585 | 0.3011 | 0.2332 |
| bb_rsi_combo | bear | 0.0823 | 0.2283 | 0.3604 | 0.2710 | 0.2290 |
| fund_score | bear | 0.0726 | 0.2099 | 0.3461 | 0.2882 | 0.2229 |
| rsi_vol_combo | bear | 0.0775 | 0.2261 | 0.3429 | 0.2753 | 0.2186 |
| mom_diff_5_20 | bear | 0.0657 | 0.2435 | 0.2696 | 0.1860 | 0.1599 |
| fund_profit_growth | bear | 0.0429 | 0.1729 | 0.2482 | 0.2301 | 0.1526 |
| fund_revenue_growth | bear | 0.0399 | 0.1838 | 0.2172 | 0.2043 | 0.1308 |
| fund_roe | bear | 0.0375 | 0.1992 | 0.1881 | 0.1817 | 0.1111 |
| bb_width_20 | bear | 0.0429 | 0.2570 | 0.1671 | 0.1097 | 0.0927 |

### 电子

- **Neutral**: ['trend_lowvol', 'fund_score'] (单因子IC=0.0551, 组合IC=0.0721)
  - weights: [0.6024, 0.3976]
- **Bull**: ['fund_gross_margin', 'fund_revenue_growth', 'fund_profit_growth'] (单因子IC=0.0447, 组合IC=0.0586)
  - bull_weights: [0.3513, 0.3444, 0.3043]
- **Bear**: ['trend_lowvol', 'fund_profit_growth'] (单因子IC=0.0859, 组合IC=0.1121)
  - bear_weights: [0.5604, 0.4396]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| trend_lowvol | neutral | 0.0666 | 0.2569 | 0.2594 | 0.2131 | 0.1573 |
| fund_score | neutral | 0.0437 | 0.2438 | 0.1791 | 0.1601 | 0.1039 |
| fund_revenue_growth | neutral | 0.0333 | 0.2041 | 0.1632 | 0.1511 | 0.0939 |
| mom_x_lowvol_20_20 | neutral | 0.0342 | 0.2248 | 0.1520 | 0.1169 | 0.0849 |
| mom_x_lowvol_20_10 | neutral | 0.0334 | 0.2226 | 0.1500 | 0.1025 | 0.0827 |
| fund_profit_growth | neutral | 0.0289 | 0.2240 | 0.1291 | 0.1439 | 0.0738 |
| mom_diff_5_20 | neutral | 0.0283 | 0.2255 | 0.1253 | 0.1043 | 0.0692 |
| volatility | neutral | 0.0121 | 0.2527 | 0.0477 | 0.1187 | 0.0267 |
| fund_gross_margin | bull | 0.0449 | 0.1776 | 0.2527 | 0.1808 | 0.1492 |
| fund_revenue_growth | bull | 0.0463 | 0.1968 | 0.2353 | 0.2434 | 0.1463 |
| fund_profit_growth | bull | 0.0428 | 0.2005 | 0.2136 | 0.2099 | 0.1292 |
| fund_score | bull | 0.0512 | 0.2545 | 0.2011 | 0.2332 | 0.1240 |
| volatility | bull | 0.0483 | 0.2328 | 0.2075 | 0.1603 | 0.1204 |
| fund_roe | bull | 0.0461 | 0.2605 | 0.1770 | 0.1691 | 0.1035 |
| trend_lowvol | bull | 0.0431 | 0.2397 | 0.1798 | 0.1050 | 0.0993 |
| mom_x_lowvol_20_10 | bull | 0.0231 | 0.2165 | 0.1068 | 0.1079 | 0.0591 |
| trend_lowvol | bear | 0.1045 | 0.2682 | 0.3898 | 0.2914 | 0.2517 |
| fund_profit_growth | bear | 0.0673 | 0.2076 | 0.3242 | 0.2183 | 0.1975 |
| momentum_reversal | bear | 0.0789 | 0.2543 | 0.3103 | 0.2452 | 0.1932 |
| mom_x_lowvol_20_20 | bear | 0.0761 | 0.2562 | 0.2971 | 0.2151 | 0.1805 |
| mom_x_lowvol_20_10 | bear | 0.0733 | 0.2535 | 0.2891 | 0.2129 | 0.1753 |
| mom_diff_5_20 | bear | 0.0633 | 0.2453 | 0.2582 | 0.2054 | 0.1556 |
| fund_score | bear | 0.0544 | 0.2509 | 0.2170 | 0.1419 | 0.1239 |
| bb_rsi_combo | bear | 0.0533 | 0.2500 | 0.2131 | 0.1613 | 0.1237 |
| mom_x_lowvol_10_10 | bear | 0.0525 | 0.2472 | 0.2123 | 0.1527 | 0.1224 |
| bb_width_20 | bear | 0.0454 | 0.2261 | 0.2007 | 0.2043 | 0.1209 |
| rsi_vol_combo | bear | 0.0517 | 0.2501 | 0.2069 | 0.1548 | 0.1195 |
| mom_x_lowvol_10_20 | bear | 0.0510 | 0.2507 | 0.2033 | 0.1570 | 0.1176 |
| mom_diff_10_20 | bear | 0.0469 | 0.2384 | 0.1969 | 0.1742 | 0.1156 |
| momentum_acceleration | bear | 0.0469 | 0.2384 | 0.1969 | 0.1742 | 0.1156 |
| fund_roe | bear | 0.0524 | 0.2649 | 0.1978 | 0.1591 | 0.1146 |
| fund_revenue_growth | bear | 0.0280 | 0.1964 | 0.1426 | 0.1720 | 0.0835 |

### 自动化/制造

- **Neutral**: ['fund_profit_growth'] (单因子IC=0.0744, 组合IC=0.0744)
  - weights: [1.0]
- **Bull**: ['fund_score'] (单因子IC=0.086, 组合IC=0.086)
  - bull_weights: [1.0]
- **Bear**: ['trend_lowvol', 'momentum_reversal'] (单因子IC=0.0951, 组合IC=0.1022)
  - bear_weights: [0.5149, 0.4851]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| fund_profit_growth | neutral | 0.0744 | 0.2661 | 0.2797 | 0.2752 | 0.1783 |
| fund_score | neutral | 0.0744 | 0.2929 | 0.2538 | 0.1906 | 0.1511 |
| tech_fund_combo | neutral | 0.0533 | 0.2896 | 0.1840 | 0.1502 | 0.1058 |
| fund_revenue_growth | neutral | 0.0497 | 0.2729 | 0.1821 | 0.1142 | 0.1014 |
| trend_lowvol | neutral | 0.0532 | 0.3095 | 0.1719 | 0.1349 | 0.0976 |
| fund_score | bull | 0.0860 | 0.3007 | 0.2861 | 0.1983 | 0.1714 |
| fund_roe | bull | 0.0607 | 0.2779 | 0.2185 | 0.1545 | 0.1261 |
| fund_gross_margin | bull | 0.0585 | 0.2775 | 0.2107 | 0.1501 | 0.1212 |
| fund_profit_growth | bull | 0.0488 | 0.2588 | 0.1885 | 0.1895 | 0.1121 |
| ret_vol_ratio_20 | bull | 0.0460 | 0.2795 | 0.1645 | 0.1399 | 0.0938 |
| fund_cf_to_profit | bull | 0.0422 | 0.2770 | 0.1524 | 0.1181 | 0.0852 |
| volatility | bull | 0.0463 | 0.3120 | 0.1484 | 0.1210 | 0.0832 |
| tech_fund_combo | bull | 0.0380 | 0.2790 | 0.1363 | 0.1254 | 0.0767 |
| rsi_factor | bull | 0.0290 | 0.2760 | 0.1050 | 0.1050 | 0.0580 |
| trend_lowvol | bear | 0.0928 | 0.2710 | 0.3424 | 0.2946 | 0.2216 |
| momentum_reversal | bear | 0.0975 | 0.2963 | 0.3289 | 0.2699 | 0.2088 |
| mom_x_lowvol_20_20 | bear | 0.0963 | 0.3000 | 0.3210 | 0.2699 | 0.2038 |
| mom_x_lowvol_20_10 | bear | 0.0857 | 0.3050 | 0.2810 | 0.2441 | 0.1748 |
| mom_diff_5_20 | bear | 0.0762 | 0.2770 | 0.2751 | 0.2527 | 0.1723 |
| rsi_vol_combo | bear | 0.0796 | 0.2826 | 0.2817 | 0.1989 | 0.1689 |
| bb_rsi_combo | bear | 0.0799 | 0.2879 | 0.2775 | 0.2097 | 0.1679 |
| fund_profit_growth | bear | 0.0648 | 0.2564 | 0.2528 | 0.2387 | 0.1566 |
| mom_x_lowvol_10_20 | bear | 0.0775 | 0.3135 | 0.2471 | 0.2237 | 0.1512 |
| mom_x_lowvol_10_10 | bear | 0.0753 | 0.3120 | 0.2413 | 0.2301 | 0.1484 |
| fund_score | bear | 0.0590 | 0.2722 | 0.2170 | 0.2226 | 0.1326 |
| fund_revenue_growth | bear | 0.0575 | 0.2725 | 0.2110 | 0.1688 | 0.1233 |
| mom_diff_10_20 | bear | 0.0524 | 0.2741 | 0.1912 | 0.1667 | 0.1115 |
| momentum_acceleration | bear | 0.0524 | 0.2741 | 0.1912 | 0.1667 | 0.1115 |
| bb_width_20 | bear | 0.0400 | 0.2784 | 0.1437 | 0.1333 | 0.0814 |
| volume_ratio | bear | 0.0293 | 0.2793 | 0.1048 | 0.1374 | 0.0596 |

### 通信/计算机

- **Neutral**: ['fund_score', 'fund_profit_growth'] (单因子IC=0.0727, 组合IC=0.0807)
  - weights: [0.5432, 0.4568]
- **Bull**: ['fund_score', 'fund_profit_growth', 'fund_revenue_growth'] (单因子IC=0.0544, 组合IC=0.0661)
  - bull_weights: [0.3463, 0.328, 0.3258]
- **Bear**: ['mom_x_lowvol_20_20', 'momentum_reversal'] (单因子IC=0.1107, 组合IC=0.1111)
  - bear_weights: [0.5003, 0.4997]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| fund_score | neutral | 0.0787 | 0.2688 | 0.2928 | 0.2383 | 0.1813 |
| fund_profit_growth | neutral | 0.0667 | 0.2615 | 0.2552 | 0.1942 | 0.1524 |
| fund_roe | neutral | 0.0670 | 0.2685 | 0.2497 | 0.2041 | 0.1504 |
| fund_revenue_growth | neutral | 0.0546 | 0.2726 | 0.2005 | 0.1745 | 0.1177 |
| tech_fund_combo | neutral | 0.0397 | 0.2767 | 0.1436 | 0.1637 | 0.0836 |
| trend_lowvol | neutral | 0.0176 | 0.3082 | 0.0572 | 0.1016 | 0.0315 |
| fund_score | bull | 0.0529 | 0.2595 | 0.2039 | 0.1603 | 0.1183 |
| fund_profit_growth | bull | 0.0540 | 0.2857 | 0.1889 | 0.1866 | 0.1121 |
| fund_revenue_growth | bull | 0.0565 | 0.3002 | 0.1881 | 0.1837 | 0.1113 |
| tech_fund_combo | bull | 0.0390 | 0.2687 | 0.1452 | 0.1385 | 0.0827 |
| ret_vol_ratio_20 | bull | 0.0340 | 0.2772 | 0.1227 | 0.1560 | 0.0709 |
| rsi_factor | bull | 0.0280 | 0.2673 | 0.1048 | 0.1181 | 0.0586 |
| fund_roe | bull | 0.0247 | 0.2732 | 0.0904 | 0.1181 | 0.0505 |
| trend_mom_v24 | bull | 0.0222 | 0.2760 | 0.0806 | 0.1239 | 0.0453 |
| trend_mom_v41 | bull | 0.0153 | 0.2656 | 0.0574 | 0.1023 | 0.0317 |
| trend_mom_v46 | bull | 0.0153 | 0.2656 | 0.0574 | 0.1023 | 0.0317 |
| mom_x_lowvol_20_20 | bear | 0.1103 | 0.3203 | 0.3442 | 0.2720 | 0.2189 |
| momentum_reversal | bear | 0.1112 | 0.3174 | 0.3503 | 0.2484 | 0.2187 |
| mom_x_lowvol_20_10 | bear | 0.1075 | 0.3207 | 0.3352 | 0.2516 | 0.2098 |
| mom_x_lowvol_10_10 | bear | 0.1004 | 0.3214 | 0.3124 | 0.2516 | 0.1955 |
| mom_x_lowvol_10_20 | bear | 0.0970 | 0.3196 | 0.3035 | 0.2452 | 0.1889 |
| mom_diff_5_20 | bear | 0.0874 | 0.3123 | 0.2799 | 0.2032 | 0.1684 |
| trend_lowvol | bear | 0.1020 | 0.3715 | 0.2745 | 0.1548 | 0.1585 |
| bb_rsi_combo | bear | 0.0809 | 0.3146 | 0.2570 | 0.1656 | 0.1498 |
| rsi_vol_combo | bear | 0.0797 | 0.3130 | 0.2547 | 0.1656 | 0.1484 |
| mom_diff_10_20 | bear | 0.0661 | 0.3037 | 0.2177 | 0.2000 | 0.1306 |
| momentum_acceleration | bear | 0.0661 | 0.3037 | 0.2177 | 0.2000 | 0.1306 |
| fund_gross_margin | bear | 0.0471 | 0.2350 | 0.2005 | 0.1505 | 0.1154 |
| fund_profit_growth | bear | 0.0493 | 0.2666 | 0.1849 | 0.1172 | 0.1033 |
| fund_score | bear | 0.0447 | 0.2884 | 0.1551 | 0.1065 | 0.0858 |
| bb_width_20 | bear | 0.0387 | 0.3341 | 0.1157 | 0.1323 | 0.0655 |
| fund_revenue_growth | bear | 0.0243 | 0.2730 | 0.0888 | 0.1194 | 0.0497 |

### 金融

- **Neutral**: ['fund_roe'] (单因子IC=0.0862, 组合IC=0.0862)
  - weights: [1.0]
- **Bull**: ['fund_score', 'fund_roe', 'volatility'] (单因子IC=0.1111, 组合IC=0.1544)
  - bull_weights: [0.3753, 0.3539, 0.2707]
- **Bear**: ['fund_profit_growth', 'fund_revenue_growth'] (单因子IC=0.0702, 组合IC=0.0875)
  - bear_weights: [0.5424, 0.4576]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| fund_roe | neutral | 0.0862 | 0.3706 | 0.2326 | 0.2194 | 0.1418 |
| fund_revenue_growth | neutral | 0.0707 | 0.3147 | 0.2248 | 0.1978 | 0.1346 |
| fund_score | neutral | 0.0784 | 0.3641 | 0.2153 | 0.2194 | 0.1313 |
| ret_vol_ratio_20 | neutral | 0.0729 | 0.3522 | 0.2069 | 0.1960 | 0.1238 |
| tech_fund_combo | neutral | 0.0706 | 0.3513 | 0.2010 | 0.1574 | 0.1163 |
| rsi_factor | neutral | 0.0671 | 0.3684 | 0.1821 | 0.1718 | 0.1067 |
| fund_profit_growth | neutral | 0.0558 | 0.3550 | 0.1571 | 0.1511 | 0.0904 |
| trend_mom_v24 | neutral | 0.0512 | 0.3608 | 0.1419 | 0.1241 | 0.0797 |
| V41_RSI_915 | neutral | 0.0476 | 0.3547 | 0.1341 | 0.1295 | 0.0757 |
| fund_score | bull | 0.1169 | 0.3301 | 0.3541 | 0.2449 | 0.2204 |
| fund_roe | bull | 0.1089 | 0.3253 | 0.3347 | 0.2420 | 0.2079 |
| volatility | bull | 0.1076 | 0.4055 | 0.2654 | 0.1983 | 0.1590 |
| tech_fund_combo | bull | 0.0919 | 0.3685 | 0.2494 | 0.2128 | 0.1512 |
| rsi_factor | bull | 0.0786 | 0.3696 | 0.2127 | 0.1210 | 0.1192 |
| volume_ratio | bull | 0.0605 | 0.3338 | 0.1813 | 0.1631 | 0.1054 |
| ret_vol_ratio_10 | bull | 0.0647 | 0.3585 | 0.1805 | 0.1487 | 0.1036 |
| fund_revenue_growth | bull | 0.0434 | 0.3062 | 0.1419 | 0.1327 | 0.0803 |
| fund_cf_to_profit | bull | 0.0345 | 0.3016 | 0.1144 | 0.1093 | 0.0635 |
| trend_mom_v24 | bull | 0.0340 | 0.3803 | 0.0895 | 0.1020 | 0.0493 |
| fund_profit_growth | bear | 0.0809 | 0.3614 | 0.2239 | 0.1419 | 0.1278 |
| fund_revenue_growth | bear | 0.0594 | 0.3186 | 0.1864 | 0.1570 | 0.1079 |
| fund_score | bear | 0.0618 | 0.3686 | 0.1677 | 0.1892 | 0.0997 |
| trend_lowvol | bear | 0.0535 | 0.4282 | 0.1250 | 0.1194 | 0.0700 |

