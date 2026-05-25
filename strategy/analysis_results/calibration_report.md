# 离线标定报告

生成时间: 2026-05-24 13:37

## 汇总

- 行业数: 19
- 总标定组合: 433
- IC>5%的组合: 340 (78.5%)

## 详细结果

### 互联网/软件

- **Neutral**: ['trend_lowvol'] (单因子IC=0.1477, 组合IC=0.1477)
  - weights: [1.0]
- **Bull**: ['volatility', 'trend_lowvol'] (单因子IC=0.1465, 组合IC=0.1635)
  - bull_weights: [0.5403, 0.4597]
- **Bear**: ['mom_x_lowvol_20_20', 'momentum_reversal'] (单因子IC=0.1894, 组合IC=0.1926)
  - bear_weights: [0.5209, 0.4791]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| trend_lowvol | neutral | 0.1477 | 0.1612 | 0.9159 | 0.5177 | 0.6951 |
| volatility | neutral | 0.0954 | 0.1495 | 0.6383 | 0.4752 | 0.4708 |
| low_downside | neutral | 0.0667 | 0.1365 | 0.4890 | 0.3475 | 0.3295 |
| fund_profit_growth | neutral | 0.0531 | 0.1185 | 0.4485 | 0.2482 | 0.2799 |
| momentum_reversal | neutral | 0.0734 | 0.1927 | 0.3810 | 0.2908 | 0.2459 |
| mom_x_lowvol_20_20 | neutral | 0.0633 | 0.1867 | 0.3388 | 0.2766 | 0.2163 |
| fund_revenue_growth | neutral | 0.0353 | 0.0949 | 0.3715 | 0.1348 | 0.2108 |
| rsi_vol_combo | neutral | 0.0571 | 0.1914 | 0.2982 | 0.2482 | 0.1861 |
| fund_score | neutral | 0.0434 | 0.1497 | 0.2899 | 0.1773 | 0.1707 |
| volatility | bull | 0.1538 | 0.1412 | 1.0894 | 0.7127 | 0.9329 |
| trend_lowvol | bull | 0.1392 | 0.1473 | 0.9451 | 0.6796 | 0.7937 |
| low_downside | bull | 0.0971 | 0.1190 | 0.8156 | 0.5470 | 0.6309 |
| mom_x_lowvol_20_20 | bull | 0.1043 | 0.1343 | 0.7762 | 0.5359 | 0.5961 |
| fund_profit_growth | bull | 0.0560 | 0.0722 | 0.7765 | 0.5028 | 0.5835 |
| momentum_reversal | bull | 0.0936 | 0.1344 | 0.6969 | 0.4917 | 0.5198 |
| fund_revenue_growth | bull | 0.0723 | 0.1031 | 0.7009 | 0.4475 | 0.5073 |
| fund_gross_margin | bull | 0.0488 | 0.1231 | 0.3964 | 0.4475 | 0.2869 |
| rsi_vol_combo | bull | 0.0495 | 0.1232 | 0.4021 | 0.3260 | 0.2666 |
| mom_x_lowvol_20_20 | bear | 0.1916 | 0.1297 | 1.4773 | 1.0000 | 1.4773 |
| momentum_reversal | bear | 0.1872 | 0.1378 | 1.3585 | 1.0000 | 1.3585 |
| fund_revenue_growth | bear | 0.1155 | 0.0906 | 1.2755 | 0.8125 | 1.1559 |
| bb_width_20 | bear | 0.1242 | 0.0993 | 1.2511 | 0.8125 | 1.1338 |
| trend_lowvol | bear | 0.1325 | 0.1133 | 1.1693 | 0.6875 | 0.9866 |
| fund_score | bear | 0.1144 | 0.1236 | 0.9257 | 0.8125 | 0.8389 |
| rsi_vol_combo | bear | 0.1150 | 0.1203 | 0.9559 | 0.6875 | 0.8065 |
| fund_gross_margin | bear | 0.1028 | 0.1137 | 0.9048 | 0.5625 | 0.7069 |
| fund_roe | bear | 0.0910 | 0.1162 | 0.7830 | 0.6875 | 0.6607 |
| fund_profit_growth | bear | 0.0897 | 0.1216 | 0.7375 | 0.7500 | 0.6453 |

### 交运

- **Neutral**: ['trend_lowvol'] (单因子IC=0.0989, 组合IC=0.0989)
  - weights: [1.0]
- **Bull**: ['low_downside', 'fund_roe'] (单因子IC=0.064, 组合IC=0.0729)
  - bull_weights: [0.5209, 0.4791]
- **Bear**: ['mom_x_lowvol_20_20'] (单因子IC=0.2713, 组合IC=0.2713)
  - bear_weights: [1.0]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| trend_lowvol | neutral | 0.0989 | 0.1578 | 0.6268 | 0.4184 | 0.4446 |
| momentum_reversal | neutral | 0.0846 | 0.1804 | 0.4691 | 0.3475 | 0.3160 |
| mom_x_lowvol_20_20 | neutral | 0.0889 | 0.1921 | 0.4626 | 0.3617 | 0.3150 |
| rsi_vol_combo | neutral | 0.0732 | 0.1536 | 0.4763 | 0.3191 | 0.3142 |
| fund_profit_growth | neutral | 0.0349 | 0.1603 | 0.2177 | 0.1206 | 0.1220 |
| low_downside | bull | 0.0722 | 0.1636 | 0.4416 | 0.3149 | 0.2903 |
| fund_roe | bull | 0.0557 | 0.1338 | 0.4166 | 0.2818 | 0.2670 |
| volatility | bull | 0.0715 | 0.1810 | 0.3950 | 0.3481 | 0.2663 |
| trend_lowvol | bull | 0.0577 | 0.1752 | 0.3297 | 0.2376 | 0.2040 |
| fund_score | bull | 0.0406 | 0.1381 | 0.2943 | 0.1934 | 0.1756 |
| fund_revenue_growth | bull | 0.0289 | 0.1067 | 0.2709 | 0.2707 | 0.1721 |
| mom_x_lowvol_20_20 | bear | 0.2713 | 0.1857 | 1.4612 | 0.9375 | 1.4156 |
| momentum_reversal | bear | 0.2674 | 0.1881 | 1.4212 | 0.9375 | 1.3768 |
| rsi_vol_combo | bear | 0.1911 | 0.1674 | 1.1417 | 0.7500 | 0.9990 |
| bb_width_20 | bear | 0.1712 | 0.1751 | 0.9781 | 0.7500 | 0.8559 |
| trend_lowvol | bear | 0.1404 | 0.1400 | 1.0030 | 0.6875 | 0.8463 |
| fund_revenue_growth | bear | 0.0477 | 0.1221 | 0.3902 | 0.3125 | 0.2560 |
| vol_confirm | bear | 0.0466 | 0.1572 | 0.2965 | 0.4375 | 0.2131 |

### 传媒

- **Neutral**: ['trend_lowvol'] (单因子IC=0.202, 组合IC=0.202)
  - weights: [1.0]
- **Bull**: ['trend_lowvol'] (单因子IC=0.1392, 组合IC=0.1392)
  - bull_weights: [1.0]
- **Bear**: ['mom_x_lowvol_20_20'] (单因子IC=0.3009, 组合IC=0.3009)
  - bear_weights: [1.0]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| trend_lowvol | neutral | 0.2020 | 0.1956 | 1.0328 | 0.7163 | 0.8863 |
| momentum_reversal | neutral | 0.1571 | 0.1982 | 0.7927 | 0.5603 | 0.6184 |
| mom_x_lowvol_20_20 | neutral | 0.1531 | 0.2036 | 0.7519 | 0.5887 | 0.5972 |
| rsi_vol_combo | neutral | 0.1322 | 0.1886 | 0.7008 | 0.4752 | 0.5169 |
| fund_profit_growth | neutral | 0.0568 | 0.1279 | 0.4443 | 0.3901 | 0.3088 |
| low_downside | neutral | 0.0969 | 0.2201 | 0.4401 | 0.3759 | 0.3028 |
| volatility | neutral | 0.1290 | 0.2760 | 0.4675 | 0.2482 | 0.2918 |
| fund_gross_margin | neutral | 0.0545 | 0.1785 | 0.3051 | 0.3333 | 0.2034 |
| fund_score | neutral | 0.0490 | 0.1810 | 0.2710 | 0.2766 | 0.1730 |
| fund_revenue_growth | neutral | 0.0377 | 0.1553 | 0.2430 | 0.1773 | 0.1430 |
| fund_roe | neutral | 0.0356 | 0.2078 | 0.1712 | 0.1489 | 0.0984 |
| trend_lowvol | bull | 0.1392 | 0.1865 | 0.7466 | 0.6464 | 0.6146 |
| fund_revenue_growth | bull | 0.0534 | 0.0994 | 0.5370 | 0.4144 | 0.3798 |
| momentum_reversal | bull | 0.0978 | 0.1706 | 0.5731 | 0.2928 | 0.3705 |
| mom_x_lowvol_20_20 | bull | 0.0930 | 0.1760 | 0.5287 | 0.2707 | 0.3359 |
| volatility | bull | 0.1039 | 0.2193 | 0.4735 | 0.3370 | 0.3165 |
| rsi_vol_combo | bull | 0.0670 | 0.1556 | 0.4305 | 0.2155 | 0.2616 |
| low_downside | bull | 0.0725 | 0.1960 | 0.3700 | 0.2044 | 0.2228 |
| fund_score | bull | 0.0209 | 0.1352 | 0.1548 | 0.1271 | 0.0872 |
| mom_x_lowvol_20_20 | bear | 0.3009 | 0.1370 | 2.1962 | 1.0000 | 2.1962 |
| momentum_reversal | bear | 0.2934 | 0.1424 | 2.0602 | 1.0000 | 2.0602 |
| rsi_vol_combo | bear | 0.1969 | 0.1523 | 1.2927 | 0.8750 | 1.2119 |
| fund_revenue_growth | bear | 0.1381 | 0.1292 | 1.0686 | 0.9375 | 1.0352 |
| bb_width_20 | bear | 0.1983 | 0.1822 | 1.0886 | 0.7500 | 0.9525 |
| trend_lowvol | bear | 0.1155 | 0.1345 | 0.8593 | 0.5625 | 0.6713 |
| fund_profit_growth | bear | 0.0525 | 0.1248 | 0.4209 | 0.3125 | 0.2762 |
| fund_score | bear | 0.0584 | 0.1356 | 0.4304 | 0.1250 | 0.2421 |

### 军工

- **Neutral**: ['mom_x_lowvol_20_20', 'momentum_reversal'] (单因子IC=0.1157, 组合IC=0.1161)
  - weights: [0.5061, 0.4939]
- **Bull**: ['volatility', 'low_downside'] (单因子IC=0.0487, 组合IC=0.0549)
  - bull_weights: [0.5493, 0.4507]
- **Bear**: ['fund_revenue_growth', 'fund_profit_growth'] (单因子IC=0.1711, 组合IC=0.2065)
  - bear_weights: [0.5711, 0.4289]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| mom_x_lowvol_20_20 | neutral | 0.1158 | 0.2126 | 0.5446 | 0.3475 | 0.3669 |
| momentum_reversal | neutral | 0.1157 | 0.2108 | 0.5487 | 0.3050 | 0.3580 |
| rsi_vol_combo | neutral | 0.0860 | 0.2084 | 0.4127 | 0.3050 | 0.2693 |
| trend_lowvol | neutral | 0.0815 | 0.2025 | 0.4024 | 0.2057 | 0.2426 |
| fund_revenue_growth | neutral | 0.0537 | 0.1496 | 0.3588 | 0.2057 | 0.2163 |
| fund_profit_growth | neutral | 0.0462 | 0.1569 | 0.2946 | 0.1773 | 0.1734 |
| volatility | bull | 0.0530 | 0.2142 | 0.2476 | 0.1602 | 0.1436 |
| low_downside | bull | 0.0444 | 0.2145 | 0.2071 | 0.1381 | 0.1178 |
| fund_revenue_growth | bear | 0.1867 | 0.1311 | 1.4236 | 0.8125 | 1.2901 |
| fund_profit_growth | bear | 0.1555 | 0.1404 | 1.1073 | 0.7500 | 0.9689 |
| momentum_reversal | bear | 0.1454 | 0.1572 | 0.9252 | 0.6875 | 0.7807 |
| bb_width_20 | bear | 0.1854 | 0.1849 | 1.0026 | 0.5000 | 0.7520 |
| mom_x_lowvol_20_20 | bear | 0.1494 | 0.1696 | 0.8810 | 0.6250 | 0.7158 |
| rsi_vol_combo | bear | 0.0843 | 0.1317 | 0.6402 | 0.5000 | 0.4801 |
| fund_score | bear | 0.0866 | 0.1767 | 0.4898 | 0.6250 | 0.3980 |
| fund_gross_margin | bear | 0.0504 | 0.1406 | 0.3587 | 0.2500 | 0.2242 |

### 农业

- **Neutral**: ['momentum_reversal', 'mom_x_lowvol_20_20', 'trend_lowvol'] (单因子IC=0.0693, 组合IC=0.0766)
  - weights: [0.3669, 0.3488, 0.2842]
- **Bull**: ['fund_profit_growth'] (单因子IC=0.0964, 组合IC=0.0964)
  - bull_weights: [1.0]
- **Bear**: ['fund_revenue_growth', 'wash_sale_score', 'momentum_reversal'] (单因子IC=0.127, 组合IC=0.1873)
  - bear_weights: [0.4215, 0.3551, 0.2234]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| momentum_reversal | neutral | 0.0751 | 0.1924 | 0.3902 | 0.3191 | 0.2574 |
| mom_x_lowvol_20_20 | neutral | 0.0745 | 0.1944 | 0.3834 | 0.2766 | 0.2447 |
| trend_lowvol | neutral | 0.0583 | 0.1826 | 0.3194 | 0.2482 | 0.1994 |
| rsi_vol_combo | neutral | 0.0561 | 0.2199 | 0.2550 | 0.2766 | 0.1628 |
| fund_revenue_growth | neutral | 0.0277 | 0.1118 | 0.2479 | 0.2199 | 0.1512 |
| fund_score | neutral | 0.0326 | 0.2369 | 0.1377 | 0.1915 | 0.0821 |
| fund_profit_growth | bull | 0.0964 | 0.1475 | 0.6535 | 0.4586 | 0.4766 |
| fund_revenue_growth | bull | 0.0380 | 0.1479 | 0.2568 | 0.3260 | 0.1703 |
| fund_revenue_growth | bear | 0.0868 | 0.0915 | 0.9485 | 0.6875 | 0.8003 |
| wash_sale_score | bear | 0.0978 | 0.1088 | 0.8989 | 0.5000 | 0.6741 |
| momentum_reversal | bear | 0.1965 | 0.3184 | 0.6171 | 0.3750 | 0.4242 |
| mom_x_lowvol_20_20 | bear | 0.1943 | 0.3198 | 0.6077 | 0.3750 | 0.4178 |
| rsi_vol_combo | bear | 0.1698 | 0.2976 | 0.5705 | 0.3750 | 0.3922 |
| trend_lowvol | bear | 0.0911 | 0.1805 | 0.5050 | 0.2500 | 0.3156 |

### 化工

- **Neutral**: ['trend_lowvol'] (单因子IC=0.0947, 组合IC=0.0947)
  - weights: [1.0]
- **Bull**: ['volatility'] (单因子IC=0.1334, 组合IC=0.1334)
  - bull_weights: [1.0]
- **Bear**: ['mom_x_lowvol_20_20'] (单因子IC=0.2167, 组合IC=0.2167)
  - bear_weights: [1.0]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| trend_lowvol | neutral | 0.0947 | 0.1260 | 0.7518 | 0.4894 | 0.5598 |
| momentum_reversal | neutral | 0.0764 | 0.1285 | 0.5943 | 0.4184 | 0.4215 |
| mom_x_lowvol_20_20 | neutral | 0.0799 | 0.1350 | 0.5916 | 0.4043 | 0.4153 |
| fund_revenue_growth | neutral | 0.0524 | 0.0956 | 0.5482 | 0.2624 | 0.3460 |
| rsi_vol_combo | neutral | 0.0579 | 0.1180 | 0.4905 | 0.3759 | 0.3374 |
| fund_profit_growth | neutral | 0.0570 | 0.1293 | 0.4412 | 0.2624 | 0.2785 |
| volatility | bull | 0.1334 | 0.1434 | 0.9304 | 0.5691 | 0.7299 |
| low_downside | bull | 0.1036 | 0.1234 | 0.8395 | 0.4807 | 0.6215 |
| mom_x_lowvol_20_20 | bull | 0.0765 | 0.1153 | 0.6635 | 0.4586 | 0.4839 |
| momentum_reversal | bull | 0.0699 | 0.1124 | 0.6221 | 0.4586 | 0.4537 |
| trend_lowvol | bull | 0.0843 | 0.1299 | 0.6486 | 0.3702 | 0.4443 |
| rsi_vol_combo | bull | 0.0504 | 0.1129 | 0.4466 | 0.3039 | 0.2912 |
| fund_roe | bull | 0.0516 | 0.1159 | 0.4454 | 0.2707 | 0.2830 |
| fund_score | bull | 0.0537 | 0.1229 | 0.4368 | 0.2818 | 0.2799 |
| fund_profit_growth | bull | 0.0320 | 0.0797 | 0.4019 | 0.2928 | 0.2598 |
| fund_revenue_growth | bull | 0.0229 | 0.0982 | 0.2336 | 0.1271 | 0.1316 |
| fund_gross_margin | bull | 0.0163 | 0.0936 | 0.1742 | 0.2265 | 0.1068 |
| mom_x_lowvol_20_20 | bear | 0.2167 | 0.1473 | 1.4713 | 0.8750 | 1.3794 |
| momentum_reversal | bear | 0.1977 | 0.1343 | 1.4722 | 0.8125 | 1.3342 |
| rsi_vol_combo | bear | 0.1184 | 0.0936 | 1.2640 | 0.8125 | 1.1455 |
| trend_lowvol | bear | 0.1256 | 0.1058 | 1.1872 | 0.7500 | 1.0388 |
| fund_revenue_growth | bear | 0.0904 | 0.0860 | 1.0503 | 0.8125 | 0.9518 |
| bb_width_20 | bear | 0.1240 | 0.1154 | 1.0749 | 0.6875 | 0.9069 |
| fund_score | bear | 0.0433 | 0.1541 | 0.2809 | 0.1875 | 0.1668 |

### 医药

- **Neutral**: ['trend_lowvol'] (单因子IC=0.1245, 组合IC=0.1245)
  - weights: [1.0]
- **Bull**: ['volatility', 'trend_lowvol'] (单因子IC=0.0994, 组合IC=0.1187)
  - bull_weights: [0.5091, 0.4909]
- **Bear**: ['mom_x_lowvol_20_20'] (单因子IC=0.2493, 组合IC=0.2493)
  - bear_weights: [1.0]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| trend_lowvol | neutral | 0.1245 | 0.1546 | 0.8054 | 0.4610 | 0.5883 |
| fund_profit_growth | neutral | 0.0512 | 0.1285 | 0.3982 | 0.1773 | 0.2344 |
| fund_revenue_growth | neutral | 0.0437 | 0.1199 | 0.3644 | 0.1489 | 0.2093 |
| momentum_reversal | neutral | 0.0596 | 0.1840 | 0.3236 | 0.2482 | 0.2020 |
| mom_x_lowvol_20_20 | neutral | 0.0543 | 0.1806 | 0.3008 | 0.2482 | 0.1877 |
| fund_score | neutral | 0.0270 | 0.1577 | 0.1711 | 0.1206 | 0.0959 |
| volatility | bull | 0.1137 | 0.1617 | 0.7034 | 0.4033 | 0.4935 |
| trend_lowvol | bull | 0.0850 | 0.1352 | 0.6287 | 0.5138 | 0.4759 |
| low_downside | bull | 0.0834 | 0.1334 | 0.6251 | 0.4144 | 0.4420 |
| fund_profit_growth | bull | 0.0477 | 0.0861 | 0.5540 | 0.3481 | 0.3734 |
| mom_x_lowvol_20_20 | bull | 0.0685 | 0.1300 | 0.5270 | 0.3591 | 0.3581 |
| momentum_reversal | bull | 0.0671 | 0.1309 | 0.5124 | 0.3591 | 0.3482 |
| rsi_vol_combo | bull | 0.0608 | 0.1254 | 0.4850 | 0.2597 | 0.3055 |
| fund_score | bull | 0.0496 | 0.1244 | 0.3987 | 0.3591 | 0.2709 |
| fund_revenue_growth | bull | 0.0429 | 0.1100 | 0.3898 | 0.2597 | 0.2455 |
| fund_roe | bull | 0.0267 | 0.1374 | 0.1943 | 0.1271 | 0.1095 |
| mom_x_lowvol_20_20 | bear | 0.2493 | 0.0932 | 2.6753 | 1.0000 | 2.6753 |
| momentum_reversal | bear | 0.2398 | 0.1018 | 2.3563 | 1.0000 | 2.3563 |
| rsi_vol_combo | bear | 0.1619 | 0.1124 | 1.4401 | 0.7500 | 1.2600 |
| trend_lowvol | bear | 0.1512 | 0.1240 | 1.2195 | 0.8750 | 1.1432 |
| bb_width_20 | bear | 0.1365 | 0.1315 | 1.0374 | 0.6250 | 0.8429 |
| fund_revenue_growth | bear | 0.0623 | 0.1134 | 0.5499 | 0.2500 | 0.3437 |
| fund_profit_growth | bear | 0.0707 | 0.1655 | 0.4272 | 0.1250 | 0.2403 |

### 半导体/光伏

- **Neutral**: ['mom_x_lowvol_20_20'] (单因子IC=0.1077, 组合IC=0.1077)
  - weights: [1.0]
- **Bull**: ['trend_lowvol', 'volatility', 'mom_x_lowvol_20_20'] (单因子IC=0.0922, 组合IC=0.1077)
  - bull_weights: [0.3863, 0.3253, 0.2884]
- **Bear**: ['fund_gross_margin', 'fund_revenue_growth', 'bb_width_20'] (单因子IC=0.2021, 组合IC=0.2681)
  - bear_weights: [0.4211, 0.4122, 0.1667]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| mom_x_lowvol_20_20 | neutral | 0.1077 | 0.1839 | 0.5856 | 0.5319 | 0.4485 |
| momentum_reversal | neutral | 0.1036 | 0.1965 | 0.5273 | 0.4610 | 0.3852 |
| trend_lowvol | neutral | 0.0686 | 0.1715 | 0.3997 | 0.2908 | 0.2580 |
| rsi_vol_combo | neutral | 0.0725 | 0.2018 | 0.3591 | 0.2766 | 0.2292 |
| fund_revenue_growth | neutral | 0.0603 | 0.1920 | 0.3140 | 0.2199 | 0.1915 |
| fund_profit_growth | neutral | 0.0511 | 0.1573 | 0.3251 | 0.1631 | 0.1891 |
| fund_score | neutral | 0.0315 | 0.1825 | 0.1729 | 0.1206 | 0.0969 |
| trend_lowvol | bull | 0.0982 | 0.1600 | 0.6134 | 0.4033 | 0.4304 |
| volatility | bull | 0.0882 | 0.1735 | 0.5085 | 0.4254 | 0.3624 |
| mom_x_lowvol_20_20 | bull | 0.0901 | 0.1845 | 0.4886 | 0.3149 | 0.3213 |
| momentum_reversal | bull | 0.0835 | 0.1810 | 0.4612 | 0.3149 | 0.3032 |
| fund_score | bull | 0.0787 | 0.1936 | 0.4066 | 0.2597 | 0.2561 |
| rsi_vol_combo | bull | 0.0643 | 0.1660 | 0.3872 | 0.2928 | 0.2503 |
| fund_roe | bull | 0.0660 | 0.1793 | 0.3679 | 0.2597 | 0.2317 |
| fund_gross_margin | bull | 0.0743 | 0.2155 | 0.3446 | 0.3370 | 0.2304 |
| fund_revenue_growth | bull | 0.0637 | 0.1996 | 0.3193 | 0.2597 | 0.2011 |
| fund_profit_growth | bull | 0.0386 | 0.1449 | 0.2664 | 0.1381 | 0.1516 |
| low_downside | bull | 0.0295 | 0.1421 | 0.2075 | 0.1823 | 0.1227 |
| fund_gross_margin | bear | 0.2087 | 0.0970 | 2.1519 | 0.9375 | 2.0847 |
| fund_revenue_growth | bear | 0.2570 | 0.1181 | 2.1767 | 0.8750 | 2.0406 |
| bb_width_20 | bear | 0.1406 | 0.1384 | 1.0158 | 0.6250 | 0.8253 |
| fund_profit_growth | bear | 0.1578 | 0.1607 | 0.9820 | 0.6250 | 0.7979 |
| mom_x_lowvol_20_20 | bear | 0.1741 | 0.1989 | 0.8751 | 0.5625 | 0.6837 |
| fund_score | bear | 0.1273 | 0.1672 | 0.7617 | 0.6250 | 0.6189 |
| fund_roe | bear | 0.0878 | 0.1400 | 0.6269 | 0.5000 | 0.4701 |
| momentum_reversal | bear | 0.1272 | 0.2277 | 0.5586 | 0.5625 | 0.4364 |

### 基建/地产/石油石化

- **Neutral**: ['trend_lowvol'] (单因子IC=0.1003, 组合IC=0.1003)
  - weights: [1.0]
- **Bull**: ['fund_profit_growth', 'volatility'] (单因子IC=0.103, 组合IC=0.1319)
  - bull_weights: [0.5309, 0.4691]
- **Bear**: ['trend_lowvol'] (单因子IC=0.2024, 组合IC=0.2024)
  - bear_weights: [1.0]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| trend_lowvol | neutral | 0.1003 | 0.1464 | 0.6851 | 0.4752 | 0.5053 |
| fund_profit_growth | neutral | 0.0587 | 0.1033 | 0.5677 | 0.4894 | 0.4227 |
| mom_x_lowvol_20_20 | neutral | 0.0739 | 0.1587 | 0.4661 | 0.3759 | 0.3206 |
| momentum_reversal | neutral | 0.0688 | 0.1547 | 0.4448 | 0.3759 | 0.3060 |
| rsi_vol_combo | neutral | 0.0442 | 0.1518 | 0.2913 | 0.2340 | 0.1797 |
| fund_roe | neutral | 0.0305 | 0.1637 | 0.1865 | 0.1915 | 0.1111 |
| fund_profit_growth | bull | 0.0814 | 0.0756 | 1.0770 | 0.7017 | 0.9164 |
| volatility | bull | 0.1245 | 0.1266 | 0.9836 | 0.6464 | 0.8097 |
| low_downside | bull | 0.0841 | 0.1196 | 0.7031 | 0.5801 | 0.5555 |
| fund_revenue_growth | bull | 0.0568 | 0.0802 | 0.7079 | 0.5470 | 0.5476 |
| trend_lowvol | bull | 0.0839 | 0.1103 | 0.7605 | 0.4254 | 0.5420 |
| fund_score | bull | 0.0782 | 0.1135 | 0.6885 | 0.5580 | 0.5364 |
| fund_roe | bull | 0.0588 | 0.1212 | 0.4852 | 0.3702 | 0.3324 |
| mom_x_lowvol_20_20 | bull | 0.0597 | 0.1499 | 0.3984 | 0.2818 | 0.2553 |
| momentum_reversal | bull | 0.0569 | 0.1541 | 0.3693 | 0.2265 | 0.2265 |
| rsi_vol_combo | bull | 0.0387 | 0.1500 | 0.2582 | 0.1381 | 0.1469 |
| trend_lowvol | bear | 0.2024 | 0.0969 | 2.0882 | 0.9375 | 2.0229 |
| fund_gross_margin | bear | 0.0696 | 0.0483 | 1.4424 | 0.7500 | 1.2621 |
| bb_width_20 | bear | 0.1841 | 0.1526 | 1.2064 | 0.8750 | 1.1310 |
| mom_x_lowvol_20_20 | bear | 0.2252 | 0.1910 | 1.1787 | 0.8125 | 1.0682 |
| momentum_reversal | bear | 0.2000 | 0.1868 | 1.0705 | 0.7500 | 0.9367 |
| fund_profit_growth | bear | 0.0969 | 0.0913 | 1.0612 | 0.7500 | 0.9285 |
| rsi_vol_combo | bear | 0.0919 | 0.1503 | 0.6118 | 0.3750 | 0.4206 |
| vol_confirm | bear | 0.0331 | 0.0898 | 0.3684 | 0.3750 | 0.2533 |

### 建材

- **Neutral**: ['trend_lowvol'] (单因子IC=0.0906, 组合IC=0.0906)
  - weights: [1.0]
- **Bull**: ['fund_gross_margin', 'volatility', 'fund_roe'] (单因子IC=0.108, 组合IC=0.1474)
  - bull_weights: [0.4094, 0.2984, 0.2923]
- **Bear**: ['bb_width_20', 'mom_x_lowvol_20_20'] (单因子IC=0.2562, 组合IC=0.2806)
  - bear_weights: [0.605, 0.395]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| trend_lowvol | neutral | 0.0906 | 0.1629 | 0.5559 | 0.4326 | 0.3982 |
| mom_x_lowvol_20_20 | neutral | 0.0698 | 0.1385 | 0.5039 | 0.4752 | 0.3717 |
| momentum_reversal | neutral | 0.0602 | 0.1382 | 0.4360 | 0.4043 | 0.3061 |
| fund_profit_growth | neutral | 0.0690 | 0.1434 | 0.4812 | 0.2624 | 0.3038 |
| fund_score | neutral | 0.0609 | 0.1782 | 0.3416 | 0.2482 | 0.2132 |
| fund_revenue_growth | neutral | 0.0471 | 0.1440 | 0.3273 | 0.2908 | 0.2112 |
| fund_gross_margin | neutral | 0.0292 | 0.1255 | 0.2325 | 0.1489 | 0.1336 |
| fund_roe | neutral | 0.0367 | 0.1680 | 0.2186 | 0.1064 | 0.1209 |
| fund_gross_margin | bull | 0.1031 | 0.1253 | 0.8232 | 0.5359 | 0.6322 |
| volatility | bull | 0.1114 | 0.1776 | 0.6271 | 0.4696 | 0.4608 |
| fund_roe | bull | 0.1096 | 0.1676 | 0.6536 | 0.3812 | 0.4513 |
| fund_score | bull | 0.0958 | 0.1634 | 0.5866 | 0.3039 | 0.3824 |
| low_downside | bull | 0.0841 | 0.1596 | 0.5270 | 0.4365 | 0.3785 |
| trend_lowvol | bull | 0.0601 | 0.1374 | 0.4377 | 0.3260 | 0.2902 |
| fund_revenue_growth | bull | 0.0475 | 0.1218 | 0.3898 | 0.2486 | 0.2434 |
| fund_profit_growth | bull | 0.0478 | 0.1291 | 0.3699 | 0.1934 | 0.2207 |
| mom_x_lowvol_20_20 | bull | 0.0489 | 0.1398 | 0.3501 | 0.2597 | 0.2205 |
| momentum_reversal | bull | 0.0424 | 0.1414 | 0.3001 | 0.2155 | 0.1824 |
| rsi_vol_combo | bull | 0.0345 | 0.1407 | 0.2449 | 0.1823 | 0.1448 |
| bb_width_20 | bear | 0.2754 | 0.1256 | 2.1918 | 0.9375 | 2.1233 |
| mom_x_lowvol_20_20 | bear | 0.2371 | 0.1603 | 1.4788 | 0.8750 | 1.3863 |
| fund_revenue_growth | bear | 0.1058 | 0.0969 | 1.0914 | 0.6875 | 0.9209 |
| momentum_reversal | bear | 0.1810 | 0.1884 | 0.9609 | 0.6250 | 0.7807 |
| fund_profit_growth | bear | 0.0979 | 0.1240 | 0.7898 | 0.5625 | 0.6170 |
| fund_score | bear | 0.1080 | 0.1311 | 0.8237 | 0.4375 | 0.5921 |
| fund_gross_margin | bear | 0.0603 | 0.0922 | 0.6541 | 0.3750 | 0.4497 |

### 新能源车/风电

- **Neutral**: ['fund_revenue_growth', 'fund_profit_growth', 'trend_lowvol'] (单因子IC=0.0677, 组合IC=0.0964)
  - weights: [0.411, 0.3589, 0.2301]
- **Bull**: ['trend_lowvol', 'volatility'] (单因子IC=0.0978, 组合IC=0.1082)
  - bull_weights: [0.5092, 0.4908]
- **Bear**: ['fund_revenue_growth', 'bb_width_20', 'fund_gross_margin'] (单因子IC=0.1277, 组合IC=0.1765)
  - bear_weights: [0.3781, 0.3301, 0.2918]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| fund_revenue_growth | neutral | 0.0662 | 0.0898 | 0.7368 | 0.5887 | 0.5853 |
| fund_profit_growth | neutral | 0.0712 | 0.0988 | 0.7207 | 0.4184 | 0.5111 |
| trend_lowvol | neutral | 0.0658 | 0.1410 | 0.4667 | 0.4043 | 0.3277 |
| mom_x_lowvol_20_20 | neutral | 0.0515 | 0.1353 | 0.3806 | 0.1915 | 0.2267 |
| momentum_reversal | neutral | 0.0449 | 0.1315 | 0.3414 | 0.1773 | 0.2010 |
| fund_gross_margin | neutral | 0.0231 | 0.1758 | 0.1315 | 0.1064 | 0.0727 |
| trend_lowvol | bull | 0.0925 | 0.1588 | 0.5822 | 0.4807 | 0.4310 |
| volatility | bull | 0.1032 | 0.1633 | 0.6319 | 0.3149 | 0.4155 |
| low_downside | bull | 0.0610 | 0.1220 | 0.4997 | 0.3260 | 0.3313 |
| mom_x_lowvol_20_20 | bull | 0.0773 | 0.1596 | 0.4844 | 0.2707 | 0.3078 |
| momentum_reversal | bull | 0.0738 | 0.1592 | 0.4634 | 0.2818 | 0.2970 |
| rsi_vol_combo | bull | 0.0589 | 0.1375 | 0.4286 | 0.2486 | 0.2676 |
| fund_profit_growth | bull | 0.0400 | 0.1047 | 0.3815 | 0.3702 | 0.2614 |
| fund_revenue_growth | bull | 0.0346 | 0.1107 | 0.3126 | 0.1271 | 0.1762 |
| fund_score | bull | 0.0186 | 0.1658 | 0.1119 | 0.1381 | 0.0637 |
| fund_revenue_growth | bear | 0.1033 | 0.0966 | 1.0689 | 1.0000 | 1.0689 |
| bb_width_20 | bear | 0.1393 | 0.1353 | 1.0297 | 0.8125 | 0.9331 |
| fund_gross_margin | bear | 0.1406 | 0.1278 | 1.1000 | 0.5000 | 0.8250 |
| fund_profit_growth | bear | 0.1040 | 0.0990 | 1.0515 | 0.5000 | 0.7886 |
| fund_score | bear | 0.1243 | 0.1537 | 0.8090 | 0.3750 | 0.5562 |
| relative_strength | bear | 0.0558 | 0.0873 | 0.6394 | 0.5625 | 0.4995 |
| mom_x_lowvol_20_20 | bear | 0.0884 | 0.1705 | 0.5187 | 0.3125 | 0.3404 |
| fund_roe | bear | 0.0565 | 0.1849 | 0.3058 | 0.3750 | 0.2102 |

### 有色/钢铁/煤炭

- **Neutral**: ['trend_lowvol', 'fund_revenue_growth', 'mom_x_lowvol_20_20'] (单因子IC=0.0582, 组合IC=0.0791)
  - weights: [0.5066, 0.252, 0.2414]
- **Bull**: ['fund_gross_margin', 'fund_roe', 'fund_score'] (单因子IC=0.1354, 组合IC=0.1629)
  - bull_weights: [0.4019, 0.328, 0.2702]
- **Bear**: ['fund_revenue_growth'] (单因子IC=0.2751, 组合IC=0.2751)
  - bear_weights: [1.0]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| trend_lowvol | neutral | 0.0763 | 0.1686 | 0.4528 | 0.2908 | 0.2922 |
| fund_revenue_growth | neutral | 0.0528 | 0.2292 | 0.2303 | 0.2624 | 0.1454 |
| mom_x_lowvol_20_20 | neutral | 0.0455 | 0.1949 | 0.2337 | 0.1915 | 0.1392 |
| fund_profit_growth | neutral | 0.0442 | 0.1975 | 0.2237 | 0.1489 | 0.1285 |
| momentum_reversal | neutral | 0.0331 | 0.1988 | 0.1667 | 0.1489 | 0.0957 |
| fund_gross_margin | bull | 0.0936 | 0.0979 | 0.9565 | 0.7459 | 0.8349 |
| fund_roe | bull | 0.1519 | 0.1786 | 0.8505 | 0.6022 | 0.6814 |
| fund_score | bull | 0.1607 | 0.2167 | 0.7416 | 0.5138 | 0.5613 |
| fund_profit_growth | bull | 0.0990 | 0.1507 | 0.6568 | 0.5138 | 0.4972 |
| fund_revenue_growth | bull | 0.1358 | 0.2191 | 0.6198 | 0.4254 | 0.4417 |
| mom_x_lowvol_20_20 | bull | 0.0664 | 0.1970 | 0.3373 | 0.3481 | 0.2273 |
| momentum_reversal | bull | 0.0606 | 0.1985 | 0.3053 | 0.2928 | 0.1974 |
| trend_lowvol | bull | 0.0468 | 0.1788 | 0.2615 | 0.2486 | 0.1633 |
| rsi_vol_combo | bull | 0.0451 | 0.1865 | 0.2420 | 0.3149 | 0.1591 |
| volatility | bull | 0.0509 | 0.1936 | 0.2631 | 0.1160 | 0.1468 |
| low_downside | bull | 0.0340 | 0.1546 | 0.2202 | 0.2486 | 0.1375 |
| fund_revenue_growth | bear | 0.2751 | 0.1272 | 2.1628 | 1.0000 | 2.1628 |
| fund_score | bear | 0.2275 | 0.1751 | 1.2991 | 0.8125 | 1.1773 |
| bb_width_20 | bear | 0.1612 | 0.1656 | 0.9732 | 0.5625 | 0.7603 |
| fund_profit_growth | bear | 0.1530 | 0.1828 | 0.8367 | 0.3125 | 0.5491 |
| mom_x_lowvol_20_20 | bear | 0.1473 | 0.2127 | 0.6925 | 0.3125 | 0.4545 |
| fund_roe | bear | 0.0973 | 0.2082 | 0.4671 | 0.5000 | 0.3503 |
| momentum_reversal | bear | 0.1102 | 0.2030 | 0.5429 | 0.1875 | 0.3224 |

### 消费

- **Neutral**: ['trend_lowvol'] (单因子IC=0.1062, 组合IC=0.1062)
  - weights: [1.0]
- **Bull**: ['volatility'] (单因子IC=0.1273, 组合IC=0.1273)
  - bull_weights: [1.0]
- **Bear**: ['bb_width_20', 'mom_x_lowvol_20_20', 'momentum_reversal'] (单因子IC=0.188, 组合IC=0.22)
  - bear_weights: [0.3668, 0.347, 0.2862]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| trend_lowvol | neutral | 0.1062 | 0.1343 | 0.7905 | 0.3901 | 0.5494 |
| fund_profit_growth | neutral | 0.0599 | 0.1028 | 0.5822 | 0.5603 | 0.4542 |
| mom_x_lowvol_20_20 | neutral | 0.0608 | 0.1627 | 0.3739 | 0.2482 | 0.2334 |
| momentum_reversal | neutral | 0.0595 | 0.1617 | 0.3681 | 0.2482 | 0.2297 |
| rsi_vol_combo | neutral | 0.0442 | 0.1434 | 0.3084 | 0.1631 | 0.1794 |
| volatility | neutral | 0.0340 | 0.1702 | 0.1995 | 0.1206 | 0.1118 |
| volatility | bull | 0.1273 | 0.1252 | 1.0169 | 0.6906 | 0.8596 |
| trend_lowvol | bull | 0.0873 | 0.1011 | 0.8638 | 0.6464 | 0.7111 |
| low_downside | bull | 0.0943 | 0.1143 | 0.8252 | 0.5028 | 0.6201 |
| mom_x_lowvol_20_20 | bull | 0.0563 | 0.1132 | 0.4975 | 0.3481 | 0.3353 |
| momentum_reversal | bull | 0.0526 | 0.1160 | 0.4538 | 0.2707 | 0.2883 |
| rsi_vol_combo | bull | 0.0434 | 0.1204 | 0.3607 | 0.2044 | 0.2172 |
| fund_profit_growth | bull | 0.0254 | 0.0703 | 0.3610 | 0.1713 | 0.2114 |
| bb_width_20 | bear | 0.1541 | 0.1205 | 1.2780 | 0.9375 | 1.2380 |
| mom_x_lowvol_20_20 | bear | 0.2177 | 0.1800 | 1.2093 | 0.9375 | 1.1715 |
| momentum_reversal | bear | 0.1924 | 0.1867 | 1.0305 | 0.8750 | 0.9661 |
| fund_revenue_growth | bear | 0.0875 | 0.1093 | 0.8006 | 0.4375 | 0.5754 |
| trend_lowvol | bear | 0.0854 | 0.1218 | 0.7011 | 0.3750 | 0.4820 |
| rsi_vol_combo | bear | 0.1077 | 0.1650 | 0.6528 | 0.4375 | 0.4692 |
| fund_profit_growth | bear | 0.0691 | 0.1287 | 0.5369 | 0.2500 | 0.3356 |

### 环保/公用

- **Neutral**: ['fund_profit_growth', 'trend_lowvol', 'rsi_vol_combo'] (单因子IC=0.0703, 组合IC=0.1084)
  - weights: [0.3818, 0.3588, 0.2594]
- **Bull**: ['fund_profit_growth', 'fund_score', 'volatility'] (单因子IC=0.0694, 组合IC=0.079)
  - bull_weights: [0.4001, 0.3235, 0.2763]
- **Bear**: ['bb_width_20', 'mom_x_lowvol_20_20'] (单因子IC=0.2281, 组合IC=0.2507)
  - bear_weights: [0.6481, 0.3519]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| fund_profit_growth | neutral | 0.0641 | 0.0939 | 0.6828 | 0.5461 | 0.5279 |
| trend_lowvol | neutral | 0.0772 | 0.1148 | 0.6724 | 0.4752 | 0.4960 |
| rsi_vol_combo | neutral | 0.0697 | 0.1323 | 0.5267 | 0.3617 | 0.3586 |
| momentum_reversal | neutral | 0.0782 | 0.1515 | 0.5160 | 0.3759 | 0.3550 |
| mom_x_lowvol_20_20 | neutral | 0.0833 | 0.1620 | 0.5141 | 0.3475 | 0.3464 |
| fund_revenue_growth | neutral | 0.0256 | 0.0705 | 0.3634 | 0.2766 | 0.2320 |
| fund_score | neutral | 0.0415 | 0.1506 | 0.2754 | 0.2908 | 0.1778 |
| fund_gross_margin | neutral | 0.0262 | 0.0907 | 0.2888 | 0.1631 | 0.1679 |
| fund_profit_growth | bull | 0.0602 | 0.0808 | 0.7452 | 0.5470 | 0.5764 |
| fund_score | bull | 0.0650 | 0.1032 | 0.6295 | 0.4807 | 0.4660 |
| volatility | bull | 0.0830 | 0.1474 | 0.5628 | 0.4144 | 0.3980 |
| fund_gross_margin | bull | 0.0329 | 0.0678 | 0.4847 | 0.3370 | 0.3240 |
| mom_x_lowvol_20_20 | bull | 0.0526 | 0.1323 | 0.3973 | 0.3591 | 0.2700 |
| fund_revenue_growth | bull | 0.0266 | 0.0739 | 0.3604 | 0.3591 | 0.2449 |
| low_downside | bull | 0.0528 | 0.1322 | 0.3992 | 0.2265 | 0.2448 |
| momentum_reversal | bull | 0.0488 | 0.1294 | 0.3768 | 0.2928 | 0.2436 |
| fund_roe | bull | 0.0442 | 0.1266 | 0.3490 | 0.1823 | 0.2063 |
| rsi_vol_combo | bull | 0.0387 | 0.1263 | 0.3066 | 0.2376 | 0.1897 |
| trend_lowvol | bull | 0.0303 | 0.1289 | 0.2349 | 0.2265 | 0.1441 |
| bb_width_20 | bear | 0.2370 | 0.1196 | 1.9812 | 0.8750 | 1.8574 |
| mom_x_lowvol_20_20 | bear | 0.2192 | 0.1902 | 1.1524 | 0.7500 | 1.0084 |
| fund_revenue_growth | bear | 0.0607 | 0.0590 | 1.0296 | 0.6250 | 0.8365 |
| momentum_reversal | bear | 0.1847 | 0.1890 | 0.9775 | 0.5625 | 0.7637 |
| fund_profit_growth | bear | 0.0585 | 0.0820 | 0.7128 | 0.6875 | 0.6015 |
| trend_lowvol | bear | 0.0842 | 0.1161 | 0.7252 | 0.4375 | 0.5212 |
| rsi_vol_combo | bear | 0.0591 | 0.1642 | 0.3598 | 0.1875 | 0.2136 |
| fund_gross_margin | bear | 0.0308 | 0.0951 | 0.3238 | 0.2500 | 0.2024 |

### 电力设备

- **Neutral**: ['trend_lowvol', 'mom_x_lowvol_20_20'] (单因子IC=0.1054, 组合IC=0.1246)
  - weights: [0.5412, 0.4588]
- **Bull**: ['fund_score', 'low_downside'] (单因子IC=0.0788, 组合IC=0.0913)
  - bull_weights: [0.5507, 0.4493]
- **Bear**: ['fund_revenue_growth', 'bb_width_20', 'mom_x_lowvol_20_20'] (单因子IC=0.1734, 组合IC=0.2084)
  - bear_weights: [0.3466, 0.3289, 0.3245]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| trend_lowvol | neutral | 0.1102 | 0.1664 | 0.6621 | 0.5319 | 0.5071 |
| mom_x_lowvol_20_20 | neutral | 0.1006 | 0.1692 | 0.5942 | 0.4468 | 0.4299 |
| fund_profit_growth | neutral | 0.0669 | 0.1235 | 0.5421 | 0.4043 | 0.3806 |
| momentum_reversal | neutral | 0.0859 | 0.1606 | 0.5348 | 0.3759 | 0.3679 |
| rsi_vol_combo | neutral | 0.0467 | 0.1423 | 0.3280 | 0.2908 | 0.2117 |
| fund_revenue_growth | neutral | 0.0446 | 0.1295 | 0.3444 | 0.1064 | 0.1905 |
| fund_score | neutral | 0.0293 | 0.1281 | 0.2290 | 0.2199 | 0.1397 |
| fund_score | bull | 0.0682 | 0.1168 | 0.5840 | 0.4475 | 0.4227 |
| low_downside | bull | 0.0894 | 0.1891 | 0.4729 | 0.4586 | 0.3449 |
| fund_revenue_growth | bull | 0.0542 | 0.1138 | 0.4757 | 0.4033 | 0.3338 |
| volatility | bull | 0.0900 | 0.2062 | 0.4364 | 0.4254 | 0.3110 |
| fund_profit_growth | bull | 0.0384 | 0.0901 | 0.4265 | 0.3812 | 0.2946 |
| fund_roe | bull | 0.0493 | 0.1141 | 0.4317 | 0.2707 | 0.2743 |
| trend_lowvol | bull | 0.0725 | 0.1966 | 0.3685 | 0.2597 | 0.2321 |
| mom_x_lowvol_20_20 | bull | 0.0548 | 0.1658 | 0.3308 | 0.1713 | 0.1937 |
| fund_gross_margin | bull | 0.0156 | 0.0701 | 0.2225 | 0.2818 | 0.1426 |
| fund_revenue_growth | bear | 0.1372 | 0.1055 | 1.3014 | 0.8125 | 1.1794 |
| bb_width_20 | bear | 0.1744 | 0.1412 | 1.2349 | 0.8125 | 1.1191 |
| mom_x_lowvol_20_20 | bear | 0.2085 | 0.1770 | 1.1780 | 0.8750 | 1.1044 |
| momentum_reversal | bear | 0.1833 | 0.1989 | 0.9216 | 0.5625 | 0.7200 |
| rsi_vol_combo | bear | 0.1156 | 0.1864 | 0.6203 | 0.4375 | 0.4458 |
| fund_profit_growth | bear | 0.0627 | 0.1487 | 0.4214 | 0.1250 | 0.2370 |
| fund_score | bear | 0.0417 | 0.1244 | 0.3349 | 0.3125 | 0.2198 |

### 电子

- **Neutral**: ['fund_profit_growth', 'fund_revenue_growth', 'trend_lowvol'] (单因子IC=0.0648, 组合IC=0.0931)
  - weights: [0.3955, 0.3206, 0.2839]
- **Bull**: ['low_downside', 'volatility'] (单因子IC=0.1071, 组合IC=0.118)
  - bull_weights: [0.5433, 0.4567]
- **Bear**: ['fund_revenue_growth', 'fund_score', 'bb_width_20'] (单因子IC=0.1465, 组合IC=0.2046)
  - bear_weights: [0.5455, 0.2449, 0.2096]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| fund_profit_growth | neutral | 0.0668 | 0.1310 | 0.5098 | 0.4894 | 0.3796 |
| fund_revenue_growth | neutral | 0.0605 | 0.1325 | 0.4567 | 0.3475 | 0.3077 |
| trend_lowvol | neutral | 0.0672 | 0.1592 | 0.4222 | 0.2908 | 0.2725 |
| fund_score | neutral | 0.0602 | 0.1771 | 0.3399 | 0.3333 | 0.2266 |
| mom_x_lowvol_20_20 | neutral | 0.0566 | 0.1496 | 0.3780 | 0.1915 | 0.2252 |
| fund_gross_margin | neutral | 0.0554 | 0.1535 | 0.3610 | 0.1631 | 0.2099 |
| momentum_reversal | neutral | 0.0517 | 0.1539 | 0.3358 | 0.1631 | 0.1953 |
| low_downside | neutral | 0.0318 | 0.1246 | 0.2555 | 0.1631 | 0.1486 |
| fund_roe | neutral | 0.0317 | 0.1571 | 0.2016 | 0.2057 | 0.1215 |
| low_downside | bull | 0.1039 | 0.1215 | 0.8549 | 0.5801 | 0.6754 |
| volatility | bull | 0.1104 | 0.1440 | 0.7667 | 0.4807 | 0.5676 |
| fund_profit_growth | bull | 0.0637 | 0.1153 | 0.5527 | 0.5028 | 0.4153 |
| trend_lowvol | bull | 0.0789 | 0.1371 | 0.5755 | 0.4254 | 0.4101 |
| fund_revenue_growth | bull | 0.0573 | 0.1405 | 0.4079 | 0.4807 | 0.3019 |
| fund_score | bull | 0.0707 | 0.1807 | 0.3916 | 0.4807 | 0.2899 |
| fund_gross_margin | bull | 0.0435 | 0.1052 | 0.4140 | 0.2155 | 0.2516 |
| fund_roe | bull | 0.0508 | 0.1669 | 0.3046 | 0.3370 | 0.2036 |
| mom_x_lowvol_20_20 | bull | 0.0309 | 0.1143 | 0.2708 | 0.1602 | 0.1571 |
| fund_revenue_growth | bear | 0.1694 | 0.0743 | 2.2786 | 0.8125 | 2.0650 |
| fund_score | bear | 0.1679 | 0.1641 | 1.0231 | 0.8125 | 0.9272 |
| bb_width_20 | bear | 0.1024 | 0.1048 | 0.9766 | 0.6250 | 0.7935 |
| fund_roe | bear | 0.1206 | 0.1443 | 0.8357 | 0.7500 | 0.7313 |
| fund_profit_growth | bear | 0.1156 | 0.1522 | 0.7595 | 0.6250 | 0.6171 |
| mom_x_lowvol_20_20 | bear | 0.1164 | 0.1591 | 0.7320 | 0.6250 | 0.5947 |
| fund_gross_margin | bear | 0.0483 | 0.1024 | 0.4710 | 0.5000 | 0.3533 |

### 自动化/制造

- **Neutral**: ['trend_lowvol', 'mom_x_lowvol_20_20'] (单因子IC=0.0623, 组合IC=0.0735)
  - weights: [0.5027, 0.4973]
- **Bull**: ['volatility'] (单因子IC=0.1487, 组合IC=0.1487)
  - bull_weights: [1.0]
- **Bear**: ['fund_gross_margin', 'bb_width_20', 'fund_revenue_growth'] (单因子IC=0.113, 组合IC=0.1381)
  - bear_weights: [0.373, 0.3642, 0.2628]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| trend_lowvol | neutral | 0.0612 | 0.1407 | 0.4352 | 0.3617 | 0.2963 |
| mom_x_lowvol_20_20 | neutral | 0.0634 | 0.1519 | 0.4174 | 0.4043 | 0.2931 |
| momentum_reversal | neutral | 0.0610 | 0.1477 | 0.4128 | 0.3759 | 0.2840 |
| fund_gross_margin | neutral | 0.0201 | 0.0600 | 0.3359 | 0.3901 | 0.2334 |
| fund_revenue_growth | neutral | 0.0376 | 0.0965 | 0.3898 | 0.1348 | 0.2211 |
| rsi_vol_combo | neutral | 0.0486 | 0.1397 | 0.3479 | 0.2199 | 0.2122 |
| volatility | bull | 0.1487 | 0.1571 | 0.9461 | 0.6022 | 0.7579 |
| low_downside | bull | 0.1137 | 0.1311 | 0.8677 | 0.6575 | 0.7191 |
| fund_profit_growth | bull | 0.0549 | 0.0752 | 0.7297 | 0.6243 | 0.5926 |
| fund_gross_margin | bull | 0.0450 | 0.0629 | 0.7160 | 0.5912 | 0.5696 |
| fund_revenue_growth | bull | 0.0489 | 0.0807 | 0.6053 | 0.4917 | 0.4515 |
| mom_x_lowvol_20_20 | bull | 0.0805 | 0.1299 | 0.6198 | 0.4365 | 0.4452 |
| fund_score | bull | 0.0669 | 0.1063 | 0.6295 | 0.3370 | 0.4208 |
| trend_lowvol | bull | 0.0850 | 0.1463 | 0.5811 | 0.3812 | 0.4013 |
| momentum_reversal | bull | 0.0684 | 0.1249 | 0.5478 | 0.3812 | 0.3783 |
| fund_roe | bull | 0.0322 | 0.0960 | 0.3356 | 0.1713 | 0.1965 |
| rsi_vol_combo | bull | 0.0399 | 0.1172 | 0.3403 | 0.1050 | 0.1880 |
| fund_gross_margin | bear | 0.0838 | 0.0548 | 1.5302 | 0.7500 | 1.3389 |
| bb_width_20 | bear | 0.1585 | 0.1174 | 1.3497 | 0.9375 | 1.3075 |
| fund_revenue_growth | bear | 0.0968 | 0.0898 | 1.0784 | 0.7500 | 0.9436 |
| mom_x_lowvol_20_20 | bear | 0.1349 | 0.1828 | 0.7375 | 0.5625 | 0.5762 |
| momentum_reversal | bear | 0.0849 | 0.1817 | 0.4672 | 0.1250 | 0.2628 |

### 通信/计算机

- **Neutral**: ['fund_profit_growth', 'fund_revenue_growth', 'fund_score'] (单因子IC=0.0643, 组合IC=0.0772)
  - weights: [0.3892, 0.3468, 0.2641]
- **Bull**: ['low_downside'] (单因子IC=0.0988, 组合IC=0.0988)
  - bull_weights: [1.0]
- **Bear**: ['fund_revenue_growth', 'fund_score'] (单因子IC=0.1512, 组合IC=0.1746)
  - bear_weights: [0.5897, 0.4103]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| fund_profit_growth | neutral | 0.0679 | 0.1291 | 0.5258 | 0.3759 | 0.3618 |
| fund_revenue_growth | neutral | 0.0590 | 0.1156 | 0.5107 | 0.2624 | 0.3223 |
| fund_score | neutral | 0.0661 | 0.1737 | 0.3803 | 0.2908 | 0.2455 |
| trend_lowvol | neutral | 0.0561 | 0.1677 | 0.3346 | 0.3333 | 0.2231 |
| fund_roe | neutral | 0.0391 | 0.1654 | 0.2362 | 0.2482 | 0.1474 |
| low_downside | neutral | 0.0247 | 0.1539 | 0.1608 | 0.1206 | 0.0901 |
| low_downside | bull | 0.0988 | 0.1659 | 0.5955 | 0.4475 | 0.4310 |
| fund_profit_growth | bull | 0.0485 | 0.0893 | 0.5434 | 0.4807 | 0.4023 |
| volatility | bull | 0.0949 | 0.1757 | 0.5399 | 0.3812 | 0.3729 |
| trend_lowvol | bull | 0.0660 | 0.1421 | 0.4648 | 0.3260 | 0.3082 |
| fund_revenue_growth | bull | 0.0486 | 0.1159 | 0.4195 | 0.3923 | 0.2920 |
| fund_score | bull | 0.0539 | 0.1345 | 0.4008 | 0.3481 | 0.2702 |
| fund_roe | bull | 0.0423 | 0.1302 | 0.3246 | 0.3370 | 0.2170 |
| mom_x_lowvol_20_20 | bull | 0.0404 | 0.1386 | 0.2916 | 0.1271 | 0.1643 |
| fund_revenue_growth | bear | 0.1394 | 0.1019 | 1.3678 | 0.8125 | 1.2395 |
| fund_score | bear | 0.1629 | 0.1712 | 0.9517 | 0.8125 | 0.8624 |
| fund_profit_growth | bear | 0.1170 | 0.1343 | 0.8708 | 0.6875 | 0.7347 |
| bb_width_20 | bear | 0.1179 | 0.1372 | 0.8592 | 0.5625 | 0.6713 |
| fund_roe | bear | 0.1133 | 0.1550 | 0.7308 | 0.5000 | 0.5481 |
| mom_x_lowvol_20_20 | bear | 0.0740 | 0.1724 | 0.4293 | 0.2500 | 0.2683 |

### 金融

- **Neutral**: ['fund_roe'] (单因子IC=0.162, 组合IC=0.162)
  - weights: [1.0]
- **Bull**: ['low_downside', 'fund_score', 'volatility'] (单因子IC=0.1395, 组合IC=0.1876)
  - bull_weights: [0.4541, 0.3222, 0.2237]
- **Bear**: ['fund_gross_margin', 'fund_profit_growth'] (单因子IC=0.1538, 组合IC=0.1887)
  - bear_weights: [0.5357, 0.4643]

| 因子 | 状态 | IC_mean | IC_std | IR | stability | combined_IR |
|------|------|---------|--------|-----|-----------|-------------|
| fund_roe | neutral | 0.1620 | 0.3003 | 0.5395 | 0.5319 | 0.4132 |
| volatility | neutral | 0.1115 | 0.2946 | 0.3785 | 0.3759 | 0.2604 |
| low_downside | neutral | 0.0889 | 0.3057 | 0.2908 | 0.2908 | 0.1877 |
| trend_lowvol | neutral | 0.0792 | 0.2678 | 0.2957 | 0.1773 | 0.1741 |
| fund_score | neutral | 0.0654 | 0.2857 | 0.2289 | 0.2199 | 0.1396 |
| low_downside | bull | 0.1787 | 0.2419 | 0.7389 | 0.5691 | 0.5797 |
| fund_score | bull | 0.1150 | 0.1962 | 0.5863 | 0.4033 | 0.4114 |
| volatility | bull | 0.1249 | 0.2876 | 0.4342 | 0.3149 | 0.2855 |
| vol_confirm | bull | 0.0716 | 0.2125 | 0.3371 | 0.3149 | 0.2217 |
| fund_profit_growth | bull | 0.1044 | 0.3043 | 0.3431 | 0.1050 | 0.1896 |
| fund_revenue_growth | bull | 0.0736 | 0.2278 | 0.3230 | 0.1713 | 0.1891 |
| trend_lowvol | bull | 0.0613 | 0.2708 | 0.2264 | 0.2486 | 0.1413 |
| fund_roe | bull | 0.0752 | 0.3107 | 0.2421 | 0.1602 | 0.1404 |
| relative_strength | bull | 0.0660 | 0.3090 | 0.2137 | 0.1713 | 0.1251 |
| fund_gross_margin | bear | 0.1650 | 0.1315 | 1.2542 | 0.7500 | 1.0974 |
| fund_profit_growth | bear | 0.1426 | 0.1265 | 1.1273 | 0.6875 | 0.9511 |
| fund_revenue_growth | bear | 0.1104 | 0.1315 | 0.8395 | 0.7500 | 0.7346 |
| mom_x_lowvol_20_20 | bear | 0.2157 | 0.3122 | 0.6910 | 0.5625 | 0.5399 |
| momentum_reversal | bear | 0.2017 | 0.3146 | 0.6413 | 0.5625 | 0.5010 |
| bb_width_20 | bear | 0.1599 | 0.2661 | 0.6010 | 0.5625 | 0.4695 |
| rsi_vol_combo | bear | 0.1100 | 0.2333 | 0.4713 | 0.5000 | 0.3535 |
| fund_score | bear | 0.1078 | 0.2708 | 0.3980 | 0.6250 | 0.3233 |
| fund_roe | bear | 0.1158 | 0.2969 | 0.3902 | 0.5625 | 0.3048 |
| trend_lowvol | bear | 0.0997 | 0.2714 | 0.3675 | 0.1250 | 0.2067 |

