#!/usr/bin/env python3
"""2026-09-10 阶段3信息源探针: SUE(一致预期) + 内部人增持 数据可得性
纯HTTP只读, 不写任何数据文件。结论口径:
- 增持: 东财 RPT_SHARE_HOLDER_INCREASE DIRECTION="增持" (现有减持同接口另一半)
- 盈利预测: 候选reportName逐个试, 看返回列/深度/覆盖率
- 评级: 候选reportName逐个试
"""
import requests
import pandas as pd

UA = {'User-Agent': 'Mozilla/5.0'}


def hit(report_name, flt='', columns='ALL', page_size=500, page=1):
    url = 'https://datacenter-web.eastmoney.com/api/data/v1/get'
    params = {
        'sortColumns': 'EITIME', 'sortTypes': '-1',
        'pageSize': str(page_size), 'pageNumber': str(page),
        'reportName': report_name, 'columns': columns,
        'source': 'WEB', 'client': 'WEB', 'filter': flt,
    }
    r = requests.get(url, params=params, timeout=30, headers=UA)
    j = r.json()
    if not j.get('success'):
        return None, f"success=False msg={j.get('message')}"
    res = j.get('result') or {}
    data = res.get('data') or []
    if not data:
        return None, f"empty result (pages={res.get('pages')}, count={res.get('count')})"
    df = pd.DataFrame(data)
    return df, f"pages={res.get('pages')} count={res.get('count')}"


def year_hist(df, col):
    s = pd.to_datetime(df[col], errors='coerce')
    return s.dt.year.value_counts().sort_index().to_dict()


def main():
    print('=' * 70)
    print('[A] 内部人增持: RPT_SHARE_HOLDER_INCREASE DIRECTION=增持')
    print('=' * 70)
    df, msg = hit('RPT_SHARE_HOLDER_INCREASE', '(DIRECTION="增持")')
    if df is None:
        print('  失败:', msg); return_early = True
    else:
        return_early = False
        print(f'  成功: {msg}')
        print(f'  列: {list(df.columns)[:14]}...')
        tcol = 'EITIME' if 'EITIME' in df.columns else df.columns[-1]
        print(f'  年份分布(EITIME): {year_hist(df, tcol)}')
        print(f'  样本: {df[["SECURITY_CODE", "SECURITY_NAME_ABBR", "CHANGE_RATIO", "HOLDER_NAME"]].head(3).to_string() if {"SECURITY_CODE","CHANGE_RATIO"}.issubset(df.columns) else df.head(2).T.to_string()}')

    print('=' * 70)
    print('[B] 机构盈利预测 (SUE输入): 候选reportName')
    print('=' * 70)
    url = 'https://datacenter-web.eastmoney.com/api/data/v1/get'
    for name, flt, extra in [('RPT_RES_PROFITPREDICT', '', {'sortColumns': 'SECURITY_CODE'}),
                             ('RPT_RES_PROFITPREDICT', '(SECURITY_CODE="300750")', {}),
                             ('RPT_DMSK_FN_PROFITPREDICT', '', {})]:
        try:
            params = {'reportName': name, 'columns': 'ALL', 'source': 'WEB',
                      'client': 'WEB', 'filter': flt, 'pageSize': '500', 'pageNumber': '1'}
            params.update(extra)
            r = requests.get(url, params=params, timeout=30, headers=UA)
            j = r.json()
            res = j.get('result') or {}
            data = res.get('data') or []
            print(f'  {name} flt={flt!r}: success={j.get("success")} count={res.get("count")} pages={res.get("pages")}')
            if data:
                print(f'    完整字段(首行原始JSON): {list(data[0].keys())}')
                print(f'    原始首行: {dict(list(data[0].items())[:22])}')
        except Exception as e:
            print(f'  {name}: EXC {e}')

    print('=' * 70)
    print('[A2] 增持历史深度: 末页最早EITIME + 年份分布(全量拉取)')
    print('=' * 70)
    frames = []
    for p in range(1, 69):
        df, _ = hit('RPT_SHARE_HOLDER_INCREASE', '(DIRECTION="增持")', page=p)
        if df is None:
            print(f'  第{p}页失败'); break
        frames.append(df)
    full = pd.concat(frames, ignore_index=True)
    full['eitime'] = pd.to_datetime(full['EITIME'], errors='coerce')
    print(f'  全量 {len(full)} 条, EITIME范围: {full["eitime"].min()} ~ {full["eitime"].max()}')
    print(f'  年份分布: {full["eitime"].dt.year.value_counts().sort_index().to_dict()}')
    print(f'  覆盖股票数: {full["SECURITY_CODE"].nunique()}')

    print('=' * 70)
    print('[C] 研报评级: 更多候选reportName')
    print('=' * 70)
    for name in ['RPT_ORGRATING', 'RPT_STOCKRATING', 'RPT_MAIN_RATING', 'RPT_RESEARCHREPORT_RATING',
                 'RPT_F10_EH_ORGRATING', 'RPT_VALUEANALYSIS_RATING']:
        df, msg = hit(name, '(SECURITY_CODE="300750")')
        if df is None:
            print(f'  {name}: {msg}')
        else:
            print(f'  {name}: {msg} 行数={len(df)} 列: {[c for c in df.columns if c not in ("SECUCODE","SECURITY_CODE")][:14]}')


if __name__ == '__main__':
    main()
