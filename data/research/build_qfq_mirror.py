#!/usr/bin/env python3
"""研究层Parquet镜像构建: backtrader_data/*_qfq.csv (5376只) → 单表长格式.

用途: 探针用DuckDB直查 (全市场截面29s→1.1s, 单股22ms); 纯增量研究资产, 生产链路零改动.
注意: 目录含 sh000001/sh000852 指数文件, 锚定正则排除 (只收 6位数字_qfq.csv);
      另有 *.preQfqFix 备份不匹配glob, 无碍.
重建时机: 数据刷新后 (bt重建完成后跑一次, ~15秒); 镜像为时点快照, 探针须知其数据态.

用法: /mnt/d/quant/.venv/bin/python data/research/build_qfq_mirror.py
查询示例:
    import duckdb
    con = duckdb.connect()
    df = con.execute(r"SELECT code, date, close FROM '/mnt/d/quant/data/research/qfq_kline.parquet' "
                     "WHERE code IN ('000001','600519') AND date >= '2026-01-01'").df()
"""
import time

import duckdb

t0 = time.time()
con = duckdb.connect()
con.execute("SET threads=3")
con.execute(r"""
COPY (
  SELECT regexp_extract(filename, '(\d{6})_qfq\.csv$', 1) AS code,
         CAST(datetime AS DATE) AS date,
         open, high, low, close, volume
  FROM read_csv('/mnt/d/quant/data/stock_data/backtrader_data/*_qfq.csv',
                filename=true, header=true)
  WHERE regexp_matches(filename, '/\d{6}_qfq\.csv$')
  ORDER BY code, date
) TO '/mnt/d/quant/data/research/qfq_kline.parquet'
(FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 250000)
""")
n = con.execute(
    "SELECT count(*), count(DISTINCT code), min(date), max(date) "
    "FROM '/mnt/d/quant/data/research/qfq_kline.parquet'").fetchone()
dup = con.execute(
    "SELECT count(*) FROM (SELECT code, date FROM "
    "'/mnt/d/quant/data/research/qfq_kline.parquet' GROUP BY code, date HAVING count(*)>1)"
).fetchone()[0]
print(f"构建完成 {time.time()-t0:.0f}s: {n[0]:,}行 / {n[1]}只 / {n[2]}~{n[3]} / 重复(code,date)={dup}")
assert dup == 0, "镜像存在重复行!"
