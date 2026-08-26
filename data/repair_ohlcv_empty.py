"""修复raw_data中OHLC/量额字段为空的行(139处).

- 成交量idx7/成交额idx8: 跨复权同值, 兄弟文件直接复制
- 价格idx3-6: 复权价不同, 用同行参考价推算比例(同日复权因子对所有价格一致):
  broken[idx] = sibling[idx] * broken[ref] / sibling[ref]
修复失败(无兄弟/无参考)的保留原样并打印.
"""
import csv
import os
import re
import shutil
from pathlib import Path

RAW = Path('/mnt/d/quant/data/stock_data/raw_data')
BACKUP = Path('/mnt/d/quant/data/backup_torn_rows_20260821')
JOURNAL = BACKUP / 'ohlcv_repair.log'
TYPES = ['none', 'qfq', 'hfq']
DATE_RE = re.compile(r'^\d{4}-\d{2}-\d{2}$')
PRICE_IDX = [3, 4, 5, 6]


def is_num(s):
    try:
        float(s)
        return True
    except (ValueError, TypeError):
        return False


def read_file(path):
    with open(path, encoding='utf-8', errors='replace', newline='') as f:
        content = f.read()
    eol = '\r\n' if '\r\n' in content else '\n'
    lines = content.split(eol)
    if lines and lines[-1] == '':
        lines = lines[:-1]
    rows = []
    for ln in lines[1:]:
        if not ln:
            rows.append(None)
            continue
        rows.append(next(csv.reader([ln])))
    return eol, lines, rows


def main():
    dirs = sorted(p for p in RAW.iterdir() if p.is_dir() and re.fullmatch(r'\d{6}', p.name))
    n_repaired, failures = 0, []
    jf = open(JOURNAL, 'w', encoding='utf-8')

    for d in dirs:
        code = d.name
        loaded = {}
        for t in TYPES:
            p = d / f'{t}.csv'
            if not p.exists():
                continue
            try:
                loaded[t] = read_file(p)
            except Exception:
                continue
        if not loaded:
            continue

        for t, (eol, lines, rows) in loaded.items():
            changed = False
            for i, row in enumerate(rows):
                if row is None or len(row) != 13 or not DATE_RE.match(row[2] or ''):
                    continue
                bad = [j for j in range(3, 9) if row[j] == '' or not is_num(row[j])]
                if not bad:
                    continue
                date = row[2]
                orig = lines[i + 1]
                # 找同日期完好的兄弟行
                sib_row = None
                for t2, (_, _, rows2) in loaded.items():
                    if t2 == t:
                        continue
                    for r2 in rows2:
                        if r2 is not None and len(r2) == 13 and r2[2] == date \
                           and all(is_num(r2[j]) for j in range(3, 9)):
                            sib_row = r2
                            break
                    if sib_row is not None:
                        break
                if sib_row is None:
                    failures.append((code, t, date, bad, '无完好兄弟行'))
                    continue
                ok = True
                for j in bad:
                    if j in (7, 8):
                        row[j] = sib_row[j]
                    else:
                        # 价格: 找同行+兄弟行都完好的参考价
                        ref = next((k for k in PRICE_IDX if k not in bad and is_num(row[k])
                                    and is_num(sib_row[k]) and float(sib_row[k]) != 0), None)
                        if ref is None:
                            failures.append((code, t, date, [j], '无参考价'))
                            ok = False
                            continue
                        ratio = float(row[ref]) / float(sib_row[ref])
                        row[j] = repr(float(sib_row[j]) * ratio)
                if not ok:
                    continue
                lines[i + 1] = ','.join(row)
                n_repaired += 1
                jf.write(f'{code}/{t}\t{date}\t{orig}\t->\t{lines[i+1]}\n')
                changed = True
            if changed:
                assert len(lines) == len(rows) + 1, f'{code}/{t} 行数对账失败'
                p = d / f'{t}.csv'
                bak = BACKUP / code
                bak.mkdir(parents=True, exist_ok=True)
                if not (bak / f'{t}.csv').exists():
                    shutil.copy2(p, bak / f'{t}.csv')
                tmp = p.with_suffix('.tmp')
                with open(tmp, 'w', encoding='utf-8', newline='') as f:
                    f.write(eol.join(lines) + eol)
                os.replace(tmp, p)
    jf.close()

    print(f'修复OHLC/量额空值行: {n_repaired}')
    print(f'失败(保留原样, 需重下): {len(failures)}')
    for x in failures[:20]:
        print(' ', x)


if __name__ == '__main__':
    main()
