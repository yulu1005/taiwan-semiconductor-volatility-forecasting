#!/bin/bash
# run_nb.sh — 把 .py（jupytext 格式）轉成 .ipynb 並執行，印出文字輸出與錯誤
for f in "$@"; do
  jupytext --to ipynb "$f" -q                                   # 轉成 notebook
  nb="${f%.py}.ipynb"                                            # notebook 檔名
  jupyter nbconvert --to notebook --execute --inplace "$nb" --ExecutePreprocessor.timeout=1800 2>&1 | grep -v "^\[NbConvertApp\]" | tail -5  # 執行
  python3 - "$nb" <<'PY'
import json,sys
nb=json.load(open(sys.argv[1]))
for c in nb['cells']:
    if c['cell_type']=='code':
        for o in c.get('outputs',[]):
            if o.get('name')=='stdout': print(''.join(o['text'])[:2500])
            if o.get('output_type')=='error': print('ERR',o['ename'],o['evalue'][:500])
            if 'data' in o and 'text/plain' in o['data'] and 'image/png' not in o['data']: print(''.join(o['data']['text/plain'])[:2500])
PY
done
