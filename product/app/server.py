#!/usr/bin/env python3
"""
公众号排版工具 — FastAPI 服务.

用法:
    .venv/bin/python product/app/server.py [--port 8000]
    # 浏览器打开 http://localhost:8000
"""

import argparse
from pathlib import Path

from flask import Flask, render_template

APP_DIR = Path(__file__).resolve().parent
TEMPLATE_DIR = APP_DIR / 'templates'

app = Flask(__name__, template_folder=str(TEMPLATE_DIR))


@app.route('/')
def index():
    return render_template('index.html')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='公众号排版工具')
    parser.add_argument('--port', type=int, default=8000)
    args = parser.parse_args()
    print(f'公众号排版工具启动: http://localhost:{args.port}')
    app.run(host='0.0.0.0', port=args.port, debug=False)
