#!/usr/bin/env python3
"""
微信公众号内联样式 HTML 渲染器 — 对标 doocs/md / mdnice 的本地实现.

把 Markdown 渲染成微信编辑器可接受的"内联样式 HTML":
微信会剥掉 <style> 与 class 属性, 只保留各标签上的内联 style,
因此整篇文章的排版样式(大标题/二级标题/表格/引用/图片/分割线)全部内联注入.

用法:
    quant/bin/python3 strategy/wechat_render.py <in.md> [--theme minimal] [--out out.html]
    quant/bin/python3 strategy/wechat_render.py --themes          # 列出可用主题
    编程接口: from wechat_render import md_to_wechat, THEME_NAMES
"""

import argparse
from pathlib import Path

from markdown_it import MarkdownIt
from bs4 import BeautifulSoup

_THEME_DEFAULTS = {
    'font_family': (
        "-apple-system, BlinkMacSystemFont, 'PingFang SC', "
        "'Hiragino Sans GB', 'Microsoft YaHei', sans-serif"),
    'font_size': '15px',
    'line_height': '1.75',
    'color': '#333333',
    'accent': '#2f6fed',            # 强调色: 表头/引用/图片边框/链接
    'h1_bg': '#2f6fed',  # 大标题横幅底色 (纯色实底, 微信不剥linear-gradient)
    'h1_color': '#ffffff',
    'h2_color': '#2f6fed',
    'h3_color': '#1f1f1f',
    'table_border': '#e8e8e8',
    'thead_bg': '#eef4ff',          # 表头 tint 底
    'thead_color': '#2f6fed',
    'zebra': 'rgba(0,0,0,0.02)',
    'quote_border': '#2f6fed',
    'quote_bg': '#f2f6ff',
    'quote_color': '#555555',
    'warn_border': '#e0483e',       # ⚠️/免责 警示块
    'warn_bg': '#fdf1f0',
    'warn_color': '#a0392f',
    'code_bg': '#f4f4f5',
    'img_radius': '8px',
}

# 三套主题: 只覆盖与默认不同的 token
_THEMES = {
    'minimal': {},  # 极简商务: 蓝 accent, 浅蓝表头, 满宽无竖线表格
    'eyecare': {
        'accent': '#27a06e',
        'h1_bg': '#27a06e',
        'h2_color': '#27a06e',
        'thead_bg': '#eaf7ef',
        'thead_color': '#1f8a5c',
        'quote_border': '#27a06e',
        'quote_bg': '#f0f9f4',
    },
    'fresh': {
        'accent': '#1b9aaa',
        'h1_bg': '#1b9aaa',
        'h2_color': '#17848f',
        'thead_bg': '#e7f5f6',
        'thead_color': '#17848f',
        'quote_border': '#1b9aaa',
        'quote_bg': '#eef9fa',
    },
}

THEME_NAMES = [
    {'id': 'minimal', 'name': '极简商务'},
    {'id': 'eyecare', 'name': '护眼绿'},
    {'id': 'fresh', 'name': '清新蓝'},
]

_MD = MarkdownIt('commonmark').enable('table')

# 文章内图片的两种绝对路径前缀 → 服务器可访问的映射
_IMG_MAP = (
    (f'{Path(__file__).resolve().parent.parent}/strategy/live_charts', 'live_charts'),
    (f'{Path(__file__).resolve().parent}/media_out/images', 'images'),
)


def _get_theme(name):
    theme = dict(_THEME_DEFAULTS)
    overrides = _THEMES.get(name, {})
    theme.update(overrides)
    theme['id'] = name if name in _THEMES else 'minimal'
    return theme


def _base_style(t):
    return (f"font-family:{t['font_family']};color:{t['color']};"
            f"font-size:{t['font_size']};line-height:{t['line_height']};"
            "letter-spacing:0.4px;word-spacing:1px;box-sizing:border-box;")


def _rewrite_img_src(src, img_root):
    """把本地绝对图片路径重写为可加载 URL. img_root='' 时保留原路径(本地打开)."""
    if not src or src.startswith(('http://', 'https://', 'data:')):
        return src
    if not img_root:
        return src  # 本地打开: 保留原始绝对路径
    for local_prefix, url_name in _IMG_MAP:
        p = str(Path(local_prefix))
        if p in src:
            rel = src.split(p, 1)[1].lstrip('/')
            return f"{img_root}/{url_name}/{rel}".replace('//', '/')
    return src


def _style_tree(soup, t, img_root):
    for tag in soup.find_all(True):
        # 微信安全: 去掉 class 等非白名单属性, 只保留内联 style 及必要属性
        attrs = dict(tag.attrs)
        tag.attrs = {}
        for keep in ('src', 'alt', 'href', 'colspan', 'rowspan'):
            if keep in attrs:
                tag.attrs[keep] = attrs[keep]
        del_attrs = set(('class', 'id', 'lang', 'data-sourcepos'))
        tag.attrs = {k: v for k, v in tag.attrs.items() if k not in del_attrs}

        tag.name = tag.name.lower()
        if tag.name == 'p':
            tag['style'] = 'margin:0 0 12px;word-break:break-word;text-align:left;'
            # 仅含图片的段落 → 居中
            if tag.find('img') and not tag.get_text(strip=True):
                tag['style'] = 'text-align:center;margin:12px 0;'
        elif tag.name == 'h1':
            tag['style'] = (f"background:{t['h1_bg']};color:{t['h1_color']};"
                            "padding:20px 16px;margin:0 0 22px;border-radius:10px;"
                            "font-size:20px;font-weight:700;text-align:center;"
                            "line-height:1.5;letter-spacing:1px;")
        elif tag.name == 'h2':
            tag['style'] = (f"border-left:4px solid {t['accent']};color:{t['h2_color']};"
                            "font-size:17px;font-weight:700;padding:2px 0 2px 10px;"
                            "margin:26px 0 12px;line-height:1.4;")
        elif tag.name == 'h3':
            tag['style'] = (f"color:{t['h3_color']};font-size:15.5px;font-weight:700;"
                            f"margin:20px 0 10px;padding-left:8px;border-left:3px solid {t['accent']};"
                            "line-height:1.4;")
        elif tag.name == 'strong' or tag.name == 'b':
            tag['style'] = 'font-weight:700;color:#111111;'
        elif tag.name == 'em':
            tag['style'] = 'font-style:italic;'
        elif tag.name == 'a':
            tag['style'] = f"color:{t['accent']};text-decoration:none;"
        elif tag.name == 'code':
            tag['style'] = (f"background:{t['code_bg']};padding:2px 6px;border-radius:4px;"
                            "font-family:ui-monospace,Menlo,Consolas,monospace;font-size:13px;")
        elif tag.name == 'pre':
            tag['style'] = (f"background:{t['code_bg']};padding:12px 14px;border-radius:8px;"
                            "overflow-x:auto;font-size:13px;line-height:1.6;margin:14px 0;")
        elif tag.name == 'hr':
            tag['style'] = 'border:none;border-top:2px solid #eeeeee;margin:24px 0;'
        elif tag.name in ('ul', 'ol'):
            tag['style'] = 'margin:6px 0 14px;padding-left:24px;'
        elif tag.name == 'li':
            tag['style'] = 'margin:0 0 8px;line-height:1.7;'
        elif tag.name == 'blockquote':
            text = tag.get_text()
            warn = ('⚠' in text) or ('免责' in text) or ('风险' in text) or ('不构成' in text)
            if warn:
                b, bg, c = t['warn_border'], t['warn_bg'], t['warn_color']
            else:
                b, bg, c = t['quote_border'], t['quote_bg'], t['quote_color']
            tag['style'] = (f"border-left:5px solid {b};background:{bg};color:{c};"
                            "padding:12px 14px;margin:14px 0;border-radius:6px;"
                            "font-size:14px;line-height:1.65;")
        elif tag.name == 'img':
            src = tag.attrs.get('src', '')
            tag.attrs['src'] = _rewrite_img_src(src, img_root)
            tag['style'] = (f"max-width:100%;height:auto;border-radius:{t['img_radius']};"
                            "display:block;margin:10px auto;box-shadow:0 2px 8px rgba(0,0,0,0.08);")
        elif tag.name == 'table':
            tag['style'] = ("width:100%;max-width:100%;table-layout:fixed;border-collapse:collapse;"
                            "margin:0;font-size:13.5px;word-break:break-all;")
            # 外包一层横向滚动容器, 防极端窄屏/宽表溢出; 固定布局下正常填满
            wrapper = soup.new_tag('div')
            wrapper['style'] = ("width:100%;max-width:100%;overflow-x:auto;margin:14px 0;"
                                "-webkit-overflow-scrolling:touch;")
            tag.wrap(wrapper)
        elif tag.name == 'th':
            tag['style'] = (f"background:{t['thead_bg']};color:{t['thead_color']};"
                            "font-weight:700;text-align:left;padding:8px 8px;"
                            f"border-bottom:2px solid {t['accent']};word-break:break-all;")
        elif tag.name == 'td':
            tag['style'] = (f"padding:8px 8px;border-bottom:1px solid {t['table_border']};"
                            "vertical-align:middle;text-align:left;word-break:break-all;")
        elif tag.name in ('thead', 'tbody', 'tr'):
            tag['style'] = ''

    # 斑马纹 (偶数行浅底), 需在去重后对 tbody 的 tr 补背景
    for tb in soup.find_all('tbody'):
        for i, tr in enumerate(tb.find_all('tr')):
            if i % 2 == 1:
                cur = tr.get('style', '')
                tr['style'] = (cur + f"background:{t['zebra']};").strip()


def md_to_wechat(md_text, theme='minimal', img_root=''):
    """Markdown → 微信内联样式 HTML (一个 <section> 包裹, 即插即用).

    Args:
        md_text: markdown 源码字符串.
        theme: 'minimal' / 'eyecare' / 'fresh'.
        img_root: 图片 URL 前缀, 服务器预览传 '/media-img', CLI(本地打开)传 ''.
    Returns:
        str: 微信可粘贴的 HTML.
    """
    t = _get_theme(theme)
    html = _MD.render(md_text or '')
    soup = BeautifulSoup(html, 'html.parser')
    _style_tree(soup, t, img_root)
    body = str(soup)
    return f'<section style="{_base_style(t)}">{body}</section>'


def main():
    ap = argparse.ArgumentParser(description='微信公众号内联样式 HTML 渲染器')
    ap.add_argument('infile', nargs='?', help='输入 markdown 文件路径')
    ap.add_argument('--theme', choices=list(_THEMES.keys()), default='minimal')
    ap.add_argument('--out', help='输出 html 路径 (默认打印到 stdout)')
    ap.add_argument('--img-root', default='',
                    help="图片URL前缀; 服务器预览传 '/media-img'")
    ap.add_argument('--themes', action='store_true', help='列出可用主题')
    args = ap.parse_args()

    if args.themes:
        for th in THEME_NAMES:
            print(f"{th['id']:10s} {th['name']}")
        return

    if not args.infile:
        ap.error('需要输入 markdown 文件, 或使用 --themes')
    md_text = Path(args.infile).read_text(encoding='utf-8')
    out = md_to_wechat(md_text, args.theme, args.img_root)
    if args.out:
        Path(args.out).write_text(out, encoding='utf-8')
        print(f"已渲染: {args.out} (主题 {args.theme})")
    else:
        print(out)


if __name__ == '__main__':
    main()