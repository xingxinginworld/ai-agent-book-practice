# -*- coding: utf-8 -*-
"""
实验 2-6 的幻灯片渲染器
========================
把"结构化元数据 + 幻灯片大纲"渲染成一份**自包含、可离线打开**的 HTML 演示文稿。

设计约束（与实验"轻量本地"定位一致）：
- 不依赖任何外部 CDN / 网络资源，双击即可在浏览器打开。
- 用系统字体，无外部字体文件。
- 支持键盘（←/→/空格）与底部按钮翻页、F 全屏、Home/End 跳首尾。
- 仅用极简 CSS + 原生 JS，单文件。
"""

HTML_HEAD = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>{title}</title>
<style>
  :root {{
    --bg: #0f172a; --panel: #1e293b; --accent: #38bdf8; --text: #e2e8f0;
    --muted: #94a3b8; --card: #172033;
  }}
  * {{ box-sizing: border-box; }}
  html, body {{ margin: 0; height: 100%; background: var(--bg); color: var(--text);
    font-family: -apple-system, "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif; }}
  #deck {{ position: relative; height: 100vh; overflow: hidden; }}
  .slide {{ position: absolute; inset: 0; display: none; flex-direction: column;
    padding: 6vh 8vw; }}
  .slide.active {{ display: flex; }}
  .slide-kicker {{ color: var(--accent); letter-spacing: .15em; font-size: 14px;
    text-transform: uppercase; margin-bottom: 12px; }}
  .slide-title {{ font-size: clamp(24px, 4.2vw, 48px); font-weight: 700; line-height: 1.2;
    margin: 0 0 28px; }}
  .slide-body {{ font-size: clamp(16px, 2.2vw, 26px); line-height: 1.7; }}
  .slide-body ul {{ padding-left: 1.1em; margin: 0; }}
  .slide-body li {{ margin: 10px 0; }}
  .meta {{ color: var(--muted); font-size: clamp(13px, 1.6vw, 18px); }}
  .cover {{ justify-content: center; }}
  .cover .slide-title {{ font-size: clamp(28px, 5vw, 60px); }}
  .bar {{ position: fixed; bottom: 0; left: 0; right: 0; display: flex; align-items: center;
    justify-content: space-between; padding: 10px 18px; background: var(--panel);
    border-top: 1px solid #334155; font-size: 14px; color: var(--muted); z-index: 5; }}
  .bar button {{ background: var(--accent); color: #06283d; border: 0; border-radius: 6px;
    padding: 6px 14px; font-size: 14px; cursor: pointer; }}
  .bar button:disabled {{ opacity: .4; cursor: default; }}
  .counter {{ font-variant-numeric: tabular-nums; }}
</style>
</head>
<body>
<div id="deck">
"""

HTML_TAIL = """</div>
<div class="bar">
  <button id="prev">← 上一页</button>
  <span class="counter"><span id="cur">1</span> / <span id="total">1</span></span>
  <button id="next">下一页 →</button>
</div>
<script>
  var slides = Array.prototype.slice.call(document.querySelectorAll('.slide'));
  var idx = 0;
  function show(i){{
    idx = Math.max(0, Math.min(slides.length-1, i));
    slides.forEach(function(s,k){{ s.classList.toggle('active', k===idx); }});
    document.getElementById('cur').textContent = idx+1;
    document.getElementById('prev').disabled = idx===0;
    document.getElementById('next').disabled = idx===slides.length-1;
  }}
  document.getElementById('prev').onclick = function(){{ show(idx-1); }};
  document.getElementById('next').onclick = function(){{ show(idx+1); }};
  document.addEventListener('keydown', function(e){{
    if(e.key==='ArrowRight'||e.key===' '||e.key==='PageDown'){{ show(idx+1); }}
    else if(e.key==='ArrowLeft'||e.key==='PageUp'){{ show(idx-1); }}
    else if(e.key==='Home'){{ show(0); }}
    else if(e.key==='End'){{ show(slides.length-1); }}
    else if(e.key==='f'||e.key==='F'){{ if(!document.fullscreenElement){{document.documentElement.requestFullscreen();}} else {{document.exitFullscreen();}} }}
  }});
  show(0);
</script>
</body>
</html>
"""

COVER_SLIDE = """  <section class="slide cover">
    <div class="slide-kicker">论文汇报 · 自动生成</div>
    <h1 class="slide-title">{title}</h1>
    <div class="meta">{authors}{venue}</div>
  </section>
"""

CONTENT_SLIDE = """  <section class="slide">
    <div class="slide-kicker">Slide {n}</div>
    <h2 class="slide-title">{title}</h2>
    <div class="slide-body"><ul>{bullets}</ul></div>
  </section>
"""


def _esc(s):
    """极简 HTML 转义，防止论文文本里的 < > & 破坏结构。"""
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def render_slides(meta, slides, max_slides=8):
    """生成完整 HTML 字符串。

    meta:   dict，至少含 title；可选 authors / venue。
    slides: list[dict]，每项 {"title":..., "bullets":[...]}。
    返回完整 HTML（str）。
    """
    title = _esc(meta.get("title") or "未命名论文汇报")
    authors = _esc(meta.get("authors") or "")
    venue = _esc(meta.get("venue") or "")
    author_line = ("作者：" + authors + "　") if authors else ""
    venue_line = ("出处：" + venue) if venue else ""
    meta_html = (author_line + venue_line).strip()

    parts = [HTML_HEAD.format(title=title)]

    # 封面
    parts.append(COVER_SLIDE.format(title=title, authors=author_line, venue=venue_line))

    # 内容页（受 max_slides 约束，封面另算，故内容最多 max_slides-1 张）
    cap = max(1, max_slides - 1)
    for i, sl in enumerate(slides[:cap], start=1):
        stitle = _esc(sl.get("title") or f"幻灯片 {i}")
        bullets = "".join("<li>{}</li>".format(_esc(b)) for b in sl.get("bullets", []))
        parts.append(CONTENT_SLIDE.format(n=i, title=stitle, bullets=bullets))

    parts.append(HTML_TAIL)
    return "\n".join(parts)
