import streamlit as st
from pathlib import Path
from datetime import datetime
import re

# 放到文件顶部 import 附近
import base64, mimetypes, re

IMG_PATTERN = re.compile(r'!\[([^\]]*)\]\(([^)]+)\)')  # ![alt](path)

def embed_local_images(md_text: str, md_file_path: Path) -> str:
    """把 markdown 中的本地图片相对路径替换为 <img src="data:...">"""
    def repl(m):
        alt, url = m.group(1), m.group(2).strip()
        # 跳过 http/https
        if re.match(r'^https?://', url, re.I):
            return m.group(0)
        # 计算相对路径
        img_path = (md_file_path.parent / url).resolve()
        # 仅允许 notes 目录内的文件，防止越界
        try:
            img_path.relative_to(BASE.resolve())
        except Exception:
            return f'![{alt}](#)'  # 非法路径占位
        if not img_path.exists():
            return f'![{alt}](#)'  # 找不到就留空
        mime, _ = mimetypes.guess_type(str(img_path))
        mime = mime or 'application/octet-stream'
        b64 = base64.b64encode(img_path.read_bytes()).decode('ascii')
        # 用 HTML img 内联，并自适应宽度
        return f'<img alt="{alt}" src="data:{mime};base64,{b64}" style="max-width:100%; height:auto;" />'
    return IMG_PATTERN.sub(repl, md_text)


# ------------------------
# 基本设置
# ------------------------
st.set_page_config(page_title="DMH's Blog", page_icon="📚", layout="wide")

BASE = Path("notes")                  # 笔记根目录
DEFAULT_CATS = ["机器学习与深度学习", "生活随笔"]
FRONT_SPLIT = "---"

BASE.mkdir(exist_ok=True)
for c in DEFAULT_CATS:
    (BASE / c).mkdir(exist_ok=True)

# ------------------------
# 工具函数
# ------------------------
def parse_post(fp: Path):
    """解析文章：支持最简单的 front matter:
    ---
    title: 标题
    date: 2025-01-01 12:00
    tags: a, b
    ---
    正文...
    """
    raw = fp.read_text(encoding="utf-8")
    title, date_str, body = fp.stem, "", raw
    if raw.startswith(FRONT_SPLIT):
        parts = raw.split(FRONT_SPLIT, 2)
        if len(parts) >= 3:
            _, header, body = parts
            for line in header.strip().splitlines():
                if ":" in line:
                    k, v = line.split(":", 1)
                    k, v = k.strip().lower(), v.strip()
                    if k == "title":
                        title = v or title
                    elif k == "date":
                        date_str = v
    # 如果没有日期，退化用文件修改时间
    try:
        dt = datetime.strptime(date_str, "%Y-%m-%d %H:%M")
    except Exception:
        dt = datetime.fromtimestamp(fp.stat().st_mtime)
    # 如果正文没有标题，尝试从第一行 # 提取
    if title == fp.stem:
        m = re.search(r"^(#+)\s+(.+)$", body.strip(), flags=re.M)
        if m:
            title = m.group(2).strip()
    return {"path": fp, "title": title, "date": dt, "body": body}

def list_posts(cat_dir: Path):
    posts = [parse_post(f) for f in cat_dir.glob("*.md")]
    posts.sort(key=lambda x: x["date"], reverse=True)  # 最新在最上
    return posts

# ------------------------
# 左侧：分类 + 文章列表
# ------------------------
st.sidebar.title("🗂️ 分类与文章")
cats = sorted([p.name for p in BASE.iterdir() if p.is_dir()])
cur_cat = st.sidebar.selectbox("选择分类", cats, index=cats.index("机器学习") if "机器学习" in cats else 0)

# 可选：搜索框（如不需要可注释）
q = st.sidebar.text_input("搜索标题关键词", "")

posts = list_posts(BASE / cur_cat)
if q.strip():
    key = q.lower()
    posts = [p for p in posts if key in p["title"].lower()]

if not posts:
    st.sidebar.info("该分类下暂无文章。请在本地目录添加 .md 文件。")
    st.write("👈 左侧选择分类/文章")
    st.stop()

# 选择当前文章（默认第一篇=最新）
titles = [p["title"] for p in posts]
default_idx = 0
picked_title = st.sidebar.radio("文章列表（最新在上）", titles, index=default_idx)
current = next(p for p in posts if p["title"] == picked_title)

# ------------------------
# 主区：当前文章的 Markdown 预览
# ------------------------
st.title(current["title"])
st.caption(f"分类：{cur_cat} ｜ 更新时间：{current['date'].strftime('%Y-%m-%d %H:%M')} ｜ 文件：{current['path'].name}")

st.markdown("---")
# st.markdown(current["body"], unsafe_allow_html=False)
html = embed_local_images(current["body"], current["path"])
st.markdown(html, unsafe_allow_html=True)