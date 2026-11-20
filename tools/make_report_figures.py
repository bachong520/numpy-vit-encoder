"""Generate the two report figures with Pillow (no matplotlib needed).

Figure 1: end-to-end data flow with shapes; student TODOs are drawn in red.
Figure 2: one TransformerBlock plus the inside of its multi-head attention.

Run:  .venv/Scripts/python.exe tools/make_report_figures.py
Out:  Experiment-Report-Template-SZU-main/figs/pipeline.png
      Experiment-Report-Template-SZU-main/figs/block.png
"""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).resolve().parent.parent / "Experiment-Report-Template-SZU-main" / "figs"
FONT = "C:/Windows/Fonts/msyh.ttc"
FONT_B = "C:/Windows/Fonts/msyhbd.ttc"
MONO = "C:/Windows/Fonts/consola.ttf"

# palette
RED_D, RED_F = (176, 42, 30), (253, 232, 229)
BLU_D, BLU_F = (44, 62, 80), (236, 240, 245)
GRN_D, GRN_F = (30, 110, 80), (231, 245, 238)
GREY = (110, 115, 122)
INK = (34, 38, 44)


def f(path, size):
    return ImageFont.truetype(path, size)


def center(d, box, text, font, fill):
    x0, y0, x1, y1 = box
    w = d.textlength(text, font=font)
    a = font.getbbox(text)
    h = a[3] - a[1]
    d.text(((x0 + x1 - w) / 2, (y0 + y1 - h) / 2 - a[1]), text, font=font, fill=fill)


def box(d, xy, lines, kind="given", title_size=25, sub_size=21, radius=14, width=3):
    """lines: [(text, role)] where role in {title, shape, tag}"""
    dcol, fill = (RED_D, RED_F) if kind == "student" else (BLU_D, BLU_F)
    if kind == "green":
        dcol, fill = GRN_D, GRN_F
    d.rounded_rectangle(xy, radius=radius, fill=fill, outline=dcol, width=width)
    x0, y0, x1, y1 = xy
    drawn = []
    for text, role in lines:
        if role == "title":
            ft, col = f(FONT_B, title_size), dcol
        elif role == "shape":
            ft, col = f(MONO, sub_size), GREY
        else:  # tag
            ft, col = f(FONT, sub_size - 3), RED_D if kind == "student" else GREY
        drawn.append((text, ft, col))
    total = sum(t.size + 7 for _, t, _ in drawn) - 7
    y = y0 + ((y1 - y0) - total) / 2
    for text, ft, col in drawn:
        tw = d.textlength(text, font=ft)
        d.text(((x0 + x1 - tw) / 2, y), text, font=ft, fill=col)
        y += ft.size + 7


def arrow(d, x0, y, x1, color=BLU_D, width=3, head=11):
    d.line([(x0, y), (x1 - head + 2, y)], fill=color, width=width)
    d.polygon([(x1, y), (x1 - head, y - head * 0.55), (x1 - head, y + head * 0.55)], fill=color)


def down(d, x, y0, y1, color=BLU_D, width=3, head=11):
    d.line([(x, y0), (x, y1 - head + 2)], fill=color, width=width)
    d.polygon([(x, y1), (x - head * 0.55, y1 - head), (x + head * 0.55, y1 - head)], fill=color)


# ---------------------------------------------------------------- figure 1
def figure_pipeline():
    W, H = 1860, 880
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)

    t = f(FONT_B, 40)
    d.text((60, 34), "NumPy 复现 ViT-B/32 图像编码器的前向数据流", font=t, fill=INK)
    leg = f(FONT, 21)
    d.rounded_rectangle([1180, 44, 1208, 70], radius=5, fill=RED_F, outline=RED_D, width=3)
    d.text((1216, 40), "本次实现的 TODO", font=leg, fill=RED_D)
    d.rounded_rectangle([1480, 44, 1508, 70], radius=5, fill=BLU_F, outline=BLU_D, width=3)
    d.text((1516, 40), "课程提供", font=leg, fill=BLU_D)

    y0, h, w, gap = 165, 150, 258, 44
    xs = [50 + i * (w + gap) for i in range(6)]
    stages = [
        ("输入图像", "(B, 3, 224, 224)", "预处理后 float32", "given"),
        ("patchify", "(B, 49, 3072)", "TODO-01", "student"),
        ("patch 投影", "(B, 49, 768)", "TODO-05a  无 bias", "student"),
        ("拼 CLS + 位置", "(B, 50, 768)", "TODO-05b", "student"),
        ("12 × Block", "(B, 50, 768)", "TODO-02/03/04", "student"),
        ("输出 token", "(B, 512)", "TODO-06  已单位化", "student"),
    ]
    for x, (a, b, c, kind) in zip(xs, stages):
        box(d, (x, y0, x + w, y0 + h),
            [(a, "title"), (b, "shape"), (c, "tag")], kind=kind)
    for i in range(5):
        arrow(d, xs[i] + w + 6, y0 + h / 2, xs[i + 1] - 6)

    # inner detail of the long stage, drawn under its box
    det_y = y0 + h + 52
    for x, txt in [(xs[4] + w / 2, "pre_norm")]:
        pass

    # pipeline annotations under the boxes
    notes = [
        "原图直接切块",
        "reshape + transpose",
        "3072 → 768",
        "token 数 49 → 50",
        "见 图 2",
        "768 → 512 → 单位长度",
    ]
    for x, note in zip(xs, notes):
        ft = f(FONT, 20)
        tw = d.textlength(note, font=ft)
        d.text((x + (w - tw) / 2, y0 + h + 14), note, font=ft, fill=GREY)

    # detail chain of the encoder body
    dy = 470
    dt = f(FONT_B, 27)
    d.text((50, dy - 44), "编码器主体（encode_tokens，学生实现部分用红框）", font=dt, fill=INK)

    seq = [
        ("patchify", "TODO-01", "student", 210),
        ("reshape\n(B,C,7,32,7,32)", "", "text", 250),
        ("transpose\n(0,2,4,1,3,5)", "", "text", 240),
        ("reshape\n(B,49,3072)", "", "text", 210),
    ]
    x = 50
    for name, tag, kind, bw in seq:
        if kind == "text":
            d.rounded_rectangle([x, dy, x + bw, dy + 92], radius=10,
                                fill=(250, 250, 250), outline=(200, 204, 210), width=2)
            lines = name.split("\n")
            yy = dy + 18
            for ln in lines:
                ft = f(MONO, 20)
                tw = d.textlength(ln, font=ft)
                d.text((x + (bw - tw) / 2, yy), ln, font=ft, fill=(70, 74, 80))
                yy += 27
        else:
            box(d, (x, dy, x + bw, dy + 92), [(name, "title"), (tag, "tag")],
                kind=kind, title_size=23, sub_size=18)
        x += bw
        if x < 1100:
            arrow(d, x + 4, dy + 46, x + 34, color=GREY, width=2, head=9)
            x += 40

    # the 12-block stack as a tall frame on the right
    fx0, fx1 = 1150, 1810
    d.rounded_rectangle([fx0, dy - 8, fx1, dy + 200], radius=14,
                        fill=(252, 246, 245), outline=RED_D, width=3)
    ft = f(FONT_B, 23)
    d.text((fx0 + 18, dy + 8), "× 12 层，每层形状恒为 (B, 50, 768)", font=ft, fill=RED_D)
    ft = f(FONT, 21)
    d.text((fx0 + 18, dy + 44), "x → norm1 → 多头注意力 → +x（残差 1）", font=ft, fill=INK)
    d.text((fx0 + 18, dy + 78), "  → norm2 → MLP(768→3072→768) → +（残差 2）", font=ft, fill=INK)
    d.text((fx0 + 18, dy + 118), "两层加法都没有可学习缩放（无 LayerScale）；", font=ft, fill=GREY)
    d.text((fx0 + 18, dy + 150), "归一化在支路上，残差主干不被归一化 —— pre-LN。", font=ft, fill=GREY)

    # bottom formula strip
    fy = 720
    ft = f(MONO, 24)
    d.rounded_rectangle([50, fy, 1810, fy + 78], radius=12,
                        fill=(245, 247, 250), outline=(210, 214, 220), width=2)
    d.text((70, fy + 12), "scores[b,h,i,j] = q[b,h,i,:] · k[b,h,j,:] / sqrt(64)      ", font=ft, fill=INK)
    d.text((70, fy + 44), "context[b,h,i,:] = sum_j P[b,h,i,j] * v[b,h,j,:]   ,   P = softmax(scores, axis=-1)", font=ft, fill=INK)

    img.save(OUT / "pipeline.png")
    return OUT / "pipeline.png"


# ---------------------------------------------------------------- figure 2
def figure_block():
    W, H = 1580, 950
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    d.text((50, 26), "单个 TransformerBlock 与多头注意力的内部", font=f(FONT_B, 34), fill=INK)

    # ---------------- left column: one block, top to bottom
    lx, lw, bh, gap = 66, 306, 56, 26
    items = [
        ("输入 x   (B, 50, 768)", "", "given"),
        ("LayerNorm 1", "给定", "given"),
        ("多头注意力", "TODO-02 / 03", "student"),
        ("x + update   （残差 1）", "TODO-04a", "student"),
        ("LayerNorm 2", "给定", "given"),
        ("Linear 768 → 3072", "给定", "given"),
        ("GELU（精确 erf 版）", "给定", "given"),
        ("Linear 3072 → 768", "给定", "given"),
        ("+ mlp_update   （残差 2）", "TODO-04b", "student"),
        ("输出   (B, 50, 768)", "", "given"),
    ]
    for i, (t1, t2, kind) in enumerate(items):
        top = 120 + i * (bh + gap)
        lines = [(t1, "title")] + ([(t2, "tag")] if t2 else [])
        box(d, (lx, top, lx + lw, top + bh), lines, kind=kind, title_size=20, sub_size=15)
        if i < len(items) - 1:
            down(d, lx + lw / 2, top + bh, top + bh + gap, head=9)

    # residual skips drawn on the left margin of the column
    def skip(xoff, y_from, y_to):
        d.line([(lx, y_from), (lx - xoff, y_from), (lx - xoff, y_to), (lx - 2, y_to)],
               fill=RED_D, width=3)
        d.polygon([(lx, y_to), (lx - 11, y_to - 6), (lx - 11, y_to + 6)], fill=RED_D)

    c = lambda i: 120 + i * (bh + gap) + bh / 2
    skip(46, c(0), c(3))
    skip(24, c(3), c(8))

    # ---------------- right panel: inside the attention module
    rx, by = 500, 160
    bw, bh2, gx, gy = 236, 88, 28, 66
    d.text((rx, 104), "MultiHeadAttention 内部（D = 768, H = 12, D/H = 64）",
           font=f(FONT_B, 24), fill=INK)

    row1 = [
        ("x", "(B, 50, 768)", "given"),
        ("qkv = x @ W^T + b", "(B, 50, 2304)", "given"),
        ("split_heads", "(B, 12, 50, 64)", "student"),
        ("scores = q @ k^T / 8", "(B, 12, 50, 50)", "student"),
    ]
    row2 = [
        ("out = merged @ W^T + b", "(B, 50, 768)", "student"),
        ("merge heads", "(B, 50, 768)", "student"),
        ("context = P @ v", "(B, 12, 50, 64)", "student"),
        ("P = softmax(scores, -1)", "(B, 12, 50, 50)", "given"),
    ]
    xs = [rx + i * (bw + gx) for i in range(4)]
    for x, (n, sh, kind) in zip(xs, row1):
        box(d, (x, by, x + bw, by + bh2), [(n, "title"), (sh, "shape")],
            kind=kind, title_size=16, sub_size=15)
    y2 = by + bh2 + gy
    for x, (n, sh, kind) in zip(xs, row2):
        box(d, (x, y2, x + bw, y2 + bh2), [(n, "title"), (sh, "shape")],
            kind=kind, title_size=16, sub_size=15)

    for i in range(3):
        arrow(d, xs[i] + bw + 2, by + bh2 / 2, xs[i + 1] - 2, color=GREY, width=2, head=8)
    # snake: row1 last cell drops into row2 last cell, then row2 runs right to left
    down(d, xs[3] + bw / 2, by + bh2 + 2, y2 - 3, color=GREY, width=2, head=8)
    for i in range(3, 0, -1):
        arrow(d, xs[i] - 2, y2 + bh2 / 2, xs[i - 1] + bw + 2, color=GREY, width=2, head=8)

    notes = [
        "k^T 必须用 np.swapaxes(-1, -2)：4 维数组的 .T 会反转全部四个轴，形状直接错。",
        "除以 sqrt(64) = 8：64 维点积的标准差约为 8，不缩放会让 softmax 饱和成 one-hot。",
        "P @ v 的内维是 50（token 数）；写成 v @ P 会因 64 != 50 报维度不匹配。",
        "q/k/v 由一次 (B,50,768) @ (768,2304) 得到，三个矩阵沿输出轴拼接存放。",
    ]
    yy = y2 + bh2 + 40
    for n in notes:
        d.text((rx, yy), n, font=f(FONT, 19), fill=GREY)
        yy += 32

    img.save(OUT / "block.png")
    return OUT / "block.png"


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for p in (figure_pipeline(), figure_block()):
        print("wrote", p)
