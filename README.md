# 用 NumPy 复现 OpenCLIP ViT-B/32 图像编码器

本项目是 Python 程序设计课程 Project 1 的提交仓库：**不使用 PyTorch / OpenCLIP，只用 NumPy**
逐层复现 OpenCLIP `ViT-B/32` 图像编码器的前向计算，输出必须与课程提供的 FP32 参考结果在
`atol=2e-5, rtol=2e-4` 内一致。

仓库地址：<https://github.com/bachong520/numpy-vit-encoder>

评分部分只有 `image_encoder.py` 中 6 个 TODO；其余文件（预处理、LayerNorm、GELU、检查脚本、
权重与参考输出）均由课程提供，未作修改。

---

## 1. 目录结构

```
.
├── project1-numpy-student/          # 课程提供的工程目录
│   ├── image_encoder.py             # ★ 唯一修改的文件（6 个 TODO）
│   ├── helpers.py                   # LayerNorm / GELU / softmax / unit_vector（给定）
│   ├── preprocess_image.py          # resize + center crop + 归一化（给定）
│   ├── runtime.py / run.py / check.py  # 加载、命令行、数值检查（给定）
│   ├── transformer_steps.py         # 课堂用的 3 token 玩具演示（给定）
│   ├── config.json / weights/manifest.json
│   ├── assets/images/ + images.json # 10 张本地测试图
│   ├── assets/reference_outputs.npz # 参考中间结果（未入库，见第 5 节）
│   └── weights/image_encoder.npz    # 152 个视觉张量，312 MB（未入库，见第 5 节）
├── Experiment-Report-Template-SZU-main/   # 实验报告（XeLaTeX，Overleaf 编译）
│   ├── main.tex
│   └── figs/pipeline.png, figs/block.png  # 自绘架构图
├── tools/make_report_figures.py     # 生成上述两张图的脚本
└── np_hmwk_E1.py … np_hmwk_E8.py    # 课前 12 题热身（不提交，仅存档）
```

---

## 2. 环境

| 组件 | 版本 | 说明 |
| --- | --- | --- |
| Python | 3.12.10 | `.venv` 虚拟环境 |
| NumPy | 2.5.4 | 全部计算只用它 |
| Pillow | 12.3.0 | 读图与预处理 |
| imageio | 2.38.2 | 课堂示例用，`check.py` 不依赖 |

不需要 GPU、PyTorch、OpenCLIP 或网络。

```bash
python -m venv .venv                     # 若还没有虚拟环境
.venv/Scripts/python.exe -m pip install -r project1-numpy-student/requirements.txt
```

> Windows 下如果终端输出中文变乱码，先执行 `set PYTHONIOENCODING=utf-8`（PowerShell 用
> `$env:PYTHONIOENCODING="utf-8"`）。数值检查本身不受影响。

---

## 3. 复现与验证

所有命令都在 `project1-numpy-student/` 目录下执行：

```bash
cd project1-numpy-student

python check.py --stage assets        # 权重张量数与图像/reference 配对
python check.py --stage preprocess    # 预处理输出
python check.py --stage patches       # TODO-01
python check.py --stage tokens        # TODO-02 + TODO-05
python check.py --stage block0        # TODO-03 + TODO-04（第一层）
python check.py --stage last_tokens   # 12 层全部
python check.py --stage output        # TODO-06
python check.py --stage batch         # 换顺序 / 重复图片
python check.py --stage all           # 以上全部
```

一次完整运行为：

```bash
python check.py --stage all
python run.py --output output_tokens.npz
```

`run.py` 按输入顺序把图像 ID 与 512 维 token 存进 npz：

```bash
python run.py --images assets/images/ILSVRC2012_val_00000002.JPEG --output one.npz
```

### 预期输出（实测）

```
assets       PASS: 152 matching visual tensors and ten image/reference pairs
preprocess   PASS shape=(10, 3, 224, 224) max_abs=0.000e+00 mean_abs=0.000e+00 (atol=2e-06, rtol=0.0002)
patches      PASS shape=(2, 49, 3072)     max_abs=0.000e+00 mean_abs=0.000e+00 (atol=0.0002, rtol=0.0002)
tokens       PASS shape=(2, 50, 768)      max_abs=4.005e-05 mean_abs=1.289e-06 (atol=0.0002, rtol=0.0002)
block0       PASS shape=(2, 50, 768)      max_abs=7.063e-06 mean_abs=3.250e-07 (atol=0.0002, rtol=0.0002)
last_tokens  PASS shape=(2, 50, 768)      max_abs=4.768e-05 mean_abs=2.696e-06 (atol=0.0002, rtol=0.0002)
output       PASS shape=(10, 512)         max_abs=5.439e-07 mean_abs=8.311e-08 (atol=2e-05,  rtol=0.0002)
batch        PASS shape=(3, 512)          max_abs=4.619e-07 mean_abs=7.632e-08 (atol=2e-05,  rtol=0.0002)
```

> ⏱ `last_tokens` / `output` / `batch` / `all` 需要 1–2 分钟。`helpers.py` 的 GELU 是
> `math.erf` 的 Python 循环实现（精确版，共约 1800 万次调用），慢但结果与参考完全同源。

---

## 4. 实现要点

| TODO | 方法 | 一句话 | 形状 |
| --- | --- | --- | --- |
| 01 | `ImageEncoder.patchify` | 6 轴 reshape 后把网格轴提到前面 | (B,3,224,224) → (B,49,3072) |
| 02 | `MultiHeadAttention.split_heads` | 768 切 12 段并让 head 成为批次轴 | (B,50,768) → (B,12,50,64) |
| 03 | `MultiHeadAttention.__call__` | q·kᵀ/√d → softmax → 加权 v → 合头 → out_proj | (B,12,50,50) → (B,50,768) |
| 04 | `TransformerBlock.__call__` | pre-LN 两条残差 + 4 倍 MLP(GELU) | (B,50,768) 不变 |
| 05 | `ImageEncoder.embed` | 无 bias 投影 + CLS 拼在 token 轴 + 位置 | (B,49,3072) → (B,50,768) |
| 06 | `ImageEncoder.__call__` | 取 CLS → post_norm → 768→512 → 单位化 | (B,50,768) → (B,512) |

三条容易踩的约定：

1. patch 内部摊平顺序是 **C→H→W**，与 `visual.conv1.weight.reshape(768,3072)` 对齐；
2. 除 `visual.proj`（已是 `[in,out]`）外，所有线性权重存 `[out,in]`，用前要 `.T`；
3. 图像塔**没有 LayerScale**，残差直接相加。

---

## 5. 为什么仓库里没有权重文件

- `weights/image_encoder.npz`：312 MB，超过 GitHub 单文件 100 MB 限制，已在 `.gitignore` 排除；
- `assets/reference_outputs.npz`：7.4 MB 参考数据，同样排除。

复现时请从课程附件中取得这两个文件，放回原位（`project1-numpy-student/weights/` 与
`project1-numpy-student/assets/`）再执行第 3 节命令。**不要**把它们提交到仓库。

---

## 6. 实验报告

`Experiment-Report-Template-SZU-main/main.tex`，用 **XeLaTeX** 编译（Overleaf 里
`Menu → Compiler → XeLaTeX`），正文（不含封面）不超过 5 页。图 1、图 2 由
`tools/make_report_figures.py` 生成（只依赖 Pillow）：

```bash
.venv/Scripts/python.exe tools/make_report_figures.py
```
