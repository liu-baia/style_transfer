# 赛博画师 —— 风格迁移（Style Transfer）

用预训练 **VGG19** 当特征提取器，把「风格图」的画风迁移到「内容图」上。

**被优化的是生成图的像素，VGG19 的参数全程冻结、不参与训练。**

优化器用 **LBFGS**（二阶 + 强 Wolfe 线搜索）而不是 Adam —— 因为被优化的"参数"只有一张图的像素，属于"参数少、要求精确"的场景。

---

## 目录结构

```
style_transfer/
├── src/                                   代码（4 个脚本）
│   ├── style_transfer_LBFGS_nomask.py     基础任务：单风格迁移
│   ├── style_transfer_ctrl.py             附加题 1：风格强度控制
│   ├── style_transfer_multi.py            附加题 2：多风格融合
│   └── style_transfer_mask.py             附加题 3：局部风格迁移
│
├── pictures/                              基础任务的输入（3 个 set）
│   ├── set1/{content_image, style_image}/     星空（油画）
│   ├── set2/{content_image, style_image}/     二次元动画
│   └── set3/{content_image, style_image}/     夕阳
├── output/                                基础任务的输出
│   └── setN/{result.png, loss.csv, compare/}
│
├── pictures_multi/                        附加题 2 的输入（1 内容 + 2 风格）
│   ├── content_image/绣球花.jpg
│   └── style_image/{01_二次元.jpg, 02_夕阳.jpeg}
├── output_multi/
│   ├── w_0.25/  w_0.50/  w_0.75/          三组不同权重
│   └── summary.csv
│
├── pictures_mask/                         附加题 3 的输入（含手工 mask）
│   └── set1/{content_image, style_image, mask/mask.png}
├── output_mask/
│   └── set1/{result.png, loss.csv}
│
├── docs/赛博画师.pdf                      实验报告（含题目解答）
└── style_to.txt                           三套 set 对应的风格说明
```

---

## 环境

| 项 | 版本 |
|---|---|
| Python | 3.11 |
| PyTorch | 2.11.0+cu128 |
| torchvision | 0.26.0+cu128 |
| Pillow | 随 torchvision |

```bash
pip install -r requirements.txt
```

**首次运行会自动下载 VGG19 预训练权重（约 548 MB）**，存到 `~/.cache/torch/`，
不会提交到仓库。

---

## 运行

所有脚本都**不需要命令行参数**，想改参数直接改文件开头的配置区。

### 基础任务：单风格迁移

```bash
cd src
python style_transfer_LBFGS_nomask.py
```

扫描 `pictures/` 下的所有 set（每个 set 需要 `content_image/` 和 `style_image/` 各一张图），
逐个生成到 `output/setN/`。

| 输出 | 内容 |
|---|---|
| `result.png` | 最终生成图 |
| `loss.csv` | 每步的内容/风格/TV 损失及加权值 |
| `compare/` | `iter_0000_content.png`（起点）+ 对数间隔的过程图 |

### 附加题 1：风格强度控制

```bash
python style_transfer_ctrl.py
```

用 `STRENGTH_LEVELS` 扫不同的风格强度。强度 `s ∈ [0,1]` 按**对数间隔**映射到权重：

$$
\beta = 10^{\,2 + 2s}
$$

输出 `output_ctrl/setN/strength_<s>/{result.png, loss.csv}` + `summary.csv`。

### 附加题 2：多风格融合

```bash
python style_transfer_multi.py
```

同时使用**两张风格图**，风格损失拆成两项：

$$
L_{style} = \beta_1 L_{style}^{(1)} + \beta_2 L_{style}^{(2)},
\qquad
\beta_1 = \beta_{total}\cdot w,\quad \beta_2 = \beta_{total}\cdot(1-w)
$$

用 `WEIGHT_RATIOS` 扫三组权重。输出 `output_multi/w_<w>/{result.png, loss.csv}` + `summary.csv`。

### 附加题 3：局部风格迁移

```bash
python style_transfer_mask.py
```

只让 mask 覆盖的区域（花朵）被风格化，**背景逐像素保持不变**。

- 内容损失只在 mask 内算
- 风格损失：特征图先乘 mask 再算 Gram，归一化用 mask 有效面积
- 输出时硬合成：`result = gen·mask + 原图·(1-mask)`

---

## 核心实现要点

### 1. 三个损失

| 损失 | 比较对象 | 取自 | 关心位置吗 |
|---|---|---|---|
| **内容** | 特征图本身（均方误差） | relu4_2 | **关心** |
| **风格** | **Gram 矩阵**（均方误差） | relu1_1 ~ relu5_1（5 层） | **完全不关心** |
| **TV** | 相邻像素差的**绝对值** | 生成图自身 | 关心（局部） |

$$
L_{total} = \alpha L_{content} + \beta L_{style} + \gamma L_{tv}
$$

默认 $\alpha=1.0$、$\beta=10^4$、$\gamma=10^{-4}$。

**注意 TV 用绝对值（L1）而不是平方**：TV 的目的是"清除孤立噪点"，需要**一视同仁**地压制所有非零跳变（L1 的梯度恒为 ±1）；而内容/风格是"追赶目标"，需要差距越大修正越多（L2 的梯度正比于差距）。

### 2. VGG19 的层索引

`.features` 是 `nn.Sequential`，**ReLU 和 Pool 也占索引**，共 37 层：

| 层名 | 索引 | 用途 |
|---|---|---|
| relu1_1 | 1 | 风格 |
| relu2_1 | 6 | 风格 |
| relu3_1 | 11 | 风格 |
| relu4_1 | 20 | 风格 |
| **relu4_2** | **22** | **内容** |
| relu5_1 | 29 | 风格 |

**要取 ReLU 之后的值**（`relu4_2` = 22），不是 Conv2d 的输出（`conv4_2` = 21）。
取错会让损失大 30 倍，而且优化器会卡死。

**验证方法**：生成图 = 内容图时（第 0 步），内容损失必须为 0。不是 0 就说明取层错了。

### 3. 生成图存在「像素空间」

```python
gen = to_pixel(content).clone().requires_grad_(True)   # 像素 [0,1]

def closure():
    gen.clamp_(0, 1)                  # 夹回合法像素范围（语义正确）
    x = to_norm(gen)                  # 送进 VGG 前才归一化
    ...
```

**不能反过来**（存归一化值却 clamp 到 `[0,1]`）：因为

$$
\text{归一化} = \frac{\text{像素} - 0.485}{0.229}
$$

所以对归一化值 clamp `[0,1]` 等价于把像素限制在 `[0.485, 0.714]`，
画面会缩成灰色窄带，**对比度损失约 42%**。

另外 `optimizer.step()` 之后 `gen` 会再次越界，**保存前必须再 clamp 一次**，
否则越界值转 uint8 时被截断成孤立极值像素 —— 就是彩色噪点。

### 4. 为什么需要 `closure`

LBFGS 每步内部要做**线搜索**（沿方向反复试步长），每试一次都要重算损失，
所以损失必须包成可反复调用的 `closure` 并 `return loss`。

---

## 实验报告

完整解答（含 12 道代码细节题、三个附加题的实验数据与分析）见
[`docs/赛博画师.pdf`](docs/赛博画师.pdf)。

---

## 说明

- 生成图尺寸默认 768×768，100 步，单张约 3~4 分钟（RTX 5070 Laptop）。
- `pictures/set2/style_image/` 为二次元风格插画，仅用于课程作业学习，未作商用。
