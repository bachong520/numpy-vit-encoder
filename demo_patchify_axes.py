"""小实验：把图片切成 patch，看清 reshape / transpose 的每个轴在干什么。

这个文件只是学习用的，不影响作业（作业只改 image_encoder.py）。
随便改数字、打印、删掉，都不会影响 check.py 的结果。
"""
import numpy as np

np.set_printoptions(precision=0, suppress=True, linewidth=130)


def part1_one_channel():
    """单通道 4x4 图，patch 2x2 → 4 个 patch，每个 patch 摊成 4 个数。"""
    img = np.arange(16, dtype=np.float32).reshape(4, 4)
    print('=' * 66)
    print('实验一：4x4 的图，patch 大小 2x2（→ 2x2=4 个 patch，每个 4 个数）')
    print('=' * 66)
    print('原图（数字就是它"在哪"）:\n', img, sep='')

    gh = gw = 2      # 网格 2x2
    p = 2            # 每个 patch 2x2

    # 第 1 步：把 (H,W) 拆成 (网格行, patch内行, 网格列, patch内列)
    x = img.reshape(gh, p, gw, p)
    print(f'\nreshape({gh},{p},{gw},{p}) 后 shape={x.shape}')
    print('x[0, :, 1, :] = 第 0 个网格行、第 1 个网格列的那个 patch =', x[0, :, 1, :].reshape(2, 2))

    # 第 2 步：把"两个网格轴"提到前面、patch 内部轴压到后面
    y = x.transpose(0, 2, 1, 3)
    print(f'transpose(0,2,1,3) 后 shape={y.shape}')

    # 第 3 步：摊平 → (patch数, 每patch的数值数)
    tokens = y.reshape(4, 4)
    print('\n最终 tokens shape=', tokens.shape)
    print(tokens)

    print('\n手工核对（用切片切出每个 patch，和老办法比）:')
    for i in range(2):
        for j in range(2):
            manual = img[i*p:(i+1)*p, j*p:(j+1)*p].reshape(-1)
            k = i * 2 + j
            ok = np.array_equal(manual, tokens[k])
            print(f'  patch({i},{j}) 行{i}列{j} → tokens[{k}] = {manual}  {"✓" if ok else "✗"}')

    # 反例：忘了 transpose 会怎样
    wrong = img.reshape(4, 4)
    print('\n反例：只 reshape、不 transpose，拿到的 4 行是:')
    print(wrong, '\n→ 每行是"图的整行"，不是"一个 patch"，所以 transpose 不能省。')


def part2_with_channels():
    """带通道和 batch：形状 [B,C,H,W] → [B, patch数, C*p*p]。"""
    B, C, H, W, p = 2, 3, 4, 4, 2
    rng = np.random.default_rng(0)
    images = rng.integers(0, 100, (B, C, H, W)).astype(np.float32)
    gh, gw = H // p, W // p

    print('\n' + '=' * 66)
    print(f'实验二：真正作业的形状 [B,C,H,W] = {images.shape}，p={p}')
    print('=' * 66)
    x = images.reshape(B, C, gh, p, gw, p)
    print(f'reshape → {x.shape}   轴序: (B, C, 网格行, patch内行, 网格列, patch内列)')
    y = x.transpose(0, 2, 4, 1, 3, 5)
    print(f'transpose(0,2,4,1,3,5) → {y.shape}   轴序: (B, 网格行, 网格列, C, patch内行, patch内列)')
    tokens = y.reshape(B, gh * gw, C * p * p)
    print(f'reshape → {tokens.shape}   = [B, 49, 3072] 的同款结构（小号版）')

    manual = []
    for i in range(gh):
        for j in range(gw):
            # 一个 patch：先取所有通道的这块区域，再按 C,H,W 顺序摊平
            block = images[0, :, i*p:(i+1)*p, j*p:(j+1)*p]
            manual.append(block.reshape(-1))
    manual = np.stack(manual)
    print(f'手工切片的 patch 矩阵 shape={manual.shape}')
    print('和 reshape+transpose 的结果是否完全相等:', np.array_equal(manual, tokens[0]))


def part3_real_shape():
    """真实作业的形状，只看 shape 不跑数值。"""
    print('\n' + '=' * 66)
    print('实验三：真实形状 224x224，patch 32x32')
    print('=' * 66)
    B, C, H, W, p = 10, 3, 224, 224, 32
    gh, gw = H // p, W // p
    print(f'网格 = {gh} x {gw} = {gh*gw} 个 patch')
    print(f'每个 patch 的数值个数 = C*p*p = {C}*{p}*{p} = {C*p*p}')
    print(f'最终 → ({B}, {gh*gw}, {C*p*p})')


if __name__ == '__main__':
    part1_one_channel()
    part2_with_channels()
    part3_real_shape()
