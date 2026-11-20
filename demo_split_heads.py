"""小实验：多头是怎么"拆"出来的 —— [B,N,D] -> [B,H,N,D/H]。

玩具参数：B=2 张图、N=3 个 token、D=4 维特征、H=2 个头、每个头 dh=2 维。
真实模型是 B=10、N=50、D=768、H=12、dh=64，拆法一模一样，只是数字变大。
"""
import numpy as np

np.set_printoptions(precision=0, suppress=True, linewidth=140)


def toy():
    B, N, D, H = 2, 3, 4, 2
    dh = D // H                                   # 每个头分到 2 维

    # 用"B和N"编号当数值，一眼能看出谁挪到哪去了：
    # x[b, n, d] = b*10 + n*1 + d*0.1  → 只用来标识位置
    x = np.arange(B * N * D, dtype=np.float32).reshape(B, N, D)
    print('=' * 70)
    print(f'输入 x: shape={x.shape} = [B={B}, N={N}, D={D}]')
    print('=' * 70)
    print(x[0], '\n↑ 只看第 0 张图：3 行 token，每行 4 个特征')

    print('\n--- 正确做法：reshape 拆维度 → transpose 换位置 ---')
    a = x.reshape(B, N, H, dh)
    print(f'① reshape(B,N,H,dh) → {a.shape}   轴序=(B, 令牌, 头, 头内特征)')
    b = a.transpose(0, 2, 1, 3)
    print(f'② transpose(0,2,1,3) → {b.shape}   轴序=(B, 头, 令牌, 头内特征)')
    print('第 0 张图、第 0 个头（前 2 维特征）:\n', b[0, 0])
    print('第 0 张图、第 1 个头（后 2 维特征）:\n', b[0, 1])

    print('\n--- 手工对照：直接按列切片再堆起来 ---')
    manual = np.stack([x[:, :, h*dh:(h+1)*dh] for h in range(H)], axis=1)
    print(f'manual shape={manual.shape}')
    print('和 reshape+transpose 完全相等:', np.array_equal(manual, b))

    print('\n--- 反例：只 reshape(0,H,N,dh)，不 transpose ---')
    wrong = x.reshape(B, H, N, dh)
    print(f'shape 恰好也是 {wrong.shape}，但内容:\n{wrong[0, 0]}')
    print('↑ 这不是"第 0 个头的 3 个 token"，而是把 12 个数硬切成 3 行 2 列。')

    print('\n--- 合并回去（注意力算完之后要做的事）---')
    back = b.transpose(0, 2, 1, 3).reshape(B, N, D)
    print('合并后和原输入相等:', np.array_equal(back, x))


def real_shapes():
    B, N, D, H = 10, 50, 768, 12
    dh = D // H
    print('\n' + '=' * 70)
    print('真实形状')
    print('=' * 70)
    print(f'[B={B}, N={N}, D={D}]  →  reshape({B},{N},{H},{dh})  →  transpose(0,2,1,3)')
    print(f'                        →  [{B}, {H}, {N}, {dh}]')
    print(f'每个头分到 {dh} 维，12 个头合计 {H*dh} 维 = 原来的 D')
    print('\n为什么必须把头挪到第 1 维？')
    print('  后面要算 scores = q @ swapaxes(k,-1,-2)，即每张图、每个头各算一张 N×N 的表。')
    print(f'  把 (B,H) 放在最前面当前两维 → 矩阵乘法自动对 (B,H) 逐个并行处理，')
    print('  得到 scores=[B,H,N,N]；如果头在中间，就对不上了。')


if __name__ == '__main__':
    toy()
    real_shapes()
