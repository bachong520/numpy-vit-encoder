import numpy as np
A=np.array([[0.20,-0.50,1.10,0.40],
            [0.80,0.30,-0.70,0.60],
            [-0.40,0.90,0.10,-0.20]])
print(f"A的秩：{A.ndim}")
print(f"A的形状：{A.shape}")
print(f"A的元素个数：{A.size}")

a_row2=A[1]
a_list_last2=A[:,-2:]
print(f"A的第二行\n{a_row2}")
print(f"A的最后两列\n{a_list_last2}")

B=A.copy()
B[B<0]=0 #负数改为0
print(f"A的副本（负数修改为0）：\n{B}")
print(f"A未改变：\n{A}")