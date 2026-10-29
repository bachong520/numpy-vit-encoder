import numpy as np

def unit_vector(a):
    len_a=np.linalg.norm(a)
    if len_a==0:
        return ValueError("输入为零向量，无法单位化")
    e_a=np.array(a/len_a)
    return e_a

a=np.array([-1.20,0.40,0.80])
try:
    e=unit_vector(a)
    print(f"a的单位向量为{np.round(e,6)}")
    print(f"allclose检验单位向量长度为1：{np.allclose(np.linalg.norm(e),1.0)}")
except ValueError as e:
    print("a为零向量，无法单位化")