import numpy as np

def distance(a,b):
    d=a-b
    print(f"a与b的差向量为{d}")
    euc_d=np.sqrt(np.sum(d**2))
    print(f"a与b的欧氏距离为{round(euc_d,6)}")
    man_d=np.sum(np.abs(d))
    print(f"a与b的曼哈顿距离{round(man_d,6)}")

p=np.array([-0.50,1.20,2.00])
q=np.array([0.80,-0.40,1.10])
distance(p,q)