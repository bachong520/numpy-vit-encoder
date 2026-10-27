import numpy as np
a=np.array([0.80,-0.60,0.70])
b=np.array([-0.20,0.90,0.40])

ab_elem=a*b
ab_dot=a@b
print("a*b=",ab_elem)
print("其shape为",ab_elem.shape) #shape应为(3,)
print("a@b=",round(ab_dot,6))
print("其shape为",ab_dot.shape)  #shape应为(),即一维向量互乘得到一个数

len_a=np.sqrt(np.sum(a**2))
len_b=np.sqrt((b*b).sum())  #各元素自相乘，即取平方
print("a的欧式模为",round(len_a,6))
print("b的欧式模为",round(len_b,6))

tol=1e-6
print("核对a的模：",abs(len_a-np.linalg.norm(a))<tol)
print("核对b的模：",abs(len_b-np.linalg.norm(b))<tol)