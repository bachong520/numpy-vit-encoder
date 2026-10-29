import numpy as np

A=np.array([[0.20,-0.50,1.10],
            [0.80,0.40,-0.30]])
c_sum=A.sum(axis=0)
r_mean=A.mean(axis=1)
gmax=A.max()
print(f"A.sum(axis=0)={c_sum},shape为{c_sum.shape}")
print(f"A.mean(axis=1)={np.round(r_mean,6)},shape为{r_mean.shape}")
print(f"A中最大值为{round(gmax,6)}")