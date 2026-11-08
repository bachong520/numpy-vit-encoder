import numpy as np
a=np.array([[[0.20,-0.40],[0.70,0.10]],
            [[0.30,0.80],[-0.20,0.50]],
            [[-0.10,0.60],[0.40,-0.30]]])
b=a+0.10
batch=np.stack([a,b],axis=0)
print(batch.shape)
blue=batch[:,2]
print(blue.shape)
bhwc=batch.transpose(0,2,3,1)   #变换
print(bhwc.shape)