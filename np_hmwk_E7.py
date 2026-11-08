import numpy as np
X=np.array([[0.80,-0.60,0.70],
            [-0.40,0.50,0.20]])
scale=np.array([1.50,0.50,-1.00])
bias=np.array([-0.10,0.20,0.30])
Y=X*scale+bias
print(Y)
print(Y.shape)