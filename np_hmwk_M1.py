import numpy as np

def angles_to(query,X):
    query=np.asarray(query,dtype=float)
    X=np.asarray(X,dtype=float)
    l_q=np.linalg.norm(query)
    if l_q==0.0:
        raise ValueError("查询向量query为零向量")
    row_X=np.linalg.norm(X,axis=1)
    dots=X@query
    cos=dots/(row_X*l_q)
    cos=np.clip(cos,-1,1)
    ang=np.degrees(np.arccos(cos))
    ang[row_X==0.0]=np.nan
    return ang

query=np.array([0.60,-0.80,0.00])
X=np.array([[0.30,-0.40,0.00],
            [-0.60,0.80,0.00],
            [0.80,0.60,0.00],
            [0.00,0.00,0.00]])
print(angles_to(query,X))