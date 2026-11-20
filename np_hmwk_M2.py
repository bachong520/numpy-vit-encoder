import numpy as np

def top_k_cosine(query,gallery,k):
    query=np.asarray(query,dtype=float)
    gallery=np.asarray(gallery,dtype=float)
    l_q=np.linalg.norm(query)
    if l_q==0.0:
        raise ValueError("query为零向量")
    gn=np.linalg.norm(gallery,axis=1)
    valid=gn>0.0
    n_valid=int(valid.sum())
    if not isinstance(k,(int,np.integer))or k<=0 or k>n_valid:
        raise ValueError(f"k必须为正整数且不超过有效行数{n_valid}")
    idx_valid=np.flatnonzero(valid)
    sims=(gallery[valid]@query)/(gn[valid]*l_q)
    order=np.argsort(-sims,kind="stable")
    chosen=idx_valid[order[:k]]
    return chosen,sims[order[:k]]

query=np.array([0.60,-0.80,0.00])
gallery=np.array([[0.60,-0.80,0.00],
                  [0.30,-0.40,0.00],
                  [0.80,0.60,0.00],
                  [-0.60,0.80,0.00],
                  [0.00,0.00,0.00]])
ids=np.array(["img_a","img_b","img_c","img_d","img_zero"])
idx,sim=top_k_cosine(query,gallery,3)
print("选中索引为",idx)
print("相似度为",np.round(sim,6))
print("选中ID为",ids[idx])