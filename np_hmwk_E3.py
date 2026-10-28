import numpy as np

def angle_between(a,b):
    a=np.asarray(a,dtype=float)
    b=np.asarray(b,dtype=float)
    na,nb=np.linalg.norm(a),np.linalg.norm(b)
    if(na==0.0 or nb==0.0):
        return ValueError("零向量无法与其他向量形成夹角")
    cos=(a@b)/(na*nb)
    cos=np.clip(cos,-1.0,1.0)
    return float(np.degrees(np.arccos(cos)))

a=np.array([1.00,-1.00])
b=np.array([1.00,1.00])
print(f"a与b间的夹角为{angle_between(a,b)}°")