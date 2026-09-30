"""Global mean-Frobenius alignment for the synthetic 80S evaluation.

Choose among the first min(100,N) particle-derived frame corrections and both
whole-dataset hands. Truth is used only for metric calculation. Fit and report
on all N particles, matching the recorded 80S evaluation convention.
"""
import numpy as np

def angles(p,t):
    pv=p[:,2,:]; tv=t[:,2,:]
    cos=np.sum(pv*tv,axis=1)/(np.linalg.norm(pv,axis=1)*np.linalg.norm(tv,axis=1))
    view=np.degrees(np.arccos(np.clip(cos,-1,1)))
    full=np.degrees(np.arccos(np.clip((np.einsum('nij,nij->n',p,t)-1)/2,-1,1)))
    return view,full

def fit(p,t):
    best={}
    for flip in (False,True):
        s=np.diag([1.,1.,-1. if flip else 1.]); pp=s@p
        for i in range(min(100,len(p))):
            q=p[i].T@s@t[i]; a=pp@q
            sq=np.sum((a-t)**2,axis=(1,2))
            scores={'paper_mean_frobenius':np.sqrt(sq).mean(),'upstream_median_squared_frobenius':np.median(sq)}
            for key,score in scores.items():
                if key not in best or score<best[key][0]: best[key]=(float(score),flip,i,q)
    result={}
    for key,(score,flip,i,q) in best.items():
        a=np.diag([1.,1.,-1. if flip else 1.])@p@q; v,f=angles(a,t)
        result[key]=dict(alignment_score=score,hand_flip=flip,candidate=i,right_alignment=q.tolist(),view_mean_deg=float(v.mean()),view_median_deg=float(np.median(v)),view_p90_deg=float(np.quantile(v,.9)),view_fraction_below5=float(np.mean(v<5)),so3_mean_deg=float(f.mean()))
    return result

def validate():
    from scipy.spatial.transform import Rotation
    t=np.broadcast_to(np.eye(3),(1,3,3))
    for axis,theta,view in [('x',20,20),('z',20,0)]:
        v,f=angles(Rotation.from_euler(axis,[theta],degrees=True).as_matrix(),t)
        np.testing.assert_allclose(v,[view],atol=1e-5); np.testing.assert_allclose(f,[theta],atol=1e-5)
    t=Rotation.random(120,random_state=1).as_matrix(); q=Rotation.random(random_state=2).as_matrix()
    for s in [np.eye(3),np.diag([1,1,-1])]:
        p=s@t@q@s
        r=fit(p,t)
        assert r['paper_mean_frobenius']['view_mean_deg']<1e-5
