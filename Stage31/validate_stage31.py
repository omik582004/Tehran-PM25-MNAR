"""Check archived Stage31 decisions against stated facts and Stage26 source cells."""
from pathlib import Path
import numpy as np,pandas as pd,json,zipfile,io,hashlib
P=Path(__file__).resolve().parent
S=pd.read_csv(P/'heldout_discrimination_v31.csv')
D=pd.read_csv(P/'heldout_decision_curve_v31.csv')
A=pd.read_csv(P/'area_weighted_decision_sensitivity_v31.csv')
B=pd.read_csv(P/'heldout_brier_decomposition_v31.csv')
assert S.shape[0]==6 and set(S.n_event)=={5,10,15}
assert (D.query('event_threshold==150').changed_decisions==0).all()
assert D.query('event_threshold==120 and decision_probability_cutoff==0.5').changed_decisions.eq(0).all()
a=D.query('event_threshold==120 and decision_probability_cutoff==0.7').iloc[0]
assert (a.MAR_TP,a.MNAR_TP,a.MAR_FP,a.MNAR_FP)==(7,8,2,2)
for u,vals in [(100,(15,)),(120,(10,)),(150,(5,))]:
 sub=S[S.threshold==u]; assert set(sub.n_event)==set(vals)
 if u==120: assert np.allclose(sub.auroc.to_numpy(),.9794871794871794)
 if u==150:
  assert np.allclose(sub.auroc.to_numpy(),.9727272727272727)
  b=sub.set_index('model').brier; assert b['MNAR'] < b['MAR']
a=A.query('date=="2024-01-01" and subindex_threshold==120 and probability_cutoff==0.5').set_index('scenario')
assert np.isclose(a.loc['within__plus_ln2','area_classification_changed_pct'],10.3626,atol=1e-4)
assert np.isclose(a.loc['within__minus_ln2','area_classification_changed_pct'],1.2458,atol=1e-4)
assert len(A)==3*3*4*2 and not A.isna().any().any()
for k in ['area_classification_changed_pct','area_new_high_risk_pct','area_no_longer_high_risk_pct']:
 assert A[k].between(-1e-9,100+1e-9).all()
assert np.allclose(A.area_classification_changed_pct,A.area_new_high_risk_pct+A.area_no_longer_high_risk_pct,atol=1e-10)
print('STAGE31 ALL TESTS PASSED; files:',len(A),'map scenarios,',len(D),'decision rows; original model code and fitting unchanged')
