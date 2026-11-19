"""Stage31 secondary post-hoc validation using previously accepted stage29 posterior predictive draws and stage26 area posterior summaries.
No fitting, no changing withheld outcomes, and no use of inadmissible chains. Exact event is Y>u.
"""
from pathlib import Path
import numpy as np,pandas as pd,zipfile,io,json
from sklearn.metrics import roc_auc_score, average_precision_score
O=Path(__file__).resolve().parent
if (O/'stage29_archive.zip').exists():
 z=zipfile.ZipFile(O/'stage29_archive.zip')
else:
 parent=zipfile.ZipFile(O/'Tehran_MNAR_Stage30_TailScore_Analysis.zip')
 key=next(n for n in parent.namelist() if n.endswith('Tehran_MNAR_Stage29_Validation_Manuscript_Code.zip'))
 z=zipfile.ZipFile(io.BytesIO(parent.read(key)))
raw={};ix={}
for m in ('G_MAR','G_MNAR'):
 d=np.load(io.BytesIO(z.read(f'Stage29/results/{m}/heldout_posterior_predictive.npz')))
 raw[m]=d['predictive_readings'];ix[m]=d['heldout_flat_index'];assert raw[m].shape[1]==49
assert np.array_equal(ix['G_MAR'],ix['G_MNAR']);index=ix['G_MAR'];days=index//35
truth=pd.read_csv(io.BytesIO(z.read('masked_holdout_truth_DO_NOT_LOAD_IN_FIT.csv'))).set_index('flat_index');y=truth.loc[index,'true_reported_subindex'].to_numpy()
counts={int(i):int(np.sum(days==i)) for i in np.unique(days)}
prs={(m,u):(r>u).mean(axis=0) for m,r in raw.items() for u in [100,120,150]}
thresholds=(.05,.10,.20,.25,.30,.50,.70,.80)
D=[];S=[];C=[]
B=20000;rng=np.random.default_rng(20261031);dblocks=[np.where(days==d)[0] for d in sorted(set(days))];boots=np.array([np.concatenate([dblocks[j] for j in rng.integers(0,len(dblocks),size=len(dblocks))]) for _ in range(B)],dtype=object)
# robust bootstrap with list of index arrays varying lengths
for u in [100,120,150]:
 event=(y>u); n_ev=int(event.sum());p0=prs['G_MAR',u];p1=prs['G_MNAR',u]
 for m,p in [('MAR',p0),('MNAR',p1)]:
  S.append(dict(threshold=u,model=m,n=49,n_event=n_ev,brier=np.mean((p-event)**2),auroc=roc_auc_score(event,p),average_precision=average_precision_score(event,p),mean_event_probability=p[event].mean(),mean_nonevent_probability=p[~event].mean()))
 delta=((p1-event)**2-(p0-event)**2)
 for grp,sel in [('event',event),('nonevent',~event),('all',np.ones(len(y),bool))]:
  vals=[]
  for ii in boots:
   ii=np.asarray(ii,dtype=int);j=ii[sel[ii]]
   if len(j):vals.append(delta[j].mean())
  C.append(dict(threshold=u,group=grp,n=int(sel.sum()),delta_brier=delta[sel].mean(),ci_low=np.quantile(vals,.025),ci_high=np.quantile(vals,.975),valid_boot=len(vals)))
 for q in thresholds:
  d0=p0>=q; d1=p1>=q; joint=(d0!=d1)
  a={}
  for model,d in [('MAR',d0),('MNAR',d1)]:
   tp=int(np.sum(d&event));fp=int(np.sum(d&~event));fn=int(np.sum(~d&event))
   a[model]=(tp,fp,fn,(tp-fp*q/(1-q))/len(y))
  pointdelta=a['MNAR'][-1]-a['MAR'][-1]
  bd=[]
  for ii in boots:
   ii=np.asarray(ii,dtype=int)
   nb1=(np.sum(d1[ii]&event[ii])-np.sum(d1[ii]&~event[ii])*q/(1-q))/len(ii)
   nb0=(np.sum(d0[ii]&event[ii])-np.sum(d0[ii]&~event[ii])*q/(1-q))/len(ii)
   bd.append(nb1-nb0)
  D.append(dict(event_threshold=u,decision_probability_cutoff=q,events=n_ev,changed_decisions=int(joint.sum()),MAR_TP=a['MAR'][0],MAR_FP=a['MAR'][1],MAR_FN=a['MAR'][2],MNAR_TP=a['MNAR'][0],MNAR_FP=a['MNAR'][1],MNAR_FN=a['MNAR'][2],MAR_net_benefit=a['MAR'][-1],MNAR_net_benefit=a['MNAR'][-1],delta_net_benefit=pointdelta,delta_ci_low=np.quantile(bd,.025),delta_ci_high=np.quantile(bd,.975),bootstrap_replicates=B))
pd.DataFrame(D).to_csv(O/'heldout_decision_curve_v31.csv',index=False)
pd.DataFrame(S).to_csv(O/'heldout_discrimination_v31.csv',index=False)
pd.DataFrame(C).to_csv(O/'heldout_brier_decomposition_v31.csv',index=False)
# spatial decisions from independently accepted stage26 cell probabilities, using source mesh and weights
df=pd.read_csv(O/'accepted_cellwise_sensitivity.csv')
A=[]
for u in [100,120,150]:
 for date in sorted(df.date.unique()):
  subset=df[(df.date==date)&(df.threshold_subindex==u)&df.scenario.isin(['within__MAR','within__minus_ln2','within__plus_ln2'])]
  pt=subset.pivot(index='cell_id',columns='scenario',values='posterior_exceedance_probability');w=subset.drop_duplicates('cell_id').set_index('cell_id').loc[pt.index,'cell_weight']; assert len(pt)==236 and not pt.isna().any().any() and np.isclose(w.sum(),1)
  for nm in ('within__MAR','within__minus_ln2','within__plus_ln2'):
   assert np.allclose(subset[subset.scenario==nm].set_index('cell_id').loc[pt.index,'cell_weight'].to_numpy(),w.to_numpy())
  for other in ('within__minus_ln2','within__plus_ln2'):
   for q in (.25,.50,.75,.90):
    b=pt['within__MAR']>=q; c=pt[other]>=q;changed=(b!=c);up=(~b&c);down=(b&~c)
    A.append(dict(date=date,subindex_threshold=u,probability_cutoff=q,scenario=other,area_classification_changed_pct=100*w[changed].sum(),area_new_high_risk_pct=100*w[up].sum(),area_no_longer_high_risk_pct=100*w[down].sum(),mar_high_risk_pct=100*w[b].sum(),scenario_high_risk_pct=100*w[c].sum(),n_cells=int(changed.sum())))
pd.DataFrame(A).to_csv(O/'area_weighted_decision_sensitivity_v31.csv',index=False)
# area event decision values are independent stage26 verified accepted summary
p=pd.read_csv(O/'Tehran_Accepted_Sensitivity_v26.csv');Q=p[(p.threshold==120)&(p.date=='2024-01-01')&p.scenario.isin(['within__MAR','within__minus_ln2','within__plus_ln2'])].copy();Q['decision_probability_cutoff']=.98;Q['illustrative_decision']=Q['event_prob_gt_30pct']>=.98;Q.to_csv(O/'regional_event_tipping_illustration_v31.csv',index=False)
M={'holdout_n':49,'calendar_days':len(dblocks),'heldout_event_counts':{u:int(np.sum(y>u)) for u in (100,120,150)},'decision_grid':list(thresholds),'caution':'Secondary, post-hoc; not an independent validation or evidence for true MNAR; observational outcomes are reported readings R=1.'}
(O/'stage31_manifest.json').write_text(json.dumps(M,indent=2))
print('event summaries');print(pd.DataFrame(S).round(6).to_string(index=False))
print('\nBrier deltas');print(pd.DataFrame(C).round(6).to_string(index=False))
print('\nDecision 120 150');print(pd.DataFrame(D).query('event_threshold>=120').round(5).to_string(index=False))
print('\nSpatial Jan1 120');print(pd.DataFrame(A).query('date=="2024-01-01" and subindex_threshold==120').round(4).to_string(index=False))
print('\nTipping Jan1');print(Q[['scenario','event_prob_gt_30pct','decision_probability_cutoff','illustrative_decision']].to_string(index=False))
