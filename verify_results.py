#!/usr/bin/env python3
"""Read-only verification of archived Tehran MNAR Stages 25--31.
Requires Python 3 and NumPy. Does not refit, alter source archives, or publish data.
Run from the unpacked release root: python verify_results.py
"""
import csv, hashlib, io, json, math, pathlib, zipfile
import numpy as np

R = pathlib.Path(__file__).resolve().parent

def check(c, msg):
    if not c: raise AssertionError(msg)
    print('PASS:', msg)

def jread(z, p): return json.loads(z.read(p).decode('utf-8'))
def csvread(z,p): return list(csv.DictReader(io.StringIO(z.read(p).decode('utf-8-sig'))))
def nested(parent, p): return zipfile.ZipFile(io.BytesIO(parent.read(p)))

class Directory:
    def __init__(self,stage): self.root=R/'stages'/str(stage)
    def read(self,name): return (self.root/name).read_bytes()
z25,z26,z29,z30,z31=map(Directory,(25,26,29,30,31))

# Verify original inputs, where the archived Stage25 panel matches the supplied CSV byte-for-byte.
original = (R/'ORIGINAL_INPUTS/Tehran_PM25_primary_35stations_2014_2025.csv').read_bytes()
check(hashlib.sha256(original).digest()==hashlib.sha256(z25.read('Tehran_Stage25/real_panel.csv')).digest(), 'Source CSV equals Stage25 real_panel.csv')
check(hashlib.sha256(original).digest()==hashlib.sha256(z26.read('Stage26/real_panel.csv')).digest(), 'Source CSV equals Stage26 real_panel.csv')

# Stage25 acceptance gate: do not substitute rejected gamma-mixed chains or preliminary fits.
for fit,required in [('coherent_G_MAR','ACCEPTED'),('coherent_G_MNAR','ACCEPTED'),('coherent_GM_MAR','REJECTED'),('coherent_GM_MNAR','REJECTED')]:
    d=jread(z25,f'Tehran_Stage25/results/{fit}/fit_diagnostics.json')
    check(d['status']==required, f'Stage25 {fit} status {required}')

# Stage26 accepted gamma/eligibility scenarios.
release=jread(z26,'Stage26/results_stage26/release_status.json')
needed={'within__MAR','within__minus_ln2','within__plus_ln2','pm10_recorded__minus_ln2','all_dates__minus_ln2'}
check(set(release['accepted_scenarios'])==needed, 'Stage26 five accepted gamma/eligibility scenarios')
check(release['fit_gates']['within__minus_ln4']['status']=='REJECTED', 'Stage26 -log(4) excluded')
for scenario in needed:
    d=release['fit_gates'][scenario]
    check(d['status']=='ACCEPTED' and d['passed']==d['total']==9, f'Stage26 {scenario} 9/9 gate')
area=csvread(z26,'Stage26/results_stage26/accepted_scenario_area_summaries.csv')
for scenario,target in [('within__MAR',65.913),('within__minus_ln2',67.863),('within__plus_ln2',60.550),('pm10_recorded__minus_ln2',65.397),('all_dates__minus_ln2',66.971)]:
    one=[r for r in area if r['scenario']==scenario and r['threshold']=='120' and r['date']=='2024-01-01']
    check(len(one)==1 and abs(float(one[0]['area_mean'])*100-target)<.002, f'Area120 Jan01: {scenario} ~{target}%')

# Stage29 diagnostics and independent re-scoring from original posterior predictive ensembles.
truth={int(r['flat_index']):float(r['true_reported_subindex']) for r in csvread(z29,'masked_holdout_truth_DO_NOT_LOAD_IN_FIT.csv')}
check(len(truth)==49,'Exactly 49 withheld reported readings')
stored={r['fit']:r for r in csvread(z29,'Stage29/results/heldout_summary.csv')}
for m,N in [('G_MAR',40000),('G_MNAR',28000)]:
    d=jread(z29,f'Stage29/results/{m}/fit_diagnostics.json')
    check(d['status']=='ACCEPTED' and d['n_pass']==d['n_gate']==9, f'Stage29 {m} full gate')
    with np.load(io.BytesIO(z29.read(f'Stage29/results/{m}/heldout_posterior_predictive.npz')),allow_pickle=False) as raw:
        ys=raw['predictive_readings'];idx=raw['heldout_flat_index']
    check(ys.shape==(N,49) and len(set(idx.tolist()))==49 and all(int(x) in truth for x in idx), f'{m}: posterior predictive dimensions/index match')
    y=np.asarray([truth[int(i)] for i in idx]);mu=ys.mean(axis=0)
    crps=[]
    coeff=2*np.arange(1,N+1)-N-1
    for k in range(49):
        x=np.sort(ys[:,k]);crps.append(np.mean(np.abs(x-y[k]))-np.dot(coeff,x)/(N*N))
    lo,hi=np.quantile(ys,[.05,.95],axis=0)
    result=dict(MAE=float(np.mean(np.abs(mu-y))),CRPS=float(np.mean(crps)),Coverage90=float(np.mean((lo<=y)&(y<=hi))),MeanPI90Width=float(np.mean(hi-lo)))
    for u in (100,120,150):result[f'Brier{u}']=float(np.mean(((ys>u).mean(axis=0)-(y>u))**2))
    for metric,value in result.items():
        check(abs(value-float(stored[m][metric]))<1e-9,f'{m} recomputed {metric}: {value:.9f}')

# Stage30: archived tail score and Stage31 post-hoc decisions.
tail=jread(z30,'metric_report.json')
row=next(r for r in tail['new_comparison'] if r['metric']=='twCRPS120')
check(abs(row['MAR']-1.8785983886989857)<1e-10 and abs(row['MNAR']-1.8808659848500582)<1e-10,'Stage30 twCRPS120 source values')
S=csvread(z31,'Stage31/heldout_discrimination_v31.csv')
check(len(S)==6 and {int(r['n_event']) for r in S}=={5,10,15},'Stage31 discrimination event counts')
A=csvread(z31,'Stage31/area_weighted_decision_sensitivity_v31.csv')
for scenario,expected in [('within__minus_ln2',69.0523),('within__plus_ln2',57.4439)]:
    a=[r for r in A if r['scenario']==scenario and r['date']=='2024-01-01' and r['subindex_threshold']=='120' and float(r['probability_cutoff'])==.5]
    check(len(a)==1 and abs(float(a[0]['scenario_high_risk_pct'])-expected)<.002,f'Stage31 {scenario} high-risk area ~{expected}%')
print('\nALL ARCHIVAL REPRODUCTION CHECKS PASSED. Full refitting has not been rerun by this verification script.')
