from pathlib import Path
import pandas as pd, matplotlib.pyplot as plt
P=Path(__file__).resolve().parent
x=pd.read_csv(P/'area_weighted_decision_sensitivity_v31.csv').query('date=="2024-01-01" and subindex_threshold==120')
fig,ax=plt.subplots(figsize=(6.8,3.6))
a=x[x.scenario=='within__minus_ln2'].sort_values('probability_cutoff');b=x[x.scenario=='within__plus_ln2'].sort_values('probability_cutoff');q=a.probability_cutoff.to_numpy()
ax.plot(q,a.mar_high_risk_pct.to_numpy(),'-o',label='MAR')
ax.plot(q,a.scenario_high_risk_pct.to_numpy(),'-s',label=r'MNAR ($\gamma=-\log 2$)')
ax.plot(q,b.scenario_high_risk_pct.to_numpy(),'-^',label=r'MNAR ($\gamma=+\log 2$)')
ax.set_xlabel('Cell-wise posterior exceedance-probability cutoff')
ax.set_ylabel('City area classified as high-risk (%)')
ax.set_ylim(0,100); ax.set_xticks(q); ax.grid(alpha=.25);ax.legend(frameon=False,fontsize=8,loc='upper right');fig.tight_layout();fig.savefig(P/'Fig_Stage31_Decision_Stability.pdf',bbox_inches='tight');fig.savefig(P/'Fig_Stage31_Decision_Stability.png',dpi=170,bbox_inches='tight');plt.close(fig)
