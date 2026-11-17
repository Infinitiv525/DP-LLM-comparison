"""Scientific summary figures from audited, frozen numeric TEST reports."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path('/mnt/project/perun2601449/dp_llm_cfd')
FINAL=ROOT/'results/ldc_final_test_20261108_v49'
OUT=ROOT/'results/ldc_final_closeout_20261108_v50a'


def main():
    audit=json.loads((OUT/'summary.json').read_text());assert audit['status']=='FINAL_OUTPUTS_AUDITED'
    report=json.loads((FINAL/'report/summary.json').read_text())
    names={'qwen3_8b':'Qwen3','ministral3_8b':'Ministral3','phi4_mini':'Phi4-mini',
           'granite33_8b':'Granite3.3','random':'Random','tpe':'TPE','fixed':'Fixed'}
    rows=report['methods'];colors=['#377eb8']*4+['#9b9b9b','#d36a23','#666666']
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,ax=plt.subplots(1,3,figsize=(15.5,5.3),gridspec_kw={'width_ratios':[1.25,1.1,1]})
    y=np.arange(7);values=np.array([r['mse_all'] for r in rows])*1e4
    ax[0].barh(y,values,color=colors);ax[0].set_yticks(y,[names[r['method']] for r in rows]);ax[0].invert_yaxis()
    ax[0].set_xlabel('Mean TEST MSE (x 10⁻⁴; lower is better)');ax[0].set_xlim(0,3.5)
    ax[0].set_title('Primary accuracy metric',loc='left',fontweight='bold')
    for i,v in enumerate(values):ax[0].text(v+.04,i,f'{v:.3f}',va='center',fontsize=9)
    c=report['primary_contrasts'];point=np.array([r['delta_test_mse'] for r in c])*1e4
    bounds=np.array([r['interval'] for r in c])*1e4
    ax[1].errorbar(point,np.arange(4),xerr=np.stack((point-bounds[:,0],bounds[:,1]-point)),fmt='o',color='#377eb8',capsize=4)
    ax[1].axvline(0,color='black',linewidth=1);ax[1].set_yticks(np.arange(4),[names[r['method']] for r in c]);ax[1].invert_yaxis()
    ax[1].set_xlabel('LLM minus TPE MSE (x 10⁻⁴)');ax[1].set_xlim(-.15,1.15)
    ax[1].set_title('Prespecified comparisons',loc='left',fontweight='bold')
    ax[1].text(.02,.03,'Positive = higher error than TPE\n98.75% Bonferroni intervals',transform=ax[1].transAxes,fontsize=9)
    speed=[r['worst_p99_ms'] for r in rows]
    ax[2].barh(y,speed,color=colors);ax[2].set_yticks(y,[names[r['method']] for r in rows]);ax[2].invert_yaxis()
    ax[2].axvline(5,color='#b2182b',linestyle='--',label='5 ms requirement');ax[2].set_xlim(0,5.5)
    ax[2].set_xlabel('Worst per-case p99 GPU query (ms)');ax[2].legend(loc='lower right',fontsize=8)
    ax[2].set_title('Full 512 x 512 field',loc='left',fontweight='bold')
    for i,v in enumerate(speed):ax[2].text(v+.06,i,f'{v:.2f}',va='center',fontsize=9)
    fig.suptitle('Shared steady 2D PINN: final FlowBench TEST',fontsize=15,fontweight='bold')
    fig.text(.5,.025,'155 models, 200 cases each. Five independent searches per method; five final seeds averaged before paired search/geometry bootstrap.\n'
        'Intervals are approximate (five searches). Geometries overlap TRAIN/TEST. GPU timing excludes transfers/rendering; single-query outliers remain in raw data.',
        ha='center',va='bottom',fontsize=9)
    fig.tight_layout(rect=[0,.13,1,.92],w_pad=2.3)
    for ext in ['png','svg']:fig.savefig(OUT/('final_test_overview.'+ext),dpi=220)
    plt.close(fig)


if __name__=='__main__':main()
