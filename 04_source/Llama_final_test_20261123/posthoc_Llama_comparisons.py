"""Exploratory posthoc Llama-vs-four-LLM comparisons from archived paired draws.
This does not update the original four predeclared LLM-vs-TPE contrasts.
Usage: python posthoc_Llama_comparisons.py --draws bootstrap_deltas_five_LLM.npy --summary combined_summary.json --output comparisons.json
"""
from pathlib import Path
import argparse,json,hashlib
import numpy as np

def calculate(draws,summary):
 assert draws.shape==(10000,5) and np.isfinite(draws).all()
 labels=['qwen3_8b','ministral3_8b','phi4_mini','granite33_8b'];means={r['method']:r['mse_all'] for r in summary['methods']};rows=[]
 for i,label in enumerate(labels):
  lo,hi=np.quantile(draws[:,4]-draws[:,i],[.00625,.99375],method='linear');point=means['Llama31_8B']-means[label];rows.append(dict(method='Llama31_8B',other=label,delta_test_MSE=point,relative_delta_percent=100*point/means[label],posthoc_family_interval_percent=98.75,interval=[float(lo),float(hi)],interval_excludes_zero=bool(lo>0 or hi<0)))
 return rows
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--draws',type=Path,required=True);p.add_argument('--summary',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();d=np.load(a.draws,allow_pickle=False);s=json.loads(a.summary.read_text());out=dict(status='POSTHOC_EXPLORATORY_COMPARISONS',analysis_was_specified_after_TEST=True,source_draws_sha256=hashlib.sha256(a.draws.read_bytes()).hexdigest(),comparisons=calculate(d,s),limits=['Approximate paired crossed bootstrap; only five independent searches.','Final seeds were averaged before search/geometry resampling.','Differences near1e-14MSE are numerical roundoff, not meaningful accuracy improvement.']);a.output.write_text(json.dumps(out,indent=2)+chr(10))
