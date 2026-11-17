"""Pinned file inventory; counts candidates, not numerically validated scenarios."""
import json,re,urllib.request
from pathlib import Path
ROOT=Path('/mnt/project/perun2601449/dp_llm_cfd');OUT=ROOT/'results/fpo_pinn_accuracy_20261030_v1'
REV='937c672604d4c9f39aceedf1fa2f0a603695264b'
def run():
 records=[];pages=[];url=f'https://huggingface.co/api/datasets/BGLab/FlowBench/tree/{REV}/FPO_NS_2D_1024x256?recursive=true&limit=1000'
 while url:
  with urllib.request.urlopen(url,timeout=60) as response:
   entries=json.load(response);link=response.headers.get('Link','')
  records.extend(entries);pages.append(dict(url=url,entries=len(entries)))
  m=re.search(r'<([^>]+)>;\s*rel="next"',link);url=m[1] if m else None
  if len(pages)>30:raise RuntimeError('Unexpected pagination')
 files={e['path']:e for e in records if e['type']=='file'};scenarios=[];missing=[]
 for name,e in sorted(files.items()):
  m=re.fullmatch(r'FPO_NS_2D_1024x256/([^/]+)/([^/]+)/Re_(\d+)\.npz',name)
  if not m:continue
  geometry=str(Path(name).parent/'input_geometry.npz');r=dict(family=m[1],geometry_id=m[2],reynolds=int(m[3]),path=name,bytes=e['size'],lfs=e.get('lfs'))
  if e['size']>0 and geometry in files and files[geometry]['size']>0:scenarios.append(r)
  else:missing.append(r)
 geometries=sorted({(r['family'],r['geometry_id']) for r in scenarios});families={}
 for family in sorted({r['family'] for r in scenarios}):
  subset=[r for r in scenarios if r['family']==family];families[family]=dict(scenarios=len(subset),geometries=len({r['geometry_id'] for r in subset}))
 result=dict(status='PINN_FPO_FILE_INVENTORY_COMPLETE',revision=REV,total_scenarios=len(scenarios),total_geometries=len(geometries),families=families,bytes=sum(r['bytes'] for r in scenarios),missing_geometry_or_empty=missing,pages=pages,scenarios=scenarios,official_split_exists=False,our_scenario_split_created=False,pilot_training_scenarios=1,pilot_validation_scenarios_separate=0,pilot_test_scenarios_separate=0,numeric_validity_of_full_pool_verified=False)
 p=OUT/'inventory.json';assert not p.exists();p.write_text(json.dumps(result,indent=2)+'\n');print({k:v for k,v in result.items() if k not in ['scenarios','pages','missing_geometry_or_empty']},flush=True)
if __name__=='__main__':run()
