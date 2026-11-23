from pathlib import Path
import json,subprocess,time,re,datetime
R=Path('/mnt/project/perun2601449/dp_llm_cfd');F=R/'results/llama_final_test_20261123_v1';job=str(json.loads((F/'submission.json').read_text())['gpu_job']);ready=F/'collector_ready.json';ready.write_text(json.dumps(dict(gpu_job=job,ready=True))+'\n');terminal={'COMPLETED','FAILED','TIMEOUT','CANCELLED','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED'};deadline=time.monotonic()+47*3600
while time.monotonic()<deadline:
 v=subprocess.run(['scontrol','show','job',job,'-o'],capture_output=True,text=True,timeout=30);assert v.returncode==0,v.stderr;fields=dict(re.findall(r'(?:^|\s)(\w+)=([^\s]+)',v.stdout))
 if fields['JobState'] in terminal:break
 time.sleep(30)
else:raise RuntimeError('Controller accounting deadline exceeded')
report=dict(status='TEST_ACCOUNTED' if fields['JobState']=='COMPLETED' and fields['ExitCode']=='0:0' else 'TEST_REVIEW_REQUIRED',captured_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),job=job,controller_fields=fields,raw_record=v.stdout,report_ready=(F/'combined_report/summary.json').exists());(F/'accounting_closeout.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report));assert report['status']=='TEST_ACCOUNTED' and report['report_ready']
