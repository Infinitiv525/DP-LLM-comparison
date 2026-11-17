"""Export only small integration receipts and current main queue metadata."""
from pathlib import Path
import hashlib
import sys
import time
import zipfile
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'llm'))
from ldc_main_common_v33 import WORK,read,save,digest,verify


def run():
    verify()
    if read(WORK/'integration/review.json')['status']!='LDC_MAIN_INTEGRATION_PASS':
        raise ValueError('Integration not complete')
    export=WORK/'exports';export.mkdir(exist_ok=True)
    paths=[]
    for name in ['protocol.json','freeze.json','operational_manifest.json','protokol_citatelny.txt','main_gate.json','integration/review.json',
                 'integration/state.json','main/state.json','main/review.json']:
        path=WORK/name
        if path.exists():paths.append(path)
    for directory in ['submissions','scheduler','integration/batches','integration/executions','integration/selections','integration/models']:
        for path in (WORK/directory).rglob('*'):
            if path.is_file() and path.suffix in {'.json','.log','.txt'} and path.stat().st_size<8*1024*1024:
                paths.append(path)
    archive=export/'integration_and_launch_receipts.zip'
    if archive.exists():raise ValueError('Do not overwrite an exported receipt snapshot')
    members={}
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for path in sorted(set(paths)):
            payload=path.read_bytes();name=str(path.relative_to(WORK))
            members[name]=dict(bytes=len(payload),sha256=hashlib.sha256(payload).hexdigest())
            z.writestr(name,payload)
    save(export/'integration_and_launch_receipt.json',dict(bytes=archive.stat().st_size,sha256=digest(archive),
        unix=time.time(),source_sha256=digest(__file__),members=members,arrays_or_weights_exported=False))
    print('SMALL_RECEIPTS_PACKED',len(members),archive.stat().st_size,flush=True)


if __name__=='__main__':run()
