"""Parallel launch node accounting and callback election; Slurm calls mocked."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import submit_ldc_parallel_v35 as launcher
from continue_ldc_parallel_v35 import peer_decision
import continue_ldc_parallel_v35 as continuation
import subprocess


class ParallelTests(unittest.TestCase):
    def fixture(self,directory,active=True,stale=False,pending=20):
        work=Path(directory);(work/'submissions').mkdir()
        records=[dict(job=12345,phase='main',gpus=8)] if active else []
        launcher.save(work/'freeze.json',dict(unit_fixture=True));launcher.save(work/'operational_manifest_v35.json',dict(unit_fixture=True))
        tasks={f'fixture_{i}':dict(status='PENDING') for i in range(pending)}
        if active or stale:tasks['running_fixture']=dict(status='RUNNING',job='12345' if active else '90000')
        launcher.save(work/'main/state.json',dict(attention_required=False,tasks=tasks))
        return work,records

    def execute(self,work,records,queue,mode):
        calls=[]
        def command(args):
            calls.append(args)
            if args[0]=='sbatch':return '900001' if '--hold' in args else '900002'
            if args[0]=='scontrol':
                self.assertTrue((work/'submissions/900001.json').exists());return ''
            raise AssertionError(args)
        with patch.object(launcher,'WORK',work),patch.object(launcher,'POOL',work/'pool'),\
             patch.object(launcher,'verify_parallel'),patch.object(launcher,'queue_snapshot',return_value=queue),\
             patch.object(launcher,'main_records',return_value=records),patch.object(launcher,'command',side_effect=command):
            launcher.run(mode)
        return calls

    def test_auxiliary_allows_other_legitimate_running_tasks(self):
        with tempfile.TemporaryDirectory(prefix='parallel35_',dir=launcher.ROOT/'test/runtime') as directory:
            work,records=self.fixture(directory)
            self.execute(work,records,{'12345':1,'12346':1},'auxiliary')
            r=launcher.read(work/'submissions/900001.json')
            self.assertEqual((r['gpus'],r['nodes']),(8,1))

    def test_two_node_rollover_fits_four_node_handover(self):
        with tempfile.TemporaryDirectory(prefix='parallel35_',dir=launcher.ROOT/'test/runtime') as directory:
            work,records=self.fixture(directory,active=False)
            self.execute(work,records,{'current_callback':1},'rollover')
            r=launcher.read(work/'submissions/900001.json')
            self.assertEqual((r['gpus'],r['nodes']),(16,2))

    def test_four_node_ceiling_rejects_fifth_reserved_node(self):
        with tempfile.TemporaryDirectory(prefix='parallel35_',dir=launcher.ROOT/'test/runtime') as directory:
            work,records=self.fixture(directory,active=False)
            with self.assertRaises(ValueError):self.execute(work,records,{'callback':1,'other':1},'rollover')
            self.assertFalse((work/'submissions/900001.json').exists())

    def test_no_third_concurrent_batch(self):
        with tempfile.TemporaryDirectory(prefix='parallel35_',dir=launcher.ROOT/'test/runtime') as directory:
            work,records=self.fixture(directory)
            records.append(dict(job=12347,phase='main',gpus=8))
            with self.assertRaises(ValueError):self.execute(work,records,{'12345':1,'12347':1},'auxiliary')

    def test_stale_task_blocks_submission(self):
        with tempfile.TemporaryDirectory(prefix='parallel35_',dir=launcher.ROOT/'test/runtime') as directory:
            work,records=self.fixture(directory,active=False,stale=True)
            with self.assertRaises(ValueError):self.execute(work,records,{},'rollover')

    def test_short_final_tail_can_use_eight_gpus(self):
        with tempfile.TemporaryDirectory(prefix='parallel35_',dir=launcher.ROOT/'test/runtime') as directory:
            work,records=self.fixture(directory,active=False,pending=5)
            self.execute(work,records,{'callback':1},'rollover')
            self.assertEqual(launcher.read(work/'submissions/900001.json')['gpus'],8)

    def test_callback_election_has_one_leader(self):
        self.assertEqual(peer_decision('200',{'201'}),'YIELD')
        self.assertEqual(peer_decision('201',{'200'}),'LEADER')
        self.assertEqual(peer_decision('201',set()),'LEADER')


class ContinuationTests(unittest.TestCase):
    def execute(self,queue,records,done=False,peer=False,failed=False):
        with tempfile.TemporaryDirectory(prefix='parallel35_callback_',dir=launcher.ROOT/'test/runtime') as directory:
            work=Path(directory);pool=work/'pool';(pool/'callbacks').mkdir(parents=True)
            if peer:launcher.save(pool/'callbacks/201.json',dict(unit_fixture=True))
            launcher.save(work/'scheduler/123/summary.json',dict(state='FAILED' if failed else 'COMPLETED',exit_code='1:0' if failed else '0:0'))
            launcher.save(work/'main/batches/123/review.json',dict(checks={'actual':True},all_done=done))
            launcher.save(work/'main/state.json',dict(tasks={'unit':dict(status='DONE' if done else 'PENDING')}))
            response=subprocess.CompletedProcess([],0,'NEXT16','')
            with patch.object(continuation,'WORK',work),patch.object(continuation,'POOL',pool),\
                 patch.object(continuation,'verify_parallel'),patch.object(continuation,'queue_snapshot',return_value=queue),\
                 patch.object(continuation,'main_records',return_value=records),\
                 patch.dict(continuation.os.environ,{'SLURM_JOB_ID':'200'}),\
                 patch.object(continuation.subprocess,'run',return_value=response) as call:
                ok=continuation.run('123')
            result=launcher.read(work/'scheduler/123/parallel_continuation.json')
            return ok,result,call.call_count,(pool/'hold.json').exists()

    def test_first_callback_waits_for_other_gpu_batch(self):
        ok,result,calls,hold=self.execute({'456':1,'200':1},[dict(job=456)])
        self.assertTrue(ok);self.assertEqual(result['status'],'WAIT_FOR_OTHER_GPU_BATCH');self.assertEqual(calls,0);self.assertFalse(hold)

    def test_lower_callback_yields_to_one_leader(self):
        ok,result,calls,hold=self.execute({'200':1,'201':1},[],peer=True)
        self.assertTrue(ok);self.assertEqual(result['status'],'YIELD_TO_LATER_CALLBACK');self.assertEqual(calls,0)

    def test_last_callback_launches_combined_successor(self):
        ok,result,calls,hold=self.execute({'200':1},[])
        self.assertTrue(ok);self.assertEqual(result['status'],'NEXT_COMBINED_BATCH_SUBMITTED');self.assertEqual(calls,1);self.assertFalse(hold)

    def test_failure_holds_pool_without_fabricating_results(self):
        ok,result,calls,hold=self.execute({'200':1},[],failed=True)
        self.assertFalse(ok);self.assertTrue(hold);self.assertEqual(calls,0)

    def test_completed_plan_never_opens_test_or_submits_more(self):
        ok,result,calls,hold=self.execute({'200':1},[],done=True)
        self.assertTrue(ok);self.assertEqual(result['status'],'TRAINING_STAGE_COMPLETE_TEST_PENDING')
        self.assertEqual(calls,0);self.assertFalse(result['main_test_opened'])


if __name__=='__main__':unittest.main()
