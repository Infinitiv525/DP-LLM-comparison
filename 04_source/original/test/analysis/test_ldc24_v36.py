"""24GPU node-budget, bootstrap and event accounting tests; no real submissions."""
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import ldc24_ops_v36 as ops
import collect_ldc24_v36 as collector


class OperationTests(unittest.TestCase):
    def fixture(self,directory,records=None):
        work=Path(directory);pool=work/'pool';(work/'submissions').mkdir();(pool/'groups').mkdir(parents=True)
        ops.save(work/'freeze.json',dict(fixture=True));ops.save(work/'operational_manifest_v36.json',dict(fixture=True))
        records=records or []
        tasks={f'p{i}':dict(status='PENDING') for i in range(30)}
        for r in records:tasks['running'+str(r['job'])]=dict(status='RUNNING',job=str(r['job']))
        ops.save(work/'main/state.json',dict(attention_required=False,tasks=tasks))
        return work,pool

    def test_dependency_is_or_of_gpu_exits(self):
        self.assertEqual(ops.after_any(['123','456']),'afterany:123?afterany:456')
        self.assertEqual(ops.after_any(['123']),'afterany:123')
        with self.assertRaises(ValueError):ops.after_any(['not_a_job'])

    def test_handover_reserves_gpu_nodes_without_premature_cpu_node(self):
        with tempfile.TemporaryDirectory(prefix='ldc24_',dir=ops.ROOT/'test/runtime') as directory:
            work,pool=self.fixture(directory);calls=[]
            def command(args):calls.append(args);return '900001'
            with patch.object(ops,'WORK',work),patch.object(ops,'POOL',pool),patch.object(ops,'verify'),\
                 patch.object(ops,'queue_snapshot',return_value={'cpu':1}),patch.object(ops,'main_records',return_value=[]),\
                 patch.object(ops,'command',side_effect=command):
                job=ops.launch_gpu(24,'group_next','123')
            self.assertEqual(job,'900001');self.assertEqual(len(calls),1)
            self.assertIn('--dependency=afterok:123',calls[0])
            self.assertIsNone(ops.read(work/'submissions/900001.json')['accounting_job'])

    def test_fifth_reserved_node_is_rejected(self):
        with tempfile.TemporaryDirectory(prefix='ldc24_',dir=ops.ROOT/'test/runtime') as directory:
            work,pool=self.fixture(directory)
            with patch.object(ops,'WORK',work),patch.object(ops,'POOL',pool),patch.object(ops,'verify'),\
                 patch.object(ops,'queue_snapshot',return_value={'cpu':1,'other':1}),patch.object(ops,'command') as command:
                with self.assertRaises(ValueError):ops.launch_gpu(24,'group_next','123')
                command.assert_not_called()

    def test_more_than_twenty_four_concurrent_gpus_is_rejected(self):
        records=[dict(job=123,gpus=24,phase='main')]
        with tempfile.TemporaryDirectory(prefix='ldc24_',dir=ops.ROOT/'test/runtime') as directory:
            work,pool=self.fixture(directory,records)
            with patch.object(ops,'WORK',work),patch.object(ops,'POOL',pool),patch.object(ops,'verify'),\
                 patch.object(ops,'queue_snapshot',return_value={'123':3}),patch.object(ops,'main_records',return_value=records):
                with self.assertRaises(ValueError):ops.launch_gpu(8,'extra')

    def test_stale_owner_prevents_new_training(self):
        records=[dict(job=123,gpus=8,phase='main')]
        with tempfile.TemporaryDirectory(prefix='ldc24_',dir=ops.ROOT/'test/runtime') as directory:
            work,pool=self.fixture(directory,records)
            with patch.object(ops,'WORK',work),patch.object(ops,'POOL',pool),patch.object(ops,'verify'),\
                 patch.object(ops,'queue_snapshot',return_value={}),patch.object(ops,'main_records',return_value=records):
                with self.assertRaises(ValueError):ops.launch_gpu(8,'extra')

    def test_bootstrap_reuses_initial_shared_collector(self):
        with tempfile.TemporaryDirectory(prefix='ldc24_',dir=ops.ROOT/'test/runtime') as directory:
            work,pool=self.fixture(directory)
            ops.save(work/'submissions/123.json',dict(phase='main',group='g0'))
            ops.save(pool/'groups/g0.json',dict(jobs=['123'],collectors=['456']))
            with patch.object(ops,'WORK',work),patch.object(ops,'POOL',pool),patch.object(ops,'verify'),\
                 patch.object(ops,'queue_snapshot',return_value={'123':1,'456':1}),patch.object(ops,'submit_collector') as submit:
                ops.bootstrap('123');submit.assert_not_called()
            self.assertEqual(ops.read(pool/'bootstrap/123.json')['collector_job'],'456')

    def test_gpu_start_attaches_collector_before_neural_work(self):
        with tempfile.TemporaryDirectory(prefix='ldc24_',dir=ops.ROOT/'test/runtime') as directory:
            work,pool=self.fixture(directory)
            ops.save(work/'submissions/123.json',dict(phase='main',group='g0'))
            ops.save(pool/'groups/g0.json',dict(jobs=['123'],collectors=[]))
            with patch.object(ops,'WORK',work),patch.object(ops,'POOL',pool),patch.object(ops,'verify'),\
                 patch.object(ops,'queue_snapshot',return_value={'123':3}),patch.object(ops,'command',return_value='456') as command:
                ops.bootstrap('123')
            self.assertEqual(len(command.call_args_list),1)
            self.assertIn('--dependency=afterany:123',command.call_args[0][0])
            self.assertEqual(ops.read(pool/'bootstrap/123.json')['status'],'ACCOUNTING_READY_BEFORE_TRAINING')

    def test_no_duplicate_collector_at_four_node_limit(self):
        with tempfile.TemporaryDirectory(prefix='ldc24_',dir=ops.ROOT/'test/runtime') as directory:
            work,pool=self.fixture(directory)
            with patch.object(ops,'WORK',work),patch.object(ops,'POOL',pool),\
                 patch.object(ops,'queue_snapshot',return_value={'gpu':3,'collector':1}),patch.object(ops,'command') as command:
                with self.assertRaises(ValueError):ops.submit_collector('g0',['123'])
                command.assert_not_called()


class CollectorTests(unittest.TestCase):
    def execute(self,done=False,failed=False):
        with tempfile.TemporaryDirectory(prefix='ldc24_collector_',dir=ops.ROOT/'test/runtime') as directory:
            work=Path(directory);pool=work/'pool';ops.save(pool/'groups/g0.json',dict(jobs=['123','456']))
            ops.save(work/'main/state.json',dict(tasks={'fixture':dict(status='DONE' if done else 'PENDING')}))
            for job in ['123','456']:ops.save(work/'main/batches'/job/'review.json',dict(checks={'receipts':True}))
            queues=iter([{'456':1},{ }]);captured=[]
            def capture(job):
                captured.append(job);bad=failed and job=='123'
                return dict(state='FAILED' if bad else 'COMPLETED',exit_code='1:0' if bad else '0:0')
            with patch.object(collector,'WORK',work),patch.object(collector,'POOL',pool),patch.object(collector,'verify'),\
                 patch.dict(collector.os.environ,{'SLURM_JOB_ID':'999'}),\
                 patch.object(collector,'queue_snapshot',side_effect=lambda:next(queues)),\
                 patch.object(collector,'terminal_state',return_value=True),patch.object(collector,'capture',side_effect=capture),\
                 patch.object(collector.time,'sleep') as sleep,\
                 patch.object(collector,'prepare_successor',return_value=dict(job='next',gpus=24)) as next_batch,\
                 patch.object(collector,'command',return_value='888'):
                ok=collector.run('g0')
            result=ops.read(pool/'collector_results/999.json')
            return ok,result,captured,sleep.call_count,next_batch.call_count

    def test_every_exit_accounted_once_while_cpu_waits_for_rest(self):
        ok,result,captured,sleeps,submits=self.execute()
        self.assertTrue(ok);self.assertEqual(captured,['123','456']);self.assertEqual(sleeps,1);self.assertEqual(submits,1)

    def test_gpu_failure_still_accounts_rest_but_never_launches_more(self):
        ok,result,captured,sleeps,submits=self.execute(failed=True)
        self.assertFalse(ok);self.assertEqual(captured,['123','456']);self.assertEqual(submits,0)

    def test_done_stage_keeps_test_closed(self):
        ok,result,captured,sleeps,submits=self.execute(done=True)
        self.assertTrue(ok);self.assertEqual(result['status'],'TRAINING_STAGE_COMPLETE_TEST_PENDING')
        self.assertFalse(result['main_test_opened']);self.assertEqual(submits,0)

    def test_saved_gpu_receipt_needs_no_expired_slurm_lookup(self):
        with tempfile.TemporaryDirectory(prefix='ldc24_retention_',dir=ops.ROOT/'test/runtime') as directory:
            work=Path(directory);pool=work/'pool'
            ops.save(pool/'groups/g0.json',dict(jobs=['123']))
            ops.save(work/'scheduler/123/summary.json',dict(state='COMPLETED',exit_code='0:0'))
            ops.save(work/'main/batches/123/review.json',dict(checks={'receipt':True}))
            ops.save(work/'main/state.json',dict(tasks={'unit':dict(status='DONE')}))
            with patch.object(collector,'WORK',work),patch.object(collector,'POOL',pool),patch.object(collector,'verify'),\
                 patch.dict(collector.os.environ,{'SLURM_JOB_ID':'999'}),\
                 patch.object(collector,'queue_snapshot',return_value={}),\
                 patch.object(collector,'terminal_state') as lookup,\
                 patch.object(collector,'capture',return_value=dict(state='COMPLETED',exit_code='0:0')),\
                 patch.object(collector,'command',return_value='888'):
                self.assertTrue(collector.run('g0'));lookup.assert_not_called()


if __name__=='__main__':unittest.main()
