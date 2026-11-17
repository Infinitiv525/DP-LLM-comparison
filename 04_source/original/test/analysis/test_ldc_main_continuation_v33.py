"""No real Slurm actions: verify continuation failure gates and registration order."""
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import continue_ldc_main_v33 as continuation
import submit_ldc_main_batch_v33 as launcher


class ContinuationTests(unittest.TestCase):
    def run_fixture(self,state='COMPLETED',all_done=False,productive=True,child_return=0):
        saved=[]
        def fixture_read(path):
            if 'submissions' in path.parts:return dict(phase='main',done_before_start=0)
            if path.name=='summary.json':return dict(state=state,exit_code='0:0' if state=='COMPLETED' else '1:0')
            return dict(checks={'receipts':True},all_done=all_done,status='LDC_MAIN_BATCH_PARTIAL_PASS',
                        all_attempts_done=1 if productive else 0)
        response=subprocess.CompletedProcess([],child_return,'next-job','error' if child_return else '')
        with patch.object(continuation,'verify'),patch.object(continuation,'read',side_effect=fixture_read),\
             patch.object(continuation,'save',side_effect=lambda p,v:saved.append(v)),\
             patch.object(continuation,'digest',return_value='fixture_digest'),\
             patch.object(continuation.subprocess,'run',return_value=response) as call:
            result=continuation.run('fixture-job')
        return result,saved[-1],call.call_count

    def test_failed_gpu_batch_never_resubmits(self):
        ok,row,calls=self.run_fixture(state='FAILED')
        self.assertFalse(ok);self.assertEqual(row['status'],'CONTINUATION_STOPPED');self.assertEqual(calls,0)

    def test_completed_training_never_opens_test_or_submits_more(self):
        ok,row,calls=self.run_fixture(all_done=True)
        self.assertTrue(ok);self.assertEqual(calls,0);self.assertFalse(row['main_test_opened'])
        self.assertEqual(row['status'],'TRAINING_STAGE_COMPLETE_TEST_PENDING')

    def test_no_progress_stops_chain(self):
        ok,row,calls=self.run_fixture(productive=False)
        self.assertFalse(ok);self.assertEqual(calls,0)

    def test_only_productive_clean_batch_can_continue(self):
        ok,row,calls=self.run_fixture()
        self.assertTrue(ok);self.assertEqual(calls,1);self.assertEqual(row['status'],'NEXT_BATCH_SUBMITTED')

    def test_launcher_failure_remains_visible(self):
        ok,row,calls=self.run_fixture(child_return=1)
        self.assertFalse(ok);self.assertEqual(row['status'],'CONTINUATION_STOPPED');self.assertEqual(calls,1)


class LauncherTests(unittest.TestCase):
    def fixture_directory(self,directory,status='PENDING'):
        work=Path(directory)
        launcher.save(work/'operational_manifest.json',dict(files={}))
        launcher.save(work/'integration/review.json',dict(status='LDC_MAIN_INTEGRATION_PASS',checks={'actual':True},successful_completed=25,job='fixture'))
        launcher.save(work/'scheduler/fixture/summary.json',dict(state='COMPLETED',exit_code='0:0'))
        launcher.save(work/'main/state.json',dict(attention_required=False,tasks={'fixture':dict(status=status)}))
        (work/'submissions').mkdir()
        launcher.save(work/'protocol.json',dict(fixture_only=True));launcher.save(work/'freeze.json',dict(fixture_only=True))
        return work

    def test_registration_happens_before_release(self):
        runtime=launcher.ROOT/'test/runtime'
        with tempfile.TemporaryDirectory(prefix='main33_launch_',dir=runtime) as directory:
            work=self.fixture_directory(directory);calls=[]
            def command(args):
                calls.append(args)
                if args[0]=='squeue':return ''
                if args[0]=='sbatch':return '900001' if '--hold' in args else '900002'
                if args[0]=='scontrol':
                    self.assertTrue((work/'submissions/900001.json').exists())
                    return ''
                raise AssertionError(args)
            with patch.object(launcher,'WORK',work),patch.object(launcher,'verify'),patch.object(launcher,'command',side_effect=command):
                launcher.run()
            record=launcher.read(work/'submissions/900001.json')
            self.assertEqual(record['accounting_job'],900002)
            self.assertIn('--dependency=afterany:900001',calls[-2])
            self.assertEqual(calls[-1],['scontrol','release','900001'])

    def test_stale_running_task_prevents_any_submission(self):
        with tempfile.TemporaryDirectory(prefix='main33_stale_',dir=launcher.ROOT/'test/runtime') as directory:
            work=self.fixture_directory(directory,'RUNNING')
            with patch.object(launcher,'WORK',work),patch.object(launcher,'verify'),patch.object(launcher,'command') as command:
                with self.assertRaises(ValueError):launcher.run()
                command.assert_not_called()


if __name__=='__main__':unittest.main()
