"""Operational tests: capacity failures do not contaminate or halt the main search."""
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

import collect_ldc_capacity_v37 as cap


class CapacityOperations(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='capacity_test_', dir=cap.ROOT / 'test/runtime')
        self.folder = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_infrastructure_failure_is_separate_from_main_hold(self):
        with patch.object(cap, 'CAP', self.folder), patch.object(cap, 'verify_capacity', side_effect=ValueError('fixture failure')):
            cap.capacity_worker()
        result = json.loads((self.folder / 'result.json').read_text())
        self.assertEqual(result['status'], 'CAPACITY_INFRASTRUCTURE_REVIEW_REQUIRED')
        self.assertFalse(result['main_test_opened'])
        self.assertFalse((self.folder / 'hold.json').exists())

    def test_main_successor_waits_for_capacity_completion(self):
        def worker():
            (self.folder / 'result.json').write_text('{"status":"fixture_complete"}')
        def collector(group):
            return bool(cap.collector.prepare_successor('123'))
        def successor(job):
            self.assertTrue((self.folder / 'result.json').exists())
            self.assertEqual(job, '123')
            return {'job':'456'}
        original = cap.collector.prepare_successor
        try:
            with patch.object(cap, 'CAP', self.folder), patch.object(cap, 'verify_capacity'), \
                 patch.object(cap, 'capacity_worker', side_effect=worker), \
                 patch.object(cap, 'prepare_successor', side_effect=successor), \
                 patch.object(cap.collector, 'run', side_effect=collector), \
                 patch.dict('os.environ', {'SLURM_JOB_ID':'123'}):
                self.assertTrue(cap.run('fixture_group'))
        finally:
            cap.collector.prepare_successor = original
        self.assertTrue(json.loads((self.folder / 'handoff.json').read_text())['collector_success'])

    def test_failed_capacity_result_does_not_choose_main_configuration(self):
        def worker():
            (self.folder / 'result.json').write_text('{"all_three_complete":false}')
        original = cap.collector.prepare_successor
        try:
            with patch.object(cap, 'CAP', self.folder), patch.object(cap, 'verify_capacity'), \
                 patch.object(cap, 'capacity_worker', side_effect=worker), \
                 patch.object(cap, 'prepare_successor', return_value={'job':'456'}) as successor, \
                 patch.object(cap.collector, 'run', side_effect=lambda group: bool(cap.collector.prepare_successor('123'))), \
                 patch.dict('os.environ', {'SLURM_JOB_ID':'123'}):
                self.assertTrue(cap.run('fixture_group'))
                successor.assert_called_once_with('123')
        finally:
            cap.collector.prepare_successor = original


if __name__ == '__main__':
    unittest.main()
