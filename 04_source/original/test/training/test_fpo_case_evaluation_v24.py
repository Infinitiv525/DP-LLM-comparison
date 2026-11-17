import json
from pathlib import Path
import tempfile
import unittest
from fpo_case_v24 import split_frames
from evaluate_fpo_case_v24 import evaluate,reserved_audit_from_full


class ReservedScoreTests(unittest.TestCase):
    def test_audit_cannot_open_before_selection(self):
        with tempfile.TemporaryDirectory() as t:
            for phase in ['audit','postfit_full']:
                with self.assertRaises(ValueError):evaluate(None,{},Path(t)/phase,phase)
                self.assertFalse((Path(t)/phase).exists())

    def test_audit_error_is_not_diluted_by_training_frames(self):
        roles=split_frames();audit=set(roles['audit'])
        rows=[]
        for frame in range(4,242):
            velocity=.3 if frame in audit else .01
            rows.append(dict(frame=frame,query_gpu_ms=3.,model=dict(velocity_relative_l2=velocity,pressure_relative_l2=2*velocity),
                persistence=dict(velocity_relative_l2=.5,pressure_relative_l2=.7),
                interior_div_rms=.001,reference_div_rms=.001,fd_momentum_loss=.1,reference_fd_momentum_loss=.01))
        full=dict(failure=False,checkpoint_selected=True,phase='postfit_full',prefix_div_rms=.001)
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);source=root/'full';source.mkdir()
            (source/'per_frame.json').write_text(json.dumps(rows))
            result=reserved_audit_from_full(full,source,root/'audit')
            self.assertEqual(result['frames'],roles['audit']);self.assertEqual(result['predicted_frames'],39)
            self.assertAlmostEqual(result['mean_errors']['velocity_relative_l2'],.3)
            self.assertFalse(result['accuracy_pass']);self.assertTrue(result['mass_pass'])
            for row in rows:
                if row['frame'] not in audit:row['model']['velocity_relative_l2']=1e8
            (source/'per_frame.json').write_text(json.dumps(rows))
            other=reserved_audit_from_full(full,source,root/'audit2')
            self.assertEqual(other['mean_errors'],result['mean_errors'])


if __name__=='__main__':unittest.main()
