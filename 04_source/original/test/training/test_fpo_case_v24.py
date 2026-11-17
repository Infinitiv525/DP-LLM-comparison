import unittest
import numpy as np
import torch
from fpo_case_v24 import (split_frames, read_training_observations,
    normalization_from_observed, training_target, known_inputs, canonicalize_labels)


class GuardedReference:
    def __init__(self, data):
        self.data = data
        self.shape = data.shape
        self.read_frames = []

    def __getitem__(self, frames):
        allowed = set(split_frames()['train'])
        if not isinstance(frames, list) or not 1 <= len(frames) <= 8 or not set(frames) <= allowed:
            raise AssertionError('Loader accessed a withheld/non-TRAIN frame')
        self.read_frames += frames
        return self.data[frames]


class CaseDataTests(unittest.TestCase):
    def fixture(self):
        rng = np.random.default_rng(24)
        data = rng.normal(size=(242, 8, 16, 3))
        data[:, :, :, 2] += np.arange(242)[:, None, None]*3
        fluid = np.ones((8, 16), dtype=bool)
        fluid[2, 3] = False
        return data, fluid

    def test_split_is_disjoint_complete_and_keeps_endpoints(self):
        roles = split_frames()
        self.assertEqual([len(roles[k]) for k in ['train','validation','audit']], [160,39,39])
        sets = [set(roles[k]) for k in ['train','validation','audit']]
        self.assertEqual(set.union(*sets), set(range(4,242)))
        self.assertFalse(sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
        self.assertTrue({4,241} <= sets[0])
        self.assertTrue(all(f-1 in sets[0] for f in roles['validation']))

    def test_loader_never_reads_withheld_and_stats_ignore_them(self):
        data, fluid = self.fixture()
        first = GuardedReference(data)
        observed = read_training_observations(first, fluid, 'cpu')
        changed = data.copy()
        withheld = split_frames()['validation'] + split_frames()['audit']
        changed[withheld] = 1e20
        second = GuardedReference(changed)
        observed2 = read_training_observations(second, fluid, 'cpu')
        self.assertEqual(first.read_frames, split_frames()['train'])
        self.assertEqual(second.read_frames, first.read_frames)
        for frame in observed:
            torch.testing.assert_close(observed[frame], observed2[frame], rtol=0, atol=0)
        normal = normalization_from_observed(observed, fluid)
        self.assertEqual(normal, normalization_from_observed(observed2, fluid))
        self.assertEqual(normal['population_fluid_points'], 160*int(fluid.sum()))

    def test_pressure_gauge_solid_mask_and_population_stats(self):
        data, fluid = self.fixture()
        untouched = data.copy()
        block = canonicalize_labels(data[[4,5]], fluid, 'cpu')
        np.testing.assert_array_equal(data, untouched)
        torch.testing.assert_close(block[:, :, :, 2][:, fluid].mean(1), torch.zeros(2), rtol=0, atol=2e-6)
        self.assertTrue(torch.all(block[:, 2, 3] == 0))
        observed = read_training_observations(GuardedReference(data), fluid, 'cpu')
        normal = normalization_from_observed(observed, fluid)
        cells = torch.cat([v[fluid].double() for v in observed.values()])
        torch.testing.assert_close(torch.tensor(normal['mean'],dtype=torch.float64), cells.mean(0), rtol=1e-10,atol=1e-10)
        torch.testing.assert_close(torch.tensor(normal['std'],dtype=torch.float64), cells.std(0,correction=0).clamp_min(1e-4),rtol=1e-10,atol=1e-10)

    def test_training_target_and_predictor_key_guards(self):
        data, fluid = self.fixture()
        observed = read_training_observations(GuardedReference(data), fluid, 'cpu')
        case = dict(observed=observed, raw=torch.zeros(8,16,6), geometry=torch.zeros(1,2,1,2),
            fluid=torch.tensor(fluid), record={'reynolds':382,'field':'must_not_reach_predictor'},
            reference=object(), normalization={'mean':[0.,0.,0.]})
        torch.testing.assert_close(training_target(case,4,[1],[1]), observed[4][[1],[1]])
        forbidden = split_frames()['validation'][0]
        case['observed'][forbidden] = torch.ones(8,16,3)
        with self.assertRaises(ValueError):training_target(case,forbidden,[1],[1])
        with self.assertRaises(ValueError):training_target(case,3,[1],[1])
        inputs = known_inputs(case)
        self.assertEqual(set(inputs), {'raw','geometry','fluid','record'})
        self.assertEqual(inputs['record'], {'reynolds':382})
        with self.assertRaises(ValueError):normalization_from_observed(case['observed'], fluid)

    def test_variance_floor_and_invalid_read_block(self):
        data = np.ones((242,8,16,3))
        fluid = np.ones((8,16),dtype=bool)
        normal = normalization_from_observed(read_training_observations(GuardedReference(data),fluid,'cpu'),fluid)
        self.assertEqual(normal['std'],[1e-4,1e-4,1e-4])
        with self.assertRaises(ValueError):read_training_observations(GuardedReference(data),fluid,'cpu',9)


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
