"""Exercise actual logger aggregation without requiring Isaac Gym or PyTorch."""
import ast
from pathlib import Path
import unittest
from types import SimpleNamespace

SOURCE = Path(__file__).resolve().parents[1] / 'rsl_rl/rsl_rl/runners/him_on_policy_runner.py'

class Tensor(list):
    @property
    def shape(self):
        return (len(self),)

    def to(self, device):
        return self

TORCH = SimpleNamespace(Tensor=Tensor, tensor=lambda values, **kw: Tensor(values),
                        cat=lambda items: Tensor(sum((list(x) for x in items), [])),
                        mean=lambda values: sum(values) / len(values))

class EpisodeLoggingTest(unittest.TestCase):
    def aggregate(self, name, infos, source=None):
        tree = ast.parse(source or SOURCE.read_text())
        method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)
        block = next(n for n in method.body if isinstance(n, ast.If))
        results = {}
        runner = SimpleNamespace(device='cpu', writer=SimpleNamespace(
            add_scalar=lambda key, value, step: results.__setitem__(key, value)))
        scope = dict(torch=TORCH, self=runner, locs={'ep_infos': infos, 'it': 0}, pad=35, ep_string='')
        exec(compile(ast.Module(body=[block], type_ignores=[]), str(SOURCE), 'exec'), scope)
        return results

    def test_missing_metrics_both_orders(self):
        for name in ('log', 'log_vision'):
            for reverse in (False, True):
                records = [{'reward': 10., 'true_save_rate': 1.}, {'reward': -4.}]
                if reverse:
                    records.reverse()
                self.assertEqual(self.aggregate(name, records),
                                 {'Episode/reward': 3., 'Episode/true_save_rate': 1.})

    def test_empty_and_vector_records(self):
        for name in ('log', 'log_vision'):
            self.assertEqual(self.aggregate(name, []), {})
            self.assertEqual(self.aggregate(name, [{}, {'reward': Tensor([2., 4.])}]),
                             {'Episode/reward': 3.})

    def test_original_bug_reproduced(self):
        original = SOURCE.read_text().replace(
            "for key in dict.fromkeys(key for info in locs['ep_infos'] for key in info):",
            "for key in locs['ep_infos'][0]:").replace(
            '                    if key not in ep_info:\n                        continue\n', '')
        with self.assertRaises(KeyError):
            self.aggregate('log', [{'true_save_rate': 1.}, {}], original)

if __name__ == '__main__':
    unittest.main()
