"""Test shot selection without starting Isaac Gym."""
import ast
from pathlib import Path
import unittest


class SelectionTests(unittest.TestCase):
    def test_selection_and_unknown_id(self):
        path = Path(__file__).resolve().parents[1] / 'legged_gym/legged_gym/scripts/evaluate_k1.py'
        tree = ast.parse(path.read_text())
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'select_shots')
        scope = {}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), 'exec'), scope)
        select = scope['select_shots']
        rows = [{'shot_id': '1'}, {'shot_id': '49'}]
        self.assertIs(select(rows, None), rows)
        self.assertEqual(select(rows, 49), [rows[1]])
        for invalid in (0, -1, 181):
            with self.assertRaises(ValueError):
                select(rows, invalid)
