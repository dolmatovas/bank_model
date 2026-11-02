"""Small end-to-end check of backward fitting and saved policy reuse."""

import tempfile
from pathlib import Path
import unittest

import torch

from .environment import Environment
from .solver import (FittedSolution, SolverConfig, generate_state_cloud,
                     rollout_financial_values, sample_initial_states, solve)


class SolverTests(unittest.TestCase):
    def test_training_starts_vary_rate_and_car(self):
        p = Environment().p
        states = sample_initial_states(100, p, torch.Generator().manual_seed(17),
                                       device=torch.device("cpu"), dtype=torch.float64)
        self.assertGreater(states[:, 0].std().item(), .01)
        self.assertGreater((states[:, 3] / p.loans).std().item(), .01)

    def test_legacy_initial_book_does_not_change_with_new_spread_curve(self):
        from dataclasses import replace
        p = Environment().p
        left = sample_initial_states(100, p, torch.Generator().manual_seed(7),
                                     device=torch.device("cpu"), dtype=torch.float64)
        right = sample_initial_states(
            100, replace(p, floating_demand_slope=.01),
            torch.Generator().manual_seed(7), device=torch.device("cpu"),
            dtype=torch.float64)
        torch.testing.assert_close(left, right, atol=0, rtol=0)

    def test_state_cloud_covers_persistent_portfolios(self):
        env = Environment().double()
        cfg = SolverConfig(horizon=5, steps=30)
        cloud = generate_state_cloud(env, cfg, 256, 19,
                                     device=torch.device("cpu"), dtype=torch.float64)
        self.assertLess(cloud[-1][:, 1].min().item(), 25)
        self.assertGreater(cloud[-1][:, 1].max().item(), 75)

    def test_backward_solution_save_load_and_rollout(self):
        torch.manual_seed(3)
        env = Environment().double()
        cfg = SolverConfig(horizon=.25, steps=2, quadrature_nodes=5,
                           action_count=9, train_paths=128, validation_paths=64,
                           target_batch=64, train_batch=64, epochs=15,
                           terminal_epochs=15,
                           patience=5, width=16, seed=3)
        device = torch.device("cpu")
        train = generate_state_cloud(env, cfg, 128, 4, device=device, dtype=torch.float64)
        valid = generate_state_cloud(env, cfg, 64, 5, device=device, dtype=torch.float64)
        solution, diagnostics = solve(env, cfg, train, valid)
        x = torch.tensor([[.08, 50., 5.625, 12.]], dtype=torch.float64)
        before_value = solution.value(0, x)
        before_action = solution.action(0, x)
        self.assertTrue(torch.isfinite(before_value).all())
        self.assertTrue(((before_action >= 0) & (before_action <= 1)).all())
        self.assertEqual(len(diagnostics), 2)
        evaluated, _ = solve(env, cfg, train, valid,
                             policy=lambda layer, state: solution.action(layer, state),
                             epoch_limit=5)
        self.assertTrue(torch.isfinite(evaluated.value(0, x)).all())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "solution.pt"
            solution.save(path, diagnostics)
            loaded = FittedSolution.load(path, dtype=torch.float64)
            torch.testing.assert_close(loaded.value(0, x), before_value, atol=0, rtol=0)
            torch.testing.assert_close(loaded.action(0, x), before_action, atol=0, rtol=0)
            rollout = rollout_financial_values(
                loaded, x.expand(8, -1), torch.zeros(2, 8, dtype=torch.float64),
                action_count=9)
            self.assertTrue(torch.isfinite(torch.tensor(rollout["mean_payoff"])))


if __name__ == "__main__":
    unittest.main()
