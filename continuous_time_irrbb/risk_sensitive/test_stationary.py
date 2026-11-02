"""Checks for the perpetual growing-book scaling and signed owner payouts."""

from dataclasses import replace
import math
from pathlib import Path
import tempfile
import unittest

import torch

from .bellman import BellmanOperator
from .environment import Environment, Parameters
from .stationary_solver import (StationaryConfig, StationarySolution,
                                StationaryValueNetwork, rescale_amounts,
                                stationary_continuation)


class StationaryTests(unittest.TestCase):
    def setUp(self):
        self.env = Environment(replace(Parameters(), loan_growth=.05,
                                       dividend_fraction=.9,
                                       signed_owner_flow=True)).double()
        self.x = torch.tensor([[.08, 50., 5.625, 12.]], dtype=torch.float64)

    def test_signed_owner_payout_and_equity_drift(self):
        x = torch.tensor([[.08, 50., 0., -50.]], dtype=torch.float64)
        profit = self.env.profit(x).item()
        self.assertLess(profit, 0)
        self.assertAlmostEqual(self.env.dividend_rate(x).item(), .9 * profit)
        self.assertAlmostEqual(self.env.drift(x, torch.tensor([.5], dtype=x.dtype))[0, 3].item(),
                               .1 * profit)

    def test_growth_normalization_is_time_homogeneous(self):
        dt, t = .25, 2.
        growth = math.exp(self.env.p.loan_growth * t)
        action = torch.tensor([.4], dtype=self.x.dtype)
        noise = torch.tensor([1.2], dtype=self.x.dtype)
        initial = self.env.step(self.x, action, noise, dt, time=0)
        later = self.env.step(rescale_amounts(self.x, growth), action, noise,
                              dt, time=t)
        torch.testing.assert_close(later.state,
                                   rescale_amounts(initial.state, growth),
                                   rtol=1e-11, atol=1e-11)
        torch.testing.assert_close(later.dividend_pv,
                                   growth * initial.dividend_pv,
                                   rtol=1e-11, atol=1e-11)

    def test_perpetual_equity_value_identity_in_par_bank(self):
        p = replace(self.env.p, deposits=100, deposit_beta=1,
                    deposit_spread=0, operating_cost=0, funding_spread=0,
                    rate_vol=0, fixed_spread=0, floating_spread_0=0,
                    floating_demand_slope=0)
        env = Environment(p).double()
        x = torch.tensor([[.08, 50., 4., 12.]], dtype=torch.float64)
        op = BellmanOperator(env, .1, 5).double()
        for action in (0., .5, 1.):
            q = op.evaluate(0, x, torch.tensor([action], dtype=x.dtype), .25,
                            lambda _time, state: state[..., 3])
            torch.testing.assert_close(q, x[..., 3], rtol=1e-10, atol=1e-10)

    def test_stationary_value_and_policy_scale_with_book(self):
        torch.manual_seed(5)
        net = StationaryValueNetwork(self.env.p.loans, 16, 1).double()
        solution = StationarySolution(self.env, StationaryConfig(
            dt=.25, quadrature_nodes=5, action_count=9, width=16, depth=1), net)
        t = 3.
        growth = math.exp(self.env.p.loan_growth * t)
        later_state = rescale_amounts(self.x, growth)
        torch.testing.assert_close(solution.value(later_state, time=t),
                                   growth * solution.value(self.x), rtol=1e-12,
                                   atol=1e-12)
        torch.testing.assert_close(solution.action(later_state, time=t),
                                   solution.action(self.x), rtol=0, atol=0)
        continuation = stationary_continuation(self.env, net)
        torch.testing.assert_close(continuation(t, later_state),
                                   growth * net(self.x), rtol=1e-12, atol=1e-12)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stationary_solution.pt"
            solution.save(path, [])
            loaded = StationarySolution.load(path, dtype=torch.float64)
            torch.testing.assert_close(loaded.value(self.x), solution.value(self.x),
                                       rtol=0, atol=0)

    def test_rate_sensitive_growth_and_pathwise_balance(self):
        env = Environment(replace(self.env.p, growth_reference_rate=.08,
                                  growth_rate_sensitivity=5)).double()
        dt = .25
        states = self.x.expand(2, -1).clone()
        states[1, 0] = .09
        action = torch.full((2,), .4, dtype=states.dtype)
        transition = env.step(states, action, torch.zeros(2, dtype=states.dtype), dt)
        issuance = env.p.amortization + env.growth_rate(states[:, 0])
        self.assertAlmostEqual((issuance[1] / issuance[0]).item(), math.exp(-.05))
        self.assertAlmostEqual(transition.next_loans[0].item(),
                               env.p.loans * math.exp(env.p.loan_growth * dt))
        self.assertLess(transition.next_loans[1].item(), transition.next_loans[0].item())
        expected_fixed = (states[:, 1] * math.exp(-env.p.amortization * dt)
                          + action * (transition.next_loans
                                      - env.p.loans * math.exp(-env.p.amortization * dt)))
        torch.testing.assert_close(transition.state[:, 1], expected_fixed)
        with self.assertRaisesRegex(ValueError, "pathwise loan balance"):
            env.profit(transition.state, time=dt)
        self.assertTrue(torch.isfinite(env.profit(transition.state, time=dt,
                                                loans=transition.next_loans)).all())

    def test_variable_growth_bellman_continuation_uses_realized_book(self):
        env = Environment(replace(self.env.p, growth_rate_sensitivity=5)).double()
        torch.manual_seed(9)
        net = StationaryValueNetwork(env.p.loans, 16, 1).double()
        continuation = stationary_continuation(env, net)
        action = torch.tensor([.4], dtype=self.x.dtype)
        step = env.step(self.x, action, torch.zeros_like(action), .25)
        expected = step.book_growth * net(rescale_amounts(
            step.state, step.book_growth.reciprocal()))
        torch.testing.assert_close(continuation(.25, step.state,
                                                 book_growth=step.book_growth), expected)
        operator = BellmanOperator(env, 0, 5).double()
        actual = operator.evaluate(0, self.x, action, .25, continuation)
        self.assertEqual(actual.shape, (1,))
        self.assertTrue(torch.isfinite(actual).all())


if __name__ == "__main__":
    unittest.main()
