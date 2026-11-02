"""Economic and Bellman identities for the proposed model."""

from dataclasses import replace
import math
import unittest

import torch

from .bellman import BellmanOperator
from .environment import Environment, Parameters


def tensor(value):
    return torch.tensor(value, dtype=torch.float64)


class EnvironmentTests(unittest.TestCase):
    def setUp(self):
        self.env = Environment()
        self.state = tensor([[0.08, 50.0, 5.0, 12.0]])

    def test_spread_memory_and_floating_pv(self):
        x = tensor([[0.08, 0.0, 0.0, 10.0]])
        dt = 0.3
        first = self.env.step(x, tensor([0.0]), tensor([0.0]), dt)
        expected_y = (1 - math.exp(-self.env.p.amortization * dt)) * (
            self.env.p.loans * (self.env.p.floating_spread_0 - self.env.p.floating_demand_slope))
        self.assertAlmostEqual(first.state[0, 2].item(), expected_y)
        self.assertAlmostEqual(first.state[0, 1].item(), 0.0)
        second = self.env.step(first.state, tensor([1.0]), tensor([0.0]), dt)
        self.assertGreater(second.state[0, 2].item(), expected_y * math.exp(-self.env.p.amortization * dt))
        torch.testing.assert_close(self.env.loan_pv(first.state),
                                   self.env.p.loans + first.state[..., 2] * self.env.annuity_q(first.state[..., 0]))

    def test_fixed_par_and_signed_terminal(self):
        env = Environment(replace(self.env.p, fixed_spread=0))
        r = tensor([0.02, 0.08, 0.2])
        f = torch.full_like(r, env.p.loans)
        x = torch.stack((r, f, f * env.new_fixed_coupon(r), torch.ones_like(r)), -1)
        torch.testing.assert_close(env.loan_pv(x), f)
        self.assertLess(env.terminal_value(tensor([[0.08, 0.0, 0.0, -30.0]])).item(), 0)

    def test_discounted_equity_and_dividend_identity(self):
        x = self.state
        out = self.env.step(x, tensor([0.4]), tensor([0.0]), 0.1)
        lhs = out.discount * out.state[..., 3] + out.dividend_pv - x[..., 3]
        rhs = self.env.exp_integral(-x[..., 0], .1) * out.ex_cash_profit_mid
        torch.testing.assert_close(lhs, rhs, atol=1e-12, rtol=1e-12)

    def test_invalid_spread_curve(self):
        with self.assertRaises(ValueError):
            Parameters(floating_spread_0=.01, floating_demand_slope=.02)

    def test_default_spreads_match_at_zero_demand_slope(self):
        p = Parameters()
        self.assertEqual(p.fixed_spread, p.floating_spread_0)
        self.assertEqual(Parameters(floating_demand_slope=0).fixed_spread,
                         Parameters(floating_demand_slope=0).floating_spread_0)

    def test_growing_book_and_random_car(self):
        p = replace(self.env.p, loan_growth=.05, dividend_fraction=.9)
        env = Environment(p).double()
        x = self.state.expand(2, -1).clone()
        actions = tensor([.4, .4])
        first = env.step(x, actions, tensor([-2., 2.]), .5, time=0)
        second = env.step(first.state, actions, tensor([0., 0.]), .5, time=.5)
        expected_f = (math.exp(-p.amortization) * 50
                      + (env.loans_at(1) - math.exp(-p.amortization) * p.loans) * .4)
        self.assertAlmostEqual(second.state[0, 1].item(), expected_f)
        self.assertAlmostEqual(env.deposits_at(1) / env.loans_at(1), p.deposits / p.loans)
        self.assertNotAlmostEqual((second.state[0, 3] / env.loans_at(1)).item(),
                                  (second.state[1, 3] / env.loans_at(1)).item())
        env.validate_state(second.state, time=1)

    def test_all_profit_payout_does_not_keep_car_constant(self):
        env = Environment(replace(self.env.p, loan_growth=.05,
                                  dividend_fraction=1)).double()
        x = self.state.clone()
        after = env.step(x, tensor([.5]), tensor([0.]), .1, time=0).state
        self.assertLess((after[0, 3] / env.loans_at(.1)).item(),
                        (x[0, 3] / env.loans_at(0)).item())


class BellmanTests(unittest.TestCase):
    def setUp(self):
        self.env = Environment()
        self.x = tensor([[.08, 50.0, 5.0, 12.0]])

    def test_gaussian_entropic_step_and_risk_neutral_limit(self):
        dt, a, gamma = .1, 30.0, .4
        action = tensor([.4])
        mean_step = self.env.step(self.x, action, tensor([0.0]), dt)
        mean_future = a * mean_step.state[..., 0]
        variance = (a * self.env.p.rate_vol) ** 2 * self.x[..., 0] * dt
        future = lambda t, x: a * x[..., 0]
        risk_value = BellmanOperator(self.env, gamma, 11).evaluate(0, self.x, action, dt, future)
        neutral_value = BellmanOperator(self.env, 0, 11).evaluate(0, self.x, action, dt, future)
        expected_neutral = mean_step.dividend_pv + mean_step.discount * mean_future
        expected_risk = expected_neutral - mean_step.discount * gamma * variance / 2
        torch.testing.assert_close(neutral_value, expected_neutral, atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(risk_value, expected_risk, atol=1e-12, rtol=1e-12)
        self.assertLess(risk_value.item(), neutral_value.item())

    def test_grid_search_finds_interior_action(self):
        p = replace(self.env.p, rate_vol=0, floating_spread_0=.05,
                    floating_demand_slope=.04, operating_cost=1.0)
        env = Environment(p)
        x = tensor([[.08, 50.0, 5.0, 12.0]])
        coupon = env.new_fixed_coupon(x[..., 0]).item()
        coefficient_f = -coupon + p.floating_spread_0 - p.floating_demand_slope
        future = lambda t, state: coefficient_f * state[..., 1] + state[..., 2]
        result = BellmanOperator(env, .2, 5).maximize(0, x, .1, future, action_count=101)
        self.assertAlmostEqual(result.action.item(), .5, places=2)
        self.assertGreater(result.value.item(), result.q_values[0, 0].item())
        self.assertGreater(result.value.item(), result.q_values[0, -1].item())

    def test_terminal_continuation_and_shapes(self):
        op = BellmanOperator(self.env, .1, 7)
        states = self.x.expand(3, -1)
        result = op.maximize(0, states, .1, lambda t, x: self.env.terminal_value(x),
                             action_count=17)
        self.assertEqual(result.action.shape, (3,))
        self.assertEqual(result.q_values.shape, (3, 17))
        self.assertTrue(torch.isfinite(result.value).all())


if __name__ == "__main__":
    unittest.main()
