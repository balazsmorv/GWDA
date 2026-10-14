import unittest

import numpy as np
import ot

from GJDOT.gjdot_label_shift import (
    class_share,
    estimate_target_proportions,
    source_marginal,
)


def _gaussian_classes(rng, n0, n1, shift=4.0):
    x = np.concatenate([rng.normal(0.0, 1.0, (n0, 2)), rng.normal(shift, 1.0, (n1, 2))])
    y = np.concatenate([np.zeros(n0, dtype=np.int64), np.ones(n1, dtype=np.int64)])
    return x, y


class SourceMarginalTest(unittest.TestCase):
    def test_uniform_matches_ot_unif(self):
        ys = np.array([0, 1, 1, 0, 1])
        np.testing.assert_array_equal(source_marginal(ys), ot.unif(5))

    def test_reweighted_puts_pi_hat_mass_on_each_class(self):
        ys = np.array([0, 0, 0, 1])
        a = source_marginal(ys, pi_hat=np.array([0.4, 0.6]))
        np.testing.assert_allclose(a, [0.4 / 3, 0.4 / 3, 0.4 / 3, 0.6])
        self.assertAlmostEqual(a.sum(), 1.0)

    def test_reweighted_with_source_prior_is_uniform(self):
        ys = np.array([0, 1, 1, 0, 1])
        a = source_marginal(ys, pi_hat=np.array([0.4, 0.6]))
        np.testing.assert_allclose(a, ot.unif(5))


class EstimatorTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.xs, self.ys = _gaussian_classes(rng, 200, 200)
        self.xt, self.yt = _gaussian_classes(rng, 210, 90)
        self.q_star = class_share(self.yt)

    def test_oracle(self):
        pi = estimate_target_proportions("oracle", y_target=self.yt)
        np.testing.assert_allclose(pi, [0.7, 0.3])

    def test_ot_recovers_share(self):
        D = ot.dist(self.xs, self.xt)
        pi = estimate_target_proportions("ot", ys=self.ys, D=D)
        self.assertLess(abs(pi[1] - self.q_star), 0.03)
        self.assertAlmostEqual(pi.sum(), 1.0)


if __name__ == "__main__":
    unittest.main()
