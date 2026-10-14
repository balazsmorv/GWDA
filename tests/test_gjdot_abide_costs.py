import unittest

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from GJDOT.gjdot_abide import build_connectome_graph, fisher_z_connectivity, top_percent_edges
from GJDOT.gjdot_costs import (
    compute_cost_matrix,
    connectivity_matrix,
    correlation_distance_matrix,
    fgw_cost_matrix,
    frobenius_cost_matrix,
)


def _random_graphs(num_graphs, num_rois=12, num_timepoints=60, seed=0, **kwargs):
    rng = np.random.default_rng(seed)
    return [
        build_connectome_graph(
            rng.standard_normal((num_timepoints, num_rois)),
            label=idx % 2,
            site="SYNTH",
            subject_id=idx,
            **kwargs,
        )
        for idx in range(num_graphs)
    ]


class ConnectomeGraphTest(unittest.TestCase):
    def test_constant_roi_gets_zero_connectivity_instead_of_nan(self):
        ts = np.random.default_rng(0).standard_normal((50, 5))
        ts[:, 2] = 3.0

        z = fisher_z_connectivity(ts)

        self.assertTrue(np.all(np.isfinite(z)))
        np.testing.assert_array_equal(z[2], 0.0)
        np.testing.assert_array_equal(np.diag(z), 0.0)
        np.testing.assert_allclose(z, z.T)

    def test_top_percent_edges_keeps_strongest_positive_pairs_undirected(self):
        z = np.array(
            [
                [0.0, 0.9, -0.5, 0.1],
                [0.9, 0.0, 0.4, -0.2],
                [-0.5, 0.4, 0.0, 0.3],
                [0.1, -0.2, 0.3, 0.0],
            ]
        )

        edge_index, edge_weight = top_percent_edges(z, percent=50.0)

        pairs = {tuple(sorted(pair)) for pair in edge_index.t().tolist()}
        self.assertEqual(pairs, {(0, 1), (1, 2), (2, 3)})
        self.assertEqual(edge_index.size(1), 6)
        self.assertTrue(torch.all(edge_weight > 0))

    def test_negative_correlations_are_never_edges(self):
        z = -np.ones((4, 4))
        np.fill_diagonal(z, 0.0)

        edge_index, _ = top_percent_edges(z, percent=100.0)

        self.assertEqual(edge_index.size(1), 0)

    def test_graph_attributes_and_batching(self):
        graphs = _random_graphs(3, num_rois=12)
        batch = next(iter(DataLoader(graphs, batch_size=3)))

        self.assertEqual(tuple(graphs[0].x.shape), (12, 12))
        self.assertEqual(tuple(graphs[0].conn.shape), (1, 12, 12))
        self.assertEqual(tuple(batch.conn.shape), (3, 12, 12))
        self.assertEqual(tuple(batch.y.shape), (3,))
        np.testing.assert_allclose(connectivity_matrix(graphs[1]), batch.conn[1].numpy())

    def test_ones_node_features(self):
        graph = _random_graphs(1, node_feature_mode="ones")[0]

        self.assertEqual(tuple(graph.x.shape), (12, 1))


class CostMatrixTest(unittest.TestCase):
    def test_frobenius_matches_direct_norm(self):
        source = _random_graphs(3, seed=1)
        target = _random_graphs(2, seed=2)

        actual = frobenius_cost_matrix(source, target)

        expected = np.array(
            [
                [np.linalg.norm(connectivity_matrix(s) - connectivity_matrix(t)) for t in target]
                for s in source
            ]
        )
        np.testing.assert_allclose(actual, expected, rtol=1e-10)

    def test_correlation_distance_is_bounded_with_zero_diagonal(self):
        distance = correlation_distance_matrix(_random_graphs(1)[0])

        np.testing.assert_array_equal(np.diag(distance), 0.0)
        self.assertTrue(np.all(distance >= 0.0) and np.all(distance <= 2.0))

    def test_fgw_is_zero_on_identical_graphs_and_symmetric(self):
        graphs = _random_graphs(3, seed=3)

        cost = fgw_cost_matrix(graphs, graphs, alpha=0.5)

        np.testing.assert_allclose(np.diag(cost), 0.0, atol=1e-8)
        np.testing.assert_allclose(cost, cost.T, atol=1e-6)
        self.assertTrue(np.all(cost[~np.eye(3, dtype=bool)] > 0))

    def test_fgw_parallel_matches_serial(self):
        source = _random_graphs(2, seed=4)
        target = _random_graphs(3, seed=5)

        serial = fgw_cost_matrix(source, target, alpha=0.3, n_jobs=1)
        parallel = fgw_cost_matrix(source, target, alpha=0.3, n_jobs=2)

        self.assertEqual(serial.shape, (2, 3))
        np.testing.assert_allclose(serial, parallel)

    def test_unknown_cost_type_raises(self):
        graphs = _random_graphs(1)
        with self.assertRaises(ValueError):
            compute_cost_matrix(graphs, graphs, cost_type="wasserstein")


if __name__ == "__main__":
    unittest.main()
