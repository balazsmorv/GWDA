import tempfile
import unittest
from pathlib import Path
from unittest import mock

import mlflow
import numpy as np

import gjdot_pipeline_ABIDE as pipeline
from GJDOT.gjdot_abide import build_connectome_graph
from GJDOT.gjdot_costs import normalize_graph_cost


def _synthetic_domain(seed, num_graphs=8, num_rois=10):
    rng = np.random.default_rng(seed)
    return tuple(
        build_connectome_graph(
            rng.standard_normal((40, num_rois)),
            label=idx % 2,
            site=f"SITE{seed}",
            subject_id=idx,
        )
        for idx in range(num_graphs)
    )


def _smoke_params(**overrides):
    params = {key: values[0] for key, values in pipeline.GRID.items()}
    params.update(
        source_domain="LEUVEN",
        target_domain="USM",
        baseline_epochs=1,
        jdot_fit_epochs=1,
        bcd_iters=2,
        batch_size=4,
    )
    params.update(overrides)
    return params


class GraphCostNormalizationTest(unittest.TestCase):
    def test_minmax_maps_onto_unit_interval(self):
        cost = np.array([[45.0, 60.0], [79.0, 50.0]])

        normalized = normalize_graph_cost(cost, "minmax")

        self.assertAlmostEqual(normalized.min(), 0.0)
        self.assertAlmostEqual(normalized.max(), 1.0)
        np.testing.assert_allclose(normalized[0, 1], 15.0 / 34.0)

    def test_max_and_none(self):
        cost = np.array([[1.0, 4.0]])

        np.testing.assert_allclose(normalize_graph_cost(cost, "max"), [[0.25, 1.0]])
        np.testing.assert_allclose(normalize_graph_cost(cost, None), cost)


class PipelineTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        tmp = Path(self.tmp.name)
        domains = {"LEUVEN": _synthetic_domain(1, num_graphs=8), "USM": _synthetic_domain(2, num_graphs=6)}
        patches = [
            mock.patch.object(pipeline, "PROJECT_ROOT", tmp),
            mock.patch.object(pipeline, "COST_N_JOBS", 1),
            mock.patch.object(pipeline, "SITE_GROUPS", {"LEUVEN": ["LEUVEN_1", "LEUVEN_2"], "USM": ["USM"]}),
            mock.patch.object(
                pipeline,
                "load_domain_graphs",
                lambda data_dir, domain, *args: domains[domain],
            ),
        ]
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)
        mlflow.set_tracking_uri(f"sqlite:///{tmp / 'mlflow.db'}")
        mlflow.create_experiment("test", artifact_location=(tmp / "mlruns").as_uri())
        mlflow.set_experiment("test")

    def tearDown(self):
        self.tmp.cleanup()

    def test_redundant_frobenius_combo_is_skipped(self):
        grid = {"fgw_alpha": [0.5, 0.9]}

        self.assertTrue(pipeline.is_redundant_cost_combo({"cost_type": "frobenius", "fgw_alpha": 0.9}, grid))
        self.assertFalse(pipeline.is_redundant_cost_combo({"cost_type": "frobenius", "fgw_alpha": 0.5}, grid))
        self.assertFalse(pipeline.is_redundant_cost_combo({"cost_type": "fgw", "fgw_alpha": 0.9}, grid))

    def test_run_experiment_end_to_end_for_both_costs(self):
        for cost_type in ("frobenius", "fgw"):
            with self.subTest(cost_type=cost_type), mlflow.start_run() as run:
                pipeline.run_experiment(_smoke_params(cost_type=cost_type))

            metrics = mlflow.get_run(run.info.run_id).data.metrics
            self.assertIn("final_target_accuracy", metrics)
            self.assertIn("baseline_target_accuracy", metrics)
            self.assertIn("baseline_source_val_accuracy", metrics)
            self.assertIn("baseline_source_val_by_epoch_accuracy", metrics)
            self.assertTrue(pipeline.make_cost_path(_smoke_params(cost_type=cost_type)).exists())

    def test_baseline_val_history_is_logged_per_epoch(self):
        with mlflow.start_run() as run:
            pipeline.run_experiment(_smoke_params(baseline_epochs=3, early_stopping_patience=10))

        history = mlflow.MlflowClient().get_metric_history(run.info.run_id, "baseline_source_val_by_epoch_accuracy")
        self.assertEqual([m.step for m in history], [1, 2, 3])

    def test_label_shift_estimators_reweight_source_marginal(self):
        for estimator in (None, "oracle", "ot"):
            with self.subTest(estimator=estimator), mlflow.start_run() as run:
                pipeline.run_experiment(_smoke_params(label_shift_estimator=estimator))

            metrics = mlflow.get_run(run.info.run_id).data.metrics
            self.assertIn("final_target_accuracy", metrics)
            self.assertIn("label_shift_q_hat", metrics)
            if estimator is None:
                self.assertAlmostEqual(metrics["label_shift_q_hat"], metrics["label_shift_q_source"])
            if estimator == "oracle":
                self.assertAlmostEqual(metrics["label_shift_q_hat"], metrics["label_shift_q_star"])

    def test_reverse_direction_reuses_transposed_cache(self):
        params = _smoke_params(cost_type="frobenius")
        with mlflow.start_run():
            pipeline.run_experiment(params)

        swapped = pipeline.swapped_domain_params(params)
        source, target = pipeline.load_domains(swapped)[1:]
        cost, path, transposed = pipeline.load_or_compute_pipeline_cost_matrix(swapped, source, target)

        self.assertTrue(transposed)
        self.assertEqual(path, pipeline.make_cost_path(params))
        self.assertEqual(cost.shape, (6, 8))

    def test_stale_cache_with_wrong_shape_raises(self):
        params = _smoke_params(cost_type="frobenius")
        path = pipeline.make_cost_path(params)
        path.parent.mkdir(parents=True)
        np.savez_compressed(path, cost_mtx=np.zeros((3, 3)))
        source, target = pipeline.load_domains(params)[1:]

        with self.assertRaises(ValueError):
            pipeline.load_or_compute_pipeline_cost_matrix(params, source, target)


if __name__ == "__main__":
    unittest.main()
