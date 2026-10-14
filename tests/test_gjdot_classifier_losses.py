import unittest
import numpy as np

from GJDOT.gjdot_classifiers import (
    GINJDOTClassifier,
    classifier_label_loss_metadata,
    resolve_classifier_label_loss,
)
from GJDOT.gjdot_ot import label_loss


class LabelLossTest(unittest.TestCase):
    def test_squared_error_is_pairwise_sum_over_full_class_vectors(self):
        y_source = np.array([[1.0, 0.0], [0.0, 1.0]])
        y_pred = np.array([[0.2, 0.8], [0.7, 0.3]])

        actual = label_loss(y_source, y_pred, label_loss_type="squared_error")

        expected = np.array([[1.28, 0.18], [0.08, 0.98]])
        np.testing.assert_allclose(actual, expected)
        self.assertEqual(actual.shape, (2, 2))

    def test_cross_entropy_regression(self):
        y_source = np.array([[1.0, 0.0], [0.0, 1.0]])
        y_pred = np.array([[0.8, 0.2], [0.25, 0.75]])

        actual = label_loss(y_source, y_pred, label_loss_type="cross_entropy")

        expected = -np.log(np.array([[0.8, 0.25], [0.2, 0.75]]))
        np.testing.assert_allclose(actual, expected)

    def test_squared_hinge_regression(self):
        y_source = np.array([[1.0, 0.0], [0.0, 1.0]])
        y_pred = np.array([[0.25, -0.5]])

        actual = label_loss(y_source, y_pred, label_loss_type="squared_hinge")

        expected = np.array([[0.75**2 + 0.5**2], [1.25**2 + 1.5**2]])
        np.testing.assert_allclose(actual, expected)


class ClassifierOwnedLossTest(unittest.TestCase):
    def test_loss_resolution_matches_classifier_training_objective(self):
        self.assertEqual(
            resolve_classifier_label_loss({"classifier": "gin", "label_loss": "cross_entropy"}),
            "cross_entropy",
        )
        self.assertEqual(
            resolve_classifier_label_loss({"classifier": "gin", "label_loss": "squared_hinge"}),
            "squared_hinge",
        )
        with self.assertRaises(ValueError):
            resolve_classifier_label_loss({"classifier": "wl_krr"})

    def test_loss_metadata_records_matching_training_and_ot_losses(self):
        expected = {
            "cross_entropy": "cross_entropy",
            "squared_hinge": "squared_hinge",
        }
        for configured_loss, resolved_loss in expected.items():
            with self.subTest(label_loss=configured_loss):
                metadata = classifier_label_loss_metadata(
                    {"classifier": "gin", "label_loss": configured_loss}
                )
                self.assertEqual(
                    metadata,
                    {
                        "classifier_training_loss": resolved_loss,
                        "ot_label_loss": resolved_loss,
                    },
                )

    def test_gin_uses_configured_cross_entropy_for_training_and_ot(self):
        classifier = object.__new__(GINJDOTClassifier)
        classifier.params = {"classifier": "gin", "label_loss": "cross_entropy"}
        classifier.predict_probs = lambda: np.array([[0.1], [0.9]])
        classifier.predict_scores = lambda: self.fail("Cross-entropy must not use raw scores.")
        supplied_probs = np.array([[0.2], [0.8]])

        actual = classifier.compute_ot_label_loss_matrix(
            np.array([[0.0], [1.0]]),
            predicted_probs=supplied_probs,
        )

        expected = label_loss(
            np.array([[0.0], [1.0]]),
            supplied_probs,
            label_loss_type="cross_entropy",
        )
        self.assertEqual(classifier.training_label_loss_name, "cross_entropy")
        self.assertEqual(classifier.ot_label_loss_name, "cross_entropy")
        np.testing.assert_allclose(actual, expected)

    def test_gin_uses_configured_squared_hinge_scores_for_ot(self):
        classifier = object.__new__(GINJDOTClassifier)
        classifier.params = {"classifier": "gin", "label_loss": "squared_hinge"}
        scores = np.array([[-0.5], [0.75]])
        classifier.predict_scores = lambda: scores

        actual = classifier.compute_ot_label_loss_matrix(
            np.array([[0.0], [1.0]]),
            predicted_probs=np.array([[0.4], [0.6]]),
        )

        expected = label_loss(
            np.array([[0.0], [1.0]]),
            scores,
            label_loss_type="squared_hinge",
        )
        self.assertEqual(classifier.training_label_loss_name, "squared_hinge")
        self.assertEqual(classifier.ot_label_loss_name, "squared_hinge")
        np.testing.assert_allclose(actual, expected)


if __name__ == "__main__":
    unittest.main()
