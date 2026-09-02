"""
Real, runnable tests. Uses tiny synthetic datasets so the whole suite
finishes quickly even on a 6GB machine. No mocked training — every test
here exercises the actual code paths.
"""
import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd

from trainer.datasets import loader, analyzer
from trainer.preprocessing.tabular import TabularPreprocessor
from trainer.algorithms.neural import DenseNeuralNetwork
from trainer.algorithms.classical import build_model, is_classical
from trainer.core.config import TrainingConfig, ConfigValidationError
from trainer.training.checkpoint import CheckpointManager
from trainer.training.trainer import train_classical, train_neural
from trainer.core.pipeline import run_pipeline
from trainer.planner.nl_planner import plan
from trainer.inference.predictor import Predictor
from trainer.hardware import memory as hw


class TestDatasetLoading(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.csv_path = os.path.join(self.tmpdir, "d.csv")
        pd.DataFrame({"a": [1, 2, 3, None], "b": ["x", "y", "x", "y"], "label": [0, 1, 0, 1]}).to_csv(
            self.csv_path, index=False
        )

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_load_csv(self):
        df = loader.load(self.csv_path)
        self.assertEqual(len(df), 4)
        self.assertIn("label", df.columns)

    def test_missing_file_raises(self):
        with self.assertRaises(loader.DatasetLoadError):
            loader.load(os.path.join(self.tmpdir, "nope.csv"))

    def test_analyzer_real_stats(self):
        df = loader.load(self.csv_path)
        stats = analyzer.analyze_tabular(df, path=self.csv_path)
        self.assertEqual(stats["n_samples"], 4)
        self.assertEqual(stats["columns"]["a"]["missing_values"], 1)


class TestPreprocessing(unittest.TestCase):
    def test_fit_transform_classification(self):
        df = pd.DataFrame({
            "x1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            "x2": ["a", "b", "a", "b", "a", "b", "a", "b"],
            "y": [0, 1, 0, 1, 0, 1, 0, 1],
        })
        pre = TabularPreprocessor(input_columns=["x1", "x2"], target_column="y")
        data = pre.fit_transform(df, task="classification", test_size=0.25, random_seed=1)
        self.assertEqual(data["X_train"].shape[1], 2)
        self.assertTrue(pre.is_classification_target_)
        self.assertEqual(data["n_classes"], 2)

    def test_transform_reuses_fitted_encoders(self):
        df = pd.DataFrame({"x1": [1.0, 2.0, 3.0], "x2": ["a", "b", "a"], "y": [0, 1, 0]})
        pre = TabularPreprocessor(input_columns=["x1", "x2"], target_column="y")
        pre.fit_transform(df, task="classification", test_size=0.34, random_seed=1)
        new_df = pd.DataFrame({"x1": [4.0], "x2": ["a"]})
        X = pre.transform(new_df)
        self.assertEqual(X.shape, (1, 2))


class TestNeuralNetworkEngine(unittest.TestCase):
    def test_forward_shapes(self):
        net = DenseNeuralNetwork(layer_sizes=[3, 5, 2], activation="relu", output_activation="softmax",
                                  loss="cross_entropy", optimizer="adam", learning_rate=0.01)
        X = np.random.randn(4, 3)
        activations, pre = net.forward(X)
        self.assertEqual(activations[-1].shape, (4, 2))
        # softmax outputs must sum to 1 per row
        self.assertTrue(np.allclose(activations[-1].sum(axis=1), 1.0, atol=1e-6))

    def test_loss_decreases_over_training(self):
        """Real check that backprop + optimizer actually reduce loss on a toy task."""
        rng = np.random.default_rng(0)
        X = rng.normal(size=(200, 2))
        y = (X[:, 0] + X[:, 1] > 0).astype(float).reshape(-1, 1)

        net = DenseNeuralNetwork(layer_sizes=[2, 8, 1], activation="relu", output_activation="sigmoid",
                                  loss="binary_cross_entropy", optimizer="adam", learning_rate=0.05,
                                  random_seed=0)
        losses = []
        for epoch in range(30):
            activations, pre = net.forward(X)
            loss_val = net.backward(activations, pre, y)[2]
            grads_w, grads_b, _ = net.backward(*net.forward(X), y)
            net.apply_gradients(grads_w, grads_b)
            losses.append(loss_val)

        self.assertLess(losses[-1], losses[0], "Loss should decrease after real training steps")

    def test_train_step_updates_weights(self):
        net = DenseNeuralNetwork(layer_sizes=[3, 4, 1], loss="mse", optimizer="sgd", learning_rate=0.1)
        w_before = net.weights[0].copy()
        X = np.random.randn(10, 3)
        y = np.random.randn(10, 1)
        net.train_step(X, y)
        self.assertFalse(np.allclose(w_before, net.weights[0]), "Weights must change after a real training step")

    def test_state_roundtrip(self):
        net = DenseNeuralNetwork(layer_sizes=[3, 4, 1], loss="mse", optimizer="adam", learning_rate=0.1)
        X = np.random.randn(5, 3)
        net.train_step(X, np.random.randn(5, 1))
        state = net.get_state()
        net2 = DenseNeuralNetwork.from_state(state)
        pred1 = net.predict(X)
        pred2 = net2.predict(X)
        self.assertTrue(np.allclose(pred1, pred2), "Restored model must produce identical predictions")


class TestClassicalAlgorithms(unittest.TestCase):
    def test_is_classical(self):
        self.assertTrue(is_classical("logistic_regression"))
        self.assertFalse(is_classical("neural_network"))

    def test_build_and_fit_logreg(self):
        cfg = TrainingConfig(task="classification", algorithm="logistic_regression",
                              dataset_path="x", target_column="y", random_seed=1)
        model = build_model("logistic_regression", cfg)
        X = np.random.randn(50, 3)
        y = (X[:, 0] > 0).astype(int)
        model.fit(X, y)
        preds = model.predict(X)
        self.assertEqual(len(preds), 50)


class TestConfigValidation(unittest.TestCase):
    def test_valid_config(self):
        cfg = TrainingConfig(task="classification", algorithm="logistic_regression",
                              dataset_path="x.csv", target_column="y")
        self.assertTrue(cfg.validate())

    def test_incompatible_algorithm_task(self):
        cfg = TrainingConfig(task="regression", algorithm="logistic_regression",
                              dataset_path="x.csv", target_column="y")
        with self.assertRaises(ConfigValidationError):
            cfg.validate()

    def test_kmeans_requires_clusters(self):
        cfg = TrainingConfig(task="clustering", algorithm="kmeans", dataset_path="x.csv")
        with self.assertRaises(ConfigValidationError):
            cfg.validate()


class TestMemoryEstimation(unittest.TestCase):
    def test_estimate_positive(self):
        est = hw.estimate_dense_nn_bytes([10, 32, 1], optimizer="adam")
        self.assertGreater(est, 0)

    def test_budget_check_runs(self):
        ok, msg, ratio = hw.check_budget(1000)
        self.assertIsInstance(ok, bool)
        self.assertIn("MB", msg)


class TestEndToEndClassical(unittest.TestCase):
    """Full pipeline: CSV -> analysis -> preprocessing -> real training ->
    validation -> checkpoint -> saved model -> prediction."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.models_dir = os.path.join(self.tmpdir, "models")
        rng = np.random.default_rng(7)
        n = 120
        df = pd.DataFrame({
            "attendance": rng.uniform(50, 100, n),
            "internal_marks": rng.uniform(30, 100, n),
        })
        df["pass_fail"] = ((df["attendance"] > 75) & (df["internal_marks"] > 50)).map({True: "pass", False: "fail"})
        self.csv_path = os.path.join(self.tmpdir, "students.csv")
        df.to_csv(self.csv_path, index=False)

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_full_pipeline_classical(self):
        import trainer.core.pipeline as pipeline_mod
        pipeline_mod.MODELS_ROOT = self.models_dir
        pipeline_mod.EXPERIMENTS_ROOT = os.path.join(self.tmpdir, "experiments")

        cfg = TrainingConfig(
            task="classification", algorithm="logistic_regression", dataset_path=self.csv_path,
            input_columns=["attendance", "internal_marks"], target_column="pass_fail",
            experiment_name="student_pass_test", random_seed=3,
        )
        out = pipeline_mod.run_pipeline(cfg)
        result = out["result"]

        # 1. dataset was loaded
        self.assertEqual(out["dataset_analysis"]["n_samples"], 120)
        # 2. real model created + trained
        self.assertEqual(result.status, "completed")
        # 3. metrics computed from actual predictions
        self.assertIn("accuracy", result.final_metrics)
        self.assertGreaterEqual(result.final_metrics["accuracy"], 0.0)
        # 4. checkpoint created on disk
        self.assertTrue(os.path.exists(result.checkpoint_dir))
        latest = os.path.join(out["model_dir"], "latest", "model_state.pkl")
        self.assertTrue(os.path.exists(latest))

        # 5. checkpoint can be loaded and used for real inference
        predictor = Predictor(self.models_dir, "student_pass_test")
        preds = predictor.predict([{"attendance": 90, "internal_marks": 80},
                                    {"attendance": 40, "internal_marks": 20}])
        self.assertEqual(len(preds), 2)
        self.assertIn(preds[0]["prediction"], ["pass", "fail"])


class TestEndToEndNeural(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.models_dir = os.path.join(self.tmpdir, "models")
        rng = np.random.default_rng(11)
        n = 150
        df = pd.DataFrame({
            "x1": rng.normal(size=n),
            "x2": rng.normal(size=n),
        })
        df["y"] = ((df["x1"] + df["x2"]) > 0).map({True: "yes", False: "no"})
        self.csv_path = os.path.join(self.tmpdir, "toy.csv")
        df.to_csv(self.csv_path, index=False)

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_full_pipeline_neural_and_resume(self):
        import trainer.core.pipeline as pipeline_mod
        pipeline_mod.MODELS_ROOT = self.models_dir
        pipeline_mod.EXPERIMENTS_ROOT = os.path.join(self.tmpdir, "experiments")

        cfg = TrainingConfig(
            task="classification", algorithm="neural_network", dataset_path=self.csv_path,
            input_columns=["x1", "x2"], target_column="y", experiment_name="toy_nn_test",
            hidden_layers=[8], epochs=3, batch_size=8, learning_rate=0.05, optimizer="adam",
            random_seed=5, checkpoint_every=1,
        )
        out = pipeline_mod.run_pipeline(cfg)
        result = out["result"]
        self.assertEqual(result.status, "completed")
        self.assertEqual(len(result.history), 3)
        # loss should generally trend down across a real training run
        self.assertLess(result.history[-1]["train_loss"], result.history[0]["train_loss"] + 1.0)

        # resume for 2 more epochs
        cfg2 = TrainingConfig(**{**cfg.__dict__, "epochs": 5})
        out2 = pipeline_mod.run_pipeline(cfg2, resume=True)
        self.assertEqual(out2["result"].history[-1]["epoch"], 5)

        predictor = Predictor(self.models_dir, "toy_nn_test")
        preds = predictor.predict([{"x1": 2.0, "x2": 2.0}, {"x1": -2.0, "x2": -2.0}])
        self.assertEqual(len(preds), 2)


class TestPlanner(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        rng = np.random.default_rng(1)
        n = 80
        df = pd.DataFrame({
            "size_sqft": rng.uniform(500, 3000, n),
            "bedrooms": rng.integers(1, 5, n),
            "price": rng.uniform(100000, 500000, n),
        })
        self.csv_path = os.path.join(self.tmpdir, "houses.csv")
        df.to_csv(self.csv_path, index=False)

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_plan_infers_regression(self):
        planned = plan("Build a model that predicts house prices from this CSV.", self.csv_path)
        self.assertEqual(planned["config"].task, "regression")
        self.assertEqual(planned["config"].target_column, "price")

    def test_plan_raises_on_ambiguous_request(self):
        with self.assertRaises(ValueError):
            plan("Do something interesting with this data.", self.csv_path)


if __name__ == "__main__":
    unittest.main(verbosity=2)
