import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from maintenance.data import (COLUMNS, FEATURES, add_training_target,
                              last_test_observations, read_engines, split_by_engine)
from maintenance.train import choose_threshold, metrics, run


def fixture(n_engines=8, cycles=8):
    return pd.DataFrame([[engine, cycle, *([float(cycle)] * 24)]
                         for engine in range(1, n_engines + 1)
                         for cycle in range(1, cycles + 1)], columns=COLUMNS)


class PipelineTests(unittest.TestCase):
    def test_split_keeps_entire_engines_separate(self):
        frame = add_training_target(fixture(), horizon=2)
        train, validation = split_by_engine(frame)
        self.assertFalse(set(train.engine_id) & set(validation.engine_id))
        self.assertEqual(len(train) + len(validation), len(frame))
        self.assertTrue((train.groupby("engine_id").size() == 8).all())

    def test_target_boundary_and_no_feature_leakage(self):
        frame = add_training_target(fixture(1, 5), horizon=2)
        self.assertEqual(frame.rul.tolist(), [4, 3, 2, 1, 0])
        self.assertEqual(frame.failure_within_horizon.tolist(), [0, 0, 1, 1, 1])
        self.assertFalse({"rul", "failure_within_horizon", "engine_id"} & set(FEATURES))

    def test_bad_horizon_rejected(self):
        for horizon in [0, -1, 1.5]:
            with self.assertRaises(ValueError):
                add_training_target(fixture(), horizon)

    def test_bad_sensor_inputs_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.txt"
            invalid_frames = [fixture().iloc[:, :-1],
                              pd.concat([fixture(), fixture().iloc[:1]])]
            invalid = fixture()
            invalid.loc[0, "sensor1"] = np.inf
            invalid_frames.append(invalid)
            invalid = fixture()
            invalid.loc[0, "cycle"] = -1
            invalid_frames.append(invalid)
            for frame in invalid_frames:
                frame.to_csv(path, sep=" ", index=False, header=False)
                with self.assertRaises(ValueError):
                    read_engines(path)

    def test_reader_sorts_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.txt"
            fixture().iloc[::-1].to_csv(path, sep=" ", index=False, header=False)
            pd.testing.assert_frame_equal(read_engines(path), fixture())

    def test_truth_aligns_with_last_cycle_not_row_order(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "truth.txt"
            path.write_text("2\n40\n")
            frame = last_test_observations(fixture(2, 5).iloc[::-1], path, horizon=2)
            self.assertEqual(frame.engine_id.tolist(), [1, 2])
            self.assertEqual(frame.cycle.tolist(), [5, 5])
            self.assertEqual(frame.failure_within_horizon.tolist(), [1, 0])
            path.write_text("2\n")
            with self.assertRaises(ValueError):
                last_test_observations(fixture(2, 5), path)
            path.write_text("2\n40\n")
            wrong_ids = fixture(2, 5)
            wrong_ids["engine_id"] += 1
            with self.assertRaises(ValueError):
                last_test_observations(wrong_ids, path)

    def test_threshold_tie_uses_largest_value(self):
        self.assertEqual(choose_threshold([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]), 0.8)
        with self.assertRaises(ValueError):
            choose_threshold([0, 0], [0.2, 0.3])

    def test_confusion_matrix_order(self):
        result = metrics([0, 0, 1, 1], [0.1, 0.8, 0.2, 0.9], 0.5)
        self.assertEqual(result["confusion_matrix_tn_fp_fn_tp"], [1, 1, 1, 1])
        self.assertEqual(result["recall"], 0.5)
        self.assertIsNone(metrics([0, 0], [0.1, 0.2], 0.5)["roc_auc"])

    def test_end_to_end_reproducible_and_test_labels_do_not_tune_threshold(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture().to_csv(root / "PM_train.txt", sep=" ", index=False, header=False)
            fixture(2, 4).to_csv(root / "PM_test.txt", sep=" ", index=False, header=False)
            (root / "PM_truth.txt").write_text("1\n8\n")
            first = run(root, root / "report1", horizon=2, trees=5)
            second = run(root, root / "report2", horizon=2, trees=5)
            self.assertEqual(first, second)
            predictions = pd.read_csv(root / "report1/test_predictions.csv")
            self.assertEqual(len(predictions), 2)
            self.assertEqual(len(first["data_sha256"]), 3)
            (root / "PM_truth.txt").write_text("8\n1\n")
            changed = run(root, root / "report3", horizon=2, trees=5)
            self.assertEqual(first["threshold"], changed["threshold"])
            updated = pd.read_csv(root / "report3/test_predictions.csv")
            np.testing.assert_array_equal(predictions.risk_score, updated.risk_score)


if __name__ == "__main__":
    unittest.main()
