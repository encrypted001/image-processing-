import unittest

import match
import numpy as np


class GeometryScoringTests(unittest.TestCase):
    def test_insufficient_matches_do_not_estimate_geometry(self):
        result = match.verify_geometry([], [], [])

        self.assertEqual(result, (None, None, 0, 0.0))
        self.assertEqual(match.confidence_score(*result), 0.0)
        self.assertEqual(match.classify_geometry(*result)[1], "DIFFERENT")

    def test_failed_geometric_verification_scores_zero(self):
        score, label = match.classify_geometry(None, None, 0, 0.0)

        self.assertEqual(score, 0.0)
        self.assertEqual(label, "DIFFERENT")

    def test_thresholds_are_configurable(self):
        homography = np.eye(3)
        mask = np.ones(80, dtype=bool)

        score, label = match.classify_geometry(
            homography,
            mask,
            80,
            0.8,
            same_threshold=90,
            similar_threshold=70,
        )

        self.assertEqual(score, 80.0)
        self.assertEqual(label, "SIMILAR")
        self.assertEqual(
            match.classify_confidence(
                10,
                same_threshold=90,
                similar_threshold=70,
                different_threshold=20,
            ),
            "DIFFERENT",
        )


if __name__ == "__main__":
    unittest.main()
