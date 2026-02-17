import unittest

from scripts.analise_credito import compute_metrics


class TestMetrics(unittest.TestCase):
    def test_compute_metrics(self):
        y_true = [1, 0, 1, 0]
        y_pred = [1, 0, 0, 0]

        m = compute_metrics(y_true, y_pred)

        self.assertEqual((m.tp, m.tn, m.fp, m.fn), (1, 2, 0, 1))
        self.assertAlmostEqual(m.accuracy, 0.75)
        self.assertAlmostEqual(m.precision, 1.0)
        self.assertAlmostEqual(m.recall, 0.5)
        self.assertAlmostEqual(round(m.f1, 3), 0.667)


if __name__ == "__main__":
    unittest.main()
