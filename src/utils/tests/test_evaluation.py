import unittest
import numpy as np

from src.utils import evaluation

class EvaluationTest(unittest.TestCase):

    def test_accuracy(self):
        # 100% accuracy
        y_pred = np.asarray([1, 0, 0, 0, 1, 1, 0, 1])
        y_true= np.asarray([1, 0, 0, 0, 1, 1, 0, 1])
        self.assertEqual(evaluation.accuracy(y_pred, y_true), 1)
        # 50% accuracy
        y_pred = np.asarray([1, 0, 0, 0, 1, 1, 0, 1])
        y_true = np.asarray([1, 0, 1, 0, 0, 1, 1, 0])
        self.assertEqual(evaluation.accuracy(y_pred, y_true), 0.5)
        # 0% accurancy
        y_pred = np.asarray([1, 0, 0, 0, 1, 1, 0, 1])
        y_true = np.asarray([0, 1, 1, 1, 0, 0, 1, 0])
        self.assertEqual(evaluation.accuracy(y_pred, y_true), 0)

if __name__ == '__main__':
    unittest.main()
