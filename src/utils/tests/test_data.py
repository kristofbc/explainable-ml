import unittest
import numpy as np
import sys

from src.utils import data

class DataTest(unittest.TestCase):
    
    def test_count_unique_value(self):
        values = np.asarray([1,1,1,2,2,3,4,5,6,6,0])
        np.random.shuffle(values) # Ensure "randomness"
        unique = data.count_unique_value(values)
        self.assertEqual(unique[0], 1)
        self.assertEqual(unique[1], 3)
        self.assertEqual(unique[2], 2)
        self.assertEqual(unique[3], 1)
        self.assertEqual(unique[4], 1)
        self.assertEqual(unique[5], 1)
        self.assertEqual(unique[6], 2)

if __name__ == '__main__':
    unittest.main()
