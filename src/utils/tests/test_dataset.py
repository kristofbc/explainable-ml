import unittest
import numpy as np
import sys

from src.utils import dataset

class DatasetTest(unittest.TestCase):

    """
        Test int value for split_dataset
    """
    def test_split_dataset_int(self):
        data = np.random.randint(0, 255, (20, 10))
        # Even split
        x, y = dataset.split_dataset(data, 10)
        self.assertEqual(x.shape, y.shape)
        self.assertFalse(np.allclose(x, y))
        # Uneven split
        x, y = dataset.split_dataset(data, 15)
        self.assertEqual(x.shape[0], 15)
        self.assertEqual(y.shape[0], data.shape[0]-15)
        self.assertFalse(np.allclose(x[:5], y[:5]))
        # Shuffled split
        x, y = dataset.split_dataset(data, 10, True)
        self.assertFalse(np.allclose(x, data[:10]))
        self.assertFalse(np.allclose(y, data[10:]))

    """
        Test float value for split_dataset
    """
    def test_split_dataset_float(self):
        data = np.random.randint(0, 255, (20, 10))
        # Even split
        x, y = dataset.split_dataset(data, 0.5)
        self.assertEqual(x.shape, y.shape)
        self.assertFalse(np.allclose(x, y))
        # Uneven split
        x, y = dataset.split_dataset(data, 0.95)
        self.assertEqual(x.shape[0], 19)
        self.assertEqual(y.shape[0], 1)
        self.assertFalse(np.allclose(x[:1], y[:1]))
        # Shuffled split
        x, y = dataset.split_dataset(data, 0.5, True)
        self.assertFalse(np.allclose(x, data[:10]))
        self.assertFalse(np.allclose(y, data[10:]))

    def test_split_train_test_dataset_int(self):
        data_x = np.random.randint(0, 255, (20, 10))
        data_y = np.random.randint(0, 255, (20,))
        # Even split
        x_train, y_train, x_test, y_test = dataset.split_train_test_dataset(data_x, data_y, 10)
        self.assertEqual(x_train.shape[0], y_train.shape[0])
        self.assertEqual(x_test.shape[0], y_test.shape[0])
        self.assertEqual(x_train.shape[0], x_test.shape[0])
        self.assertEqual(len(y_train.shape), 1)
        self.assertEqual(len(y_test.shape), 1)
        # Uneven split
        x_train, y_train, x_test, y_test = dataset.split_train_test_dataset(data_x, data_y, 15)
        self.assertEqual(x_train.shape[0], y_train.shape[0])
        self.assertEqual(x_test.shape[0], y_test.shape[0])
        self.assertNotEqual(x_train.shape[0], x_test.shape[0])
        self.assertEqual(len(y_train.shape), 1)
        self.assertEqual(len(y_test.shape), 1)
        # Shuffled split
        x_train, y_train, x_test, y_test = dataset.split_train_test_dataset(data_x, data_y, 10, True)
        self.assertEqual(x_train.shape[0], y_train.shape[0])
        self.assertEqual(x_test.shape[0], y_test.shape[0])
        self.assertEqual(x_train.shape[0], x_test.shape[0])
        self.assertEqual(len(y_train.shape), 1)
        self.assertEqual(len(y_test.shape), 1)


    def test_split_train_test_dataset_float(self):
        data_x = np.random.randint(0, 255, (20, 10))
        data_y = np.random.randint(0, 255, (20,))
        # Even split
        x_train, y_train, x_test, y_test = dataset.split_train_test_dataset(data_x, data_y, 0.5)
        self.assertEqual(x_train.shape[0], y_train.shape[0])
        self.assertEqual(x_test.shape[0], y_test.shape[0])
        self.assertEqual(x_train.shape[0], x_test.shape[0])
        self.assertEqual(len(y_train.shape), 1)
        self.assertEqual(len(y_test.shape), 1)
        # Uneven split
        x_train, y_train, x_test, y_test = dataset.split_train_test_dataset(data_x, data_y, 0.80)
        self.assertEqual(x_train.shape[0], y_train.shape[0])
        self.assertEqual(x_test.shape[0], y_test.shape[0])
        self.assertNotEqual(x_train.shape[0], x_test.shape[0])
        self.assertEqual(len(y_train.shape), 1)
        self.assertEqual(len(y_test.shape), 1)
        # Shuffled split
        x_train, y_train, x_test, y_test = dataset.split_train_test_dataset(data_x, data_y, 0.5, True)
        self.assertEqual(x_train.shape[0], y_train.shape[0])
        self.assertEqual(x_test.shape[0], y_test.shape[0])
        self.assertEqual(x_train.shape[0], x_test.shape[0])
        self.assertEqual(len(y_train.shape), 1)
        self.assertEqual(len(y_test.shape), 1)



if __name__ == '__main__':
        unittest.main()
