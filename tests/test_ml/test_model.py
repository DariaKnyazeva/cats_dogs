import joblib
import mock
import numpy as np
from dask_ml.wrappers import ParallelPostFit
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
import unittest

from src.ml import SKLinearImageModel


class SKLinearImageModelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.num_samples = 100
        cls.X = np.random.random_sample((cls.num_samples, 150, 150, 3))
        cls.y = np.random.random_sample(cls.num_samples)

    @mock.patch('joblib.dump')
    def test_save(self, mock_joblib):
        testable = SKLinearImageModel(pkl_file=None)
        testable.classifier = mock.MagicMock()
        testable.save('1.pkl')

        mock_joblib.assert_called_once_with(testable.classifier, '1.pkl')

    @mock.patch.object(joblib, 'load', return_value=SGDClassifier())
    def test_load(self, mock_joblib):
        testable = SKLinearImageModel(pkl_file=None)
        testable.load('1.pkl')

        mock_joblib.assert_called_once_with('1.pkl')

    @mock.patch.object(ParallelPostFit, 'predict_proba')
    @mock.patch.object(ParallelPostFit, 'fit')
    @mock.patch('src.ml.model.SKLinearImageModel._preprocess_dataset')
    def test_train(self, mock_preprocess_ds, mock_fit, mock_predict_proba):
        X_train, X_test, y_train, y_test = train_test_split(
            self.X,
            self.y,
            test_size=0.2,
            shuffle=True,
            random_state=42,
        )
        X_train_prepared = X_train * 0.2
        mock_preprocess_ds.return_value = X_train_prepared
        mock_fit.return_value = ParallelPostFit()

        testable = SKLinearImageModel(pkl_file=None)
        testable.train(X_train, y_train, X_test, y_test, verbose=False)

        np.testing.assert_array_equal(X_train_prepared,
                                      mock_fit.call_args[0][0])
        np.testing.assert_array_equal(y_train,
                                      mock_fit.call_args[0][1])

        mock_predict_proba.assert_called_once()

    @mock.patch('src.ml.model.SKLinearImageModel._preprocess_dataset')
    @mock.patch.object(ParallelPostFit, 'predict_proba')
    @mock.patch('src.ml.model.SKLinearImageModel.load')
    def test_predict(self, mock_load, mock_predict, mock_preprocess_ds):
        mock_load.return_value = ParallelPostFit()
        mock_preprocess_ds.return_value = [1, 2, 3]

        testable = SKLinearImageModel(pkl_file='trained_models/hog_sklearn.pkl')
        testable.predict(self.X)

        mock_load.assert_called_once()
        mock_predict.assert_called_once_with([1, 2, 3])

    def test_predict_proba_to_label_cat(self):
        testable = SKLinearImageModel(pkl_file=None)
        proba = (0.85, 0.15)
        self.assertEqual('cat', testable._predict_proba_to_label(proba))

    def test_predict_proba_to_label_dog(self):
        testable = SKLinearImageModel(pkl_file=None)
        proba = (0.15, 0.85)
        self.assertEqual('dog', testable._predict_proba_to_label(proba))

    def test_predict_proba_to_label_unknown(self):
        testable = SKLinearImageModel(pkl_file=None)
        proba = (0.45, 0.65)
        self.assertEqual('unknown', testable._predict_proba_to_label(proba))
