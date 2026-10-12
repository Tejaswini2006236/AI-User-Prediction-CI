
import unittest

from ai_prediction import (
    load_data,
    preprocess_data,
    train_model,
    predict_ai_user
)


class TestAIPrediction(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        cls.df = load_data(
            "AI_User_Prediction_Dataset_600.csv"
        )

        cls.X, cls.y = preprocess_data(cls.df)

        cls.model, cls.accuracy = train_model(
            cls.X, cls.y
        )

    def test_dataset_loading(self):

        self.assertEqual(len(self.df), 600)

    def test_target_column(self):

        self.assertNotIn("AI_User", self.X.columns)

        self.assertNotIn("User_ID", self.X.columns)

    def test_model_accuracy(self):

        self.assertGreaterEqual(self.accuracy, 0.0)

        self.assertLessEqual(self.accuracy, 1.0)

    def test_prediction_output(self):

        sample = self.X.iloc[[0]]

        prediction = predict_ai_user(
            self.model,
            sample
        )

        self.assertIn(prediction, [0, 1])


if __name__ == "__main__":

    unittest.main()
