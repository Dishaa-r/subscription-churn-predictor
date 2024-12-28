import unittest
from model import predict_churn

class TestChurnModel(unittest.TestCase):
    def test_predict_churn_high_risk(self):
        data = {
            "age": 45,
            "subscription_duration": 1,
            "interaction_frequency": 1,
            "activity_score": 0.2
        }
        churn_probability, suggestion = predict_churn(data)
        self.assertGreater(churn_probability, 50)
        self.assertEqual(suggestion, "Offer a discount or call the customer.")

if __name__ == "__main__":
    unittest.main()
