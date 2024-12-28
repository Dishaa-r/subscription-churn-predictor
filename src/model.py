from sklearn.tree import DecisionTreeClassifier
import numpy as np

# Mock model for demonstration
def train_mock_model():
    # Training dataset: [age, subscription_duration, interaction_frequency, activity_score]
    X = np.array([
        [25, 12, 8, 0.75],  # Non-churn example
        [40, 2, 1, 0.30],   # Churn example
        [30, 6, 5, 0.50],   # Non-churn example
        [50, 1, 2, 0.20]    # Churn example
    ])
    y = np.array([0, 1, 0, 1])  # 0 = Non-churn, 1 = Churn

    # Train a simple Decision Tree Classifier
    model = DecisionTreeClassifier()
    model.fit(X, y)
    return model

# Load mock model
model = train_mock_model()

def predict_churn(data):
    # Convert input data into a NumPy array for prediction
    input_data = np.array([
        data["age"], 
        data["subscription_duration"], 
        data["interaction_frequency"], 
        data["activity_score"]
    ]).reshape(1, -1)
    
    # Get prediction and probability
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]  # Probability of churn
    
    # Suggest action
    suggestion = "Offer a discount or call the customer." if probability > 0.5 else "Keep monitoring engagement."

    return round(probability * 100, 2), suggestion
