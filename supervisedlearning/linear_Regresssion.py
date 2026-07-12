import pandas as pd
from sklearn.linear_model import LinearRegression

# Load CSV file
data = pd.read_csv("data.csv")

# Input (Feature)
X = data[["X"]]

# Output (Target)
y = data["Y"]

# Create model
model = LinearRegression()

# Train model
model.fit(X, y)

# Predict for X = 6
prediction = model.predict([[6]])

print("Prediction:", prediction[0])