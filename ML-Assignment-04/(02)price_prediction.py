import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset
df = pd.read_csv("house_price.csv")

# Display first 5 rows
print("First 5 rows:")
print(df.head())

# Select only area and price
X = df[["sqft"]]
y = df["totalprice"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create Linear Regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Predict prices
y_pred = model.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Results")
print("MAE:", mae)
print("MSE:", mse)
print("R2 Score:", r2)

# Model equation
print("\nRegression Equation:")
print("Price =", model.coef_[0], "* Area +", model.intercept_)

# Predict price for a new house
area = 1500
predicted_price = model.predict([[area]])

print("\nPredicted price for", area, "sq.ft:",
      predicted_price[0])

# Visualization
plt.scatter(X_test, y_test)
plt.plot(X_test, y_pred)
plt.xlabel("House Area (sq.ft)")
plt.ylabel("House Price")
plt.title("House Price Prediction using Linear Regression")
plt.show()