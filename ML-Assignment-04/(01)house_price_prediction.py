import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# 1. Load Dataset
df = pd.read_csv("Housing.csv")

print("First 5 rows:")
print(df.head())

# 2. Select Area and Price
X = df[["area"]]
y = df["price"]

# 3. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# 4. Model Initialization & Training
model = LinearRegression()
model.fit(X_train, y_train)

# 5. Prediction
y_pred = model.predict(X_test)

# 6. Metrics Calculation
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nSlope (b1):", model.coef_[0])
print("Intercept (b0):", model.intercept_)

print("\n--- Evaluation Metrics ---")
print(f"MAE  : {mae:.2f}")
print(f"MSE  : {mse:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"R2   : {r2:.4f}")

# 7. Predict price for a new house
area = 1500
predicted_price = model.predict([[area]])

print("\nPredicted price for", area, "sq.ft:",
      predicted_price[0])

# 8. Plotting
plt.figure(figsize=(7, 5))

plt.scatter(X_test, y_test,
            color="blue", label="Actual Data")

plt.plot(X_test, y_pred,
         color="red", linewidth=2, label="Regression Line")

plt.xlabel("House Area (sq.ft)")
plt.ylabel("House Price")
plt.title("House Area vs House Price")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.show()