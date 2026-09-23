import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
df = pd.read_csv("house_price.csv")
print("First 5 rows:")
print(df.head())
X = df[["sqft"]]
y = df["totalprice"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Results")
print("MAE:", mae)
print("MSE:", mse)
print("R2 Score:", r2)
print("\nRegression Equation:")
print("Price =", model.coef_[0], "* Area +", model.intercept_)

area = 1500
predicted_price = model.predict([[area]])

print("\nPredicted price for", area, "sq.ft:",
      predicted_price[0])
plt.scatter(X_test, y_test)
plt.plot(X_test, y_pred)
plt.xlabel("House Area (sq.ft)")
plt.ylabel("House Price")
plt.title("House Price Prediction using Linear Regression")
plt.show()
