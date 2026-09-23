import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
df = pd.read_csv("house_price.csv")
X = df[["sqft"]]
y = df["totalprice"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
linear = LinearRegression()
linear.fit(X_train, y_train)
linear_pred = linear.predict(X_test)
poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

polynomial = LinearRegression()
polynomial.fit(X_train_poly, y_train)
poly_pred = polynomial.predict(X_test_poly)

print("Linear Regression R2:", r2_score(y_test, linear_pred))
print("Polynomial Regression R2:", r2_score(y_test, poly_pred))
