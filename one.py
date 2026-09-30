import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv("boston_houses.csv")

X = df[['RM']]
y = df['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)
ridge_pred = ridge.predict(X_test)

lasso = Lasso(alpha=0.1)
lasso.fit(X_train, y_train)
lasso_pred = lasso.predict(X_test)

print("Ridge Regression MSE:", mean_squared_error(y_test, ridge_pred))
print("Ridge Regression R2:", r2_score(y_test, ridge_pred))
print("Lasso Regression MSE:", mean_squared_error(y_test, lasso_pred))
print("Lasso Regression R2:", r2_score(y_test, lasso_pred))

sample_house = pd.DataFrame({'RM': [5.0]})
price_ridge = ridge.predict(sample_house)[0]
price_lasso = lasso.predict(sample_house)[0]

print(f"\nPredicted Price for a 5-room house using Ridge: {price_ridge:.2f}")
print(f"Predicted Price for a 5-room house using Lasso: {price_lasso:.2f}")
print(f"Comparison Difference: {abs(price_ridge - price_lasso):.4f}")
