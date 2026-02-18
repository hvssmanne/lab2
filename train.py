import os
import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score

os.makedirs("outputs", exist_ok=True)

red = pd.read_csv("dataset/wine+quality/winequality-red.csv", sep=';')
white = pd.read_csv("dataset/wine+quality/winequality-white.csv", sep=';')

red["type"] = 0
white["type"] = 1

data = pd.concat([red, white], ignore_index=True)

X = data.drop("quality", axis=1)
y = data["quality"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Ridge model
model = Ridge(alpha=0.1)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Evaluation Metrics")
print("-------------------")
print(f"MSE: {mse:.4f}")
print(f"R2 Score: {r2:.4f}")

joblib.dump(model, "outputs/model.pkl")

results = {
    "model": "Ridge",
    "alpha": 0.1,
    "test_size": 0.2,
    "MSE": mse,
    "R2": r2
}

with open("outputs/results.json", "w") as f:
    json.dump(results, f, indent=4)
