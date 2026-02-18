import os
import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

os.makedirs("outputs", exist_ok=True)

red = pd.read_csv("dataset/wine+quality/winequality-red.csv", sep=';')

white = pd.read_csv("dataset/wine+quality/winequality-white.csv", sep=';')

red["type"] = 0
white["type"] = 1

data = pd.concat([red, white], ignore_index=True)
X = data.drop("quality", axis=1)
y = data["quality"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

model = LinearRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MSE: {mse}")
print(f"R2: {r2}")

joblib.dump(model, "outputs/model.pkl")

results = {
    "MSE": mse,
    "R2": r2
}

with open("outputs/results.json", "w") as f:
    json.dump(results, f, indent=4)
