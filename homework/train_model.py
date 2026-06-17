import pickle
import pandas as pd
from sklearn.linear_model import LinearRegression

# Cargar datos
df = pd.read_csv("files/input/house_data.csv")

# Variables predictoras
features = df[
    [
        "bedrooms",
        "bathrooms",
        "sqft_living",
        "sqft_lot",
        "floors",
        "waterfront",
        "condition",
    ]
]

# Variable objetivo
target = df["price"]

# Entrenar modelo
estimator = LinearRegression()
estimator.fit(features, target)

# Guardar modelo
with open("homework/house_predictor.pkl", "wb") as file:
    pickle.dump(estimator, file)

print("Modelo guardado correctamente en homework/house_predictor.pkl")