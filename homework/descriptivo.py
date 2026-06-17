import pandas as pd  #  type: ignore

df = pd.read_csv("../files/input/house_data.csv")

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

target = df[["price"]]



# %%
features.bathrooms.value_counts().sort_index().plot(kind="bar")