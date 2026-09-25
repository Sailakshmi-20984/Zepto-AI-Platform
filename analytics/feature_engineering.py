from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

def get_preprocessor():
    return ColumnTransformer([
        ("num", StandardScaler(), ["distance_km", "item_count", "prep_time_min", "rider_rating"]),
        ("cat", OneHotEncoder(drop="first"), ["traffic_density"])
    ])