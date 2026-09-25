import os, pandas as pd, numpy as np

DATA_PATH = os.path.join(os.path.dirname(__file__), "customer_orders.csv")

def generate_dataset():
    np.random.seed(42)
    n = 1000
    df = pd.DataFrame({
        "order_id": [f"ORD-{i}" for i in range(1000, 1000 + n)],
        "distance_km": np.random.uniform(0.5, 12.0, n),
        "item_count": np.random.randint(1, 15, n),
        "prep_time_min": np.random.uniform(2.0, 15.0, n),
        "rider_rating": np.random.uniform(3.5, 5.0, n),
        "traffic_density": np.random.choice(["Low", "Medium", "High"], n)
    })
    score = (df["distance_km"] * 1.5) + (df["item_count"] * 0.8) + (df["prep_time_min"] * 1.2)
    df["is_late"] = (score > 22.0).astype(int)
    df.to_csv(DATA_PATH, index=False)
    print(f"Generated analytics dataset -> {DATA_PATH}")

if __name__ == "__main__":
    generate_dataset()