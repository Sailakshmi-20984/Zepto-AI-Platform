import os, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from feature_engineering import get_preprocessor

DATA_PATH = os.path.join(os.path.dirname(__file__), "customer_orders.csv")

def train_and_evaluate():
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=["order_id", "is_late"])
    y = df["is_late"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    preprocessor = get_preprocessor()
    
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_proc, y_train)
    
    preds = model.predict(X_test_proc)
    print("\n--- Model Evaluation ---")
    print(classification_report(y_test, preds))

if __name__ == "__main__":
    train_and_evaluate()