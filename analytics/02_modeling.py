import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, mean_absolute_error, 
                             mean_squared_error, r2_score)
from imblearn.over_sampling import SMOTE

analytics_dir = os.path.dirname(__file__)
csv_path = os.path.join(analytics_dir, "titanic.csv")
df = pd.read_csv(csv_path)

X = df.drop(columns=["survived", "alive", "deck"], errors="ignore")
y = df["survived"]

# 1. Stratified Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 2. Pipeline Preprocessor (Fit on Train ONLY)
num_features = ["age", "fare", "sibsp", "parch"]
cat_features = ["sex", "embarked", "pclass", "who"]

num_transformer = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
cat_transformer = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("encoder", OneHotEncoder(handle_unknown="ignore"))])

preprocessor = ColumnTransformer([
    ("num", num_transformer, num_features),
    ("cat", cat_transformer, cat_features)
])

# 3. Train 3 Classifiers
classifiers = {
    "Logistic Regression": LogisticRegression(random_state=42),
    "Decision Tree": DecisionTreeClassifier(max_depth=4, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, oob_score=True, random_state=42)
}

results = []
for name, clf in classifiers.items():
    pipe = Pipeline([("preprocessor", preprocessor), ("clf", clf)])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    y_prob = pipe.predict_proba(X_test)[:, 1]
    
    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred),
        "AUC": roc_auc_score(y_test, y_prob)
    })

print("\n--- Classifier Metrics ---")
print(pd.DataFrame(results))

# 4. Decision Tree Plot
dt_pipe = Pipeline([("preprocessor", preprocessor), ("clf", DecisionTreeClassifier(max_depth=3, random_state=42))])
dt_pipe.fit(X_train, y_train)
plt.figure(figsize=(12, 6))
plot_tree(dt_pipe.named_steps["clf"], filled=True, class_names=["Died", "Survived"])
plt.savefig(os.path.join(analytics_dir, "decision_tree.png"))
plt.close()

# 5. Imbalance Handling & Hyperparameter Tuning
param_grid = {
    "clf__n_estimators": [50, 100],
    "clf__max_depth": [5, 10, None],
    "clf__max_features": ["sqrt", "log2"]
}
rf_oob = Pipeline([("preprocessor", preprocessor), ("clf", RandomForestClassifier(oob_score=True, random_state=42))])
grid = GridSearchCV(rf_oob, param_grid, cv=5, scoring="f1").fit(X_train, y_train)
best_rf = grid.best_estimator_

print(f"\nBest RF Params: {grid.best_params_}")
print(f"RF OOB Score: {best_rf.named_steps['clf'].oob_score_:.4f}")

# 6. Linear Regression Side-Task (Predict Fare)
X_reg = df.drop(columns=["fare", "survived", "alive", "deck"], errors="ignore")
y_reg = df["fare"].fillna(df["fare"].median())
X_tr_r, X_te_r, y_tr_r, y_te_r = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

reg_preprocessor = ColumnTransformer([
    ("num", num_transformer, ["age", "sibsp", "parch"]),
    ("cat", cat_transformer, ["sex", "embarked", "pclass"])
])

reg_pipe = Pipeline([("preprocessor", reg_preprocessor), ("reg", LinearRegression())])
reg_pipe.fit(X_tr_r, y_tr_r)
y_reg_pred = reg_pipe.predict(X_te_r)

mae = mean_absolute_error(y_te_r, y_reg_pred)
rmse = np.sqrt(mean_squared_error(y_te_r, y_reg_pred))
r2 = r2_score(y_te_r, y_reg_pred)
adj_r2 = 1 - (1 - r2) * (len(y_te_r) - 1) / (len(y_te_r) - X_te_r.shape[1] - 1)

print(f"\nFare Regression Metrics: MAE={mae:.2f}, RMSE={rmse:.2f}, R2={r2:.2f}, Adj R2={adj_r2:.2f}")

# 7. Export Fitted End-to-End Pipeline Object
pipeline_path = os.path.join(analytics_dir, "titanic_pipeline.joblib")
joblib.dump(best_rf, pipeline_path)

# Verify reload
loaded_pipe = joblib.load(pipeline_path)
print(f"\nReload Verification Output: {loaded_pipe.predict(X_test.iloc[:2])}")