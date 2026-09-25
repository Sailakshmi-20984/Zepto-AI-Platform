# Zepto Data, Analytics & AI Platform (Capstone Project)

This repository contains the end-to-end data engineering, exploratory data analysis, predictive modeling, and generative AI support assistant pipeline built for Zepto.

---

## 📁 Repository Structure

```text
zepto-data-ai-platform/
├── data_pipeline/         # Module 1: Scraping, Cleaning, & SQLite Relational Pipeline
├── analytics/              # Module 2: EDA, Machine Learning Models, and Joblib Pipeline
├── support_assistant/      # Module 3: RAG Assistant, ChromaDB, LangGraph, and FastAPI Service
└── README.md               # Master Documentation
```

---

# 🚀 Module 1 — Data Pipeline (`/data_pipeline`)

## Overview & Data Source

* **Source:** Scraped from `books.toscrape.com`, capturing books across multiple categories to build a robust relational database.
* **Currency Conversion Rate:** Uses the required project-defined fixed baseline rate: **1 GBP = 105.50 INR**.

## Pipeline Execution Steps

### Scraping

Extracts the following information:

* Book title
* Raw price in GBP
* Star rating
* Availability
* Category

### Cleaning & Transformation

* Removes the currency symbol and creates the `price_gbp` float column.
* Maps text ratings (One–Five) to integers (1–5).
* Converts availability text into the `in_stock` Boolean column.
* Calculates `price_inr` using the fixed 105.50 multiplier.

### Relational Database Storage

The processed data is stored in an SQLite database named:

```text
zepto_data.db
```

The database uses a normalized two-table structure:

* `categories`
* `books`

A Primary Key / Foreign Key relationship connects the two tables.

---

# 📊 Module 2 — Analytics Pipeline (`/analytics`)

## 1. Profiling, Cleaning & Offline Fallback

### Dataset

The Titanic dataset is loaded using Seaborn:

```python
sns.load_dataset('titanic')
```

The dataset is then saved as an offline fallback CSV:

```text
analytics/titanic.csv
```

### Missing Value Strategy

* **`deck`:** Dropped because more than 30% of its values are missing.
* **`age`:** Imputed using median age grouped by class and gender.
* **`embarked` / `embark_town`:** Rows with missing values are removed because the missing percentage is below the 5% threshold.

---

## 2. EDA & Statistical Insights

### Outlier Detection

Outliers are identified using the IQR rule:

```text
Lower Bound = Q1 - 1.5 × IQR
Upper Bound = Q3 + 1.5 × IQR
```

Outlier counts for `age` and `fare` are calculated and logged in the analysis notebooks.

### Fare Skewness

The fare distribution is right-skewed, confirmed using the relationship:

```text
Mean > Median > Mode
```

### Bivariate Survival Analysis

Survival patterns are analyzed across:

* Sex
* Passenger class (`pclass`)
* Combined sex and passenger class

### Correlation Analysis

Correlation analysis is performed using six core numeric variables:

* `survived`
* `pclass`
* `age`
* `sibsp`
* `parch`
* `fare`

Redundant variables such as `adult_male` and `alone` are excluded.

The two strongest absolute off-diagonal correlations are identified and interpreted.

---

## 3. Predictive Modeling & Evaluation

### Train/Test Split

A stratified train/test split is used to preserve the class distribution between training and testing datasets.

### Leakage-Free Preprocessing

Preprocessing is implemented using:

* `Pipeline`
* `ColumnTransformer`

The preprocessing steps are fitted only on the training data to prevent data leakage.

### Classification Models

The following classifiers are evaluated:

1. Logistic Regression
2. Decision Tree
3. Random Forest

The Decision Tree is also visualized using `plot_tree`.

### Imbalance Handling

Model performance is compared using:

* Baseline training
* `class_weight='balanced'`
* SMOTE oversampling applied only to the training data

### Hyperparameter Tuning

Random Forest is tuned using `GridSearchCV`.

Out-of-bag evaluation is also enabled using:

```python
oob_score=True
```

### Regression Side Task

Multivariate Linear Regression is used to predict passenger fare.

---

## Model Comparison

| Model Type     | Model Name / Variant                  | Accuracy / R² | Precision | Recall | F1-Score |  AUC / RMSE |
| -------------- | ------------------------------------- | ------------: | --------: | -----: | -------: | ----------: |
| Classification | Logistic Regression                   |         ~0.81 |     ~0.78 |  ~0.74 |    ~0.76 |       ~0.85 |
| Classification | Decision Tree                         |         ~0.79 |     ~0.75 |  ~0.72 |    ~0.73 |       ~0.81 |
| Classification | Random Forest (Tuned)                 |         ~0.83 |     ~0.81 |  ~0.77 |    ~0.79 |       ~0.88 |
| Regression     | Multivariate Linear Regression (fare) |     R²: ~0.42 |         — |      — |        — | RMSE: ~35.2 |

### Final Recommendation

The **Tuned Random Forest classifier** is selected as the final classification model because it provides the best overall balance of precision, recall, F1-score, and AUC.

It achieves approximately:

* **Accuracy:** 0.83
* **Precision:** 0.81
* **Recall:** 0.77
* **F1-Score:** 0.79
* **AUC:** 0.88

### Model Persistence

The final fitted pipeline is saved using Joblib:

```python
joblib.dump(full_pipeline, "analytics/titanic_pipeline.joblib")
```

---

# 🤖 Module 3 — Support Assistant (`/support_assistant`)

## RAG Pipeline Architecture

The support assistant uses a Retrieval-Augmented Generation (RAG) architecture to answer questions using a collection of local policy documents.

### 1. Ingestion

The system reads eight policy text files:

```text
doc_01.txt
doc_02.txt
doc_03.txt
doc_04.txt
doc_05.txt
doc_06.txt
doc_07.txt
doc_08.txt
```

The documents cover topics such as:

* Delivery
* Returns
* Membership
* Order tracking
* Cancellations
* Items
* Gift cards
* Support hours

### 2. Embedding

Local vector embeddings are generated using Sentence Transformers with:

```text
all-MiniLM-L6-v2
```

### 3. Vector Storage

The generated embeddings are stored locally in a persistent ChromaDB collection:

```text
zepto_policies
```

### 4. Retrieval & Routing

LangGraph `StateGraph` is used to route incoming queries.

The workflow follows:

```text
User Query
    ↓
Intent Classification
    ↓
Retrieve & Answer / Direct Answer
```

### 5. Generation

The assistant supports an offline deterministic mock generation mode using:

```text
MOCK_LLM=1
```

This mode validates the generated response using Pydantic JSON output validation.

Setting:

```text
MOCK_LLM=0
```

enables the real LLM extension implemented in the project.

---

# 🔌 Example API Request & Response

The FastAPI service exposes the following endpoint:

```text
POST /ask
```

### Request

```json
{
  "query": "What is the return policy?"
}
```

### Response

```json
{
  "answer": "Based on the retrieved context: Grocery and perishable items may be reported for a return within 24 hours of delivery if damaged, spoiled, or incorrect; non-perishable packaged items may be returned within 7 days of delivery in applicable cases.",
  "sources": [
    "doc_02.txt",
    "doc_01.txt",
    "doc_03.txt"
  ],
  "confidence": 1.0
}
```

---

# 🐳 Docker Containerization

A local Dockerfile is included inside:

```text
/support_assistant
```

### Build the Docker Image

```powershell
docker build -t zepto-support-assistant support_assistant/
```

### Run the Container

```powershell
docker run -p 7860:7860 zepto-support-assistant
```

The FastAPI service can then be accessed locally through the configured application port.

---

# 🛠️ Technologies Used

### Data Engineering

* Python
* Requests
* BeautifulSoup
* SQLite
* Pandas

### Data Analysis & Machine Learning

* Pandas
* NumPy
* Seaborn
* Matplotlib
* Scikit-learn
* Imbalanced-learn
* Joblib

### Generative AI & RAG

* Sentence Transformers
* ChromaDB
* LangGraph
* Pydantic
* FastAPI

### Deployment

* Docker

---

# 📌 Project Summary

This project demonstrates an end-to-end AI/ML workflow covering:

1. Web scraping and data collection
2. Data cleaning and transformation
3. Relational database design
4. Exploratory data analysis
5. Statistical analysis
6. Machine learning model development
7. Model evaluation and tuning
8. Model persistence
9. Retrieval-Augmented Generation
10. Local vector database management
11. API development using FastAPI
12. Docker containerization

The project combines **data engineering, analytics, machine learning, and generative AI** into a single end-to-end platform.

---

## ✨ UI Refresh (Lightweight)

The support assistant now includes a lightweight browser interface while retaining the original FastAPI `/ask` API. The refresh adds quick-topic cards, a conversation panel, local session history, a light/dark theme toggle, character count, clearer service status, source/confidence metadata, responsive mobile styling, and a small policy-library shortcut.

Run the assistant from `support_assistant/` as before and open `http://localhost:7860/` in a browser.
