import sqlite3
import pandas as pd
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "zepto_data.db")
conn = sqlite3.connect(DB_PATH)

print("--- Query 1: SELECT / WHERE ---")
q1 = "SELECT title, price_inr FROM books WHERE rating = 5 LIMIT 3;"
print(pd.read_sql(q1, conn))

print("\n--- Query 2: ORDER BY / LIMIT ---")
q2 = "SELECT title, price_gbp FROM books ORDER BY price_gbp DESC LIMIT 3;"
print(pd.read_sql(q2, conn))

print("\n--- Query 3: DISTINCT ---")
q3 = "SELECT DISTINCT rating FROM books ORDER BY rating ASC;"
print(pd.read_sql(q3, conn))

print("\n--- Query 4: BETWEEN ---")
q4 = "SELECT title, price_inr FROM books WHERE price_inr BETWEEN 2000 AND 4000 LIMIT 3;"
print(pd.read_sql(q4, conn))

print("\n--- Query 5: SQL JOIN (10 highest-rated books per category) ---")
q5 = """
SELECT b.title, c.category_name, b.rating, b.price_inr 
FROM books b 
JOIN categories c ON b.category_id = c.category_id 
ORDER BY b.rating DESC, b.price_inr DESC 
LIMIT 10;
"""
sql_join_df = pd.read_sql(q5, conn)
print(sql_join_df)

print("\n--- Pandas Equivalent (pd.merge) Verification ---")
books_df = pd.read_sql("SELECT * FROM books", conn)
categories_df = pd.read_sql("SELECT * FROM categories", conn)

merged_df = pd.merge(books_df, categories_df, on="category_id")
merged_df = merged_df[["title", "category_name", "rating", "price_inr"]]
pd_merge_df = merged_df.sort_values(by=["rating", "price_inr"], ascending=[False, False]).head(10).reset_index(drop=True)

print(pd_merge_df)

print("\n--- Match Verification ---")
print("Are outputs identical?:", sql_join_df.equals(pd_merge_df))

conn.close()