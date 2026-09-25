import sqlite3
import json
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "zepto_data.db")
JSON_PATH = os.path.join(os.path.dirname(__file__), "raw_scraped_data.json")

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

cursor.executescript("""
DROP TABLE IF EXISTS books;
DROP TABLE IF EXISTS categories;

CREATE TABLE categories (
    category_id INTEGER PRIMARY KEY AUTOINCREMENT,
    category_name TEXT UNIQUE NOT NULL
);

CREATE TABLE books (
    book_id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    price_gbp REAL NOT NULL,
    price_inr REAL NOT NULL,
    rating INTEGER NOT NULL,
    in_stock INTEGER NOT NULL,
    category_id INTEGER,
    FOREIGN KEY (category_id) REFERENCES categories(category_id)
);
""")

with open(JSON_PATH, "r") as f:
    books_data = json.load(f)

for item in books_data:
    cursor.execute("INSERT OR IGNORE INTO categories (category_name) VALUES (?)", (item["category"],))
    cursor.execute("SELECT category_id FROM categories WHERE category_name = ?", (item["category"],))
    cat_id = cursor.fetchone()[0]
    
    cursor.execute("""
        INSERT INTO books (title, price_gbp, price_inr, rating, in_stock, category_id)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (item["title"], item["price_gbp"], item["price_inr"], item["rating"], item["in_stock"], cat_id))

conn.commit()
conn.close()
print("Database initialized and populated successfully.")