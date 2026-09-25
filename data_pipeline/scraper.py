import requests
from bs4 import BeautifulSoup
import json
import os

BASE_URL = "http://books.toscrape.com/catalogue/category/books_1/index.html"
RATING_MAP = {"One": 1, "Two": 2, "Three": 3, "Four": 4, "Five": 5}
GBP_TO_INR = 105.50

categories = [
    "http://books.toscrape.com/catalogue/category/books/travel_2/index.html",
    "http://books.toscrape.com/catalogue/category/books/mystery_3/index.html",
    "http://books.toscrape.com/catalogue/category/books/historical-fiction_4/index.html"
]

all_books = []

for cat_url in categories:
    res = requests.get(cat_url)
    soup = BeautifulSoup(res.content, "html.parser")
    cat_name = soup.find("h1").text.strip()
    
    products = soup.find_all("article", class_="product_pod")
    for p in products:
        title = p.find("h3").find("a")["title"]
        price_str = p.find("p", class_="price_color").text.replace("£", "").strip()
        price_gbp = float(price_str)
        
        rating_class = p.find("p", class_="star-rating")["class"][1]
        rating = RATING_MAP.get(rating_class, 3)
        
        availability = p.find("p", class_="instock availability").text.strip()
        in_stock = "In stock" in availability
        
        all_books.append({
            "title": title,
            "category": cat_name,
            "price_gbp": price_gbp,
            "price_inr": round(price_gbp * GBP_TO_INR, 2),
            "rating": rating,
            "in_stock": 1 if in_stock else 0
        })

out_path = os.path.join(os.path.dirname(__file__), "raw_scraped_data.json")
with open(out_path, "w") as f:
    json.dump(all_books, f, indent=4)

print(f"Scraped {len(all_books)} books across {len(categories)} categories.")