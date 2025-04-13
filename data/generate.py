import pandas as pd
import random
from faker import Faker
from datetime import datetime, timedelta


"""
A small python script that generates a large amount of fake data CVSs for the db
"""

fake = Faker()

# Generate Authors
num_authors = 8000
authors = []
for i in range(1, num_authors + 1):
    authors.append({
        "Author_id (P)": i,
        "Name": fake.name(),
        "Country": fake.country()
    })
authors_df = pd.DataFrame(authors)

# Generate Books
num_books = 100000
books = []
for i in range(1, num_books + 1):
    books.append({
        "Book_id (P)": i + 100,
        "Title": fake.sentence(nb_words=14).rstrip('.'),
        "Author_id": random.randint(1, num_authors),
        "Genre": random.choice(["Fiction", "Dystopian", "Magical Realism", "Fantasy", "Sci-Fi", "Non-fiction", "Mystery"]),
        "Published_Year": random.randint(1900, 2023)
    })
books_df = pd.DataFrame(books)

# Generate Sales
num_sales = 1000000
sales = []
start_date = datetime(2022, 1, 1)
for i in range(1, num_sales + 1):
    sales.append({
        "Sale_id (P)": i + 10000,
        "Book_id": random.randint(101, 100 + num_books),
        "Sale_Date": (start_date + timedelta(days=random.randint(0, 730))).strftime('%Y-%m-%d'),
        "Quantity": random.randint(1, 10),
        "Store_Location": fake.city()
    })
sales_df = pd.DataFrame(sales)

# Save CSVs
authors_path = "Authors.csv"
books_path = "Books.csv"
sales_path = "Sales.csv"

authors_df.to_csv(authors_path, index=False)
books_df.to_csv(books_path, index=False)
sales_df.to_csv(sales_path, index=False)

if __name__ == "__main__":
    pass