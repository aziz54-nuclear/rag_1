"""
database.py
-----------
Creates and populates an SQLite database with 30 e-commerce products.
Each product is stored as: id, title, description, manufacturer, price, category, color.

This is the "vector database" layer. In production you would swap SQLite
for Chroma / Pinecone / pgvector and store pre-computed embeddings.
Here we store the raw text fields and compute embeddings at search time.
"""

import sqlite3
import json
import os

DB_PATH = "products.db"

PRODUCTS = [
    (1,  "MacBook Pro 14 M3 Pro",          "Apple M3 Pro chip 18GB RAM 512GB SSD Liquid Retina XDR display excellent battery life professional performance",        "Apple",     1999, "Laptops",     "Silver"),
    (2,  "Dell XPS 15 OLED",               "Intel Core i7 32GB DDR5 1TB SSD 15 inch OLED touchscreen thin light premium creator laptop",                          "Dell",      1799, "Laptops",     "Platinum"),
    (3,  "ASUS ROG Strix G16 Gaming",       "AMD Ryzen 9 RTX 4070 graphics 32GB RAM 1TB SSD 165Hz display RGB keyboard black gaming laptop",                       "ASUS",      1299, "Laptops",     "Black"),
    (4,  "Lenovo ThinkPad X1 Carbon",       "Intel Core i7 vPro 16GB RAM 512GB SSD ultra thin 14 inch business laptop MIL-STD durability excellent keyboard",      "Lenovo",    1549, "Laptops",     "Black"),
    (5,  "HP Spectre x360 14",              "Intel Evo OLED display 2-in-1 convertible 16GB RAM 512GB SSD silver fingerprint reader stylus included",              "HP",        1349, "Laptops",     "Silver"),
    (6,  "Samsung Galaxy S24 Ultra",        "Snapdragon 8 Gen 3 12GB RAM 256GB 200MP camera 6.8 inch AMOLED S Pen titanium black smartphone",                      "Samsung",    999, "Smartphones", "Black"),
    (7,  "iPhone 16 Pro Max",               "Apple A18 Pro chip 48MP camera 5x optical zoom 6.9 inch Super Retina titanium design 256GB",                         "Apple",     1199, "Smartphones", "Titanium"),
    (8,  "Google Pixel 9 Pro",              "Google Tensor G4 50MP triple camera 6.3 inch LTPO 16GB RAM 256GB AI photo editing satellite SOS",                     "Google",     999, "Smartphones", "Hazel"),
    (9,  "OnePlus 12 5G",                   "Snapdragon 8 Gen 3 Hasselblad 50MP camera 100W fast charging 6.82 inch AMOLED 16GB RAM 256GB black",                  "OnePlus",    799, "Smartphones", "Black"),
    (10, "Sony WH-1000XM5",                 "Industry leading noise cancelling headphones 30hr battery LDAC Hi-Res audio multipoint black wireless",               "Sony",       349, "Audio",       "Black"),
    (11, "Apple AirPods Pro 2nd Gen",       "Active noise cancellation transparency mode H2 chip spatial audio USB-C case 30hr total battery white",               "Apple",      249, "Audio",       "White"),
    (12, "Bose QuietComfort 45",            "World class noise cancelling 24hr battery 2.5hr quick charge plush ear cushions adjustable EQ black wireless",        "Bose",       279, "Audio",       "Black"),
    (13, "Jabra Evolve2 85",                "Professional wireless headset 37hr battery ANC busy light Bluetooth multipoint calls and music black",                 "Jabra",      449, "Audio",       "Black"),
    (14, "Sony PlayStation 5 Slim",         "PS5 console disc drive 1TB SSD 4K gaming 120fps DualSense controller white design",                                   "Sony",       499, "Gaming",      "White"),
    (15, "Xbox Series X",                   "Most powerful Xbox 4K gaming 120fps 1TB SSD Game Pass Quick Resume backward compatible black",                        "Microsoft",  499, "Gaming",      "Black"),
    (16, "Nintendo Switch OLED",            "7 inch OLED screen handheld tabletop modes 64GB storage enhanced audio LAN port white neon",                          "Nintendo",   349, "Gaming",      "White"),
    (17, "Logitech G Pro X Superlight 2",   "Ultra lightweight wireless gaming mouse HERO 25K sensor 95g POWERPLAY 95hr battery white",                            "Logitech",   159, "Gaming",      "White"),
    (18, "Samsung 65 QLED 4K TV",           "Quantum Dot 120Hz HDR10 Dolby Vision smart TV 4K upscaling gaming mode black",                                        "Samsung",   1099, "TVs",         "Black"),
    (19, "LG C3 OLED 55 TV",               "OLED evo panel a9 AI processor 120Hz perfect blacks webOS Google Assistant 4K HDR Dolby Atmos black",                 "LG",        1299, "TVs",         "Black"),
    (20, "Sony Bravia XR A80L OLED 65",    "XR OLED Cognitive Processor 4K 120Hz Dolby Vision PS5 gaming Google TV black",                                         "Sony",      1799, "TVs",         "Black"),
    (21, "Canon EOS R6 Mark II",            "24MP full frame mirrorless 40fps RAW burst 6K RAW video in-body stabilization dual card slots professional camera",   "Canon",     2499, "Cameras",     "Black"),
    (22, "Sony Alpha A7 IV Mirrorless",     "33MP BSI sensor 4K60 10bit video 759 phase detect AF 10fps burst dual CFexpress professional mirrorless",             "Sony",      2499, "Cameras",     "Black"),
    (23, "DJI Air 3 Drone",                 "Dual 48MP cameras 4K 60fps 46min flight omnidirectional obstacle sensing RC controller 600g drone",                   "DJI",       1099, "Cameras",     "Gray"),
    (24, "iPad Pro 13 M4",                  "Apple M4 chip Ultra Retina XDR OLED 256GB Apple Pencil Pro 5G space gray thinnest Apple product tablet",              "Apple",     1299, "Tablets",     "Gray"),
    (25, "Samsung Galaxy Tab S9 Ultra",     "14.6 inch AMOLED Snapdragon 8 Gen 2 12GB RAM 256GB S Pen 12000mAh battery graphite tablet",                          "Samsung",   1199, "Tablets",     "Graphite"),
    (26, "Apple Watch Series 9",            "Always on Retina S9 chip blood oxygen ECG crash detection 18hr battery aluminum midnight smartwatch",                 "Apple",      399, "Wearables",   "Midnight"),
    (27, "Samsung Galaxy Watch 6 Classic",  "46mm rotating bezel AMOLED sleep tracking body composition Wear OS 5ATM black stainless smartwatch",                  "Samsung",    429, "Wearables",   "Black"),
    (28, "ASUS ROG Swift 27 4K Monitor",    "27 inch IPS 4K 144Hz G-Sync HDR600 1ms USB-C ergonomic gaming monitor black",                                         "ASUS",       899, "Monitors",    "Black"),
    (29, "LG UltraWide 34 Curved Monitor",  "34 inch 21:9 curved IPS WQHD 3440x1440 144Hz FreeSync USB-C picture by picture productivity monitor black",          "LG",         599, "Monitors",    "Black"),
    (30, "Anker PowerBank 26800mAh",        "26800mAh 65W USB-C fast charging power 3 devices LED display black travel portable charger",                           "Anker",       89, "Accessories", "Black"),
]


def create_db(path: str = DB_PATH) -> None:
    """Create the SQLite DB and insert all products."""
    conn = sqlite3.connect(path)
    cur = conn.cursor()

    cur.execute("DROP TABLE IF EXISTS products")
    cur.execute("""
        CREATE TABLE products (
            id           INTEGER PRIMARY KEY,
            title        TEXT NOT NULL,
            description  TEXT NOT NULL,
            manufacturer TEXT NOT NULL,
            price        REAL NOT NULL,
            category     TEXT NOT NULL,
            color        TEXT NOT NULL
        )
    """)

    cur.executemany(
        "INSERT INTO products VALUES (?,?,?,?,?,?,?)",
        PRODUCTS
    )
    conn.commit()
    conn.close()
    print(f"[DB] Created '{path}' with {len(PRODUCTS)} products.")


def load_products(path: str = DB_PATH) -> list[dict]:
    """Return all products as a list of dicts."""
    if not os.path.exists(path):
        create_db(path)

    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM products").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def product_to_text(p: dict) -> str:
    """
    Convert a product dict into a single searchable string.
    Title is repeated to boost its weight in TF-IDF / BM25.
    """
    return (
        f"{p['title']} {p['title']} "
        f"{p['description']} "
        f"{p['manufacturer']} "
        f"{p['category']} "
        f"{p['color']}"
    )


if __name__ == "__main__":
    create_db()
    products = load_products()
    print(f"[DB] Loaded {len(products)} products.")
    print(f"[DB] Sample: {products[0]}")
