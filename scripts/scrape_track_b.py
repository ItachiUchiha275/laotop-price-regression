"""
Track B: Combined scraper — ryans.com + startech.com.bd
=====================================================
Target: ≥300 unique laptop listings with fields:
  Brand, Model, Processor, RAM, Storage, Display, GPU, Price(BDT), 
  Availability, Source, Product_URL

Output: data/track_b/raw/track_b_listings.csv (clean, human-readable CSV)
No raw HTML files are saved — only the final CSV.

Respects robots.txt (Allow: /, 1.5s delay between requests).
"""
import os, sys, json, csv, time, re
import requests
from bs4 import BeautifulSoup

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "track_b", "raw")
os.makedirs(RAW_DIR, exist_ok=True)
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
DELAY = 1.5

OUTPUT_FIELDS = [
    "Brand", "Model", "Processor", "RAM", "Storage", "Display",
    "GPU", "Price_BDT", "Availability", "Source", "Product_URL"
]

# ════════════════════════════════════════════════════════════════
# SCRAPER: RYANS.COM
# ════════════════════════════════════════════════════════════════
def scrape_ryans():
    """Scrape ryans.com/category/laptop-all-laptop (13 pages, ~260 listings)."""
    base = "https://www.ryans.com/category/laptop-all-laptop"
    products = []

    # Get first page to determine total pages
    r = requests.get(base, headers=HEADERS, timeout=30)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")

    # Parse pagination
    page_links = soup.select(".pagination .page-link")
    total_pages = 1
    for a in page_links:
        txt = a.get_text(strip=True)
        if txt.isdigit():
            total_pages = max(total_pages, int(txt))
    print(f"  Ryans pages: {total_pages}")

    def parse_ryans_card(card):
        """Extract product info from a .card.h-100 element."""
        # Structured specs from data-attributes JSON
        specs = {}
        preview_btn = card.select_one(".product-preview-btn")
        if preview_btn:
            attrs_json = preview_btn.get("data-attributes")
            if attrs_json:
                try:
                    data = json.loads(attrs_json).get("data", {})
                    for k, v in data.items():
                        key = k.strip().lower().replace(" ", "_").replace(".", "")
                        specs[key] = v.strip() if v else ""
                except json.JSONDecodeError:
                    pass

        # Product name: use img alt (full name, not truncated like stretched-link)
        name = ""
        link = card.select_one("a.stretched-link")
        img = card.select_one(".card-img-top")
        if img:
            name = img.get("alt", "").strip()
        if not name or len(name) < 5:
            if link:
                name = link.get_text(strip=True)
        # Final fallback
        name_el = card.select_one(".product-name")
        if name_el and (not name or len(name) < 5):
            name = name_el.get_text(strip=True)

        # Clean model name: remove trailing model codes like "33.01.523.06"
        clean_name = name
        clean_name = re.sub(r'\s*\.{2,}[\d.]+$', '', clean_name)
        clean_name = re.sub(r'\s*\.{2,}[.\d\s]+$', '', clean_name)
        name = clean_name.strip()

        # Brand: first word of product name
        brand_raw = name.split()[0] if name else ""
        brand = brand_raw

        # URL
        url = link.get("href", "") if link else ""

        # Price: "Tk 64,500"
        price_raw = ""
        price_el = card.select_one(".pr-text")
        if price_el:
            price_raw = price_el.get_text(strip=True)

        # Parse price: strip "Tk" and commas
        price_num = ""
        if price_raw:
            p = price_raw.replace("Tk", "").replace(",", "").strip()
            try:
                price_num = str(int(float(p)))
            except ValueError:
                price_num = ""

        # Extract fields from specs dict
        processor = specs.get("processor_type_", "")
        if not processor:
            m = re.search(r'(Intel|AMD|Apple|Qualcomm|MediaTek)\s+\w+\s+[\w\d-]+', name)
            if m:
                processor = m.group(0)
            else:
                m2 = re.search(r'(Core\s+i\d[\w-]*|Ryzen\s+[\w\d]+|Snapdragon\s+\w+|M\d?\s*[Pp]ro?)', name)
                if m2:
                    processor = m2.group(0)

        ram = specs.get("ram", "")
        storage = specs.get("storage", "")
        display = specs.get("display_size_(inch)", "")
        gpu = specs.get("graphics_chipset", "")
        # Clean GPU name — some have "Graphics Chipset:" prefix
        if gpu and ":" in str(gpu):
            gpu = gpu.split(":")[-1].strip()

        # Build display string
        display_str = display + '"' if display else ""
        # Extract screen res from name if possible
        res_match = re.search(r'(\d+x\d+)', name, re.IGNORECASE)
        if res_match:
            if display_str:
                display_str = display_str + " " + res_match.group(1)
            else:
                display_str = res_match.group(1)

        availability = "In Stock"

        return {
            "Brand": brand,
            "Model": name,
            "Processor": processor,
            "RAM": ram,
            "Storage": storage,
            "Display": display_str if display_str else "",
            "GPU": gpu,
            "Price_BDT": price_num,
            "Availability": availability,
            "Source": "ryans.com",
            "Product_URL": url,
        }

    # Parse page 1
    for card in soup.select(".card.h-100"):
        products.append(parse_ryans_card(card))
    print(f"  Page 1: {len(soup.select('.card.h-100'))} items")

    # Pages 2..N
    for page in range(2, total_pages + 1):
        time.sleep(DELAY)
        url = f"{base}?page={page}"
        try:
            r2 = requests.get(url, headers=HEADERS, timeout=30)
            r2.raise_for_status()
            s2 = BeautifulSoup(r2.text, "html.parser")
            cards = s2.select(".card.h-100")
            for card in cards:
                products.append(parse_ryans_card(card))
            print(f"  Page {page}: {len(cards)} items")
        except Exception as e:
            print(f"  Page {page}: FAILED — {e}")

    return products


# ════════════════════════════════════════════════════════════════
# SCRAPER: STARTECH.COM.BD
# ════════════════════════════════════════════════════════════════
def scrape_startech():
    """Scrape startech.com.bd laptop + ultrabook pages."""
    base = "https://www.startech.com.bd"
    categories = ["/laptop-notebook/laptop", "/laptop-notebook/ultrabook"]
    products = []

    def total_pages(soup):
        pag = soup.select_one(".pagination")
        if not pag:
            return 1
        nums = []
        for a in pag.find_all("a"):
            t = a.get_text(strip=True)
            if t.isdigit():
                nums.append(int(t))
        active = pag.select_one(".active span")
        if active and active.get_text(strip=True).isdigit():
            nums.append(int(active.get_text(strip=True)))
        return max(nums) if nums else 1

    def parse_startech(item):
        name_el = item.select_one(".p-item-name a")
        name = name_el.get_text(strip=True) if name_el else ""
        url = name_el.get("href", "") if name_el else ""
        price_el = item.select_one(".p-item-price span")
        price_raw = price_el.get_text(strip=True) if price_el else ""
        # Parse price: "145,000৳" → "145000"
        price_num = ""
        if price_raw:
            p = price_raw.replace("৳", "").replace(",", "").strip()
            try:
                price_num = str(int(float(p)))
            except ValueError:
                price_num = ""

        specs = {}
        for li in item.select(".short-description li"):
            txt = li.get_text(strip=True)
            low = txt.lower()
            if "processor" in low:
                specs["Processor"] = txt.split(":", 1)[1].strip() if ":" in txt else txt
            elif "ram" in low:
                v = txt.split(":", 1)[1].strip() if ":" in txt else txt
                specs["RAM"] = v.split(",")[0].strip()
                m = re.search(r'[Ss]torage\s*:\s*(.+?)(?:$|,)', txt)
                if m:
                    specs["Storage"] = m.group(1).strip()
            elif "storage" in low:
                specs["Storage"] = txt.split(":", 1)[1].strip() if ":" in txt else txt
            elif "display" in low:
                specs["Display"] = txt.split(":", 1)[1].strip() if ":" in txt else txt
            elif any(g in low for g in ["graphics", "gpu"]):
                specs["GPU"] = txt.split(":", 1)[1].strip() if ":" in txt else txt

        brand = name.split()[0] if name else ""
        availability = "In Stock" if item.select_one(".btn-add-cart") else "Out of Stock"

        return {
            "Brand": brand,
            "Model": name,
            "Processor": specs.get("Processor", ""),
            "RAM": specs.get("RAM", ""),
            "Storage": specs.get("Storage", ""),
            "Display": specs.get("Display", ""),
            "GPU": specs.get("GPU", ""),
            "Price_BDT": price_num,
            "Availability": availability,
            "Source": "startech.com.bd",
            "Product_URL": url,
        }

    for cat in categories:
        cat_url = base + cat
        print(f"\n  Startech category: {cat}")
        r = requests.get(cat_url, headers=HEADERS, timeout=30)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        pages = total_pages(soup)
        print(f"  Pages: {pages}")

        for item in soup.select(".p-item"):
            products.append(parse_startech(item))
        print(f"  Page 1: {len(soup.select('.p-item'))} items")

        for pg in range(2, pages + 1):
            time.sleep(DELAY)
            try:
                r2 = requests.get(f"{cat_url}?page={pg}", headers=HEADERS, timeout=30)
                r2.raise_for_status()
                s2 = BeautifulSoup(r2.text, "html.parser")
                items = s2.select(".p-item")
                for item in items:
                    products.append(parse_startech(item))
                print(f"  Page {pg}: {len(items)} items")
            except Exception as e:
                print(f"  Page {pg}: FAILED — {e}")

    return products


# ════════════════════════════════════════════════════════════════
# DETAIL PAGE SCRAPER: recover missing GPU from Startech detail pages
# ════════════════════════════════════════════════════════════════
def recover_missing_gpu(products):
    """Visit individual product pages for Startech listings missing GPU info."""
    startech_missing = [p for p in products if p["Source"] == "startech.com.bd" and not p["GPU"]]
    print(f"\n  Recovering GPU from {len(startech_missing)} Startech detail pages...")
    recovered = 0
    for i, p in enumerate(startech_missing):
        url = p["Product_URL"]
        if not url:
            continue
        time.sleep(DELAY)
        try:
            r = requests.get(url, headers=HEADERS, timeout=30)
            r.raise_for_status()
            soup = BeautifulSoup(r.text, "html.parser")
            # Find spec tables
            for table in soup.select(".table"):
                rows = table.select("tr")
                for row in rows:
                    cells = row.select("td")
                    if len(cells) >= 2:
                        key = cells[0].get_text(strip=True).lower()
                        val = cells[1].get_text(strip=True)
                        if "graphics model" in key or "graphics type" in key or "graphics" == key:
                            if val and val.lower() not in ("n/a", "none", ""):
                                p["GPU"] = val
                                recovered += 1
                                break
                if p["GPU"]:
                    break
            # If still no GPU, try the product description area
            if not p["GPU"]:
                desc = soup.select_one(".product-desc, #tab-description, .description")
                if desc:
                    text = desc.get_text()
                    for keyword in ["integrated", "graphics", "adreno", "iris", "uhd", "radeon"]:
                        if keyword in text.lower():
                            # Extract a short GPU mention
                            m = re.search(r'([\w\s]+Graphics[\w\s]*)', text, re.IGNORECASE)
                            if m:
                                p["GPU"] = m.group(1).strip()[:80]
                                recovered += 1
                                break
        except Exception as e:
            pass
        if (i + 1) % 20 == 0:
            print(f"    ... {i+1}/{len(startech_missing)} detail pages visited ({recovered} GPUs found)")
    print(f"  Recovered GPU for {recovered}/{len(startech_missing)} listings")
    return products


# ════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════
def main():
    print("=" * 60)
    print("Scraping ryans.com ...")
    print("=" * 60)
    ryans_products = scrape_ryans()
    print(f"\n  Ryans total: {len(ryans_products)}")

    print("\n" + "=" * 60)
    print("Scraping startech.com.bd ...")
    print("=" * 60)
    startech_products = scrape_startech()
    print(f"\n  Startech total: {len(startech_products)}")

    # Recover missing GPU from Startech detail pages
    all_products = ryans_products + startech_products
    all_products = recover_missing_gpu(all_products)
    print(f"\n{'=' * 60}")
    print(f"Combined: {len(all_products)}")

    # Normalize brand names across both sources
    brand_fixes = {
        "ASUS": "ASUS", "Asus": "ASUS", "asus": "ASUS",
        "MSI": "MSI", "Msi": "MSI", "msi": "MSI",
        "HP": "HP", "Hp": "HP",
        "Dell": "Dell", "DELL": "Dell",
        "Lenovo": "Lenovo", "LENOVO": "Lenovo",
        "Acer": "Acer", "ACER": "Acer",
        "Apple": "Apple", "MacBook": "Apple", "MACBOOK": "Apple",
        "Microsoft": "Microsoft", "microsoft": "Microsoft",
        "Gigabyte": "Gigabyte", "GIGABYTE": "Gigabyte",
        "Walton": "Walton", "WALTON": "Walton",
        "TECNO": "Tecno", "Chuwi": "Chuwi",
    }
    for p in all_products:
        raw = p["Brand"]
        p["Brand"] = brand_fixes.get(raw, raw)
        # Remove leading parenthesis garbage
        if p["Brand"].startswith("("):
            p["Brand"] = p["Model"].split()[0] if p["Model"] else ""
            p["Brand"] = brand_fixes.get(p["Brand"], p["Brand"])

    # Deduplicate by Product_URL
    seen = set()
    unique = []
    for p in all_products:
        if p["Product_URL"] and p["Product_URL"] not in seen:
            seen.add(p["Product_URL"])
            unique.append(p)
        elif not p["Product_URL"]:
            unique.append(p)

    print(f"After dedup by URL: {len(unique)}")
    print(f"  Ryans: {sum(1 for p in unique if p['Source'] == 'ryans.com')}")
    print(f"  Startech: {sum(1 for p in unique if p['Source'] == 'startech.com.bd')}")

    # Save clean CSV
    csv_path = os.path.join(RAW_DIR, "track_b_listings.csv")
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS)
        w.writeheader()
        w.writerows(unique)
    print(f"\nSaved: {csv_path}")

    # Metadata
    meta = {
        "sources": ["ryans.com", "startech.com.bd"],
        "products_by_source": {
            "ryans.com": len(ryans_products),
            "startech.com.bd": len(startech_products),
        },
        "total_unique": len(unique),
        "scrape_date": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(os.path.join(RAW_DIR, "scrape_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print("\nDone.")

if __name__ == "__main__":
    main()
