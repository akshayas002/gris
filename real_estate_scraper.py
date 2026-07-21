"""
Real Estate Scraper — Day 3 + Day 4
Targets: MagicBricks & 99acres (Hyderabad listings)
Stack: Playwright + BeautifulSoup + anti-detection + CSV/JSON export
"""

import asyncio
import json
import csv
import random
import time
import logging
import re
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Optional

from playwright.async_api import async_playwright, TimeoutError as PWTimeout
from bs4 import BeautifulSoup

# ─── Logging setup ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("scraper.log", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)


# ─── Config ───────────────────────────────────────────────────────────────────
OUTPUT_DIR = Path("data")
OUTPUT_DIR.mkdir(exist_ok=True)

MAX_PAGES   = 5          # pages to scrape per source (increase for more data)
MIN_DELAY   = 2.5        # seconds between requests (be respectful)
MAX_DELAY   = 5.5
HEADLESS    = True       # set False to watch the browser during debugging


# ─── User-Agent pool (rotated per request) ────────────────────────────────────
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:125.0) Gecko/20100101 Firefox/125.0",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_4_1) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4.1 Safari/605.1.15",
]


# ─── Data model ───────────────────────────────────────────────────────────────
@dataclass
class Property:
    source:        str  = ""          # "magicbricks" or "99acres"
    title:         str  = ""
    price:         str  = ""          # raw string, e.g. "₹45 Lakh"
    price_numeric: Optional[float] = None   # normalised INR value
    location:      str  = ""
    city:          str  = "Hyderabad"
    area_sqft:     str  = ""
    property_type: str  = ""          # Flat / Villa / Plot / etc.
    bedrooms:      str  = ""
    bathrooms:     str  = ""
    amenities:     list = field(default_factory=list)
    listing_url:   str  = ""
    scraped_at:    str  = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self):
        d = asdict(self)
        d["amenities"] = ", ".join(d["amenities"])
        return d


# ─── Price normaliser ─────────────────────────────────────────────────────────
def parse_price(raw: str) -> Optional[float]:
    """
    '₹45 Lakh'  → 4_500_000
    '1.2 Crore' → 12_000_000
    '₹4,50,000' → 450_000
    Returns None if unparseable.
    """
    if not raw:
        return None
    raw = raw.replace("₹", "").replace(",", "").strip().lower()
    try:
        if "crore" in raw:
            num = float(re.search(r"[\d.]+", raw).group())
            return round(num * 1e7)
        elif "lakh" in raw or "lac" in raw:
            num = float(re.search(r"[\d.]+", raw).group())
            return round(num * 1e5)
        else:
            num = float(re.search(r"[\d.]+", raw).group())
            return round(num)
    except Exception:
        return None


# ─── Anti-detection helpers ───────────────────────────────────────────────────
async def stealth_goto(page, url: str) -> bool:
    """Navigate with retry logic and random delay."""
    for attempt in range(3):
        try:
            await page.goto(url, wait_until="domcontentloaded", timeout=30_000)
            await asyncio.sleep(random.uniform(MIN_DELAY, MAX_DELAY))
            return True
        except PWTimeout:
            log.warning(f"Timeout on {url} (attempt {attempt+1}/3)")
            await asyncio.sleep(3)
        except Exception as e:
            log.error(f"Navigation error: {e}")
            return False
    return False


async def random_scroll(page):
    """Scroll in random increments to trigger lazy-loading and look human."""
    for _ in range(random.randint(3, 6)):
        scroll_y = random.randint(300, 700)
        await page.evaluate(f"window.scrollBy(0, {scroll_y})")
        await asyncio.sleep(random.uniform(0.4, 1.0))


# ─── MagicBricks scraper ──────────────────────────────────────────────────────
async def scrape_magicbricks(context) -> list[Property]:
    """
    Scrapes buy listings in Hyderabad from MagicBricks.
    URL pattern: https://www.magicbricks.com/property-for-sale/residential-real-estate?proptype=Multistorey-Apartment,Builder-Floor-Apartment,Penthouse,Studio-Apartment&cityName=Hyderabad&page=N
    """
    results: list[Property] = []
    base_url = (
        "https://www.magicbricks.com/property-for-sale/residential-real-estate"
        "?proptype=Multistorey-Apartment,Builder-Floor-Apartment,Penthouse,Studio-Apartment"
        "&cityName=Hyderabad&page={page}"
    )

    page = await context.new_page()
    await page.set_extra_http_headers({"User-Agent": random.choice(USER_AGENTS)})

    for pg in range(1, MAX_PAGES + 1):
        url = base_url.format(page=pg)
        log.info(f"[MagicBricks] Page {pg} → {url}")

        ok = await stealth_goto(page, url)
        if not ok:
            log.warning(f"[MagicBricks] Skipping page {pg}")
            continue

        await random_scroll(page)

        # Wait for listing cards
        try:
            await page.wait_for_selector(".mb-srp__card", timeout=15_000)
        except PWTimeout:
            log.warning(f"[MagicBricks] No cards found on page {pg} — site may have changed structure")
            # Save raw HTML for debugging
            html = await page.content()
            (OUTPUT_DIR / f"mb_debug_p{pg}.html").write_text(html, encoding="utf-8")
            continue

        soup = BeautifulSoup(await page.content(), "html.parser")
        cards = soup.select(".mb-srp__card")
        log.info(f"[MagicBricks] Found {len(cards)} cards on page {pg}")

        for card in cards:
            try:
                prop = Property(source="magicbricks")

                # Title / property name
                title_el = card.select_one(".mb-srp__card--title")
                prop.title = title_el.get_text(strip=True) if title_el else ""

                # Price
                price_el = card.select_one(".mb-srp__card--price")
                prop.price = price_el.get_text(strip=True) if price_el else ""
                prop.price_numeric = parse_price(prop.price)

                # Location
                loc_el = card.select_one(".mb-srp__card--locality")
                prop.location = loc_el.get_text(strip=True) if loc_el else ""

                # Area
                area_el = card.select_one('[data-summary="carpet-area"] .mb-srp__card--summary--val')
                if not area_el:
                    area_el = card.select_one(".mb-srp__card--summary--val")
                prop.area_sqft = area_el.get_text(strip=True) if area_el else ""

                # Bedrooms / bathrooms from summary items
                summary_items = card.select(".mb-srp__card--summary--item")
                for item in summary_items:
                    label_el = item.select_one(".mb-srp__card--summary--label")
                    val_el   = item.select_one(".mb-srp__card--summary--val")
                    if not label_el or not val_el:
                        continue
                    label = label_el.get_text(strip=True).lower()
                    val   = val_el.get_text(strip=True)
                    if "bed" in label:
                        prop.bedrooms = val
                    elif "bath" in label:
                        prop.bathrooms = val
                    elif "carpet" in label or "area" in label:
                        prop.area_sqft = prop.area_sqft or val

                # Property type from tag
                type_el = card.select_one(".mb-srp__card--type")
                prop.property_type = type_el.get_text(strip=True) if type_el else ""

                # Amenities
                amenity_els = card.select(".mb-srp__card--amenities--list li")
                prop.amenities = [a.get_text(strip=True) for a in amenity_els if a.get_text(strip=True)]

                # Listing URL
                link_el = card.select_one("a.mb-srp__card--photo")
                if not link_el:
                    link_el = card.select_one("a[href*='/property-for-sale/']")
                if link_el:
                    href = link_el.get("href", "")
                    prop.listing_url = href if href.startswith("http") else f"https://www.magicbricks.com{href}"

                if prop.price or prop.title:
                    results.append(prop)

            except Exception as e:
                log.error(f"[MagicBricks] Card parse error: {e}")

    await page.close()
    log.info(f"[MagicBricks] Total scraped: {len(results)}")
    return results


# ─── 99acres scraper ──────────────────────────────────────────────────────────
async def scrape_99acres(context) -> list[Property]:
    """
    Scrapes buy listings in Hyderabad from 99acres.
    URL: https://www.99acres.com/property-in-hyderabad-ffid?search_type=b&search_location=Hyderabad&page=N
    """
    results: list[Property] = []
    base_url = (
        "https://www.99acres.com/property-in-hyderabad-ffid"
        "?search_type=b&search_location=Hyderabad&page={page}"
    )

    page = await context.new_page()
    await page.set_extra_http_headers({"User-Agent": random.choice(USER_AGENTS)})

    for pg in range(1, MAX_PAGES + 1):
        url = base_url.format(page=pg)
        log.info(f"[99acres] Page {pg} → {url}")

        ok = await stealth_goto(page, url)
        if not ok:
            log.warning(f"[99acres] Skipping page {pg}")
            continue

        await random_scroll(page)

        # Wait for listing cards (99acres uses different selectors per update)
        selectors_to_try = [
            ".srpTuple__propertyTuple",
            "[data-tracking-id='property_card']",
            ".tupleNew__outerTuple",
            "article.srpTuple",
        ]
        found_selector = None
        for sel in selectors_to_try:
            try:
                await page.wait_for_selector(sel, timeout=8_000)
                found_selector = sel
                break
            except PWTimeout:
                continue

        if not found_selector:
            log.warning(f"[99acres] No cards found on page {pg} — saving debug HTML")
            html = await page.content()
            (OUTPUT_DIR / f"99_debug_p{pg}.html").write_text(html, encoding="utf-8")
            continue

        soup = BeautifulSoup(await page.content(), "html.parser")
        cards = soup.select(found_selector)
        log.info(f"[99acres] Found {len(cards)} cards using '{found_selector}' on page {pg}")

        for card in cards:
            try:
                prop = Property(source="99acres")

                # Title
                for sel in [".srpTuple__propertyName", ".tupleNew__propName", "h2"]:
                    el = card.select_one(sel)
                    if el:
                        prop.title = el.get_text(strip=True)
                        break

                # Price
                for sel in [".srpTuple__price", ".tupleNew__priceValWrap", "[class*='price']"]:
                    el = card.select_one(sel)
                    if el:
                        prop.price = el.get_text(strip=True)
                        break
                prop.price_numeric = parse_price(prop.price)

                # Location
                for sel in [".srpTuple__address", ".tupleNew__locality", "[class*='locality']"]:
                    el = card.select_one(sel)
                    if el:
                        prop.location = el.get_text(strip=True)
                        break

                # Area
                area_text = ""
                for sel in [".srpTuple__area", ".tupleNew__areaVal", "[class*='area']"]:
                    el = card.select_one(sel)
                    if el:
                        area_text = el.get_text(strip=True)
                        break
                prop.area_sqft = area_text

                # Bedrooms
                for sel in [".srpTuple__bedroomBathroom", "[class*='bedroom']", "[class*='bhk']"]:
                    el = card.select_one(sel)
                    if el:
                        txt = el.get_text(strip=True)
                        bhk_match = re.search(r"(\d+)\s*BHK", txt, re.I)
                        bath_match = re.search(r"(\d+)\s*Bath", txt, re.I)
                        if bhk_match:
                            prop.bedrooms = bhk_match.group(1)
                        if bath_match:
                            prop.bathrooms = bath_match.group(1)
                        break

                # Property type
                for sel in [".srpTuple__propertyType", "[class*='propType']"]:
                    el = card.select_one(sel)
                    if el:
                        prop.property_type = el.get_text(strip=True)
                        break

                # Amenities
                for sel in [".srpTuple__amenities li", "[class*='amenity']"]:
                    els = card.select(sel)
                    if els:
                        prop.amenities = [e.get_text(strip=True) for e in els if e.get_text(strip=True)]
                        break

                # URL
                link = card.select_one("a[href]")
                if link:
                    href = link["href"]
                    prop.listing_url = href if href.startswith("http") else f"https://www.99acres.com{href}"

                if prop.price or prop.title:
                    results.append(prop)

            except Exception as e:
                log.error(f"[99acres] Card parse error: {e}")

    await page.close()
    log.info(f"[99acres] Total scraped: {len(results)}")
    return results


# ─── Export helpers ───────────────────────────────────────────────────────────
def save_json(properties: list[Property], path: Path):
    data = [p.to_dict() for p in properties]
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info(f"Saved JSON → {path}  ({len(data)} records)")


def save_csv(properties: list[Property], path: Path):
    if not properties:
        log.warning("No data to save to CSV")
        return
    rows = [p.to_dict() for p in properties]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    log.info(f"Saved CSV  → {path}  ({len(rows)} records)")


def print_summary(properties: list[Property]):
    log.info("─" * 60)
    log.info(f"TOTAL LISTINGS: {len(properties)}")
    by_source = {}
    for p in properties:
        by_source[p.source] = by_source.get(p.source, 0) + 1
    for src, count in by_source.items():
        log.info(f"  {src:20s}: {count}")
    priced = [p for p in properties if p.price_numeric]
    if priced:
        avg = sum(p.price_numeric for p in priced) / len(priced)
        log.info(f"  Avg price (parsed): ₹{avg:,.0f}")
    log.info("─" * 60)


# ─── Main ─────────────────────────────────────────────────────────────────────
async def main():
    log.info("Starting real estate scraper 🏠")
    all_properties: list[Property] = []

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(
            headless=HEADLESS,
            args=[
                "--disable-blink-features=AutomationControlled",   # hide automation flag
                "--no-sandbox",
                "--disable-dev-shm-usage",
                "--disable-gpu",
            ],
        )

        # Browser context with realistic fingerprint
        context = await browser.new_context(
            viewport={"width": 1366, "height": 768},
            user_agent=random.choice(USER_AGENTS),
            locale="en-IN",
            timezone_id="Asia/Kolkata",
            # Mask webdriver flag
            java_script_enabled=True,
        )

        # Inject stealth script — hides navigator.webdriver
        await context.add_init_script("""
            Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
            Object.defineProperty(navigator, 'plugins', { get: () => [1, 2, 3] });
            Object.defineProperty(navigator, 'languages', { get: () => ['en-IN', 'en'] });
            window.chrome = { runtime: {} };
        """)

        # Scrape both sources
        try:
            mb_results = await scrape_magicbricks(context)
            all_properties.extend(mb_results)
        except Exception as e:
            log.error(f"MagicBricks scraper failed: {e}")

        # Small pause between sites
        await asyncio.sleep(random.uniform(3, 6))

        try:
            acres_results = await scrape_99acres(context)
            all_properties.extend(acres_results)
        except Exception as e:
            log.error(f"99acres scraper failed: {e}")

        await browser.close()

    # Save outputs
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M")
    save_json(all_properties, OUTPUT_DIR / f"listings_{ts}.json")
    save_csv(all_properties, OUTPUT_DIR / f"listings_{ts}.csv")

    # Also save a 'latest' copy for easy pipeline access
    save_json(all_properties, OUTPUT_DIR / "listings_latest.json")
    save_csv(all_properties, OUTPUT_DIR / "listings_latest.csv")

    print_summary(all_properties)


if __name__ == "__main__":
    asyncio.run(main())
