"""
Catalog loading and persistence.

Responsibilities (and nothing else):
  - read the CSV and validate that the required columns exist
  - clean messy values (blank cells, "$1,299.00", stray whitespace, Excel's UTF-8 BOM)
  - split review_snippets on " | "
  - save / load processed_products.json

Uses only the standard-library csv module (no pandas): fewer compiled dependencies,
faster imports, and nothing for Windows Application Control to block.
"""
import csv
import json
import re
from pathlib import Path
from typing import Optional

from app.config import get_settings
from app.models.schemas import ProcessedProduct, RawProduct

REQUIRED_COLUMNS = [
    "product_id", "category", "brand", "title", "description",
    "price_usd", "average_rating", "num_reviews", "review_snippets",
]


def _to_float(value: str) -> Optional[float]:
    cleaned = re.sub(r"[^\d.\-]", "", value or "")
    try:
        return float(cleaned) if cleaned not in ("", "-", ".") else None
    except ValueError:
        return None


def _to_int(value: str) -> Optional[int]:
    f = _to_float(value)
    return int(f) if f is not None else None


def _split_reviews(value: str) -> list[str]:
    return [r.strip() for r in re.split(r"\s*\|\s*", value or "") if r.strip()]


def load_catalog(path: Optional[Path] = None) -> list[RawProduct]:
    """Load the CSV into validated RawProduct objects."""
    path = Path(path) if path else get_settings().catalog_csv
    if not path.exists():
        raise FileNotFoundError(f"Catalog CSV not found: {path}")

    # utf-8-sig silently drops the BOM that Excel adds, so the first column name stays clean
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        columns = [(c or "").strip() for c in (reader.fieldnames or [])]
        missing = [c for c in REQUIRED_COLUMNS if c not in columns]
        if missing:
            raise ValueError(f"Catalog is missing required columns: {missing}")
        rows = [{(k or "").strip(): (v or "") for k, v in row.items()} for row in reader]

    products: list[RawProduct] = []
    seen: set[str] = set()
    for row in rows:
        pid = row["product_id"].strip()
        if not pid:
            raise ValueError("Found a row with an empty product_id")
        if pid in seen:
            raise ValueError(f"Duplicate product_id: {pid}")
        seen.add(pid)
        products.append(
            RawProduct(
                product_id=pid,
                category=row["category"].strip().lower(),
                brand=row["brand"].strip(),
                title=row["title"].strip(),
                description=row["description"].strip(),
                price_usd=_to_float(row["price_usd"]),
                average_rating=_to_float(row["average_rating"]),
                num_reviews=_to_int(row["num_reviews"]),
                review_snippets=_split_reviews(row["review_snippets"]),
            )
        )
    return products


def save_processed(products: list[ProcessedProduct], path: Optional[Path] = None) -> Path:
    path = Path(path) if path else get_settings().processed_products
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump([p.model_dump() for p in products], f, indent=2, ensure_ascii=False)
    return path


def load_processed(path: Optional[Path] = None) -> list[ProcessedProduct]:
    """Load processed products; returns [] if the file does not exist yet."""
    path = Path(path) if path else get_settings().processed_products
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [ProcessedProduct.model_validate(item) for item in json.load(f)]