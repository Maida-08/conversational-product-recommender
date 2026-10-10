# Synthetic Development Dataset

> **This is invented data. It is NOT the original assignment dataset (`amazon_dataset_curated.csv`).**
> It was created because the original CSV was not available during development.
> Product names, brands, prices, ratings and reviews are all fictional. Any resemblance to real products is coincidental.

## Files

| File | Purpose |
|---|---|
| `amazon_dataset_synthetic_100.csv` | The catalog: 100 rows, exactly the 9 assignment columns |
| `build_synthetic_dataset.py` | Reproducible generator + validation (run it to regenerate both files) |
| `synthetic_test_manifest.json` | Ground-truth labels for planted cases and an example query playbook |
| `SYNTHETIC_DATA.md` | This file |

**The recommendation pipeline never reads the manifest.** It exists only for tests and documentation, so the system cannot cheat by looking up the answer.

## Schema

`product_id, category, brand, title, description, price_usd, average_rating, num_reviews, review_snippets`

`review_snippets` holds 3 to 4 excerpts separated by `" | "`, as in the assignment.

## Composition

| Category | Rows |
|---|---|
| footwear | 13 |
| electronics | 13 |
| home | 12 |
| beauty | 12 |
| sports | 12 |
| kitchen | 12 |
| outdoor | 12 |
| toys | 12 |
| pets, office (one-offs) | 1 each |

Realism choices: fictional brands only, mixed units (oz / g / lb / kg), a range of description styles (attribute-rich, marketing-style, terse), and ratings and review counts that scale sensibly with price and popularity.

## Planted test situations

| Situation | Examples |
|---|---|
| Strong match | P001 (waterproof trail runner), P014 (basic earbuds), P034 (large purifier) |
| Partial match | P002 (light but not waterproof), P011 (basic shoe with unknown weight/waterproofing) |
| Budget conflict | P001 (about $9.50 over an $80 ask), P076 (40L pack vs a $40 ask) |
| Missing attribute (must become `unknown`, not `false`) | P011, P024, P085, P049 |
| Review-only negative signal | P003 (canvas sneaker soaks in rain), P008, P051 |
| Review-only positive signal | P024 (sweat and rain), P085 (headlamp in downpour), P060, P049 |
| Description vs. review contradiction | P006, P037, P055, P068, P079, P086, P080 (hedged) |
| A lone review that must not overturn the description | P007, P043, P059 |
| Upsell families (concrete data-backed benefit) | P014 to P015, P018 to P019, P022 to P023, P045 to P046, P051 to P052, P063 to P064, P065 to P066, P071 to P072, P076 to P077, P093 to P094, P033 to P034 |
| Upsell decoys (must NOT be pushed) | P016 (over-ear, different form factor), P017 (cheaper but rated 3.4), P010 (pricey, no relevant benefit) |
| Not wireless (must not match wireless queries) | P026 |
| Gift candidates (for the clarifying question) | P027, P038, P042, P050, P073, P032 |
| Ambiguous / no match | "something nice for my mom" has no single right category. There is deliberately no standing desk (only P100, a laptop stand, as a loose neighbor) |

## Scenario analogs (from the assignment's four sample scenarios)

| Assignment scenario | Synthetic analog |
|---|---|
| A: lightweight waterproof running shoe under $80 | P001 top pick, P002 secondary |
| B: something nice for my mom | clarification |
| C: casual sneaker in light rain | P003 flagged, P004 alternative |
| D: cheap basic earbuds | P014 primary, P015 upsell |

## Caveat

Tuning ranking weights and thresholds against a dataset we wrote ourselves risks overfitting to our own assumptions. The write-up will say so, and the pipeline reads only the CSV, so it will run unchanged on the real dataset when available.
