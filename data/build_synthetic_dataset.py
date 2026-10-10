"""
Builds the SYNTHETIC development catalog for the Conversational Product Search project.

!! This is invented data. It is NOT the original assignment dataset. !!
All brands are fictional. See SYNTHETIC_DATA.md for the planted test situations.

Outputs (written next to this script):
  - amazon_dataset_synthetic_100.csv   -> the catalog (9 columns, assignment schema)
  - synthetic_test_manifest.json       -> planted traits + example queries.
                                          The pipeline must NEVER read this file;
                                          it exists for tests and documentation only.

Run:  python build_synthetic_dataset.py
"""
import csv
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
CSV_PATH = HERE / "amazon_dataset_synthetic_100.csv"
MANIFEST_PATH = HERE / "synthetic_test_manifest.json"

COLUMNS = [
    "product_id", "category", "brand", "title", "description",
    "price_usd", "average_rating", "num_reviews", "review_snippets",
]

# (id, category, brand, title, description, price, rating, n_reviews, [reviews])
P = []


def add(pid, cat, brand, title, desc, price, rating, n, reviews):
    P.append((pid, cat, brand, title, desc, price, rating, n, reviews))


# ----------------------------------------------------------------- FOOTWEAR
add("P001", "footwear", "Aquaridge", "Aquaridge StormShield Waterproof Trail Running Shoe",
    "Fully waterproof trail runner with a sealed-seam membrane and gusseted tongue. Aggressive 5mm lugs for wet trails. Weighs 10.2 oz (289 g) per shoe in men's size 9. Synthetic mesh upper with TPU overlays.",
    89.50, 4.5, 812,
    ["Feet stayed bone dry through a two-hour downpour", "Grippy on wet rock, but needed a week to break in", "Pricey for a running shoe, though it is the only one that kept me dry"])
add("P002", "footwear", "TrailFox", "TrailFox Featherlite Running Shoe",
    "Ultralight road-to-trail runner at just 7.8 oz (221 g). Breathable engineered mesh upper, 8mm drop and a responsive foam midsole. Built for tempo runs and daily miles.",
    64.99, 4.4, 1543,
    ["Barely feel them on my feet, super light", "Mesh soaked through in the first rain, wet socks the whole run", "Great daily trainer for dry weather"])
add("P003", "footwear", "UrbanStep", "UrbanStep Canvas Casual Sneaker",
    "Classic low-top sneaker in cotton canvas with a vulcanized rubber sole and cushioned insole. Available in six colors. A clean everyday look for casual wear.",
    54.00, 4.2, 966,
    ["Comfortable and looks great with jeans", "The fabric got soaked in light rain and took a day to dry", "Canvas soaked through in a drizzle, would not wear these in wet weather"])
add("P004", "footwear", "Northpath", "Northpath Leather Hiking Shoe",
    "Waterproof full-grain leather hiking shoe with a bootie-style membrane and a lugged rubber outsole. Supportive midsole for day hikes. Weighs 1 lb 14 oz (850 g) per pair.",
    119.00, 4.6, 704,
    ["Kept my feet dry crossing shallow streams", "Sturdy but a little heavy for long days", "Handled steady rain on a day hike without leaking"])
add("P005", "footwear", "Cloudstride", "Cloudstride Max Cushion Road Running Shoe",
    "Maximum-cushion road running shoe with a knit upper. Not waterproof; designed for breathability in warm weather. 9.6 oz (272 g) per shoe. Rocker geometry for long runs.",
    129.00, 4.7, 2210,
    ["Plush ride for marathon training", "Knit upper breathes well and dries fast", "Excellent cushioning but pricey"])
add("P006", "footwear", "Tempo", "Tempo AquaRun Waterproof Lightweight Runner",
    "Lightweight, waterproof running shoe with a water-repellent coated upper and bootie construction. Reflective details for low-light runs. Everyday training in any conditions.",
    74.99, 4.1, 388,
    ["Leaked through the seams on the first stormy run", "The waterproof claim is wrong, my socks were soaked after one puddle", "Fine in a light drizzle and a comfortable fit"])
add("P007", "footwear", "Granite Peak", "Granite Peak Waterproof Hiking Boot",
    "Mid-cut hiking boot with a waterproof membrane and full-grain leather upper. Lugged rubber outsole, padded collar and solid ankle support. Weighs 2 lb 6 oz (1.08 kg) per pair.",
    139.00, 4.6, 1287,
    ["Dry feet through three days of rain in the Cascades", "Great ankle support and very durable", "Started to leak at the toe after about six months", "Excellent traction on muddy trails"])
add("P008", "footwear", "SoftStep", "SoftStep Slip-On Comfort Walking Shoe",
    "Slip-on walking shoe with a stretch knit upper, memory-foam insole and lightweight EVA sole. Machine washable. Made for all-day comfort.",
    49.00, 4.3, 2035,
    ["Easy on and off, very comfortable for errands", "Light as a slipper", "Not great in the rain, the knit gets damp fast"])
add("P009", "footwear", "Riverline", "Riverline Quick-Dry Water Sandal",
    "Sport sandal with adjustable straps, quick-drying webbing and a grippy rubber sole. Designed for water and trail use. Weighs 11 oz (312 g) per pair.",
    45.00, 4.4, 820,
    ["Wore them kayaking and hiking, they drain instantly", "Straps rubbed a bit at first", "Good grip on wet rocks"])
add("P010", "footwear", "Apex", "Apex Carbon Plate Racing Shoe",
    "Race-day shoe with a carbon-fiber plate and ultra-light foam. 6.9 oz (196 g) per shoe. Built for speed on the road; not intended for daily training or trail use.",
    189.00, 4.5, 640,
    ["Fast and springy, set a half-marathon PR", "Wears out quickly, best saved for race day", "Firm at slow paces"])
add("P011", "footwear", "PaceLine", "PaceLine Basic Road Running Shoe",
    "No-frills road running shoe with a mesh upper, cushioned EVA midsole and durable rubber outsole. A dependable entry-level trainer for casual joggers.",
    39.99, 4.0, 3200,
    ["Great value for casual jogging", "Comfortable right out of the box", "Feels light enough for 5K runs"])
add("P012", "footwear", "Boreal", "Boreal ThermoGrip Insulated Snow Boot",
    "Waterproof insulated snow boot with a rubber shell, removable felt liner and aggressive traction sole. Rated to -25F. Weighs 3 lb 4 oz (1.47 kg) per pair.",
    109.00, 4.5, 930,
    ["Warm and dry in deep snow", "Heavy but keeps feet warm at -20F", "No leaks after a whole winter"])
add("P013", "footwear", "KidFlex", "KidFlex Light-Up Kids Sneaker",
    "Kids sneaker with LED lights in the heel, hook-and-loop straps and a flexible sole. Sizes toddler through big kid.",
    29.00, 4.0, 1120,
    ["My daughter loves the lights", "Lights stopped working after two months", "Runs a half size small"])

# -------------------------------------------------------------- ELECTRONICS
add("P014", "electronics", "SonicPulse", "SonicPulse Wireless Earbuds Lite",
    "Basic true-wireless earbuds with Bluetooth 5.3, touch controls and a compact charging case. 6 hours per charge, 18 hours with the case. No active noise cancellation. IPX4 splash resistance.",
    24.99, 4.1, 5400,
    ["Solid sound for the price", "Perfect basic earbuds, no frills", "Battery lasts about 5 hours, fine for commuting"])
add("P015", "electronics", "SonicPulse", "SonicPulse Wireless Earbuds Pro",
    "Premium true-wireless earbuds with hybrid active noise cancellation, transparency mode and Bluetooth 5.3. Up to 8 hours per charge and 30 hours with the wireless charging case. IPX4 rated.",
    59.99, 4.6, 3100,
    ["The ANC blocks out the train nicely", "Battery easily lasts a work week", "Best sound at this price and the mic is great for calls"])
add("P016", "electronics", "SonicPulse", "SonicPulse Studio ANC Over-Ear Headphones",
    "Over-ear wireless headphones with adaptive active noise cancellation, 45-hour battery, memory-foam ear cups and multipoint Bluetooth. Folds flat for travel.",
    149.00, 4.7, 1820,
    ["Comfortable for all-day wear", "Noise cancelling is excellent on flights", "Bulky compared to earbuds"])
add("P017", "electronics", "BassLine", "BassLine Buds X Wireless Earbuds",
    "Budget wireless earbuds with boosted bass, Bluetooth 5.0 and a 5-hour battery. Includes a pocket-size charging case.",
    19.99, 3.4, 2100,
    ["Left earbud stopped working after a month", "Bass is good for the price", "Connection drops all the time"])
add("P018", "electronics", "VoltCore", "VoltCore 10000mAh Slim Power Bank",
    "Slim 10,000mAh power bank with USB-C and USB-A ports and 12W output. Weighs 7.4 oz (210 g). Charges most phones about two times.",
    27.99, 4.4, 3800,
    ["Fits in a jeans pocket and charges twice", "Charges slowly compared to my old one", "Solid build"])
add("P019", "electronics", "VoltCore", "VoltCore 20000mAh Fast-Charge Power Bank Pro",
    "20,000mAh power bank with 45W USB-C Power Delivery that can charge laptops, plus a digital battery display. Weighs 14.1 oz (400 g). Three output ports.",
    44.99, 4.6, 2100,
    ["Charged my laptop and phone at the same time", "Heavy but worth it for travel", "Display is handy"])
add("P020", "electronics", "ClearView", "ClearView 27-inch 4K IPS Monitor",
    "27-inch 4K IPS monitor at 60Hz with USB-C 65W power delivery and a height-adjustable stand. Covers 99% sRGB.",
    329.00, 4.5, 1450,
    ["Sharp text and one-cable laptop setup", "Colors look great out of the box", "Speakers are weak, use headphones"])
add("P021", "electronics", "PixelTab", "PixelTab 10 Tablet",
    "10.1-inch tablet with 64GB storage, 10 hours of rated battery life and a quad-core processor. Weighs 1.1 lb (500 g).",
    189.00, 4.2, 2600,
    ["Great for streaming and kids' games", "Battery lasts about 7 hours of video", "Slows down with many apps open"])
add("P022", "electronics", "Orbit", "Orbit Band 5 Fitness Tracker",
    "Fitness tracker with heart-rate and sleep tracking, 5ATM water resistance and up to 14 days of battery. Weighs 0.9 oz (26 g).",
    49.00, 4.3, 4200,
    ["Wore it swimming daily with no issues", "Battery is closer to 10 days", "Step count seems accurate"])
add("P023", "electronics", "Orbit", "Orbit Band 5 Plus GPS Fitness Tracker",
    "Fitness tracker with built-in GPS, SpO2, heart-rate and sleep tracking and 5ATM water resistance. Up to 7 days of battery. Weighs 1.1 oz (31 g).",
    89.00, 4.5, 1900,
    ["GPS locks fast and maps my runs well", "Worth the upgrade for leaving the phone at home", "Battery lasts about a week"])
add("P024", "electronics", "AeroPods", "AeroPods Sport Ear-Hook Wireless Earbuds",
    "Secure ear-hook wireless earbuds made for workouts. Bluetooth 5.2, 9 hours per charge, and three tip sizes in the box.",
    44.99, 4.4, 1750,
    ["Survived sweaty runs and a rainy 10K without any problem", "Stay put during burpees", "Sound is decent, the ear hooks take getting used to"])
add("P025", "electronics", "BoomCube", "BoomCube Waterproof Bluetooth Speaker",
    "Portable Bluetooth speaker with IPX7 waterproofing that floats. 12 hours of battery and a built-in carabiner loop. Weighs 1.2 lb (545 g).",
    39.99, 4.5, 6300,
    ["Dropped it in the pool and it still works", "Loud for its size", "Battery is closer to 9-10 hours at high volume"])
add("P026", "electronics", "ClearTone", "ClearTone USB-C Wired Earphones",
    "Wired in-ear earphones with a USB-C connector and inline microphone. Tangle-resistant cable. Not wireless.",
    12.99, 4.0, 2900,
    ["Good sound for cheap and no charging needed", "Mic is okay for calls", "Cable feels thin"])

# --------------------------------------------------------------------- HOME
add("P027", "home", "CozyNest", "CozyNest Chunky Knit Throw Blanket",
    "Soft chunky-knit throw made of 100% acrylic, 50 x 60 in, machine washable. A cozy layer for the couch or bed and a popular gift.",
    34.99, 4.7, 3400,
    ["So cozy, my mom loved it as a gift", "Sheds a little at first", "Beautiful and warm"])
add("P028", "home", "CozyNest", "CozyNest Microfiber Fleece Blanket",
    "Lightweight microfiber fleece blanket, 50 x 60 in, weighs 1.2 lb (545 g). Soft, wrinkle-resistant and machine washable.",
    19.99, 4.4, 8200,
    ["Great value and very soft", "Thinner than I expected", "Washes well"])
add("P029", "home", "LumaGlow", "LumaGlow Ultrasonic Aroma Diffuser",
    "Ultrasonic essential-oil diffuser with 7 LED colors, a 300 ml tank, auto shut-off and up to 10 hours of runtime.",
    28.00, 4.3, 2500,
    ["Quiet and relaxing", "Runs closer to 6-7 hours on high", "Nice gift for a friend"])
add("P030", "home", "SereneSleep", "SereneSleep Memory Foam Pillow",
    "Contoured memory-foam pillow with a cooling gel layer and a washable bamboo cover. Suits back and side sleepers.",
    39.00, 4.4, 3900,
    ["My neck pain is gone", "Smells odd for the first few days", "Not really cooling"])
add("P031", "home", "BrightNest", "BrightNest LED Desk Lamp",
    "Dimmable LED desk lamp with five color temperatures, USB charging port, touch controls and a flexible arm.",
    32.99, 4.5, 2700,
    ["Bright and adjustable, great for reading", "USB port is handy", "Base is a little light"])
add("P032", "home", "Hearth & Grain", "Hearth & Grain Soy Candle Trio",
    "Three hand-poured soy candles in lavender, vanilla and cedar. About 40 hours of burn time each. Comes gift-boxed.",
    24.00, 4.6, 1300,
    ["Lovely scent that fills the room", "Gave it as a gift and it was a hit", "Burns evenly"])
add("P033", "home", "PureAir", "PureAir HEPA Air Purifier Compact",
    "True HEPA air purifier for rooms up to 250 sq ft with a 24 dB sleep mode and 3-stage filtration. Weighs 6.5 lb (2.9 kg).",
    89.00, 4.4, 2200,
    ["Bedroom air feels fresher and it is very quiet", "Filters are pricey", "Good for allergies"])
add("P034", "home", "PureAir", "PureAir HEPA Air Purifier Max",
    "True HEPA air purifier for rooms up to 700 sq ft with an air-quality sensor, auto mode and app control. Weighs 14 lb (6.4 kg).",
    179.00, 4.6, 1500,
    ["Handles my open-plan living room easily", "Auto mode is smart and quiet", "Large footprint"])
add("P035", "home", "TidyBox", "TidyBox Woven Storage Baskets 3-Pack",
    "Set of three foldable woven fabric storage baskets with handles. Sizes for shelves and closets.",
    26.00, 4.3, 1800,
    ["Look nice on a shelf", "Sturdy enough for toys", "Smaller than the photos suggest"])
add("P036", "home", "SoftTouch", "SoftTouch Cotton Bath Towel Set 6-Piece",
    "Six-piece towel set in 100% cotton, 600 GSM: two bath towels, two hand towels and two washcloths.",
    44.00, 4.5, 2400,
    ["Thick and absorbent", "Lint for the first few washes", "Colors stay bright"])
add("P037", "home", "SwiftVac", "SwiftVac Cordless Stick Vacuum",
    "Cordless stick vacuum with up to 40 minutes of runtime, a HEPA filter and a detachable handheld unit. Weighs 5.5 lb (2.5 kg).",
    149.00, 4.0, 1350,
    ["Strong suction on carpet", "Battery only lasted about 18 minutes on max", "Barely got through my apartment, closer to 20 minutes than 40"])
add("P038", "home", "CalmHug", "CalmHug Weighted Blanket 15 lb",
    "15 lb glass-bead weighted blanket, 48 x 72 in, with a breathable cotton cover. Designed for calming, deeper sleep.",
    69.00, 4.5, 3100,
    ["Helps me fall asleep faster", "Too warm for hot sleepers", "Heavy but calming; a great gift for my sister"])

# ------------------------------------------------------------------- BEAUTY
add("P039", "beauty", "Velvet Dew", "Velvet Dew Hydrating Face Moisturizer",
    "Lightweight daily moisturizer with hyaluronic acid and ceramides for normal to dry skin. 1.7 oz. Fragrance-free.",
    22.00, 4.5, 3600,
    ["Hydrates all day without feeling greasy", "Absorbs quickly under makeup", "Small jar for the price"])
add("P040", "beauty", "Velvet Dew", "Velvet Dew Night Repair Cream",
    "Rich night cream with retinol and peptides for dry and mature skin. 1.7 oz.",
    38.00, 4.6, 1700,
    ["Skin looks smoother after three weeks", "A little irritating at first", "Luxurious texture"])
add("P041", "beauty", "SunVeil", "SunVeil Daily Sunscreen SPF 50",
    "Broad-spectrum SPF 50 lightweight lotion, water resistant for 80 minutes. Reef-friendly formula. 3 oz.",
    16.00, 4.4, 5100,
    ["No white cast and did not sting", "Held up through a swim", "Runs out fast"])
add("P042", "beauty", "Bloomhaus", "Bloomhaus Lavender Spa Gift Set",
    "Gift box with lavender bath salts, body butter, shower gel and pillow mist. A relaxing present for her.",
    42.00, 4.7, 2100,
    ["Bought for my mom and she adored it", "Beautiful packaging", "Scent is calming, not overpowering"])
add("P043", "beauty", "Glow Ritual", "Glow Ritual Vitamin C Serum",
    "15% vitamin C serum with hyaluronic acid for brightening. 1 oz.",
    29.00, 4.2, 4400,
    ["Skin looked brighter after four weeks", "Turned orange and oxidized after two months", "Stings a bit at first"])
add("P044", "beauty", "PureLoc", "PureLoc Sulfate-Free Shampoo",
    "Sulfate-free shampoo for color-treated hair, 16 oz, with coconut oil and keratin.",
    14.00, 4.3, 3000,
    ["Color lasts longer", "Gentle on my scalp", "Doesn't lather much"])
add("P045", "beauty", "SilkWave", "SilkWave Ionic Hair Dryer",
    "1875W ionic hair dryer with three heat settings and a cool shot button. Weighs 1.1 lb (500 g).",
    59.00, 4.4, 2600,
    ["Dries my hair quickly with less frizz", "A bit loud", "Light enough for long styling"])
add("P046", "beauty", "SilkWave", "SilkWave Pro Brushless Ionic Hair Dryer",
    "Brushless-motor ionic hair dryer with diffuser, five heat settings and a 1.0 lb (450 g) body. Runs quieter than standard dryers.",
    129.00, 4.6, 900,
    ["Noticeably quieter and faster", "Diffuser is great for curls", "Pricey but salon quality"])
add("P047", "beauty", "Petal Soft", "Petal Soft Lip Balm Trio",
    "Set of three moisturizing lip balms in mint, vanilla and berry with shea butter.",
    9.00, 4.5, 6000,
    ["Keeps my lips soft all day", "Great stocking stuffer", "Mint is strong"])
add("P048", "beauty", "Aura", "Aura Matte Liquid Lipstick",
    "Long-wear matte liquid lipstick in 12 shades. Transfer-resistant formula.",
    18.00, 4.1, 2100,
    ["Stays put through lunch", "Dries my lips out", "Shade matches the swatch"])
add("P049", "beauty", "ClearSkin", "ClearSkin Gentle Foaming Cleanser",
    "Gentle foaming facial cleanser for sensitive skin, pH balanced, 6.7 oz.",
    13.00, 4.4, 3300,
    ["No scent at all and it works for my sensitive skin", "Doesn't remove waterproof mascara", "Did not irritate my skin"])
add("P050", "beauty", "Ember", "Ember Rose Eau de Parfum",
    "50 ml eau de parfum with notes of rose, amber and musk.",
    58.00, 4.3, 900,
    ["Smells elegant and lasts most of the day", "Gifted to my mother and she wears it daily", "Bottle is smaller than expected"])

# ------------------------------------------------------------------- SPORTS
add("P051", "sports", "FlexCore", "FlexCore Yoga Mat 6mm",
    "6 mm TPE yoga mat, 72 x 24 in, weighs 2.2 lb (1 kg). Textured surface with a carry strap.",
    24.99, 4.4, 4700,
    ["Good cushioning for the price", "Gets slippery with sweaty hands", "Edges curl a little"])
add("P052", "sports", "FlexCore", "FlexCore Pro Non-Slip Yoga Mat 8mm",
    "8 mm natural-rubber yoga mat with alignment lines and a non-slip surface. 72 x 26 in, weighs 4.4 lb (2 kg).",
    54.00, 4.7, 1500,
    ["Stays put even in hot yoga", "Extra cushioning is great for knees", "Heavy to carry"])
add("P053", "sports", "IronPeak", "IronPeak Adjustable Dumbbell Pair 5-52 lb",
    "Pair of adjustable dumbbells from 5 to 52.5 lb each with a quick-dial selector and storage tray.",
    279.00, 4.6, 1200,
    ["Replaces a whole rack in my apartment", "Dial is quick to change", "Handles feel a bit thick"])
add("P054", "sports", "SprintLine", "SprintLine Resistance Bands Set of 5",
    "Set of five latex resistance bands from 10 to 50 lb with door anchor, handles and carry bag.",
    17.00, 4.4, 5200,
    ["Great for travel workouts", "One band snapped after a few months", "Good range of resistance"])
add("P055", "sports", "AquaLap", "AquaLap Anti-Fog Swim Goggles",
    "Swim goggles with anti-fog lenses, UV protection and an adjustable silicone strap.",
    14.00, 4.0, 2200,
    ["Comfortable seal with no leaking", "Fogged up after ten minutes", "Fine for casual laps"])
add("P056", "sports", "CourtKing", "CourtKing Indoor/Outdoor Basketball",
    "Size 7 composite-leather basketball for indoor and outdoor courts with deep channels for grip.",
    27.00, 4.5, 3100,
    ["Great grip and holds air", "Handled outdoor concrete well", "A bit slick out of the box"])
add("P057", "sports", "ZenRoll", "ZenRoll Foam Roller 18 in",
    "High-density 18-inch foam roller for muscle recovery. Weighs 1.0 lb (450 g).",
    21.00, 4.5, 2800,
    ["Firm and works for sore quads", "Holds shape after months", "Very firm for beginners"])
add("P058", "sports", "SwiftRide", "SwiftRide Road Cycling Helmet",
    "Road cycling helmet with MIPS, 18 vents and an adjustable dial fit. Weighs 9.5 oz (270 g).",
    64.00, 4.5, 1900,
    ["Light and well ventilated", "Fit dial is easy to use", "Visor sold separately"])
add("P059", "sports", "HydroFlow", "HydroFlow Insulated Water Bottle 32 oz",
    "Double-wall stainless steel bottle that keeps drinks cold for 24 hours or hot for 12. Leakproof lid. 32 oz.",
    29.00, 4.5, 5100,
    ["Ice still frozen the next morning", "Leaks when tipped over in my bag", "Fits my cup holder"])
add("P060", "sports", "PaceMate", "PaceMate Lightweight Running Belt",
    "Lightweight stretch running belt with a zip pocket for a phone and keys. Adjustable waist strap. Weighs 2.1 oz (60 g).",
    16.00, 4.3, 3300,
    ["Doesn't bounce while I run", "Kept my phone dry on a rainy run", "Fits my large phone"])
add("P061", "sports", "GripMax", "GripMax Speed Jump Rope",
    "Adjustable speed jump rope with ball-bearing handles and a steel cable.",
    11.00, 4.4, 4100,
    ["Spins smoothly", "Good for double-unders", "Cable frays over time"])
add("P062", "sports", "StrideWear", "StrideWear Quick-Dry Running Shorts",
    "Quick-dry 2-in-1 running shorts with a zip pocket and reflective trim. Weighs 4.5 oz (128 g).",
    28.00, 4.3, 1600,
    ["Comfortable and no chafing", "Pocket fits my phone", "Runs slightly small"])

# ------------------------------------------------------------------ KITCHEN
add("P063", "kitchen", "BrewMate", "BrewMate Electric Kettle 1.7L",
    "1.7L stainless steel electric kettle, 1500W, with auto shut-off, boil-dry protection and a cordless 360-degree base.",
    34.99, 4.6, 6100,
    ["Boils fast and shuts off reliably", "Spout pours cleanly", "Lid is a little stiff"])
add("P064", "kitchen", "BrewMate", "BrewMate Precision Variable-Temp Kettle",
    "1.0L gooseneck kettle with adjustable temperatures from 140F to 212F, a one-hour keep-warm function and a precision pour spout.",
    69.00, 4.6, 1600,
    ["Perfect for pour-over and green tea", "Keep-warm mode is convenient", "Smaller capacity than my old kettle"])
add("P065", "kitchen", "BlendRight", "BlendRight Personal Blender 600W",
    "600W personal blender with two 20 oz to-go cups. Blades handle fruit and soft ingredients.",
    39.00, 4.2, 3900,
    ["Perfect for morning smoothies", "Struggles with ice", "Easy to clean"])
add("P066", "kitchen", "BlendRight", "BlendRight Power Blender 1200W",
    "1200W countertop blender with a 64 oz pitcher that crushes ice, plus a self-clean cycle and variable speed dial.",
    119.00, 4.6, 2000,
    ["Crushes ice and makes silky soups", "Loud but powerful", "Big pitcher for family batches"])
add("P067", "kitchen", "IronSear", "IronSear Pre-Seasoned Cast Iron Skillet 10 in",
    "10-inch pre-seasoned cast iron skillet, oven safe to 500F. Weighs 5 lb (2.3 kg). Works on gas, induction and campfires.",
    32.00, 4.7, 7200,
    ["Amazing sear and lasts forever", "Heavy but that's cast iron", "Seasoning improved with use"])
add("P068", "kitchen", "CeraChef", "CeraChef Ceramic Nonstick Pan Set 3-Piece",
    "PFOA-free ceramic nonstick 3-piece pan set with a durable long-lasting coating. Induction compatible, dishwasher safe.",
    59.00, 4.1, 2800,
    ["Eggs slide right off at first", "The coating started peeling after four months", "The nonstick wore out quickly, not durable"])
add("P069", "kitchen", "ChefLine", "ChefLine 8-inch Chef Knife",
    "8-inch high-carbon stainless steel chef knife with a full tang and pakkawood handle. Weighs 8.1 oz (230 g).",
    44.00, 4.6, 3000,
    ["Razor sharp out of the box", "Well balanced", "Not dishwasher safe"])
add("P070", "kitchen", "FreshLock", "FreshLock Glass Food Containers 10-Piece",
    "Ten-piece glass food storage set with snap-lock lids. Microwave, oven and dishwasher safe.",
    29.00, 4.5, 3800,
    ["Lids seal well and no stains", "Heavier than plastic", "Good size variety"])
add("P071", "kitchen", "CrispWave", "CrispWave Air Fryer 4 qt",
    "4-quart digital air fryer with 8 presets and a nonstick dishwasher-safe basket. Suits one to two people.",
    69.00, 4.5, 8800,
    ["Crisps fries with almost no oil", "Compact and easy to clean", "Basket is small for a family"])
add("P072", "kitchen", "CrispWave", "CrispWave Dual-Basket Air Fryer 8 qt",
    "8-quart air fryer with two independent baskets and a sync-finish mode so two foods cook at once. Nonstick dishwasher-safe baskets.",
    129.00, 4.6, 2400,
    ["Cook chicken and fries at the same time", "Big enough for family dinners", "Takes a lot of counter space"])
add("P073", "kitchen", "PourPerfect", "PourPerfect Pour-Over Coffee Gift Set",
    "Glass pour-over dripper with carafe, reusable stainless filter and measuring scoop, packed in a gift box.",
    36.00, 4.6, 900,
    ["Beautiful set and makes great coffee", "Gift for my mom and she uses it daily", "Glass is fragile"])
add("P074", "kitchen", "SnapLid", "SnapLid Reusable Silicone Bag Set",
    "Set of six reusable silicone food storage bags in assorted sizes. Freezer and dishwasher safe.",
    15.00, 4.3, 2500,
    ["Replaced my plastic bags", "Hard to dry inside", "Seal is reliable"])

# ------------------------------------------------------------------ OUTDOOR
add("P075", "outdoor", "Ridgeline", "Ridgeline 2-Person Backpacking Tent",
    "Freestanding 3-season two-person tent with a waterproof rain fly and taped seams. Weighs 4 lb 6 oz (1.99 kg) packed.",
    159.00, 4.5, 1900,
    ["Stayed dry through an overnight thunderstorm", "Easy setup in five minutes", "Tight for two adults with gear"])
add("P076", "outdoor", "TrailMate", "TrailMate 40L Hiking Backpack",
    "40L daypack with a ventilated back panel, padded hip belt, hydration sleeve and an included rain cover. Weighs 2 lb 3 oz (1 kg).",
    79.00, 4.4, 2300,
    ["Comfortable on all-day hikes", "Rain cover works well", "Zippers feel a bit flimsy"])
add("P077", "outdoor", "TrailMate", "TrailMate Pro 55L Backpacking Pack",
    "55L backpacking pack with an adjustable suspension frame, load-lifter straps and an included rain cover. Weighs 3 lb 5 oz (1.5 kg).",
    139.00, 4.6, 1100,
    ["Carries 35 lb comfortably for days", "Great fit adjustments", "Overkill for day trips"])
add("P078", "outdoor", "Emberlight", "Emberlight Rechargeable Camping Lantern",
    "USB-rechargeable camping lantern with 500 lumens, three modes and up to 12 hours of runtime on low. Weighs 10.6 oz (300 g).",
    19.99, 4.5, 3300,
    ["Bright enough for the whole campsite", "Charges quickly", "Handle is a little short"])
add("P079", "outdoor", "SummitWarm", "SummitWarm 20F Mummy Sleeping Bag",
    "Mummy sleeping bag rated to 20F with synthetic insulation and a compression sack. Weighs 3 lb 2 oz (1.4 kg).",
    89.00, 4.4, 2000,
    ["Warm and packs small", "Comfortable down to about 30F, not 20F", "Zipper snags sometimes"])
add("P080", "outdoor", "RainShield", "RainShield Packable Rain Jacket",
    "Packable waterproof rain jacket with taped seams and an adjustable hood. Stuffs into its own pocket. Weighs 8.5 oz (241 g).",
    59.00, 4.2, 1700,
    ["Kept me dry in light rain and drizzle", "Soaked through in a heavy downpour after about an hour", "Packs down tiny"])
add("P081", "outdoor", "PathFinder", "PathFinder Carbon Trekking Poles",
    "Pair of carbon-fiber trekking poles with cork grips and flip-lock adjustment. Weighs 15 oz (425 g) per pair.",
    59.00, 4.5, 1400,
    ["Light and strong on steep descents", "Locks hold firm", "Tips wear out fast"])
add("P082", "outdoor", "AquaPure", "AquaPure Portable Water Filter",
    "0.1-micron hollow-fiber water filter that treats up to 1,000 L. Weighs 2 oz (57 g). Screws onto standard bottles.",
    34.00, 4.5, 3600,
    ["Made stream water taste clean", "Flow slows if you don't backflush", "Tiny and light"])
add("P083", "outdoor", "CampChef", "CampChef Compact Canister Camp Stove",
    "Compact canister camp stove that boils 1L of water in about 3.5 minutes. Weighs 3 oz (85 g) with a folding pot support.",
    29.00, 4.4, 2700,
    ["Boils fast and packs tiny", "Wobbly with large pots", "Wind reduces performance a lot"])
add("P084", "outdoor", "SunGuard", "SunGuard Wide-Brim Sun Hat UPF 50+",
    "Wide-brim hiking sun hat with UPF 50+ fabric, chin cord and a packable crown.",
    24.00, 4.3, 1900,
    ["Great shade on a long hike", "Packs flat in my bag", "Runs slightly large"])
add("P085", "outdoor", "Beacon", "Beacon 300-Lumen Rechargeable Headlamp",
    "Rechargeable headlamp with 300 lumens, a red-light mode and a tilting head. Weighs 3 oz (85 g).",
    22.00, 4.4, 2900,
    ["Worked fine through a full downpour on a night hike", "Bright and comfortable", "Battery lasts about 4 hours on high"])
add("P086", "outdoor", "CoolTrek", "CoolTrek 30-Quart Hard Cooler",
    "30-quart hard-sided cooler with rotomolded construction that keeps ice for up to 3 days. Weighs 12 lb (5.4 kg).",
    59.00, 4.3, 1500,
    ["Very tough and bear-resistant looking", "Ice was gone in a day and a half in a hot car", "Heavy when full"])

# --------------------------------------------------------------------- TOYS
add("P087", "toys", "BrickBuilders", "BrickBuilders 500-Piece Creative Set",
    "500-piece building brick set with a storage tub and idea booklet. Recommended for ages 6+.",
    29.99, 4.7, 3900,
    ["Hours of creative play", "Bricks fit well with other sets", "A few pieces were missing"])
add("P088", "toys", "PlushPals", "PlushPals Cuddle Bear 16 in",
    "Soft 16-inch plush bear with embroidered eyes. Machine washable. Suitable from birth.",
    18.00, 4.6, 2700,
    ["Super soft and my baby loves it", "Washes without losing shape", "Slightly smaller than expected"])
add("P089", "toys", "PuzzleQuest", "PuzzleQuest 1000-Piece Landscape Puzzle",
    "1000-piece jigsaw puzzle of a mountain landscape on thick recycled board. Finished size 27 x 20 in.",
    16.99, 4.5, 1800,
    ["Pieces are thick and fit snugly", "Fun family project", "Image is a bit dark"])
add("P090", "toys", "RoboTinker", "RoboTinker STEM Robot Building Kit",
    "STEM kit that builds six different robots, no soldering required. Recommended for ages 8+. Needs 4 AA batteries (not included).",
    44.99, 4.0, 850,
    ["Instructions were unclear and my kid needed help", "Kids loved it once built", "Missing screws in one kit"])
add("P091", "toys", "SplashZone", "SplashZone Water Blaster",
    "Large-capacity water blaster with a 1L tank and 30-foot range. For ages 5+.",
    14.00, 4.2, 2900,
    ["Perfect for summer backyard fun", "Trigger stiff at first", "Leaks a little at the cap"])
add("P092", "toys", "TinyTots", "TinyTots Wooden Stacking Blocks",
    "Set of 30 wooden stacking blocks painted with non-toxic water-based paint. Suitable for ages 1+.",
    23.00, 4.7, 1600,
    ["Sturdy and my toddler loves them", "Smooth edges", "Paint chipped slightly on one block"])
add("P093", "toys", "RC Racer", "RC Racer Stunt Car",
    "2.4GHz remote-control stunt car with 20 minutes of battery per charge, 360-degree flips and rechargeable battery. Recommended for ages 8+.",
    39.00, 4.3, 2300,
    ["Fast and survives crashes", "Battery only lasts 15 minutes", "Fun for the price"])
add("P094", "toys", "RC Racer", "RC Racer Stunt Car Pro",
    "2.4GHz remote-control stunt car with two rechargeable batteries for 40 minutes of play, all-terrain tires and speeds up to 20 mph. Recommended for ages 8+.",
    69.00, 4.6, 1200,
    ["Tires grip on grass and gravel", "Two batteries doubled our playtime", "Noticeably faster than the basic model"])
add("P095", "toys", "GameNight", "GameNight Family Party Board Game",
    "Family party board game for 2 to 8 players. Recommended for ages 10+. Plays in about 30 minutes.",
    27.00, 4.6, 2100,
    ["Big hit at our family dinners", "Easy rules", "Could use more cards"])
add("P096", "toys", "ArtBox", "ArtBox Kids Washable Painting Set",
    "Washable painting set with 12 colors, three brushes and a mixing tray. Recommended for ages 3+.",
    19.00, 4.4, 1700,
    ["Colors are vibrant and washed out of clothes", "Brushes are cheap", "Good starter set"])
add("P097", "toys", "Starlight", "Starlight Dream Projector Night Light",
    "Night light that projects stars and moons onto the ceiling, with a timer and three colors. USB powered.",
    21.00, 4.3, 3200,
    ["My kids love falling asleep to it", "Projection is dim in a bright room", "Timer is handy"])
add("P098", "toys", "Little Chef", "Little Chef Wooden Play Kitchen",
    "Wooden play kitchen with a stove, sink, oven and storage shelves. Assembly required. Weighs 22 lb (10 kg). Recommended for ages 3+.",
    59.00, 4.4, 1300,
    ["Took two hours to assemble but sturdy", "Kids play with it daily", "Bulky for small rooms"])

# ---------------------------------------------------------------- ONE-OFFS
add("P099", "pets", "PawPal", "PawPal Orthopedic Dog Bed Medium",
    "Orthopedic memory-foam dog bed with a removable washable cover and non-slip base. Fits dogs up to 50 lb.",
    49.00, 4.6, 2700,
    ["My senior dog sleeps better", "Cover washes well", "Runs a little small"])
add("P100", "office", "DeskMate", "DeskMate Adjustable Laptop Stand",
    "Aluminum laptop stand with six height settings that folds flat. Fits laptops from 10 to 16 inches. Weighs 1.0 lb (450 g).",
    29.99, 4.5, 4300,
    ["Sturdy and fixed my neck strain", "Folds flat for travel", "Rubber pads fell off after a while"])


# ------------------------------------------------------- PLANTED TEST TRAITS
# Ground truth for tests/documentation ONLY. The pipeline must never import this.
TRAITS = {
    "P001": ["strong_match", "budget_conflict", "only_fully_waterproof_running_shoe", "scenario_A_top_pick"],
    "P002": ["partial_match", "missing_attribute:waterproof", "review_negative_evidence:water", "scenario_A_secondary"],
    "P003": ["review_only_negative:rain", "scenario_C_trap", "multiple_agreeing_reviews"],
    "P004": ["good_alternative_for_rain", "scenario_C_recommended_alternative"],
    "P005": ["explicit_not_waterproof", "over_typical_budget"],
    "P006": ["contradiction:desc_waterproof_vs_reviews_leak", "two_agreeing_negative_reviews", "one_hedged_positive"],
    "P007": ["lone_contradicting_review_should_not_overturn_description"],
    "P008": ["review_only_negative:rain_mild"],
    "P010": ["pricier_decoy_no_relevant_benefit_for_budget_shopper"],
    "P011": ["missing_attribute:weight", "missing_attribute:waterproof", "weak_review_signal:lightweight"],
    "P014": ["scenario_D_primary", "cheap_basic"],
    "P015": ["scenario_D_valid_upsell", "upsell_benefit:ANC+battery"],
    "P016": ["different_form_factor_not_a_natural_upsell_for_earbuds"],
    "P017": ["cheap_but_low_quality_decoy", "rating_below_primary"],
    "P018": ["upsell_family:power_bank_basic"],
    "P019": ["upsell_family:power_bank_pro", "upsell_benefit:laptop_charging+capacity"],
    "P022": ["upsell_family:tracker_basic", "review_contradicts_spec:battery_days"],
    "P023": ["upsell_family:tracker_gps", "upsell_benefit:GPS"],
    "P024": ["review_only_positive:sweat_and_rain", "description_silent_on_water_resistance"],
    "P025": ["explicit_waterproof_IPX7", "minor_review_contradiction:battery"],
    "P026": ["not_wireless_should_not_match_wireless_queries"],
    "P027": ["gift_candidate", "cozy"],
    "P028": ["cozy_budget_alternative"],
    "P033": ["upsell_family:purifier_compact"],
    "P034": ["upsell_family:purifier_large", "upsell_benefit:room_coverage"],
    "P037": ["contradiction:battery_minutes_vs_reviews", "two_agreeing_negative_reviews"],
    "P038": ["gift_candidate", "cozy"],
    "P042": ["gift_candidate:mom"],
    "P043": ["lone_negative_review:oxidation"],
    "P045": ["upsell_family:dryer_standard", "lightweight_stated"],
    "P046": ["upsell_family:dryer_pro", "upsell_benefit:quieter+diffuser"],
    "P049": ["review_only_positive:fragrance_free", "description_silent_on_fragrance"],
    "P050": ["gift_candidate:mom"],
    "P051": ["review_only_negative:slippery", "upsell_family:yoga_mat_basic"],
    "P052": ["upsell_family:yoga_mat_pro", "upsell_benefit:non_slip"],
    "P055": ["contradiction:anti_fog_claim_vs_review"],
    "P059": ["lone_negative_review:leak"],
    "P060": ["review_only_positive:rain_dry"],
    "P063": ["upsell_family:kettle_basic"],
    "P064": ["upsell_family:kettle_variable_temp", "upsell_benefit:temperature_control"],
    "P065": ["upsell_family:blender_personal"],
    "P066": ["upsell_family:blender_power", "upsell_benefit:crushes_ice"],
    "P068": ["contradiction:durable_claim_vs_reviews_peeling", "two_agreeing_negative_reviews"],
    "P071": ["upsell_family:air_fryer_4qt"],
    "P072": ["upsell_family:air_fryer_8qt_dual", "upsell_benefit:two_baskets"],
    "P073": ["gift_candidate:mom"],
    "P075": ["waterproof_stated:rain_fly"],
    "P076": ["upsell_family:pack_40L", "budget_conflict_for_cheap_backpack_queries"],
    "P077": ["upsell_family:pack_55L", "upsell_benefit:frame+capacity"],
    "P079": ["contradiction:temperature_rating_vs_review"],
    "P080": ["hedged_contradiction:waterproof_holds_in_light_rain_only"],
    "P085": ["review_only_positive:waterproof", "description_silent_on_waterproof"],
    "P086": ["contradiction:ice_retention_vs_review"],
    "P093": ["upsell_family:rc_car_basic"],
    "P094": ["upsell_family:rc_car_pro", "upsell_benefit:2_batteries+faster"],
    "P099": ["off_category_one_off:pets"],
    "P100": ["off_category_one_off:office", "no_standing_desk_in_catalog"],
}

QUERY_PLAYBOOK = [
    {"id": "Q1_specific", "query": "I need a lightweight waterproof running shoe, ideally under $80.",
     "expect": "P001 top pick (over budget, flagged); P002 secondary (not waterproof per reviews); P006 flagged as review contradiction"},
    {"id": "Q2_budget", "query": "Cheap running shoes under $45.",
     "expect": "P011 top; unknown weight/waterproof surfaced as unknown, not false"},
    {"id": "Q3_ambiguous", "query": "Something nice for my mom.",
     "expect": "clarification, no ranked list; options drawn from catalog categories"},
    {"id": "Q4_review_only", "query": "I want a casual sneaker that won't fall apart in light rain.",
     "expect": "P003 flagged/down-ranked due to reviews; P004 offered as alternative"},
    {"id": "Q5_upsell", "query": "Looking for cheap wireless earbuds, just need something basic.",
     "expect": "P014 primary; P015 as brief optional upsell (ANC + 30h); P017 not preferred (low rating); P026 excluded (wired)"},
    {"id": "Q6_no_match", "query": "I need a standing desk for my home office.",
     "expect": "no strong match; honest message; optionally P100 as loosely related, labeled poor"},
    {"id": "Q7_budget_conflict", "query": "A 40L hiking backpack under $40.",
     "expect": "P076 over budget, flagged; no silent hiding"},
    {"id": "Q8_review_positive", "query": "Wireless earbuds for running that can handle sweat and rain.",
     "expect": "P024 supported by reviews only (uncertain, communicated); P014/P015 IPX4 stated"},
    {"id": "Q9_contradiction", "query": "Waterproof running shoe under $80 that's really lightweight.",
     "expect": "P006 shown with contradiction warning rather than confident match"},
    {"id": "Q10_no_upsell", "query": "I want the best air purifier for a large open-plan living room.",
     "expect": "P034 primary; no upsell because primary already top of family"},
    {"id": "Q11_lone_review", "query": "Waterproof hiking boot.",
     "expect": "P007 still recommended; lone leak review surfaced softly, not treated as truth"},
]


def validate(rows):
    ids = [r[0] for r in rows]
    assert len(rows) == 100, f"expected 100 rows, got {len(rows)}"
    assert len(set(ids)) == 100, "duplicate product_ids"
    assert set(TRAITS) <= set(ids), "manifest references unknown product ids"
    titles = [r[3] for r in rows]
    assert len(set(titles)) == 100, "duplicate titles"
    for pid, cat, brand, title, desc, price, rating, n, reviews in rows:
        assert price > 0 and 1.0 <= rating <= 5.0 and n > 0, pid
        assert 3 <= len(reviews) <= 4, f"{pid} needs 3-4 review snippets"
        assert all(" | " not in r for r in reviews + [desc]), f"{pid} contains the separator"
        assert desc.strip() and title.strip() and brand.strip(), pid


def main():
    validate(P)
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(COLUMNS)
        for pid, cat, brand, title, desc, price, rating, n, reviews in P:
            w.writerow([pid, cat, brand, title, desc, f"{price:.2f}", rating, n, " | ".join(reviews)])
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "warning": "SYNTHETIC development data. Not the original assignment dataset. "
                       "The recommendation pipeline must never read this file; it is for tests and docs only.",
            "traits": TRAITS,
            "query_playbook": QUERY_PLAYBOOK,
        }, f, indent=2)
    print(f"Wrote {CSV_PATH.name}: {len(P)} rows")
    print(f"Wrote {MANIFEST_PATH.name}: {len(TRAITS)} annotated products, {len(QUERY_PLAYBOOK)} playbook queries")


if __name__ == "__main__":
    main()
