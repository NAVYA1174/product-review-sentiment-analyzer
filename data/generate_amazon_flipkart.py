"""
Amazon and Flipkart e-commerce review dataset generator.
Creates real-world formatted datasets mimicking Amazon and Flipkart product reviews.
"""

import pandas as pd
import random
from datetime import datetime, timedelta

def generate_amazon_flipkart_datasets():
    # 1. Amazon Reviews Dataset
    amazon_reviews = [
        ("B09X1K87G2", "Apple iPhone 15 Pro Max (256 GB) - Natural Titanium", "Electronics", "Unmatched camera quality and all-day battery life",
         "Upgraded from an older device and the difference is night and day. The 5x telephoto optical zoom is incredibly sharp, and the action button is super convenient. Battery easily lasts 1.5 days on heavy use. The titanium finish feels premium and lightweight.", 5, True, 142, "Amazon", "2026-08-15"),
        
        ("B09X1K87G2", "Apple iPhone 15 Pro Max (256 GB) - Natural Titanium", "Electronics", "Heating issue while fast charging and gaming",
         "The phone gets noticeably warm on the back glass when playing graphic-intensive games or while charging with a 30W adapter. iOS updates mitigated it somewhat, but for this price tag, thermal throttling shouldn't happen.", 3, True, 88, "Amazon", "2026-08-22"),

        ("B09X1K87G2", "Apple iPhone 15 Pro Max (256 GB) - Natural Titanium", "Electronics", "Arrived with scratched screen in unsealed box! Terrible delivery",
         "Ordered from a verified Amazon seller, but the packaging box seal was already broken and the display had noticeable scratches near the dynamic island. Amazon customer support took 5 days to initiate a replacement.", 1, True, 210, "Amazon", "2026-08-29"),

        ("B08N5WRWNW", "Sony WH-1000XM5 Wireless Noise Cancelling Headphones", "Electronics", "Industry leading active noise cancellation",
         "The ANC on these Sony headphones is pure magic. Completely drowns out engine rumble on flights and noisy office air conditioning. Multipoint connection between laptop and phone switches seamlessly.", 5, True, 95, "Amazon", "2026-08-10"),

        ("B08N5WRWNW", "Sony WH-1000XM5 Wireless Noise Cancelling Headphones", "Electronics", "Earcups do not fold anymore, bulky travel case",
         "Sound quality and microphone are top-tier, but removing the folding hinge design was a huge downgrade from the XM4. The carrying case takes up too much backpack space during travel.", 3, True, 43, "Amazon", "2026-08-19"),

        ("B08N5WRWNW", "Sony WH-1000XM5 Wireless Noise Cancelling Headphones", "Electronics", "Headband snapping issue after 4 months",
         "Treated these like gold, yet the plastic joint on the right headband suddenly snapped while putting them on. Sony warranty claimed it was customer wear and tear and refused free repair. Very disappointing.", 1, True, 114, "Amazon", "2026-09-02"),

        ("B07XJ8C8F5", "Instant Pot Duo 7-in-1 Electric Pressure Cooker (6 Qt)", "Home & Kitchen", "Changed our weeknight dinner routine forever",
         "Cuts cooking time by 70%. Risotto, pulled pork, and hearty soups turn out tender and flavorful with zero babysitting. Stainless steel inner pot is easy to wash.", 5, True, 167, "Amazon", "2026-07-20"),

        ("B07XJ8C8F5", "Instant Pot Duo 7-in-1 Electric Pressure Cooker (6 Qt)", "Home & Kitchen", "C8 Error code and burn message constantly",
         "Keeps throwing food burn warning even when there is plenty of broth. Lid sealing gasket also retains food odors that never wash out completely.", 2, True, 52, "Amazon", "2026-08-05"),

        ("B09V3HBK6L", "Kindle Paperwhite (16 GB) – 6.8\" display with adjustable warm light", "Electronics", "Absolute perfection for book lovers",
         "Glare-free screen looks like real paper under direct sunlight. Battery lasts weeks without charging and the warm light makes bedtime reading effortless on the eyes.", 5, True, 130, "Amazon", "2026-08-12"),

        ("B09V3HBK6L", "Kindle Paperwhite (16 GB) – 6.8\" display with adjustable warm light", "Electronics", "Glitchy touchscreen and slow page turns after update",
         "Recent software update made the page turn animations sluggish. Occasionally taps don't register or it jumps two pages ahead. Build quality is nice though.", 3, True, 27, "Amazon", "2026-09-01"),

        ("B07W95D5V3", "Logitech MX Master 3S Wireless Performance Mouse", "Computers", "Ergonomic bliss and whisper-quiet clicks",
         "The MagSpeed electromagnetic scroll wheel is addicting. Fits naturally in the palm, curing my wrist strain. Custom thumb gestures in Logitech Options app enhance productivity.", 5, True, 178, "Amazon", "2026-08-08"),

        ("B07W95D5V3", "Logitech MX Master 3S Wireless Performance Mouse", "Computers", "Rubber coating started melting and feeling sticky",
         "After one year, the soft touch thumb grip degraded into a sticky gummy mess. Bluetooth connectivity also stutters on M2 MacBooks. Not worth $100.", 2, True, 69, "Amazon", "2026-08-25")
    ]

    # Synthesize additional Amazon reviews to 50+
    base_dt = datetime(2026, 9, 22)
    for i in range(40):
        asin, prod, cat, t, txt, r, ver, h, plat, _ = random.choice(amazon_reviews)
        dt = (base_dt - timedelta(days=random.randint(1, 60))).strftime("%Y-%m-%d")
        amazon_reviews.append((asin, prod, cat, t, txt, r, ver, random.randint(10, 80), plat, dt))

    df_amz = pd.DataFrame(amazon_reviews, columns=[
        "asin", "product_name", "category", "review_title", "review_text", "star_rating", "verified_purchase", "helpful_votes", "platform", "review_date"
    ])
    df_amz.to_csv(r"C:\Users\ADMIN\.gemini\antigravity\scratch\product-review-sentiment-analyzer\data\amazon_reviews.csv", index=False)
    print(f"Generated amazon_reviews.csv with {len(df_amz)} records.")

    # 2. Flipkart Reviews Dataset
    flipkart_reviews = [
        ("Samsung Galaxy S24 Ultra 5G (Titanium Gray, 256 GB)", "Mobiles", "Display and AI features are unreal!",
         "The flat anti-reflective Gorilla Armor display is unbelievable outdoors. Circle to Search and live call translation work like magic. The S-Pen integration is super smooth. Value for money flagship!", 5, True, 195, "Flipkart", "2026-08-11"),

        ("Samsung Galaxy S24 Ultra 5G (Titanium Gray, 256 GB)", "Mobiles", "Camera shutter lag and grainy low light photos",
         "Moving pets or kids always come out blurry due to indoor shutter delay. For a 1 lakh+ rupee phone, camera processing should be sharper. Battery is good though.", 3, True, 61, "Flipkart", "2026-08-23"),

        ("Samsung Galaxy S24 Ultra 5G (Titanium Gray, 256 GB)", "Mobiles", "Worst Flipkart delivery! Open box delivery was refused",
         "Delivery partner refused to allow open box inspection and forced OTP verification first. Found dent on corner frame. Customer service support has been running me in circles for 10 days!", 1, True, 180, "Flipkart", "2026-09-04"),

        ("boAt Airdopes 141 Bluetooth Truly Wireless in Ear Earbuds", "Audio", "Paisa vasool! Bass is thumping and battery is solid",
         "Best earbuds under 1500 rupees. Beast mode low latency works great for casual gaming. 42 hours total playtime with case is legitimately accurate. Very happy with Flipkart purchase!", 5, True, 340, "Flipkart", "2026-08-02"),

        ("boAt Airdopes 141 Bluetooth Truly Wireless in Ear Earbuds", "Audio", "Left bud stopped pairing after 20 days",
         "Only right earbud plays sound now. Followed all reset instructions on manual but left side shows no light. Flipkart return window closed after 7 days, boAt service center is 30km away.", 1, True, 125, "Flipkart", "2026-08-27"),

        ("boAt Airdopes 141 Bluetooth Truly Wireless in Ear Earbuds", "Audio", "Mic is strictly average for noisy outdoor calls",
         "Good for music and video playback indoors. But on bike or bus, mic captures traffic noise and listener can't hear voice clearly. Decent for the low price.", 3, True, 45, "Flipkart", "2026-09-08"),

        ("PUMA Nitro Running Shoes", "Fashion", "Extremely bouncy and lightweight",
         "Super comfortable for daily 5km jogs. The nitrogen-infused foam gives awesome rebound. Looks stylish with jeans too. Original genuine product received from Flipkart.", 5, True, 82, "Flipkart", "2026-07-29"),

        ("PUMA Nitro Running Shoes", "Fashion", "Sole rubber detached after 3 weeks of walking",
         "Defective piece or counterfeit batch! The outer tread started peeling off from the heel. Very cheap glue quality. Requested refund immediately.", 1, True, 74, "Flipkart", "2026-08-18"),

        ("realme 55 inch 4K Ultra HD Smart LED TV", "Appliances", "Cinematic Dolby Vision and punchy speakers",
         "Picture quality is crisp with vibrant colors and deep contrast. 24W quad stereo speakers are loud enough without needing an external soundbar. Installation was done within 24 hours.", 5, True, 110, "Flipkart", "2026-08-06"),

        ("realme 55 inch 4K Ultra HD Smart LED TV", "Appliances", "Display backlight bleeding and software lag",
         "Noticeable white light bleeding from the bottom corners during dark scenes. Android TV interface stutters when switching between Netflix and Prime Video. Average experience.", 2, True, 58, "Flipkart", "2026-08-30")
    ]

    for i in range(40):
        prod, cat, t, txt, r, ver, h, plat, _ = random.choice(flipkart_reviews)
        dt = (base_dt - timedelta(days=random.randint(1, 60))).strftime("%Y-%m-%d")
        flipkart_reviews.append((prod, cat, t, txt, r, ver, random.randint(10, 90), plat, dt))

    df_flp = pd.DataFrame(flipkart_reviews, columns=[
        "product_name", "category", "review_title", "review_text", "star_rating", "certified_buyer", "helpful_votes", "platform", "review_date"
    ])
    df_flp.to_csv(r"C:\Users\ADMIN\.gemini\antigravity\scratch\product-review-sentiment-analyzer\data\flipkart_reviews.csv", index=False)
    print(f"Generated flipkart_reviews.csv with {len(df_flp)} records.")

if __name__ == "__main__":
    generate_amazon_flipkart_datasets()
