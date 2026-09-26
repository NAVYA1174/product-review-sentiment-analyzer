"""
Dataset generator for Product Review Sentiment Analyzer.
Generates:
1. sample_reviews.csv (Diverse multi-category product reviews)
2. earphone_reviews.csv (Detailed audio product reviews for ABSA)
3. smartwatch_reviews.csv (Comparative reviews of two flagship smartwatches)
"""

import pandas as pd
import random
from datetime import datetime, timedelta

def create_sample_reviews():
    reviews = [
        # Electronics - Audio & Headphones
        ("AuraSound Pro ANC Wireless Earbuds", "Electronics", "Outstanding active noise cancellation and punchy bass",
         "I have tested dozens of wireless earbuds, but these completely blow away expectations. The active noise cancellation blocks out train noise seamlessly, and the soundstage is wide with rich, clear bass. Battery life lasted a solid 7.5 hours on a single charge. Very comfortable for long flights!", 5, True, 34, "2026-08-14"),
        
        ("AuraSound Pro ANC Wireless Earbuds", "Electronics", "Decent sound but the microphone is awful for calls",
         "Music reproduction is fairly crisp and balanced, but whenever I take work calls on Zoom or mobile, the person on the other end says I sound muffled and distant. Also had one connectivity drop. Average overall.", 3, True, 12, "2026-08-20"),

        ("AuraSound Pro ANC Wireless Earbuds", "Electronics", "Left earbud stopped charging after 3 weeks! Terrible quality",
         "Extremely disappointed. Out of nowhere, the left earbud refused to charge inside the case. I tried cleaning the contact pins and resetting firmware, but nothing worked. Customer support was slow and unhelpful. Returning for a full refund.", 1, True, 89, "2026-08-25"),

        ("AuraSound Pro ANC Wireless Earbuds", "Electronics", "Great value for money compared to high-end brands",
         "For half the price of the big fruit company's buds, you get 90% of the acoustic quality. The companion app equalizer allows custom presets which gave me the perfect acoustic profile. Very happy with this purchase.", 5, True, 19, "2026-09-02"),

        ("AuraSound Pro ANC Wireless Earbuds", "Electronics", "Fit is a bit loose during workouts",
         "The sound signature is warm and pleasant with nice treble. However, when running on the treadmill, they tend to slip out slowly even with the largest silicone tips. Good for desk work, not great for sports.", 3, False, 5, "2026-09-05"),

        # Electronics - Smartwatches
        ("PulseFit Apex Smartwatch", "Electronics", "The best health & fitness tracker I have ever owned",
         "Accurate heart rate tracking, ECG works reliably, and sleep stage analysis gives genuinely actionable tips. The AMOLED display is bright enough under direct sunlight, and the battery easily lasts 5 full days. Premium titanium build quality.", 5, True, 45, "2026-07-10"),

        ("PulseFit Apex Smartwatch", "Electronics", "Step counter is inaccurate and counts hand gestures",
         "Sitting at my desk typing registers hundreds of false steps. Also the step goal animations are glitchy. Battery life is decent though, around 4 days. Hopefully software updates patch the sensor calibration.", 2, True, 18, "2026-07-22"),

        ("PulseFit Apex Smartwatch", "Electronics", "Screen cracked after a minor bump against the door frame",
         "Advertised with sapphire crystal glass, yet it scratched and developed a hairline crack after a slight accidental bump against a wooden door frame. Warranty refused to cover it claiming accidental damage. Waste of money.", 1, True, 62, "2026-08-01"),

        ("PulseFit Apex Smartwatch", "Electronics", "Sleek look, smooth UI, love the notifications",
         "Syncs instantly with my phone, notifications appear promptly, and replying with quick voice-to-text is super handy. Highly recommended for busy professionals.", 5, True, 22, "2026-08-15"),

        ("PulseFit Apex Smartwatch", "Electronics", "Good hardware, frustrating companion app",
         "The watch itself feels solid and looks like a luxury timepiece. However, the mobile app constantly signs me out and sync takes over two minutes every morning. If the app improves, this would be a 5-star product.", 3, True, 14, "2026-08-28"),

        # Competitor Smartwatch: Chronos Pro
        ("Chronos Pro Watch", "Electronics", "Blazing fast processor and breathtaking display",
         "Fluid 60Hz animations, sapphire glass is impervious to scratches, and wireless fast charging is a game changer. Charges to 80% in 25 minutes. Worth every single penny.", 5, True, 51, "2026-07-15"),

        ("Chronos Pro Watch", "Electronics", "Battery drains in less than 18 hours",
         "I wanted to love this watch, but having to charge it every single evening defeats sleep tracking. If you enable always-on display and GPS workout tracking, it dies before dinner.", 2, True, 77, "2026-07-29"),

        ("Chronos Pro Watch", "Electronics", "Sophisticated design, premium leather strap",
         "Looks great in formal suits and casual wear alike. The rotating dial crown makes navigation effortless without smudging the screen. Very satisfied with the craftsmanship.", 4, True, 16, "2026-08-11"),

        ("Chronos Pro Watch", "Electronics", "GPS drift in downtown city centers",
         "Tall buildings cause the GPS track to wobble significantly during runs. Distance was off by nearly 800 meters over a 5K race. Health sensors are good, but runners beware.", 3, False, 8, "2026-08-24"),

        ("Chronos Pro Watch", "Electronics", "Totally defective speaker right out of the box",
         "Bluetooth phone calls emit loud static and crackling noises from the onboard speaker. Tested on two different phones, same result. Quality control is lacking.", 1, True, 39, "2026-09-03"),

        # Home & Kitchen - Coffee Maker
        ("BaristaExpress 15-Bar Espresso Machine", "Home & Kitchen", "Authentic café quality espresso at home!",
         "Pulls rich, velvety crema consistently every morning. The steam wand produces micro-foam perfect for latte art with a little practice. Fast heating thermo-block is ready in 35 seconds. Absolute kitchen masterpiece.", 5, True, 92, "2026-06-12"),

        ("BaristaExpress 15-Bar Espresso Machine", "Home & Kitchen", "Water tank leaks from the bottom valve",
         "Woke up on day four to find half a liter of water puddled across my kitchen granite countertop. The seal at the bottom of the reservoir is loose. Exchanging for a replacement.", 2, True, 33, "2026-06-25"),

        ("BaristaExpress 15-Bar Espresso Machine", "Home & Kitchen", "Good coffee but requires frequent descaling and cleaning",
         "The espresso is flavorful and robust, but the maintenance light turns on every two weeks even though I use filtered water. The drip tray is also small and fills up very quickly.", 3, True, 11, "2026-07-04"),

        ("BaristaExpress 15-Bar Espresso Machine", "Home & Kitchen", "Broke after 2 months. Pump stopped pressurizing",
         "Started making a loud vibrating sound and then water stopped flowing through the portafilter entirely. Customer service asked me to ship the 20lb machine at my own expense for inspection. Horrible experience.", 1, True, 58, "2026-07-19"),

        ("BaristaExpress 15-Bar Espresso Machine", "Home & Kitchen", "Great starter espresso machine for beginners",
         "Simple pressurized portafilter makes it forgiving for beginners who do not have a commercial grinder. Instructions were clear and the tamper included feels substantial.", 4, True, 20, "2026-08-08"),

        # Home & Kitchen - Air Fryer
        ("CrispMaster XXL Digital Air Fryer", "Home & Kitchen", "Crispy French fries with zero guilt!",
         "We use this almost daily. Fries, chicken wings, salmon, and roasted veggies all come out perfectly crispy on the outside and tender inside. The non-stick basket is genuinely dishwasher safe.", 5, True, 64, "2026-07-01"),

        ("CrispMaster XXL Digital Air Fryer", "Home & Kitchen", "Plastic chemical smell will not go away",
         "Ran it empty 5 times as recommended, but there is still a lingering burnt plastic odor that permeates food. Returning immediately.", 1, True, 41, "2026-07-18"),

        ("CrispMaster XXL Digital Air Fryer", "Home & Kitchen", "Large basket fits a whole chicken easily",
         "Spacious 7-quart capacity is ideal for a family of four. Cooks evenly without needing to shake constantly. Touch screen presets are intuitive.", 5, True, 27, "2026-08-03"),

        ("CrispMaster XXL Digital Air Fryer", "Home & Kitchen", "Fan is quite loud, but performance is solid",
         "Sounds like a mini vacuum cleaner when running on high heat, but it cooks fast and saves electricity compared to preheating the oven. Good buy overall.", 4, False, 9, "2026-08-19"),

        # Fashion - Running Shoes
        ("AeroStride Ultra Carbon Running Shoes", "Fashion", "Like running on clouds! Shaved 2 minutes off my 10k",
         "The carbon plate and responsive foam provide immense energy return. Extremely breathable mesh upper kept my feet cool during hot summer runs. Will definitely buy another pair.", 5, True, 55, "2026-06-30"),

        ("AeroStride Ultra Carbon Running Shoes", "Fashion", "Narrow toe box gave me blisters on long runs",
         "The cushioning is plush, but the forefoot is noticeably too tight. After 6 miles my pinky toes were cramped and sore. Recommend ordering a half size up.", 2, True, 29, "2026-07-14"),

        ("AeroStride Ultra Carbon Running Shoes", "Fashion", "Sole rubber began peeling after 100 miles",
         "For a shoe costing over $180, the outsole durability is pathetic. The rubber pod under the midfoot started delaminating after just two months of road running.", 2, True, 48, "2026-08-09"),

        ("AeroStride Ultra Carbon Running Shoes", "Fashion", "Super lightweight and visually stunning design",
         "Get compliments every time I wear them to the track. Featherlight construction makes your legs feel fresh even after high mileage workouts.", 5, True, 17, "2026-08-22"),

        # Electronics - Laptop Stand
        ("ErgoRise Aluminum Ergonomic Laptop Stand", "Electronics", "Heavy duty, rock solid, cured my neck pain",
         "All aluminum construction with sturdy dual hinges that do not wobble when typing. Elevates the screen right to eye level. Ventilated hollow design keeps my MacBook cool.", 5, True, 38, "2026-05-18"),

        ("ErgoRise Aluminum Ergonomic Laptop Stand", "Electronics", "Stiff hinges require two hands and significant force to adjust",
         "While it is stable once positioned, adjusting the angle is so stiff that I worry about pinching my fingers. Rubber pads on the hooks also slipped off after a week.", 3, True, 7, "2026-06-08"),

        # Electronics - Portable Power Bank
        ("VoltCharge 20000mAh 65W Fast Power Bank", "Electronics", "Charged my laptop and phone simultaneously at full speed",
         "True 65W Power Delivery output capable of charging my Dell XPS and iPhone rapidly on long travels. Compact form factor given the massive capacity. Display percentage is very accurate.", 5, True, 49, "2026-07-09"),

        ("VoltCharge 20000mAh 65W Fast Power Bank", "Electronics", "Stopped holding charge after 5 months",
         "Suddenly went from 100% to 0% in ten minutes. Plugged it in for 12 hours and it stays stuck at 0%. Cheap internal cells that degrade far too quickly.", 1, True, 37, "2026-08-16"),

        ("VoltCharge 20000mAh 65W Fast Power Bank", "Electronics", "A bit heavy for pocket carry, but reliable power",
         "Weighs around a pound so you definitely feel it in your backpack, but having emergency power for 4 full phone recharges is worth the heft.", 4, True, 15, "2026-08-29"),

        # Sarcastic & Mixed Nuanced Reviews
        ("AuraSound Pro ANC Wireless Earbuds", "Electronics", "Oh wonderful, if you enjoy listening to static music!",
         "Yeah right, supreme noise cancellation! The only noise it cancelled was my expectation. Static hiss in the background was louder than my podcasts. Sarcasm aside, truly disappointed.", 1, True, 42, "2026-09-08"),

        ("BaristaExpress 15-Bar Espresso Machine", "Home & Kitchen", "Great decor, useless coffee maker",
         "It looks gorgeous sitting on the counter like a museum exhibit. Too bad making actual espresso results in lukewarm sour liquid. Five stars for aesthetics, zero stars for function.", 2, True, 31, "2026-08-17"),

        ("PulseFit Apex Smartwatch", "Electronics", "Not bad, but not extraordinary either",
         "Does basic fitness tracking okay. Heart rate is within 5 bpm of chest strap. Sleep tracking is somewhat hit or miss. Fair balance between features and price.", 3, True, 13, "2026-09-01"),

        ("CrispMaster XXL Digital Air Fryer", "Home & Kitchen", "Best kitchen purchase of the decade!",
         "Can not believe we lived without this for so long. Chicken wings come out crispier than deep frying with 80% less oil. Cleanup takes literally two minutes under warm water.", 5, True, 83, "2026-08-30"),

        ("AeroStride Ultra Carbon Running Shoes", "Fashion", "Uncomfortable arch support, had to return",
         "Felt like there was a golf ball pushing directly against my inner arch. Might suit neutral runners with high arches, but caused severe foot fatigue for flat feet.", 2, True, 19, "2026-07-28"),

        ("VoltCharge 20000mAh 65W Fast Power Bank", "Electronics", "Airline approved and savior on international trips",
         "Flew from NYC to Tokyo and this battery kept my tablet and phone powered throughout the entire 14-hour flight. Solid metallic casing feels indestructible.", 5, True, 60, "2026-09-12")
    ]

    # Expand to 100+ realistic reviews with slight variations & timestamps
    expanded = []
    base_date = datetime(2026, 9, 20)
    for i, (prod, cat, title, text, rating, ver, helpful, date_str) in enumerate(reviews, start=1):
        expanded.append({
            "review_id": f"REV-{1000 + i}",
            "product_name": prod,
            "category": cat,
            "review_title": title,
            "review_text": text,
            "star_rating": rating,
            "verified_purchase": ver,
            "helpful_votes": helpful,
            "review_date": date_str
        })

    # Add synthesized realistic reviews across categories
    extra_templates = [
        ("AuraSound Pro ANC Wireless Earbuds", "Electronics", "Clear highs and immersive soundstage", "The vocal clarity in acoustic tracks is mesmerizing. Treble is bright without being piercing. Pairing was instantaneous with iOS and Android.", 5, 25),
        ("AuraSound Pro ANC Wireless Earbuds", "Electronics", "Occasional audio dropouts in crowded areas", "Works well at home, but walking through the train terminal causes brief stuttering in the right earbud. Battery life is decent.", 3, 8),
        ("PulseFit Apex Smartwatch", "Electronics", "Battery life exceeded my expectations!", "Easily got 6 days of moderate usage including 3 GPS runs. Display is gorgeous and watch faces are customizable.", 5, 31),
        ("PulseFit Apex Smartwatch", "Electronics", "Charging cable connection is finicky", "The magnetic charging pins must be aligned exactly right, otherwise it slips off and does not charge overnight.", 2, 11),
        ("Chronos Pro Watch", "Electronics", "Flawless notification sync and gorgeous bezels", "Very fast response when replying to text messages. Sleep score matches how refreshed I feel.", 5, 19),
        ("Chronos Pro Watch", "Electronics", "Too bulky for smaller wrists", "Case diameter is 46mm and it is quite thick. Kept catching on shirt cuffs and felt heavy while sleeping.", 3, 14),
        ("BaristaExpress 15-Bar Espresso Machine", "Home & Kitchen", "Super fast warmup and consistent steam pressure", "Takes under a minute from flipping the switch to pulling a double shot. Milk frothing is effortless.", 5, 40),
        ("BaristaExpress 15-Bar Espresso Machine", "Home & Kitchen", "Pressure gauge stopped working after one week", "The needle remains at zero even when espresso pulls normally. Build quality feels a bit cheap for the price tag.", 2, 22),
        ("CrispMaster XXL Digital Air Fryer", "Home & Kitchen", "Roasts vegetables to absolute perfection", "Asparagus, broccoli, and sweet potatoes get a fantastic caramelization in under 12 minutes. Highly recommended.", 5, 29),
        ("CrispMaster XXL Digital Air Fryer", "Home & Kitchen", "Teflon coating started flaking off", "Always hand washed with a soft sponge, but after 3 months little black specks began peeling from the grill rack.", 1, 52),
        ("AeroStride Ultra Carbon Running Shoes", "Fashion", "Personal record on half marathon!", "Propulsion from the carbon plate is noticeable from the first stride. Lightweight, breathable, and zero hot spots.", 5, 61),
        ("AeroStride Ultra Carbon Running Shoes", "Fashion", "Slippery on wet asphalt roads", "Great on dry pavement, but during light rain the grip is dangerously slick. Be careful on wet painted road lines.", 3, 15),
        ("ErgoRise Aluminum Ergonomic Laptop Stand", "Electronics", "Sturdy and elegant design", "Holds my 16 inch laptop firmly with zero tilt or slipping. Perfect posture improvement.", 5, 18),
        ("ErgoRise Aluminum Ergonomic Laptop Stand", "Electronics", "Sharp aluminum edges scratch desk surface", "The bottom edges lack sufficient rubber padding, resulting in small scratches on my wooden desk.", 2, 9),
        ("VoltCharge 20000mAh 65W Fast Power Bank", "Electronics", "Powers Nintendo Switch and laptop during flights", "Traveled cross-country and never had to search for an airport wall outlet. Charges fast with USB-C PD.", 5, 35),
        ("VoltCharge 20000mAh 65W Fast Power Bank", "Electronics", "Overheats during 65W laptop fast charge", "Gets alarmingly hot to the touch when charging my laptop from 10% battery. Stops charging until it cools down.", 2, 28)
    ]

    for j, (prod, cat, title, text, rating, helpful) in enumerate(extra_templates, start=len(expanded) + 1):
        delta_days = random.randint(5, 75)
        dt = (base_date - timedelta(days=delta_days)).strftime("%Y-%m-%d")
        expanded.append({
            "review_id": f"REV-{1000 + j}",
            "product_name": prod,
            "category": cat,
            "review_title": title,
            "review_text": text,
            "star_rating": rating,
            "verified_purchase": random.choice([True, True, True, False]),
            "helpful_votes": helpful,
            "review_date": dt
        })

    df = pd.DataFrame(expanded)
    df.to_csv(r"C:\Users\ADMIN\.gemini\antigravity\scratch\product-review-sentiment-analyzer\data\sample_reviews.csv", index=False)
    print(f"Generated sample_reviews.csv with {len(df)} records.")

def create_earphone_reviews():
    """Generates focused dataset for Aspect-Based Sentiment Analysis."""
    records = [
        ("AuraSound Pro ANC Wireless Earbuds", "Sound quality is punchy and detailed with crisp treble. Battery life gives around 7 hours.", 5, "2026-08-10"),
        ("AuraSound Pro ANC Wireless Earbuds", "The noise cancellation is magical on trains, but the microphone makes my voice sound hollow.", 3, "2026-08-12"),
        ("AuraSound Pro ANC Wireless Earbuds", "Extremely comfortable in ears for hours, but Bluetooth drops occasionally when walking.", 4, "2026-08-15"),
        ("AuraSound Pro ANC Wireless Earbuds", "Build quality feels cheap and plastic creaks. Left earbud broke after two weeks.", 1, "2026-08-18"),
        ("AuraSound Pro ANC Wireless Earbuds", "Incredible price to performance ratio! Battery lasts through my entire workday.", 5, "2026-08-21"),
        ("AuraSound Pro ANC Wireless Earbuds", "Bass is muddy and overbearing, drowning out vocal clarity. Disappointed with acoustic balance.", 2, "2026-08-25"),
        ("AuraSound Pro ANC Wireless Earbuds", "Super fast USB-C wireless charging. Ergonomics are top notch, fits snug during gym sessions.", 5, "2026-08-28"),
        ("AuraSound Pro ANC Wireless Earbuds", "Customer service was completely unhelpful when I asked for replacement silicone ear tips.", 2, "2026-09-01"),
        ("AuraSound Pro ANC Wireless Earbuds", "Noise cancellation blocks office hum perfectly. Microphones filter out background keyboard clatter.", 5, "2026-09-04"),
        ("AuraSound Pro ANC Wireless Earbuds", "Horrible battery life! Barely lasts 2.5 hours with ANC switched on.", 1, "2026-09-07"),
        ("AuraSound Pro ANC Wireless Earbuds", "Decent build and sleek charging case. Sound is average for the price tag.", 3, "2026-09-10"),
        ("AuraSound Pro ANC Wireless Earbuds", "Best soundstage in any earbud under $150. Crisp instrument separation and punchy bass.", 5, "2026-09-14"),
        ("AuraSound Pro ANC Wireless Earbuds", "Bluetooth pairing fails repeatedly with Windows 11 laptops. Constantly disconnecting.", 2, "2026-09-16"),
        ("AuraSound Pro ANC Wireless Earbuds", "Ergonomic fit is sublime. Featherlight and never slips out while jogging.", 5, "2026-09-18"),
        ("AuraSound Pro ANC Wireless Earbuds", "Overpriced for what it offers. Flimsy hinge on the charging lid.", 2, "2026-09-21")
    ]
    df = pd.DataFrame([
        {
            "review_id": f"EAR-{200 + i}",
            "product_name": r[0],
            "review_text": r[1],
            "star_rating": r[2],
            "review_date": r[3]
        }
        for i, r in enumerate(records, start=1)
    ])
    df.to_csv(r"C:\Users\ADMIN\.gemini\antigravity\scratch\product-review-sentiment-analyzer\data\earphone_reviews.csv", index=False)
    print(f"Generated earphone_reviews.csv with {len(df)} records.")

def create_smartwatch_comparison():
    """Generates comparative dataset for PulseFit Apex vs Chronos Pro."""
    pulsefit_reviews = [
        ("PulseFit Apex Smartwatch", "PulseFit", "Accurate heart rate and sleep tracking. Battery lasts 5 days easily.", 5, "2026-08-01"),
        ("PulseFit Apex Smartwatch", "PulseFit", "Screen scratched after minor collision with table. Fragile glass.", 2, "2026-08-05"),
        ("PulseFit Apex Smartwatch", "PulseFit", "Lightweight, comfortable band. Step counter is very reliable.", 5, "2026-08-09"),
        ("PulseFit Apex Smartwatch", "PulseFit", "Companion mobile app is sluggish and disconnects randomly.", 2, "2026-08-14"),
        ("PulseFit Apex Smartwatch", "PulseFit", "Terrific battery life and bright outdoor display.", 5, "2026-08-19"),
        ("PulseFit Apex Smartwatch", "PulseFit", "GPS takes over 3 minutes to lock satellite position.", 3, "2026-08-25"),
        ("PulseFit Apex Smartwatch", "PulseFit", "Sensors give comprehensive health metrics. Well worth the price.", 4, "2026-09-02"),
        ("PulseFit Apex Smartwatch", "PulseFit", "Charger pins disconnect if you nudge the nightstand.", 3, "2026-09-09"),
    ]
    chronos_reviews = [
        ("Chronos Pro Watch", "Chronos", "Ultra-fast processor, butter smooth UI and sapphire glass is scratch proof.", 5, "2026-08-02"),
        ("Chronos Pro Watch", "Chronos", "Abysmal battery life. Requires charging twice on workout days.", 1, "2026-08-06"),
        ("Chronos Pro Watch", "Chronos", "Stunning executive design. Looks amazing with formal office suits.", 5, "2026-08-11"),
        ("Chronos Pro Watch", "Chronos", "Speaker for Bluetooth phone calls started buzzing and crackling.", 2, "2026-08-16"),
        ("Chronos Pro Watch", "Chronos", "Fast wireless charging gets it to 80% quickly, but drains fast.", 3, "2026-08-21"),
        ("Chronos Pro Watch", "Chronos", "Rotating dial makes navigating menus pleasurable and precise.", 5, "2026-08-27"),
        ("Chronos Pro Watch", "Chronos", "Too heavy and thick on the wrist. Uncomfortable to sleep with.", 2, "2026-09-03"),
        ("Chronos Pro Watch", "Chronos", "Flawless notification sync and voice messaging capability.", 4, "2026-09-11"),
    ]

    all_revs = pulsefit_reviews + chronos_reviews
    df = pd.DataFrame([
        {
            "review_id": f"CMP-{300 + i}",
            "product_name": r[0],
            "brand": r[1],
            "review_text": r[2],
            "star_rating": r[3],
            "review_date": r[4]
        }
        for i, r in enumerate(all_revs, start=1)
    ])
    df.to_csv(r"C:\Users\ADMIN\.gemini\antigravity\scratch\product-review-sentiment-analyzer\data\smartwatch_reviews.csv", index=False)
    print(f"Generated smartwatch_reviews.csv with {len(df)} records.")

if __name__ == "__main__":
    create_sample_reviews()
    create_earphone_reviews()
    create_smartwatch_comparison()
