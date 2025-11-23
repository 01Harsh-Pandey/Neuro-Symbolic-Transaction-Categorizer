import yaml
import pandas as pd
import random
import string
import os
from faker import Faker

# Initialize Faker
fake = Faker()

# Configuration
OUTPUT_FILE = "data/synthetic_dataset.csv"
TAXONOMY_FILE = "data/taxonomy.yaml"
NUM_SAMPLES = 10000 

# --- TYPO ENGINE (The Expert's "Secret Sauce") ---
def swap_adjacent(s):
    if len(s) < 2: return s
    i = random.randint(0, len(s)-2)
    return s[:i] + s[i+1] + s[i] + s[i+2:]

def delete_char(s):
    if len(s) < 2: return s
    i = random.randint(0, len(s)-1)
    return s[:i] + s[i+1:]

def insert_random_char(s):
    i = random.randint(0, len(s))
    c = random.choice(string.ascii_lowercase + "0123456789")
    return s[:i] + c + s[i:]

def apply_typos(text, severity=1):
    """Applies 1 or more typo operations based on severity."""
    s = text
    ops = [swap_adjacent, delete_char, insert_random_char]
    for _ in range(severity):
        op = random.choice(ops)
        s = op(s)
    return s

# --- TEMPLATE ENGINE ---
def generate_template_noise(keyword):
    """Injects structural noise (Merchant IDs, Cities, 'Pending')."""
    templates = [
        f"{keyword.upper()}",
        f"{keyword.upper()} * {fake.bothify(text='##??##')}",
        f"POS PURCHASE {keyword.upper()} {fake.city().upper()}",
        f"{fake.bothify(text='??#')} {keyword.upper()} STORE {fake.random_int(100, 999)}",
        f"PAYPAL *{keyword.upper()} {fake.bothify(text='###')}",
        f"{keyword.capitalize()} {fake.date_this_year()}",
        f"CHECKCARD {fake.date_this_year()} {keyword.upper()}",
        f"{keyword.upper()} {fake.state_abbr()} {fake.zipcode()}",
        f"ONLINE PAYMENT {keyword.upper()} {fake.bothify(text='#######')}",
        f"MOBILE {keyword.upper()} APP {fake.bothify(text='###')}",
        f"CONTACTLESS {keyword.upper()} {fake.bothify(text='CHIP###')}",
        f"RECURRING {keyword.upper()} {fake.month_name().upper()}",
        f"{fake.bothify(text='###')} {keyword.upper()} {fake.bothify(text='INV####')}",
        f"{keyword.upper()} {fake.date_this_month().strftime('%m/%d')} {fake.bothify(text='AUTH###')}",
        f"INTL {keyword.upper()} {fake.country_code().upper()} {fake.bothify(text='FEE##')}"
    ]
    return random.choice(templates)

def load_taxonomy(filepath):
    with open(filepath, 'r') as f:
        return yaml.safe_load(f)

def generate_realistic_amount(category_name, subcategory_name):
    """Generate realistic transaction amounts based on category"""
    amount_profiles = {
        "Transportation": {
            "Rideshare": (5.00, 75.00),
            "Fuel": (25.00, 120.00),
            "Public Transit": (2.50, 25.00),
            "Parking": (8.00, 45.00)
        },
        "Food & Dining": {
            "Coffee Shops": (3.50, 12.00),
            "Restaurants": (15.00, 150.00),
            "Groceries": (25.00, 250.00),
            "Fast Food": (8.00, 25.00)
        },
        "Shopping": {
            "Electronics": (50.00, 2000.00),
            "E-Commerce": (15.00, 300.00),
            "Clothing": (20.00, 200.00),
            "Department Stores": (25.00, 350.00)
        },
        "Bills & Utilities": {
            "Internet": (45.00, 120.00),
            "Electricity": (60.00, 300.00),
            "Water": (30.00, 100.00),
            "Phone": (40.00, 150.00)
        },
        "Income": {
            "Salary": (2000.00, 10000.00),
            "Transfers": (50.00, 2000.00),
            "Freelance": (100.00, 2500.00),
            "Investments": (25.00, 500.00)
        }
    }
    
    min_amt, max_amt = amount_profiles.get(category_name, {}).get(subcategory_name, (1.00, 100.00))
    return round(random.uniform(min_amt, max_amt), 2)

def main():
    print(f"Loading taxonomy from {TAXONOMY_FILE}...")
    data = load_taxonomy(TAXONOMY_FILE)['taxonomy']
    
    dataset = []
    
    print(f"Generating {NUM_SAMPLES} robust samples...")
    
    for i in range(NUM_SAMPLES):
        # 1. Select Ground Truth
        category_obj = random.choice(data)
        sub_cat_obj = random.choice(category_obj['subcategories'])
        keyword = random.choice(sub_cat_obj['keywords'])
        
        # 2. Generate Base String (Template Noise)
        # We assume base strings mostly have SOME template noise (e.g., city names)
        # because "clean" keywords rarely exist in isolation in bank logs.
        base_text = generate_template_noise(keyword)
        
        # 3. Apply Noise Strategy (The Expert's Ratio)
        rand_val = random.random()
        
        if rand_val < 0.55:
            # 55%: Clean-ish / Light Template 
            # (Just the base template, no extra mangling)
            final_text = base_text
            noise_type = "clean_template"
            
        elif rand_val < 0.75:
            # 20%: Heavy Template Variants (Suffixes/Prefixes)
            # Add extra junk to the ends
            prefixes = ["", f"{fake.random_int(1000,9999)} ", "AUTH ", "PENDING ", "CHK "]
            suffixes = ["", f" REF:{fake.bothify('??##')}", f" ID:{fake.bothify('###')}", " PENDING"]
            
            final_text = f"{random.choice(prefixes)}{base_text}{random.choice(suffixes)}"
            noise_type = "heavy_template"
            
        elif rand_val < 0.95:
            # 20%: Typo Injection (1 typo)
            final_text = apply_typos(base_text, severity=1)
            noise_type = "typo_light"
            
        else:
            # 5%: Heavy Noise (Typos + Structural garbage)
            # This stresses the model to look for subword fragments
            garbage_prefixes = ["Xfer ", "ACH ", "POS ", "TFR ", ""]
            garbage_suffixes = [f" {fake.bothify('###??')}", " CONFIRMED", " SECURE", ""]
            
            heavy_base = f"{random.choice(garbage_prefixes)}{base_text}{random.choice(garbage_suffixes)}"
            final_text = apply_typos(heavy_base, severity=2)
            noise_type = "heavy_noise"

        # Generate realistic amount and date
        amount = generate_realistic_amount(category_obj['name'], sub_cat_obj['name'])
        date = fake.date_between(start_date='-1y', end_date='today')
        transaction_id = f"TXN{fake.bothify(text='##########')}"
        
        # Determine transaction type
        txn_type = "credit" if category_obj['name'] == "Income" else "debit"
        
        dataset.append({
            "transaction_id": transaction_id,
            "date": date,
            "raw_text": final_text,
            "clean_keyword": keyword,
            "category": category_obj['name'],
            "subcategory": sub_cat_obj['name'],
            "label": f"{category_obj['name']} > {sub_cat_obj['name']}",
            "amount": amount,
            "type": txn_type,
            "noise_type": noise_type # Useful for analyzing error rates later
        })
        
        # Progress indicator
        if (i + 1) % 1000 == 0:
            print(f"Generated {i + 1}/{NUM_SAMPLES} samples...")
    
    # Save
    df = pd.DataFrame(dataset)
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)
    
    print(f"✅ Generated {len(df)} samples.")
    print("\n📊 Noise Distribution Check:")
    noise_dist = df['noise_type'].value_counts(normalize=True).sort_index()
    for noise_type, percentage in noise_dist.items():
        print(f"  {noise_type}: {percentage:.1%}")
    
    print("\n💰 Amount Statistics by Category:")
    for category in df['category'].unique():
        cat_data = df[df['category'] == category]
        avg_amount = cat_data['amount'].mean()
        print(f"  {category}: ${avg_amount:.2f} avg")
    
    print("\n🔍 Sample Data (showing different noise types):")
    sample_df = df.groupby('noise_type').head(1)[['raw_text', 'clean_keyword', 'noise_type', 'category', 'amount']]
    for _, row in sample_df.iterrows():
        print(f"  [{row['noise_type']:14}] ${row['amount']:6.2f} {row['category']:15} -> '{row['raw_text']}'")

if __name__ == "__main__":
    main()