"""
OPTIONAL: How to use your Kaggle data with the recommender system

If you want to replace the sample data with your actual travel dataset,
follow this guide.

The recommender works with ANY travel data as long as it has the right columns.
"""

import pandas as pd
import numpy as np

# ============================================================================
# OPTION 1: SIMPLE - Load CSV directly
# ============================================================================

def load_kaggle_data(csv_path):
    """
    Load your Kaggle travel data CSV file
        Example:
    """

        # df = load_kaggle_data('C:\Users\Riya\Desktop\projects\travel rec\data\clean_travel_data.csv')
    
    df = pd.read_csv('clean_travel_data.csv')
    
    # Normalize column names to lowercase
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
    
    print(f"Loaded {len(df)} rows")
    print(f"Columns: {df.columns.tolist()}")
    
    return df


# ============================================================================
# OPTION 2: CLEAN - Remove problems from your combined dataset
# ============================================================================

def clean_kaggle_data(csv_path):
    """
    Clean your problematic combined dataset
    
    Fixes:
    - Removes fake 'destination_name' column
    - Removes duplicate cities
    - Validates feature columns
    - Fills missing values
    """
    
    df = load_kaggle_data(csv_path)
    
    print("\n🔧 CLEANING DATA...")
    
    # CRITICAL FIX: Remove the fake destination_name column
    if 'destination_name' in df.columns:
        print("  ✅ Removing fake 'destination_name' column")
        df = df.drop('destination_name', axis=1)
    
    # Keep only rows with valid cities
    required_cols = ['city', 'country']
    df = df.dropna(subset=required_cols)
    print(f"  ✅ Kept rows with valid city/country")
    
    # Remove duplicate city-country pairs
    before = len(df)
    df = df.drop_duplicates(subset=['city', 'country'], keep='first')
    after = len(df)
    print(f"  ✅ Removed {before - after} duplicates")
    
    # Feature columns you need
    feature_cols = ['culture', 'adventure', 'nature', 'beaches', 
                   'nightlife', 'cuisine', 'wellness', 'urban', 'seclusion']
    
    # type,avg_cost_(usd/day),best_season,avg_rating,annual_visitors_(m),
    # unesco_site,id,city,region,short_description,latitude,longitude,avg_temp_monthly,
    # ideal_durations,budget_level,culture,adventure,nature,beaches,nightlife,cuisine,
    # wellness,urban,seclusion
    
    # Add missing feature columns with neutral defaults (3/5)
    for col in feature_cols:
        if col not in df.columns:
            print(f"  ⚠️  Missing '{col}' - using default value 3/5")
            df[col] = 3.0
        else:
            # Ensure values are in 1-5 range
            df[col] = df[col].clip(1, 5)
            # Fill missing values
            df[col] = df[col].fillna(3.0)
    
    print(f"  ✅ Features validated\n")
    
    return df


# ============================================================================
# OPTION 3: CONVERT - Turn CSV into format for recommender
# ============================================================================

def prepare_for_recommender(csv_path, output_path='clean_travel_data.csv'):
    """
    Clean and prepare data, save as CSV ready for recommender
    
    Usage:
        prepare_for_recommender('your_data.csv')
        # Now use this in travel_recommender.py:
        # df = pd.read_csv('clean_travel_data.csv')
    """
    
    df = clean_kaggle_data(csv_path)
    df.to_csv(output_path, index=False)
    print(f"✅ Cleaned data saved to: {output_path}")
    print(f"   Ready to use with travel_recommender.py!")
    
    return df


# ============================================================================
# OPTION 4: INTEGRATE - Load Kaggle data into recommender
# ============================================================================

def use_kaggle_data_in_recommender(csv_path):
    """
    Modify travel_recommender.py to use your Kaggle data instead of sample
    
    Steps:
    1. In travel_recommender.py, find this line (around line 115):
       df = pd.DataFrame(DESTINATIONS_DATA)
    
    2. Replace it with:
       df = pd.read_csv(csv_path)
       df = df.drop_duplicates(subset=['city', 'country'], keep='first')
    
    3. Make sure these columns exist in your CSV:
       - city
       - country
       - culture, adventure, nature, beaches, nightlife, cuisine, wellness, urban, seclusion (1-5 scale)
       - Optional: rating, cost_per_day, best_season
    
    Done! Now you're using your real data.
    """
    pass


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

if __name__ == "__main__":
    
    print("="*70)
    print("HOW TO USE YOUR KAGGLE DATA WITH THE RECOMMENDER")
    print("="*70 + "\n")
    
    # Example 1: Just load it
    print("EXAMPLE 1: Load your CSV file")
    print("-"*70)
    print("""
    import pandas as pd
    df = pd.read_csv('your_travel_data.csv')
    
    # Make sure it has these columns:
    # - city
    # - country
    # - culture, adventure, nature, beaches, nightlife, cuisine, wellness, urban, seclusion
    """)
    
    # Example 2: Clean it first
    print("\nEXAMPLE 2: Clean your problematic combined dataset")
    print("-"*70)
    print("""
    from data_loader import clean_kaggle_data
    
    df = clean_kaggle_data('data/combined_travel_data.csv')
    # Now use df in the recommender
    """)
    
    # Example 3: Save cleaned data
    print("\nEXAMPLE 3: Save cleaned data for later")
    print("-"*70)
    print("""
    from data_loader import prepare_for_recommender
    
    # This will clean and save
    prepare_for_recommender('data/combined_travel_data.csv', 'clean_data.csv')
    
    # Then in your script:
    df = pd.read_csv('clean_data.csv')
    """)
    
    # Example 4: Full integration
    print("\nEXAMPLE 4: Full integration with recommender")
    print("-"*70)
    print("""
    In travel_recommender.py, line 115, change:
    
    FROM:
        df = pd.DataFrame(DESTINATIONS_DATA)
    
    TO:
        df = pd.read_csv('your_kaggle_data.csv')
        df = df.drop_duplicates(subset=['city', 'country'], keep='first')
    
    Then run:
        python travel_recommender.py
    """)
    
    print("\n" + "="*70)
    print("IMPORTANT COLUMNS NEEDED IN YOUR CSV:")
    print("="*70)
    print("""
    REQUIRED:
    - city (string)
    - country (string)
    - culture (1-5)
    - adventure (1-5)
    - nature (1-5)
    - beaches (1-5)
    - nightlife (1-5)
    - cuisine (1-5)
    - wellness (1-5)
    - urban (1-5)
    - seclusion (1-5)
    
    OPTIONAL (nice to have):
    - rating (0-5)
    - cost_per_day (number)
    - best_season (string)
    - latitude (number)
    - longitude (number)
    """)
    
    print("\n✅ If you need help:")
    print("   1. Check your column names match above")
    print("   2. Ensure feature columns are 1-5 scale")
    print("   3. Remove 'destination_name' if it exists")
    print("   4. Run the recommender with your data!")
