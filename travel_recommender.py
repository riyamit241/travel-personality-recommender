import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import json


#part 1: LOADING DATASET
df = pd.read_csv('clean_travel_data.csv')   
df = df.drop_duplicates(subset=['city', 'country'], keep='first')

if 'destination_name' in df.columns:
    df = df.drop('destination_name', axis=1)

# ============================================================================
# PART 2: PERSONALITY MAPPER - Convert personality to travel preferences
# ============================================================================

class PersonalityToTravelMapper:
    """Converts Big Five personality scores to travel preferences"""
    
    def __init__(self):
        self.travel_features = ['culture', 'adventure', 'nature', 'beaches', 
                               'nightlife', 'cuisine', 'wellness', 'urban', 'seclusion']
    
    def mbti_to_big5(self, mbti_type):
        """Convert 16 Personalities MBTI type to Big Five scores (0-100)"""
        
        e_i = mbti_type[0]      # E=Extroverted (80), I=Introverted (40)
        s_n = mbti_type[1]      # N=iNtuitive/Openness (80), S=Sensing (40)
        t_f = mbti_type[2]      # F=Feeling/Agreeable (75), T=Thinking (45)
        j_p = mbti_type[3]      # J=Judging/Conscientious (80), P=Perceiving (40)
        
        big5_scores = {
            'openness': 80 if s_n == 'N' else 40,
            'conscientiousness': 80 if j_p == 'J' else 40,
            'extraversion': 80 if e_i == 'E' else 40,
            'agreeableness': 75 if t_f == 'F' else 45,
            'neuroticism': 50  # Average for now
        }
        
        return big5_scores
    
    def big5_to_travel_preferences(self, big5_scores):
        """Map Big Five personality to travel preferences (0-1 scale)"""
        
        # Normalize to 0-1
        o = big5_scores['openness'] / 100
        c = big5_scores['conscientiousness'] / 100
        e = big5_scores['extraversion'] / 100
        a = big5_scores['agreeableness'] / 100
        n = big5_scores['neuroticism'] / 100
        
        prefs = np.array([
            o*0.6 + a*0.4,                          # culture
            o*0.7 + (1-n)*0.3,                      # adventure
            o*0.5 + a*0.3 + (1-e)*0.2,             # nature
            o*0.5 + (1-c)*0.5,                      # beaches
            e*0.8 + (1-n)*0.2,                      # nightlife
            o*0.5 + a*0.5,                          # cuisine
            a*0.6 + n*0.4,                          # wellness
            e*0.6 + c*0.4,                          # urban
            o*0.3 + a*0.3 + (1-e)*0.4              # seclusion
        ])
        
        return np.clip(prefs, 0, 1)
    
    def mbti_to_travel_preferences(self, mbti_type):
        """One-step conversion: MBTI -> Travel Preferences"""
        big5 = self.mbti_to_big5(mbti_type)
        return self.big5_to_travel_preferences(big5)


# ============================================================================
# PART 3: RECOMMENDER ENGINE - Content-based filtering with cosine similarity
# ============================================================================

class TravelRecommender:
    """Recommend travel destinations based on personality"""
    
    def __init__(self, destinations_df):
        self.df = destinations_df.copy()
        self.feature_cols = ['culture', 'adventure', 'nature', 'beaches', 
                            'nightlife', 'cuisine', 'wellness', 'urban', 'seclusion']
        
        # Create normalized feature matrix (0-1 scale)
        features = self.df[self.feature_cols].values
        self.feature_matrix = (features - 1) / 4  # Convert 1-5 scale to 0-1
        self.feature_matrix = np.clip(self.feature_matrix, 0, 1)
        
        self.mapper = PersonalityToTravelMapper()
    
    def recommend(self, user_preferences, top_n=5):
        """
        Get top N recommendations based on user travel preferences
        
        Args:
            user_preferences: np.array of shape (9,) with values 0-1
            top_n: number of recommendations
        
        Returns:
            DataFrame with top N destinations and match scores
        """
        
        # Reshape for cosine similarity
        user_vector = user_preferences.reshape(1, -1)
        
        # Calculate cosine similarity with all destinations
        similarities = cosine_similarity(user_vector, self.feature_matrix)[0]
        
        # Add to dataframe
        results = self.df.copy()
        results['match_score'] = similarities
        
        # Remove duplicates (keep first occurrence of each city-country)
        results = results.drop_duplicates(subset=['city', 'country'], keep='first')
        
        # Get top N
        top_recommendations = results.nlargest(top_n, 'match_score')[
            ['city', 'country', 'avg_rating', 'avg_cost_(usd/day)', 'best_season', 'match_score']
        ].reset_index(drop=True)
        
        return top_recommendations
    
    def recommend_by_mbti(self, mbti_type, top_n=5):
        """Get recommendations from 16 Personalities MBTI type"""
        user_prefs = self.mapper.mbti_to_travel_preferences(mbti_type)
        return self.recommend(user_prefs, top_n=top_n)
    
    def explain_match(self, city, country, user_preferences):
        """Explain why a destination matches the user"""
        
        match = self.df[(self.df['city'] == city) & (self.df['country'] == country)]
        if match.empty:
            return None
        
        dest_features = match[self.feature_cols].iloc[0].values
        dest_features_norm = (dest_features - 1) / 4
        
        # Calculate alignment for each feature
        feature_alignment = []
        for i, feat in enumerate(self.feature_cols):
            user_pref = user_preferences[i]
            dest_score = dest_features_norm[i]
            alignment = 1 - abs(user_pref - dest_score)
            feature_alignment.append((feat, user_pref, dest_score, alignment))
        
        # Top 3 matches
        feature_alignment.sort(key=lambda x: x[3], reverse=True)
        
        return feature_alignment[:3]
    
    def get_recommendations(self, personality_input, top_n=5):
        """
        Accepts 16 Personalities MBTI string or Big Five dict,
        returns top destinations as a list of dicts.
        """
        if isinstance(personality_input, str):
            prefs = self.mapper.mbti_to_travel_preferences(personality_input)
        elif isinstance(personality_input, dict):
            prefs = self.mapper.big5_to_travel_preferences(personality_input)
        else:
            raise ValueError("Invalid personality input")

        rec_df = self.recommend(prefs, top_n=top_n)
        return rec_df.to_dict(orient='records')



# ============================================================================
# PART 4: MAIN APPLICATION
# ============================================================================

def print_header():
    print("\n" + "="*70)
    print(" 🌍 PERSONALITY-BASED TRAVEL RECOMMENDER 🌍")
    print("="*70)


def get_user_personality():
    """Get personality input from user"""
    print("\n" + "-"*70)
    print("HOW TO USE:")
    print("-"*70)
    print("\nEnter your 16 Personalities type (MBTI)")
    print("Examples: ENFP, ISTJ, INFP, ESTP, INTP, etc.\n")
    print("Don't know your type?")
    print("  → Take the free test at: https://www.16personalities.com/\n")
    print("Personality codes:")
    print("  E=Extroverted  | I=Introverted")
    print("  S=Sensing      | N=iNtuitive")
    print("  T=Thinking     | F=Feeling")
    print("  J=Judging      | P=Perceiving\n")
    
    while True:
        user_input = input("👤 Enter your MBTI type (or 'demo' for example): ").strip().upper()
        
        if user_input == 'DEMO':
            return 'ENFP'
        
        if len(user_input) == 4 and all(c in 'EISTJFPN' for c in user_input):
            # Validate format
            if user_input[0] in 'EI' and user_input[1] in 'SN' and user_input[2] in 'TF' and user_input[3] in 'JP':
                return user_input
        
        print("❌ Invalid format. Please try again (e.g., ENFP)")


def display_recommendations(recommendations, mbti_type):
    """Display recommendations in a nice format"""
    
    print("\n" + "="*70)
    print(f"🎯 TOP DESTINATIONS FOR {mbti_type} PERSONALITY")
    print("="*70 + "\n")
    
    for idx, (_, row) in enumerate(recommendations.iterrows(), 1):
        match_pct = row['match_score'] * 100
        bar_length = int(match_pct / 5)
        bar = "█" * bar_length + "░" * (20 - bar_length)
        
        print(f"{idx}. {row['city']}, {row['country']}")
        print(f"   Match: {bar} {match_pct:.0f}%")
        print(f"   ⭐ Rating: {row['avg_rating']:.1f}/5.0  |  💰 ${row['avg_cost_(usd/day)']:.0f}/day  |  🗓️  {row['best_season']}")
        print()


def main():
    """Main application flow"""
    
    print_header()
    
   # Load dataset
    #print("\n📂 Loading travel destinations...")
    #df = pd.DataFrame(DESTINATIONS_DATA)
    #print(f"✅ Loaded {len(df)} destinations")
    
    # Create recommender
    print("🎯 Initializing recommender system...")
    recommender = TravelRecommender(df)
    print("✅ Ready!\n")
    
    # Get user personality
    mbti_type = get_user_personality()
    
    # Get recommendations
    print(f"\n🔍 Finding best destinations for {mbti_type} personalities...")
    print("(This means: " + explain_mbti(mbti_type) + ")\n")
    
    recommendations = recommender.recommend_by_mbti(mbti_type, top_n=5)
    
    # Display results
    display_recommendations(recommendations, mbti_type)
    
    # Show why matches work
    print("-"*70)
    print("WHY THESE MATCHES?\n")
    
    user_prefs = recommender.mapper.mbti_to_travel_preferences(mbti_type)
    
    for idx, (_, row) in enumerate(recommendations.iterrows(), 1):
        city = row['city']
        country = row['country']
        alignment = recommender.explain_match(city, country, user_prefs)
        
        if alignment:
            print(f"{idx}. {city}, {country}")
            for feat, user_score, dest_score, match_strength in alignment:
                print(f"   • {feat.upper()}: You like it {user_score*5+1:.1f}/5, it has {dest_score*4+1:.1f}/5 ({match_strength*100:.0f}% match)")
            print()
    
    # Save option
    print("-"*70)
    save = input("\n💾 Save recommendations to CSV? (y/n): ").strip().lower()
    if save == 'y':
        recommendations.to_csv('recommendations.csv', index=False)
        print("✅ Saved to recommendations.csv")
    
    print("\n" + "="*70)
    print("✨ Thanks for using Travel Recommender!")
    print("="*70 + "\n")


def explain_mbti(mbti_type):
    """Explain what an MBTI type means"""
    
    explanations = {
        'E': 'extroverted',
        'I': 'introverted',
        'S': 'practical',
        'N': 'innovative',
        'T': 'logical',
        'F': 'empathetic',
        'J': 'organized',
        'P': 'spontaneous'
    }
    
    parts = [explanations.get(c, c) for c in mbti_type]
    return f"{parts[0]}, {parts[1]}, {parts[2]}, {parts[3]}"


if __name__ == "__main__":
    main()
