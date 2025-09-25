import json
import pandas as pd
import numpy as np
from scipy import stats
from collections import defaultdict

def load_wvs_data(file_path):
    """Load WVS data from JSONL file"""
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return pd.DataFrame(data)

def transform_god_importance(df):
    """Q164: God importance - 1 (not important) to 10 (very important)"""
    # Traditional response: 8-10, already correctly scaled
    return df['Q164_aid'].fillna(df['Q164_aid'].median())

def transform_child_qualities(df):
    """Q7-Q17: Child qualities - Traditional values obedience & religious faith vs. independence"""
    # Traditional values: Q17 (Obedience), Q15 (Religious faith)  
    # Secular values: Q8 (Independence), Q14 (Determination)
    # 1 = mentioned, 2 = not mentioned
    
    traditional_score = 0
    
    # Q17: Obedience (traditional value)
    if 'Q17_aid' in df.columns:
        # 1 = mentioned (traditional), 2 = not mentioned
        # Convert to traditional score: mentioned = +1, not mentioned = -1
        obedience_score = df['Q17_aid'].fillna(2).apply(lambda x: 1 if x == 1 else -1)
        traditional_score += obedience_score
    
    # Q15: Religious faith (traditional value)
    if 'Q15_aid' in df.columns:
        faith_score = df['Q15_aid'].fillna(2).apply(lambda x: 1 if x == 1 else -1)
        traditional_score += faith_score
    
    # Q8: Independence (secular value) - reverse scoring
    if 'Q8_aid' in df.columns:
        # Independence mentioned = less traditional
        independence_score = df['Q8_aid'].fillna(2).apply(lambda x: -1 if x == 1 else 1)
        traditional_score += independence_score
    
    # Q14: Determination (secular value) - reverse scoring  
    if 'Q14_aid' in df.columns:
        determination_score = df['Q14_aid'].fillna(2).apply(lambda x: -1 if x == 1 else 1)
        traditional_score += determination_score
    
    # Scale to positive values (add 4 to shift range from [-4,4] to [0,8])
    return traditional_score + 4

def transform_abortion_justification(df):
    """Q184: Abortion justification - 1 (never) to 10 (always)"""
    # Traditional response: 1 (never justifiable)
    # Transform: Reverse so higher = more traditional
    scores = df['Q184_aid'].fillna(df['Q184_aid'].median())
    return 11 - scores

def transform_national_pride(df):
    """Q254: National pride - 1 (very proud) to 4 (not proud)"""
    # Traditional response: 1-2 (very/quite proud)
    # Transform: Reverse so higher = more traditional
    scores = df['Q254_aid'].fillna(df['Q254_aid'].median())
    return 5 - scores

def transform_respect_authority(df):
    """Q45: Respect for authority - 1 (strongly agree) to 5 (strongly disagree)"""
    # Traditional response: 1-2 (agree with more respect)
    # Transform: Reverse so higher = more traditional
    scores = df['Q45_aid'].fillna(df['Q45_aid'].median())
    return 6 - scores

def transform_materialism(df):
    """Q152-Q153: Materialism vs Post-materialism priorities"""
    # Materialist/Survival priorities: 1 (economic growth), 2 (strong defense)
    # Post-materialist/Self-expression: 3 (more say in jobs/communities), 4 (beautiful cities)
    # Higher score = more materialist/survival-oriented
    
    survival_score = 0
    
    # Q152: Most important goal
    if 'Q152_aid' in df.columns:
        q152_materialist = df['Q152_aid'].fillna(3).apply(lambda x: 2 if x in [1, 2] else 0)
        survival_score += q152_materialist
    
    # Q153: Next most important goal  
    if 'Q153_aid' in df.columns:
        q153_materialist = df['Q153_aid'].fillna(3).apply(lambda x: 2 if x in [1, 2] else 0)
        survival_score += q153_materialist
    
    return survival_score

def transform_happiness(df):
    """Q46: Happiness level - 1 (very happy) to 4 (not happy)"""
    # Survival response: higher values (not very happy)
    # Already correctly scaled
    return df['Q46_aid'].fillna(df['Q46_aid'].median())

def transform_homosexuality_justification(df):
    """Q182: Homosexuality justification - 1 (never) to 10 (always)"""
    # Survival/Traditional response: 1 (never justifiable)
    # Transform: Reverse so higher = more survival/traditional
    scores = df['Q182_aid'].fillna(df['Q182_aid'].median())
    return 11 - scores

def transform_petition_participation(df):
    """Q209: Political participation - 1 (have done) to 3 (would never do)"""
    # Survival response: 3 (would never do)
    # Already correctly scaled
    return df['Q209_aid'].fillna(df['Q209_aid'].median())

def transform_interpersonal_trust(df):
    """Q57: Interpersonal trust - 1 (can't be careful) to 2 (can trust)"""
    # Survival response: 1 (can't be too careful)
    # Transform: Reverse so higher = more survival-oriented
    scores = df['Q57_aid'].fillna(df['Q57_aid'].median())
    return 3 - scores

def calculate_dimension_scores(df):
    """Calculate Traditional/Secular and Survival/Self-Expression scores"""
    
    # Traditional vs. Secular-Rational dimension
    traditional_vars = {
        'god_trad_score': transform_god_importance(df),
        'child_trad_score': transform_child_qualities(df),
        'abortion_trad_score': transform_abortion_justification(df),
        'pride_trad_score': transform_national_pride(df),
        'authority_trad_score': transform_respect_authority(df)
    }
    
    # Survival vs. Self-Expression dimension
    survival_vars = {
        'materialism_surv_score': transform_materialism(df),
        'happiness_surv_score': transform_happiness(df),
        'homosexuality_surv_score': transform_homosexuality_justification(df),
        'petition_surv_score': transform_petition_participation(df),
        'trust_surv_score': transform_interpersonal_trust(df)
    }
    
    # Calculate composite scores
    traditional_score = pd.DataFrame(traditional_vars).mean(axis=1)
    survival_score = pd.DataFrame(survival_vars).mean(axis=1)
    
    # Standardize scores
    traditional_z = stats.zscore(traditional_score, nan_policy='omit')
    survival_z = stats.zscore(survival_score, nan_policy='omit')
    
    return traditional_z, survival_z, traditional_vars, survival_vars

def classify_quadrant(traditional_z, survival_z, threshold=0):
    """Classify into cultural quadrants"""
    if traditional_z >= threshold and survival_z >= threshold:
        return "Traditional-Survival"
    elif traditional_z >= threshold and survival_z < threshold:
        return "Traditional-Self-Expression"
    elif traditional_z < threshold and survival_z >= threshold:
        return "Secular-Survival"
    else:
        return "Secular-Self-Expression"

def main():
    # Load data
    print("Loading WVS data...")
    df = load_wvs_data('indievalue/IndieValue/demographics_in_nl_statements_combined_full_set.jsonl')
    
    print(f"Loaded {len(df)} respondents")
    
    # Filter for valid responses on key questions
    key_questions = ['Q164_aid', 'Q17_aid', 'Q15_aid', 'Q8_aid', 'Q14_aid', 'Q184_aid', 'Q254_aid', 
                     'Q45_aid', 'Q152_aid', 'Q153_aid', 'Q46_aid', 'Q182_aid', 
                     'Q209_aid', 'Q57_aid']
    
    # Check for missing values (allowing for some flexibility)
    valid_mask = pd.Series([True] * len(df))
    for q in key_questions:
        if q in df.columns:
            # Allow some missing values but not too many
            valid_mask &= (df[q] != -99) & (df[q] != -2) & (df[q] != -4)
    
    df_clean = df[valid_mask].copy()
    print(f"Clean sample with valid responses: {len(df_clean)} respondents")
    
    # Calculate dimension scores
    print("Calculating dimension scores...")
    traditional_z, survival_z, traditional_vars, survival_vars = calculate_dimension_scores(df_clean)
    
    # Add scores to dataframe
    df_clean['traditional_z'] = traditional_z
    df_clean['survival_z'] = survival_z
    
    # Classify quadrants
    print("Classifying quadrants...")
    df_clean['quadrant'] = df_clean.apply(
        lambda row: classify_quadrant(row['traditional_z'], row['survival_z']), 
        axis=1
    )
    
    # Remove any rows with NaN in quadrant classification
    df_clean = df_clean.dropna(subset=['quadrant'])
    
    # Create output structure
    quadrant_groups = defaultdict(list)
    
    for _, row in df_clean.iterrows():
        user_id = row['D_INTERVIEW']
        quadrant = row['quadrant']
        quadrant_groups[quadrant].append(int(user_id))
    
    # Print statistics
    print("\n=== QUADRANT DISTRIBUTION ===")
    for quadrant, users in quadrant_groups.items():
        print(f"{quadrant}: {len(users)} users")
    
    # Print sample characteristics
    print("\n=== SAMPLE CHARACTERISTICS ===")
    quadrant_stats = df_clean.groupby('quadrant')[['traditional_z', 'survival_z']].agg(['mean', 'std', 'count'])
    print(quadrant_stats.round(3))
    
    # Save results
    output_file = 'wvs_quadrant_classification.json'
    with open(output_file, 'w') as f:
        json.dump(dict(quadrant_groups), f, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    
    # Show sanity check examples
    print("\n=== SANITY CHECK EXAMPLES ===")
    show_sanity_check_examples(df_clean, traditional_vars, survival_vars)
    
    return df_clean, quadrant_groups

def show_sanity_check_examples(df, traditional_vars, survival_vars):
    """Show example responses from each quadrant for sanity checking"""
    
    for quadrant in df['quadrant'].unique():
        print(f"\n--- {quadrant} ---")
        quadrant_df = df[df['quadrant'] == quadrant]
        
        if len(quadrant_df) > 0:
            # Get a sample user
            sample_user = quadrant_df.iloc[0]
            
            print(f"User ID: {sample_user['D_INTERVIEW']}")
            print(f"Traditional Score: {sample_user['traditional_z']:.3f}")
            print(f"Survival Score: {sample_user['survival_z']:.3f}")
            
            # Show key question responses
            print("Key responses:")
            print(f"  God importance (Q164): {sample_user.get('Q164_aid', 'N/A')}")
            print(f"  Child qualities - Obedience (Q17): {sample_user.get('Q17_aid', 'N/A')} (1=mentioned, 2=not)")
            print(f"  Child qualities - Religious faith (Q15): {sample_user.get('Q15_aid', 'N/A')} (1=mentioned, 2=not)")
            print(f"  Child qualities - Independence (Q8): {sample_user.get('Q8_aid', 'N/A')} (1=mentioned, 2=not)")
            print(f"  Child qualities - Determination (Q14): {sample_user.get('Q14_aid', 'N/A')} (1=mentioned, 2=not)")
            print(f"  Abortion justification (Q184): {sample_user.get('Q184_aid', 'N/A')}")
            print(f"  National pride (Q254): {sample_user.get('Q254_aid', 'N/A')}")
            print(f"  Respect authority (Q45): {sample_user.get('Q45_aid', 'N/A')}")
            print(f"  Materialism priority 1 (Q152): {sample_user.get('Q152_aid', 'N/A')} (1=econ growth, 2=defense, 3=participation, 4=beauty)")
            print(f"  Materialism priority 2 (Q153): {sample_user.get('Q153_aid', 'N/A')} (1=econ growth, 2=defense, 3=participation, 4=beauty)")
            print(f"  Happiness level (Q46): {sample_user.get('Q46_aid', 'N/A')}")
            print(f"  Homosexuality justification (Q182): {sample_user.get('Q182_aid', 'N/A')}")
            print(f"  Petition participation (Q209): {sample_user.get('Q209_aid', 'N/A')}")
            print(f"  Interpersonal trust (Q57): {sample_user.get('Q57_aid', 'N/A')}")

if __name__ == "__main__":
    df_clean, quadrant_groups = main() 