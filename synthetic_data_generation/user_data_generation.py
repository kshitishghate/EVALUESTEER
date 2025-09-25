import json
import pandas as pd
import numpy as np
import itertools
from collections import defaultdict
import os
from tqdm import tqdm

def load_distinct_value_profiles(file_path):
    """Load the distinct value profiles from JSON file"""
    with open(file_path, 'r') as f:
        return json.load(f)

def load_question_metadata(is_refined=False):
    """Load question metadata from statements_meta_data.jsonl"""
    metadata_path = "../indievalue/data/meta_data/refined_statements_meta_data.jsonl" if is_refined else "../indievalue/data/meta_data/statements_meta_data.jsonl"
    metadata = []
    with open(metadata_path, 'r') as f:
        for line in f:
            metadata.append(json.loads(line))
    return pd.DataFrame(metadata)

def load_synthetic_data(file_path):
    """Load the synthetic data generated with style variations"""
    with open(file_path, 'r', encoding="utf-8") as f:
        return json.load(f)

def generate_all_style_combinations():
    """
    Generate all 2^4 = 16 combinations of style preferences for 4 families:
    verbosity, readability, confidence, sentiment
    """
    style_families = {
        "verbosity": ["verbose", "concise"],
        "readability": ["high_reading_difficulty", "low_reading_difficulty"],
        "confidence": ["high_confidence", "low_confidence"], 
        "sentiment": ["warm", "cold"]
    }
    
    # Get all possible combinations (2^4 = 16)
    family_names = list(style_families.keys())
    all_combinations = []
    
    for combination in itertools.product(*[style_families[family] for family in family_names]):
        style_profile = dict(zip(family_names, combination))
        all_combinations.append(style_profile)
    
    return all_combinations

def create_style_profile_code(style_profile):
    """Create a compact code for a style profile"""
    style_codes = {
        "verbose": "v", "concise": "c",
        "high_reading_difficulty": "hrd", "low_reading_difficulty": "lrd",
        "high_confidence": "hc", "low_confidence": "lc", 
        "warm": "w", "cold": "co"
    }
    
    codes = []
    for family in ["verbosity", "readability", "confidence", "sentiment"]:
        preference = style_profile[family]
        codes.append(style_codes[preference])
    
    return "_".join(codes)

def create_user_profile_combinations():
    """
    Create all combinations of value profiles (18) and style profiles (16).
    Returns a list of 288 user profile combinations.
    """
    # Load value profiles
    value_profiles_data = load_distinct_value_profiles('wvs_user_profile_generation/distinct_value_profiles.json')
    value_profiles = []
    
    # Extract the distinct profiles from all quadrants
    for quadrant, profiles in value_profiles_data['distinct_profiles'].items():
        for profile in profiles:
            value_profiles.append({
                'value_profile_id': profile['user_id'],
                'quadrant': profile['quadrant'],
                'profile_label': profile['profile_label'],
                'diversity_score': profile['diversity_score'],
                'demographics': profile['demographics'],
                'key_responses': profile['key_responses']
            })
    
    # Generate all style combinations
    style_combinations = generate_all_style_combinations()
    
    # Create all user profile combinations
    user_profile_combinations = []
    
    for i, (value_profile, style_profile) in enumerate(itertools.product(value_profiles, style_combinations)):
        user_profile_id = f"user_profile_{i:03d}"
        style_code = create_style_profile_code(style_profile)
        
        combined_profile = {
            'user_profile_id': user_profile_id,
            'value_profile_id': value_profile['value_profile_id'],
            'quadrant': value_profile['quadrant'],
            'value_profile_label': value_profile['profile_label'],
            'style_profile': style_profile,
            'style_code': style_code,
            'diversity_score': value_profile['diversity_score'],
            'demographics': value_profile['demographics'],
            'key_responses': value_profile['key_responses']
        }
        
        user_profile_combinations.append(combined_profile)
    
    return user_profile_combinations

def map_wvs_responses_to_completions(metadata_df, synthetic_data):
    """Create a mapping between WVS responses and synthetic data completions."""
    response_mappings = {}
    
    for item in synthetic_data:
        question_id = item['question_id']
        
        # Find the metadata for this question
        q_metadata_rows = metadata_df[metadata_df['question_id'] == question_id]
        
        if len(q_metadata_rows) == 0:
            print(f"Warning: No metadata found for question {question_id}")
            continue
            
        q_metadata = q_metadata_rows.iloc[0]
        
        # Get the answer_ids_to_grouped_answer_ids mapping
        answer_mapping = q_metadata['answer_ids_to_grouped_answer_ids']
        
        # Create a mapping from answer_id to completion (A or B)
        response_to_completion = {}
        
        for aid, group_id in answer_mapping.items():
            # Convert aid to int since it's stored as string in the metadata
            aid_int = int(aid)
            
            # Map to completion based on group_id
            if group_id == 0:
                response_to_completion[aid_int] = "completion_A"
            else:
                response_to_completion[aid_int] = "completion_B"
        
        response_mappings[question_id] = response_to_completion
    
    return response_mappings

def generate_preference_combinations_for_family(family_name, style_profile):
    """
    Generate all preference combinations for a specific style family.
    Returns a list of tuples (preferred_key, non_preferred_key, combination_type, preference_rule)
    """
    family_styles = {
        "verbosity": ["verbose", "concise"],
        "readability": ["high_reading_difficulty", "low_reading_difficulty"],
        "confidence": ["high_confidence", "low_confidence"],
        "sentiment": ["warm", "cold"]
    }
    
    if family_name not in family_styles:
        raise ValueError(f"Unknown family: {family_name}")
    
    style1, style2 = family_styles[family_name]
    user_preferred_style = style_profile[family_name]
    
    combinations = []
    
    # A vs B combinations - preference based on WVS response
    combinations.extend([
        (f"completion_A_{style1}", f"completion_B_{style1}", f"A_{style1}_vs_B_{style1}", "wvs_based"),
        (f"completion_A_{style2}", f"completion_B_{style1}", f"A_{style2}_vs_B_{style1}", "wvs_based"),
        (f"completion_A_{style1}", f"completion_B_{style2}", f"A_{style1}_vs_B_{style2}", "wvs_based"),
        (f"completion_A_{style2}", f"completion_B_{style2}", f"A_{style2}_vs_B_{style2}", "wvs_based")
    ])
    
    # Within-family comparisons - based on user's style preference
    combinations.extend([
        (f"completion_A_{user_preferred_style}", f"completion_A_{style1 if user_preferred_style != style1 else style2}", 
         f"A_{style1}_vs_A_{style2}", f"prefer_{user_preferred_style}"),
        (f"completion_B_{user_preferred_style}", f"completion_B_{style1 if user_preferred_style != style1 else style2}", 
         f"B_{style1}_vs_B_{style2}", f"prefer_{user_preferred_style}")
    ])
    
    return combinations

def generate_user_preferences_for_profile(user_profile, synthetic_data, response_mappings):
    """
    Generate user preferences for a specific user profile across all style families.
    """
    user_preferences = []
    
    # Create a dictionary for faster lookup of synthetic data by question_id
    synthetic_data_dict = {item['question_id']: item for item in synthetic_data}
    
    # Style families to process
    style_families = ["verbosity", "readability", "confidence", "sentiment"]
    
    # For each style family
    for family_name in style_families:
        # Get preference combinations for this family
        preference_combinations = generate_preference_combinations_for_family(family_name, user_profile['style_profile'])
        
        # For each question that has a mapping and synthetic data
        for question_id, response_mapping in response_mappings.items():
            # Skip if this question doesn't have synthetic data
            if question_id not in synthetic_data_dict:
                continue
                
            synthetic_item = synthetic_data_dict[question_id]
            
            # Check if all required style variations exist for this family
            family_styles = {
                "verbosity": ["verbose", "concise"],
                "readability": ["high_reading_difficulty", "low_reading_difficulty"],
                "confidence": ["high_confidence", "low_confidence"],
                "sentiment": ["warm", "cold"]
            }
            
            style1, style2 = family_styles[family_name]
            required_keys = [
                f"completion_A_{style1}", f"completion_B_{style1}",
                f"completion_A_{style2}", f"completion_B_{style2}"
            ]
            
            if not all(key in synthetic_item for key in required_keys):
                print(f"Warning: Missing {family_name} variations for question {question_id}")
                continue
            
            # Get the user's response to this WVS question from their key_responses
            user_response = user_profile['key_responses'].get(question_id)
            
            # Skip if the user didn't answer this question
            if user_response is None:
                continue
            
            # Skip if the user's response isn't in our mapping
            if user_response not in response_mapping:
                continue
            
            # Generate preferences for each combination
            for comp1_key, comp2_key, combination_type, preference_rule in preference_combinations:
                # Determine preference based on the rule
                if preference_rule == "wvs_based":
                    # For A vs B comparisons, use WVS response to determine preference
                    user_preferred_base = response_mapping[user_response]  # Either "completion_A" or "completion_B"
                    
                    # Check if comp1 or comp2 aligns with the user's preferred base
                    comp1_base = comp1_key.split('_')[0] + "_" + comp1_key.split('_')[1]  # e.g., "completion_A"
                    comp2_base = comp2_key.split('_')[0] + "_" + comp2_key.split('_')[1]  # e.g., "completion_B"
                    
                    if comp1_base == user_preferred_base:
                        preferred_completion = synthetic_item[comp1_key]
                        non_preferred_completion = synthetic_item[comp2_key]
                        actual_preferred_key = comp1_key
                        actual_non_preferred_key = comp2_key
                    else:
                        preferred_completion = synthetic_item[comp2_key]
                        non_preferred_completion = synthetic_item[comp1_key]
                        actual_preferred_key = comp2_key
                        actual_non_preferred_key = comp1_key
                        
                elif preference_rule.startswith("prefer_"):
                    # For within-family comparisons, use the style preference
                    preferred_style = preference_rule.replace("prefer_", "")
                    
                    if comp1_key.endswith(preferred_style):
                        preferred_completion = synthetic_item[comp1_key]
                        non_preferred_completion = synthetic_item[comp2_key]
                        actual_preferred_key = comp1_key
                        actual_non_preferred_key = comp2_key
                    else:
                        preferred_completion = synthetic_item[comp2_key]
                        non_preferred_completion = synthetic_item[comp1_key]
                        actual_preferred_key = comp2_key
                        actual_non_preferred_key = comp1_key
                
                # Create a preference data point
                preference = {
                    "user_profile_id": user_profile['user_profile_id'],
                    "value_profile_id": user_profile['value_profile_id'],
                    "quadrant": user_profile['quadrant'],
                    "value_profile_label": user_profile['value_profile_label'],
                    "style_code": user_profile['style_code'],
                    "question_id": question_id,
                    "style_family": family_name,
                    "combination_type": combination_type,
                    "preference_rule": preference_rule,
                    "style_profile": user_profile['style_profile'],
                    "prompt": synthetic_item["prompt"],
                    "preferred_completion": preferred_completion,
                    "non_preferred_completion": non_preferred_completion,
                    "preferred_completion_key": actual_preferred_key,
                    "non_preferred_completion_key": actual_non_preferred_key,
                    "wvs_response": user_response,
                    "wvs_question": synthetic_item["wvs_question"],
                    "statement_1": synthetic_item["statement_1"],
                    "statement_2": synthetic_item["statement_2"],
                    "demographics": user_profile['demographics']
                }
                
                user_preferences.append(preference)
    
    return user_preferences

def create_user_preference_dataset(user_preferences):
    """Create a dataset with unique IDs for each user-question-combination triplet."""
    # Create a DataFrame from the user preferences
    df = pd.DataFrame(user_preferences)
    
    # Add a unique ID for each preference pair
    df['preference_id'] = [f"pref_{i:06d}" for i in range(len(df))]
    
    return df

def create_preference_statistics(user_profile_combinations):
    """Create statistics about the preference combinations."""
    stats = {
        "total_user_profiles": len(user_profile_combinations),
        "total_value_profiles": len(set(p['value_profile_id'] for p in user_profile_combinations)),
        "total_style_profiles": len(set(p['style_code'] for p in user_profile_combinations)),
        "quadrant_distribution": {},
        "style_family_combinations": {},
        "efficient_representation": {}
    }
    
    # Count quadrant distribution
    for profile in user_profile_combinations:
        quadrant = profile['quadrant']
        stats["quadrant_distribution"][quadrant] = stats["quadrant_distribution"].get(quadrant, 0) + 1
    
    # Create efficient representation
    unique_style_codes = sorted(set(p['style_code'] for p in user_profile_combinations))
    unique_value_profiles = sorted(set(p['value_profile_id'] for p in user_profile_combinations))
    
    stats["efficient_representation"] = {
        "style_codes": unique_style_codes,
        "value_profile_ids": unique_value_profiles,
        "total_combinations": len(unique_style_codes) * len(unique_value_profiles)
    }
    
    # Style family breakdown
    for profile in user_profile_combinations:
        style_profile = profile['style_profile']
        for family, preference in style_profile.items():
            if family not in stats["style_family_combinations"]:
                stats["style_family_combinations"][family] = {}
            stats["style_family_combinations"][family][preference] = \
                stats["style_family_combinations"][family].get(preference, 0) + 1
    
    return stats

def generate_compact_dataset(user_profile_combinations, synthetic_data, response_mappings, max_profiles=None):
    """
    Generate preferences for a subset of user profiles for efficiency.
    If max_profiles is None, generate for all profiles.
    """
    if max_profiles:
        selected_profiles = user_profile_combinations[:max_profiles]
        print(f"Generating preferences for {max_profiles} user profiles (out of {len(user_profile_combinations)} total)")
    else:
        selected_profiles = user_profile_combinations
        print(f"Generating preferences for all {len(user_profile_combinations)} user profiles")
    
    all_user_preferences = []
    
    for user_profile in tqdm(selected_profiles, desc="Generating user preferences"):
        profile_preferences = generate_user_preferences_for_profile(
            user_profile, synthetic_data, response_mappings
        )
        all_user_preferences.extend(profile_preferences)
    
    return all_user_preferences

def main():
    print("="*80)
    print("ENHANCED USER PREFERENCE DATA GENERATION ")
    print("="*80)
    print("Features:")
    print("- Uses 18 distinct value profiles from WVS data")
    print("- Generates all 16 style profile combinations (2^4)")
    print("- Focuses on 4 style families: verbosity, readability, confidence, sentiment")
    print("- Creates 288 total user profile combinations (18 × 16)")
    print("="*80)
    
    # Generate all user profile combinations
    print("\nGenerating user profile combinations...")
    user_profile_combinations = create_user_profile_combinations()
    print(f"Created {len(user_profile_combinations)} user profile combinations")
    
    # Create statistics
    print("\nCreating preference statistics...")
    stats = create_preference_statistics(user_profile_combinations)
    
    print(f"\nStatistics:")
    print(f"Total user profiles: {stats['total_user_profiles']}")
    print(f"Unique value profiles: {stats['total_value_profiles']}")
    print(f"Unique style profiles: {stats['total_style_profiles']}")
    print(f"\nQuadrant distribution:")
    for quadrant, count in stats['quadrant_distribution'].items():
        print(f"  {quadrant}: {count}")
    
    print(f"\nStyle family combinations:")
    for family, preferences in stats['style_family_combinations'].items():
        print(f"  {family}:")
        for pref, count in preferences.items():
            print(f"    {pref}: {count}")
    
    # Save user profile combinations
    print(f"\nSaving user profile combinations...")
    with open('user_profile_combinations_.json', 'w', encoding='utf-8') as f:
        json.dump({
            'statistics': stats,
            'user_profiles': user_profile_combinations
        }, f, indent=2, ensure_ascii=False)
    
    # Ask user about preference generation
    print(f"\nDo you want to generate preference data? This will be computationally intensive.")
    print(f"Options:")
    print(f"1. Generate for all {len(user_profile_combinations)} profiles (full dataset)")
    print(f"2. Generate for a subset (e.g., 50 profiles for testing)")
    print(f"3. Skip preference generation (only create profile combinations)")
    
    choice = input("Enter your choice (1/2/3): ").strip()
    
    if choice == "3":
        print("Skipping preference generation. User profile combinations saved.")
        return
    
    # Load required data for preference generation
    print("\nLoading required data...")
    
    # Load question metadata
    print("Loading question metadata...")
    metadata_df = load_question_metadata(is_refined=False)
    print(f"Loaded metadata for {len(metadata_df)} questions")
    
    # Load the synthetic data with style variations (including readability)
    synthetic_file = 'prism_wvs_generated_data_with_style_variations.json'
    if not os.path.exists(synthetic_file):
        print(f"Error: Synthetic data file {synthetic_file} not found.")
        print("Please ensure you have generated style variations including readability.")
        return
    
    print("Loading synthetic data with style variations...")
    synthetic_data = load_synthetic_data(synthetic_file)
    print(f"Loaded {len(synthetic_data)} synthetic data points")
    
    # Map WVS responses to synthetic data completions
    print("Mapping WVS responses to synthetic data completions...")
    response_mappings = map_wvs_responses_to_completions(metadata_df, synthetic_data)
    print(f"Created response mappings for {len(response_mappings)} questions")
    
    # Generate preferences based on user choice
    if choice == "1":
        max_profiles = None
        output_suffix = "full"
    elif choice == "2":
        try:
            max_profiles = int(input("Enter number of profiles to process: "))
            output_suffix = f"subset_{max_profiles}"
        except ValueError:
            max_profiles = 50
            output_suffix = "subset_50"
            print("Invalid input, using 50 profiles")
    
    print(f"\nGenerating user preferences...")
    all_user_preferences = generate_compact_dataset(
        user_profile_combinations, synthetic_data, response_mappings, max_profiles
    )
    
    print(f"\nTotal preferences generated: {len(all_user_preferences)}")
    
    if len(all_user_preferences) == 0:
        print("No preferences generated. Check if synthetic data has required style variations.")
        return
    
    # Create user preference dataset
    print("Creating user preference dataset...")
    preference_df = create_user_preference_dataset(all_user_preferences)
    
    # Print summary statistics
    print(f"\nDataset Summary:")
    print(f"Total preference pairs: {len(preference_df)}")
    print(f"Unique user profiles: {len(preference_df['user_profile_id'].unique())}")
    print(f"Unique questions: {len(preference_df['question_id'].unique())}")
    print(f"\nStyle families:")
    for family, count in preference_df['style_family'].value_counts().items():
        print(f"  {family}: {count}")
    print(f"\nQuadrants:")
    for quadrant, count in preference_df['quadrant'].value_counts().items():
        print(f"  {quadrant}: {count}")
    
    # Save preference dataset
    output_file = f"user_preference_dataset__{output_suffix}.json"
    print(f"\nSaving user preference dataset to {output_file}...")
    
    with open(output_file, 'w', encoding="utf-8") as f:
        json.dump(preference_df.to_dict('records'), f, indent=2, ensure_ascii=False)
    
    print("User preference dataset generation complete!")
    print(f"Output files:")
    print(f"  - User profiles: user_profile_combinations_.json")
    print(f"  - Preference dataset: {output_file}")

if __name__ == "__main__":
    main() 
