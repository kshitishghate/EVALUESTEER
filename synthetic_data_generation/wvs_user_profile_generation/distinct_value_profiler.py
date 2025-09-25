import json
import pandas as pd
import numpy as np
from collections import defaultdict

def calculate_user_diversity(df):
    """
    Calculate diversity of responses for each user across all WVS questions.
    Returns a diversity score for each user.
    """
    # Get all question columns (Q1, Q2, etc.)
    question_cols = [col for col in df.columns if col.startswith('Q') and not col.endswith('_aid')]
    
    # Calculate diversity score for each user
    diversity_scores = {}
    
    for _, user_row in df.iterrows():
        user_id = user_row['D_INTERVIEW']
        
        # Count unique responses for this user
        response_counts = defaultdict(int)
        valid_responses = 0
        
        for q_col in question_cols:
            if pd.notna(user_row[q_col]):
                response_counts[user_row[q_col]] += 1
                valid_responses += 1
        
        if valid_responses == 0:
            diversity_scores[user_id] = 0
            continue
        
        # Calculate entropy as diversity measure
        probabilities = [count/valid_responses for count in response_counts.values()]
        entropy = -sum(p * np.log2(p) for p in probabilities)
        
        # Coverage factor
        coverage_factor = min(1.0, valid_responses / len(question_cols))
        
        # Combine entropy and coverage for final diversity score
        diversity_scores[user_id] = entropy * coverage_factor
    
    return diversity_scores

def load_wvs_data():
    """Load WVS data and classification results"""
    # Load original data
    df = pd.read_json('indievalue/IndieValue/demographics_in_nl_statements_combined_full_set.jsonl', lines=True)
    
    # Load classification results
    with open('wvs_quadrant_classification.json', 'r') as f:
        quadrant_groups = json.load(f)
    
    return df, quadrant_groups

def define_value_items():
    """Define the key value items that characterize each dimension"""
    
    # Traditional vs Secular dimension items (5 items each)
    traditional_items = {
        'high_god_importance': {
            'description': 'High religious importance',
            'filter': lambda df: df['Q164_aid'] >= 8,
            'label': 'Religious'
        },
        'anti_autonomy_children': {
            'description': 'Values obedience/faith over independence/determination in children',
            'filter': lambda df: ((df['Q17_aid'] == 1) | (df['Q15_aid'] == 1)) & ((df['Q8_aid'] == 2) | (df['Q14_aid'] == 2)),
            'label': 'Anti-autonomy'
        },
        'low_abortion_tolerance': {
            'description': 'Opposes abortion',
            'filter': lambda df: df['Q184_aid'] <= 2,
            'label': 'Pro-life'
        },
        'high_national_pride': {
            'description': 'Very proud of country',
            'filter': lambda df: df['Q254_aid'] == 1,
            'label': 'Patriotic'
        },
        'respect_authority': {
            'description': 'Strongly respects authority',
            'filter': lambda df: df['Q45_aid'] <= 2,
            'label': 'Authority-respecting'
        }
    }
    
    # Secular items (opposite of traditional, 5 items each)
    secular_items = {
        'low_god_importance': {
            'description': 'Low religious importance',
            'filter': lambda df: df['Q164_aid'] <= 3,
            'label': 'Secular'
        },
        'pro_autonomy_children': {
            'description': 'Values independence/determination over obedience/faith in children',
            'filter': lambda df: ((df['Q8_aid'] == 1) | (df['Q14_aid'] == 1)) & ((df['Q17_aid'] == 2) | (df['Q15_aid'] == 2)),
            'label': 'Pro-autonomy'
        },
        'high_abortion_tolerance': {
            'description': 'Supports abortion rights',
            'filter': lambda df: df['Q184_aid'] >= 7,
            'label': 'Pro-choice'
        },
        'low_national_pride': {
            'description': 'Not very proud of country',
            'filter': lambda df: df['Q254_aid'] >= 3,
            'label': 'Cosmopolitan'
        },
        'question_authority': {
            'description': 'Questions authority',
            'filter': lambda df: df['Q45_aid'] >= 4,
            'label': 'Authority-questioning'
        }
    }
    
    # Survival vs Self-Expression dimension items
    survival_items = {
        'materialist_priorities': {
            'description': 'Prioritizes economic growth and defense',
            'filter': lambda df: (df['Q152_aid'].isin([1, 2])) | (df['Q153_aid'].isin([1, 2])),
            'label': 'Materialist'
        },
        'low_happiness': {
            'description': 'Reports lower happiness',
            'filter': lambda df: df['Q46_aid'] >= 3,
            'label': 'Pessimistic'
        },
        'low_homosexuality_tolerance': {
            'description': 'Opposes homosexuality',
            'filter': lambda df: df['Q182_aid'] <= 2,
            'label': 'Socially conservative'
        },
        'low_political_participation': {
            'description': 'Avoids political participation',
            'filter': lambda df: df['Q209_aid'] == 3,
            'label': 'Politically disengaged'
        },
        'low_interpersonal_trust': {
            'description': 'Distrusts others',
            'filter': lambda df: df['Q57_aid'] == 1,
            'label': 'Suspicious'
        }
    }
    
    # Self-Expression items (opposite of survival)
    self_expression_items = {
        'postmaterialist_priorities': {
            'description': 'Prioritizes participation and beauty',
            'filter': lambda df: (df['Q152_aid'].isin([3, 4])) | (df['Q153_aid'].isin([3, 4])),
            'label': 'Post-materialist'
        },
        'high_happiness': {
            'description': 'Reports higher happiness',
            'filter': lambda df: df['Q46_aid'] <= 2,
            'label': 'Optimistic'
        },
        'high_homosexuality_tolerance': {
            'description': 'Supports homosexuality rights',
            'filter': lambda df: df['Q182_aid'] >= 7,
            'label': 'Socially liberal'
        },
        'high_political_participation': {
            'description': 'Engages in political participation',
            'filter': lambda df: df['Q209_aid'] <= 2,
            'label': 'Politically engaged'
        },
        'high_interpersonal_trust': {
            'description': 'Trusts others',
            'filter': lambda df: df['Q57_aid'] == 2,
            'label': 'Trusting'
        }
    }
    
    return traditional_items, secular_items, survival_items, self_expression_items

def load_wvq_questions():
    """Load question IDs from WVQ.csv file"""
    # Manual list of WVQ question IDs to avoid CSV parsing issues
    wvq_q_ids = [
        # SOCIAL VALUES
        27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41,
        # MIGRATION  
        122, 123, 124, 125, 126, 127, 128, 129,
        # SECURITY
        132, 133, 134, 135, 136, 137, 138,
        # SCIENCE & TECHNOLOGY
        158, 159, 160, 161, 162,
        # RELIGIOUS VALUES
        169, 170,
        # ETHICAL VALUES AND NORMS
        196, 197, 198,
        # POLITICAL INTEREST & POLITICAL PARTICIPATION
        224, 225, 226, 227, 228, 229, 230, 231, 232, 233
    ]
    
    wvq_questions = [f'Q{q_id}_aid' for q_id in wvq_q_ids]
    return wvq_questions


def filter_valid_responses(candidates):
    """Filter out users with invalid (negative, zero, or missing) responses to key questions"""
    
    # Define key WVS classification questions that must have valid responses
    key_wvs_questions = [
        'Q164_aid',  # God importance
        'Q17_aid',   # Child qualities - Obedience
        'Q15_aid',   # Child qualities - Religious faith
        'Q8_aid',    # Child qualities - Independence
        'Q14_aid',   # Child qualities - Determination
        'Q184_aid',  # Abortion justification
        'Q254_aid',  # National pride
        'Q45_aid',   # Respect for authority
        'Q152_aid',  # Materialism priority 1
        'Q153_aid',  # Materialism priority 2
        'Q46_aid',   # Happiness level
        'Q182_aid',  # Homosexuality justification
        'Q209_aid',  # Petition participation (or Q218_aid)
        'Q57_aid'    # Interpersonal trust
    ]
    
    # Load WVQ evaluation questions
    wvq_questions = load_wvq_questions()
    
    # Combine all questions that need validation
    all_required_questions = key_wvs_questions + wvq_questions
    
    # Remove duplicates while preserving order
    seen = set()
    unique_questions = []
    for q in all_required_questions:
        if q not in seen:
            unique_questions.append(q)
            seen.add(q)
    
    # Check which questions are actually present in the dataset
    available_questions = [q for q in unique_questions if q in candidates.columns]
    
    if not available_questions:
        print("    Warning: No required questions found with '_aid' suffix")
        return candidates
    
    # Count how many of each type are available
    available_wvs = len([q for q in key_wvs_questions if q in candidates.columns])
    available_wvq = len([q for q in wvq_questions if q in candidates.columns])
    print(f"    Validating {len(available_questions)} questions ({available_wvs}/{len(key_wvs_questions)} WVS + {available_wvq}/{len(wvq_questions)} WVQ)")
    
    # Filter for valid responses: must be positive and not missing
    valid_mask = True
    for question in available_questions:
        # Valid responses are > 0 (positive) and not NaN
        question_valid = (candidates[question] > 0) & (candidates[question].notna())
        valid_mask = valid_mask & question_valid
    
    valid_candidates = candidates[valid_mask]
    
    return valid_candidates


def sample_distinct_profiles(df, quadrant_groups, diversity_scores, n_profiles_per_quadrant=None):
    """Sample distinct profiles where each represents a different value item, prioritizing diversity"""
    
    traditional_items, secular_items, survival_items, self_expression_items = define_value_items()
    
    # Determine minimum number of profiles needed (5x5 = 25 combinations, but we aim for 5 per quadrant)
    if n_profiles_per_quadrant is None:
        n_profiles_per_quadrant = 5  # Now that both dimensions have 5 items each
    
    print(f"Sampling {n_profiles_per_quadrant} profiles per quadrant...")
    print(f"Total profiles: {n_profiles_per_quadrant * 4}")
    print(f"Prioritizing users with highest diversity scores...")
    print(f"Ensuring each user can only be selected once...")
    
    distinct_profiles = {}
    
    # Keep track of already selected users to ensure no duplicates
    selected_user_ids = set()
    
    # Sample for each quadrant
    quadrant_configs = {
        'Traditional-Survival': (traditional_items, survival_items),
        'Traditional-Self-Expression': (traditional_items, self_expression_items),
        'Secular-Survival': (secular_items, survival_items),
        'Secular-Self-Expression': (secular_items, self_expression_items)
    }
    
    for quadrant, (dim1_items, dim2_items) in quadrant_configs.items():
        print(f"\nSampling {quadrant}...")
        
        # Get users in this quadrant
        quadrant_users = quadrant_groups[quadrant]
        quadrant_df = df[df['D_INTERVIEW'].isin(quadrant_users)].copy()
        
        profiles = []
        used_combinations = set()
        
        # Create combinations of dimension 1 and dimension 2 items
        dim1_keys = list(dim1_items.keys())
        dim2_keys = list(dim2_items.keys())
        
        for i in range(n_profiles_per_quadrant):
            dim1_key = dim1_keys[i % len(dim1_keys)]
            dim2_key = dim2_keys[i % len(dim2_keys)]
            
            # Avoid duplicate combinations
            combination = (dim1_key, dim2_key)
            if combination in used_combinations:
                # Find alternative
                for alt_dim1 in dim1_keys:
                    for alt_dim2 in dim2_keys:
                        alt_combo = (alt_dim1, alt_dim2)
                        if alt_combo not in used_combinations:
                            dim1_key, dim2_key = alt_dim1, alt_dim2
                            combination = alt_combo
                            break
                    if combination not in used_combinations:
                        break
            
            used_combinations.add(combination)
            
            # Filter users who satisfy both criteria
            dim1_filter = dim1_items[dim1_key]['filter'](quadrant_df)
            dim2_filter = dim2_items[dim2_key]['filter'](quadrant_df)
            
            # Get candidates that satisfy both criteria
            potential_candidates = quadrant_df[dim1_filter & dim2_filter]
            
            # Track filtering steps for reporting
            after_value_criteria = len(potential_candidates)
            
            # Filter out already selected users
            candidates = potential_candidates[~potential_candidates['D_INTERVIEW'].isin(selected_user_ids)]
            after_user_filter = len(candidates)
            
            # Filter out users with invalid responses to key questions
            if len(candidates) > 0:
                candidates = filter_valid_responses(candidates)
            after_valid_filter = len(candidates)
            
            if len(candidates) > 0:
                # Rank candidates by diversity score and select the most diverse
                candidate_diversity_scores = []
                for _, candidate in candidates.iterrows():
                    user_id = candidate['D_INTERVIEW']
                    diversity_score = diversity_scores.get(user_id, 0)
                    candidate_diversity_scores.append((diversity_score, candidate))
                
                # Sort by diversity score (descending) and select the most diverse
                candidate_diversity_scores.sort(key=lambda x: x[0], reverse=True)
                selected_user = candidate_diversity_scores[0][1]
                selected_diversity = candidate_diversity_scores[0][0]
                selected_user_id = selected_user['D_INTERVIEW']
                
                # Add this user to the selected set to prevent future selection
                selected_user_ids.add(selected_user_id)
                
                profile = {
                    'user_id': int(selected_user['D_INTERVIEW']),
                    'quadrant': quadrant,
                    'diversity_score': selected_diversity,
                    'primary_traditional_secular_item': dim1_key,
                    'primary_traditional_secular_label': dim1_items[dim1_key]['label'],
                    'primary_traditional_secular_description': dim1_items[dim1_key]['description'],
                    'primary_survival_expression_item': dim2_key,
                    'primary_survival_expression_label': dim2_items[dim2_key]['label'],
                    'primary_survival_expression_description': dim2_items[dim2_key]['description'],
                    'profile_label': f"{dim1_items[dim1_key]['label']} + {dim2_items[dim2_key]['label']}",
                    'demographics': {
                        'gender': selected_user.get('Q260', 'N/A'),
                        'age': selected_user.get('X003R', 'N/A'),
                        'country': selected_user.get('B_COUNTRY', 'N/A'),
                        'education': selected_user.get('Q275', 'N/A')
                    },
                    'key_responses': {
                        'Q164_aid': int(selected_user['Q164_aid']),  # God importance
                        'Q17_aid': int(selected_user['Q17_aid']),   # Obedience
                        'Q15_aid': int(selected_user['Q15_aid']),   # Religious faith
                        'Q8_aid': int(selected_user['Q8_aid']),     # Independence
                        'Q14_aid': int(selected_user['Q14_aid']),   # Determination
                        'Q184_aid': int(selected_user['Q184_aid']), # Abortion
                        'Q254_aid': int(selected_user['Q254_aid']), # National pride
                        'Q45_aid': int(selected_user['Q45_aid']),   # Authority
                        'Q152_aid': int(selected_user['Q152_aid']), # Materialism 1
                        'Q153_aid': int(selected_user['Q153_aid']), # Materialism 2
                        'Q46_aid': int(selected_user['Q46_aid']),   # Happiness
                        'Q182_aid': int(selected_user['Q182_aid']), # Homosexuality
                        'Q209_aid': int(selected_user['Q209_aid']), # Political participation
                        'Q57_aid': int(selected_user['Q57_aid'])    # Trust
                    }
                }
                
                profiles.append(profile)
                print(f"  Profile {i+1}: {profile['profile_label']} (User {profile['user_id']}, diversity: {selected_diversity:.4f})")
                print(f"    Candidates: {after_value_criteria} total → {after_user_filter} available → {after_valid_filter} with valid responses")
            else:
                if after_value_criteria == 0:
                    print(f"  Profile {i+1}: No candidates found for {dim1_key} + {dim2_key}")
                elif after_user_filter == 0:
                    print(f"  Profile {i+1}: No available candidates for {dim1_key} + {dim2_key} ({after_value_criteria} already selected)")
                elif after_valid_filter == 0:
                    print(f"  Profile {i+1}: No candidates with valid responses for {dim1_key} + {dim2_key} ({after_user_filter} available but invalid)")
                else:
                    print(f"  Profile {i+1}: Unexpected error for {dim1_key} + {dim2_key}")
        
        distinct_profiles[quadrant] = profiles
    
    return distinct_profiles

def save_distinct_profiles(distinct_profiles):
    """Save the distinct profiles to JSON file"""
    
    # Calculate diversity statistics for selected profiles
    all_selected_profiles = []
    for profiles in distinct_profiles.values():
        all_selected_profiles.extend(profiles)
    
    if all_selected_profiles:
        diversity_scores = [p['diversity_score'] for p in all_selected_profiles]
        diversity_stats = {
            'mean_diversity': np.mean(diversity_scores),
            'median_diversity': np.median(diversity_scores),
            'max_diversity': np.max(diversity_scores),
            'min_diversity': np.min(diversity_scores)
        }
    else:
        diversity_stats = {}
    
    # Create summary statistics
    summary = {
        'total_profiles': sum(len(profiles) for profiles in distinct_profiles.values()),
        'profiles_per_quadrant': {quadrant: len(profiles) for quadrant, profiles in distinct_profiles.items()},
        'diversity_statistics': diversity_stats,
        'unique_combinations': []
    }
    
    # Collect all unique combinations
    for quadrant, profiles in distinct_profiles.items():
        for profile in profiles:
            combination = {
                'quadrant': quadrant,
                'combination': profile['profile_label'],
                'traditional_secular': profile['primary_traditional_secular_label'],
                'survival_expression': profile['primary_survival_expression_label']
            }
            summary['unique_combinations'].append(combination)
    
    # Create output structure
    output = {
        'summary': summary,
        'distinct_profiles': distinct_profiles
    }
    
    # Save to file
    with open('distinct_value_profiles.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to: distinct_value_profiles.json")
    return output

def main():
    print("=== DISTINCT VALUE PROFILER ===")
    print("Creating minimum distinct profiles representing different value items per quadrant")
    print("Prioritizing users with highest response diversity scores")
    
    # Load data
    print("\nLoading data...")
    df, quadrant_groups = load_wvs_data()
    
    # Calculate diversity scores for all users
    print("Calculating user response diversity...")
    diversity_scores = calculate_user_diversity(df)
    print(f"Calculated diversity scores for {len(diversity_scores)} users")
    
    # Print diversity statistics
    diversity_values = list(diversity_scores.values())
    print(f"Diversity score statistics:")
    print(f"  Mean: {np.mean(diversity_values):.4f}")
    print(f"  Median: {np.median(diversity_values):.4f}")
    print(f"  Max: {np.max(diversity_values):.4f}")
    print(f"  Min: {np.min(diversity_values):.4f}")
    
    # Define value items
    traditional_items, secular_items, survival_items, self_expression_items = define_value_items()
    
    print(f"\nValue items identified:")
    print(f"  Traditional items: {len(traditional_items)}")
    print(f"  Secular items: {len(secular_items)}")
    print(f"  Survival items: {len(survival_items)}")
    print(f"  Self-expression items: {len(self_expression_items)}")
    
    # Sample distinct profiles using diversity scores
    distinct_profiles = sample_distinct_profiles(df, quadrant_groups, diversity_scores)
    
    # Save results
    output = save_distinct_profiles(distinct_profiles)
    
    # Print summary
    print(f"\n=== SUMMARY ===")
    print(f"Total distinct profiles: {output['summary']['total_profiles']}")
    print(f"Profiles per quadrant: {max(output['summary']['profiles_per_quadrant'].values())}")
    
    # Check for unique users
    all_user_ids = []
    for profiles in output['distinct_profiles'].values():
        for profile in profiles:
            all_user_ids.append(profile['user_id'])
    
    print(f"Unique users selected: {len(set(all_user_ids))} (should equal total profiles: {len(all_user_ids)})")
    if len(set(all_user_ids)) == len(all_user_ids):
        print("✅ SUCCESS: Each profile has a unique user!")
    else:
        print("WARNING: Some users were selected multiple times")
    
    # Print diversity statistics for selected profiles
    if 'diversity_statistics' in output['summary'] and output['summary']['diversity_statistics']:
        diversity_stats = output['summary']['diversity_statistics']
        print(f"\nSelected profiles diversity statistics:")
        print(f"  Mean diversity: {diversity_stats['mean_diversity']:.4f}")
        print(f"  Median diversity: {diversity_stats['median_diversity']:.4f}")
        print(f"  Max diversity: {diversity_stats['max_diversity']:.4f}")
        print(f"  Min diversity: {diversity_stats['min_diversity']:.4f}")
    
    print(f"\nDistinct combinations created:")
    for combo in output['summary']['unique_combinations'][:10]:  # Show first 10
        print(f"  {combo['quadrant']}: {combo['combination']}")
    if len(output['summary']['unique_combinations']) > 10:
        print(f"  ... and {len(output['summary']['unique_combinations']) - 10} more")

if __name__ == "__main__":
    main() 