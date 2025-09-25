import json
import pandas as pd
import numpy as np
import random
from collections import defaultdict
import os
from tqdm import tqdm
from typing import List, Dict, Any, Tuple, Optional, Literal
import time
import traceback
import sys
from pathlib import Path

# Add parent directories to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))
sys.path.append(str(Path(__file__).parent.parent))

# Import data manager
from data_management.data_manager_v3 import DataManagerV3

try:
    from vllm import LLM, SamplingParams
except ImportError:
    print("Warning: vLLM not installed. Please install with: pip install vllm")
    LLM = None
    SamplingParams = None

try:
    from transformers import AutoTokenizer
except ImportError:
    print("Warning: transformers not installed. Please install with: pip install transformers")
    AutoTokenizer = None

# Pydantic for structured JSON output
try:
    from pydantic import BaseModel, Field, ValidationError
except ImportError:
    print("Warning: Pydantic not installed. Please install with: pip install pydantic")
    BaseModel = None
    Field = None
    ValidationError = None

# COT Response schema
if BaseModel is not None:
    class COTResponse(BaseModel):
        """Structured response schema for Chain-of-Thought evaluations"""
        reasoning: str = Field(description="Step-by-step analysis of user preferences and response alignment")
        final_choice: Literal["A", "B"] = Field(description="The final answer: either A or B")
        confidence: Optional[float] = Field(default=None, ge=0, le=1, description="Optional confidence score between 0 and 1")


class EvaluationEngineV3:
    """
    Enhanced evaluation engine for v3 framework with systematic user profile combinations.
    Supports setting-specific user profile filtering to avoid redundant evaluations.
    """
    
    def __init__(self, data_manager: DataManagerV3):
        self.data_manager = data_manager
        
        # Load all data
        self.user_profiles = data_manager.load_user_profiles_v3()
        self.distinct_value_profiles = data_manager.load_distinct_value_profiles()
        self.synthetic_data = data_manager.filter_synthetic_data_by_questions()
        
        print(f"Evaluation engine initialized with {len(self.user_profiles['user_profiles'])} user profiles")
        print(f"{len(self.synthetic_data)} questions available for evaluation")
    
    def get_user_profiles_for_setting(self, setting_name: str) -> List[Dict[str, Any]]:
        """
        Get the appropriate user profiles for a specific evaluation setting.
        This is the key method that ensures proper filtering to avoid redundant evaluations.
        
        Args:
            setting_name: Name of the evaluation setting
            
        Returns:
            List of user profiles to use for this setting
        """
        all_profiles = self.user_profiles['user_profiles']
        
        if setting_name in ['simple_only', 'simple_prompting']:
            # For simple prompting, we don't use user context at all
            # We only need one representative profile per unique combination since context is ignored
            # Use the first profile as a representative
            return [all_profiles[0]]
            
        elif setting_name in ['full_wvs_only', 'full_wvs_context_only']:
            # For WVS-only evaluation, we need one profile per unique value_profile_id
            # Style profiles become irrelevant
            seen_value_profiles = set()
            filtered_profiles = []
            
            for profile in all_profiles:
                value_profile_id = profile['value_profile_id']
                if value_profile_id not in seen_value_profiles:
                    seen_value_profiles.add(value_profile_id)
                    filtered_profiles.append(profile)
            
            return filtered_profiles
            
        elif setting_name in ['full_style_only', 'full_style_context_only']:
            # For style-only evaluation, we need one profile per unique style_profile
            # Value profiles become irrelevant
            seen_style_profiles = set()
            filtered_profiles = []
            
            for profile in all_profiles:
                # Convert style_profile dict to a hashable tuple for comparison
                style_profile = profile['style_profile']
                style_key = tuple(sorted(style_profile.items())) if style_profile else ()
                
                if style_key not in seen_style_profiles:
                    seen_style_profiles.add(style_key)
                    filtered_profiles.append(profile)
            
            return filtered_profiles
            
        elif setting_name in ['full_combined', 'full_combined_context', 'cot_only']:
            # For combined evaluation or COT, we need all user profiles
            # This covers all 18 value × 16 style combinations
            return all_profiles
            
        elif setting_name == 'all_settings':
            # For general all_settings, return all profiles (backward compatibility)
            print(f"Warning: Using all profiles for all_settings setting")
            exit()
            return None
            
        # ===== ABLATION-SPECIFIC SETTINGS =====
        elif setting_name in ['ablation_wvs_only', 'ablation_wvs_only_cot']:
            # For WVS-only ablation, we need one profile per unique value_profile_id
            # Style profiles become irrelevant since we're only testing WVS context
            seen_value_profiles = set()
            filtered_profiles = []
            
            for profile in all_profiles:
                value_profile_id = profile['value_profile_id']
                if value_profile_id not in seen_value_profiles:
                    seen_value_profiles.add(value_profile_id)
                    filtered_profiles.append(profile)
            
            return filtered_profiles
            
        elif setting_name in ['ablation_wvs_style_neutral', 'ablation_wvs_style_prefer_wvs', 
                              'ablation_wvs_style_prefer_style', 'ablation_wvs_style_neutral_cot',
                              'ablation_wvs_style_prefer_wvs_cot', 'ablation_wvs_style_prefer_style_cot']:
            # For combined WVS + style ablation, we need all user profiles
            # This covers all 18 value × 16 style combinations since both contexts matter
            return all_profiles
            
        else:
            # Default: return all profiles for unknown settings
            print(f"Warning: Unknown setting '{setting_name}', using all profiles")
            return all_profiles
    
    def get_evaluation_counts_for_setting(self, setting_name: str) -> Dict[str, int]:
        """
        Calculate evaluation counts for a specific setting using proper filtering.
        
        Args:
            setting_name: Name of the evaluation setting
            
        Returns:
            Dictionary with detailed evaluation count breakdown
        """
        # Get filtered user profiles for this setting
        profiles_for_setting = self.get_user_profiles_for_setting(setting_name)
        
        # Calculate base statistics
        n_questions = len(self.synthetic_data)
        n_style_families = 4  # verbosity, readability, confidence, sentiment
        n_combinations_per_family = 6  # 4 WVS-based + 2 style-based
        n_profiles = len(profiles_for_setting)
        
        base_combinations_per_profile = n_questions * n_style_families * n_combinations_per_family
        total_evaluations = base_combinations_per_profile * n_profiles
        
        # Additional counts for combined settings
        counts = {
            'questions': n_questions,
            'style_families': n_style_families,
            'combinations_per_family': n_combinations_per_family,
            'user_profiles_used': n_profiles,
            'base_combinations_per_profile': base_combinations_per_profile,
            'total_evaluations': total_evaluations
        }
        
        # Add unique profile counts
        if n_profiles > 0:
            unique_value_profiles = len(set(p['value_profile_id'] for p in profiles_for_setting))
            unique_style_codes = len(set(p.get('style_code', 'unknown') for p in profiles_for_setting))
            
            counts.update({
                'unique_value_profiles': unique_value_profiles,
                'unique_style_profiles': unique_style_codes,
                'setting_name': setting_name
            })
        
        return counts
    
    def generate_preference_pairs_for_question(self, question_data: Dict[str, Any], 
                                              user_profile: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate preference pairs for a specific question and user profile.
        Following the v2 logic but adapted for v3 data structure.
        
        Args:
            question_data: Synthetic question data with style variations
            user_profile: User profile with value and style preferences
            
        Returns:
            List of preference pair dictionaries
        """
        preference_pairs = []
        
        # Style families in v3 (4 families)
        style_families = ['verbosity', 'readability', 'confidence', 'sentiment']
        
        for style_family in style_families:
            # Generate 6 preference combinations per family per question
            family_pairs = self._generate_family_preference_pairs(
                question_data, user_profile, style_family
            )
            preference_pairs.extend(family_pairs)
        
        return preference_pairs
    
    def _generate_family_preference_pairs(self, question_data: Dict[str, Any], 
                                        user_profile: Dict[str, Any], 
                                        style_family: str) -> List[Dict[str, Any]]:
        """
        Generate 6 preference pairs for a specific style family.
        
        4 A vs B comparisons (WVS-based) + 2 within-family comparisons (style-based)
        """
        pairs = []
        user_style_pref = user_profile['style_profile'][style_family]
        
        # Style variations mapping
        style_variations = {
            'verbosity': ['verbose', 'concise'],
            'readability': ['high_reading_difficulty', 'low_reading_difficulty'],
            'confidence': ['high_confidence', 'low_confidence'],
            'sentiment': ['warm', 'cold']
        }
        
        style1, style2 = style_variations[style_family]
        
        # Get completions with style variations
        try:
            completion_A_style1 = question_data[f'completion_A_{style1}']
            completion_A_style2 = question_data[f'completion_A_{style2}']
            completion_B_style1 = question_data[f'completion_B_{style1}']
            completion_B_style2 = question_data[f'completion_B_{style2}']
        except KeyError as e:
            print(f"Warning: Missing style variation {e} for question {question_data['question_id']}")
            return []
        
        # A vs B Comparisons (4 pairs) - WVS-based preferences
        wvs_pairs = [
            (completion_A_style1, completion_B_style1, f'A_{style1}_vs_B_{style1}'),
            (completion_A_style2, completion_B_style1, f'A_{style2}_vs_B_{style1}'),
            (completion_A_style1, completion_B_style2, f'A_{style1}_vs_B_{style2}'),
            (completion_A_style2, completion_B_style2, f'A_{style2}_vs_B_{style2}'),
        ]
        
        for comp_a, comp_b, combination_type in wvs_pairs:
            # WVS response determines preference (from question metadata)
            wvs_response = question_data.get('wvs_response', 1)  # Default to 1 if missing
            
            if wvs_response == 1:
                preferred_completion = comp_a
                non_preferred_completion = comp_b
                preferred_key = combination_type.split('_vs_')[0]
                non_preferred_key = combination_type.split('_vs_')[1]
            elif wvs_response == 2:
                preferred_completion = comp_b
                non_preferred_completion = comp_a
                preferred_key = combination_type.split('_vs_')[1]
                non_preferred_key = combination_type.split('_vs_')[0]
            else:
                continue  # Skip invalid responses
            
            pairs.append({
                'user_profile_id': user_profile['user_profile_id'],
                'value_profile_id': user_profile['value_profile_id'],
                'question_id': question_data['question_id'],
                'style_family': style_family,
                'combination_type': combination_type,
                'preference_rule': 'wvs_based',
                'style_profile': user_profile['style_profile'],
                'prompt': question_data['prompt'],
                'preferred_completion': preferred_completion,
                'non_preferred_completion': non_preferred_completion,
                'preferred_completion_key': preferred_key,
                'non_preferred_completion_key': non_preferred_key,
                'wvs_response': wvs_response,
                'wvs_question': question_data.get('wvs_question', ''),
                'statement_1': question_data.get('statement_1', ''),
                'statement_2': question_data.get('statement_2', ''),
                'quadrant': user_profile['quadrant'],
                'style_code': user_profile['style_code']
            })
        
        # Within-Family Comparisons (2 pairs) - Style-based preferences
        style_pairs = [
            (completion_A_style1, completion_A_style2, f'A_{style1}_vs_A_{style2}'),
            (completion_B_style1, completion_B_style2, f'B_{style1}_vs_B_{style2}'),
        ]
        
        for comp_a, comp_b, combination_type in style_pairs:
            # User's style preference determines the winner
            if user_style_pref == style1:
                preferred_completion = comp_a
                non_preferred_completion = comp_b
                preferred_key = combination_type.split('_vs_')[0]
                non_preferred_key = combination_type.split('_vs_')[1]
            else:  # user_style_pref == style2
                preferred_completion = comp_b
                non_preferred_completion = comp_a
                preferred_key = combination_type.split('_vs_')[1]
                non_preferred_key = combination_type.split('_vs_')[0]
            
            pairs.append({
                'user_profile_id': user_profile['user_profile_id'],
                'value_profile_id': user_profile['value_profile_id'],
                'question_id': question_data['question_id'],
                'style_family': style_family,
                'combination_type': combination_type,
                'preference_rule': 'style_based',
                'style_profile': user_profile['style_profile'],
                'prompt': question_data['prompt'],
                'preferred_completion': preferred_completion,
                'non_preferred_completion': non_preferred_completion,
                'preferred_completion_key': preferred_key,
                'non_preferred_completion_key': non_preferred_key,
                'wvs_response': question_data.get('wvs_response', 1),
                'wvs_question': question_data.get('wvs_question', ''),
                'statement_1': question_data.get('statement_1', ''),
                'statement_2': question_data.get('statement_2', ''),
                'quadrant': user_profile['quadrant'],
                'style_code': user_profile['style_code']
            })
        
        return pairs
    
    def generate_all_preference_pairs(self, question_ids: List[str] = None, 
                                    user_profile_ids: List[str] = None,
                                    setting_name: str = 'all_settings') -> List[Dict[str, Any]]:
        """
        Generate preference pairs for the specified questions and user profiles.
        Uses setting-specific filtering to avoid redundant evaluations.
        
        For ablation studies, this method intelligently handles different ablation settings:
        - ablation_wvs_only, ablation_wvs_only_cot: Returns preference pairs for 18 value profiles only
        - ablation_wvs_style_*: Returns preference pairs for all 288 user profiles
        
        Args:
            question_ids: List of question IDs to include (None = all filtered questions)
            user_profile_ids: List of user profile IDs to include (None = use setting-based filtering)
            setting_name: Evaluation setting name for filtering user profiles
            
        Returns:
            List of all generated preference pairs
        """
        # Filter questions
        if question_ids is None:
            questions_to_use = self.synthetic_data
        else:
            question_map = {q['question_id']: q for q in self.synthetic_data}
            questions_to_use = [question_map[q_id] for q_id in question_ids if q_id in question_map]
        
        # Filter user profiles based on setting or explicit list
        if user_profile_ids is not None:
            # Use explicit list if provided
            profile_map = {p['user_profile_id']: p for p in self.user_profiles['user_profiles']}
            profiles_to_use = [profile_map[p_id] for p_id in user_profile_ids if p_id in profile_map]
            print(f"Using {len(profiles_to_use)} explicitly specified user profiles")
        else:
            # Use setting-based filtering
            profiles_to_use = self.get_user_profiles_for_setting(setting_name)
            print(f"Using {len(profiles_to_use)} user profiles for setting '{setting_name}'")
        
        all_pairs = []
        
        print(f"Generating preference pairs for {len(questions_to_use)} questions × {len(profiles_to_use)} profiles...")
        
        for question_data in tqdm(questions_to_use, desc="Processing questions"):
            for user_profile in profiles_to_use:
                pairs = self.generate_preference_pairs_for_question(question_data, user_profile)
                all_pairs.extend(pairs)
        
        print(f"Generated {len(all_pairs)} total preference pairs")
        
        # Verify the counts match expectations
        expected_counts = self.get_evaluation_counts_for_setting(setting_name)
        if len(all_pairs) != expected_counts['total_evaluations']:
            print(f"Warning: Generated {len(all_pairs)} pairs, expected {expected_counts['total_evaluations']}")
        else:
            print(f"Preference pair count matches expected: {len(all_pairs)}")
            
        return all_pairs
    
    def generate_ablation_preference_pairs(self, question_ids: List[str] = None,
                                         user_profile_ids: List[str] = None,
                                         ablation_mode: str = 'all_settings') -> Dict[str, List[Dict[str, Any]]]:
        """
        Generate preference pairs specifically for ablation studies.
        
        This method intelligently splits preference pairs by ablation type:
        - WVS-only settings: Use 18 value profiles (style doesn't matter)
        - Combined settings: Use all 288 profiles (both value and style matter)
        
        Args:
            question_ids: List of question IDs to include (None = all filtered questions)
            user_profile_ids: List of user profile IDs to include (None = use setting-based filtering)
            ablation_mode: 'all_settings' includes COT, 'all_settings_no_cot' excludes COT
            
        Returns:
            Dictionary mapping setting categories to their preference pairs:
            {
                'wvs_only': [...],      # For ablation_wvs_only settings (18 profiles)
                'combined': [...]       # For ablation_wvs_style_* settings (288 profiles)
            }
        """
        # Define setting categories
        wvs_only_settings = ['ablation_wvs_only']
        combined_settings = ['ablation_wvs_style_neutral', 'ablation_wvs_style_prefer_wvs', 
                            'ablation_wvs_style_prefer_style']
        
        if ablation_mode == 'all_settings':
            # Add COT versions
            wvs_only_settings.append('ablation_wvs_only_cot')
            combined_settings.extend(['ablation_wvs_style_neutral_cot', 'ablation_wvs_style_prefer_wvs_cot',
                                     'ablation_wvs_style_prefer_style_cot'])
        
        result = {}
        
        # Generate preference pairs for WVS-only settings (18 value profiles)
        print(f"Generating preference pairs for WVS-only ablation settings...")
        wvs_only_pairs = self.generate_all_preference_pairs(
            question_ids=question_ids,
            user_profile_ids=user_profile_ids,
            setting_name='ablation_wvs_only'  # This will filter to 18 value profiles
        )
        result['wvs_only'] = wvs_only_pairs
        print(f"Generated {len(wvs_only_pairs)} preference pairs for WVS-only settings")
        
        # Generate preference pairs for combined settings (288 profiles)
        print(f"Generating preference pairs for combined ablation settings...")
        combined_pairs = self.generate_all_preference_pairs(
            question_ids=question_ids,
            user_profile_ids=user_profile_ids,
            setting_name='ablation_wvs_style_neutral'  # This will use all 288 profiles
        )
        result['combined'] = combined_pairs
        print(f"Generated {len(combined_pairs)} preference pairs for combined settings")
        
        # Print summary
        total_unique_pairs = len(wvs_only_pairs) + len(combined_pairs)
        print(f"Total unique preference pairs generated: {total_unique_pairs}")
        print(f"  - WVS-only settings (18 value profiles): {len(wvs_only_pairs)}")
        print(f"  - Combined settings (288 profiles): {len(combined_pairs)}")
        
        return result
    
    def create_unique_evaluation_id(self, preference_pair: Dict[str, Any], setting_name: str) -> str:
        """
        Create a unique evaluation ID for deduplication.
        
        Args:
            preference_pair: Preference pair dictionary
            setting_name: Evaluation setting name
            
        Returns:
            Unique evaluation ID string
        """
        components = [
            preference_pair['user_profile_id'],
            preference_pair['question_id'],
            preference_pair['style_family'],
            preference_pair['combination_type'],
            setting_name
        ]
        return "_".join(components)
    
    def get_evaluation_count_by_setting(self, preference_pairs: List[Dict[str, Any]] = None, 
                                       setting_name: str = None) -> Dict[str, int]:
        """
        Calculate evaluation counts for different settings using proper filtering.
        
        Args:
            preference_pairs: List of preference pairs (deprecated, use setting_name instead)
            setting_name: Setting name to calculate counts for
            
        Returns:
            Dictionary with evaluation counts by setting
        """
        if setting_name:
            # New approach: use setting-specific filtering
            return self.get_evaluation_counts_for_setting(setting_name)
        
        # Legacy approach for backward compatibility
        if preference_pairs is None:
            print("Warning: No preference_pairs or setting_name provided")
            return {}
            
        base_count = len(preference_pairs)
        
        # Count unique value profiles and style profiles in the data
        unique_value_profiles = len(set(p['value_profile_id'] for p in preference_pairs))
        unique_style_codes = len(set(p['style_code'] for p in preference_pairs))
        
        counts = {
            'simple_prompting': base_count,
            'full_wvs_only': base_count * unique_value_profiles if unique_value_profiles > 0 else base_count,
            'full_style_only': base_count * unique_style_codes if unique_style_codes > 0 else base_count,
            'full_combined_per_priority': base_count * unique_value_profiles * unique_style_codes if unique_value_profiles > 0 and unique_style_codes > 0 else base_count,
            'full_combined_all_priorities': base_count * unique_value_profiles * unique_style_codes * 3 if unique_value_profiles > 0 and unique_style_codes > 0 else base_count * 3
        }
        
        # Add COT versions (double each)
        for setting in list(counts.keys()):
            counts[f'{setting}_cot'] = counts[setting]
        
        return counts
    
    def save_preference_pairs(self, preference_pairs: List[Dict[str, Any]], output_file: str):
        """
        Save generated preference pairs to JSON file.
        
        Args:
            preference_pairs: List of preference pairs
            output_file: Output file path
        """
        # Create metadata
        metadata = {
            'total_preference_pairs': len(preference_pairs),
            'unique_questions': len(set(p['question_id'] for p in preference_pairs)),
            'unique_user_profiles': len(set(p['user_profile_id'] for p in preference_pairs)),
            'unique_value_profiles': len(set(p['value_profile_id'] for p in preference_pairs)),
            'unique_style_codes': len(set(p['style_code'] for p in preference_pairs)),
            'style_families': list(set(p['style_family'] for p in preference_pairs)),
            'quadrants': list(set(p['quadrant'] for p in preference_pairs)),
            'preference_rules': list(set(p['preference_rule'] for p in preference_pairs)),
            'generation_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'evaluation_counts': self.get_evaluation_count_by_setting(preference_pairs)
        }
        
        # Create output structure
        output_data = {
            'metadata': metadata,
            'preference_pairs': preference_pairs
        }
        
        # Save to file
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"Saved {len(preference_pairs)} preference pairs to {output_file}")
        return output_data


# Example usage and testing
if __name__ == "__main__":
    print("=== Evaluation Engine V3 Testing ===")
    
    # Initialize evaluation engine
    engine = EvaluationEngineV3()
    
    # Test with a small subset
    print("\n1. Testing preference pair generation with subset...")
    
    # Use first 2 questions and first 4 user profiles for testing
    test_questions = [engine.synthetic_data[i]['question_id'] for i in range(min(2, len(engine.synthetic_data)))]
    test_profiles = [engine.user_profiles['user_profiles'][i]['user_profile_id'] for i in range(min(4, len(engine.user_profiles['user_profiles'])))]
    
    print(f"   Test questions: {test_questions}")
    print(f"   Test profiles: {test_profiles}")
    
    # Generate preference pairs
    preference_pairs = engine.generate_all_preference_pairs(
        question_ids=test_questions,
        user_profile_ids=test_profiles
    )
    
    # Show statistics
    print(f"\n2. Test results:")
    print(f"   Generated {len(preference_pairs)} preference pairs")
    
    if preference_pairs:
        print(f"   Style families: {set(p['style_family'] for p in preference_pairs)}")
        print(f"   Preference rules: {set(p['preference_rule'] for p in preference_pairs)}")
        print(f"   Quadrants: {set(p['quadrant'] for p in preference_pairs)}")
        
        # Show evaluation counts
        counts = engine.get_evaluation_count_by_setting(preference_pairs)
        print(f"\n3. Evaluation count projections:")
        for setting, count in counts.items():
            if not setting.endswith('_cot'):  # Show non-COT first
                print(f"   {setting}: {count:,}")
        
        # Save test results
        test_output_file = "test_preference_pairs_v3.json"
        engine.save_preference_pairs(preference_pairs, test_output_file)
        
        print("\nEvaluation Engine V3 testing completed!")
        print(f"Test data saved to {test_output_file}")
    else:
        print("   No preference pairs generated - check data compatibility") 