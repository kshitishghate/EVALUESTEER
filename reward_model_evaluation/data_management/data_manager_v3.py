import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
import sys
import os

# Add parent directories to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent / "synthetic_data_generation_v2"))

class DataManagerV3:
    """
    Comprehensive data manager 
    """
    
    def __init__(self, base_path: str = None):
        """
        Initialize the data manager with paths to data sources.
        
        Args:
            base_path: Base path to the project directory
        """
        if base_path is None:
            self.base_path = Path(__file__).parent.parent.parent
        else:
            self.base_path = Path(base_path)
            
        # Data file paths
        self.data_paths = {
            'user_profiles_v3': self.base_path / "synthetic_data_generation/user_profile_combinations.json",
            'distinct_value_profiles': self.base_path / "synthetic_data_generation/wvs_user_profile_generation/distinct_value_profiles.json",
            'synthetic_data': self.base_path / "synthetic_data_generation/prism_wvs_generated_data_with_style_variations.json",
            'wvs_human_data': self.base_path / "indievalue/IndieValue/demographics_in_nl_statements_combined_full_set.jsonl",
            'question_metadata': self.base_path / "indievalue/data/meta_data/statements_meta_data.jsonl"
        }
        
        # Cache for loaded data
        self._cache = {}
        
        # Question sets for filtering
        # Note: The synthetic data uses standard Q1, Q2, etc. format
        # These are the core questions used for user classification
        self.key_wvs_questions = [
            'Q164',  # God importance
            'Q17',   # Child qualities - Obedience
            'Q15',   # Child qualities - Religious faith
            'Q8',    # Child qualities - Independence
            'Q14',   # Child qualities - Determination
            'Q184',  # Abortion justification
            'Q254',  # National pride
            'Q45',   # Respect for authority
            'Q154',  # Materialism priority 1
            'Q155',  # Materialism priority 2
            'Q46',   # Happiness level
            'Q182',  # Homosexuality justification
            'Q209',  # Petition participation
            'Q57'    # Interpersonal trust
        ]
        
        # Map to _aid versions for WVS human data access
        self.key_wvs_questions_aid = [q + '_aid' for q in self.key_wvs_questions]
        
        # self.wvq_question_ids = [
        #     # SOCIAL VALUES (15 questions)
        #     27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41,
        #     # MIGRATION (8 questions)
        #     122, 123, 124, 125, 126, 127, 128, 129,
        #     # SECURITY (7 questions)
        #     132, 133, 134, 135, 136, 137, 138,
        #     # SCIENCE & TECHNOLOGY (5 questions)
        #     158, 159, 160, 161, 162,
        #     # RELIGIOUS VALUES (2 questions)
        #     169, 170,
        #     # ETHICAL VALUES AND NORMS (3 questions)
        #     196, 197, 198,
        #     # POLITICAL INTEREST & POLITICAL PARTICIPATION (10 questions)
        #     224, 225, 226, 227, 228, 229, 230, 231, 232, 233
        # ]
        
        self.wvq_question_ids = [
            # SOCIAL VALUES (15 questions)
            27, 28, 
            # MIGRATION (8 questions)
            122, 123, 
            # SECURITY (7 questions)
            132, 133,
            # SCIENCE & TECHNOLOGY (5 questions)
            158, 159, 
            # RELIGIOUS VALUES (2 questions)
            169, 170,
            # ETHICAL VALUES AND NORMS (3 questions)
            196, 197,
            # POLITICAL INTEREST & POLITICAL PARTICIPATION (10 questions)
            224, 225
        ]
        
        # Convert WVQ question IDs to the format used in data
        self.wvq_questions = [f'Q{q_id}' for q_id in self.wvq_question_ids]
        
        # All 64 filtered questions (14 key WVS + 50 WVQ)
        self.filtered_questions = self.key_wvs_questions + self.wvq_questions
        
    def load_user_profiles_v3(self) -> Dict[str, Any]:
        """
        Load the 288 user profile combinations from v3 data generation.
        
        Returns:
            Dictionary containing user profiles and statistics
        """
        if 'user_profiles_v3' not in self._cache:
            with open(self.data_paths['user_profiles_v3'], 'r') as f:
                data = json.load(f)
            self._cache['user_profiles_v3'] = data
            print(f"Loaded {data['statistics']['total_user_profiles']} user profiles")
        
        return self._cache['user_profiles_v3']
    
    def load_distinct_value_profiles(self) -> Dict[str, Any]:
        """
        Load the 18 distinct value profiles with detailed WVS responses.
        
        Returns:
            Dictionary containing distinct value profiles
        """
        if 'distinct_value_profiles' not in self._cache:
            with open(self.data_paths['distinct_value_profiles'], 'r') as f:
                data = json.load(f)
            self._cache['distinct_value_profiles'] = data
            print(f"Loaded {data['summary']['total_profiles']} distinct value profiles")
        
        return self._cache['distinct_value_profiles']
    
    def load_synthetic_data(self) -> List[Dict[str, Any]]:
        """
        Load synthetic data with style variations.
        
        Returns:
            List of synthetic question-answer pairs with style variations
        """
        if 'synthetic_data' not in self._cache:
            with open(self.data_paths['synthetic_data'], 'r') as f:
                data = json.load(f)
            self._cache['synthetic_data'] = data
            print(f"Loaded {len(data)} synthetic questions")
        
        return self._cache['synthetic_data']
    
    def load_wvs_human_data(self) -> pd.DataFrame:
        """
        Load WVS human response data.
        
        Returns:
            DataFrame containing WVS human responses
        """
        if 'wvs_human_data' not in self._cache:
            data = []
            with open(self.data_paths['wvs_human_data'], 'r') as f:
                for line in f:
                    data.append(json.loads(line))
            df = pd.DataFrame(data)
            self._cache['wvs_human_data'] = df
            print(f"Loaded WVS data for {len(df)} users")
        
        return self._cache['wvs_human_data']
    
    def load_question_metadata(self) -> pd.DataFrame:
        """
        Load question metadata for context generation.
        
        Returns:
            DataFrame containing question metadata
        """
        if 'question_metadata' not in self._cache:
            metadata = []
            with open(self.data_paths['question_metadata'], 'r') as f:
                for line in f:
                    metadata.append(json.loads(line))
            df = pd.DataFrame(metadata)
            self._cache['question_metadata'] = df
            print(f"Loaded metadata for {len(df)} questions")
        
        return self._cache['question_metadata']
    
    def get_filtered_questions(self) -> List[str]:
        """
        Get the list of 64 filtered questions for evaluation.
        
        Returns:
            List of question IDs for the 64 filtered questions
        """
        return self.filtered_questions.copy()
    
    def filter_synthetic_data_by_questions(self, question_ids: List[str] = None) -> List[Dict[str, Any]]:
        """
        Filter synthetic data to only include the specified questions.
        
        Args:
            question_ids: List of question IDs to include (defaults to filtered_questions)
            
        Returns:
            Filtered list of synthetic data items
        """
        if question_ids is None:
            question_ids = self.filtered_questions
            
        synthetic_data = self.load_synthetic_data()
        
        # Create mapping from question_id to data item
        question_map = {item['question_id']: item for item in synthetic_data}
        
        filtered_data = []
        for q_id in question_ids:
            if q_id in question_map:
                filtered_data.append(question_map[q_id])
            else:
                print(f"Warning: Question {q_id} not found in synthetic data")
        
        print(f"Filtered synthetic data: {len(filtered_data)}/{len(synthetic_data)} questions")
        return filtered_data
    
    def get_user_profile_by_id(self, user_profile_id: str) -> Dict[str, Any]:
        """
        Get a specific user profile by ID.
        
        Args:
            user_profile_id: User profile ID (e.g., "user_profile_042")
            
        Returns:
            User profile dictionary
        """
        user_profiles = self.load_user_profiles_v3()
        
        for profile in user_profiles['user_profiles']:
            if profile['user_profile_id'] == user_profile_id:
                return profile
                
        raise ValueError(f"User profile {user_profile_id} not found")
    
    def get_value_profile_by_id(self, value_profile_id: int) -> Dict[str, Any]:
        """
        Get a specific value profile by WVS user ID.
        
        Args:
            value_profile_id: WVS user ID (e.g., 231070731)
            
        Returns:
            Value profile dictionary
        """
        distinct_profiles = self.load_distinct_value_profiles()
        
        for quadrant, profiles in distinct_profiles['distinct_profiles'].items():
            for profile in profiles:
                if profile['user_id'] == value_profile_id:
                    return profile
                    
        raise ValueError(f"Value profile {value_profile_id} not found")
    
    def generate_full_wvs_context(self, value_profile_id: int) -> str:
        """
        Generate full WVS context string for a value profile.
        
        Args:
            value_profile_id: WVS user ID
            
        Returns:
            Formatted context string describing user's values and beliefs
        """
        # Get the value profile
        value_profile = self.get_value_profile_by_id(value_profile_id)
        
        # Load WVS and metadata for context generation
        wvs_df = self.load_wvs_human_data()
        metadata_df = self.load_question_metadata()
        
        # Get user's row in WVS data
        user_rows = wvs_df[wvs_df['D_INTERVIEW'] == value_profile_id]
        if len(user_rows) == 0:
            return f"No WVS context available for user {value_profile_id}."
        
        user_row = user_rows.iloc[0]
        
        # Get all question columns (Q1, Q2, etc.)
        question_cols = [col for col in wvs_df.columns if col.startswith('Q') and not col.endswith('_aid')]
        
        context_statements = []
        
        for q_col in question_cols:
            user_response = user_row.get(q_col)
            
            # Skip if the user didn't answer this question
            if pd.isna(user_response):
                continue
            
            # Skip if the response is not numeric
            try:
                user_response = int(user_response)
            except (ValueError, TypeError):
                continue
            
            # Skip invalid values like -99
            if user_response < 0:
                continue
                
            # Find the metadata for this question
            q_metadata_rows = metadata_df[metadata_df['question_id'] == q_col]
            
            if len(q_metadata_rows) == 0:
                continue
                
            q_metadata = q_metadata_rows.iloc[0]
            
            # Get the answer_ids_to_grouped_answer_ids mapping
            answer_mapping = q_metadata['answer_ids_to_grouped_answer_ids']
            
            # Find which group this user's response belongs to
            group_id = None
            for aid, gid in answer_mapping.items():
                if int(aid) == user_response:
                    group_id = gid
                    break
            
            if group_id is not None:
                # Get the statement that corresponds to this group
                converted_statements = q_metadata['converted_statements']
                if group_id < len(converted_statements):
                    statement = converted_statements[group_id]
                    context_statements.append(f"- {statement}")
        
        if not context_statements:
            return f"No specific values identified from user {value_profile_id}'s WVS responses."
        
        context = f"Based on this user's World Values Survey responses, the following statements describe their core values and beliefs:\n\n" + "\n".join(context_statements)
        return context
    
    def generate_full_style_context(self, style_profile: Dict[str, str]) -> str:
        """
        Generate full style context string for a style profile.
        
        Args:
            style_profile: Dictionary containing style preferences
            
        Returns:
            Formatted context string describing user's style preferences
        """
        if not style_profile:
            return "No style preferences available."
        
        context_parts = []
        context_parts.append("This user has the following style preferences:")
        
        if 'verbosity' in style_profile:
            verbosity = style_profile['verbosity']
            if verbosity == 'verbose':
                context_parts.append("- They prefer detailed, comprehensive explanations with thorough elaboration")
            else:  # concise
                context_parts.append("- They prefer brief, to-the-point responses that get straight to the core message")
        
        if 'readability' in style_profile:
            readability = style_profile['readability']
            if readability == 'high_reading_difficulty':
                context_parts.append("- They prefer sophisticated, complex language with advanced vocabulary and intricate sentence structures")
            else:  # low_reading_difficulty
                context_parts.append("- They prefer simple, accessible language that is easy to understand and follow")
        
        if 'confidence' in style_profile:
            confidence = style_profile['confidence']
            if confidence == 'high_confidence':
                context_parts.append("- They prefer confident, assertive responses with clear, definitive statements")
            else:  # low_confidence
                context_parts.append("- They prefer humble, tentative responses that acknowledge uncertainty and limitations")
        
        if 'sentiment' in style_profile:
            sentiment = style_profile['sentiment']
            if sentiment == 'warm':
                context_parts.append("- They prefer warm, friendly responses with an encouraging and supportive tone")
            else:  # cold
                context_parts.append("- They prefer formal, professional responses with an objective and detached tone")
        
        return "\n".join(context_parts)
    
    def generate_combined_context(self, value_profile_id: int, style_profile: Dict[str, str], 
                                 wvs_first: bool = True) -> str:
        """
        Generate combined WVS and style context.
        
        Args:
            value_profile_id: WVS user ID
            style_profile: Style preference dictionary
            wvs_first: Whether to put WVS context first
            
        Returns:
            Combined context string
        """
        wvs_context = self.generate_full_wvs_context(value_profile_id)
        style_context = self.generate_full_style_context(style_profile)
        
        if wvs_first:
            return f"User's Values and Beliefs:\n{wvs_context}\n\nUser's Style Preferences:\n{style_context}"
        else:
            return f"User's Style Preferences:\n{style_context}\n\nUser's Values and Beliefs:\n{wvs_context}"
    
    def get_evaluation_statistics(self) -> Dict[str, Any]:
        """
        Calculate expected evaluation statistics for different settings.
        
        Returns:
            Dictionary containing evaluation count projections
        """
        n_questions = len(self.get_filtered_questions())
        n_style_families = 4
        n_combinations_per_family = 6
        n_user_profiles = 288
        n_value_profiles = 18
        n_style_profiles = 16
        
        base_combinations = n_questions * n_style_families * n_combinations_per_family
        
        stats = {
            'data_scope': {
                'total_questions': n_questions,
                'key_wvs_questions': len(self.key_wvs_questions),
                'wvq_questions': len(self.wvq_questions),
                'style_families': n_style_families,
                'combinations_per_family': n_combinations_per_family,
                'user_profiles': n_user_profiles,
                'value_profiles': n_value_profiles,
                'style_profiles': n_style_profiles
            },
            'evaluation_counts': {
                'simple_prompting': base_combinations,
                'full_wvs_only': base_combinations * n_value_profiles,
                'full_style_only': base_combinations * n_style_profiles,
                'full_combined_per_priority': base_combinations * n_value_profiles * n_style_profiles,
                'full_combined_all_priorities': base_combinations * n_value_profiles * n_style_profiles * 3
            }
        }
        
        # Add totals
        stats['totals'] = {
            'primary_settings_without_cot': (
                stats['evaluation_counts']['simple_prompting'] +
                stats['evaluation_counts']['full_wvs_only'] +
                stats['evaluation_counts']['full_style_only'] +
                stats['evaluation_counts']['full_combined_all_priorities']
            ),
            'primary_settings_with_cot': 0  # Will be double the above
        }
        stats['totals']['primary_settings_with_cot'] = stats['totals']['primary_settings_without_cot'] * 2
        
        return stats
    
    def validate_data_integrity(self) -> Dict[str, bool]:
        """
        Validate the integrity and completeness of all data sources.
        
        Returns:
            Dictionary indicating validation status for each component
        """
        validation_results = {}
        
        try:
            # Validate user profiles
            user_profiles = self.load_user_profiles_v3()
            validation_results['user_profiles_v3'] = (
                len(user_profiles['user_profiles']) == 288 and
                user_profiles['statistics']['total_user_profiles'] == 288
            )
        except Exception as e:
            print(f"User profiles validation failed: {e}")
            validation_results['user_profiles_v3'] = False
        
        try:
            # Validate distinct value profiles
            value_profiles = self.load_distinct_value_profiles()
            validation_results['distinct_value_profiles'] = (
                value_profiles['summary']['total_profiles'] == 18
            )
        except Exception as e:
            print(f"Value profiles validation failed: {e}")
            validation_results['distinct_value_profiles'] = False
        
        try:
            # Validate synthetic data
            synthetic_data = self.load_synthetic_data()
            validation_results['synthetic_data'] = len(synthetic_data) > 200  # Should have ~233 questions
        except Exception as e:
            print(f"Synthetic data validation failed: {e}")
            validation_results['synthetic_data'] = False
        
        try:
            # Validate filtered questions
            filtered_data = self.filter_synthetic_data_by_questions()
            validation_results['filtered_questions'] = len(filtered_data) == 64
        except Exception as e:
            print(f"Filtered questions validation failed: {e}")
            validation_results['filtered_questions'] = False
        
        try:
            # Validate WVS human data
            wvs_df = self.load_wvs_human_data()
            validation_results['wvs_human_data'] = len(wvs_df) > 80000  # Should have ~85k users
        except Exception as e:
            print(f"WVS human data validation failed: {e}")
            validation_results['wvs_human_data'] = False
        
        return validation_results
    
    def get_available_key_questions(self) -> Tuple[List[str], List[str]]:
        """
        Check which key WVS questions are available in the synthetic data.
        
        Returns:
            Tuple of (available_questions, missing_questions)
        """
        synthetic_data = self.load_synthetic_data()
        available_q_ids = {item['question_id'] for item in synthetic_data}
        
        available_key_questions = []
        missing_key_questions = []
        
        for q_id in self.key_wvs_questions:
            if q_id in available_q_ids:
                available_key_questions.append(q_id)
            else:
                missing_key_questions.append(q_id)
        
        return available_key_questions, missing_key_questions


# Example usage and testing
if __name__ == "__main__":
    # Initialize data manager
    dm = DataManagerV3()
    
    # Test data loading
    print("=== Data Manager V3 Testing ===")
    
    # Validate data integrity
    print("\n1. Validating data integrity...")
    validation = dm.validate_data_integrity()
    for component, is_valid in validation.items():
        status = "Valid" if is_valid else "Invalid"
        print(f"   {status} {component}: {'PASS' if is_valid else 'FAIL'}")
    
    # Show evaluation statistics
    print("\n2. Evaluation statistics...")
    stats = dm.get_evaluation_statistics()
    print(f"   Questions: {stats['data_scope']['total_questions']} (64 filtered)")
    print(f"   User profiles: {stats['data_scope']['user_profiles']}")
    print(f"   Simple prompting: {stats['evaluation_counts']['simple_prompting']:,} evaluations")
    print(f"   Full WVS only: {stats['evaluation_counts']['full_wvs_only']:,} evaluations")
    print(f"   Full style only: {stats['evaluation_counts']['full_style_only']:,} evaluations")
    print(f"   Full combined (all priorities): {stats['evaluation_counts']['full_combined_all_priorities']:,} evaluations")
    print(f"   Total (without COT): {stats['totals']['primary_settings_without_cot']:,} evaluations")
    print(f"   Total (with COT): {stats['totals']['primary_settings_with_cot']:,} evaluations")
    
    # Test context generation
    print("\n3. Testing context generation...")
    user_profiles = dm.load_user_profiles_v3()
    sample_profile = user_profiles['user_profiles'][0]
    
    # Generate WVS context
    wvs_context = dm.generate_full_wvs_context(sample_profile['value_profile_id'])
    print(f"   WVS context length: {len(wvs_context)} characters")
    
    # Generate style context
    style_context = dm.generate_full_style_context(sample_profile['style_profile'])
    print(f"   Style context length: {len(style_context)} characters")
    
    # Generate combined context
    combined_context = dm.generate_combined_context(
        sample_profile['value_profile_id'], 
        sample_profile['style_profile']
    )
    print(f"   Combined context length: {len(combined_context)} characters")
    
    print("\nData Manager V3 testing completed!") 
