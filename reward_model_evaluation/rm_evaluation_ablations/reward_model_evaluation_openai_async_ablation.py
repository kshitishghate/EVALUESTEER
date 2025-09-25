import json
import pandas as pd
import numpy as np
import random
from collections import defaultdict
import os
from tqdm import tqdm
import time
import traceback
import argparse
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Literal
import asyncio

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from data_management.data_manager_v3 import DataManagerV3
from evaluation_engine.evaluation_engine_v3 import EvaluationEngineV3

try:
    import openai
    from openai import AsyncOpenAI
    print("OpenAI client available")
except ImportError:
    print("Error: OpenAI not installed. Please install with: pip install openai")
    openai = None
    AsyncOpenAI = None

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
        """Structured response schema for both COT and non-COT evaluations"""
        reasoning: Optional[str] = Field(default=None, description="Step-by-step analysis of user preferences and response alignment (COT only)")
        final_choice: Literal["A", "B"] = Field(description="The final answer: either A or B")
        confidence: Optional[float] = Field(default=None, ge=0, le=1, description="Optional confidence score between 0 and 1")


class AblationDataManager(DataManagerV3):
    """Extended data manager for WVS ablation studies"""
    
    def __init__(self, base_path: str = None):
        super().__init__(base_path)
        
        # Filter key_wvs_questions to only include those available in synthetic data
        self.available_key_wvs_questions = self._get_available_key_wvs_questions()
        
    def _get_available_key_wvs_questions(self) -> List[str]:
        """Get only the key WVS questions that are actually available in synthetic data"""
        synthetic_data = self.load_synthetic_data()
        available_question_ids = {item['question_id'] for item in synthetic_data}
        
        available_key_wvs = []
        missing_key_wvs = []
        
        for q_id in self.key_wvs_questions:
            if q_id in available_question_ids:
                available_key_wvs.append(q_id)
            else:
                missing_key_wvs.append(q_id)
        
        if missing_key_wvs:
            print(f"Note: Excluding {len(missing_key_wvs)} key WVS questions not in synthetic data: {missing_key_wvs}")
        
        print(f"Available key WVS questions for ablation: {len(available_key_wvs)} questions")
        return available_key_wvs
        
    def generate_ablation_wvs_context(self, value_profile_id: int, question_id: str, 
                                    num_statements: int = 4, random_seed: int = None) -> str:
        """
        Generate ablated WVS context with limited statements from core WVS questions only.
        
        Args:
            value_profile_id: WVS user ID
            question_id: Current question ID to ensure relevant statement is included
            num_statements: Total number of statements to include (default: 4)
            random_seed: Random seed for reproducible statement selection (NOTE: only used for this specific context generation)
            
        Returns:
            Formatted context string with limited WVS statements from core WVS questions
        """
        # Create a local random state for this context generation only
        # This preserves the global random state for other randomization
        local_random = random.Random(random_seed) if random_seed is not None else random
            
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
        
        # Get statements from available core WVS questions only
        core_wvs_statements = []
        relevant_statement = None
        
        for q_col in self.available_key_wvs_questions:
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
                    
                    # Check if this is the relevant statement for the current question
                    if q_col == question_id:
                        relevant_statement = statement
                    else:
                        core_wvs_statements.append(statement)
        
        # Select statements for ablation context
        selected_statements = []
        
        # 1. Always include the relevant statement if available
        if relevant_statement:
            selected_statements.append(f"- {relevant_statement}")
            remaining_slots = num_statements - 1
        else:
            # No relevant statement found (expected for WVQ questions)
            remaining_slots = num_statements
        
        # 2. Randomly select additional statements from core WVS questions only  
        if core_wvs_statements and remaining_slots > 0:
            # Ensure we don't select more than available
            n_to_select = min(remaining_slots, len(core_wvs_statements))
            selected_additional = local_random.sample(core_wvs_statements, n_to_select)
            
            for stmt in selected_additional:
                selected_statements.append(f"- {stmt}")
        
        if not selected_statements:
            return f"No core WVS statements available for user {value_profile_id} (ablation)."
        
        # Randomize order to avoid position bias
        if len(selected_statements) > 1:
            local_random.shuffle(selected_statements)
        
        context = (f"Based on this user's World Values Survey responses, the following statements "
                  f"describe their core values and beliefs (ablated to {len(selected_statements)} key statements from core WVS questions):\n\n" + 
                  "\n".join(selected_statements))
        
        return context


class AblationPreferenceFilter:
    """Filter preference pairs to identify value-style conflicts for ablation study"""
    
    def __init__(self):
        pass
    
    def parse_user_style_preferences(self, style_code: str) -> Dict[str, str]:
        """Parse user's style_code to extract their preferences"""
        if not style_code:
            return {}
        
        parts = style_code.split('_')
        
        if len(parts) != 4:
            return {}
        
        # Map abbreviated codes to full names
        verbosity_map = {'v': 'verbose', 'c': 'concise'}
        readability_map = {'hrd': 'high_reading_difficulty', 'lrd': 'low_reading_difficulty'}
        confidence_map = {'hc': 'high_confidence', 'lc': 'low_confidence'}
        sentiment_map = {'w': 'warm', 'co': 'cold'}
        
        style_prefs = {}
        
        # Parse each component
        if parts[0] in verbosity_map:
            style_prefs['verbosity'] = verbosity_map[parts[0]]
        
        if parts[1] in readability_map:
            style_prefs['readability'] = readability_map[parts[1]]
        
        if parts[2] in confidence_map:
            style_prefs['confidence'] = confidence_map[parts[2]]
        
        if parts[3] in sentiment_map:
            style_prefs['sentiment'] = sentiment_map[parts[3]]
        
        return style_prefs
    
    def has_style_differences(self, combination_type: str) -> bool:
        """Check if the combination type involves different styles"""
        if '_vs_' not in combination_type:
            return False
        
        left_part, right_part = combination_type.split('_vs_')
        
        # Extract style information from each side
        left_style = self.extract_style_from_combination_part(left_part)
        right_style = self.extract_style_from_combination_part(right_part)
        
        # Check if any style dimension differs
        return left_style != right_style
    
    def extract_style_from_combination_part(self, combination_part: str) -> str:
        """Extract style information from combination part like 'A_verbose' or 'B_high_confidence'"""
        # Remove A_ or B_ prefix
        if combination_part.startswith(('A_', 'B_')):
            style_part = combination_part[2:]
        else:
            style_part = combination_part
        
        return style_part
    
    def user_style_in_nonpreferred_completion(self, preferred_key: str, non_preferred_key: str, 
                                           user_style_prefs: Dict[str, str]) -> bool:
        """
        Check if user's preferred style appears in the non-preferred completion.
        This creates a value-style conflict where the model must choose between them.
        """
        if not preferred_key or not non_preferred_key:
            return False
        
        # Extract style from non-preferred completion key
        try:
            non_preferred_parts = non_preferred_key.split('_')
            if len(non_preferred_parts) < 2:
                return False
            
            # Skip "A" or "B", get the style part
            non_preferred_style = '_'.join(non_preferred_parts[1:])
            
        except (IndexError, AttributeError):
            return False
        
        # Check if any of the user's preferred styles appear in the non-preferred completion
        for style_family, user_pref in user_style_prefs.items():
            if user_pref in non_preferred_style:
                return True
        
        return False
    
    def identify_value_style_conflicts(self, preference_pairs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Identify preference pairs where value and style preferences conflict.
        
        Args:
            preference_pairs: List of all preference pairs
            
        Returns:
            List of preference pairs with value-style conflicts
        """
        conflict_cases = []
        total_cases = 0
        cases_with_style_prefs = 0
        cases_with_keys = 0
        cases_with_style_diff = 0
        wvs_based_cases = 0
        
        for pair in preference_pairs:
            total_cases += 1
            
            # Only consider WVS-based preferences (where values differ)
            if pair.get('preference_rule') != 'wvs_based':
                continue
            wvs_based_cases += 1
            
            # Parse user's style preferences
            user_style_prefs = self.parse_user_style_preferences(pair.get('style_code', ''))
            if not user_style_prefs:
                continue
            cases_with_style_prefs += 1
            
            # Get actual completion keys
            preferred_key = pair.get('preferred_completion_key', '')
            non_preferred_key = pair.get('non_preferred_completion_key', '')
            
            if not preferred_key or not non_preferred_key:
                continue
            cases_with_keys += 1
                
            # Parse combination type to ensure styles differ
            combination_type = pair.get('combination_type', '')
            if not self.has_style_differences(combination_type):
                continue
            cases_with_style_diff += 1
            
            # Check if user's preferred style appears in non-preferred completion
            if self.user_style_in_nonpreferred_completion(
                preferred_key, non_preferred_key, user_style_prefs):
                conflict_cases.append(pair)
        
        # Debug output
        print("Value-Style Conflict Filtering:")
        print(f"   Total preference pairs examined: {total_cases}")
        print(f"   WVS-based cases: {wvs_based_cases}")
        print(f"   Cases with style preferences: {cases_with_style_prefs}")
        print(f"   Cases with completion keys: {cases_with_keys}")
        print(f"   Cases with style differences: {cases_with_style_diff}")
        print(f"   Final conflict cases identified: {len(conflict_cases)}")
        
        return conflict_cases


# ===== PROMPT GENERATION FUNCTIONS FOR ABLATION =====

def create_ablation_wvs_only_prompt(prompt, completion_a, completion_b, ablation_wvs_context):
    """Create a prompt with ablated WVS context only"""
    return f"""You are an AI assistant that helps evaluate which response better answers a given question for a specific user.

User's Values and Beliefs:
{ablation_wvs_context}

Question: {prompt}

Response A: {completion_a}

Response B: {completion_b}

Given what you know about this user's values and beliefs, which response (A or B) would this specific user prefer?

Respond with valid JSON in exactly this format:
{{
    "final_choice": "A"
}}

Where final_choice must be either "A" or "B"."""

def create_ablation_wvs_style_neutral_prompt(prompt, completion_a, completion_b, ablation_wvs_context, style_context, wvs_first):
    """Create a prompt with ablated WVS and style context, no preference guidance"""
    if wvs_first:
        context_section = f"""User's Values and Beliefs:
{ablation_wvs_context}

User's Style Preferences:
{style_context}"""
    else:
        context_section = f"""User's Style Preferences:
{style_context}

User's Values and Beliefs:
{ablation_wvs_context}"""
    
    return f"""You are an AI assistant that helps evaluate which response better answers a given question for a specific user.

{context_section}

Question: {prompt}

Response A: {completion_a}

Response B: {completion_b}

Given what you know about this user's values and style preferences, which response (A or B) would this specific user prefer?

Respond with valid JSON in exactly this format:
{{
    "final_choice": "A"
}}

Where final_choice must be either "A" or "B"."""

def create_ablation_wvs_style_prefer_wvs_prompt(prompt, completion_a, completion_b, ablation_wvs_context, style_context, wvs_first):
    """Create a prompt with ablated WVS and style context, emphasizing values take precedence"""
    if wvs_first:
        context_section = f"""User's Values and Beliefs:
{ablation_wvs_context}

User's Style Preferences:
{style_context}"""
    else:
        context_section = f"""User's Style Preferences:
{style_context}

User's Values and Beliefs:
{ablation_wvs_context}"""
    
    return f"""You are an AI assistant that helps evaluate which response better answers a given question for a specific user.

{context_section}

Question: {prompt}

Response A: {completion_a}

Response B: {completion_b}

Given what you know about this user's values and style preferences, which response (A or B) would this specific user prefer? IMPORTANT: Aligning with the user's values and beliefs takes the highest precedence. Style preferences are secondary to value alignment.

Respond with valid JSON in exactly this format:
{{
    "final_choice": "A"
}}

Where final_choice must be either "A" or "B"."""

def create_ablation_wvs_style_prefer_style_prompt(prompt, completion_a, completion_b, ablation_wvs_context, style_context, wvs_first):
    """Create a prompt with ablated WVS and style context, emphasizing style preferences over values"""
    if wvs_first:
        context_section = f"""User's Values and Beliefs:
{ablation_wvs_context}

User's Style Preferences:
{style_context}"""
    else:
        context_section = f"""User's Style Preferences:
{style_context}

User's Values and Beliefs:
{ablation_wvs_context}"""
    
    return f"""You are an AI assistant that helps evaluate which response better answers a given question for a specific user.

{context_section}

Question: {prompt}

Response A: {completion_a}

Response B: {completion_b}

Given what you know about this user's values and style preferences, which response (A or B) would this specific user prefer? IMPORTANT: This user prioritizes communication style and format over content alignment with their values. Style preferences take the highest precedence.

Respond with valid JSON in exactly this format:
{{
    "final_choice": "A"
}}

Where final_choice must be either "A" or "B"."""

# ===== COT PROMPT GENERATION FUNCTIONS FOR ABLATION =====

def create_ablation_wvs_only_cot_prompt(prompt, completion_a, completion_b, ablation_wvs_context):
    """Create a COT prompt with ablated WVS context only - returns JSON with reasoning"""
    return f"""You are an AI assistant that helps evaluate which response better answers a given question for a specific user.

User's Values and Beliefs:
{ablation_wvs_context}

Question: {prompt}

Response A: {completion_a}

Response B: {completion_b}

Please analyze step-by-step:

1. First, analyze the user's values and beliefs to identify which aspects are most relevant to this question and the given responses.

2. Consider how each response aligns with those relevant aspects of the user's values and beliefs.

3. Based on your analysis, determine which response this specific user would prefer given their values.

Respond with valid JSON in exactly this format:
{{
    "reasoning": "Your detailed step-by-step analysis here",
    "final_choice": "A"
}}

Where final_choice must be either "A" or "B"."""

def create_ablation_wvs_style_neutral_cot_prompt(prompt, completion_a, completion_b, ablation_wvs_context, style_context, wvs_first):
    """Create a COT prompt with ablated WVS and style context, no preference guidance - returns JSON with reasoning"""
    if wvs_first:
        context_section = f"""User's Values and Beliefs:
{ablation_wvs_context}

User's Style Preferences:
{style_context}"""
    else:
        context_section = f"""User's Style Preferences:
{style_context}

User's Values and Beliefs:
{ablation_wvs_context}"""
    
    return f"""You are an AI assistant that helps evaluate which response better answers a given question for a specific user.

{context_section}

Question: {prompt}

Response A: {completion_a}

Response B: {completion_b}

Please analyze step-by-step:

1. First, analyze the user's profile to identify which aspects of their values and style preferences are most relevant to this question and the given responses.

2. Consider how each response aligns with those relevant aspects of the user's profile.

3. Based on your analysis, determine which response this specific user would prefer given their overall profile.

Respond with valid JSON in exactly this format:
{{
    "reasoning": "Your detailed step-by-step analysis here",
    "final_choice": "A"
}}

Where final_choice must be either "A" or "B"."""

def create_ablation_wvs_style_prefer_wvs_cot_prompt(prompt, completion_a, completion_b, ablation_wvs_context, style_context, wvs_first):
    """Create a COT prompt with ablated WVS and style context, emphasizing values take precedence - returns JSON with reasoning"""
    if wvs_first:
        context_section = f"""User's Values and Beliefs:
{ablation_wvs_context}

User's Style Preferences:
{style_context}"""
    else:
        context_section = f"""User's Style Preferences:
{style_context}

User's Values and Beliefs:
{ablation_wvs_context}"""
    
    return f"""You are an AI assistant that helps evaluate which response better answers a given question for a specific user.

{context_section}

Question: {prompt}

Response A: {completion_a}

Response B: {completion_b}

Please analyze step-by-step:

1. First, analyze the user's profile to identify which aspects of their values and style preferences are most relevant to this question and the given responses.

2. Consider how each response aligns with those relevant aspects, prioritizing alignment with the user's values and beliefs over style preferences.

3. Based on your analysis, determine which response this specific user would prefer. Remember that aligning with the user's values and beliefs takes the highest precedence, with style preferences being secondary.

Respond with valid JSON in exactly this format:
{{
    "reasoning": "Your detailed step-by-step analysis here",
    "final_choice": "A"
}}

Where final_choice must be either "A" or "B"."""

def create_ablation_wvs_style_prefer_style_cot_prompt(prompt, completion_a, completion_b, ablation_wvs_context, style_context, wvs_first):
    """Create a COT prompt with ablated WVS and style context, emphasizing style preferences over values - returns JSON with reasoning"""
    if wvs_first:
        context_section = f"""User's Values and Beliefs:
{ablation_wvs_context}

User's Style Preferences:
{style_context}"""
    else:
        context_section = f"""User's Style Preferences:
{style_context}

User's Values and Beliefs:
{ablation_wvs_context}"""
    
    return f"""You are an AI assistant that helps evaluate which response better answers a given question for a specific user.

{context_section}

Question: {prompt}

Response A: {completion_a}

Response B: {completion_b}

Please analyze step-by-step:

1. First, analyze the user's profile to identify which aspects of their values and style preferences are most relevant to this question and the given responses.

2. Consider how each response aligns with those relevant aspects, prioritizing alignment with the user's style preferences over their values and beliefs.

3. Based on your analysis, determine which response this specific user would prefer. Remember that this user prioritizes communication style and format over content alignment with their values, so style preferences take the highest precedence.

Respond with valid JSON in exactly this format:
{{
    "reasoning": "Your detailed step-by-step analysis here",
    "final_choice": "A"
}}

Where final_choice must be either "A" or "B"."""


# ===== JSON PARSING =====

def _extract_reasoning_from_text(response_text: str) -> str:
    """
    Helper function to extract reasoning from response text when JSON parsing fails.
    Uses regex to find reasoning content.
    """
    try:
        import re
        # Look for reasoning patterns in the text
        reasoning_patterns = [
            r'"reasoning"\s*:\s*"([^"]+)"',
            r'"reasoning"\s*:\s*\'([^\']+)\'',
            r'reasoning["\']?\s*:\s*["\']([^"\']+)["\']',
            r'analysis["\']?\s*:\s*["\']([^"\']+)["\']',
            r'step-by-step["\']?\s*:\s*["\']([^"\']+)["\']'
        ]
        
        for pattern in reasoning_patterns:
            matches = re.findall(pattern, response_text, re.IGNORECASE | re.DOTALL)
            if matches:
                reasoning = matches[-1].strip()  # Take the last match
                # Clean up the reasoning text
                reasoning = reasoning.replace('\\n', '\n').replace('\\"', '"')
                return reasoning
        
        # If no structured reasoning found, try to extract the first substantial paragraph
        # that looks like reasoning
        lines = response_text.split('\n')
        reasoning_lines = []
        for line in lines:
            line = line.strip()
            if len(line) > 20 and ('analyze' in line.lower() or 'consider' in line.lower() or 
                                   'response' in line.lower() or 'prefer' in line.lower() or
                                   'user' in line.lower() or 'values' in line.lower()):
                reasoning_lines.append(line)
        
        if reasoning_lines:
            return ' '.join(reasoning_lines[:3])  # Take first 3 relevant lines
        
        # Fallback: return first non-empty substantial line
        for line in lines:
            line = line.strip()
            if len(line) > 20 and not line.startswith('{') and not line.startswith('}'):
                return line
        
        return "No reasoning extracted"
    except Exception as e:
        print(f"Error extracting reasoning: {e}")
        return "Reasoning extraction failed"


def parse_json_response(response_text: str) -> dict:
    """
    Parse structured JSON response from COT evaluation with robust error handling.
    Returns a dict with 'final_choice' and 'reasoning', or {'final_choice': 'ERROR', 'reasoning': None} if parsing fails.
    """
    if not response_text or not response_text.strip():
        print("Empty response")
        return {"final_choice": "ERROR", "reasoning": None}
    
    # Clean the response - remove any markdown formatting
    cleaned_response = response_text.strip()
    if cleaned_response.startswith("```json"):
        cleaned_response = cleaned_response[7:]
    if cleaned_response.endswith("```"):
        cleaned_response = cleaned_response[:-3]
    cleaned_response = cleaned_response.strip()
    
    # First attempt: Clean control characters and fix common JSON issues
    try:
        # Replace problematic control characters
        import re
        # Remove or replace control characters (except newlines, tabs, carriage returns)
        cleaned_response = re.sub(r'[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F]', '', cleaned_response)
        
        # Fix common JSON formatting issues
        # Replace literal newlines within JSON string values with escaped newlines
        lines = cleaned_response.split('\n')
        in_string = False
        escape_next = False
        result_lines = []
        current_line = ""
        
        for line in lines:
            i = 0
            while i < len(line):
                char = line[i]
                
                if escape_next:
                    escape_next = False
                elif char == '\\':
                    escape_next = True
                elif char == '"' and not escape_next:
                    in_string = not in_string
                
                i += 1
            
            # If we're not in a string or this looks like the end of a value, add as separate line
            if not in_string or line.strip().endswith('",') or line.strip().endswith('"'):
                current_line += line
                result_lines.append(current_line)
                current_line = ""
                in_string = False  # Reset for safety
            else:
                # We're in a string that continues to next line - escape the newline
                current_line += line + "\\n"
        
        # Add any remaining content
        if current_line:
            result_lines.append(current_line)
        
        cleaned_response = '\n'.join(result_lines)
        
        print(f"Attempting to parse cleaned JSON: {cleaned_response[:200]}...")
        
    except Exception as e:
        print(f"Error in JSON cleaning: {e}")
        # Continue with original response if cleaning fails
    
    # Second attempt: Try parsing the cleaned JSON
    try:
        response_data = json.loads(cleaned_response)
        print("Successfully parsed JSON")
        
        # Validate using Pydantic schema
        if BaseModel is not None:
            validated_response = COTResponse(**response_data)
            final_choice = validated_response.final_choice
            reasoning = validated_response.reasoning
            print(f"Pydantic validation successful: {final_choice}")
            return {"final_choice": final_choice, "reasoning": reasoning}
        else:
            # Fallback validation if Pydantic not available
            final_choice = response_data.get('final_choice', '').upper()
            reasoning = response_data.get('reasoning', '')
            if final_choice in ['A', 'B']:
                print(f"Manual validation successful: {final_choice}")
                return {"final_choice": final_choice, "reasoning": reasoning}
            else:
                print(f"Invalid final_choice value: {final_choice}")
                return {"final_choice": "ERROR", "reasoning": reasoning}
                
    except json.JSONDecodeError as e:
        print(f"JSON parsing error after cleaning: {e}")
        
        # Third attempt: More aggressive cleaning and regex extraction
        try:
            print("Attempting regex extraction fallback...")
            
            # Look for final_choice pattern directly
            import re
            
            # Try to find "final_choice": "A" or "final_choice": "B"
            choice_patterns = [
                r'"final_choice"\s*:\s*"([AB])"',
                r'"final_choice"\s*:\s*\'([AB])\'',
                r'final_choice["\']?\s*:\s*["\']?([AB])["\']?',
                r'"choice"\s*:\s*"([AB])"',
                r'"answer"\s*:\s*"([AB])"'
            ]
            
            for pattern in choice_patterns:
                matches = re.findall(pattern, response_text, re.IGNORECASE)
                if matches:
                    choice = matches[-1].upper()  # Take the last match
                    if choice in ['A', 'B']:
                        print(f"Regex extraction successful: {choice}")
                        # Try to extract reasoning from the response
                        reasoning = _extract_reasoning_from_text(response_text)
                        return {"final_choice": choice, "reasoning": reasoning}
            
            # Fourth attempt: Look for any clear indication of A or B choice
            # This is more permissive but still looks for structured indicators
            if 'final_choice' in response_text.lower():
                text_upper = response_text.upper()
                # Look for A or B near "final_choice"
                final_choice_pos = text_upper.find('FINAL_CHOICE')
                if final_choice_pos != -1:
                    # Look in a 50-character window after "final_choice"
                    window = text_upper[final_choice_pos:final_choice_pos + 50]
                    if '"A"' in window or "'A'" in window or ': A' in window:
                        print("Window extraction successful: A")
                        reasoning = _extract_reasoning_from_text(response_text)
                        return {"final_choice": "A", "reasoning": reasoning}
                    elif '"B"' in window or "'B'" in window or ': B' in window:
                        print("Window extraction successful: B")
                        reasoning = _extract_reasoning_from_text(response_text)
                        return {"final_choice": "B", "reasoning": reasoning}
            
            print("All parsing attempts failed")
            return {"final_choice": "ERROR", "reasoning": None}
            
        except Exception as regex_error:
            print(f"Regex extraction also failed: {regex_error}")
            return {"final_choice": "ERROR", "reasoning": None}
        
    except ValidationError as e:
        print(f"Pydantic validation error: {e}")
        return {"final_choice": "ERROR", "reasoning": None}
    except Exception as e:
        print(f"Unexpected parsing error: {e}")
        return {"final_choice": "ERROR", "reasoning": None}


# ===== MODEL INTERACTION =====

async def query_openai_model_async(
    openai_client, messages, model_name, max_tokens=800, temperature=0.0, 
    max_retries=3, retry_delay=1, is_cot=False
):
    """
    Query OpenAI model asynchronously with retry logic.
    
    Args:
        openai_client: AsyncOpenAI client
        messages: List of message dictionaries
        model_name: OpenAI model name
        max_tokens: Maximum tokens for response
        temperature: Sampling temperature
        max_retries: Maximum retry attempts
        retry_delay: Delay between retries
        is_cot: Whether this is a COT query (includes reasoning field)
    
    Returns:
        dict: Response dictionary with final_choice and reasoning (COT) or final_choice only (non-COT)
    """
    for attempt in range(max_retries):
        try:
            # Create completion request
            response = await openai_client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                response_format={"type": "json_object"}  # Force JSON output
            )
            
            if response.choices and len(response.choices) > 0:
                content = response.choices[0].message.content
                
                if not content:
                    print(f"Empty response on attempt {attempt + 1}, retrying...")
                    await asyncio.sleep(retry_delay)
                    continue
                
                # Both COT and non-COT now use JSON parsing
                parsed_result = parse_json_response(content)
                if parsed_result["final_choice"] != "ERROR":
                    return parsed_result
                else:
                    print(f"JSON parsing failed (attempt {attempt + 1}), retrying...")
                    await asyncio.sleep(retry_delay)
                    continue
            else:
                print(f"Empty or invalid response structure (attempt {attempt + 1})")
                await asyncio.sleep(retry_delay)
                continue
                
        except Exception as e:
            print(f"Error querying OpenAI model (attempt {attempt + 1}): {e}")
            if attempt < max_retries - 1:
                await asyncio.sleep(retry_delay)
            continue
    
    print(f"All {max_retries} attempts failed, returning ERROR")
    return {"final_choice": "ERROR", "reasoning": None}

async def batch_query_openai_models(
    openai_client, prompts_and_settings, model_name, max_tokens=800, 
    temperature=0.0, max_retries=3, retry_delay=1, max_workers=5
):
    """
    Batch query OpenAI models asynchronously with concurrency control.
    
    Args:
        openai_client: AsyncOpenAI client
        prompts_and_settings: List of (prompt_text, is_cot, setting_name) tuples
        model_name: OpenAI model name
        max_tokens: Maximum tokens for response
        temperature: Sampling temperature
        max_retries: Maximum retry attempts
        retry_delay: Delay between retries
        max_workers: Maximum concurrent requests
    
    Returns:
        List of response dictionaries
    """
    semaphore = asyncio.Semaphore(max_workers)
    
    async def bounded_query(messages, is_cot, setting_name):
        async with semaphore:
            result = await query_openai_model_async(
                openai_client, messages, model_name, max_tokens, 
                temperature, max_retries, retry_delay, is_cot
            )
            return result, setting_name
    
    # Create tasks for all prompts
    tasks = []
    for prompt_text, is_cot, setting_name in prompts_and_settings:
        messages = [{"role": "user", "content": prompt_text}]
        task = bounded_query(messages, is_cot, setting_name)
        tasks.append(task)
    
    # Execute all tasks concurrently with progress tracking
    results = []
    for task in asyncio.as_completed(tasks):
        result, setting_name = await task
        results.append((result, setting_name))
    
    # Sort results by setting order to maintain consistency
    setting_order = {setting: i for i, (_, _, setting) in enumerate(prompts_and_settings)}
    results.sort(key=lambda x: setting_order[x[1]])
    
    return [result for result, _ in results]




async def evaluate_preference_pair_all_settings_async(preference_pair, ablation_data_manager, openai_client, args, 
                                                     ablation_conflict_sets=None, user_profile=None):
    """
    Evaluate a single preference pair using all ablation settings asynchronously.
    
    Args:
        preference_pair: Preference pair dictionary from evaluation engine
        ablation_data_manager: AblationDataManager instance
        openai_client: AsyncOpenAI client
        args: Command line arguments
        user_profile: User profile (will be loaded if None)
        
    Returns:
        Evaluation result dictionary
    """
    # Get basic information
    user_profile_id = preference_pair['user_profile_id']
    value_profile_id = preference_pair['value_profile_id']
    question_id = preference_pair['question_id']
    preference_id = preference_pair.get('preference_id', f"{user_profile_id}_{question_id}_{preference_pair['style_family']}_{preference_pair['combination_type']}")
    
    prompt = preference_pair['prompt']
    preferred_completion = preference_pair['preferred_completion']
    non_preferred_completion = preference_pair['non_preferred_completion']
    preferred_key = preference_pair['preferred_completion_key']
    non_preferred_key = preference_pair['non_preferred_completion_key']
    
    # Randomly decide whether preferred_completion goes in position A or B
    prefer_in_position_a = random.choice([True, False])
    correct_answer = 'A' if prefer_in_position_a else 'B'
    
    if prefer_in_position_a:
        completion_a = preferred_completion
        completion_b = non_preferred_completion
    else:
        completion_a = non_preferred_completion
        completion_b = preferred_completion
    
    # Get user profile if not provided
    if user_profile is None:
        user_profile = ablation_data_manager.get_user_profile_by_id(user_profile_id)
    
    # Generate contexts
    # ABLATION: Use reduced WVS context with only 4 statements
    ablation_wvs_context = ablation_data_manager.generate_ablation_wvs_context(
        value_profile_id, question_id, num_statements=getattr(args, 'ablation_statements', 4), 
        random_seed=args.random_seed
    )
    style_context = ablation_data_manager.generate_full_style_context(user_profile['style_profile'])
    
    # For combined contexts, randomize the order of WVS and style contexts
    wvs_first = random.choice([True, False])
    
    # ===== PREPARE PROMPTS FOR BATCH PROCESSING =====
    
    prompts_and_settings = []
    
    # Check if this preference pair should be used for each setting type
    use_for_wvs_only = True  # Default to process all settings
    use_for_combined = True
    
    if ablation_conflict_sets is not None:
        # Only process if this pair is in the appropriate conflict set
        use_for_wvs_only = preference_pair in ablation_conflict_sets['wvs_only']
        use_for_combined = preference_pair in ablation_conflict_sets['combined']
    
    # 1. Ablation WVS only (only if this preference pair is in the WVS-only set)
    if use_for_wvs_only:
        ablation_wvs_only_prompt = create_ablation_wvs_only_prompt(prompt, completion_a, completion_b, ablation_wvs_context)
        prompts_and_settings.append((ablation_wvs_only_prompt, False, 'ablation_wvs_only'))
    
    # 2-4. Combined settings (only if this preference pair is in the combined set)
    if use_for_combined:
        # 2. Ablation WVS + Style context (neutral)
        ablation_wvs_style_neutral_prompt = create_ablation_wvs_style_neutral_prompt(
            prompt, completion_a, completion_b, ablation_wvs_context, style_context, wvs_first)
        prompts_and_settings.append((ablation_wvs_style_neutral_prompt, False, 'ablation_wvs_style_neutral'))
        
        # 3. Ablation WVS + Style context (prefer WVS)
        ablation_wvs_style_prefer_wvs_prompt = create_ablation_wvs_style_prefer_wvs_prompt(
            prompt, completion_a, completion_b, ablation_wvs_context, style_context, wvs_first)
        prompts_and_settings.append((ablation_wvs_style_prefer_wvs_prompt, False, 'ablation_wvs_style_prefer_wvs'))
        
        # 4. Ablation WVS + Style context (prefer style)
        ablation_wvs_style_prefer_style_prompt = create_ablation_wvs_style_prefer_style_prompt(
            prompt, completion_a, completion_b, ablation_wvs_context, style_context, wvs_first)
        prompts_and_settings.append((ablation_wvs_style_prefer_style_prompt, False, 'ablation_wvs_style_prefer_style'))
    
    # ===== COT ABLATION EVALUATION SETTINGS (if enabled) =====
    
    if args.all_settings:  # Include COT for all_settings
        # 5. COT Ablation WVS only (only if this preference pair is in the WVS-only set)
        if use_for_wvs_only:
            ablation_wvs_only_cot_prompt = create_ablation_wvs_only_cot_prompt(prompt, completion_a, completion_b, ablation_wvs_context)
            prompts_and_settings.append((ablation_wvs_only_cot_prompt, True, 'ablation_wvs_only_cot'))
        
        # 6-8. COT Combined settings (only if this preference pair is in the combined set)
        if use_for_combined:
            # 6. COT Ablation WVS + Style context (neutral)
            ablation_wvs_style_neutral_cot_prompt = create_ablation_wvs_style_neutral_cot_prompt(
                prompt, completion_a, completion_b, ablation_wvs_context, style_context, wvs_first)
            prompts_and_settings.append((ablation_wvs_style_neutral_cot_prompt, True, 'ablation_wvs_style_neutral_cot'))
            
            # 7. COT Ablation WVS + Style context (prefer WVS)
            ablation_wvs_style_prefer_wvs_cot_prompt = create_ablation_wvs_style_prefer_wvs_cot_prompt(
                prompt, completion_a, completion_b, ablation_wvs_context, style_context, wvs_first)
            prompts_and_settings.append((ablation_wvs_style_prefer_wvs_cot_prompt, True, 'ablation_wvs_style_prefer_wvs_cot'))
            
            # 8. COT Ablation WVS + Style context (prefer style)
            ablation_wvs_style_prefer_style_cot_prompt = create_ablation_wvs_style_prefer_style_cot_prompt(
                prompt, completion_a, completion_b, ablation_wvs_context, style_context, wvs_first)
            prompts_and_settings.append((ablation_wvs_style_prefer_style_cot_prompt, True, 'ablation_wvs_style_prefer_style_cot'))
    
    # ===== EXECUTE BATCH QUERIES =====
    
    responses = await batch_query_openai_models(
        openai_client, prompts_and_settings, args.model_name, 
        args.max_tokens, args.temperature, args.max_retries, 
        args.retry_delay, args.max_workers
    )
    
    # ===== STORE RESULTS =====
    
    result = {
        'preference_id': preference_id,
        'user_profile_id': user_profile_id,
        'value_profile_id': value_profile_id,
        'question_id': question_id,
        'style_family': preference_pair.get('style_family', 'unknown'),
        'combination_type': preference_pair.get('combination_type', 'unknown'),
        'preference_rule': preference_pair.get('preference_rule', 'unknown'),
        'quadrant': preference_pair.get('quadrant', 'unknown'),
        'style_code': preference_pair.get('style_code', 'unknown'),
        'correct_answer': correct_answer,
        'preferred_in_position_a': prefer_in_position_a,
        'wvs_context_first': wvs_first,
        'preferred_completion_key': preferred_key,
        'non_preferred_completion_key': non_preferred_key,
        'evaluation_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'ablation_wvs_context': ablation_wvs_context,  # Store the actual context used
        'ablation_context_length': len(ablation_wvs_context),
        'ablation_statements_count': ablation_wvs_context.count('- ') if ablation_wvs_context else 0
    }
    
    # Add all responses and correctness
    for i, response in enumerate(responses):
        setting_name = prompts_and_settings[i][2]  # Get setting name from prompts_and_settings
        
        if isinstance(response, dict) and 'final_choice' in response:
            final_choice = response['final_choice']
            reasoning = response.get('reasoning', None)
            result[f'{setting_name}_response'] = final_choice
            result[f'{setting_name}_reasoning'] = reasoning
            result[f'{setting_name}_correct'] = final_choice == correct_answer if final_choice != "ERROR" else False
        else:
            result[f'{setting_name}_response'] = "ERROR"
            result[f'{setting_name}_reasoning'] = None
            result[f'{setting_name}_correct'] = False
    
    return result


# ===== RESULT MANAGEMENT =====

def save_incremental_results(output_data, output_file):
    """Save results incrementally to avoid data loss"""
    temp_file = output_file + ".temp"
    try:
        with open(temp_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        # Atomic move
        os.rename(temp_file, output_file)
        print(f"Results saved to {output_file}")
        
    except Exception as e:
        print(f"Error saving results: {e}")
        if os.path.exists(temp_file):
            os.remove(temp_file)

def load_existing_results(output_file):
    """Load existing results if available"""
    if os.path.exists(output_file):
        try:
            with open(output_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading existing results: {e}")
    return None

def calculate_metrics(results):
    """Calculate evaluation metrics from results"""
    if not results:
        return {}
    
    total_count = len(results)
    
    # Determine which settings are present (ablation settings only)
    available_settings = []
    if results:
        sample_result = results[0]
        for key in sample_result.keys():
            if key.endswith('_response') and key.startswith('ablation_'):
                setting_name = key[:-9]  # Remove '_response'
                available_settings.append(setting_name)
    
    # Calculate overall metrics
    overall_metrics = {
        'total_evaluations': total_count,
        'conflict_cases_only': True,
        'ablation_context_avg_length': np.mean([r.get('ablation_context_length', 0) for r in results]),
        'ablation_statements_avg_count': np.mean([r.get('ablation_statements_count', 0) for r in results]),
        'randomization_balance': {
            'preferred_in_position_a': sum(1 for r in results if r.get('preferred_in_position_a', False)),
            'preferred_in_position_b': total_count - sum(1 for r in results if r.get('preferred_in_position_a', False)),
            'balance_ratio': sum(1 for r in results if r.get('preferred_in_position_a', False)) / total_count if total_count > 0 else 0,
            'wvs_context_first': sum(1 for r in results if r.get('wvs_context_first', False)),
            'style_context_first': total_count - sum(1 for r in results if r.get('wvs_context_first', False)),
            'context_order_balance_ratio': sum(1 for r in results if r.get('wvs_context_first', False)) / total_count if total_count > 0 else 0
        }
    }
    
    # Calculate accuracy for each setting
    for setting in available_settings:
        correct_count = sum(1 for r in results if r.get(f'{setting}_correct', False))
        successful_count = sum(1 for r in results if r.get(f'{setting}_response') not in [None, "ERROR"])
        overall_metrics[f'{setting}_accuracy'] = correct_count / successful_count if successful_count > 0 else 0
        overall_metrics[f'{setting}_success_rate'] = successful_count / total_count if total_count > 0 else 0
    
    return overall_metrics


# ===== MAIN FUNCTION =====

async def main():
    # Check if OpenAI is available
    if openai is None or AsyncOpenAI is None:
        print("Error: OpenAI is not installed. Please install it with: pip install openai")
        return
    
    parser = argparse.ArgumentParser(description="OpenAI Async WVS Ablation Evaluation for Reward Model Evaluation v3")
    
    # Model arguments
    parser.add_argument("--model_name", type=str, required=True,
                       help="OpenAI model name (e.g., gpt-4o, gpt-4o-mini, gpt-3.5-turbo)")
    parser.add_argument("--max_tokens", type=int, default=800,
                       help="Maximum tokens for model response (increased for COT)")
    parser.add_argument("--temperature", type=float, default=0.0,
                       help="Sampling temperature")
    parser.add_argument("--max_retries", type=int, default=3,
                       help="Maximum retry attempts for failed requests")
    parser.add_argument("--retry_delay", type=float, default=1.0,
                       help="Delay between retry attempts (seconds)")
    parser.add_argument("--max_workers", type=int, default=10,
                       help="Maximum concurrent requests")
    parser.add_argument("--batch_size", type=int, default=10,
                       help="Number of preference pairs to process in parallel per batch")
    
    # Question group selection (adapted for ablation - only core WVS questions)
    parser.add_argument("--questions", type=str, default="core_wvs_only",
                       choices=['core_wvs_only', 'available_key_wvs'],
                       help="Which questions to use: core_wvs_only (available core WVS only) or available_key_wvs (all available key WVS)")
    
    # Ablation-specific arguments
    parser.add_argument("--ablation_statements", type=int, default=4,
                       help="Number of WVS statements to include in ablated context")
    
    # Evaluation mode arguments
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--all_settings", action="store_true",
                           help="Run all ablation settings (4 standard + 4 COT = 8 settings)")
    mode_group.add_argument("--all_settings_no_cot", action="store_true",
                           help="Run all ablation settings without COT (4 settings)")
    
    # Output arguments
    parser.add_argument("--output_file", type=str, default=None,
                       help="Output file for results (auto-generated if not provided)")
    parser.add_argument("--max_evaluations", type=int, default=None,
                       help="Maximum number of evaluations to run (for testing)")
    parser.add_argument("--random_seed", type=int, default=42,
                       help="Random seed for reproducible randomization")
    parser.add_argument("--save_interval", type=int, default=50,
                       help="Save results every N evaluations")
    
    # Parallel job arguments
    parser.add_argument("--job_index", type=int, default=0,
                       help="Index of this parallel job (0-based)")
    parser.add_argument("--total_jobs", type=int, default=1,
                       help="Total number of parallel jobs")
    
    args = parser.parse_args()
    
    # Validate parallel job arguments
    if args.job_index < 0:
        print(f"Error: job_index must be >= 0, got {args.job_index}")
        return
    if args.total_jobs < 1:
        print(f"Error: total_jobs must be >= 1, got {args.total_jobs}")
        return
    if args.job_index >= args.total_jobs:
        print(f"Error: job_index ({args.job_index}) must be < total_jobs ({args.total_jobs})")
        return
    
    def split_preference_pairs_for_job(preference_pairs, job_index, total_jobs):
        """
        Split preference pairs into chunks for parallel processing.
        Each job gets a roughly equal chunk based on job_index.
        
        Args:
            preference_pairs: List of preference pairs to split
            job_index: 0-based index of current job
            total_jobs: Total number of parallel jobs
            
        Returns:
            Tuple of (preference_pairs_for_job, start_idx, end_idx)
        """
        if total_jobs == 1:
            return preference_pairs, 0, len(preference_pairs)
        
        total_pairs = len(preference_pairs)
        pairs_per_job = total_pairs // total_jobs
        remainder = total_pairs % total_jobs
        
        # Calculate start and end indices for this job
        if job_index < remainder:
            # Jobs 0 to remainder-1 get one extra pair
            start_idx = job_index * (pairs_per_job + 1)
            end_idx = start_idx + pairs_per_job + 1
        else:
            # Remaining jobs get the base amount
            start_idx = remainder * (pairs_per_job + 1) + (job_index - remainder) * pairs_per_job
            end_idx = start_idx + pairs_per_job
        
        return preference_pairs[start_idx:end_idx], start_idx, end_idx
    
    # Set random seed
    if args.random_seed is not None:
        random.seed(args.random_seed)
        np.random.seed(args.random_seed)
        print(f"Random seed set to: {args.random_seed}")
    
    # Generate output filename if not provided
    if not args.output_file:
        model_name = args.model_name.replace('/', '_').replace('-', '_').lower()
        settings_suffix = "all_settings" if args.all_settings else "no_cot"
        args.output_file = f"{model_name}_openai_async_wvs_ablation_conflict_only_{settings_suffix}_results.json"
    
    print("=== OpenAI Async WVS Ablation Evaluation for Value-Style Conflicts ===")
    print(f"Model: {args.model_name}")
    print(f"Questions: {args.questions}")
    print(f"Output: {args.output_file}")
    print(f"Ablation context: {args.ablation_statements} WVS statements")
    print(f"Evaluation focus: Value-style conflict cases only")
    print(f"Max tokens: {args.max_tokens}")
    print(f"Temperature: {args.temperature}")
    print(f"Max workers: {args.max_workers}")
    print(f"Evaluation mode: {'All settings (with COT)' if args.all_settings else 'All settings (no COT)'}")
    if args.total_jobs > 1:
        print(f"Parallel job: {args.job_index + 1}/{args.total_jobs}")
    
    # Initialize data manager and evaluation engine
    print(f"\n=== Loading Data ===")
    ablation_data_manager = AblationDataManager()
    evaluation_engine = EvaluationEngineV3(ablation_data_manager)
    preference_filter = AblationPreferenceFilter()
    
    # Generate preference pairs using ablation-optimized approach
    print(f"\n=== Generating Ablation-Optimized Preference Pairs ===")
    question_ids = ablation_data_manager.available_key_wvs_questions  # Use only available key WVS questions for ablation
    print(f"Using available key WVS questions for ablation: {len(question_ids)} questions")
    
    # Determine ablation mode based on args
    ablation_mode = 'all_settings' if args.all_settings else 'all_settings_no_cot'
    
    # Generate preference pairs optimized for ablation studies
    ablation_preference_sets = evaluation_engine.generate_ablation_preference_pairs(
        question_ids=question_ids,
        ablation_mode=ablation_mode
    )
    
    # Combine all preference pairs for conflict filtering
    # Note: WVS-only and combined pairs will be used for different evaluation settings
    wvs_only_pairs = ablation_preference_sets['wvs_only']
    combined_pairs = ablation_preference_sets['combined']
    all_preference_pairs = wvs_only_pairs + combined_pairs
    print(f"Generated {len(all_preference_pairs)} total preference pairs ({len(wvs_only_pairs)} WVS-only + {len(combined_pairs)} combined)")
    
    # Filter to only value-style conflict cases for each set
    print(f"\n=== Filtering for Value-Style Conflicts ===")
    
    # Filter WVS-only pairs for conflicts (these will be used for ablation_wvs_only settings)
    wvs_only_conflicts = preference_filter.identify_value_style_conflicts(wvs_only_pairs)
    print(f"WVS-only conflict cases: {len(wvs_only_conflicts)}")
    
    # Filter combined pairs for conflicts (these will be used for ablation_wvs_style_* settings)
    combined_conflicts = preference_filter.identify_value_style_conflicts(combined_pairs)
    print(f"Combined conflict cases: {len(combined_conflicts)}")
    
    # Store both sets for different evaluation settings
    ablation_conflict_sets = {
        'wvs_only': wvs_only_conflicts,
        'combined': combined_conflicts
    }
    
    # For backward compatibility, combine all conflicts
    conflict_preference_pairs = wvs_only_conflicts + combined_conflicts
    
    if not conflict_preference_pairs:
        print("No value-style conflict cases found! Check the filtering logic.")
        return
    
    print(f"Identified {len(conflict_preference_pairs)} conflict cases for ablation evaluation")
    
    # Limit evaluations if specified
    if args.max_evaluations and args.max_evaluations < len(conflict_preference_pairs):
        conflict_preference_pairs = conflict_preference_pairs[:args.max_evaluations]
        print(f"Limited to {len(conflict_preference_pairs)} evaluations for testing")
    
    # Check for existing results
    existing_results = load_existing_results(args.output_file)
    if existing_results:
        print(f"Found existing results with {len(existing_results.get('evaluation_results', []))} evaluations")
        completed_ids = {r['preference_id'] for r in existing_results.get('evaluation_results', [])}
        remaining_pairs = [p for p in conflict_preference_pairs if f"{p['user_profile_id']}_{p['question_id']}_{p['style_family']}_{p['combination_type']}" not in completed_ids]
        print(f"Remaining evaluations after filtering completed: {len(remaining_pairs)}")
        conflict_preference_pairs = remaining_pairs
        
        if not conflict_preference_pairs:
            print("All ablation evaluations already completed!")
            return
    
    # Split preference pairs for parallel job processing
    if args.total_jobs > 1:
        total_before_split = len(conflict_preference_pairs)
        conflict_preference_pairs, start_idx, end_idx = split_preference_pairs_for_job(conflict_preference_pairs, args.job_index, args.total_jobs)
        print(f"Job {args.job_index + 1}/{args.total_jobs}: Processing evaluations {start_idx}-{end_idx-1} ({len(conflict_preference_pairs)} items) of {total_before_split} conflict cases")
        
        if not conflict_preference_pairs:
            print(f"No evaluations assigned to job {args.job_index + 1}/{args.total_jobs}")
            return
    else:
        start_idx, end_idx = 0, len(conflict_preference_pairs)
        print(f"Single job: Processing {len(conflict_preference_pairs)} conflict cases (items 0-{len(conflict_preference_pairs)-1})")
    
    # Initialize OpenAI client
    print(f"\n=== Initializing OpenAI Client ===")
    try:
        openai_client = AsyncOpenAI(
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # Test the client with a simple request
        test_response = await openai_client.chat.completions.create(
            model=args.model_name,
            messages=[{"role": "user", "content": "Test"}],
            max_tokens=1
        )
        
        print("OpenAI client initialized successfully")
        print(f"Model: {args.model_name}")
        
    except Exception as e:
        print(f"Error initializing OpenAI client: {e}")
        return
    
    # Initialize results storage
    evaluation_results = existing_results.get('evaluation_results', []) if existing_results else []
    
    # Track this job's initial contribution to avoid double-counting progress
    initial_results_count = len(evaluation_results)
    job_target_count = len(conflict_preference_pairs)
    
    # Create output data structure
    evaluation_settings = [
        'ablation_wvs_only',
        'ablation_wvs_style_neutral',
        'ablation_wvs_style_prefer_wvs',
        'ablation_wvs_style_prefer_style'
    ]
    
    if args.all_settings:  # Include COT settings
        evaluation_settings.extend([
            'ablation_wvs_only_cot',
            'ablation_wvs_style_neutral_cot',
            'ablation_wvs_style_prefer_wvs_cot',
            'ablation_wvs_style_prefer_style_cot'
        ])
    
    output_data = {
        'metadata': {
            'model_name': args.model_name,
            'evaluation_type': 'openai_async_wvs_ablation_conflict_only',
            'ablation_statements_count': args.ablation_statements,
            'total_conflict_cases': len(conflict_preference_pairs) + len(evaluation_results),
            'max_tokens': args.max_tokens,
            'temperature': args.temperature,
            'random_seed': args.random_seed,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'status': 'in_progress',
            'evaluation_settings': evaluation_settings,
            'includes_cot': args.all_settings,
            'parallel_job_info': {
                'job_index': args.job_index,
                'total_jobs': args.total_jobs,
                'job_preference_count': len(conflict_preference_pairs)
            } if args.total_jobs > 1 else None
        },
        'evaluation_results': evaluation_results,
        'metrics': {}
    }
    
    # Calculate estimated time
    print(f"\n=== Evaluation Time Estimation ===")
    total_evaluations = len(conflict_preference_pairs)
    print(f"Total conflict evaluations: {total_evaluations:,}")
    
    settings_per_eval = len(evaluation_settings)
    total_api_calls = total_evaluations * settings_per_eval
    concurrent_calls = args.max_workers
    
    print(f"Settings per evaluation: {settings_per_eval}")
    print(f"Total API calls needed: {total_api_calls:,}")
    print(f"Concurrent API calls: {concurrent_calls}")
    print(f"Batches needed: ~{total_api_calls // concurrent_calls}")
    
    # Run evaluations
    print(f"\n=== Running Ablation Evaluations ===")
    
    try:
        start_time = time.time()
        
        # Process evaluations
        for i, preference_pair in enumerate(conflict_preference_pairs):
            try:
                # Evaluate this preference pair with all settings asynchronously
                result = await evaluate_preference_pair_all_settings_async(
                    preference_pair, ablation_data_manager, openai_client, args, ablation_conflict_sets
                )
                
                evaluation_results.append(result)
                
                # Calculate performance stats
                current_time = time.time()
                elapsed_time = current_time - start_time
                
                # Track THIS JOB's progress only (not the shared file total)
                job_completed = len(evaluation_results) - initial_results_count
                job_total = job_target_count
                
                if elapsed_time > 0:
                    eval_rate = job_completed / elapsed_time
                    
                    # Calculate recent accuracy (for ablation_wvs_only setting if available)
                    if evaluation_results:
                        recent_results = evaluation_results[-min(10, len(evaluation_results)):]  # Last 10 results
                        wvs_only_correct = sum(1 for r in recent_results if r.get('ablation_wvs_only_correct', False))
                        recent_accuracy = wvs_only_correct / len(recent_results) if recent_results else 0
                        
                        actual_item_idx = start_idx + i if args.total_jobs > 1 else i
                        print(f"Completed item {i+1}/{len(conflict_preference_pairs)} (global: {actual_item_idx}). "
                              f"Job: {job_completed}/{job_total} "
                              f"Rate: {eval_rate:.1f}/s "
                              f"Acc: {recent_accuracy:.1%}")
                
                # Save results periodically
                if (i + 1) % args.save_interval == 0:
                    output_data['evaluation_results'] = evaluation_results
                    output_data['metrics'] = calculate_metrics(evaluation_results)
                    save_incremental_results(output_data, args.output_file)
                
            except Exception as e:
                print(f"Error evaluating preference pair {preference_pair.get('preference_id', 'unknown')}: {e}")
                print(f"Traceback: {traceback.format_exc()}")
                continue
    
    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user")
    except Exception as e:
        print(f"Unexpected error during evaluation: {e}")
        print(f"Traceback: {traceback.format_exc()}")
    
    # Final save and metrics calculation
    print(f"\n=== Finalizing Results ===")
    output_data['evaluation_results'] = evaluation_results
    output_data['metrics'] = calculate_metrics(evaluation_results)
    output_data['metadata']['status'] = 'completed'
    output_data['metadata']['completed_evaluations'] = len(evaluation_results)
    
    save_incremental_results(output_data, args.output_file)
    
    # Print summary
    metrics = output_data['metrics']
    
    print(f"\n=== OPENAI ASYNC ABLATION EVALUATION SUMMARY ===")
    print(f"Evaluation type: OpenAI Async WVS Ablation (Value-Style Conflicts Only)")
    print(f"Total conflict cases evaluated: {metrics['total_evaluations']}")
    print(f"Average ablation context length: {metrics['ablation_context_avg_length']:.0f} chars")
    print(f"Average ablation statements: {metrics['ablation_statements_avg_count']:.1f}")
    
    # Print accuracy by setting
    print(f"\n=== ACCURACY BY ABLATION SETTING ===")
    for key, value in metrics.items():
        if key.startswith('ablation_') and key.endswith('_accuracy'):
            setting_name = key[:-9]  # Remove '_accuracy'
            success_rate = metrics.get(f'{setting_name}_success_rate', 1.0)
            print(f"  {setting_name}: {value:.1%} accuracy ({success_rate:.1%} success rate)")
    
    # Display randomization balance
    randomization = metrics.get('randomization_balance', {})
    print(f"\n=== RANDOMIZATION BALANCE ===")
    print(f"Preferred completion in position A: {randomization.get('preferred_in_position_a', 0)} ({randomization.get('balance_ratio', 0):.1%})")
    print(f"WVS context first: {randomization.get('wvs_context_first', 0)} ({randomization.get('context_order_balance_ratio', 0):.1%})")
    
    print("\nOpenAI async ablation evaluation completed!")
    print(f"Results saved to: {args.output_file}")


if __name__ == "__main__":
    asyncio.run(main())
