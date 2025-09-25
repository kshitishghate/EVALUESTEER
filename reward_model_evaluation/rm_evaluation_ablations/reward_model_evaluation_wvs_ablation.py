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

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from data_management.data_manager_v3 import DataManagerV3
from evaluation_engine.evaluation_engine_v3 import EvaluationEngineV3

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
        """Structured response schema for evaluations"""
        reasoning: Optional[str] = Field(default=None, description="Step-by-step analysis (if COT)")
        final_choice: Literal["A", "B"] = Field(description="The final answer: either A or B")
        confidence: Optional[float] = Field(default=None, ge=0, le=1, description="Optional confidence score")


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
    
    def get_question_to_wvs_mapping(self) -> Dict[str, str]:
        """
        Create a mapping from question IDs to their corresponding WVS question IDs.
        This is used to identify the "relevant" WVS statement for each evaluation question.
        
        Returns:
            Dictionary mapping question_id -> wvs_question_id
        """
        # Load synthetic data to examine the mapping
        synthetic_data = self.load_synthetic_data()
        
        mapping = {}
        for item in synthetic_data:
            question_id = item.get('question_id')
            wvs_question = item.get('wvs_question', '')
            
            # Extract WVS question ID from wvs_question field
            # Look for patterns like "Q164" in the wvs_question text
            import re
            wvs_match = re.search(r'Q(\d+)', wvs_question)
            if wvs_match:
                wvs_q_id = f"Q{wvs_match.group(1)}"
                mapping[question_id] = wvs_q_id
            else:
                # Fallback: assume question_id maps directly to itself for WVS questions
                if question_id in self.key_wvs_questions:
                    mapping[question_id] = question_id
        
        return mapping


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
        # 1. Replace unescaped quotes within strings (this is a heuristic approach)
        # Look for patterns like: "text with "quotes" inside"
        # This is complex, so we'll try a simpler approach first
        
        # 2. Ensure proper newline handling within JSON strings
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

def wrap_as_chat(prompt: str, tok: AutoTokenizer) -> str:
    """Wrap prompt in chat template"""
    system_message = "You are an AI assistant that evaluates which answer (A or B) a user would prefer. Always respond with valid JSON."
    
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": prompt},
    ]
    return tok.apply_chat_template(messages,
                                tokenize=False,
                                add_generation_prompt=True)

def query_vllm_model(llm_model, sampling_params, prompt, args, max_retries=3, is_cot=False):
    """
    Query the vLLM model with retry logic.
    
    Args:
        llm_model: The vLLM model instance
        sampling_params: Sampling parameters for generation
        prompt: The input prompt
        args: Arguments namespace
        max_retries: Maximum number of retry attempts
        is_cot: Whether this is a COT query (includes reasoning field)
    
    Returns:
        dict: Response dictionary with final_choice and reasoning (COT) or final_choice only (non-COT)
    """
    for attempt in range(max_retries):
        try:
            # For COT queries, try to use guided generation if available
            current_sampling_params = sampling_params
            if is_cot:
                try:
                    # Try to enable JSON schema guidance for COT responses
                    # Note: This may not be available in all vLLM versions
                    json_schema = COTResponse.model_json_schema() if BaseModel is not None else None
                    if json_schema and hasattr(sampling_params, 'guided_json'):
                        current_sampling_params = SamplingParams(
                            temperature=sampling_params.temperature,
                            top_p=sampling_params.top_p,
                            max_tokens=sampling_params.max_tokens,
                            guided_json=json_schema
                        )
                        print("Using guided JSON generation for COT")
                except Exception as e:
                    print(f"Note: Guided JSON generation not available: {e}")
                    # Fall back to regular sampling
                    pass
            
            # Generate response using vLLM
            outputs = llm_model.generate([prompt], current_sampling_params)
            
            if outputs and len(outputs) > 0 and outputs[0].outputs:
                result = outputs[0].outputs[0].text.strip()
                
                print(f"Attempt {attempt + 1}: Raw response: '{result[:100]}...'")
                
                # If empty response, retry
                if not result:
                    print(f"Empty response on attempt {attempt + 1}, retrying...")
                    continue
                
                # Both COT and non-COT now use JSON parsing
                parsed_result = parse_json_response(result)
                if parsed_result["final_choice"] != "ERROR":
                    return parsed_result
                else:
                    print(f"JSON parsing failed (attempt {attempt + 1}), retrying...")
                    continue
            else:
                print(f"Empty or invalid response structure (attempt {attempt + 1})")
                continue
                
        except Exception as e:
            print(f"Error querying model (attempt {attempt + 1}): {e}")
            if attempt < max_retries - 1:
                time.sleep(1)
            continue
    
    print(f"All {max_retries} attempts failed, returning ERROR")
    # Return consistent format
    return {"final_choice": "ERROR", "reasoning": None}

# ===== BATCHED EVALUATION FUNCTIONS FOR ABLATION =====

def get_preference_pairs_for_setting(setting_name, ablation_conflict_sets, fallback_pairs):
    """
    Get the appropriate preference pairs for a specific ablation setting.
    
    Args:
        setting_name: Name of the ablation setting
        ablation_conflict_sets: Dict with 'wvs_only' and 'combined' preference pairs
        fallback_pairs: Fallback preference pairs if ablation_conflict_sets is None
        
    Returns:
        List of preference pairs appropriate for this setting
    """
    if ablation_conflict_sets is None:
        return fallback_pairs
    
    # Route settings to appropriate preference pair sets
    if setting_name in ['ablation_wvs_only', 'ablation_wvs_only_cot']:
        return ablation_conflict_sets['wvs_only']
    elif setting_name in ['ablation_wvs_style_neutral', 'ablation_wvs_style_prefer_wvs', 
                          'ablation_wvs_style_prefer_style', 'ablation_wvs_style_neutral_cot',
                          'ablation_wvs_style_prefer_wvs_cot', 'ablation_wvs_style_prefer_style_cot']:
        return ablation_conflict_sets['combined']
    else:
        # Default: use combined pairs for unknown settings
        return ablation_conflict_sets.get('combined', fallback_pairs)


def collect_prompts_from_preference_pairs_ablation(preference_pairs, ablation_data_manager, args, user_profile_cache, 
                                                  ablation_conflict_sets=None, tokenizer=None):
    """
    Collect all prompts from a batch of preference pairs for batched vLLM inference (ABLATION VERSION).
    
    This function intelligently routes different preference pair sets to different ablation settings:
    - WVS-only settings use 18-profile preference pairs
    - Combined settings use 288-profile preference pairs
    
    Args:
        preference_pairs: List of preference pair dictionaries (used as fallback)
        ablation_data_manager: AblationDataManager instance
        args: Command line arguments
        user_profile_cache: Cache for user profiles
        ablation_conflict_sets: Dict with 'wvs_only' and 'combined' preference pair sets
        tokenizer: Pre-loaded tokenizer (optional)
    
    Returns:
        prompts: List of formatted prompts ready for vLLM
        prompt_metadata: List of metadata for mapping responses back to preference pairs
    """
    prompts = []
    prompt_metadata = []
    
    # Use provided tokenizer or try to load one
    tok = tokenizer
    if tok is None and AutoTokenizer is not None:
        try:
            tok = AutoTokenizer.from_pretrained(args.model_path)
        except:
            print("Warning: Could not load tokenizer, using raw prompts")
    
    # Define ablation settings and their required preference pair sets
    ablation_settings = [
        ('ablation_wvs_only', False),
        ('ablation_wvs_style_neutral', False),
        ('ablation_wvs_style_prefer_wvs', False),
        ('ablation_wvs_style_prefer_style', False)
    ]
    
    if args.all_settings:  # Add COT versions
        ablation_settings.extend([
            ('ablation_wvs_only_cot', True),
            ('ablation_wvs_style_neutral_cot', True),
            ('ablation_wvs_style_prefer_wvs_cot', True),
            ('ablation_wvs_style_prefer_style_cot', True)
        ])
    
    # Generate prompts for each setting using the appropriate preference pairs
    for setting_name, is_cot in ablation_settings:
        # Get the appropriate preference pairs for this setting
        setting_pairs = get_preference_pairs_for_setting(setting_name, ablation_conflict_sets, preference_pairs)
        
        print(f"Generating prompts for {setting_name}: {len(setting_pairs)} preference pairs")
        
        for preference_pair in setting_pairs:
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
            
            # Get user profile from cache
            if user_profile_id not in user_profile_cache:
                user_profile_cache[user_profile_id] = ablation_data_manager.get_user_profile_by_id(user_profile_id)
            user_profile = user_profile_cache[user_profile_id]
            
            # Generate contexts
            # ABLATION: Use reduced WVS context with only 4 statements
            ablation_wvs_context = ablation_data_manager.generate_ablation_wvs_context(
                value_profile_id, question_id, num_statements=getattr(args, 'ablation_statements', 4), 
                random_seed=args.random_seed
            )
            style_context = ablation_data_manager.generate_full_style_context(user_profile['style_profile'])
            
            # For combined contexts, randomize the order of WVS and style contexts
            wvs_first = random.choice([True, False])
            
            # Store metadata for this prompt
            base_metadata = {
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
                'setting': setting_name,
                'is_cot': is_cot
            }
            
            # Generate the appropriate prompt for this setting
            if setting_name == 'ablation_wvs_only':
                prompt_text = create_ablation_wvs_only_prompt(prompt, completion_a, completion_b, ablation_wvs_context)
            elif setting_name == 'ablation_wvs_style_neutral':
                prompt_text = create_ablation_wvs_style_neutral_prompt(prompt, completion_a, completion_b, ablation_wvs_context, style_context, wvs_first)
            elif setting_name == 'ablation_wvs_style_prefer_wvs':
                prompt_text = create_ablation_wvs_style_prefer_wvs_prompt(prompt, completion_a, completion_b, ablation_wvs_context, style_context, wvs_first)
            elif setting_name == 'ablation_wvs_style_prefer_style':
                prompt_text = create_ablation_wvs_style_prefer_style_prompt(prompt, completion_a, completion_b, ablation_wvs_context, style_context, wvs_first)
            elif setting_name == 'ablation_wvs_only_cot':
                prompt_text = create_ablation_wvs_only_cot_prompt(prompt, completion_a, completion_b, ablation_wvs_context)
            elif setting_name == 'ablation_wvs_style_neutral_cot':
                prompt_text = create_ablation_wvs_style_neutral_cot_prompt(prompt, completion_a, completion_b, ablation_wvs_context, style_context, wvs_first)
            elif setting_name == 'ablation_wvs_style_prefer_wvs_cot':
                prompt_text = create_ablation_wvs_style_prefer_wvs_cot_prompt(prompt, completion_a, completion_b, ablation_wvs_context, style_context, wvs_first)
            elif setting_name == 'ablation_wvs_style_prefer_style_cot':
                prompt_text = create_ablation_wvs_style_prefer_style_cot_prompt(prompt, completion_a, completion_b, ablation_wvs_context, style_context, wvs_first)
            else:
                # Default fallback
                prompt_text = create_ablation_wvs_only_prompt(prompt, completion_a, completion_b, ablation_wvs_context)
            
            # Apply chat formatting if tokenizer available
            if tok:
                prompt_text = wrap_as_chat(prompt_text, tok)
            
            prompts.append(prompt_text)
            prompt_metadata.append(base_metadata)
    
    return prompts, prompt_metadata


def query_vllm_model_batch(llm_model, sampling_params, prompts, prompt_metadata, args, max_retries=3):
    """
    Query the vLLM model with a batch of prompts with retry logic.
    
    Args:
        llm_model: The vLLM model instance
        sampling_params: Sampling parameters for generation
        prompts: List of formatted prompts ready for vLLM
        prompt_metadata: List of metadata for mapping responses back
        args: Arguments namespace
        max_retries: Maximum number of retry attempts
    
    Returns:
        List of response dictionaries with final_choice and reasoning
    """
    for attempt in range(max_retries):
        try:
            print(f"Sending batch of {len(prompts)} prompts to vLLM (attempt {attempt + 1})")
            
            # Generate responses using vLLM batch inference
            outputs = llm_model.generate(prompts, sampling_params)
            
            if not outputs or len(outputs) != len(prompts):
                print(f"Invalid batch response structure (attempt {attempt + 1})")
                continue
            
            # Parse all responses
            parsed_responses = []
            failed_count = 0
            
            for i, output in enumerate(outputs):
                if output.outputs and len(output.outputs) > 0:
                    result = output.outputs[0].text.strip()
                    metadata = prompt_metadata[i]
                    
                    if not result:
                        print(f"Empty response for prompt {i} (attempt {attempt + 1})")
                        failed_count += 1
                        parsed_responses.append({"final_choice": "ERROR", "reasoning": None})
                        continue
                    
                    # Parse JSON response (both COT and non-COT use JSON now)
                    parsed_result = parse_json_response(result)
                    parsed_responses.append(parsed_result)
                    
                    if parsed_result["final_choice"] == "ERROR":
                        failed_count += 1
                else:
                    print(f"Empty output structure for prompt {i} (attempt {attempt + 1})")
                    failed_count += 1
                    parsed_responses.append({"final_choice": "ERROR", "reasoning": None})
            
            # If most responses parsed successfully, return the batch
            success_rate = (len(prompts) - failed_count) / len(prompts)
            if success_rate >= 0.8:  # At least 80% success rate
                print(f"Batch completed with {success_rate:.1%} success rate ({failed_count} failures)")
                return parsed_responses
            else:
                print(f"Batch failed with {success_rate:.1%} success rate (attempt {attempt + 1})")
                if attempt < max_retries - 1:
                    continue
                else:
                    print(f"Returning batch with failures after {max_retries} attempts")
                    return parsed_responses
                
        except Exception as e:
            print(f"Error in batch inference (attempt {attempt + 1}): {e}")
            if attempt < max_retries - 1:
                time.sleep(1)
            continue
    
    print(f"All {max_retries} batch attempts failed, returning errors")
    return [{"final_choice": "ERROR", "reasoning": None} for _ in prompts]


def evaluate_preference_pairs_batch_optimized_ablation(preference_pairs, ablation_data_manager, llm_model, sampling_params, args, 
                                                     ablation_conflict_sets=None, tokenizer=None):
    """
    Evaluate a batch of preference pairs using optimized batched vLLM inference (ABLATION VERSION).
    
    Args:
        preference_pairs: List of preference pair dictionaries
        ablation_data_manager: AblationDataManager instance  
        llm_model: vLLM model instance  
        sampling_params: Sampling parameters
        args: Command line arguments
        tokenizer: Pre-loaded tokenizer (optional, for efficiency)
        
    Returns:
        List of evaluation result dictionaries
    """
    user_profile_cache = {}
    
    # Collect all prompts from the batch
    print(f"Collecting prompts from {len(preference_pairs)} preference pairs...")
    prompts, prompt_metadata = collect_prompts_from_preference_pairs_ablation(
        preference_pairs, ablation_data_manager, args, user_profile_cache, ablation_conflict_sets, tokenizer)
    
    print(f"Generated {len(prompts)} total prompts for batch inference")
    
    # Query vLLM with all prompts at once
    responses = query_vllm_model_batch(llm_model, sampling_params, prompts, prompt_metadata, args)
    
    # Group responses back by preference pair using multiple fields approach (consistent with other scripts)
    results_by_preference = {}
    
    for response, metadata in zip(responses, prompt_metadata):
        preference_id = metadata['preference_id']
        setting = metadata['setting']
        
        # Store the base metadata on first encounter for this preference pair
        if preference_id not in results_by_preference:
            # Get ablation context info for this preference pair
            preference_pair = next((p for p in preference_pairs if p.get('user_profile_id') == metadata['user_profile_id'] and p.get('question_id') == metadata['question_id']), None)
            ablation_wvs_context = ""
            ablation_context_length = 0
            ablation_statements_count = 0
            
            if preference_pair:
                try:
                    ablation_wvs_context = ablation_data_manager.generate_ablation_wvs_context(
                        metadata['value_profile_id'], metadata['question_id'], 
                        num_statements=getattr(args, 'ablation_statements', 4), 
                        random_seed=args.random_seed
                    )
                    ablation_context_length = len(ablation_wvs_context)
                    ablation_statements_count = ablation_wvs_context.count('- ') if ablation_wvs_context else 0
                except:
                    pass  # Use defaults if context generation fails
            
            results_by_preference[preference_id] = {
                'preference_id': metadata['preference_id'],
                'user_profile_id': metadata['user_profile_id'],
                'value_profile_id': metadata['value_profile_id'],
                'question_id': metadata['question_id'],
                'style_family': metadata['style_family'],
                'combination_type': metadata['combination_type'],
                'preference_rule': metadata['preference_rule'],
                'quadrant': metadata['quadrant'],
                'style_code': metadata['style_code'],
                'correct_answer': metadata['correct_answer'],
                'preferred_in_position_a': metadata['preferred_in_position_a'],
                'wvs_context_first': metadata['wvs_context_first'],
                'preferred_completion_key': metadata['preferred_completion_key'],
                'non_preferred_completion_key': metadata['non_preferred_completion_key'],
                'evaluation_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'ablation_wvs_context': ablation_wvs_context,  # Store the actual context used
                'ablation_context_length': ablation_context_length,
                'ablation_statements_count': ablation_statements_count
            }
        
        # Add response data for this setting as separate fields
        final_choice = response['final_choice']
        reasoning = response.get('reasoning', None)
        correct_answer = metadata['correct_answer']
        
        results_by_preference[preference_id][f'{setting}_response'] = final_choice
        results_by_preference[preference_id][f'{setting}_reasoning'] = reasoning
        results_by_preference[preference_id][f'{setting}_correct'] = final_choice == correct_answer if final_choice != "ERROR" else False
    
    # Convert back to list format
    evaluation_results = list(results_by_preference.values())
    
    print(f"Successfully processed {len(evaluation_results)} preference pairs with multiple setting fields")
    return evaluation_results

# ===== MAIN EVALUATION FUNCTION =====

def evaluate_preference_pair_ablation_settings(preference_pair, ablation_data_manager, llm_model, sampling_params, args, user_profile=None):
    """
    Evaluate a single preference pair using ablation settings.
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
        value_profile_id, question_id, num_statements=4, random_seed=args.random_seed
    )
    style_context = ablation_data_manager.generate_full_style_context(user_profile['style_profile'])
    
    # Get tokenizer for chat formatting
    tok = None
    if AutoTokenizer is not None:
        try:
            tok = AutoTokenizer.from_pretrained(args.model_path)
        except:
            print("Warning: Could not load tokenizer, using raw prompts")
    
    # All evaluation responses
    responses = {}
    
    # For combined contexts, randomize the order of WVS and style contexts
    wvs_first = random.choice([True, False])
    
    # ===== ABLATION EVALUATION SETTINGS =====
    
    # 1. Ablation WVS only
    ablation_wvs_only_prompt = create_ablation_wvs_only_prompt(prompt, completion_a, completion_b, ablation_wvs_context)
    if tok:
        ablation_wvs_only_prompt = wrap_as_chat(ablation_wvs_only_prompt, tok)
    responses['ablation_wvs_only'] = query_vllm_model(llm_model, sampling_params, ablation_wvs_only_prompt, args)
    
    # 2. Ablation WVS + Style context (neutral)
    ablation_wvs_style_neutral_prompt = create_ablation_wvs_style_neutral_prompt(
        prompt, completion_a, completion_b, ablation_wvs_context, style_context, wvs_first)
    if tok:
        ablation_wvs_style_neutral_prompt = wrap_as_chat(ablation_wvs_style_neutral_prompt, tok)
    responses['ablation_wvs_style_neutral'] = query_vllm_model(llm_model, sampling_params, ablation_wvs_style_neutral_prompt, args)
    
    # 3. Ablation WVS + Style context (prefer WVS)
    ablation_wvs_style_prefer_wvs_prompt = create_ablation_wvs_style_prefer_wvs_prompt(
        prompt, completion_a, completion_b, ablation_wvs_context, style_context, wvs_first)
    if tok:
        ablation_wvs_style_prefer_wvs_prompt = wrap_as_chat(ablation_wvs_style_prefer_wvs_prompt, tok)
    responses['ablation_wvs_style_prefer_wvs'] = query_vllm_model(llm_model, sampling_params, ablation_wvs_style_prefer_wvs_prompt, args)
    
    # 4. Ablation WVS + Style context (prefer style)
    ablation_wvs_style_prefer_style_prompt = create_ablation_wvs_style_prefer_style_prompt(
        prompt, completion_a, completion_b, ablation_wvs_context, style_context, wvs_first)
    if tok:
        ablation_wvs_style_prefer_style_prompt = wrap_as_chat(ablation_wvs_style_prefer_style_prompt, tok)
    responses['ablation_wvs_style_prefer_style'] = query_vllm_model(llm_model, sampling_params, ablation_wvs_style_prefer_style_prompt, args)
    
    # ===== COT ABLATION EVALUATION SETTINGS (if enabled) =====
    
    if args.all_settings:  # Include COT for all_settings
        # 5. COT Ablation WVS only
        ablation_wvs_only_cot_prompt = create_ablation_wvs_only_cot_prompt(prompt, completion_a, completion_b, ablation_wvs_context)
        if tok:
            ablation_wvs_only_cot_prompt = wrap_as_chat(ablation_wvs_only_cot_prompt, tok)
        responses['ablation_wvs_only_cot'] = query_vllm_model(llm_model, sampling_params, ablation_wvs_only_cot_prompt, args, is_cot=True)
    
        # 6. COT Ablation WVS + Style context (neutral)
        ablation_wvs_style_neutral_cot_prompt = create_ablation_wvs_style_neutral_cot_prompt(
            prompt, completion_a, completion_b, ablation_wvs_context, style_context, wvs_first)
        if tok:
            ablation_wvs_style_neutral_cot_prompt = wrap_as_chat(ablation_wvs_style_neutral_cot_prompt, tok)
        responses['ablation_wvs_style_neutral_cot'] = query_vllm_model(llm_model, sampling_params, ablation_wvs_style_neutral_cot_prompt, args, is_cot=True)
        
        # 7. COT Ablation WVS + Style context (prefer WVS)
        ablation_wvs_style_prefer_wvs_cot_prompt = create_ablation_wvs_style_prefer_wvs_cot_prompt(
            prompt, completion_a, completion_b, ablation_wvs_context, style_context, wvs_first)
        if tok:
            ablation_wvs_style_prefer_wvs_cot_prompt = wrap_as_chat(ablation_wvs_style_prefer_wvs_cot_prompt, tok)
        responses['ablation_wvs_style_prefer_wvs_cot'] = query_vllm_model(llm_model, sampling_params, ablation_wvs_style_prefer_wvs_cot_prompt, args, is_cot=True)
        
        # 8. COT Ablation WVS + Style context (prefer style)
        ablation_wvs_style_prefer_style_cot_prompt = create_ablation_wvs_style_prefer_style_cot_prompt(
            prompt, completion_a, completion_b, ablation_wvs_context, style_context, wvs_first)
        if tok:
            ablation_wvs_style_prefer_style_cot_prompt = wrap_as_chat(ablation_wvs_style_prefer_style_cot_prompt, tok)
        responses['ablation_wvs_style_prefer_style_cot'] = query_vllm_model(llm_model, sampling_params, ablation_wvs_style_prefer_style_cot_prompt, args, is_cot=True)
    
    # Store the evaluation result with all responses
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
    for setting_name, response in responses.items():
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
    """Save results incrementally with file locking to prevent race conditions"""
    import fcntl
    import random
    import time
    
    max_retries = 5
    for attempt in range(max_retries):
        lock_file = output_file + ".lock"
        temp_suffix = f".temp.{os.getpid()}.{random.randint(1000,9999)}"
        temp_file = output_file + temp_suffix
        
        try:
            # Ensure output directory exists
            output_dir = os.path.dirname(output_file)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
            
            # Acquire file lock with timeout
            with open(lock_file, 'w') as lock_f:
                try:
                    # Try to acquire exclusive lock with timeout
                    fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    
                    # Re-read the current file to get latest results from other jobs
                    current_results = []
                    if os.path.exists(output_file):
                        try:
                            with open(output_file, 'r') as f:
                                current_data = json.load(f)
                                current_results = current_data.get('evaluation_results', [])
                        except:
                            pass  # File might be corrupted, proceed with our data
                    
                    # Merge our results with current results, avoiding duplicates
                    our_results = output_data.get('evaluation_results', [])
                    existing_ids = {r.get('preference_id') for r in current_results}
                    
                    # Only add our results that aren't already present
                    new_results = [r for r in our_results if r.get('preference_id') not in existing_ids]
                    
                    if new_results:
                        # Merge results
                        merged_results = current_results + new_results
                        output_data['evaluation_results'] = merged_results
                        output_data['metadata']['completed_evaluations'] = len(merged_results)
                        
                        print(f"Merging {len(new_results)} new results with {len(current_results)} existing (total: {len(merged_results)})")
                    else:
                        print(f"No new results to add (all {len(our_results)} already exist)")
                        return
                    
                    # Write merged data to temp file
                    with open(temp_file, 'w') as f:
                        json.dump(output_data, f, indent=2)
                    
                    # Atomic move
                    os.rename(temp_file, output_file)
                    print(f"Results saved to {output_file} ({len(merged_results)} total)")
                    
                    # Release lock
                    fcntl.flock(lock_f.fileno(), fcntl.LOCK_UN)
                    return
                    
                except BlockingIOError:
                    # Lock is held by another process, wait and retry
                    wait_time = random.uniform(0.1, 0.5) * (attempt + 1)
                    print(f"File locked, waiting {wait_time:.1f}s (attempt {attempt+1}/{max_retries})")
                    time.sleep(wait_time)
                    continue
                    
        except Exception as e:
            print(f"Error saving results (attempt {attempt+1}/{max_retries}): {e}")
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except:
                    pass
            if attempt < max_retries - 1:
                time.sleep(random.uniform(0.5, 1.0))
            continue
        finally:
            # Clean up lock file
            try:
                if os.path.exists(lock_file):
                    os.remove(lock_file)
            except:
                pass
    
    print(f"Failed to save results after {max_retries} attempts")

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
    
    # Determine which settings are present
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

def main():
    # Check if vLLM is available
    if LLM is None or SamplingParams is None:
        print("Error: vLLM is not installed. Please install it with: pip install vllm")
        return
    
    parser = argparse.ArgumentParser(description="WVS Ablation Evaluation for Reward Model Evaluation v3")
    
    # Model arguments
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to the model (HuggingFace model path or local path)")
    parser.add_argument("--max_tokens", type=int, default=800,
                       help="Maximum tokens for model response (increased for COT)")
    parser.add_argument("--temperature", type=float, default=0.0,
                       help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=1.0,
                       help="Top-p sampling parameter")
    parser.add_argument("--tensor_parallel_size", type=int, default=1,
                       help="Number of GPUs for tensor parallelism")
    parser.add_argument("--max_model_len", type=int, default=None,
                       help="Maximum model sequence length")
    parser.add_argument("--batch_size", type=int, default=64,
                       help="Number of preference pairs to process per batch")
    
    # vLLM Batching arguments (from main script)
    parser.add_argument("--enable_vllm_batching", action="store_true", default=True,
                       help="Enable optimized vLLM batching for maximum throughput (default: True)")
    parser.add_argument("--disable_vllm_batching", dest="enable_vllm_batching", action="store_false",
                       help="Disable vLLM batching and use individual calls (for debugging)")
    
    # Evaluation mode arguments
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--all_settings", action="store_true",
                           help="Run all ablation settings (4 standard + 4 COT = 8 settings)")
    mode_group.add_argument("--all_settings_no_cot", action="store_true",
                           help="Run all ablation settings without COT (4 settings)")
    
    # Question group selection (adapted for ablation - only core WVS questions)
    parser.add_argument("--questions", type=str, default="core_wvs_only",
                       choices=['core_wvs_only', 'available_key_wvs'],
                       help="Which questions to use: core_wvs_only (available core WVS only) or available_key_wvs (all available key WVS)")
    
    # Ablation-specific arguments
    parser.add_argument("--ablation_statements", type=int, default=4,
                       help="Number of WVS statements to include in ablated context")
    
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
        model_name = args.model_path.split('/')[-1].lower().replace('-', '_')
        args.output_file = f"{model_name}_wvs_ablation_conflict_only_results.json"
    
    print("=== WVS Ablation Evaluation for Value-Style Conflicts ===")
    print(f"Model: {args.model_path}")
    print(f"Questions: {args.questions}")
    print(f"Output: {args.output_file}")
    print(f"Ablation context: {args.ablation_statements} WVS statements")
    print(f"Evaluation focus: Value-style conflict cases only")
    print(f"Max tokens: {args.max_tokens}")
    print(f"Temperature: {args.temperature}")
    print(f"Tensor parallel size: {args.tensor_parallel_size}")
    print(f"Batch size: {args.batch_size}")
    print(f"vLLM batching: {'ENABLED' if args.enable_vllm_batching else 'DISABLED'}")
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
        print(f"DEBUG: Job {args.job_index}/{args.total_jobs} - Before split: {total_before_split} pairs")
        conflict_preference_pairs, start_idx, end_idx = split_preference_pairs_for_job(conflict_preference_pairs, args.job_index, args.total_jobs)
        print(f"DEBUG: Job {args.job_index}/{args.total_jobs} - After split: {len(conflict_preference_pairs)} pairs")
        print(f"Job {args.job_index + 1}/{args.total_jobs}: Processing evaluations {start_idx}-{end_idx-1} ({len(conflict_preference_pairs)} items) of {total_before_split} conflict cases")
        
        if not conflict_preference_pairs:
            print(f"No evaluations assigned to job {args.job_index + 1}/{args.total_jobs}")
            return
    else:
        start_idx, end_idx = 0, len(conflict_preference_pairs)
        print(f"Single job: Processing {len(conflict_preference_pairs)} conflict cases (items 0-{len(conflict_preference_pairs)-1})")
    
    # Initialize vLLM model
    print(f"\n=== Initializing Model ===")
    try:
        llm_kwargs = {
            "model": args.model_path,
            "tensor_parallel_size": args.tensor_parallel_size,
            "trust_remote_code": True
        }
        
        if args.max_model_len is not None:
            llm_kwargs["max_model_len"] = args.max_model_len
        
        llm_model = LLM(**llm_kwargs)
        
        sampling_params = SamplingParams(
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens
        )
        
        print("Model loaded successfully")
        
        # Pre-load tokenizer for chat formatting (optimization for batching)
        shared_tokenizer = None
        if AutoTokenizer is not None and args.enable_vllm_batching:
            try:
                shared_tokenizer = AutoTokenizer.from_pretrained(args.model_path)
                print("Tokenizer loaded for batching optimization")
            except:
                print("Warning: Could not load tokenizer, will use raw prompts")
        
    except Exception as e:
        print(f"Error initializing model: {e}")
        return
    
    # Initialize results storage
    evaluation_results = existing_results.get('evaluation_results', []) if existing_results else []
    
    # Track this job's initial contribution to avoid double-counting progress
    initial_results_count = len(evaluation_results)
    job_target_count = len(conflict_preference_pairs)
    
    # Create output data structure
    output_data = {
        'metadata': {
            'model_path': args.model_path,
            'evaluation_type': 'wvs_ablation_conflict_only',
            'ablation_statements_count': args.ablation_statements,
            'total_conflict_cases': len(conflict_preference_pairs) + len(evaluation_results),
            'max_tokens': args.max_tokens,
            'temperature': args.temperature,
            'random_seed': args.random_seed,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'status': 'in_progress',
            'evaluation_settings': [
                'ablation_wvs_only',
                'ablation_wvs_style_neutral',
                'ablation_wvs_style_prefer_wvs',
                'ablation_wvs_style_prefer_style'
            ] + (['ablation_wvs_only_cot',
                'ablation_wvs_style_neutral_cot',
                'ablation_wvs_style_prefer_wvs_cot',
                'ablation_wvs_style_prefer_style_cot'] if args.all_settings else []),
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
    
    if args.enable_vllm_batching:
        # Count total prompts per batch based on ablation settings
        prompts_per_pair = 4  # 4 standard ablation settings
        if args.all_settings:  # Add COT settings
            prompts_per_pair = 8
        
        total_batches = (total_evaluations + args.batch_size - 1) // args.batch_size
        avg_prompts_per_vllm_call = prompts_per_pair * args.batch_size
        print(f"Prompts per preference pair: {prompts_per_pair}")
        print(f"Average prompts per vLLM call: {avg_prompts_per_vllm_call}")
        print(f"Total vLLM calls needed: ~{total_batches} (vs {total_evaluations * prompts_per_pair} without batching)")
        print(f"Estimated throughput improvement: {prompts_per_pair * args.batch_size:.0f}x per vLLM call")
    else:
        print("Note: vLLM batching disabled - using individual calls (slower)")
    
    # Run evaluations
    print(f"\n=== Running Ablation Evaluations ===")
    print(f"Processing {len(conflict_preference_pairs)} conflict cases in batches of {args.batch_size}")
    
    try:
        start_time = time.time()
        
        # Process preference pairs in batches with optimized vLLM batching
        total_items = len(conflict_preference_pairs)
        
        for batch_start in range(0, total_items, args.batch_size):
            batch_end = min(batch_start + args.batch_size, total_items)
            batch_pairs = conflict_preference_pairs[batch_start:batch_end]
            
            print(f"\nProcessing batch {batch_start//args.batch_size + 1}/{(total_items-1)//args.batch_size + 1} "
                  f"(items {batch_start+1}-{batch_end}/{total_items})")
            
            try:
                # Use optimized batch evaluation if enabled, otherwise fallback to individual processing
                if args.enable_vllm_batching:
                    batch_results = evaluate_preference_pairs_batch_optimized_ablation(
                        batch_pairs, ablation_data_manager, llm_model, sampling_params, args, ablation_conflict_sets, shared_tokenizer
                    )
                    evaluation_results.extend(batch_results)
                else:
                    # Process individually when batching is disabled
                    user_profile_cache = {}
                    for preference_pair in batch_pairs:
                        try:
                            # Get user profile from cache or load it
                            user_profile_id = preference_pair['user_profile_id']
                            if user_profile_id not in user_profile_cache:
                                user_profile_cache[user_profile_id] = ablation_data_manager.get_user_profile_by_id(user_profile_id)
                            user_profile = user_profile_cache[user_profile_id]
                            
                            # Evaluate this preference pair individually
                            result = evaluate_preference_pair_ablation_settings(
                                preference_pair, ablation_data_manager, llm_model, sampling_params, args, user_profile
                            )
                            
                            evaluation_results.append(result)
                            
                        except Exception as e2:
                            print(f"Error evaluating preference pair {preference_pair.get('preference_id', 'unknown')}: {e2}")
                            continue
                
            except Exception as e:
                print(f"Error in batch evaluation: {e}")
                print(f"Traceback: {traceback.format_exc()}")
                
                # Fallback to individual processing if batch fails
                print("Falling back to individual processing for this batch...")
                user_profile_cache = user_profile_cache if 'user_profile_cache' in locals() else {}
                
                for preference_pair in batch_pairs:
                    try:
                        # Get user profile from cache or load it
                        user_profile_id = preference_pair['user_profile_id']
                        if user_profile_id not in user_profile_cache:
                            user_profile_cache[user_profile_id] = ablation_data_manager.get_user_profile_by_id(user_profile_id)
                        user_profile = user_profile_cache[user_profile_id]
                        
                        # Evaluate this preference pair individually
                        result = evaluate_preference_pair_ablation_settings(
                            preference_pair, ablation_data_manager, llm_model, sampling_params, args, user_profile
                        )
                        
                        evaluation_results.append(result)
                        
                    except Exception as e2:
                        print(f"Error evaluating individual preference pair {preference_pair.get('preference_id', 'unknown')}: {e2}")
                        continue
            
            # Calculate performance stats
            current_time = time.time()
            elapsed_time = current_time - start_time
            
            # Track THIS JOB's progress only (not the shared file total)
            job_completed = len(evaluation_results) - initial_results_count
            job_total = job_target_count
            
            if elapsed_time > 0:
                eval_rate = job_completed / elapsed_time
                remaining_evals = job_total - job_completed
                remaining_time = remaining_evals / eval_rate if eval_rate > 0 else 0
                
                # Calculate recent accuracy (for ablation_wvs_only setting if available)
                if evaluation_results:
                    recent_results = evaluation_results[-min(20, len(evaluation_results)):]  # Last 20 results
                    wvs_only_correct = sum(1 for r in recent_results if r.get('ablation_wvs_only_correct', False))
                    recent_accuracy = wvs_only_correct / len(recent_results) if recent_results else 0
                    
                    actual_item_start = start_idx + batch_start if args.total_jobs > 1 else batch_start
                    actual_item_end = start_idx + batch_end - 1 if args.total_jobs > 1 else batch_end - 1
                    print(f"Completed batch {batch_start//args.batch_size + 1}. "
                          f"Job: {job_completed}/{job_total} "
                          f"Rate: {eval_rate:.1f}/s "
                          f"Acc: {recent_accuracy:.1%} "
                          f"Range: {actual_item_start}-{actual_item_end}")
            
            # Save results after each batch
            output_data['evaluation_results'] = evaluation_results
            output_data['metrics'] = calculate_metrics(evaluation_results)
            save_incremental_results(output_data, args.output_file)
    
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
    
    print(f"\n=== ABLATION EVALUATION SUMMARY ===")
    print(f"Evaluation type: WVS Ablation (Value-Style Conflicts Only)")
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
    
    print("\nAblation evaluation completed!")
    print(f"Results saved to: {args.output_file}")

if __name__ == "__main__":
    main()
