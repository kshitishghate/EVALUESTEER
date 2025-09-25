import os
import json
import re
import time
from typing import Dict, List, Tuple, Optional
from openai import OpenAI


client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
# Style families and their prompts - each family generates contrasting pairs
STYLE_FAMILY_PROMPTS = {
    "verbosity": """You are a text transformation assistant. Your task is to create TWO versions of the given text that contrast on VERBOSITY while preserving the exact semantic content and meaning. All other aspects (fluency, confidence, sentiment) should remain high quality and natural.

Create:
1. VERBOSE version: Significantly more detailed, expanded, and comprehensive
2. CONCISE version: Significantly more brief, direct, and to-the-point

Guidelines for VERBOSE version:
- Add detailed explanations and elaborations for each point made
- Include more examples and supporting details
- Use longer, more complex and sophisticated sentence structures
- Add qualifying statements and nuanced explanations
- Include relevant background context where appropriate
- Expand on implications and consequences of the arguments
- Use more descriptive language and precise adjectives

Guidelines for CONCISE version:
- Remove unnecessary words and redundant phrases
- Use shorter, more direct and impactful sentences
- Eliminate repetition while preserving key points
- Focus on core arguments and essential information
- Use more precise and economical language
- Remove filler words and unnecessary qualifiers
- Condense complex ideas into their essential elements

Both versions must preserve all original ideas and arguments while creating maximum contrast on verbosity.""",

    "confidence": """You are a text transformation assistant. Your task is to create TWO versions of the given text that contrast on CONFIDENCE while preserving the exact semantic content and meaning. All other aspects (fluency, verbosity, sentiment) should remain high quality and natural.

Create:
1. HIGH CONFIDENCE version: Extremely confident and certain
2. LOW CONFIDENCE version: Uncertain and tentative

Guidelines for HIGH CONFIDENCE version:
- Use definitive language and strong, unwavering assertions
- Remove all hedging words and uncertainty markers
- Use phrases like "certainly", "definitely", "without doubt", "absolutely"
- Present arguments as established facts rather than mere opinions
- Use authoritative tone and demonstrate strong conviction
- Express complete certainty in the position and reasoning

Guidelines for LOW CONFIDENCE version:
- Add hedging words and uncertainty markers throughout
- Use phrases like "perhaps", "might be", "it seems", "possibly", "I think", "maybe"
- Present arguments as tentative suggestions rather than firm conclusions
- Include expressions of self-doubt and qualification
- Use tentative tone and express notable uncertainty
- Express the position with appropriate reservation and humility

Both versions must preserve all original ideas and arguments while creating maximum contrast on confidence level.""",

    "diversity": """You are a text transformation assistant. Your task is to create TWO versions of the given text that contrast on DIVERSITY/POLARIZATION while preserving the exact semantic content and meaning. All other aspects (fluency, verbosity, confidence) should remain high quality and natural.

Create:
1. EXTREME version: More polarized and uncompromising
2. NEUTRAL version: More balanced and diplomatic

Guidelines for EXTREME version:
- Amplify the existing position to its most uncompromising form
- Use stronger, more intense and dramatic language
- Remove moderation and nuance from the arguments
- Make the stance more black-and-white and absolute
- Use more emphatic and passionate expressions
- Eliminate middle-ground positions and compromises
- Present the view in its most radical form

Guidelines for NEUTRAL version:
- Add nuance and acknowledge the complexity of issues
- Include recognition of multiple perspectives where appropriate
- Use more balanced, measured, and diplomatic language
- Acknowledge potential counterarguments or limitations
- Present the position in a more inclusive and considerate way
- Add qualifying statements that show awareness of other viewpoints
- Use more moderate and diplomatic expressions

Both versions must preserve all original ideas and arguments while creating maximum contrast on polarization/balance.""",

    "sentiment": """You are a text transformation assistant. Your task is to create TWO versions of the given text that contrast on SENTIMENT while preserving the exact semantic content and meaning. All other aspects (fluency, verbosity, confidence) should remain high quality and natural.

Create:
1. WARM version: Extremely warm, positive, and caring
2. COLD version: Cold, detached, and formal

Guidelines for WARM version:
- Use friendly, enthusiastic, and genuinely positive language
- Add warmth, empathy, and emotional connection
- Use encouraging, supportive, and uplifting phrases
- Express genuine care, concern, and understanding
- Include positive framing of ideas and hopeful perspectives
- Use inclusive, welcoming, and compassionate language
- Show empathy and emotional intelligence

Guidelines for COLD version:
- Use formal, distant, and impersonal language
- Remove emotional expressions, warmth, and personal connection
- Use clinical, objective, and matter-of-fact phrasing
- Express ideas in a businesslike and professional manner
- Remove personal investment and emotional engagement
- Use more formal, bureaucratic, and institutional language
- Present information without emotional coloring or bias

Both versions must preserve all original ideas and arguments while creating maximum contrast on emotional tone.""",

    "readability": """You are a text transformation assistant. Your task is to create TWO versions of the given text that contrast on READABILITY while preserving the exact semantic content and meaning. All other aspects (fluency, confidence, sentiment) should remain high quality and natural.

Create:
1. HIGH READING DIFFICULTY version: Significantly more complex and challenging to read
2. LOW READING DIFFICULTY version: Significantly more accessible and easy to read

Guidelines for HIGH READING DIFFICULTY version:
- Use complex, sophisticated, and technical vocabulary where appropriate
- Employ longer, multi-clause sentences with nested structures
- Include abstract concepts and theoretical language
- Use formal academic or professional register and tone
- Employ passive voice constructions where suitable
- Include subordinate clauses and complex grammatical structures
- Use precise but less common terminology and jargon
- Create more cognitively demanding sentence patterns

Guidelines for LOW READING DIFFICULTY version:
- Use simple, common, and everyday vocabulary
- Employ shorter, straightforward sentences with clear structure
- Use concrete examples and accessible language
- Use conversational and informal register and tone
- Employ active voice constructions predominantly
- Use simple, direct grammatical structures
- Replace technical terms with plain language equivalents
- Create easily digestible and straightforward sentence patterns

Both versions must preserve all original ideas and arguments while creating maximum contrast on reading difficulty and cognitive accessibility."""
}

# Mapping of style families to their output field names
STYLE_FAMILY_MAPPING = {
    "verbosity": ["verbose", "concise"],
    "confidence": ["high_confidence", "low_confidence"],
    "diversity": ["extreme", "neutral"],
    "sentiment": ["warm", "cold"],
    "readability": ["high_reading_difficulty", "low_reading_difficulty"]
}

def extract_paired_content_from_response(response_text: str, style_family: str) -> Tuple[str, str]:
    """Extract both variations from a paired response."""
    
    style_names = STYLE_FAMILY_MAPPING[style_family]
    first_style, second_style = style_names
    
    # First try: Standard JSON parsing
    try:
        result = json.loads(response_text)
        first_content = result.get(f"{first_style}_version", "")
        second_content = result.get(f"{second_style}_version", "")
        if first_content and second_content:
            return first_content, second_content
    except json.JSONDecodeError:
        pass
    
    # Second try: Look for both versions in JSON-like structure
    patterns = [
        (f'"{first_style}_version"\\s*:\\s*"([^"]*(?:\\\\.[^"]*)*)"', f'"{second_style}_version"\\s*:\\s*"([^"]*(?:\\\\.[^"]*)*)"'),
        (f'"version_1"\\s*:\\s*"([^"]*(?:\\\\.[^"]*)*)"', f'"version_2"\\s*:\\s*"([^"]*(?:\\\\.[^"]*)*)"'),
        (f'"first"\\s*:\\s*"([^"]*(?:\\\\.[^"]*)*)"', f'"second"\\s*:\\s*"([^"]*(?:\\\\.[^"]*)*)"')
    ]
    
    for first_pattern, second_pattern in patterns:
        first_match = re.search(first_pattern, response_text, re.DOTALL | re.IGNORECASE)
        second_match = re.search(second_pattern, response_text, re.DOTALL | re.IGNORECASE)
        
        if first_match and second_match:
            first_content = first_match.group(1).replace('\\"', '"').replace('\\n', '\n').replace('\\\\', '\\')
            second_content = second_match.group(1).replace('\\"', '"').replace('\\n', '\n').replace('\\\\', '\\')
            return first_content, second_content
    
    # Third try: Look for numbered sections
    section_patterns = [
        (r'1\.?\s*(?:VERBOSE|HIGH CONFIDENCE|EXTREME|WARM)(?:\s+version)?:?\s*([^\n]*(?:\n(?!2\.)[^\n]*)*)', 
         r'2\.?\s*(?:CONCISE|LOW CONFIDENCE|NEUTRAL|COLD)(?:\s+version)?:?\s*([^\n]*(?:\n(?!1\.|\d+\.)[^\n]*)*)'),
        (r'(?:Version\s+)?1:?\s*([^\n]*(?:\n(?!(?:Version\s+)?2:)[^\n]*)*)', 
         r'(?:Version\s+)?2:?\s*([^\n]*(?:\n(?!(?:Version\s+)?1:|\d+:)[^\n]*)*)')
    ]
    
    for first_pattern, second_pattern in section_patterns:
        first_match = re.search(first_pattern, response_text, re.DOTALL | re.IGNORECASE)
        second_match = re.search(second_pattern, response_text, re.DOTALL | re.IGNORECASE)
        
        if first_match and second_match:
            first_content = first_match.group(1).strip()
            second_content = second_match.group(1).strip()
            if len(first_content) > 20 and len(second_content) > 20:
                return first_content, second_content
    
    return "", ""

def generate_style_family_variations(original_text: str, style_family: str, temperature: float = 0.7, max_retries: int = 3) -> Tuple[str, str]:
    """Generate a pair of contrasting style variations from the same family."""
    
    if style_family not in STYLE_FAMILY_PROMPTS:
        return f"Error: Unknown style family '{style_family}'", f"Error: Unknown style family '{style_family}'"
    
    system_prompt = STYLE_FAMILY_PROMPTS[style_family]
    style_names = STYLE_FAMILY_MAPPING[style_family]
    first_style, second_style = style_names
    
    # Add JSON format instruction to the system prompt
    system_prompt += f"""

CRITICAL: You MUST return your response in EXACTLY this JSON format with no additional text, markdown, or formatting:
{{
  "{first_style}_version": "...",
  "{second_style}_version": "..."
}}

Ensure both versions are complete and substantially different from each other on the target dimension while preserving semantic meaning.
Do NOT include any explanations, comments, or other text outside of this JSON structure."""
    
    user_payload = {
        "original_text": original_text
    }
    
    for attempt in range(max_retries + 1):
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                temperature=temperature,
                max_tokens=2500,  # Increased for paired generations
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": json.dumps(user_payload)}
                ]
            )
            
            response_text = response.choices[0].message.content.strip()
            first_content, second_content = extract_paired_content_from_response(response_text, style_family)
            
            if first_content and second_content:
                return first_content, second_content
            else:
                print(f"  Warning: Could not extract paired content from response (attempt {attempt + 1}): {response_text[:100]}...")
                if attempt < max_retries:
                    print(f"  Retrying...")
                    time.sleep(1)
                    continue
                else:
                    return f"Error: Could not extract paired content after {max_retries + 1} attempts", f"Error: Could not extract paired content after {max_retries + 1} attempts"
            
        except Exception as e:
            print(f"  Error generating {style_family} variations (attempt {attempt + 1}): {str(e)}")
            if attempt < max_retries:
                print(f"  Retrying...")
                time.sleep(2)
                continue
            else:
                return f"Error: Could not generate {style_family} variations after {max_retries + 1} attempts - {str(e)}", f"Error: Could not generate {style_family} variations after {max_retries + 1} attempts - {str(e)}"
    
    return "Error: Unexpected error in generation", "Error: Unexpected error in generation"

def process_single_item(item: Dict, style_families: List[str]) -> Dict:
    """Process a single item to generate all style family variations using paired API calls."""
    
    new_item = item.copy()
    
    # Process completion_A variations (one call per family)
    if 'completion_A' in item:
        for family in style_families:
            style_names = STYLE_FAMILY_MAPPING[family]
            print(f"    Generating completion_A {family} family ({style_names[0]} & {style_names[1]}) in single call...")
            
            first_content, second_content = generate_style_family_variations(item['completion_A'], family)
            new_item[f"completion_A_{style_names[0]}"] = first_content
            new_item[f"completion_A_{style_names[1]}"] = second_content
            
            time.sleep(0.5)  # Brief pause between family calls
    
    # Process completion_B variations (one call per family)
    if 'completion_B' in item:
        for family in style_families:
            style_names = STYLE_FAMILY_MAPPING[family]
            print(f"    Generating completion_B {family} family ({style_names[0]} & {style_names[1]}) in single call...")
            
            first_content, second_content = generate_style_family_variations(item['completion_B'], family)
            new_item[f"completion_B_{style_names[0]}"] = first_content
            new_item[f"completion_B_{style_names[1]}"] = second_content
            
            time.sleep(0.5)  # Brief pause between family calls
    
    return new_item

def identify_and_fix_errors(data: List[Dict], style_families: List[str]) -> Tuple[List[Dict], int]:
    """Identify items with errors and fix them using the paired generation approach."""
    
    fixed_count = 0
    error_fields = []
    
    # Identify all fields that might have errors
    for family in style_families:
        style_names = STYLE_FAMILY_MAPPING[family]
        for style in style_names:
            error_fields.extend([f"completion_A_{style}", f"completion_B_{style}"])
    
    # Find items with errors
    items_to_fix = []
    for i, item in enumerate(data):
        needs_fixing = False
        error_fields_in_item = []
        
        for field in error_fields:
            if field in item and item[field].startswith('Error:'):
                needs_fixing = True
                error_fields_in_item.append(field)
        
        if needs_fixing:
            items_to_fix.append((i, error_fields_in_item))
    
    print(f"Found {len(items_to_fix)} items with errors that need fixing.")
    
    if not items_to_fix:
        print("No errors found! All items are already processed correctly.")
        return data, 0
    
    # Group errors by family for paired fixing
    for idx, (item_index, error_fields_in_item) in enumerate(items_to_fix):
        item = data[item_index]
        print(f"Fixing item {idx + 1}/{len(items_to_fix)} (Question ID: {item.get('question_id', 'Unknown')}, Index: {item_index})")
        
        # Group errors by completion and family
        families_to_fix = {"completion_A": set(), "completion_B": set()}
        
        for error_field in error_fields_in_item:
            if error_field.startswith('completion_A_'):
                # Find which family this style belongs to
                style = error_field.replace('completion_A_', '')
                for family, style_names in STYLE_FAMILY_MAPPING.items():
                    if style in style_names:
                        families_to_fix["completion_A"].add(family)
                        break
            elif error_field.startswith('completion_B_'):
                style = error_field.replace('completion_B_', '')
                for family, style_names in STYLE_FAMILY_MAPPING.items():
                    if style in style_names:
                        families_to_fix["completion_B"].add(family)
                        break
        
        # Fix entire families at once
        for completion_field in ["completion_A", "completion_B"]:
            if completion_field in item:
                for family in families_to_fix[completion_field]:
                    style_names = STYLE_FAMILY_MAPPING[family]
                    print(f"  Regenerating {completion_field} {family} family...")
                    
                    first_content, second_content = generate_style_family_variations(item[completion_field], family)
                    
                    if not first_content.startswith('Error:') and not second_content.startswith('Error:'):
                        item[f"{completion_field}_{style_names[0]}"] = first_content
                        item[f"{completion_field}_{style_names[1]}"] = second_content
                        fixed_count += 2  # Fixed both variations in the family
                        print(f"  Fixed {completion_field} {family} family ({style_names[0]} & {style_names[1]})")
                    else:
                        print(f"  Still failed: {first_content[:50]}... / {second_content[:50]}...")
                    
                    time.sleep(1)  # Pause between error fixing attempts
    
    return data, fixed_count

def process_data_with_style_variations(input_file_path: str, output_file_path: str, style_families: List[str]):
    """Process the JSON data to add style family variations of completions."""
    
    # Load the original data
    print(f"Loading data from {input_file_path}...")
    with open(input_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Loaded {len(data)} items.")
    print(f"Style families to generate: {', '.join(style_families)}")
    print("Note: Each family generates contrasting pairs in a single API call for maximum diversity.")
    
    # Calculate total API calls
    total_calls_per_item = len(style_families) * 2  # families * (completion_A + completion_B)
    total_calls = len(data) * total_calls_per_item
    print(f"Total API calls needed: {total_calls} ({total_calls_per_item} per item)")
    
    # Process each item
    processed_data = []
    for i, item in enumerate(data):
        print(f"Processing item {i+1}/{len(data)} (Question ID: {item.get('question_id', 'Unknown')})")
        
        # Generate all style family variations for this item
        new_item = process_single_item(item, style_families)
        processed_data.append(new_item)
        
        # Save progress after each item
        print(f"  Saving progress after item {i+1}...")
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(processed_data, f, indent=2, ensure_ascii=False)
    
    # Identify and fix any errors
    print("\nChecking for errors and fixing them...")
    processed_data, fixed_count = identify_and_fix_errors(processed_data, style_families)
    
    # Save the final result
    print(f"Saving final result to {output_file_path}...")
    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, indent=2, ensure_ascii=False)
    
    print(f"Processing complete! Fixed {fixed_count} errors. Output saved to {output_file_path}")

def get_style_families_to_generate():
    """Interactive function to select which style families to generate."""
    
    available_families = {
        "verbosity": ["verbose", "concise"],
        "confidence": ["high_confidence", "low_confidence"],
        "diversity": ["extreme", "neutral"],
        "sentiment": ["warm", "cold"],
        "readability": ["high_reading_difficulty", "low_reading_difficulty"]
    }
    
    print("Available style families:")
    for i, (family, variations) in enumerate(available_families.items(), 1):
        print(f"{i}. {family}: {variations[0]} & {variations[1]}")
    
    print("\nOptions:")
    print("A. Generate ALL style families (default)")
    print("S. Select specific families")
    
    choice = input("\nEnter your choice (A/S) or press Enter for all: ").strip().upper()
    
    if choice == "S":
        selected_families = []
        print("\nSelect families to generate (enter numbers separated by commas):")
        print("Example: 1,3,5 for verbosity, diversity, and readability")
        
        while True:
            try:
                selection = input("Enter family numbers: ").strip()
                if not selection:
                    print("No selection made. Using all families.")
                    return list(available_families.keys())
                
                numbers = [int(x.strip()) for x in selection.split(',')]
                family_names = list(available_families.keys())
                
                for num in numbers:
                    if 1 <= num <= len(family_names):
                        family = family_names[num - 1]
                        if family not in selected_families:
                            selected_families.append(family)
                    else:
                        print(f"Invalid number: {num}")
                        continue
                
                if selected_families:
                    print(f"\nSelected families: {', '.join(selected_families)}")
                    return selected_families
                else:
                    print("No valid families selected. Please try again.")
                    
            except ValueError:
                print("Invalid input. Please enter numbers separated by commas.")
    
    # Default: return all families
    return list(available_families.keys())

def main():
    """Main function to run the paired style variation process."""
    
    input_file = "prism_wvs_generated_data_v2.json"
    
    # Get style families to generate
    style_families = get_style_families_to_generate()
    
    # Create output filename based on selected families
    if len(style_families) == 5:  # All families
        output_file = "prism_wvs_generated_data_v2_with_style_variations_v2.json"
    else:
        # Create a descriptive filename for partial generation
        family_codes = {
            "verbosity": "verb",
            "confidence": "conf", 
            "diversity": "div",
            "sentiment": "sent",
            "readability": "read"
        }
        codes = "_".join([family_codes[f] for f in style_families])
        output_file = f"prism_wvs_generated_data_v2_with_style_variations_{codes}.json"
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file {input_file} not found!")
        return
    
    print("Starting paired style variation generation process...")
    print(f"Input file: {input_file}")
    print(f"Output file: {output_file}")
    print(f"\nStyle families to be generated ({len(style_families)} families):")
    
    family_descriptions = {
        "verbosity": "verbose & concise",
        "confidence": "high_confidence & low_confidence",
        "diversity": "extreme & neutral", 
        "sentiment": "warm & cold",
        "readability": "high_reading_difficulty & low_reading_difficulty"
    }
    
    for i, family in enumerate(style_families, 1):
        print(f"{i}. {family.capitalize()}: {family_descriptions[family]} (single call)")
    
    print("\nEach family generates contrasting pairs in a single API call for maximum diversity.")
    print("Different families use separate calls to maintain independence.\n")
    
    try:
        process_data_with_style_variations(input_file, output_file, style_families)
        print("Process completed successfully!")
        
    except Exception as e:
        print(f"Error during processing: {str(e)}")
        raise

if __name__ == "__main__":
    main() 
