from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import re
import logging

# Set up logging
logger = logging.getLogger(__name__)

def prepare_input_vector(answers: Dict[str, Any], feature_cols: List[str]) -> np.ndarray:
    """Enhanced conversion of answers into numeric array following feature_cols order.
    
    Args:
        answers: Dictionary of question-answer pairs
        feature_cols: List of feature columns in expected order
        
    Returns:
        numpy array shaped (1, n_features) ready for model prediction
    """
    vec = []
    
    for i, c in enumerate(feature_cols):
        v = answers.get(c, 0)
        
        # First try direct float conversion
        try:
            v_float = float(v)
            vec.append(v_float)
            continue
        except (ValueError, TypeError):
            pass
        
        # Enhanced string processing with comprehensive mapping
        try:
            if v is None:
                v_float = 0.0
            else:
                s = str(v).strip().lower()
                
                # Comprehensive categorical mapping
                categorical_map = {
                    # Yes/True variations
                    'yes': 1.0, 'y': 1.0, 'true': 1.0, '1': 1.0, '✅': 1.0, '✅ yes': 1.0,
                    'always': 3.0, 'very often': 3.0, '⭐': 3.0, '🌟': 3.0,
                    'often': 2.0, 'frequently': 2.0, '👍': 2.0, 'regularly': 2.0,
                    
                    # No/False variations
                    'no': 0.0, 'n': 0.0, 'false': 0.0, '0': 0.0, '❌': 0.0, '❌ no': 0.0,
                    'never': 0.0, 'none': 0.0, 'not at all': 0.0,
                    
                    # Intermediate values
                    'sometimes': 1.0, 'maybe': 1.0, 'some-time': 1.0, 'occasionally': 1.0,
                    '🙂': 1.0, 'possibly': 1.0, 'unsure': 1.0, 'not sure': 1.0,
                    
                    # Age categories (Q13)
                    'before 9 months': 0.0, 'before 9': 0.0, '<9 months': 0.0, '👶': 0.0,
                    '👶 before 9 months': 0.0, 'under 9 months': 0.0,
                    '9-12 months': 1.0, '9 to 12': 1.0, '9-12': 1.0, '🧒': 1.0,
                    '🧒 9-12 months': 1.0, 'between 9-12 months': 1.0,
                    '1-2 years': 2.0, '1 to 2': 2.0, '1-2': 2.0, '👦': 2.0,
                    '👦 1-2 years': 2.0, 'between 1-2 years': 2.0,
                    'after 2 years': 3.0, '>2 years': 3.0, 'after 2': 3.0, '👨': 3.0,
                    '👨 after 2 years': 3.0, 'over 2 years': 3.0,
                    
                    # Word count ranges (Q14)
                    '0-10 words': 0.0, '0-10': 0.0, '�️': 0.0, '�️ 0-10 words': 0.0,
                    'few words': 0.0, 'less than 10': 0.0, '<10': 0.0,
                    '11-30 words': 1.0, '11-30': 1.0, '🗣️': 1.0, '🗣️ 11-30 words': 1.0,
                    'some words': 1.0, '10-30': 1.0,
                    '31-50 words': 2.0, '31-50': 2.0, '💬': 2.0, '💬 31-50 words': 2.0,
                    'many words': 2.0, '30-50': 2.0,
                    '50+ words': 3.0, '50+': 3.0, '🎯': 3.0, '🎯 50+ words': 3.0,
                    'lots of words': 3.0, '>50': 3.0,
                    
                    # Play hours ranges (Q15)
                    '0-2 hours': 0.0, '0-2': 0.0, '🏠': 0.0, '🏠 0-2 hours': 0.0,
                    'less than 2': 0.0, '<2': 0.0, 'little': 0.0,
                    '3-5 hours': 1.0, '3-5': 1.0, '👥': 1.0, '👥 3-5 hours': 1.0,
                    'some': 1.0, '2-5': 1.0,
                    '6-10 hours': 2.0, '6-10': 2.0, '🤝': 2.0, '🤝 6-10 hours': 2.0,
                    'regular': 2.0, '5-10': 2.0,
                    '10+ hours': 3.0, '10+': 3.0, '🌟': 3.0, '🌟 10+ hours': 3.0,
                    'lots': 3.0, '>10': 3.0
                }
                
                # Check exact match first
                if s in categorical_map:
                    v_float = categorical_map[s]
                else:
                    # Handle partial matches and numeric extraction
                    v_float = extract_numeric_from_text(s, c)
                
            vec.append(v_float)
            
        except Exception as e:
            logger.warning(f"Error processing feature '{c}' with value '{v}': {e}")
            vec.append(0.0)  # Default fallback
    
    return np.array(vec).reshape(1, -1)

def extract_numeric_from_text(text: str, feature_name: str) -> float:
    """Extract numeric value from text based on feature context."""
    text_lower = text.lower()
    
    # Handle hour-related features
    if 'hour' in feature_name.lower() or 'q15' in feature_name.lower():
        digits = re.findall(r'\d+\.?\d*', text_lower)
        if digits:
            hours = float(digits[0])
            # Normalize to 0-3 scale
            if hours <= 2:
                return 0.0
            elif hours <= 5:
                return 1.0
            elif hours <= 10:
                return 2.0
            else:
                return 3.0
        return 0.0
    
    # Handle word count features
    elif 'word' in feature_name.lower() or 'q14' in feature_name.lower():
        digits = re.findall(r'\d+\.?\d*', text_lower)
        if digits:
            word_count = float(digits[0])
            # Normalize to 0-3 scale
            if word_count <= 10:
                return 0.0
            elif word_count <= 30:
                return 1.0
            elif word_count <= 50:
                return 2.0
            else:
                return 3.0
        return 0.0
    
    # Handle age features
    elif 'age' in feature_name.lower() or 'q13' in feature_name.lower():
        if 'before' in text_lower or '<9' in text_lower or 'under' in text_lower:
            return 0.0
        elif '9' in text_lower and '12' in text_lower:
            return 1.0
        elif '1' in text_lower and '2' in text_lower:
            return 2.0
        elif 'after' in text_lower or '>2' in text_lower or 'over' in text_lower:
            return 3.0
        # Extract months/years
        digits = re.findall(r'\d+', text_lower)
        if digits:
            age = float(digits[0])
            if 'month' in text_lower:
                if age < 9:
                    return 0.0
                elif age <= 12:
                    return 1.0
            elif 'year' in text_lower:
                if age <= 2:
                    return 2.0
                else:
                    return 3.0
        return 0.0
    
    # Generic numeric extraction
    else:
        digits = re.findall(r'\d+\.?\d*', text_lower)
        if digits:
            return float(digits[0])
        
        # Text to number mapping for common responses
        text_to_number = {
            'never': 0.0, 'none': 0.0, 'zero': 0.0,
            'rarely': 0.5, 'seldom': 0.5,
            'sometimes': 1.0, 'occasionally': 1.0, 'maybe': 1.0,
            'often': 2.0, 'frequently': 2.0, 'usually': 2.0,
            'always': 3.0, 'very often': 3.0, 'constantly': 3.0
        }
        
        for key, value in text_to_number.items():
            if key in text_lower:
                return value
        
        return 0.0

def risk_category(prob: float) -> Tuple[str, str, str]:
    """Calculate risk category with detailed information.
    
    Args:
        prob: Probability score between 0 and 1
        
    Returns:
        Tuple of (category, color_hex, description)
    """
    if prob < 0.2:
        return "Low", "#10b981", "Minimal indicators detected"
    elif prob < 0.35:
        return "Low-Moderate", "#84cc16", "Few indicators present"
    elif prob < 0.5:
        return "Moderate", "#f59e0b", "Some indicators detected"
    elif prob < 0.7:
        return "Moderate-High", "#ef4444", "Multiple indicators present"
    else:
        return "High", "#dc2626", "Strong indicators detected"

def validate_answers(answers: Dict[str, Any], expected_questions: List[str]) -> Dict[str, Any]:
    """Validate and standardize answer formats.
    
    Args:
        answers: Raw answers dictionary
        expected_questions: List of expected question keys
        
    Returns:
        Validated and standardized answers dictionary
    """
    validated = {}
    
    for q, answer in answers.items():
        if q in expected_questions:
            if isinstance(answer, str):
                answer_lower = answer.strip().lower()
                
                # Enhanced validation mapping
                if answer_lower in ['yes', 'y', 'true', '1', '✅', '✅ yes']:
                    validated[q] = 1.0
                elif answer_lower in ['no', 'n', 'false', '0', '❌', '❌ no']:
                    validated[q] = 0.0
                elif answer_lower in ['sometimes', 'maybe', 'some-time', 'occasionally']:
                    validated[q] = 1.0
                elif answer_lower in ['often', 'frequently', 'usually']:
                    validated[q] = 2.0
                elif answer_lower in ['always', 'very often', 'constantly']:
                    validated[q] = 3.0
                else:
                    # Try to preserve original but log warning
                    validated[q] = answer
                    logger.debug(f"Unusual answer format for {q}: {answer}")
            else:
                validated[q] = answer
        else:
            validated[q] = answer
            logger.warning(f"Unexpected question key: {q}")
    
    return validated

def get_answer_statistics(answers: Dict[str, Any]) -> Dict[str, Any]:
    """Get comprehensive statistics about the provided answers.
    
    Args:
        answers: Dictionary of question-answer pairs
        
    Returns:
        Dictionary with detailed statistics
    """
    stats = {
        "total_answers": len(answers),
        "numeric_answers": 0,
        "string_answers": 0,
        "null_answers": 0,
        "answer_types": {},
        "answer_values": {},
        "completion_percentage": 0,
        "risk_indicators": 0
    }
    
    total_questions = len(answers)
    if total_questions == 0:
        return stats
    
    for q, answer in answers.items():
        answer_type = type(answer).__name__
        stats["answer_types"][answer_type] = stats["answer_types"].get(answer_type, 0) + 1
        
        if answer is None:
            stats["null_answers"] += 1
        elif isinstance(answer, (int, float)):
            stats["numeric_answers"] += 1
            # Count potential risk indicators (values > 0 for binary questions)
            if answer > 0 and any(indicator in q.lower() for indicator in ['q1', 'q2', 'q3', 'q4', 'q5', 'q6', 'q7', 'q8', 'q9', 'q10', 'q11']):
                stats["risk_indicators"] += 1
        elif isinstance(answer, str):
            stats["string_answers"] += 1
        
        # Store sample values (first 10)
        if len(stats["answer_values"]) < 10:
            stats["answer_values"][q] = str(answer)[:100]  # Limit length
    
    # Calculate completion percentage
    stats["completion_percentage"] = ((total_questions - stats["null_answers"]) / total_questions) * 100
    
    return stats

def preprocess_special_features(answers: Dict[str, Any]) -> Dict[str, Any]:
    """Preprocess special features (Q13-Q15) to ensure proper formatting.
    
    Args:
        answers: Raw answers dictionary
        
    Returns:
        Preprocessed answers with standardized formats
    """
    processed = answers.copy()
    
    for q, answer in answers.items():
        if answer is None:
            processed[q] = 0.0
            continue
            
        q_lower = q.lower()
        
        # Q13: Age of first words
        if 'first words' in q_lower or 'q13' in q_lower:
            if isinstance(answer, str):
                answer_lower = answer.lower()
                if any(term in answer_lower for term in ['before 9', '<9', 'under 9']):
                    processed[q] = 0.0
                elif any(term in answer_lower for term in ['9-12', '9 to 12', '9-12 months']):
                    processed[q] = 1.0
                elif any(term in answer_lower for term in ['1-2', '1 to 2', '1-2 years']):
                    processed[q] = 2.0
                elif any(term in answer_lower for term in ['after 2', '>2', 'over 2']):
                    processed[q] = 3.0
                else:
                    # Try to extract numeric value
                    digits = re.findall(r'\d+', answer_lower)
                    if digits:
                        age = float(digits[0])
                        if 'month' in answer_lower:
                            processed[q] = 0.0 if age < 9 else 1.0 if age <= 12 else 2.0
                        elif 'year' in answer_lower:
                            processed[q] = 2.0 if age <= 2 else 3.0
        
        # Q14: Word count
        elif 'word' in q_lower or 'q14' in q_lower:
            if isinstance(answer, str):
                digits = re.findall(r'\d+', str(answer))
                if digits:
                    word_count = float(digits[0])
                    if word_count <= 10:
                        processed[q] = 0.0
                    elif word_count <= 30:
                        processed[q] = 1.0
                    elif word_count <= 50:
                        processed[q] = 2.0
                    else:
                        processed[q] = 3.0
                else:
                    processed[q] = 0.0
        
        # Q15: Play hours
        elif 'hour' in q_lower or 'play' in q_lower or 'q15' in q_lower:
            if isinstance(answer, str):
                digits = re.findall(r'\d+', str(answer))
                if digits:
                    hours = float(digits[0])
                    if hours <= 2:
                        processed[q] = 0.0
                    elif hours <= 5:
                        processed[q] = 1.0
                    elif hours <= 10:
                        processed[q] = 2.0
                    else:
                        processed[q] = 3.0
                else:
                    processed[q] = 0.0
    
    return processed

def generate_risk_explanation(probability: float, risk_category: str, answers: Dict[str, Any]) -> str:
    """Generate a human-readable explanation of the risk assessment.
    
    Args:
        probability: Calculated probability
        risk_category: Risk category string
        answers: User answers for context
        
    Returns:
        Explanation text
    """
    base_explanations = {
        "Low": "The screening indicates minimal signs associated with ASD. However, continue to monitor development and consult a professional if you have concerns.",
        "Low-Moderate": "Some mild indicators were noted. Regular developmental monitoring is recommended.",
        "Moderate": "Several indicators suggest further evaluation may be beneficial. Consider consulting a healthcare provider.",
        "Moderate-High": "Multiple significant indicators were identified. Professional assessment is recommended.",
        "High": "Strong indicators suggest a comprehensive evaluation by a specialist would be appropriate."
    }
    
    explanation = base_explanations.get(risk_category, "Assessment completed.")
    
    # Add specific observations based on answers
    observations = []
    
    # Check specific high-impact questions
    if answers.get("Q1. Does your child look at you when you call their name?", 0) == 0:
        observations.append("Lack of response to name")
    
    if answers.get("Q3. Does your child point to share interest?", 0) == 0:
        observations.append("Limited joint attention")
    
    if answers.get("Q6. Does your child engage in pretend play?", 0) == 0:
        observations.append("Limited pretend play")
    
    if answers.get("Q15. Hours per week playing with other children?", 0) <= 1:
        observations.append("Limited social interaction")
    
    if observations:
        explanation += f" Notable observations: {', '.join(observations)}."
    
    return explanation

def check_data_quality(answers: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Check the quality and completeness of the provided answers.
    
    Args:
        answers: Dictionary of answers
        
    Returns:
        Tuple of (is_acceptable, warning_messages)
    """
    warnings = []
    
    # Check for null/missing values
    null_count = sum(1 for answer in answers.values() if answer is None or answer == "")
    if null_count > 0:
        warnings.append(f"{null_count} questions have missing answers")
    
    # Check for inconsistent answer formats
    numeric_count = sum(1 for answer in answers.values() if isinstance(answer, (int, float)))
    string_count = sum(1 for answer in answers.values() if isinstance(answer, str))
    
    if string_count > numeric_count and len(answers) > 5:
        warnings.append("Most answers are in text format - numeric conversion applied")
    
    # Check critical questions
    critical_questions = [
        "Q1. Does your child look at you when you call their name?",
        "Q3. Does your child point to share interest?",
        "Q6. Does your child engage in pretend play?"
    ]
    
    missing_critical = [q for q in critical_questions if answers.get(q) in [None, ""]]
    if missing_critical:
        warnings.append(f"Critical questions missing: {len(missing_critical)}")
    
    is_acceptable = len(warnings) < 3 and null_count < len(answers) * 0.3
    
    return is_acceptable, warnings