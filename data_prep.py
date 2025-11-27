"""
Data preparation utilities for Final.csv (ASD screening).
Updated for consistent question formatting and enhanced preprocessing.
"""
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import logging
import re

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATA_PATH = Path("Final.csv")

def create_question_mapping():
    """Create mapping between CSV questions and Streamlit questions"""
    question_mapping = {
        # CSV Questions → Streamlit Questions
        "Q NO 1: Does your child look at you when you call his/her name?": 
            "Q1. Does your child look at you when you call their name?",
        
        "Q NO 2 : Does your child point at things they want (e.g., a toy or snack)?": 
            "Q2. Does your child point at things they want?",
        
        "Q NO 03: Does your child point at objects to share interest with you (e.g., pointing at a plane in the sky)?": 
            "Q3. Does your child point to share interest?",
        
        "Q NO 04 : Does your child look at your face when talking or playing with you?": 
            "Q4. Does your child look at your face while talking/playing?",
        
        "Q NO 05: Does your child follow your gaze if you look at something (e.g., if you look at a toy, does the child look too)?": 
            "Q5. Does your child follow your gaze?",
        
        "Q NO 6 : Does your child engage in pretend play (e.g., pretending to cook, talking to dolls, or playing with toy cars as if they are real)?": 
            "Q6. Does your child engage in pretend play?",
        
        " Q NO 07 : If you or someone else in the family is visibly upset, does your child show signs of wanting to comfort them?": 
            "Q7. If someone is upset, does your child show comfort attempts?",
        
        "Q NO 08 :  Does your child get upset when their daily routine is changed (e.g., eating at a different time, taking a different route to school)?": 
            "Q8. Does your child get upset when routine changes?",
        
        "Q NO 09 : Does your child use simple gestures?": 
            "Q9. Does your child use simple gestures (wave, etc)?",
        
        "Q NO 10 : Did you ever expect the child was nearly deaf?": 
            "Q10. Did you ever suspect the child was nearly deaf?",
        
        "Q NO 11: Was there a time when the child insisted on listening to some certain music?": 
            "Q11. Does the child insist on specific music or sounds?",
        
        " Q NO 12 : During the first year, did the child react to bright lights, bright colors, or unusual sounds?": 
            "Q12. In the first year, did the child react to lights/sounds?",
        
        " Q NO 13 : At what age did the child say their first words?": 
            "Q13. At what age did the child say first words?",
        
        " Q  NO 14 : Approximately how many words can your child say clearly?": 
            "Q14. Approximately how many words can the child say?",
        
        " Q no 15 : How many hours per week does your child spend playing with other children?": 
            "Q15. Hours per week playing with other children?"
    }
    return question_mapping

def preprocess_special_answers(df: pd.DataFrame) -> pd.DataFrame:
    """Enhanced preprocessing for special question types (Q12-Q15) with better error handling"""
    df_processed = df.copy()
    
    # Define question mappings for special processing
    question_mapping = create_question_mapping()
    reverse_mapping = {v: k for k, v in question_mapping.items()}
    
    # Q12: Lights/sounds reaction - convert to binary
    q12_streamlit = "Q12. In the first year, did the child react to lights/sounds?"
    q12_csv = reverse_mapping.get(q12_streamlit, q12_streamlit)
    if q12_csv in df_processed.columns:
        try:
            df_processed[q12_csv] = df_processed[q12_csv].astype(str).str.strip().str.lower()
            df_processed[q12_csv] = df_processed[q12_csv].map({
                'yes': 1, 'y': 1, 'true': 1, '1': 1,
                'no': 0, 'n': 0, 'false': 0, '0': 0,
                'sometimes': 1, 'maybe': 1, 'some-time': 1
            }).fillna(df_processed[q12_csv])
            logger.info("Preprocessed Q12: Converted to binary format")
        except Exception as e:
            logger.error(f"Error preprocessing Q12: {e}")
    
    # Q13: Age of first words - convert to categories
    q13_streamlit = "Q13. At what age did the child say first words?"
    q13_csv = reverse_mapping.get(q13_streamlit, q13_streamlit)
    if q13_csv in df_processed.columns:
        try:
            df_processed[q13_csv] = df_processed[q13_csv].astype(str).str.strip().str.lower()
            df_processed[q13_csv] = df_processed[q13_csv].map({
                'before 9 months': 0,
                '9-12 months': 1,
                '1-2 years': 2, 
                'after 2 years': 3
            }).fillna(df_processed[q13_csv])
            logger.info("Preprocessed Q13: Converted age categories to numeric")
        except Exception as e:
            logger.error(f"Error preprocessing Q13: {e}")
    
    # Q14: Word count - extract numbers and normalize
    q14_streamlit = "Q14. Approximately how many words can the child say?"
    q14_csv = reverse_mapping.get(q14_streamlit, q14_streamlit)
    if q14_csv in df_processed.columns:
        def extract_word_count(val):
            if pd.isna(val):
                return 0
            try:
                val_str = str(val).strip().lower()
                # Extract numbers
                numbers = re.findall(r'\d+', val_str)
                if numbers:
                    word_count = float(numbers[0])
                    # Normalize to 0-3 scale similar to Streamlit app
                    if word_count <= 10:
                        return 0
                    elif word_count <= 30:
                        return 1
                    elif word_count <= 50:
                        return 2
                    else:
                        return 3
                return 0
            except Exception:
                return 0
        
        try:
            df_processed[q14_csv] = df_processed[q14_csv].apply(extract_word_count)
            logger.info("Preprocessed Q14: Extracted and normalized word counts")
        except Exception as e:
            logger.error(f"Error preprocessing Q14: {e}")
    
    # Q15: Play hours - extract numbers and normalize
    q15_streamlit = "Q15. Hours per week playing with other children?"
    q15_csv = reverse_mapping.get(q15_streamlit, q15_streamlit)
    if q15_csv in df_processed.columns:
        def extract_play_hours(val):
            if pd.isna(val):
                return 0
            try:
                val_str = str(val).strip().lower()
                # Extract numbers
                numbers = re.findall(r'\d+', val_str)
                if numbers:
                    hours = float(numbers[0])
                    # Normalize to 0-3 scale similar to Streamlit app
                    if hours <= 2:
                        return 0
                    elif hours <= 5:
                        return 1
                    elif hours <= 10:
                        return 2
                    else:
                        return 3
                return 0
            except Exception:
                return 0
        
        try:
            df_processed[q15_csv] = df_processed[q15_csv].apply(extract_play_hours)
            logger.info("Preprocessed Q15: Extracted and normalized play hours")
        except Exception as e:
            logger.error(f"Error preprocessing Q15: {e}")
    
    return df_processed

def rename_dataframe_columns(df):
    """Rename dataframe columns from long CSV format to short Streamlit format"""
    question_mapping = create_question_mapping()
    
    # Create the actual mapping for columns that exist in the dataframe
    column_mapping = {}
    for csv_q, streamlit_q in question_mapping.items():
        if csv_q in df.columns:
            column_mapping[csv_q] = streamlit_q
    
    # Rename the columns
    df_renamed = df.rename(columns=column_mapping)
    
    logger.info(f"Renamed {len(column_mapping)} columns from CSV to Streamlit format")
    logger.info(f"Original columns: {list(df.columns)}")
    logger.info(f"Renamed columns: {list(df_renamed.columns)}")
    
    return df_renamed

def load_dataset(path: str = None) -> pd.DataFrame:
    """Load dataset with enhanced error handling and preprocessing"""
    path = path or DATA_PATH
    try:
        df = pd.read_csv(path)
        logger.info(f"Loaded dataset from {path} with {len(df)} rows and {len(df.columns)} columns")
        
        # Apply special preprocessing for Q12-Q15 first
        df = preprocess_special_answers(df)
        
        # Rename columns to Streamlit format
        df = rename_dataframe_columns(df)
        
        return df
    except FileNotFoundError:
        logger.error(f"Dataset file not found at {path}")
        raise
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        raise

def basic_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """Enhanced data cleaning with detailed logging and better handling"""
    df = df.copy()
    initial_shape = df.shape
    
    # Remove duplicates
    df = df.drop_duplicates().reset_index(drop=True)
    duplicates_removed = initial_shape[0] - len(df)
    
    # Drop rows that are entirely empty
    df = df.dropna(how="all")
    empty_rows_removed = initial_shape[0] - duplicates_removed - len(df)
    
    # Remove columns with all missing values
    initial_cols = len(df.columns)
    df = df.dropna(axis=1, how='all')
    empty_cols_removed = initial_cols - len(df.columns)
    
    # Fill remaining NaN values with 0 for numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)
    
    logger.info(f"Data cleaning completed:")
    logger.info(f"  - Removed {duplicates_removed} duplicate rows")
    logger.info(f"  - Removed {empty_rows_removed} empty rows") 
    logger.info(f"  - Removed {empty_cols_removed} empty columns")
    logger.info(f"  - Final shape: {df.shape}")
    
    return df

def infer_features_and_target(df: pd.DataFrame):
    """Return (feature_cols, target_col) with enhanced detection and validation"""
    # Check for target column with case insensitivity
    target_candidates = ['target', 'TARGET', 'Target']
    target = None
    
    for candidate in target_candidates:
        if candidate in df.columns:
            target = candidate
            break
    
    # If no target found, use last column
    if target is None:
        target = df.columns[-1]
        logger.info(f"No 'target' column found. Using last column '{target}' as target")
    else:
        logger.info(f"Using '{target}' as target column")
    
    features = [c for c in df.columns if c != target]
    
    # Validate that we have features
    if not features:
        raise ValueError("No feature columns found in dataset")
    
    logger.info(f"Identified {len(features)} feature columns and target '{target}'")
    
    # Log feature names for verification
    logger.info("Feature columns:")
    for i, feature in enumerate(features, 1):
        logger.info(f"  {i:2d}. {feature}")
    
    return features, target

def train_test_split_df(X, y, test_size=0.2, random_state=42):
    """Enhanced train-test split with stratification and validation"""
    if len(np.unique(y)) > 1:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=random_state, 
            stratify=y
        )
        logger.info(f"Stratified train-test split: Train={len(X_train)}, Test={len(X_test)}")
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=random_state
        )
        logger.warning("Only one class in target, cannot stratify")
        logger.info(f"Simple train-test split: Train={len(X_train)}, Test={len(X_test)}")
    
    return X_train, X_test, y_train, y_test

def get_dataset_stats(df: pd.DataFrame, target_col: str):
    """Get comprehensive dataset statistics with better error handling"""
    stats = {
        "total_samples": len(df),
        "total_features": len(df.columns) - 1,  # excluding target
        "target_distribution": {},
        "missing_values": df.isnull().sum().to_dict(),
        "data_types": df.dtypes.to_dict(),
        "feature_ranges": {}
    }
    
    try:
        stats["target_distribution"] = df[target_col].value_counts().to_dict()
    except Exception as e:
        logger.error(f"Error getting target distribution: {e}")
    
    # Calculate ranges for numeric features
    for col in df.columns:
        if col != target_col and pd.api.types.is_numeric_dtype(df[col]):
            try:
                stats["feature_ranges"][col] = {
                    "min": df[col].min(),
                    "max": df[col].max(),
                    "mean": df[col].mean()
                }
            except Exception as e:
                logger.error(f"Error calculating ranges for {col}: {e}")
    
    logger.info("Dataset Statistics:")
    logger.info(f"  Total samples: {stats['total_samples']}")
    logger.info(f"  Total features: {stats['total_features']}")
    logger.info(f"  Target distribution: {stats['target_distribution']}")
    logger.info(f"  Missing values: {sum(stats['missing_values'].values())} total")
    
    return stats

def analyze_question_types(df: pd.DataFrame, features: list):
    """Analyze and report on different question types in the dataset"""
    question_categories = {
        "binary_questions": [],
        "scale_questions": [],
        "age_questions": [],
        "count_questions": []
    }
    
    for feature in features:
        feature_lower = feature.lower()
        
        # Categorize based on question content
        if any(keyword in feature_lower for keyword in ['light', 'sound', 'deaf', 'music']):
            question_categories["binary_questions"].append(feature)
        elif any(keyword in feature_lower for keyword in ['age', 'year', 'month']):
            question_categories["age_questions"].append(feature)
        elif any(keyword in feature_lower for keyword in ['word', 'hour']):
            question_categories["count_questions"].append(feature)
        else:
            question_categories["scale_questions"].append(feature)
    
    logger.info("Question Type Analysis:")
    for category, questions in question_categories.items():
        logger.info(f"  {category}: {len(questions)} questions")
        for q in questions:
            logger.info(f"    - {q}")
    
    return question_categories

def convert_answers_to_model_format(answers: dict, feature_cols: list) -> np.ndarray:
    """Convert user answers to model input format with robust error handling"""
    vec = []
    for col in feature_cols:
        try:
            value = answers.get(col, 0)
            
            # Handle different data types
            if isinstance(value, (int, float)):
                vec.append(float(value))
            elif isinstance(value, str):
                value_lower = value.strip().lower()
                
                # Map categorical values to numeric
                if value_lower in ['yes', 'y', 'true', '1', '✅ yes']:
                    vec.append(1.0)
                elif value_lower in ['no', 'n', 'false', '0', '❌ no']:
                    vec.append(0.0)
                elif value_lower in ['sometimes', 'maybe', 'some-time']:
                    vec.append(1.0)
                elif 'before 9' in value_lower:
                    vec.append(0.0)
                elif '9-12' in value_lower:
                    vec.append(1.0)
                elif '1-2' in value_lower:
                    vec.append(2.0)
                elif 'after 2' in value_lower:
                    vec.append(3.0)
                else:
                    # Try to extract numeric value
                    numbers = re.findall(r'\d+', value_lower)
                    if numbers:
                        vec.append(float(numbers[0]))
                    else:
                        vec.append(0.0)
            else:
                vec.append(0.0)
                
        except Exception as e:
            logger.warning(f"Error processing column {col}: {e}, using default value 0")
            vec.append(0.0)
    
    return np.array(vec).reshape(1, -1)

if __name__ == "__main__":
    # Test the data preparation pipeline
    try:
        df = load_dataset()
        df = basic_cleaning(df)
        features, target = infer_features_and_target(df)
        
        stats = get_dataset_stats(df, target)
        question_categories = analyze_question_types(df, features)
        
        print("\n" + "="*50)
        print("DATA PREPARATION SUMMARY")
        print("="*50)
        print(f"Final dataset shape: {df.shape}")
        print(f"Features: {len(features)}")
        print(f"Target: {target}")
        print(f"Target distribution: {stats['target_distribution']}")
        
        print(f"\nQuestion Categories:")
        for category, questions in question_categories.items():
            print(f"  {category}: {len(questions)}")
        
        print(f"\nFeature names:")
        for i, feature in enumerate(features, 1):
            print(f"  {i:2d}. {feature}")
        
    except Exception as e:
        print(f"Error in data preparation: {e}")