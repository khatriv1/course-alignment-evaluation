# sight_alignment_evaluation/utils/data_loader.py

"""
Data loading utilities for the SIGHT dataset.
Simplified to ONLY handle SIGHT format data.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional

def load_and_preprocess_sight_data(file_path: str):
    """
    Load and preprocess SIGHT data.
    """
    print(f"Loading data from: {file_path}")
    
    # Read CSV file
    try:
        df = pd.read_csv(file_path)
        print(f"Loaded {len(df)} rows with {len(df.columns)} columns")
    except Exception as e:
        raise Exception(f"Error reading CSV file: {str(e)}")
    
    # Process SIGHT format
    print("Processing SIGHT dataset format.")
    return process_sight_format(df)

def process_sight_format(df):
    """Process SIGHT dataset format."""
    processed_data = []
    
    # The 9 SIGHT categories
    categories = {
        'general': 'general',
        'confusion': 'confusion', 
        'pedagogy': 'pedagogy',
        'setup': 'setup',
        'gratitude': 'gratitude',
        'personal_experience': 'personal_experience',
        'clarification': 'clarification',
        'non_english': 'non_english',
        'na': 'na'
    }
    
    # Process each comment
    for idx, row in df.iterrows():
        if pd.isna(row.get('comment', None)):
            continue
            
        # Basic comment info
        comment_data = {
            'comment_id': row.get('comment_id', idx),
            'comment': row['comment'],
            'video_id': row.get('video_id', ''),
            'video_name': row.get('video_name', ''),
            'playlist_name': row.get('playlist_name', ''),
            'datasplit': row.get('datasplit', '')
        }
        
        # Extract human annotations
        for category in categories.keys():
            # Get H1's annotation (Human 1)
            h1_col = f'annotator_H1_{category}'
            h1_value = row.get(h1_col, np.nan)
            comment_data[f'H1_{category}'] = bool(h1_value) if not pd.isna(h1_value) else None
            
            # Get H2's annotation (Human 2)
            h2_col = f'annotator_H2_{category}'
            h2_value = row.get(h2_col, np.nan)
            comment_data[f'H2_{category}'] = bool(h2_value) if not pd.isna(h2_value) else None
            
            # Calculate consensus (both humans must agree)
            if not pd.isna(h1_value) and not pd.isna(h2_value):
                comment_data[f'consensus_{category}'] = bool(h1_value) and bool(h2_value)
            else:
                comment_data[f'consensus_{category}'] = None
        
        processed_data.append(comment_data)
    
    result_df = pd.DataFrame(processed_data)
    print(f"\nProcessed {len(result_df)} comments with annotations")
    
    # Print annotation statistics
    print("\nAnnotation statistics (consensus):")
    for category in categories.keys():
        consensus_col = f'consensus_{category}'
        if consensus_col in result_df.columns:
            count = result_df[consensus_col].sum()
            total = len(result_df)
            print(f"  {category}: {count}/{total} ({count/total*100:.1f}%)")
    
    return result_df

def get_comment_categories(row: pd.Series, annotator: str = 'consensus') -> List[str]:
    """
    Extract categories assigned to a comment.
    
    Args:
        row: A row from the dataframe
        annotator: Which annotator to use ('H1', 'H2', or 'consensus')
    
    Returns:
        List of category names that were marked as True
    """
    categories = [
        'general', 'confusion', 'pedagogy', 'setup', 'gratitude', 
        'personal_experience', 'clarification', 'non_english', 'na'
    ]
    
    assigned_categories = []
    for category in categories:
        col_name = f'{annotator}_{category}'
        if col_name in row and row[col_name] is True:
            assigned_categories.append(category)
    
    return assigned_categories

def filter_annotated_comments(df: pd.DataFrame, min_annotations: int = 1, 
                            annotator: str = 'consensus') -> pd.DataFrame:
    """
    Filter to keep only comments that have at least min_annotations categories assigned.
    """
    categories = [
        'general', 'confusion', 'pedagogy', 'setup', 'gratitude', 
        'personal_experience', 'clarification', 'non_english', 'na'
    ]
    
    # Count how many categories each comment has
    annotation_counts = []
    for idx, row in df.iterrows():
        count = 0
        for category in categories:
            col_name = f'{annotator}_{category}'
            if col_name in row and row[col_name] is True:
                count += 1
        annotation_counts.append(count)
    
    df['annotation_count'] = annotation_counts
    filtered_df = df[df['annotation_count'] >= min_annotations].copy()
    
    print(f"Filtered to {len(filtered_df)} comments with at least {min_annotations} annotations")
    return filtered_df

# OLD CODE ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# # sight_alignment_evaluation/utils/data_loader.py

# """
# Data loading utilities for the SIGHT dataset.
# Modified to handle different data formats.
# """

# import pandas as pd
# import numpy as np
# from typing import List, Dict, Tuple, Optional

# def load_and_preprocess_sight_data(file_path: str):
#     """
#     Load and preprocess data for SIGHT evaluation.
#     Now handles both SIGHT format and course alignment format.
#     """
#     print(f"Loading data from: {file_path}")
    
#     # Read CSV file
#     try:
#         df = pd.read_csv(file_path)
#         print(f"Loaded {len(df)} rows with {len(df.columns)} columns")
#     except Exception as e:
#         raise Exception(f"Error reading CSV file: {str(e)}")
    
#     # Check if this is the course alignment dataset (has 'title' and 'description' columns)
#     if 'title' in df.columns and 'description' in df.columns:
#         print("Detected course alignment dataset format. Converting for SIGHT evaluation...")
#         return convert_course_data_to_sight_format(df)
    
#     # Check if this is the SIGHT dataset format
#     elif 'comment' in df.columns:
#         print("Detected SIGHT dataset format.")
#         return process_sight_format(df)
    
#     else:
#         raise Exception(f"Unknown dataset format. Expected either 'comment' column (SIGHT) or 'title'+'description' columns (course alignment)")

# def convert_course_data_to_sight_format(df):
#     """
#     Convert course alignment data to SIGHT-like format for testing.
#     This creates synthetic comment data from course descriptions.
#     """
#     processed_data = []
    
#     for idx, row in df.iterrows():
#         # Create synthetic "comments" from learning objectives
#         for i in range(1, 11):  # z1 to z10
#             z_col = f'z{i}'
#             score_col = f'z{i}_score'
            
#             if z_col in df.columns and score_col in df.columns:
#                 learning_objective = row[z_col]
#                 score = row[score_col]
                
#                 if pd.notna(learning_objective) and pd.notna(score):
#                     # Create a synthetic comment-like entry
#                     comment_data = {
#                         'comment_id': f"{idx}_{i}",
#                         'comment': f"This course on {row['title']} should cover: {learning_objective}",
#                         'video_name': row['title'],
#                         'playlist_name': row.get('domain', 'Unknown'),
#                         'datasplit': 'train'
#                     }
                    
#                     # Create dummy human annotations for testing
#                     # For now, randomly assign some categories based on the score
#                     categories = ['general', 'confusion', 'pedagogy', 'setup', 'gratitude', 
#                                 'personal_experience', 'clarification', 'non_english', 'na']
                    
#                     for category in categories:
#                         # Create some synthetic annotations for testing
#                         if category == 'general':
#                             # Higher scores get "general" positive feedback
#                             annotation = score >= 4
#                         elif category == 'confusion':
#                             # Lower scores get "confusion" 
#                             annotation = score <= 2
#                         elif category == 'pedagogy':
#                             # Medium scores get pedagogy comments
#                             annotation = score == 3
#                         else:
#                             # Random for other categories
#                             annotation = np.random.choice([True, False], p=[0.1, 0.9])
                        
#                         comment_data[f'H1_{category}'] = annotation
#                         comment_data[f'H2_{category}'] = annotation  # Same for both annotators
#                         comment_data[f'consensus_{category}'] = annotation
                    
#                     processed_data.append(comment_data)
    
#     result_df = pd.DataFrame(processed_data)
#     print(f"\nConverted to {len(result_df)} synthetic comments for SIGHT evaluation")
#     print("NOTE: This is synthetic data for testing. Real SIGHT evaluation needs actual YouTube comments.")
    
#     return result_df

# def process_sight_format(df):
#     """Process actual SIGHT dataset format."""
#     processed_data = []
    
#     # Map SIGHT category names to our internal names
#     sight_categories = {
#         'general': 'general',
#         'confusion': 'confusion', 
#         'pedagogy': 'pedagogy',
#         'setup': 'setup',
#         'gratitude': 'gratitude',
#         'personal_experience': 'personal_experience',
#         'clarification': 'clarification',
#         'non_english': 'non_english',
#         'na': 'na'
#     }
    
#     for idx, row in df.iterrows():
#         if pd.isna(row.get('comment', None)):
#             continue
            
#         comment_data = {
#             'comment_id': row.get('comment_id', idx),
#             'comment': row['comment'],
#             'video_id': row.get('video_id', ''),
#             'video_name': row.get('video_name', ''),
#             'playlist_name': row.get('playlist_name', ''),
#             'datasplit': row.get('datasplit', '')
#         }
        
#         # Extract human annotations (H1 and H2)
#         human_annotations = {}
#         for internal_cat, sight_cat in sight_categories.items():
#             h1_col = f'annotator_H1_{sight_cat}'
#             h2_col = f'annotator_H2_{sight_cat}'
            
#             h1_score = row.get(h1_col, np.nan)
#             h2_score = row.get(h2_col, np.nan)
            
#             # Convert to boolean (1.0 -> True, 0.0 -> False, NaN -> None)
#             human_annotations[f'H1_{internal_cat}'] = None if pd.isna(h1_score) else bool(h1_score)
#             human_annotations[f'H2_{internal_cat}'] = None if pd.isna(h2_score) else bool(h2_score)
            
#             # Consensus: both annotators agree on positive label
#             if not pd.isna(h1_score) and not pd.isna(h2_score):
#                 human_annotations[f'consensus_{internal_cat}'] = bool(h1_score) and bool(h2_score)
#             else:
#                 human_annotations[f'consensus_{internal_cat}'] = None
        
#         comment_data.update(human_annotations)
#         processed_data.append(comment_data)
    
#     result_df = pd.DataFrame(processed_data)
#     print(f"\nProcessed {len(result_df)} comments with annotations")
    
#     # Print annotation statistics
#     print("\nAnnotation statistics (H1):")
#     for category in sight_categories.keys():
#         h1_col = f'H1_{category}'
#         if h1_col in result_df.columns:
#             count = result_df[h1_col].sum() if result_df[h1_col].notna().any() else 0
#             total = result_df[h1_col].notna().sum()
#             if total > 0:
#                 print(f"  {category}: {count}/{total} ({count/total*100:.1f}%)")
#             else:
#                 print(f"  {category}: 0/0")
    
#     return result_df

# def get_comment_categories(row: pd.Series, annotator: str = 'consensus') -> List[str]:
#     """Extract categories assigned to a comment by specified annotator."""
#     categories = [
#         'general', 'confusion', 'pedagogy', 'setup', 'gratitude', 
#         'personal_experience', 'clarification', 'non_english', 'na'
#     ]
    
#     assigned_categories = []
#     for category in categories:
#         col_name = f'{annotator}_{category}'
#         if col_name in row and row[col_name] is True:
#             assigned_categories.append(category)
    
#     return assigned_categories

# def filter_annotated_comments(df: pd.DataFrame, min_annotations: int = 1, 
#                             annotator: str = 'consensus') -> pd.DataFrame:
#     """Filter to comments that have at least min_annotations."""
#     categories = [
#         'general', 'confusion', 'pedagogy', 'setup', 'gratitude', 
#         'personal_experience', 'clarification', 'non_english', 'na'
#     ]
    
#     annotation_counts = []
#     for idx, row in df.iterrows():
#         count = 0
#         for category in categories:
#             col_name = f'{annotator}_{category}'
#             if col_name in row and row[col_name] is True:
#                 count += 1
#         annotation_counts.append(count)
    
#     df['annotation_count'] = annotation_counts
#     filtered_df = df[df['annotation_count'] >= min_annotations].copy()
    
#     print(f"Filtered to {len(filtered_df)} comments with at least {min_annotations} annotations")
#     return filtered_df

