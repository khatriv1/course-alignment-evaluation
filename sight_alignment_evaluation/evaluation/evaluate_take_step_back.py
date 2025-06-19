# sight_alignment_evaluation/evaluation/evaluate_take_step_back.py

import sys
import os
import pandas as pd
import openai
import time
from typing import Dict, List

# Add parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from prompting.take_a_step_back import get_take_step_back_prediction_all_categories
from utils.data_loader import load_and_preprocess_sight_data, get_comment_categories, filter_annotated_comments
from utils.metrics import  calculate_agreement_metrics, plot_category_performance, print_detailed_results

def evaluate_take_step_back(data_path: str, api_key: str, output_dir: str = "results/take_step_back", limit: int = None):
    """Evaluate Take a Step Back prompting technique on SIGHT dataset."""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load and preprocess data
    try:
        df = load_and_preprocess_sight_data(data_path)
        if df.empty:
            raise Exception("No valid examples found in the data file")
        
        # Filter to comments with consensus annotations
        df = filter_annotated_comments(df, min_annotations=1, annotator='consensus')
        
        if limit:
            df = df.head(limit)
        print(f"\nEvaluating on {len(df)} comments")
        
    except Exception as e:
        raise Exception(f"Error loading or processing data: {str(e)}")
    
    # Initialize OpenAI client
    client = openai.OpenAI(api_key=api_key)
    
    # SIGHT categories
    categories = [
        'general', 'confusion', 'pedagogy', 'setup', 'gratitude', 
        'personal_experience', 'clarification', 'non_english', 'na'
    ]
    
    # Store results
    human_labels = {}  # comment_id -> list of categories
    model_labels = {}  # comment_id -> list of categories
    detailed_results = []
    
    # Process each comment
    total = len(df)
    for seq, (_, row) in enumerate(df.iterrows(), start=1):
        print(f"\nProcessing comment {seq}/{total}")
        print(f"Comment: {row['comment'][:100]}...")
        
        comment_id = str(row['comment_id'])
        
        try:
            # Get human annotations (consensus)
            human_cats = get_comment_categories(row, annotator='consensus')
            human_labels[comment_id] = human_cats
            
            # Get model predictions with Take a Step Back
            model_cats = get_take_step_back_prediction_all_categories(
                comment=row['comment'],
                video_name=row.get('video_name', ''),
                playlist_name=row.get('playlist_name', ''),
                client=client
            )
            model_labels[comment_id] = model_cats
            
            # Store detailed result
            detailed_results.append({
                'comment_id': comment_id,
                'comment': row['comment'],
                'video_name': row.get('video_name', ''),
                'playlist_name': row.get('playlist_name', ''),
                'human_categories': ', '.join(human_cats),
                'model_categories': ', '.join(model_cats),
                'exact_match': set(human_cats) == set(model_cats)
            })
            
            print(f"Human: {human_cats}")
            print(f"Model: {model_cats}")
            print(f"Match: {set(human_cats) == set(model_cats)}")
            
        except Exception as e:
            print(f"Error processing comment {comment_id}: {str(e)}")
            continue
        
        time.sleep(1)  # Rate limiting
    
    if not human_labels:
        raise Exception("No valid predictions were generated")
    
    # Calculate metrics
    metrics =  calculate_agreement_metrics(human_labels, model_labels, categories)
    
    # Create visualization
    plot_category_performance(
        metrics, 
        categories, 
        'Take a Step Back',
        f"{output_dir}/take_step_back_performance.png"
    )
    
    # Print results
    print_detailed_results(metrics, categories, 'Take a Step Back')
    
    # Save detailed results
    results_df = pd.DataFrame(detailed_results)
    results_df.to_csv(f"{output_dir}/detailed_results.csv", index=False)
    print(f"\nDetailed results saved to {output_dir}/detailed_results.csv")
    
    # Save metrics summary
    metrics_summary = {
        'technique': 'Technique Name',
        'accuracy': metrics['accuracy'],
        'kappa': metrics['kappa'],
        'alpha': metrics['alpha'],
        'icc': metrics['icc']
    }
    
    summary_df = pd.DataFrame([metrics_summary])
    summary_df.to_csv(f"{output_dir}/metrics_summary.csv", index=False)
    
    return detailed_results, metrics

if __name__ == "__main__":
    # Import config
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import config
    
    # Run evaluation
    try:
        print("\nStarting Take a Step Back evaluation on SIGHT dataset...")
        print(f"Using data file: {config.DATA_PATH}")
        
        results, metrics = evaluate_take_step_back(
            data_path=config.DATA_PATH,
            api_key=config.OPENAI_API_KEY,
            limit=10  # Set to small number for testing
        )
        print("\nEvaluation completed successfully!")
        
    except Exception as e:
        print(f"\nError during evaluation: {str(e)}")
        import traceback
        print("\nFull error traceback:")
        print(traceback.format_exc())