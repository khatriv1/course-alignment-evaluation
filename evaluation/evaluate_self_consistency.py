import sys
import os
import pandas as pd
import openai
import time
import numpy as np

# Add parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Now we can import from sibling directories
from prompting.self_consistency import get_self_consistency_prediction
from utils.metrics import calculate_agreement_metrics, plot_simple_metrics, print_simple_summary

def load_and_preprocess_data(file_path):
    """Load and preprocess human score data."""
    print(f"Loading data from: {file_path}")
    
    # Read CSV file
    df = pd.read_csv(file_path)
    print(f"Loaded {len(df)} rows")
    
    # Create a list to store transformed data
    transformed_data = []
    
    # Process each row
    for idx, row in df.iterrows():
        course_title = row['title']
        description = row['description']
        
        # Process each z-column and its corresponding score
        for i in range(1, 11):  # z1 to z10
            z_col = f'z{i}'
            score_col = f'z{i}_score'
            
            if z_col in df.columns and score_col in df.columns:
                learning_objective = row[z_col]
                human_score = row[score_col]
                
                if pd.notna(learning_objective) and pd.notna(human_score):
                    transformed_data.append({
                        'course_title': course_title,
                        'description': description,
                        'learning_objective': learning_objective,
                        'human_score': int(human_score)
                    })
    
    # Convert to DataFrame
    result_df = pd.DataFrame(transformed_data)
    print(f"\nProcessed {len(result_df)} learning objectives")
    
    return result_df

def evaluate_self_consistency(data_path, api_key, output_dir="results/self_consistency", limit=None):
    """Evaluate Self-Consistency prompting technique."""
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load and preprocess data
    try:
        df = load_and_preprocess_data(data_path)
        if df.empty:
            raise Exception("No valid examples found in the data file")
            
        if limit:
            df = df.head(limit)
        print(f"\nProcessing {len(df)} examples")
        
    except Exception as e:
        raise Exception(f"Error loading or processing data: {str(e)}")
    
    # Initialize OpenAI client
    client = openai.OpenAI(api_key=api_key)
    
    # Store results
    results = {
        'human_scores': [],
        'model_scores': [],
        'examples': []
    }
    
    # Process each example
    total = len(df)
    for idx, row in df.iterrows():
        print(f"\nProcessing example {idx + 1}/{total}")
        print(f"Course: {row['course_title']}")
        print(f"Learning Objective: {row['learning_objective'][:100]}...")
        
        try:
            # Get model prediction (without showing human score)
            score = get_self_consistency_prediction(
                row['course_title'],
                row['description'],
                row['learning_objective'],
                client,
                num_samples=3  # Use 3 samples for efficiency, can be adjusted
            )
            
            if score is not None:
                results['human_scores'].append(row['human_score'])
                results['model_scores'].append(score)
                results['examples'].append({
                    'course_title': row['course_title'],
                    'learning_objective': row['learning_objective'],
                    'human_score': row['human_score'],
                    'model_score': score,
                    'error': abs(row['human_score'] - score)
                })
                print(f"Human score: {row['human_score']}, Model score: {score}")
            else:
                print("Failed to get valid prediction for this example")
            
        except Exception as e:
            print(f"Error processing example: {str(e)}")
            continue
        
        time.sleep(1)  # Rate limiting
    
    if not results['human_scores']:
        raise Exception("No valid predictions were generated")
    
    # Calculate metrics
    metrics = calculate_agreement_metrics(
        results['human_scores'],
        results['model_scores']
    )
    
    # Create visualization
    plot_simple_metrics(
        results['human_scores'],
        results['model_scores'],
        'Self-Consistency',
        f"{output_dir}/self_consistency_metrics.png"
    )
    
    # Print results summary
    print_simple_summary(metrics, 'Self-Consistency')
    
    # Save detailed results
    results_df = pd.DataFrame(results['examples'])
    results_df.to_csv(f"{output_dir}/detailed_results.csv", index=False)
    print(f"\nDetailed results saved to {output_dir}/detailed_results.csv")
    
    return results, metrics

if __name__ == "__main__":
    # Import config from parent directory
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import config
    
    # Run evaluation
    try:
        print("\nStarting Self-Consistency evaluation...")
        print(f"Using data file: {config.DATA_PATH}")
        
        results, metrics = evaluate_self_consistency(
            data_path=config.DATA_PATH,
            api_key=config.OPENAI_API_KEY,
            limit=5  # Set to a number like 5 for testing
        )
        print("\nEvaluation completed successfully!")
        
    except Exception as e:
        print(f"\nError during evaluation: {str(e)}")
        import traceback
        print("\nFull error traceback:")
        print(traceback.format_exc())