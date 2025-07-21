# course_alignment_evaluation/evaluation/evaluate_active_prompt.py
# MINIMAL VERSION: Fixed imports + reduced parameters

import sys
import os
import pandas as pd
import openai
import time
import numpy as np

# Add parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Fixed imports to match the available functions
from prompting.active_prompt import get_active_prompt_prediction, prepare_active_prompting_data
from utils.metrics import calculate_agreement_metrics, plot_simple_metrics, print_simple_summary

def load_and_preprocess_data(file_path):
    """Load and preprocess human score data"""
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

def evaluate_active_prompt(data_path, api_key, output_dir="results/active_prompt", limit=None):
    """Evaluate Active Prompting technique with MINIMAL parameters"""
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load and preprocess data
    try:
        df = load_and_preprocess_data(data_path)
        if df.empty:
            raise Exception("No valid examples found in the data file")
        
        print(f"Loaded {len(df)} total course-objective pairs")
        
        if limit:
            # REDUCED uncertainty pool size
            uncertainty_size = min(20, max(10, len(df) // 4))  # REDUCED from min(500, max(50, len(df) // 2))
            eval_size = min(limit, 15)  # Cap evaluation size
            
            # Sample for uncertainty estimation (smaller pool)
            uncertainty_df = df.sample(n=uncertainty_size, random_state=42)
            
            # Sample for evaluation
            remaining_df = df.drop(uncertainty_df.index) if len(df) > uncertainty_size + eval_size else df
            if len(remaining_df) >= eval_size:
                eval_df = remaining_df.sample(n=eval_size, random_state=43)
            else:
                eval_df = df.sample(n=eval_size, random_state=43)
                
            print(f"Using {len(uncertainty_df)} pairs for uncertainty estimation (REDUCED)")
            print(f"Evaluating on {len(eval_df)} pairs")
        else:
            # Default small sizes for testing
            uncertainty_size = min(20, len(df))
            eval_size = min(10, len(df))
            
            uncertainty_df = df.sample(n=uncertainty_size, random_state=42)
            eval_df = df.sample(n=eval_size, random_state=43)
            
            print(f"Using {len(uncertainty_df)} pairs for uncertainty estimation")
            print(f"Evaluating on {len(eval_df)} pairs")
            
    except Exception as e:
        raise Exception(f"Error loading or processing data: {str(e)}")
    
    # Initialize OpenAI client
    client = openai.OpenAI(api_key=api_key)
    
    print("\n" + "="*60)
    print("ACTIVE PROMPTING: UNCERTAINTY ESTIMATION PHASE (MINIMAL)")
    print("="*60)
    print("Optimized parameters:")
    print(f"  • Pool size: {len(uncertainty_df)} pairs")
    print(f"  • k_samples: 2 per pair")
    print(f"  • Examples: 3 total")
    print(f"  • Expected API calls: ~{len(uncertainty_df) * 2} for uncertainty")
    print("="*60)
    
    # STAGE 1-3: Uncertainty Estimation, Selection, and Annotation
    try:
        uncertainty_examples = prepare_active_prompting_data(uncertainty_df, client, n_examples=3)  # REDUCED from 5 to 3
        print(f" Active Prompting preparation completed with {len(uncertainty_examples)} examples")
        
        # Show what was created
        if uncertainty_examples:
            print(f"  Created examples from uncertain pairs:")
            for i, ex in enumerate(uncertainty_examples[:3]):
                print(f"    {i+1}. Score {ex['score']} - {ex['course'][:30]}...")
        
    except Exception as e:
        print(f" Uncertainty estimation failed: {e}")
        print(" Using fallback method...")
        
        # Fallback: create examples from random selection
        sample_pairs = uncertainty_df.sample(n=min(3, len(uncertainty_df)), random_state=42)
        uncertainty_examples = []
        
        for _, row in sample_pairs.iterrows():
            # Create reasoning based on score
            score = row['human_score']
            if score >= 4:
                reasoning = "Strong alignment - course content directly supports the objective."
            elif score == 3:
                reasoning = "Moderate alignment - transferable skills but not direct coverage."
            elif score == 2:
                reasoning = "Weak alignment - same domain but limited relevance."
            else:
                reasoning = "Poor alignment - minimal connection between course and objective."
            
            uncertainty_examples.append({
                'course': row['course_title'],
                'description': row['description'],
                'objective': row['learning_objective'],
                'reasoning': reasoning,
                'score': score
            })
        
        print(f" Fallback preparation completed with {len(uncertainty_examples)} examples")
    
    print("\n" + "="*60)
    print("ACTIVE PROMPTING: EVALUATION PHASE (MINIMAL)")
    print("="*60)
    
    # Store results
    results = {
        'human_scores': [],
        'model_scores': [],
        'examples': []
    }
    
    # STAGE 4: Inference with Selected Examples
    total = len(eval_df)
    successful_predictions = 0
    
    for idx, (_, row) in enumerate(eval_df.iterrows(), start=1):
        print(f"\nProcessing example {idx}/{total}")
        print(f"Course: {row['course_title'][:50]}...")
        print(f"Objective: {row['learning_objective'][:60]}...")
        
        try:
            # Get model prediction using Active Prompting
            score = get_active_prompt_prediction(
                row['course_title'],
                row['description'],
                row['learning_objective'],
                client,
                uncertainty_examples=uncertainty_examples
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
                successful_predictions += 1
                print(f"Human: {row['human_score']}, Model: {score}")
            else:
                print(" Failed to get valid prediction")
            
        except Exception as e:
            print(f" Error processing example: {str(e)}")
            continue
        
        time.sleep(0.3)  # Faster rate limiting
    
    if not results['human_scores']:
        raise Exception("No valid predictions were generated")
    
    print(f"\n✅ Successfully processed {successful_predictions}/{total} pairs")
    
    # Calculate metrics
    metrics = calculate_agreement_metrics(
        results['human_scores'],
        results['model_scores']
    )
    
    # Create visualization
    try:
        plot_simple_metrics(
            results['human_scores'],
            results['model_scores'],
            'Active Prompting (MINIMAL)',
            f"{output_dir}/active_prompt_metrics.png"
        )
        print(f" Performance plot saved to {output_dir}/active_prompt_metrics.png")
    except Exception as e:
        print(f" Could not create plot: {e}")
    
    # Print results summary
    print_simple_summary(metrics, 'Active Prompting (MINIMAL)')
    
    # Save detailed results
    results_df = pd.DataFrame(results['examples'])
    results_df.to_csv(f"{output_dir}/detailed_results.csv", index=False)
    print(f" Detailed results saved to {output_dir}/detailed_results.csv")
    
    # Save metrics summary
    metrics_summary = {
        'technique': 'Active Prompting (MINIMAL)',
        'accuracy': metrics['accuracy'],
        'kappa': metrics['kappa'],
        'alpha': metrics['alpha'],
        'icc': metrics['icc'],
        'uncertainty_examples_used': len(uncertainty_examples),
        'uncertainty_pool_size': len(uncertainty_df),
        'successful_predictions': successful_predictions
    }
    
    summary_df = pd.DataFrame([metrics_summary])
    summary_df.to_csv(f"{output_dir}/metrics_summary.csv", index=False)
    print(f"📊 Metrics summary saved to {output_dir}/metrics_summary.csv")
    
    return results, metrics

if __name__ == "__main__":
    # Import config from parent directory
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import config
    
    # Run evaluation
    try:
        print(" Starting MINIMAL Active Prompting evaluation on Course Alignment dataset...")
        print(" Optimized for fast testing:")
        print("   • Small uncertainty pool (20 pairs)")
        print("   • Fewer samples (k=2 vs k=10)")
        print("   • Fewer examples (3 vs 5)")
        print("   • Simplified prompts")
        print(f" Using data file: {config.DATA_PATH}")
        
        # Ask user for limit
        try:
            limit_input = input("\n Enter number of pairs to evaluate (or press Enter for 10): ").strip()
            limit = int(limit_input) if limit_input else 10
            
            if limit > 15:
                print(f" Reducing limit from {limit} to 15 for faster testing")
                limit = 15
                
        except ValueError:
            limit = 10
        
        print(f"\n🎯 Configuration:")
        print(f"   • Uncertainty pool: ~20 pairs")
        print(f"   • Evaluation set: {limit} pairs")
        print(f"   • Expected API calls: ~40 for uncertainty + ~{limit} for evaluation")
        print(f"   • Estimated time: 2-3 minutes")
        
        # Confirm before starting
        confirm = input("\nContinue? (y/n): ").strip().lower()
        if confirm not in ['y', 'yes', '']:
            print("Cancelled")
            exit()
        
        start_time = time.time()
        
        results, metrics = evaluate_active_prompt(
            data_path=config.DATA_PATH,
            api_key=config.OPENAI_API_KEY,
            limit=limit
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        print("\n" + "="*60)
        print(" MINIMAL ACTIVE PROMPTING EVALUATION COMPLETED!")
        print("="*60)
        print(f" Total time: {duration:.1f} seconds")
        print(f" Accuracy: {metrics['accuracy']:.1%}")
        print(f" Cohen's Kappa: {metrics['kappa']:.3f}")
        print(f" Results saved in: results/active_prompt/")
        
    except KeyboardInterrupt:
        print("\n\n Evaluation stopped by user")
        
    except Exception as e:
        print(f"\n Error during evaluation: {str(e)}")
        print("\n Troubleshooting tips:")
        print("   • Check your OpenAI API key")
        print("   • Ensure data file exists")
        print("   • Try with smaller limit (e.g., 5)")
        
        import traceback
        print("\n📋 Full error traceback:")
        print(traceback.format_exc())