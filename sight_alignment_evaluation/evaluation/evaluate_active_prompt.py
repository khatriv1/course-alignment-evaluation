# sight_alignment_evaluation/evaluation/evaluate_active_prompt.py

import sys
import os
import pandas as pd
import openai
import time
from typing import Dict, List

# Add parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from prompting.active_prompt import get_active_prompt_prediction_all_categories, prepare_active_prompting_data, prepare_active_prompting_data_fallback
from utils.data_loader import load_and_preprocess_sight_data, get_comment_categories, filter_annotated_comments
from utils.metrics import calculate_agreement_metrics, plot_category_performance, print_detailed_results

def evaluate_active_prompt(data_path: str, api_key: str, output_dir: str = "results/active_prompt", limit: int = None):
    """Evaluate Active Prompting technique on SIGHT dataset with TRUE Active Prompting methodology."""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load and preprocess data
    try:
        df = load_and_preprocess_sight_data(data_path)
        if df.empty:
            raise Exception("No valid examples found in the data file")
        
        # Filter to comments with consensus annotations
        df = filter_annotated_comments(df, min_annotations=1, annotator='consensus')
        
        print(f"Loaded {len(df)} total comments with annotations")
        
        if limit:
            # Split data: use portion for uncertainty estimation, portion for evaluation
            uncertainty_size = min(30, max(10, len(df) // 4))  # REDUCED from min(1000, max(150, len(df) // 2))
            eval_size = min(limit, 15)  # Cap evaluation size for testing
            
            # Sample for uncertainty estimation (smaller pool)
            uncertainty_df = df.sample(n=uncertainty_size, random_state=42)
            
            # Sample for evaluation (separate from uncertainty pool if possible)
            remaining_df = df.drop(uncertainty_df.index) if len(df) > uncertainty_size + eval_size else df
            if len(remaining_df) >= eval_size:
                eval_df = remaining_df.sample(n=eval_size, random_state=43)
            else:
                eval_df = df.sample(n=eval_size, random_state=43)
                
            print(f"Using {len(uncertainty_df)} comments for uncertainty estimation")
            print(f"Evaluating on {len(eval_df)} comments")
        else:
            # Use smaller default sizes
            uncertainty_size = min(30, len(df))
            eval_size = min(10, len(df))
            
            uncertainty_df = df.sample(n=uncertainty_size, random_state=42)
            eval_df = df.sample(n=eval_size, random_state=43)
            
            print(f"Using {len(uncertainty_df)} comments for uncertainty estimation")
            print(f"Evaluating on {len(eval_df)} comments")
        
    except Exception as e:
        raise Exception(f"Error loading or processing data: {str(e)}")
    
    # Initialize OpenAI client
    client = openai.OpenAI(api_key=api_key)
    
    # SIGHT categories
    categories = [
        'general', 'confusion', 'pedagogy', 'setup', 'gratitude', 
        'personal_experience', 'clarification', 'non_english', 'na'
    ]
    
    print("\n" + "="*60)
    print("ACTIVE PROMPTING: UNCERTAINTY ESTIMATION PHASE")
    print("="*60)
    print("Following Diao et al. 2023 methodology:")
    print("  1. Uncertainty Estimation (k=3 samples, disagreement metric)")
    print("  2. Selection (top-N most uncertain examples)")  
    print("  3. Annotation (ground truth + reasoning)")
    print("  4. Inference (using selected examples)")
    print("="*60)
    
    # STAGE 1-3: Uncertainty Estimation, Selection, and Annotation
    try:
        uncertainty_examples = prepare_active_prompting_data(uncertainty_df, client, n_examples=3)
        print(f"✅ Active Prompting preparation completed with examples for {len(uncertainty_examples)} categories")
        
        # Show what was created
        for category, examples in uncertainty_examples.items():
            if examples:
                print(f"  {category}: {len(examples)} uncertain examples")
            else:
                print(f"  {category}: 0 examples (will use basic prompting)")
                
    except Exception as e:
        print(f"⚠️ Uncertainty estimation failed: {e}")
        print("⚠️ Falling back to random selection with ground truth...")
        
        try:
            uncertainty_examples = prepare_active_prompting_data_fallback(uncertainty_df, n_examples=3)
            print(f"✅ Fallback preparation completed with examples for {len(uncertainty_examples)} categories")
        except Exception as fallback_error:
            print(f"❌ Fallback also failed: {fallback_error}")
            print("❌ Using minimal default examples...")
            
            # Minimal fallback
            uncertainty_examples = {}
            for category in categories:
                uncertainty_examples[category] = []
    
    print("\n" + "="*60)
    print("ACTIVE PROMPTING: EVALUATION PHASE")
    print("="*60)
    
    # Store results
    human_labels = {}  # comment_id -> list of categories
    model_labels = {}  # comment_id -> list of categories
    detailed_results = []
    
    # STAGE 4: Inference with Selected Examples
    total = len(eval_df)
    successful_predictions = 0
    
    for seq, (_, row) in enumerate(eval_df.iterrows(), start=1):
        print(f"\nProcessing comment {seq}/{total}")
        print(f"Comment: {row['comment'][:100]}...")
        
        comment_id = str(row.get('comment_id', seq))
        
        try:
            # Get human annotations (consensus)
            human_cats = get_comment_categories(row, annotator='consensus')
            human_labels[comment_id] = human_cats
            
            # Get model predictions with Active Prompting using uncertainty-selected examples
            model_cats = get_active_prompt_prediction_all_categories(
                comment=row['comment'],
                video_name=row.get('video_name', ''),
                playlist_name=row.get('playlist_name', ''),
                client=client,
                uncertainty_examples=uncertainty_examples
            )
            model_labels[comment_id] = model_cats
            successful_predictions += 1
            
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
            print(f"❌ Error processing comment {comment_id}: {str(e)}")
            # Continue processing other comments
            continue
        
        time.sleep(0.5)  # Rate limiting between comments
    
    if not human_labels:
        raise Exception("No valid predictions were generated")
    
    print(f"\n✅ Successfully processed {successful_predictions}/{total} comments")
    
    # Calculate metrics
    metrics = calculate_agreement_metrics(human_labels, model_labels, categories)
    
    # Create visualization
    try:
        plot_category_performance(
            metrics, 
            categories, 
            'Active Prompting',
            f"{output_dir}/active_prompt_performance.png"
        )
        print(f"📊 Performance plot saved to {output_dir}/active_prompt_performance.png")
    except Exception as e:
        print(f"⚠️ Could not create plot: {e}")
    
    # Print results
    print_detailed_results(metrics, categories, 'Active Prompting')
    
    # Save detailed results
    results_df = pd.DataFrame(detailed_results)
    results_df.to_csv(f"{output_dir}/detailed_results.csv", index=False)
    print(f"💾 Detailed results saved to {output_dir}/detailed_results.csv")
    
    # Save metrics summary
    metrics_summary = {
        'technique': 'Active Prompting',
        'accuracy': metrics['accuracy'],
        'kappa': metrics['kappa'],
        'alpha': metrics['alpha'],
        'icc': metrics['icc'],
        'total_comments_processed': successful_predictions,
        'uncertainty_pool_size': len(uncertainty_df),
        'categories_with_uncertain_examples': len([c for c in categories if uncertainty_examples.get(c, [])])
    }
    
    summary_df = pd.DataFrame([metrics_summary])
    summary_df.to_csv(f"{output_dir}/metrics_summary.csv", index=False)
    print(f"📊 Metrics summary saved to {output_dir}/metrics_summary.csv")
    
    return detailed_results, metrics

if __name__ == "__main__":
    # Import config
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import config
    
    # Run evaluation
    try:
        print("🚀 Starting OPTIMIZED Active Prompting evaluation on SIGHT dataset...")
        print("📋 This implements the Diao et al. 2023 methodology with optimized parameters:")
        print("   • Reduced pool size (30 comments vs 150)")
        print("   • Fewer uncertainty samples (k=3 vs k=10)")  
        print("   • Testing 3 categories first (general, confusion, pedagogy)")
        print("   • Better progress tracking and error handling")
        print(f"📁 Using data file: {config.DATA_PATH}")
        
        # Ask user for limit
        try:
            limit_input = input("\n📝 Enter number of comments to evaluate (or press Enter for 10): ").strip()
            limit = int(limit_input) if limit_input else 10
            
            if limit > 20:
                print(f"⚠️  Reducing limit from {limit} to 20 for faster testing")
                limit = 20
                
        except ValueError:
            limit = 10
        
        print(f"\n🎯 Configuration:")
        print(f"   • Uncertainty pool: ~30 comments")
        print(f"   • Evaluation set: {limit} comments")
        print(f"   • Uncertainty samples: k=3 per comment")
        print(f"   • Expected API calls: ~270 for uncertainty + ~{limit * 9} for evaluation")
        print(f"   • Estimated time: 3-5 minutes")
        
        # Confirm before starting
        confirm = input("\n▶️  Continue? (y/n): ").strip().lower()
        if confirm not in ['y', 'yes', '']:
            print("❌ Cancelled")
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
        print("✅ OPTIMIZED ACTIVE PROMPTING EVALUATION COMPLETED!")
        print("="*60)
        print(f"⏱️  Total time: {duration:.1f} seconds")
        print(f"🎯 Accuracy: {metrics['accuracy']:.1%}")
        print(f"📊 Cohen's Kappa: {metrics['kappa']:.3f}")
        print(f"📈 Results saved in: results/active_prompt/")
        print("\n🔄 Next steps:")
        print("   • Review results in detailed_results.csv")
        print("   • Check uncertainty examples that were selected")
        print("   • Scale up to more categories and larger datasets")
        print("   • Compare with other prompting techniques")
        
    except KeyboardInterrupt:
        print("\n\n⏹️  Evaluation stopped by user")
        print("💡 You can restart with smaller parameters if needed")
        
    except Exception as e:
        print(f"\n❌ Error during evaluation: {str(e)}")
        print("\n🔧 Troubleshooting tips:")
        print("   • Check your OpenAI API key is valid")
        print("   • Ensure data file exists and has required columns")
        print("   • Try with smaller limit (e.g., 5 comments)")
        print("   • Check internet connection for API calls")
        
        import traceback
        print("\n📋 Full error traceback:")
        print(traceback.format_exc())