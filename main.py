import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from evaluation.evaluate_zero_shot import evaluate_zero_shot
from evaluation.evaluate_few_shot import evaluate_few_shot
from evaluation.evaluate_cot import evaluate_cot
from evaluation.evaluate_self_consistency import evaluate_self_consistency
from evaluation.evaluate_contrastive_cot import evaluate_contrastive_cot
from evaluation.evaluate_rar import evaluate_rar
from evaluation.evaluate_step_back import evaluate_step_back
from evaluation.evaluate_auto_cot import evaluate_auto_cot
import config

def create_comparison_visualization(comparison_df, output_dir):
    """Create comparison visualizations for all techniques."""
    # Use a default style
    plt.style.use('default')
    
    # Set figure size and colors
    plt.figure(figsize=(15, 10))  # Increased height to accommodate more techniques
    colors = ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c']
    
    # Convert all metrics to percentages
    comparison_df_percent = comparison_df.copy()
    for col in ['Agreement (Kappa)', 'Consistency (Alpha)', 'Correlation (ICC)']:
        comparison_df_percent[col] = comparison_df_percent[col] * 100

    # Prepare data
    metrics = [
        'Accuracy (%)',
        'Agreement (Kappa)',
        'Consistency (Alpha)',
        'Correlation (ICC)'
    ]
    
    techniques = comparison_df_percent['Technique'].tolist()
    x = np.arange(len(techniques))
    width = 0.18  # Reduced width to fit more bars
    multiplier = 0
    
    # Plot each metric group
    for idx, metric in enumerate(metrics):
        offset = width * multiplier
        plt.bar(x + offset, comparison_df_percent[metric], width,
                label=metric, color=colors[idx], alpha=0.8)
        multiplier += 1
    
    # Customize plot
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.xlabel('Prompting Technique', fontsize=12, fontweight='bold')
    plt.ylabel('Score (%)', fontsize=12, fontweight='bold')
    plt.title('Comparison of Prompting Techniques',
             fontsize=14, fontweight='bold', pad=20)
    
    # Set y-axis to percentage scale
    plt.ylim(-50, 100)  # Adjust range to accommodate negative values
    
    # Adjust x-ticks
    plt.xticks(x + width * 1.5, techniques, rotation=45, ha='right')  # Rotated labels for better readability
    plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
    
    # Add value labels
    for i in range(len(techniques)):
        for j, metric in enumerate(metrics):
            value = comparison_df_percent[metric].iloc[i]
            plt.text(i + width * j, value + (1 if value >= 0 else -3),
                    f'{value:.1f}%',
                    ha='center', va='bottom', fontweight='bold', fontsize=8)  # Reduced font size
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(f"{output_dir}/technique_comparison.png",
                dpi=300, bbox_inches='tight')
    plt.close()

def run_all_evaluations(data_path, api_key, output_dir="results", limit=None, techniques=None):
    """Run all prompting techniques and create comparison."""
    # Create output directory
    output_dir = os.path.join(output_dir, f"evaluation_report")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nResults will be saved in: {output_dir}")
    
    # Store results from each technique
    all_results = {}
    detailed_results = []
    
    # Use provided techniques or default to all if none provided
    if techniques is None:
        techniques = {
            'Zero-shot': evaluate_zero_shot,
            'Few-shot': evaluate_few_shot,
            'Chain of Thought': evaluate_cot,
            'Self-Consistency': evaluate_self_consistency,
            'Auto Chain of Thought': evaluate_auto_cot,
            'Contrastive Chain of Thought': evaluate_contrastive_cot,
            'Rephrase and Respond': evaluate_rar,
            'Take a Step Back': evaluate_step_back
        }
    
    # Process each technique
    for technique_name, evaluate_func in techniques.items():
        print(f"\nRunning {technique_name} evaluation...")
        
        # Create technique-specific directory
        technique_dir = os.path.join(output_dir, technique_name.lower().replace(' ', '_'))
        os.makedirs(technique_dir, exist_ok=True)
        
        # Run evaluation
        results, metrics = evaluate_func(data_path, api_key,
                                      output_dir=technique_dir, limit=limit)
        all_results[technique_name] = metrics
        
        # Collect detailed results
        results_df = pd.DataFrame(results['examples'])
        results_df['Technique'] = technique_name
        detailed_results.append(results_df)
    
    # Create comparison DataFrame
    comparison_df = pd.DataFrame({
        'Technique': list(all_results.keys()),
        'Accuracy (%)': [metrics['accuracy'] for metrics in all_results.values()],
        'Agreement (Kappa)': [metrics['kappa'] for metrics in all_results.values()],
        'Consistency (Alpha)': [metrics['alpha'] for metrics in all_results.values()],
        'Correlation (ICC)': [metrics['icc'] for metrics in all_results.values()]
    })
    
    # Save comparison results
    comparison_df.to_csv(f"{output_dir}/technique_comparison.csv", index=False)
    
    # Create comparison visualization
    create_comparison_visualization(comparison_df, output_dir)
    
    # Combine and save detailed results
    all_detailed_results = pd.concat(detailed_results, ignore_index=True)
    all_detailed_results.to_csv(f"{output_dir}/all_detailed_results.csv",
                              index=False)
    
    # Generate summary report
    with open(f"{output_dir}/summary_report.txt", 'w') as f:
        f.write("=== Prompting Techniques Comparison Summary ===\n\n")
        
        # Overall metrics
        metrics_df = comparison_df.copy()
        for col in ['Agreement (Kappa)', 'Consistency (Alpha)', 'Correlation (ICC)']:
            metrics_df[col] = metrics_df[col] * 100
        f.write("Overall Metrics Comparison (all values in %):\n")
        f.write(metrics_df.to_string())
        f.write("\n\n")
        
        # Best performing technique for each metric
        f.write("Best Performing Techniques:\n")
        for metric in ['Accuracy (%)', 'Agreement (Kappa)',
                      'Consistency (Alpha)', 'Correlation (ICC)']:
            best_technique = metrics_df.loc[metrics_df[metric].idxmax()]
            f.write(f"{metric}: {best_technique['Technique']} "
                   f"({best_technique[metric]:.1f}%)\n")
        
        # Additional statistics
        f.write("\nDetailed Statistics:\n")
        for technique in techniques.keys():
            technique_data = all_detailed_results[
                all_detailed_results['Technique'] == technique]
            f.write(f"\n{technique}:\n")
            f.write(f"Total examples: {len(technique_data)}\n")
            f.write(f"Average error: {technique_data['error'].mean():.2f}\n")
            f.write(f"Perfect matches: "
                   f"{(technique_data['human_score'] == technique_data['model_score']).sum()}\n")
        
        # Technique descriptions and references
        f.write("\n\n=== Technique Descriptions ===\n\n")
        f.write("Zero-shot: Provides only instructions and evaluation criteria without examples.\n")
        f.write("Few-shot: Includes a few examples to demonstrate the task.\n")
        f.write("Chain of Thought (CoT): Guides the model to produce a step-by-step reasoning process before giving the final answer.\n")
        f.write("Self-Consistency: Samples multiple reasoning paths and takes the most consistent answer.\n")
        f.write("Auto Chain of Thought (Auto-CoT): Automatically generates reasoning chains without human examples.\n")
        f.write("Contrastive Chain of Thought (CCoT): Provides both correct and incorrect reasoning paths to enhance accuracy.\n")
        f.write("Rephrase and Respond (RaR): First rephrases the problem to ensure clear understanding, then answers.\n")
        f.write("Take a Step Back: Approaches the problem from a more abstract level before detailed analysis.\n")
    
    print("\n=== Evaluation Complete ===")
    print(f"All results saved in: {output_dir}")
    print("\nGenerated files:")
    print("- technique_comparison.csv (Overall metrics)")
    print("- technique_comparison.png (Visualization)")
    print("- all_detailed_results.csv (Detailed predictions)")
    print("- summary_report.txt (Complete analysis)")
    print("Individual technique results in subdirectories")
    
    return comparison_df, all_detailed_results

if __name__ == "__main__":
    print("=== Course Alignment Evaluation Program ===")
    print(f"Using data file: {config.DATA_PATH}")
    
    # Ask user if they want to test with limited examples
    print("\nNOTE: A minimum of 10 examples is required for statistically valid results.")
    limit_input = input("Enter number of examples to test with (minimum 10, or press Enter for all): ")
    
    # Validate the minimum number of examples
    if limit_input.strip():
        try:
            limit = int(limit_input)
            if limit < 10:
                print("ERROR: You must evaluate at least 10 examples for statistically valid results.")
                print("Please run the program again with at least 10 examples.")
                sys.exit(1)
        except ValueError:
            print("Invalid input. Please enter a number or press Enter for all examples.")
            sys.exit(1)
    else:
        limit = None
    
    # Ask which techniques to evaluate
    print("\nAvailable techniques:")
    techniques_list = [
        "1. Zero-shot",
        "2. Few-shot",
        "3. Chain of Thought",
        "4. Self-Consistency",
        "5. Auto Chain of Thought",
        "6. Contrastive Chain of Thought",
        "7. Rephrase and Respond",
        "8. Take a Step Back",
        "9. All techniques"
    ]
    for technique in techniques_list:
        print(technique)
    
    # Improved error handling for technique selection
    technique_input = input("\nEnter the numbers of techniques to evaluate (comma-separated) or 9 for all: ")
    try:
        # Handle empty input gracefully
        if not technique_input.strip():
            selected_indices = [9]  # Default to all techniques
        else:
            selected_indices = [int(idx.strip()) for idx in technique_input.split(",") if idx.strip()]
        
        # Make sure we have at least one valid technique
        if not selected_indices:
            selected_indices = [9]  # Default to all techniques
    except ValueError:
        print("Invalid input. Defaulting to evaluating all techniques.")
        selected_indices = [9]  # Default to all techniques
    
    # Map evaluation functions (defining them outside the conditional blocks)
    evaluation_map = {
        "Zero-shot": evaluate_zero_shot,
        "Few-shot": evaluate_few_shot,
        "Chain of Thought": evaluate_cot,
        "Self-Consistency": evaluate_self_consistency,
        "Auto Chain of Thought": evaluate_auto_cot,
        "Contrastive Chain of Thought": evaluate_contrastive_cot,
        "Rephrase and Respond": evaluate_rar,
        "Take a Step Back": evaluate_step_back
    }
    
    # Run evaluations with improved error handling
    try:
        # If 9 is selected or list is empty, run all techniques
        if 9 in selected_indices:
            comparison_df, detailed_results = run_all_evaluations(
                data_path=config.DATA_PATH,
                api_key=config.OPENAI_API_KEY,
                limit=limit,
                techniques=evaluation_map  # Pass all techniques
            )
        else:
            # Map indices to technique names
            technique_map = {
                1: "Zero-shot",
                2: "Few-shot",
                3: "Chain of Thought",
                4: "Self-Consistency",
                5: "Auto Chain of Thought",
                6: "Contrastive Chain of Thought",
                7: "Rephrase and Respond",
                8: "Take a Step Back"
            }
            
            # Filter out any invalid indices
            valid_indices = [idx for idx in selected_indices if idx in technique_map]
            if not valid_indices:
                print("No valid techniques selected. Defaulting to all techniques.")
                comparison_df, detailed_results = run_all_evaluations(
                    data_path=config.DATA_PATH,
                    api_key=config.OPENAI_API_KEY,
                    limit=limit,
                    techniques=evaluation_map  # Pass all techniques
                )
            else:
                # Create filtered techniques dictionary
                selected_techniques = {}
                for idx in valid_indices:
                    technique_name = technique_map[idx]
                    if technique_name in evaluation_map:
                        selected_techniques[technique_name] = evaluation_map[technique_name]
                
                # Run evaluation with only the selected techniques
                comparison_df, detailed_results = run_all_evaluations(
                    data_path=config.DATA_PATH,
                    api_key=config.OPENAI_API_KEY,
                    limit=limit,
                    techniques=selected_techniques  # Pass only selected techniques
                )
            
        print("\nAll evaluations completed successfully!")
        
    except Exception as e:
        print(f"\nError during evaluation: {str(e)}")
        import traceback
        print("\nFull error traceback:")
        print(traceback.format_exc())