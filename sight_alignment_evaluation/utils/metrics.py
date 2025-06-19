# sight_alignment_evaluation/utils/metrics.py

"""
Evaluation metrics for SIGHT comment classification.
Using the 4 specified metrics: Accuracy, Cohen's Kappa, Krippendorff's Alpha, ICC
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import cohen_kappa_score, accuracy_score
from scipy import stats
import krippendorff
from typing import Dict, List, Tuple

def calculate_agreement_metrics(human_labels: Dict[str, List[str]], 
                               model_labels: Dict[str, List[str]], 
                               categories: List[str]) -> Dict[str, float]:
    """
    Calculate the 4 specified metrics for SIGHT comment classification.
    
    Args:
        human_labels: Dict mapping comment_id to list of human-assigned categories
        model_labels: Dict mapping comment_id to list of model-assigned categories  
        categories: List of all possible categories
    
    Returns:
        Dictionary containing the 4 metrics
    """
    # Convert to binary matrices for calculations
    comment_ids = list(human_labels.keys())
    n_comments = len(comment_ids)
    n_categories = len(categories)
    
    human_matrix = np.zeros((n_comments, n_categories), dtype=int)
    model_matrix = np.zeros((n_comments, n_categories), dtype=int)
    
    category_to_idx = {cat: idx for idx, cat in enumerate(categories)}
    
    for i, comment_id in enumerate(comment_ids):
        # Human labels
        for category in human_labels[comment_id]:
            if category in category_to_idx:
                human_matrix[i, category_to_idx[category]] = 1
        
        # Model labels  
        for category in model_labels[comment_id]:
            if category in category_to_idx:
                model_matrix[i, category_to_idx[category]] = 1
    
    # Flatten for overall metrics
    human_flat = human_matrix.flatten()
    model_flat = model_matrix.flatten()
    
    # 1. ACCURACY - Exact match accuracy
    exact_matches = np.all(human_matrix == model_matrix, axis=1)
    accuracy = np.mean(exact_matches) * 100
    
    # 2. COHEN'S KAPPA (κ) - Agreement beyond chance
    kappa = cohen_kappa_score(human_flat, model_flat)
    
    # 3. KRIPPENDORFF'S ALPHA (α) - Reliability measure
    # Prepare data for Krippendorff's alpha
    data = np.array([human_flat, model_flat])
    alpha = krippendorff.alpha(data, level_of_measurement='nominal')
    
    # 4. INTRACLASS CORRELATION (ICC) - Correlation between scores
    # Using ICC(2,1) - two-way random effects, single measurement, absolute agreement
    # Reshape data for ICC calculation
    ratings = np.column_stack([human_flat, model_flat])
    
    # Calculate ICC using the formula for ICC(2,1)
    n_items = len(human_flat)
    mean_human = np.mean(human_flat)
    mean_model = np.mean(model_flat)
    
    # Between-subject variance
    subject_means = np.mean(ratings, axis=1)
    ms_between = n_items * np.var(subject_means, ddof=1) * 2
    
    # Within-subject variance
    ms_within = np.mean(np.var(ratings, axis=1, ddof=1))
    
    # Error variance
    ms_error = ms_within
    
    # ICC calculation
    if ms_between > 0:
        icc = (ms_between - ms_error) / (ms_between + ms_error)
    else:
        icc = 0.0
    
    # Per-category metrics
    category_metrics = {}
    for i, category in enumerate(categories):
        human_cat = human_matrix[:, i]
        model_cat = model_matrix[:, i]
        
        if len(np.unique(human_cat)) > 1 and len(np.unique(model_cat)) > 1:
            cat_accuracy = accuracy_score(human_cat, model_cat) * 100
            cat_kappa = cohen_kappa_score(human_cat, model_cat)
            
            # Category-specific Krippendorff's alpha
            cat_data = np.array([human_cat, model_cat])
            cat_alpha = krippendorff.alpha(cat_data, level_of_measurement='nominal')
            
            # Category-specific correlation
            cat_corr = np.corrcoef(human_cat, model_cat)[0, 1]
            
        else:
            cat_accuracy = np.mean(human_cat == model_cat) * 100
            cat_kappa = 0.0
            cat_alpha = 0.0
            cat_corr = 0.0
        
        category_metrics[category] = {
            'accuracy': cat_accuracy,
            'kappa': cat_kappa,
            'alpha': cat_alpha,
            'correlation': cat_corr
        }
    
    return {
        'accuracy': accuracy,
        'kappa': kappa,
        'alpha': alpha,
        'icc': icc,
        'category_metrics': category_metrics
    }


def plot_category_performance(metrics: Dict[str, float], 
                            categories: List[str], 
                            technique_name: str, 
                            save_path: str = None):
    """
    Create visualization of per-category performance using the 4 metrics.
    """
    category_metrics = metrics['category_metrics']
    
    # Prepare data for plotting
    metric_names = ['Accuracy', 'Cohen\'s κ', 'Krippendorff\'s α', 'Correlation']
    metric_keys = ['accuracy', 'kappa', 'alpha', 'correlation']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, (metric_name, metric_key) in enumerate(zip(metric_names, metric_keys)):
        scores = []
        for cat in categories:
            if metric_key == 'accuracy':
                scores.append(category_metrics[cat][metric_key] / 100)  # Convert to 0-1 scale
            else:
                scores.append(category_metrics[cat][metric_key])
        
        bars = axes[i].bar(categories, scores, alpha=0.7)
        axes[i].set_title(f'{metric_name} by Category', fontsize=12, fontweight='bold')
        axes[i].set_ylabel(metric_name)
        axes[i].set_ylim(-0.1, 1.1 if metric_key != 'accuracy' else 110)
        axes[i].tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, score in zip(bars, scores):
            height = bar.get_height()
            if metric_key == 'accuracy':
                axes[i].text(bar.get_x() + bar.get_width()/2., height + 1,
                            f'{score*100:.1f}%', ha='center', va='bottom', fontsize=9)
            else:
                axes[i].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                            f'{score:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.suptitle(f'Category Performance: {technique_name}', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.close()


def print_detailed_results(metrics: Dict[str, float], 
                         categories: List[str], 
                         technique_name: str):
    """
    Print detailed results summary with the 4 metrics.
    """
    print(f"\n=== {technique_name} Results ===")
    print(f"Overall Metrics:")
    print(f"  Accuracy: {metrics['accuracy']:.1f}%")
    print(f"  Cohen's Kappa (κ): {metrics['kappa']:.3f}")
    print(f"  Krippendorff's Alpha (α): {metrics['alpha']:.3f}")
    print(f"  Intraclass Correlation (ICC): {metrics['icc']:.3f}")
    
    print("\nPer-Category Results:")
    category_metrics = metrics['category_metrics']
    for category in categories:
        cat_metrics = category_metrics[category]
        print(f"  {category:20s}: Accuracy={cat_metrics['accuracy']:.1f}%, "
              f"κ={cat_metrics['kappa']:.3f}, "
              f"α={cat_metrics['alpha']:.3f}, "
              f"Corr={cat_metrics['correlation']:.3f}")
    
    # Overall interpretation based on Kappa
    kappa = metrics['kappa']
    print(f"\nOverall Agreement Level (κ={kappa:.3f}): ", end="")
    if kappa > 0.8:
        print("Almost Perfect Agreement")
    elif kappa > 0.6:
        print("Substantial Agreement")  
    elif kappa > 0.4:
        print("Moderate Agreement")
    elif kappa > 0.2:
        print("Fair Agreement")
    elif kappa > 0:
        print("Slight Agreement")
    else:
        print("Poor Agreement")


# Keep the original function name for compatibility but use new metrics
def calculate_multilabel_metrics(human_labels: Dict[str, List[str]], 
                                model_labels: Dict[str, List[str]], 
                                categories: List[str]) -> Dict[str, float]:
    """
    Calculate metrics for multi-label classification.
    This now calculates the 4 specified metrics.
    """
    return calculate_agreement_metrics(human_labels, model_labels, categories)