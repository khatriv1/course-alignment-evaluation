import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import cohen_kappa_score, confusion_matrix
from scipy import stats
import krippendorff

def calculate_agreement_metrics(human_scores, model_scores):
    """Calculate basic metrics and agreement scores."""
    human_scores = np.array(human_scores)
    model_scores = np.array(model_scores)
    
    # Basic accuracy (simple to understand)
    accuracy = np.mean(human_scores == model_scores) * 100  # Convert to percentage
    
    # Agreement metrics
    kappa = cohen_kappa_score(human_scores, model_scores, weights='linear')
    
    # Krippendorff's alpha
    data = np.array([human_scores, model_scores])
    alpha = krippendorff.alpha(data, level_of_measurement='ordinal')
    
    # ICC
    icc = stats.pearsonr(human_scores, model_scores)[0]
    
    return {
        'accuracy': accuracy,
        'kappa': kappa,
        'alpha': alpha,
        'icc': icc
    }

def plot_simple_metrics(human_scores, model_scores, technique_name, save_path=None):
    """Create a simple visualization of metrics that's easy to understand."""
    metrics = calculate_agreement_metrics(human_scores, model_scores)
    
    plt.figure(figsize=(12, 6))
    
    # Define metrics to plot with friendly names
    metrics_to_plot = [
        ('Accuracy', metrics['accuracy'], '%'),
        ('Agreement (κ)', metrics['kappa'] * 100, '%'),
        ('Consistency (α)', metrics['alpha'] * 100, '%'),
        ('Correlation (ICC)', metrics['icc'] * 100, '%')
    ]
    
    # Create bar plot
    names, values, units = zip(*metrics_to_plot)
    colors = ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c']
    
    bars = plt.bar(names, [abs(v) for v in values], color=colors)
    
    # Customize the plot
    plt.title(f'Performance Metrics: {technique_name}', pad=20, fontsize=14, fontweight='bold')
    plt.ylabel('Score (%)', fontsize=12)
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = abs(value)
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.1f}%',
                ha='center', va='bottom')
    
    # Add explanations below the chart
    explanations = [
        "Accuracy: How often the model gives exactly the same score as humans",
        "Agreement (κ): How well the model agrees with humans, accounting for chance",
        "Consistency (α): How reliably the model matches human scoring patterns",
        "Correlation (ICC): How closely the model's scores track with human scores"
    ]
    
    plt.figtext(0.1, -0.15, '\n'.join(explanations), fontsize=10, ha='left')
    
    # Adjust layout to make room for explanations
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def print_simple_summary(metrics, technique_name):
    """Print a simple, easy-to-understand summary of the metrics."""
    print(f"\n=== Summary for {technique_name} ===")
    print(f"\nAccuracy: {metrics['accuracy']:.1f}%")
    print("(How often the model exactly matches human scores)")
    
    print(f"\nAgreement (κ): {metrics['kappa']*100:.1f}%")
    print("(How well the model agrees with humans beyond chance)")
    
    print(f"\nConsistency (α): {metrics['alpha']*100:.1f}%")
    print("(How reliably the model follows human scoring patterns)")
    
    print(f"\nCorrelation (ICC): {metrics['icc']*100:.1f}%")
    print("(How closely model scores align with human scores)")
    
    # Overall interpretation
    avg_performance = (metrics['kappa'] + metrics['alpha'] + metrics['icc']) / 3
    print("\nOverall Performance:", end=" ")
    if avg_performance > 0.8:
        print("Excellent")
    elif avg_performance > 0.6:
        print("Good")
    elif avg_performance > 0.4:
        print("Fair")
    else:
        print("Needs Improvement")

# testing 