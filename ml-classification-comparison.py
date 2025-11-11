"""
Comparing Linear Regression vs Logistic Regression
Based on: Introduction to Data Science by Rafael Irizarry
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

def make_data(n=1000, p=0.5, mu_0=0, mu_1=2, sigma_0=1, sigma_1=1):
    """
    Generate synthetic binary classification data
    
    Parameters:
    -----------
    n : int, number of samples
    p : float, probability of class 1
    mu_0 : float, mean of class 0
    mu_1 : float, mean of class 1
    sigma_0 : float, standard deviation of class 0
    sigma_1 : float, standard deviation of class 1
    
    Returns:
    --------
    dict with 'train' and 'test' DataFrames
    """
    # Generate binary outcomes
    y = np.random.binomial(1, p, n)
    
    # Generate features from two normal distributions
    f_0 = np.random.normal(mu_0, sigma_0, n)
    f_1 = np.random.normal(mu_1, sigma_1, n)
    
    # Assign features based on class
    x = np.where(y == 1, f_1, f_0)
    
    # Create dataframe
    df = pd.DataFrame({'x': x, 'y': y})
    
    # Split into train and test (50-50 split)
    train_df, test_df = train_test_split(df, test_size=0.5, 
                                         stratify=df['y'], 
                                         random_state=42)
    
    return {'train': train_df, 'test': test_df}


def compare_models(data):
    """
    Compare Linear Regression vs Logistic Regression
    
    Parameters:
    -----------
    data : dict with 'train' and 'test' DataFrames
    
    Returns:
    --------
    dict with accuracies for both models
    """
    # Extract train and test data
    X_train = data['train'][['x']].values
    y_train = data['train']['y'].values
    X_test = data['test'][['x']].values
    y_test = data['test']['y'].values
    
    # Linear Regression
    lin_reg = LinearRegression()
    lin_reg.fit(X_train, y_train)
    y_pred_lin = lin_reg.predict(X_test)
    # Convert to binary predictions (threshold at 0.5)
    y_pred_lin_binary = (y_pred_lin > 0.5).astype(int)
    acc_linear = accuracy_score(y_test, y_pred_lin_binary)
    
    # Logistic Regression
    log_reg = LogisticRegression()
    log_reg.fit(X_train, y_train)
    y_pred_log = log_reg.predict(X_test)
    acc_logistic = accuracy_score(y_test, y_pred_log)
    
    return {
        'linear': acc_linear,
        'logistic': acc_logistic,
        'linear_model': lin_reg,
        'logistic_model': log_reg
    }


# ============================================================================
# EXERCISE 1: Compare accuracy of linear regression and logistic regression
# ============================================================================
print("=" * 70)
print("EXERCISE 1: Single Dataset Comparison")
print("=" * 70)

# Generate data
dat = make_data()

# Visualize the data
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Density plot by class
for class_label in [0, 1]:
    data_class = dat['train'][dat['train']['y'] == class_label]['x']
    axes[0].hist(data_class, bins=30, alpha=0.5, density=True, 
                 label=f'Class {class_label}')
axes[0].set_xlabel('x')
axes[0].set_ylabel('Density')
axes[0].set_title('Distribution of x by Class (Training Data)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Plot 2: Scatter plot
colors = ['red' if y == 0 else 'blue' for y in dat['train']['y']]
axes[1].scatter(dat['train']['x'], dat['train']['y'], c=colors, alpha=0.5)
axes[1].set_xlabel('x')
axes[1].set_ylabel('y')
axes[1].set_title('Training Data: x vs y')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/exercise1_data_visualization.png', dpi=300, bbox_inches='tight')
print("\n✓ Data visualization saved")

# Compare models
results = compare_models(dat)

print(f"\nModel Comparison Results:")
print(f"  Linear Regression Accuracy:   {results['linear']:.4f}")
print(f"  Logistic Regression Accuracy: {results['logistic']:.4f}")
print(f"  Difference:                   {abs(results['linear'] - results['logistic']):.4f}")


# ============================================================================
# EXERCISE 2: Repeat simulation 100 times
# ============================================================================
print("\n" + "=" * 70)
print("EXERCISE 2: 100 Simulations")
print("=" * 70)

n_simulations = 100
accuracies_linear = []
accuracies_logistic = []

for i in range(n_simulations):
    # Use different random seed for each simulation
    np.random.seed(i)
    data = make_data()
    results = compare_models(data)
    accuracies_linear.append(results['linear'])
    accuracies_logistic.append(results['logistic'])

# Calculate statistics
mean_linear = np.mean(accuracies_linear)
mean_logistic = np.mean(accuracies_logistic)
std_linear = np.std(accuracies_linear)
std_logistic = np.std(accuracies_logistic)

print(f"\nResults from {n_simulations} simulations:")
print(f"  Linear Regression:")
print(f"    Mean Accuracy: {mean_linear:.4f} ± {std_linear:.4f}")
print(f"  Logistic Regression:")
print(f"    Mean Accuracy: {mean_logistic:.4f} ± {std_logistic:.4f}")
print(f"  Average Difference: {abs(mean_linear - mean_logistic):.4f}")

# Visualize results
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Histogram comparison
axes[0].hist(accuracies_linear, bins=20, alpha=0.6, label='Linear Regression', color='blue')
axes[0].hist(accuracies_logistic, bins=20, alpha=0.6, label='Logistic Regression', color='red')
axes[0].axvline(mean_linear, color='blue', linestyle='--', linewidth=2, label=f'Mean Linear: {mean_linear:.3f}')
axes[0].axvline(mean_logistic, color='red', linestyle='--', linewidth=2, label=f'Mean Logistic: {mean_logistic:.3f}')
axes[0].set_xlabel('Accuracy')
axes[0].set_ylabel('Frequency')
axes[0].set_title(f'Distribution of Accuracies ({n_simulations} Simulations)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Plot 2: Scatter plot
axes[1].scatter(accuracies_linear, accuracies_logistic, alpha=0.5)
axes[1].plot([0.5, 1], [0.5, 1], 'r--', label='Perfect Agreement')
axes[1].set_xlabel('Linear Regression Accuracy')
axes[1].set_ylabel('Logistic Regression Accuracy')
axes[1].set_title('Linear vs Logistic Regression Accuracy')
axes[1].legend()
axes[1].grid(True, alpha=0.3)
axes[1].axis('equal')

plt.tight_layout()
plt.savefig('outputs/exercise2_simulation_results.png', dpi=300, bbox_inches='tight')
print("\n✓ Simulation results visualization saved")


# ============================================================================
# EXERCISE 3: Vary the difference between classes (delta)
# ============================================================================
print("\n" + "=" * 70)
print("EXERCISE 3: Accuracy vs Delta (Class Separation)")
print("=" * 70)

delta_values = np.linspace(0, 3, 25)
n_reps = 50  # Number of repetitions for each delta

results_by_delta = {
    'delta': [],
    'linear_mean': [],
    'linear_std': [],
    'logistic_mean': [],
    'logistic_std': []
}

print(f"\nRunning {len(delta_values)} different deltas with {n_reps} repetitions each...")
print(f"Total simulations: {len(delta_values) * n_reps}")

for delta in delta_values:
    acc_linear_temp = []
    acc_logistic_temp = []
    
    for rep in range(n_reps):
        np.random.seed(rep * 100 + int(delta * 100))
        # mu_0 = 0, mu_1 = mu_0 + delta
        data = make_data(mu_0=0, mu_1=delta)
        results = compare_models(data)
        acc_linear_temp.append(results['linear'])
        acc_logistic_temp.append(results['logistic'])
    
    results_by_delta['delta'].append(delta)
    results_by_delta['linear_mean'].append(np.mean(acc_linear_temp))
    results_by_delta['linear_std'].append(np.std(acc_linear_temp))
    results_by_delta['logistic_mean'].append(np.mean(acc_logistic_temp))
    results_by_delta['logistic_std'].append(np.std(acc_logistic_temp))

print("✓ Simulations complete")

# Create comprehensive visualization
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Plot 1: Mean accuracy vs delta
axes[0, 0].plot(results_by_delta['delta'], results_by_delta['linear_mean'], 
                'o-', label='Linear Regression', linewidth=2, markersize=6)
axes[0, 0].plot(results_by_delta['delta'], results_by_delta['logistic_mean'], 
                's-', label='Logistic Regression', linewidth=2, markersize=6)
axes[0, 0].axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Random Guess')
axes[0, 0].set_xlabel('Delta (μ₁ - μ₀)', fontsize=12)
axes[0, 0].set_ylabel('Mean Accuracy', fontsize=12)
axes[0, 0].set_title('Mean Accuracy vs Class Separation', fontsize=14, fontweight='bold')
axes[0, 0].legend(fontsize=10)
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].set_ylim([0.45, 1.05])

# Plot 2: Accuracy with error bands
axes[0, 1].plot(results_by_delta['delta'], results_by_delta['linear_mean'], 
                'b-', label='Linear Regression', linewidth=2)
axes[0, 1].fill_between(results_by_delta['delta'], 
                        np.array(results_by_delta['linear_mean']) - np.array(results_by_delta['linear_std']),
                        np.array(results_by_delta['linear_mean']) + np.array(results_by_delta['linear_std']),
                        alpha=0.3, color='blue')
axes[0, 1].plot(results_by_delta['delta'], results_by_delta['logistic_mean'], 
                'r-', label='Logistic Regression', linewidth=2)
axes[0, 1].fill_between(results_by_delta['delta'], 
                        np.array(results_by_delta['logistic_mean']) - np.array(results_by_delta['logistic_std']),
                        np.array(results_by_delta['logistic_mean']) + np.array(results_by_delta['logistic_std']),
                        alpha=0.3, color='red')
axes[0, 1].set_xlabel('Delta (μ₁ - μ₀)', fontsize=12)
axes[0, 1].set_ylabel('Accuracy', fontsize=12)
axes[0, 1].set_title('Accuracy with Standard Deviation Bands', fontsize=14, fontweight='bold')
axes[0, 1].legend(fontsize=10)
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].set_ylim([0.45, 1.05])

# Plot 3: Difference between methods
difference = np.array(results_by_delta['logistic_mean']) - np.array(results_by_delta['linear_mean'])
axes[1, 0].plot(results_by_delta['delta'], difference, 'go-', linewidth=2, markersize=6)
axes[1, 0].axhline(y=0, color='black', linestyle='-', alpha=0.5)
axes[1, 0].set_xlabel('Delta (μ₁ - μ₀)', fontsize=12)
axes[1, 0].set_ylabel('Accuracy Difference\n(Logistic - Linear)', fontsize=12)
axes[1, 0].set_title('Performance Difference Between Methods', fontsize=14, fontweight='bold')
axes[1, 0].grid(True, alpha=0.3)

# Plot 4: Standard deviation comparison
axes[1, 1].plot(results_by_delta['delta'], results_by_delta['linear_std'], 
                'o-', label='Linear Regression', linewidth=2, markersize=6)
axes[1, 1].plot(results_by_delta['delta'], results_by_delta['logistic_std'], 
                's-', label='Logistic Regression', linewidth=2, markersize=6)
axes[1, 1].set_xlabel('Delta (μ₁ - μ₀)', fontsize=12)
axes[1, 1].set_ylabel('Standard Deviation of Accuracy', fontsize=12)
axes[1, 1].set_title('Variability of Accuracy vs Class Separation', fontsize=14, fontweight='bold')
axes[1, 1].legend(fontsize=10)
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/exercise3_accuracy_vs_delta.png', dpi=300, bbox_inches='tight')
print("\n✓ Accuracy vs Delta visualization saved")

# Summary statistics for key delta values
print("\nKey Findings:")
print("-" * 70)
for idx in [0, 12, 24]:  # delta = 0, 1.5, 3
    delta = results_by_delta['delta'][idx]
    lin_acc = results_by_delta['linear_mean'][idx]
    log_acc = results_by_delta['logistic_mean'][idx]
    print(f"\nDelta = {delta:.2f}:")
    print(f"  Linear Regression:   {lin_acc:.4f} ± {results_by_delta['linear_std'][idx]:.4f}")
    print(f"  Logistic Regression: {log_acc:.4f} ± {results_by_delta['logistic_std'][idx]:.4f}")
    print(f"  Difference:          {log_acc - lin_acc:.4f}")


# ============================================================================
# SAVE RESULTS TO CSV
# ============================================================================
results_df = pd.DataFrame(results_by_delta)
results_df.to_csv('outputs/exercise3_results.csv', index=False)
print("\n" + "=" * 70)
print("✓ Results saved to CSV file")
print("=" * 70)

# Print final summary
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print("\n1. Single Dataset (Exercise 1):")
print(f"   Both methods perform similarly with small differences")
print(f"\n2. Multiple Simulations (Exercise 2):")
print(f"   Average accuracies are nearly identical:")
print(f"   - Linear: {mean_linear:.4f}, Logistic: {mean_logistic:.4f}")
print(f"\n3. Varying Class Separation (Exercise 3):")
print(f"   - When delta = 0 (no separation): ~50% accuracy (random)")
print(f"   - As delta increases: accuracy approaches 100%")
print(f"   - Both methods converge to similar performance")
print("\nConclusion:")
print("Linear and logistic regression perform practically the same for this")
print("simple 1D binary classification problem, especially when classes are")
print("well-separated. The difference between methods is negligible in practice.")
print("=" * 70)
