"""
BrainCEMISID Statistical Analysis Engine (Week 6)
-----------------------------------------------
This script performs statistical validation of the cognitive architecture
compared to a baseline LLM using t-Student tests, effect size, and visualizations.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def calculate_cohens_d(x, y):
    """Calculates Cohen's d effect size."""
    nx = len(x)
    ny = len(y)
    dof = nx + ny - 2
    return (np.mean(x) - np.mean(y)) / np.sqrt(((nx-1)*np.std(x, ddof=1)**2 + (ny-1)*np.std(y, ddof=1)**2) / dof)

def run_analysis(csv_path='analysis/metrics_summary.csv', output_dir='analysis/plots'):
    # 0. Setup
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    # 1. Load Data
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip() # Remove any whitespace
    
    # Define models
    brain_data = df[df['model_type'] == 'BrainCEMISID']
    
    print("--- BrainCEMISID Statistical Report ---")
    print(f"Total Samples (Rows): {len(df)}")
    
    # 2. Statistical Testing (t-Student) — all 9 metrics + latency
    metrics = {
        'coherence_score': 'Coherence',
        'planning_effectiveness': 'Planning',
        'behavioral_alignment': 'Behavioral Alignment',
        'fact_recall': 'Fact Recall',
        'rule_adherence': 'Rule Adherence',
        'persona_consistency': 'Persona Consistency',
        'emotional_trajectory': 'Emotional Trajectory',
        'sensory_integration': 'Sensory Integration',
        'decision_consistency': 'Decision Consistency',
        'latency_ms': 'Attention Latency'
    }
    
    comparison_models = [m for m in df['model_type'].unique() if m != 'BrainCEMISID']
    all_p_values = {}
    for comp_model in comparison_models:
        print("\n" + "="*50)
        print(f"COMPARISON: BrainCEMISID vs {comp_model}")
        print("="*50)
        comp_data = df[df['model_type'] == comp_model]
        
        results = {}
        for col, name in metrics.items():
            if col not in brain_data.columns or col not in comp_data.columns:
                print(f"\nMetric: {name} — SKIPPED (column '{col}' not found in data)")
                continue
            
            # Need variance in both arrays to run a t-test. If one is all 0, t-test can fail or be NaN.
            if len(brain_data[col].dropna()) < 2 or len(comp_data[col].dropna()) < 2:
                continue
                
            variance_brain = np.var(brain_data[col], ddof=1)
            variance_comp = np.var(comp_data[col], ddof=1)
            
            # If both have zero variance and identical means, it's not sig
            if variance_brain == 0 and variance_comp == 0:
                t_stat, p_val = 0.0, 1.0
                d = 0.0
            else:
                t_stat, p_val = stats.ttest_ind(brain_data[col], comp_data[col], equal_var=False)
                d = calculate_cohens_d(brain_data[col], comp_data[col])
                
            results[name] = {'t': t_stat, 'p': p_val, 'd': d}
            
            print(f"\nMetric: {name}")
            print(f"  Means: Brain={brain_data[col].mean():.2f} | {comp_model}={comp_data[col].mean():.2f}")
            print(f"  p-value: {p_val:.4f} ({'Significant' if p_val < 0.05 else 'Not Significant'})")
            print(f"  Effect Size (Cohen's d): {d:.2f}")

        all_p_values[comp_model] = {m: r['p'] for m, r in results.items()}

        # Auto Summary for Baseline
        if comp_model == 'Baseline':
            print("\n" + "*"*40)
            print(f"HYPOTHESIS SUMMARY (vs Baseline)")
            print("*"*40)
            
            sig_metrics = [m for m, r in results.items() if r['p'] < 0.05]
            brain_wins = [m for m, r in results.items() if r['p'] < 0.05 and r['d'] > 0]
            brain_ties = [m for m, r in results.items() if r['p'] >= 0.05 and m != 'Attention Latency']
            
            if sig_metrics:
                print(f"The primary hypothesis is SUPPORTED (p < 0.05 for {', '.join(sig_metrics)}).")
                print("BrainCEMISID shows scientific significance in cognitive performance over the Baseline LLM.")
                if brain_wins:
                    print(f"\n  🏆 Brain WINS on: {', '.join(brain_wins)}")
                if brain_ties:
                    print(f"  🤝 Tied (no sig. diff): {', '.join(brain_ties)}")
            else:
                print("The hypothesis is REJECTED (p > 0.05 across key metrics).")
                print("No significant performance difference was detected in this sample size.")
            print("*"*40)

    # 3. Context Retention (Memory Hits % over time)
    max_facts = 10 
    brain_data_sorted = brain_data.sort_values(by=['scenario_id', 'step_id'])
    brain_data_sorted['retention_pct'] = (brain_data_sorted['memory_hits'] / max_facts) * 100
    avg_retention = brain_data_sorted['retention_pct'].mean()
    print(f"\nContext Retention (Avg): {avg_retention:.1f}%")

    # 4. Visualizations
    sns.set_theme(style="whitegrid", palette="coolwarm")
    
    # Plot 1: Full 9-Metric Performance Comparison
    score_cols = [c for c in ['coherence_score', 'planning_effectiveness', 'behavioral_alignment',
                              'fact_recall', 'rule_adherence', 'persona_consistency',
                              'emotional_trajectory', 'sensory_integration', 'decision_consistency']
                  if c in df.columns]
    
    plt.figure(figsize=(16, 7))
    comparison_data = df.melt(id_vars=['model_type'], value_vars=score_cols, 
                               var_name='Metric', value_name='Score')
    label_map = {
        'coherence_score': 'Coherence', 
        'planning_effectiveness': 'Planning',
        'behavioral_alignment': 'Alignment',
        'fact_recall': 'Fact Recall',
        'rule_adherence': 'Rule Adherence',
        'persona_consistency': 'Persona',
        'emotional_trajectory': 'Emotion',
        'sensory_integration': 'Sensory',
        'decision_consistency': 'Decision'
    }
    comparison_data['Metric'] = comparison_data['Metric'].map(label_map)
    
    ax = sns.barplot(data=comparison_data, x='Metric', y='Score', hue='model_type', capsize=.1)
    plt.title("Performance Gap: BrainCEMISID vs Baseline (All Metrics)", fontsize=14, fontweight='bold')
    plt.ylabel("Score (Mean)")
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/performance_comparison.png")
    plt.close()

    # Plot 2: Behavioral Drift (Emotional Valence)
    plt.figure(figsize=(10, 6))
    for scenario in df['scenario_id'].unique():
        scenario_data = brain_data[brain_data['scenario_id'] == scenario]
        plt.plot(scenario_data['step_id'], scenario_data['emotional_valence'], marker='o', label=scenario)
    
    plt.title("Behavioral Drift: Emotional Evolution over Frames", fontsize=14, fontweight='bold')
    plt.xlabel("Simulation Frame (Time)")
    plt.ylabel("Emotional Valence (0.0 to 1.0)")
    plt.legend(title="Scenario")
    plt.savefig(f"{output_dir}/behavioral_drift.png")
    plt.close()

    # Plot 3: Sensory-Emotional Heatmap
    plt.figure(figsize=(10, 6))
    pivot_table = pd.pivot_table(brain_data, index="scenario_id", columns="step_id", values="emotional_valence", aggfunc="mean")
    sns.heatmap(pivot_table, annot=True, cmap="YlOrRd", cbar_kws={'label': 'Valence Intensity'})
    plt.title("Stimulus-Response Mapping (Heatmap)", fontsize=14, fontweight='bold')
    plt.savefig(f"{output_dir}/sensory_heatmap.png")
    plt.close()

    # Plot 4: Box and Whisker Plot
    plt.figure(figsize=(18, 8))
    sns.boxplot(data=comparison_data, x='Metric', y='Score', hue='model_type',
                palette={'BrainCEMISID': '#4C72B0', 'Baseline': '#C44E52', 
                         'Brain_No_Memory': '#55A868', 'Brain_No_Emotion': '#8172B3'})
    plt.title("Performance Variance and Consistency (Box Plots)", fontsize=16, fontweight='bold')
    plt.ylabel("Score")
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/performance_variance_box.png")
    plt.close()

    # Plot 5: Radar Chart
    # Calculate means
    radar_data = comparison_data.groupby(['model_type', 'Metric'])['Score'].mean().reset_index()
    radar_pivot = radar_data.pivot(index='model_type', columns='Metric', values='Score')
    categories = list(radar_pivot.columns)
    N = len(categories)
    
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    colors = {'BrainCEMISID': '#4C72B0', 'Baseline': '#C44E52', 'Brain_No_Memory': '#55A868', 'Brain_No_Emotion': '#8172B3'}
    
    for model in radar_pivot.index:
        values = radar_pivot.loc[model].values.flatten().tolist()
        values += values[:1]
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=model, color=colors.get(model, 'gray'))
        if model == 'BrainCEMISID':
            ax.fill(angles, values, color=colors[model], alpha=0.25)
            
    plt.xticks(angles[:-1], categories, size=11)
    ax.set_rlabel_position(0)
    plt.yticks([2, 4, 6, 8, 10], ["2", "4", "6", "8", "10"], color="grey", size=10)
    plt.ylim(0, 10)
    plt.title("Cognitive Profile Radar Chart", size=16, fontweight='bold', y=1.1)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()
    plt.savefig(f"{output_dir}/cognitive_radar.png")
    plt.close()

    # Plot 6: Latency vs. Performance Trade-off Scatter Plot
    # Calculate global performance (mean of all 9 metrics per frame)
    df['global_performance'] = df[score_cols].mean(axis=1)
    
    plt.figure(figsize=(12, 8))
    sns.scatterplot(data=df, x='latency_ms', y='global_performance', hue='model_type',
                    palette=colors, alpha=0.7, s=80)
    plt.title("Compute Trade-off: Latency vs. Global Performance", fontsize=16, fontweight='bold')
    plt.xlabel("Attention Latency (ms)")
    plt.ylabel("Global Coherence (Mean of 9 Metrics)")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/latency_tradeoff_scatter.png")
    plt.close()

    # Plot 7: P-Value Significance Matrix
    # Convert all_p_values to a DataFrame
    p_df = pd.DataFrame(all_p_values)
    if not p_df.empty:
        plt.figure(figsize=(10, 8))
        # Custom colormap: Green for significant (p < 0.05), Gray/White for > 0.05
        from matplotlib.colors import ListedColormap
        # We map p-values. Values < 0.05 will be green, >= 0.05 will be light gray.
        sns.heatmap(p_df, annot=True, cmap="Greens_r", vmin=0.0, vmax=0.05, 
                    cbar_kws={'label': 'P-Value (< 0.05 is Significant)'},
                    fmt=".4f", mask=(p_df >= 0.05))
        
        # Overlay the non-significant ones in gray
        sns.heatmap(p_df, annot=True, cmap=ListedColormap(['#DDDDDD']), vmin=0.0, vmax=1.0,
                    cbar=False, fmt=".4f", mask=(p_df < 0.05))
                    
        plt.title("Statistical Significance Matrix (vs BrainCEMISID)", fontsize=16, fontweight='bold')
        plt.xlabel("Comparison Model")
        plt.ylabel("Metric")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/p_value_matrix.png")
        plt.close()

    # Automated Conclusion (Moved inside loop)

if __name__ == "__main__":
    run_analysis()

