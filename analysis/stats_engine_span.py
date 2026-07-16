"""
BrainCEMISID Motor de Análisis Estadístico (Semana 6) - Versión en Español
-----------------------------------------------
Este script realiza la validación estadística de la arquitectura cognitiva
en comparación con un LLM de referencia usando pruebas t-Student, tamaño del efecto y visualizaciones.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def calculate_cohens_d(x, y):
    """Calcula el tamaño del efecto (d de Cohen)."""
    nx = len(x)
    ny = len(y)
    dof = nx + ny - 2
    return (np.mean(x) - np.mean(y)) / np.sqrt(((nx-1)*np.std(x, ddof=1)**2 + (ny-1)*np.std(y, ddof=1)**2) / dof)

def run_analysis(csv_path='analysis/metrics_summary.csv', output_dir='analysis/plots_es'):
    # 0. Configuración inicial
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    if not os.path.exists(csv_path):
        print(f"Error: No se encontró {csv_path}.")
        return

    # 1. Cargar Datos
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip() # Limpiar espacios en blanco
    
    # Definir modelos
    brain_data = df[df['model_type'] == 'BrainCEMISID']
    
    print("--- Reporte Estadístico BrainCEMISID ---")
    print(f"Muestras Totales (Filas): {len(df)}")
    
    # 2. Pruebas Estadísticas (t-Student) — las 9 métricas + latencia
    metrics = {
        'coherence_score': 'Coherencia',
        'planning_effectiveness': 'Planificacion',
        'behavioral_alignment': 'Alineacion Conductual',
        'fact_recall': 'Recuperación de hechos',
        'rule_adherence': 'Adherencia a Reglas',
        'persona_consistency': 'Consistencia de Persona',
        'emotional_trajectory': 'Trayectoria Emocional',
        'sensory_integration': 'Integracion Sensorial',
        'decision_consistency': 'Consistencia de Decision',
        'latency_ms': 'Latencia de Atencion'
    }
    
    comparison_models = [m for m in df['model_type'].unique() if m != 'BrainCEMISID']
    all_p_values = {}
    for comp_model in comparison_models:
        print("\n" + "="*50)
        print(f"COMPARACIÓN: BrainCEMISID vs {comp_model}")
        print("="*50)
        comp_data = df[df['model_type'] == comp_model]
        
        results = {}
        for col, name in metrics.items():
            if col not in brain_data.columns or col not in comp_data.columns:
                print(f"\nMétrica: {name} — OMITIDA (columna '{col}' no encontrada en datos)")
                continue
            
            if len(brain_data[col].dropna()) < 2 or len(comp_data[col].dropna()) < 2:
                continue
                
            variance_brain = np.var(brain_data[col], ddof=1)
            variance_comp = np.var(comp_data[col], ddof=1)
            
            if variance_brain == 0 and variance_comp == 0:
                t_stat, p_val = 0.0, 1.0
                d = 0.0
            else:
                t_stat, p_val = stats.ttest_ind(brain_data[col], comp_data[col], equal_var=False)
                d = calculate_cohens_d(brain_data[col], comp_data[col])
                
            results[name] = {'t': t_stat, 'p': p_val, 'd': d}
            
            print(f"\nMétrica: {name}")
            print(f"  Medias: Brain={brain_data[col].mean():.2f} | {comp_model}={comp_data[col].mean():.2f}")
            print(f"  Estadístico p: {p_val:.4f} ({'Significativo' if p_val < 0.05 else 'No Significativo'})")
            print(f"  Tamaño de Efecto (d de Cohen): {d:.2f}")

        all_p_values[comp_model] = {m: r['p'] for m, r in results.items()}

        if comp_model == 'Baseline':
            print("\n" + "*"*40)
            print(f"RESUMEN DE HIPÓTESIS (vs Baseline)")
            print("*"*40)
            
            sig_metrics = [m for m, r in results.items() if r['p'] < 0.05]
            brain_wins = [m for m, r in results.items() if r['p'] < 0.05 and r['d'] > 0]
            brain_ties = [m for m, r in results.items() if r['p'] >= 0.05 and m != 'Latencia de Atencion']
            
            if sig_metrics:
                print(f"La hipótesis primaria está RESPALDADA (p < 0.05 para {', '.join(sig_metrics)}).")
                print("BrainCEMISID muestra significancia científica en rendimiento cognitivo sobre el Baseline LLM.")
                if brain_wins:
                    print(f"\n  🏆 Brain GANA en: {', '.join(brain_wins)}")
                if brain_ties:
                    print(f"  🤝 Empate (sin dif. sig): {', '.join(brain_ties)}")
            else:
                print("La hipótesis está RECHAZADA (p > 0.05).")
            print("*"*40)

    # 3. Retención de Contexto (Hits de Memoria % en el tiempo)
    max_facts = 10 
    brain_data_sorted = brain_data.sort_values(by=['scenario_id', 'step_id'])
    brain_data_sorted['retention_pct'] = (brain_data_sorted['memory_hits'] / max_facts) * 100
    avg_retention = brain_data_sorted['retention_pct'].mean()
    print(f"\nRetención de Contexto (Promedio): {avg_retention:.1f}%")

    # 4. Visualizaciones en Español
    sns.set_theme(style="whitegrid", palette="coolwarm")
    
    score_cols = [c for c in ['coherence_score', 'planning_effectiveness', 'behavioral_alignment',
                              'fact_recall', 'rule_adherence', 'persona_consistency',
                              'emotional_trajectory', 'sensory_integration', 'decision_consistency']
                  if c in df.columns]
    
    label_map = {
        'coherence_score': 'Coherencia',
        'planning_effectiveness': 'Planificacion',
        'behavioral_alignment': 'Alineacion Conductual',
        'fact_recall': 'Recuperación de hechos',
        'rule_adherence': 'Adherencia a Reglas',
        'persona_consistency': 'Consistencia de Persona',
        'emotional_trajectory': 'Trayectoria Emocional',
        'sensory_integration': 'Integracion Sensorial',
        'decision_consistency': 'Consistencia de Decision'
    }

    colors = {'BrainCEMISID': '#4C72B0', 'Baseline': '#C44E52', 'Brain_No_Memory': '#55A868', 'Brain_No_Emotion': '#8172B3'}

    # Gráfico Nuevo: SOLO BrainCEMISID vs Baseline (Medias de las Métricas Cognitivas)
    plt.figure(figsize=(14, 7))
    subset_df = df[df['model_type'].isin(['BrainCEMISID', 'Baseline'])]
    subset_melted = subset_df.melt(id_vars=['model_type'], value_vars=score_cols, 
                               var_name='Métrica', value_name='Puntuación')
    subset_melted['Métrica'] = subset_melted['Métrica'].map(label_map)
    
    sns.barplot(data=subset_melted, x='Métrica', y='Puntuación', hue='model_type', capsize=.1, palette={'BrainCEMISID': '#4C72B0', 'Baseline': '#C44E52'})
    plt.title("Comparación de Rendimiento Promedio: BrainCEMISID vs Baseline", fontsize=16, fontweight='bold')
    plt.ylabel("Puntuación Promedio (0-10)")
    plt.xlabel("Métricas de Evaluación")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/rendimiento_promedio_brain_vs_baseline.png")
    plt.close()

    # Gráfico 1: Comparación Completa de Rendimiento (Todos los modelos)
    plt.figure(figsize=(16, 7))
    comparison_data = df.melt(id_vars=['model_type'], value_vars=score_cols, 
                               var_name='Métrica', value_name='Puntuación')
    comparison_data['Métrica'] = comparison_data['Métrica'].map(label_map)
    
    ax = sns.barplot(data=comparison_data, x='Métrica', y='Puntuación', hue='model_type', capsize=.1, palette=colors)
    plt.title("Brecha de Rendimiento: BrainCEMISID vs Variantes (Todas las Métricas)", fontsize=14, fontweight='bold')
    plt.ylabel("Puntuación Promedio")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/comparacion_rendimiento_total.png")
    plt.close()

    # Gráfico 2: Evolución Emocional
    plt.figure(figsize=(10, 6))
    for scenario in df['scenario_id'].unique():
        scenario_data = brain_data[brain_data['scenario_id'] == scenario]
        plt.plot(scenario_data['step_id'], scenario_data['emotional_valence'], marker='o', label=scenario)
    
    plt.title("Deriva Conductual: Evolución Emocional en el Tiempo", fontsize=14, fontweight='bold')
    plt.xlabel("Marco de Simulación (Tiempo)")
    plt.ylabel("Valencia Emocional (0.0 a 1.0)")
    plt.legend(title="Escenario")
    plt.savefig(f"{output_dir}/deriva_conductual.png")
    plt.close()

    # Gráfico 3: Mapa de Calor Sensorial-Emocional
    plt.figure(figsize=(10, 6))
    pivot_table = pd.pivot_table(brain_data, index="scenario_id", columns="step_id", values="emotional_valence", aggfunc="mean")
    sns.heatmap(pivot_table, annot=True, cmap="YlOrRd", cbar_kws={'label': 'Intensidad de Valencia'})
    plt.title("Mapeo Estímulo-Respuesta (Mapa de Calor)", fontsize=14, fontweight='bold')
    plt.savefig(f"{output_dir}/mapa_calor_sensorial.png")
    plt.close()

    # Gráfico 4: Gráfico de Cajas
    plt.figure(figsize=(18, 8))
    sns.boxplot(data=comparison_data, x='Métrica', y='Puntuación', hue='model_type', palette=colors)
    plt.title("Varianza y Consistencia del Rendimiento (Diagramas de Caja)", fontsize=16, fontweight='bold')
    plt.ylabel("Puntuación")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/varianza_rendimiento_cajas.png")
    plt.close()

    # Gráfico 5: Gráfico de Radar
    radar_data = comparison_data[comparison_data['model_type'].isin(['BrainCEMISID', 'Baseline'])].groupby(['model_type', 'Métrica'])['Puntuación'].mean().reset_index()
    radar_pivot = radar_data.pivot(index='model_type', columns='Métrica', values='Puntuación')
    categories = list(radar_pivot.columns)
    N = len(categories)
    
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    for model in radar_pivot.index:
        values = radar_pivot.loc[model].values.flatten().tolist()
        values += values[:1]
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=model, color=colors.get(model, 'gray'))
        if model == 'BrainCEMISID':
            ax.fill(angles, values, color=colors[model], alpha=0.25)
            
    plt.xticks(angles[:-1], categories, size=11, rotation=45)
    ax.set_rlabel_position(0)
    plt.yticks([2, 4, 6, 8, 10], ["2", "4", "6", "8", "10"], color="grey", size=10)
    plt.ylim(0, 10)
    plt.title("Gráfico de Radar del Perfil Cognitivo", size=16, fontweight='bold', y=1.1)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()
    plt.savefig(f"{output_dir}/radar_cognitivo.png")
    plt.close()

    # Gráfico 6: Trade-off de Latencia vs Rendimiento
    df['rendimiento_global'] = df[score_cols].mean(axis=1)
    
    plt.figure(figsize=(12, 8))
    sns.scatterplot(data=df, x='latency_ms', y='rendimiento_global', hue='model_type',
                    palette=colors, alpha=0.7, s=80)
    plt.title("Trade-off Computacional: Latencia vs. Rendimiento Global", fontsize=16, fontweight='bold')
    plt.xlabel("Latencia de Atención (ms)")
    plt.ylabel("Coherencia Global (Promedio de 9 Métricas)")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/dispersion_tradeoff_latencia.png")
    plt.close()

    # Gráfico 7: Matriz de Significancia P-Value
    p_df = pd.DataFrame(all_p_values)
    if not p_df.empty:
        plt.figure(figsize=(12, 8))
        from matplotlib.colors import ListedColormap
        sns.heatmap(p_df, annot=True, cmap="Greens_r", vmin=0.0, vmax=0.05, 
                    cbar_kws={'label': 'Valor-P (< 0.05 es Significativo)'},
                    fmt=".4f", mask=(p_df >= 0.05))
        
        sns.heatmap(p_df, annot=True, cmap=ListedColormap(['#DDDDDD']), vmin=0.0, vmax=1.0,
                    cbar=False, fmt=".4f", mask=(p_df < 0.05))
                    
        plt.title("Matriz de Significancia Estadística (vs BrainCEMISID)", fontsize=16, fontweight='bold')
        plt.xlabel("Modelo de Comparación")
        plt.ylabel("Métrica")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/matriz_valores_p.png")
        plt.close()

    # Gráfico Nuevo: Gráfico de Barras de Latencia
    plt.figure(figsize=(10, 6))
    subset_latency = df[df['model_type'].isin(['BrainCEMISID', 'Baseline'])]
    latency_data = subset_latency.groupby('model_type')['latency_ms'].mean().reset_index()
    ax = sns.barplot(data=latency_data, x='model_type', y='latency_ms', palette={'BrainCEMISID': '#4C72B0', 'Baseline': '#C44E52'})
    plt.title("Comparación de Latencia Promedio: BrainCEMISID vs Baseline", fontsize=16, fontweight='bold')
    plt.ylabel("Latencia (Milisegundos)")
    plt.xlabel("Modelo")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/comparacion_latencia_barras.png")
    plt.close()

if __name__ == "__main__":
    run_analysis()
