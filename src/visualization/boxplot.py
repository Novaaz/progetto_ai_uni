"""
Script semplice per boxplot da CSV nelle sottocartelle di plots.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime

def extract_and_group_data(results):
    """
    Estrae total_reward e raggruppa per tipo (s_/d_) e noise level.
    
    Returns:
        dict: {'static': {'0.05': [rewards]}, 'dynamic': {'0.10': [rewards]}}
    """
    grouped = {'static': {}, 'dynamic': {}}
    
    for result in results:
        name = result['name']
        reward = result['total_reward']
        
        # Determina tipo e noise level dal nome
        if name.startswith('s_'):
            model_type = 'static'
            noise = name.replace('s_', '').split('_')[0]  # s_0.05_xxx -> 0.05
        elif name.startswith('d_'):
            model_type = 'dynamic' 
            noise = name.replace('d_', '').split('_')[0]  # d_0.10_xxx -> 0.10
        else:
            continue
        
        # Aggiungi alla lista
        if noise not in grouped[model_type]:
            grouped[model_type][noise] = []
        grouped[model_type][noise].append(reward)
    
    return grouped

def calculate_boxplot_values(grouped_data):
    """
    Calcola min, q1, median, q3, max per ogni gruppo.
    """
    stats = {}
    
    for model_type, noise_dict in grouped_data.items():
        stats[model_type] = {}
        for noise, rewards in noise_dict.items():
            if rewards:
                values = np.array(rewards)
                stats[model_type][noise] = {
                    'min': np.min(values),
                    'q1': np.percentile(values, 25),
                    'median': np.percentile(values, 50), 
                    'q3': np.percentile(values, 75),
                    'max': np.max(values),
                    'values': rewards
                }
    
    return stats

def plot_boxplots(directory, results):
    """
    Crea boxplot separati per static e dynamic.
    """
    # Estrai e raggruppa
    grouped = extract_and_group_data(results)
    
    # Calcola le statistiche dei boxplot
    stats = calculate_boxplot_values(grouped)
    
    # Crea directory
    os.makedirs(directory, exist_ok=True)
    
    # Plot per static
    if grouped['static']:
        plt.figure(figsize=(10, 6))
        data_static = []
        for noise, rewards in grouped['static'].items():
            for reward in rewards:
                data_static.append({'Noise': noise, 'Reward': reward})
        
        df = pd.DataFrame(data_static)
        sns.boxplot(data=df, x='Noise', y='Reward')
        plt.title('Static Models - Performance by Noise Level')
        plt.savefig(os.path.join(directory, 'static_boxplot.png'))
        plt.close()
    
    # Plot per dynamic  
    if grouped['dynamic']:
        plt.figure(figsize=(10, 6))
        data_dynamic = []
        for noise, rewards in grouped['dynamic'].items():
            for reward in rewards:
                data_dynamic.append({'Noise': noise, 'Reward': reward})
        
        df = pd.DataFrame(data_dynamic)
        sns.boxplot(data=df, x='Noise', y='Reward')
        plt.title('Dynamic Models - Performance by Noise Level')
        plt.savefig(os.path.join(directory, 'dynamic_boxplot.png'))
        plt.close()
    
    # Salva le statistiche dei boxplot
    save_boxplot_stats_csv(stats, directory)
    
    print(f"Boxplot salvati in: {directory}")

def save_boxplot_stats_csv(stats, directory):
    """
    Salva le statistiche dei boxplot in un CSV con timestamp.
    """
    #timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"analysis.csv"
    filepath = os.path.join(directory, filename)
    
    # Converti stats in lista di righe per DataFrame
    rows = []
    for model_type in ['static', 'dynamic']:
        for noise_level, stat_data in stats[model_type].items():
            rows.append({
                'Model_Type': model_type,
                'Noise_Level': noise_level,
                'Min': stat_data['min'],
                'Q1': stat_data['q1'],
                'Median': stat_data['median'],
                'Q3': stat_data['q3'],
                'Max': stat_data['max'],
                'Count': len(stat_data['values'])
            })
    
    # Crea DataFrame e salva
    df = pd.DataFrame(rows)
    df.to_csv(filepath, index=False)
    print(f"Statistiche boxplot salvate in: {filepath}")
