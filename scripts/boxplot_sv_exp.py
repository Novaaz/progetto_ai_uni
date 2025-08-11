"""
Script per visualizzare e confrontare le performance dei modelli single-variable.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob  # ← AGGIUNTO QUESTO IMPORT
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.constants import *

def load_test_results(csv_path=None):
    """
    Carica i risultati dei test da CSV.
    """
    if csv_path is None:
        # Cerca l'ultimo file di test
        csv_dir = os.path.join(RESULTS_DIR, CSV_DIR)
        test_files = glob.glob(os.path.join(csv_dir, "single_var_test_*.csv"))
        if not test_files:
            raise FileNotFoundError("Nessun file di test trovato!")
        csv_path = max(test_files, key=os.path.getctime)
        print(f"📂 Caricando: {os.path.basename(csv_path)}")
    
    df = pd.read_csv(csv_path)
    # Filtra solo risultati validi
    df = df[df['total_reward'] > -999999]
    print(f"✅ Caricati {len(df)} risultati validi")
    return df

def plot_performance_overview(df, save_dir):
    """
    1. OVERVIEW GENERALE - Boxplot per variabile
    """
    plt.figure(figsize=(16, 10))
    
    # Ordina per reward medio
    var_means = df.groupby('variable_name')['total_reward'].mean().sort_values(ascending=False)
    ordered_vars = var_means.index.tolist()
    
    # Boxplot
    sns.boxplot(data=df, x='variable_name', y='total_reward', order=ordered_vars)
    plt.xticks(rotation=45, ha='right')
    plt.title('Performance per Variabile (Boxplot)', fontsize=16, fontweight='bold')
    plt.xlabel('Variabile', fontsize=12)
    plt.ylabel('Total Reward', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Aggiungi statistiche
    for i, var in enumerate(ordered_vars):
        var_data = df[df['variable_name'] == var]['total_reward']
        mean_val = var_data.mean()
        plt.text(i, mean_val, f'{mean_val:.0f}', ha='center', va='bottom', 
                fontweight='bold', fontsize=10, color='red')
    
    plt.savefig(os.path.join(save_dir, 'overview_boxplot.png'), dpi=300, bbox_inches='tight')
    plt.show()

def plot_noise_robustness(df, save_dir):
    """
    3. ROBUSTEZZA AL RUMORE - Coefficiente di variazione
    """
    # Calcola robustezza (std/mean) per ogni variabile
    robustness = df.groupby('variable_name')['total_reward'].agg(['mean', 'std'])
    robustness['cv'] = robustness['std'] / robustness['mean']  # Coefficiente di variazione
    robustness = robustness.sort_values('cv')
    
    plt.figure(figsize=(12, 8))
    
    bars = plt.bar(range(len(robustness)), robustness['cv'], 
                   color=plt.cm.RdYlGn_r(np.linspace(0, 1, len(robustness))))
    
    plt.xticks(range(len(robustness)), robustness.index, rotation=45, ha='right')
    plt.ylabel('Coefficiente di Variazione (std/mean)', fontsize=12)
    plt.title('Robustezza al Rumore per Variabile\n(Valori più bassi = più robusta)', 
              fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')
    
    # Valori sopra le barre
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'noise_robustness.png'), dpi=300, bbox_inches='tight')
    plt.show()

def generate_summary_stats(df):
    """
    Genera statistiche riassuntive testuali.
    """
    print("\n📊 === STATISTICHE RIASSUNTIVE ===")
    print("="*50)
    
    # Statistiche generali
    print(f"Modelli testati: {len(df)}")
    print(f"Variabili analizzate: {df['variable_name'].nunique()}")
    print(f"Livelli di rumore: {sorted(df['noise_level'].unique())}")
    print(f"Reward medio globale: {df['total_reward'].mean():.2f}")
    print(f"Reward mediano: {df['total_reward'].median():.2f}")
    
    # Top/Bottom variabili
    var_means = df.groupby('variable_name')['total_reward'].mean().sort_values(ascending=False)
    print(f"\n🏆 Top 3 variabili:")
    for i, (var, reward) in enumerate(var_means.head(3).items(), 1):
        print(f"   {i}. {var}: {reward:.2f}")
    
    print(f"\n💔 Bottom 3 variabili:")
    for i, (var, reward) in enumerate(var_means.tail(3).items(), 1):
        print(f"   {i}. {var}: {reward:.2f}")
    
    # Robustezza
    robustness = df.groupby('variable_name')['total_reward'].agg(['mean', 'std'])
    robustness['cv'] = robustness['std'] / robustness['mean']
    most_robust = robustness.sort_values('cv').head(3)
    
    print(f"\n🛡️ Variabili più robuste (basso CV):")
    for i, (var, stats) in enumerate(most_robust.iterrows(), 1):
        print(f"   {i}. {var}: CV={stats['cv']:.3f}")

def main():
    """
    Funzione principale per generare tutti i grafici di confronto.
    """
    print("📈 === ANALISI VISUAL PERFORMANCE SINGLE-VARIABLE ===")
    print("="*60)
    
    try:
        # 1. CARICA DATI
        csv_input = input("Path CSV risultati (ENTER per ultimo): ").strip()
        df = load_test_results(csv_input if csv_input else None)
        
        # 2. CREA DIRECTORY OUTPUT
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(RESULTS_DIR, "plots", f"single_var_analysis_{timestamp}")
        os.makedirs(output_dir, exist_ok=True)
        print(f"📁 Grafici salvati in: {output_dir}")
        
        # 3. GENERA TUTTI I GRAFICI
        print("\n🎨 Generando grafici...")
        
        print("   📊 1/2 Overview Boxplot...")
        plot_performance_overview(df, output_dir)
        
        print("   🛡️ 2/2 Robustezza...")
        plot_noise_robustness(df, output_dir)
        
        # 4. STATISTICHE TESTUALI
        generate_summary_stats(df)
        
        print(f"\n✅ Analisi completata!")
        print(f"📁 Tutti i grafici salvati in: {output_dir}")
        
    except Exception as e:
        print(f"❌ Errore: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()