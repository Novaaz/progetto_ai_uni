"""
Script dedicato alla generazione di boxplot da risultati di analisi
"""

import os
import sys
import pandas as pd

# Aggiungi la directory root al path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.utils.constants import *
from src.utils.core import *
from src.visualization.boxplot import plot_boxplot, plot_mean

def load_latest_analysis_results():
    """Carica i risultati dall'ultima analisi aggregata"""
    analysis_dir = os.path.join(RESULTS_DIR, CSV_DIR)
    
    if not os.path.exists(analysis_dir):
        raise FileNotFoundError(f"Directory analisi non trovata: {analysis_dir}")
    
    # Cerca file analysis_*.csv
    csv_files = [f for f in os.listdir(analysis_dir) if f.startswith('analysis_') and f.endswith('.csv')]
    if not csv_files:
        raise FileNotFoundError("Nessun file di analisi trovato!")
    
    latest_csv = max(csv_files, key=lambda x: os.path.getmtime(os.path.join(analysis_dir, x)))
    csv_path = os.path.join(analysis_dir, latest_csv)
    
    print(f"📁 Caricando analisi da: {latest_csv}")
    
    # Carica e converti per boxplot
    df = pd.read_csv(csv_path)
    all_results = []
    for _, row in df.iterrows():
        all_results.append({
            'name': row['Model_Name'],
            'total_reward': row['Median'],
            'step_rewards': None,
            'seed': 'aggregated'
        })
    
    print(f"📊 Caricati {len(all_results)} risultati aggregati")
    return all_results, csv_path

def load_individual_results():
    """Carica risultati individuali per boxplot dettagliati"""
    results_dir = os.path.join(RESULTS_DIR, CSV_DIR)
    individual_results = []
    
    # Cerca file individuali (escludi analysis_, best_models_, multi_env_test_)
    for filename in os.listdir(results_dir):
        if (filename.endswith('.csv') and 
            not filename.startswith(('analysis_', 'best_models_', 'multi_env_test_'))):
            try:
                file_path = os.path.join(results_dir, filename)
                df = pd.read_csv(file_path)
                if len(df) > 0:
                    result = df.iloc[0].to_dict()
                    individual_results.append(result)
            except Exception as e:
                print(f"  ⚠️  Errore caricamento {filename}: {e}")
    
    print(f"📊 Caricati {len(individual_results)} risultati individuali")
    return individual_results

def load_multi_env_results():
    """Carica risultati dai test multi-ambiente"""
    analysis_dir = os.path.join(RESULTS_DIR, CSV_DIR)
    
    # Cerca file multi_env_test_*.csv
    csv_files = [f for f in os.listdir(analysis_dir) if f.startswith('multi_env_test_') and f.endswith('.csv')]
    if not csv_files:
        return None, None
    
    latest_csv = max(csv_files, key=lambda x: os.path.getmtime(os.path.join(analysis_dir, x)))
    csv_path = os.path.join(analysis_dir, latest_csv)
    
    print(f"📁 Caricando test multi-env da: {latest_csv}")
    
    # Carica e converti per boxplot
    df = pd.read_csv(csv_path)
    all_results = []
    
    for _, row in df.iterrows():
        # Nome per raggruppamento
        if row['is_ensemble']:
            display_name = 'ensemble'
        else:
            model_type = row['model_type']
            noise_level = row['model_noise_level']
            display_name = f"{model_type}_{noise_level:.3f}"
        
        all_results.append({
            'name': display_name,
            'total_reward': row['total_reward'],
            'step_rewards': None,
            'seed': row['env_id'],  # Usa env_id come distinguisher
            'model_type': row['model_type'],
            'is_ensemble': row['is_ensemble']
        })
    
    print(f"📊 Caricati {len(all_results)} risultati multi-env")
    print(f"🌍 Ambienti: {df['env_id'].nunique()}, 🤖 Modelli: {df['model_name'].nunique()}")
    
    return all_results, csv_path

def print_multi_env_stats(all_results):
    """Stampa statistiche per multi-ambiente"""
    if not all_results:
        return
    
    df = pd.DataFrame(all_results)
    
    print(f"\n📊 === STATISTICHE MULTI-AMBIENTE ===")
    
    # Statistiche per modello
    print(f"\n🤖 Performance per Modello:")
    model_stats = df.groupby('name')['total_reward'].agg(['mean', 'std', 'count']).sort_values('mean', ascending=False)
    
    for model_name, stats in model_stats.iterrows():
        print(f"  • {model_name:15s}: {stats['mean']:7.2f} ± {stats['std']:5.2f} ({int(stats['count'])} tests)")
    
    # Statistiche per tipo
    print(f"\n📈 Performance per Tipo:")
    type_stats = df.groupby('model_type')['total_reward'].agg(['mean', 'std', 'count']).sort_values('mean', ascending=False)
    
    for model_type, stats in type_stats.iterrows():
        print(f"  • {model_type:10s}: {stats['mean']:7.2f} ± {stats['std']:5.2f} ({int(stats['count'])} tests)")

def main():
    """Funzione principale per generazione boxplot"""
    print("📊 === GENERATORE BOXPLOT ===")
    print("="*40)
    
    try:
        # Menu di scelta semplificato
        print("\n🔧 Opzioni disponibili:")
        print("1. Boxplot da risultati individuali")
        print("2. Boxplot da test multi-ambiente") 
        print("3. Entrambi")
        
        choice = input("\nScegli opzione (1-3, default 1): ").strip() or "1"
        
        # Directory output
        plots_dir = os.path.join(RESULTS_DIR, PLOTS_DIR)
        os.makedirs(plots_dir, exist_ok=True)
        
        # Opzione 1: Risultati individuali
        if choice in ['1', '3']:
            print("\n🎨 Generando boxplot da risultati individuali...")
            try:
                individual_results = load_individual_results()
                
                if individual_results:
                    individual_plots_dir = os.path.join(plots_dir, 'individual')
                    os.makedirs(individual_plots_dir, exist_ok=True)
                    
                    plot_boxplot(individual_plots_dir, individual_results)
                    plot_mean(individual_plots_dir, individual_results)
                    
                    print(f"  ✅ Boxplot individuali salvati in: {individual_plots_dir}")
                else:
                    print("  ⚠️  Nessun risultato individuale trovato")
            except Exception as e:
                print(f"  ❌ Errore boxplot individuali: {e}")
        
        # Opzione 2: Test multi-ambiente
        if choice in ['2', '3']:
            print("\n🎨 Generando boxplot da test multi-ambiente...")
            try:
                multi_env_results, multi_env_file = load_multi_env_results()
                
                if multi_env_results:
                    multi_env_plots_dir = os.path.join(plots_dir, 'multi_environment')
                    os.makedirs(multi_env_plots_dir, exist_ok=True)
                    
                    plot_boxplot(multi_env_plots_dir, multi_env_results)
                    plot_mean(multi_env_plots_dir, multi_env_results)
                    
                    print_multi_env_stats(multi_env_results)
                    
                    print(f"  ✅ Boxplot multi-ambiente salvati in: {multi_env_plots_dir}")
                    print(f"  📁 Basati su: {os.path.basename(multi_env_file)}")
                else:
                    print("  ⚠️  Nessun risultato multi-ambiente trovato")
                    print("  💡 Esegui prima: python scripts/testing_on_new_env.py")
            except Exception as e:
                print(f"  ❌ Errore boxplot multi-ambiente: {e}")
        
        print(f"\n✅ Generazione boxplot completata!")
        print(f"📁 Directory output: {plots_dir}")
        
    except Exception as e:
        print(f"❌ Errore generico: {e}")

if __name__ == "__main__":
    main()