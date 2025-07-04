import config 
import src.training.train_functions as train_functions
from config import *
from src.training.train_functions import *
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gymnasium.wrappers import TransformObservation
import os
from datetime import datetime

def save_rewards_to_csv(training_data, save_dir):
    """
    Salva i dati delle reward in formato CSV con episodi come indice e modelli come colonne.
    
    Parametri:
    training_data: dict - Dati di training per tutti i modelli
    save_dir: str - Directory di salvataggio
    
    Returns:
    str: Path del file CSV salvato
    """
    # Prepara i dati per il DataFrame
    max_episodes = 0
    model_rewards = {}
    
    # Trova il numero massimo di episodi e prepara i dati
    for model_name, data in training_data.items():
        rewards = data['rewards']
        model_rewards[model_name] = rewards
        max_episodes = max(max_episodes, len(rewards))
    
    # Crea il DataFrame con episodi come indice
    df_data = {}
    for model_name, rewards in model_rewards.items():
        # Estendi con NaN se necessario per uniformare la lunghezza
        padded_rewards = rewards + [np.nan] * (max_episodes - len(rewards))
        df_data[model_name.replace('_', ' ').title()] = padded_rewards
    
    # Crea DataFrame
    df = pd.DataFrame(df_data)
    df.index.name = 'Episodio'
    df.index = df.index + 1  # Inizia episodi da 1 invece di 0
    
    # Crea riga di metadati come primo commento nel CSV
    metadata_line = f"# SEED: {RANDOM_SEED}, EPISODI: {EPISODES}, DATA: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    
    # Salva CSV
    csv_path = os.path.join(save_dir, "training_rewards.csv")
    
    # Scrivi prima i metadati come commento
    with open(csv_path, 'w') as f:
        f.write(metadata_line)
        # Aggiungi il DataFrame
        df.to_csv(f, index=True)
    
    print(f"📊 CSV salvato: {csv_path}")
    return csv_path

# Inizio esecuzione
execution_start = datetime.now()
print(f"🚀 Avvio simulazione: {execution_start.strftime('%Y-%m-%d %H:%M:%S')}")

original_models, training_data = train_offline_models()

# Crea il grafico per tutti i modelli
plt.figure(figsize=(14, 10))

# Colori per i diversi modelli
colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
model_names = list(training_data.keys())

all_rewards = []

# Plotta ogni modello
for i, (model_name, data) in enumerate(training_data.items()):
	episode_numbers, grouped_rewards = group_rewards_by_episodes(data['rewards'], group_size=1)
	
	if len(grouped_rewards) > 0:
		plt.plot(episode_numbers, grouped_rewards, 
				marker='o', markersize=4, linewidth=2, 
				color=colors[i % len(colors)], alpha=0.8, 
				label=f'{model_name.replace("_", " ").title()}')
		all_rewards.extend(grouped_rewards)

# Personalizza il grafico
plt.title('Andamento delle Reward durante il Training\n(Confronto tra tutti i modelli)', 
		fontsize=16, fontweight='bold')
plt.xlabel('Episodi di Valutazione', fontsize=12)
plt.ylabel('Reward Media', fontsize=12)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=11)

# Aggiungi statistiche come testo
if len(all_rewards) > 0:
	min_reward = min(all_rewards)
	max_reward = max(all_rewards)
	avg_reward = np.mean(all_rewards)
	
	stats_text = f"""Statistiche Generali:
• Reward minima: {min_reward:.3f}
• Reward massima: {max_reward:.3f}
• Reward media: {avg_reward:.3f}
• Modelli trainati: {len(model_names)}	• Punti nel grafico: {len(all_rewards)}"""
	
	plt.text(0.02, 0.98, stats_text, 
			transform=plt.gca().transAxes, 
			fontsize=10,
			verticalalignment='top',
			bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

# Crea cartella di salvataggio con nome basato su data/ora di inizio
timestamp = execution_start.strftime("%Y%m%d_%H%M%S")
save_dir = os.path.join("plots", f"sim_{timestamp}")
os.makedirs(save_dir, exist_ok=True)
print(f"📁 Directory di salvataggio: {save_dir}")

# Salva i dati in CSV
csv_path = save_rewards_to_csv(training_data, save_dir)

# Salva e mostra il grafico
plt.tight_layout()
plot_path = os.path.join(save_dir, 'training_rewards_plot.png')
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
plt.show()

print(f"✅ Grafico salvato: {plot_path}")
print(f"📊 Modelli trainati: {', '.join(model_names)}")
print(f"🔢 Punti totali nel grafico: {len(all_rewards)}")
print(f"📁 Tutti i file salvati in: {save_dir}")

# Stampa riepilogo finale
end_time = datetime.now()
duration = (end_time - execution_start).total_seconds()
print(f"\n⏱️ Simulazione completata in {duration/60:.2f} minuti ({duration:.1f} secondi)")
print(f"🎯 SEED utilizzato: {RANDOM_SEED}")
print(f"📈 Episodi per modello: {EPISODES}")
