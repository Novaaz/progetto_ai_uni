"""
Script semplice per boxplot da CSV nelle sottocartelle di plots.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
from src.utils.constants import EXCLUDED_NOISE_LEVELS

def extract_and_group_data(results):
	"""
	Estrae total_reward e raggruppa per modello senza separare static/dynamic.
	Supporta sia dati generati (s_/d_) che multi-ambiente (ensemble, sac_xxx, etc.).
	
	Returns:
		dict: {model_name: [rewards]}
	"""
	grouped = {}
	
	for result in results:
		name = result['name']
		reward = result['total_reward']
		
		# Estrai il livello di rumore per filtrare
		noise_level = None
		if name.startswith('s_') or name.startswith('d_'):
			try:
				noise_level = float(name.split('_')[1])
			except:
				noise_level = None
		elif '_' in name and not name.lower() == 'ensemble':
			try:
				noise_level = float(name.split('_')[1])
			except:
				noise_level = None
		
		# FILTRO: Salta se il rumore è nella lista di esclusione
		if noise_level is not None:
			if any(abs(noise_level - excluded) < 0.001 for excluded in EXCLUDED_NOISE_LEVELS):
				continue  # Salta questo modello
		
		# Crea una chiave leggibile per il modello
		if name.startswith('s_'):
			# s_0.05_xxx -> Static_0.05
			noise = name.replace('s_', '').split('_')[0]
			model_key = f"Static_{noise}"
		elif name.startswith('d_'):
			# d_0.10_xxx -> Dynamic_0.10
			noise = name.replace('d_', '').split('_')[0]
			model_key = f"Dynamic_{noise}"
		elif name == 'ensemble':
			model_key = "Ensemble"
		else:
			model_key = name.title()
		
		# Aggiungi alla lista
		if model_key not in grouped:
			grouped[model_key] = []
		grouped[model_key].append(reward)
	
	return grouped

def calculate_stats(grouped_data):
	"""
	Calcola statistiche per ogni modello.
	"""
	stats = {}
	
	for model_name, rewards in grouped_data.items():
		if rewards:
			values = np.array(rewards)
			stats[model_name] = {
				'min': np.min(values),
				'q1': np.percentile(values, 25),
				'mean': np.mean(values),
				'std': np.std(values),
				'median': np.percentile(values, 50), 
				'q3': np.percentile(values, 75),
				'max': np.max(values),
				'count': len(values),
				'values': rewards
			}
	
	return stats

def plot_boxplot(directory, results):
	"""
	Crea un singolo boxplot con tutti i modelli.
	"""
	# Estrai e raggruppa
	grouped = extract_and_group_data(results)
	
	if not grouped:
		print("⚠️  Nessun dato da plottare")
		return
	
	# Calcola statistiche
	stats = calculate_stats(grouped)
	
	# Crea directory
	os.makedirs(directory, exist_ok=True)
	
	# Prepara dati per boxplot
	plot_data = []
	for model_name, rewards in grouped.items():
		for reward in rewards:
			plot_data.append({'Model': model_name, 'Total_Reward': reward})
	
	# Crea il plot
	plt.figure(figsize=(14, 8))
	df = pd.DataFrame(plot_data)
	
	# Ordina i modelli per mediana (migliori a sinistra)
	model_order = sorted(grouped.keys(), 
						key=lambda x: np.median(grouped[x]), 
						reverse=True)
	
	sns.boxplot(data=df, x='Model', y='Total_Reward', order=model_order)
	plt.title('Model Performance Comparison - All Models', fontsize=16, fontweight='bold')
	plt.xlabel('Model', fontsize=12)
	plt.ylabel('Total Reward', fontsize=12)
	plt.xticks(rotation=45, ha='right')
	plt.grid(True, alpha=0.3)
	plt.tight_layout()
	
	# Salva il plot
	plt.savefig(os.path.join(directory, 'unified_boxplot.png'), dpi=300, bbox_inches='tight')
	plt.close()
	
	# Salva statistiche
	save_stats_csv(stats, directory)
	
	print(f"📊 Boxplot unificato salvato in: {directory}")
	print(f"🏆 Migliori modelli (per mediana): {model_order[:3]}")

def plot_mean(directory, results):
	"""
	Crea un grafico a linee con le medie di tutti i modelli.
	"""
	# Estrai e raggruppa
	grouped = extract_and_group_data(results)
	
	if not grouped:
		print("⚠️  Nessun dato da plottare")
		return
	
	# Calcola statistiche
	stats = calculate_stats(grouped)
	
	# Crea directory
	os.makedirs(directory, exist_ok=True)
	
	# Prepara dati per lineplot
	plot_data = []
	for model_name, rewards in grouped.items():
		for reward in rewards:
			plot_data.append({'Model': model_name, 'Total_Reward': reward})
	
	# Crea il plot
	plt.figure(figsize=(14, 8))
	df = pd.DataFrame(plot_data)
	
	# Ordina i modelli per mediana (migliori a sinistra)
	model_order = sorted(grouped.keys(), 
						key=lambda x: np.median(grouped[x]), 
						reverse=True)
	
	# Filtra il DataFrame per ordinare i dati
	df['Model'] = pd.Categorical(df['Model'], categories=model_order, ordered=True)
	df = df.sort_values('Model')
	
	# LINEPLOT senza parametro order
	sns.lineplot(data=df, x='Model', y='Total_Reward', marker='o')
	plt.title('Model Performance - Mean Rewards', fontsize=16, fontweight='bold')
	plt.xlabel('Model', fontsize=12)
	plt.ylabel('Mean Total Reward', fontsize=12)
	plt.xticks(rotation=45, ha='right')
	plt.grid(True, alpha=0.3)
	plt.tight_layout()
	
	# Salva il plot
	plt.savefig(os.path.join(directory, 'unified_mean.png'), dpi=300, bbox_inches='tight')
	plt.close()
	
	# Salva statistiche
	save_stats_csv(stats, directory)
	
	print(f"📈 Grafico medie unificato salvato in: {directory}")

def save_stats_csv(stats, directory):
	"""
	Salva le statistiche unificate in un CSV.
	"""
	filename = "unified_analysis.csv"
	filepath = os.path.join(directory, filename)
	
	# Converti stats in lista di righe per DataFrame
	rows = []
	for model_name, stat_data in stats.items():
		rows.append({
			'Model_Name': model_name,
			'Mean': stat_data['mean'],
			'Std': stat_data['std'],
			'Min': stat_data['min'],
			'Q1': stat_data['q1'],
			'Median': stat_data['median'],
			'Q3': stat_data['q3'],
			'Max': stat_data['max'],
			'Count': stat_data['count']
		})
	
	# Ordina per mediana
	rows.sort(key=lambda x: x['Median'], reverse=True)
	
	# Crea DataFrame e salva
	df = pd.DataFrame(rows)
	df.to_csv(filepath, index=False)
	print(f"📄 Statistiche unificate salvate in: {filepath}")

# def extract_and_group_data(results):
#     """
#     Estrae total_reward e raggruppa per tipo (s_/d_) e noise level.
#     
#     Returns:
#         dict: {'static': {'0.05': [rewards]}, 'dynamic': {'0.10': [rewards]}}
#     """
#     grouped = {'static': {}, 'dynamic': {}}
#     
#     for result in results:
#         name = result['name']
#         reward = result['total_reward']
#         
#         # Determina tipo e noise level dal nome
#         if name.startswith('s_'):
#             model_type = 'static'
#             noise = name.replace('s_', '').split('_')[0]  # s_0.05_xxx -> 0.05
#         elif name.startswith('d_'):
#             model_type = 'dynamic' 
#             noise = name.replace('d_', '').split('_')[0]  # d_0.10_xxx -> 0.10
#         else:
#             continue
#         
#         # Aggiungi alla lista
#         if noise not in grouped[model_type]:
#             grouped[model_type][noise] = []
#         grouped[model_type][noise].append(reward)
#     
#     return grouped

