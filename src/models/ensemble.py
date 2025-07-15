import numpy as np
from stable_baselines3 import SAC
from ..utils.constants import *
from ..utils.core import *
import os
import pickle
import zipfile
import tempfile
from datetime import datetime

class WeightedEnsemble:
	"""
	Ensemble che combina predizioni di multipli modelli SAC con pesi dinamici
	"""
	def __init__(self, model_paths, env, initial_weights=None):
		"""
		Parameters:
		model_paths: list - Lista di path ai modelli .zip
		env: Environment - Ambiente per caricare i modelli
		initial_weights: list - Pesi iniziali (se None, usa pesi uniformi)
		"""
		self.env = env
		self.model_paths = model_paths
		self.models = []
		self.model_info = []
		
		# Carica tutti i modelli con controllo errori
		print(f"🤖 Caricando {len(model_paths)} modelli per ensemble...")
		for i, path in enumerate(model_paths):
			if not os.path.exists(path):
				print(f"  ❌ File non trovato: {path}")
				self.models.append(None)
				self.model_info.append({'path': path, 'valid': False, 'index': i})
				continue
				
			try:
				model = SAC.load(path, env)
				self.models.append(model)
				
				# Estrai info dal nome del file
				filename = os.path.basename(path)
				noise_level = self._extract_noise_level(filename)
				model_type = "static" if filename.startswith("s_") else "dynamic"
				
				self.model_info.append({
					'path': path,
					'noise_level': noise_level,
					'type': model_type,
					'index': i,
					'valid': True
				})
				print(f"  ✅ Modello {i}: {filename} (noise: {noise_level:.3f})")
				
			except Exception as e:
				print(f"  ❌ Errore caricando {path}: {e}")
				self.models.append(None)
				self.model_info.append({'path': path, 'valid': False, 'index': i})
		
		# Verifica che ci siano modelli validi
		valid_models = [m for m in self.models if m is not None]
		if len(valid_models) == 0:
			raise ValueError("Nessun modello valido caricato!")
		
		print(f"📊 Ensemble inizializzato: {len(valid_models)}/{len(model_paths)} modelli validi")
		
		# Inizializza pesi solo per modelli validi
		n_models = len(self.models)
		if initial_weights is None:
			self.weights = np.ones(n_models) / n_models  # Pesi uniformi
		else:
			self.weights = np.array(initial_weights)
			self.weights = self.weights / self.weights.sum()  # Normalizza
		
		# Azzera pesi per modelli non validi
		for i, model in enumerate(self.models):
			if model is None:
				self.weights[i] = 0.0
		
		# Rinormalizza pesi
		if self.weights.sum() > 0:
			self.weights = self.weights / self.weights.sum()
		
		print(f"   Pesi iniziali: {self.weights}")
		
		# Buffer per performance tracking
		self.performance_history = []
		self.weight_history = []
	
	def _extract_noise_level(self, filename):
		"""Estrae il livello di rumore dal nome del file"""
		try:
			# Formato: s_0.15.zip o d_0.25.zip
			parts = filename.replace('.zip', '').split('_')
			return float(parts[1])
		except:
			return 0.0
	
	def predict(self, obs, deterministic=True, method='weighted_average'):
		"""
		Predice azione usando ensemble con controllo errori
		"""
		valid_models = [(i, model) for i, model in enumerate(self.models) if model is not None]
		
		if len(valid_models) == 0:
			raise ValueError("Nessun modello valido disponibile per predizione!")
		
		if method == 'weighted_average':
			return self._weighted_average_predict(obs, deterministic, valid_models)
		elif method == 'best_only':
			return self._best_only_predict(obs, deterministic, valid_models)
		else:
			raise ValueError(f"Metodo '{method}' non supportato")
	
	def _weighted_average_predict(self, obs, deterministic, valid_models):
		"""Combina predizioni con media pesata solo per modelli validi"""
		actions = []
		valid_weights = []
		
		# Ottieni predizioni solo da modelli validi
		for i, model in valid_models:
			try:
				action, _ = model.predict(obs, deterministic=deterministic)
				actions.append(action)
				valid_weights.append(self.weights[i])
			except Exception as e:
				print(f"Errore nella predizione modello {i}: {e}")
				continue
		
		if len(actions) == 0:
			raise ValueError("Nessuna predizione valida ottenuta!")
		
		# Normalizza pesi validi
		valid_weights = np.array(valid_weights)
		if valid_weights.sum() > 0:
			valid_weights = valid_weights / valid_weights.sum()
		else:
			valid_weights = np.ones(len(valid_weights)) / len(valid_weights)
		
		# Media pesata
		actions = np.array(actions)
		weighted_action = np.average(actions, weights=valid_weights, axis=0)
		
		return weighted_action, None
	
	def _best_only_predict(self, obs, deterministic):
		"""Usa solo il modello migliore"""
		best_model_idx = np.argmax(self.weights)
		return self.models[best_model_idx].predict(obs, deterministic=deterministic)
	
	def update_weights(self, recent_performance, learning_rate=None, method=None):
		"""
		Aggiorna i pesi usando configurazione se parametri non forniti
		
		Parameters:
		recent_performance: list - Performance recenti di ogni modello
		learning_rate: float - Velocità di aggiornamento pesi
		method: str - Metodo di aggiornamento ('exponential', 'linear', 'softmax')
		"""
		# Usa configurazione di default se non forniti
		if learning_rate is None:
			learning_rate = ENSEMBLE_CONFIG['learning_rate']
		if method is None:
			method = ENSEMBLE_CONFIG['update_method']
		
		if len(recent_performance) != len(self.models):
			raise ValueError("Numero performance != numero modelli")
		
		performances = np.array(recent_performance)
		
		if method == 'exponential':
			# Aggiornamento esponenziale (favorisce modelli migliori)
			exp_perf = np.exp(performances / np.max(performances))
			new_weights = exp_perf / exp_perf.sum()
			
		elif method == 'softmax':
			# Softmax con temperatura dalla configurazione
			temperature = ENSEMBLE_CONFIG['temperature']
			exp_perf = np.exp(performances / temperature)
			new_weights = exp_perf / exp_perf.sum()
			
		elif method == 'linear':
			normalized_perf = (performances - np.min(performances)) / (np.max(performances) - np.min(performances) + 1e-8)
			new_weights = normalized_perf / normalized_perf.sum()
		
		# Smooth update (evita cambiamenti troppo bruschi)
		self.weights = (1 - learning_rate) * self.weights + learning_rate * new_weights
		
		self.weight_history.append(self.weights.copy())
		self.performance_history.append(performances.copy())
		
		print(f"📊 Pesi aggiornati (metodo: {method}, lr: {learning_rate}): {self.weights}")
		
	def get_model_info(self):
		"""Restituisce informazioni sui modelli nell'ensemble"""
		return self.model_info
	
	def get_best_model(self):
		"""Restituisce il modello con peso maggiore"""
		best_idx = np.argmax(self.weights)
		return self.models[best_idx], self.model_info[best_idx]
	
	def save_ensemble(self, save_path, include_models=True, metadata=None):
		"""
		Salva l'ensemble in un file zip
		
		Parameters:
		save_path: str - Path dove salvare il file zip
		include_models: bool - Se includere i modelli SAC nel zip (default: True)
		metadata: dict - Metadati aggiuntivi da salvare
		
		Returns:
		str - Path del file salvato
		"""
		if not save_path.endswith('.zip'):
			save_path += '.zip'
		
		os.makedirs(os.path.dirname(save_path), exist_ok=True)
		
		print(f"💾 Salvando ensemble in: {save_path}")
		
		with tempfile.TemporaryDirectory() as temp_dir:
			
			config_data = {
				'ensemble_type': 'WeightedEnsemble',
				'weights': self.weights.tolist(),
				'model_info': self.model_info,
				'model_paths': self.model_paths,
				'performance_history': self.performance_history,
				'weight_history': [w.tolist() for w in self.weight_history],
				'num_models': len(self.models),
				'valid_models': [i for i, m in enumerate(self.models) if m is not None],
				'creation_time': datetime.now().isoformat(),
				'metadata': metadata or {}
			}
			
			config_path = os.path.join(temp_dir, 'ensemble_config.pkl')
			with open(config_path, 'wb') as f:
				pickle.dump(config_data, f)
			
			model_files = []
			if include_models:
				models_dir = os.path.join(temp_dir, 'models')
				os.makedirs(models_dir, exist_ok=True)
				
				for i, model in enumerate(self.models):
					if model is not None:
						try:
							model_filename = f'model_{i}.zip'
							model_path = os.path.join(models_dir, model_filename)
							model.save(model_path)
							model_files.append(model_filename)
							print(f"  ✅ Modello {i} salvato")
						except Exception as e:
							print(f"  ❌ Errore salvando modello {i}: {e}")
							model_files.append(None)
					else:
						model_files.append(None)
				
				# Aggiorna config con i file dei modelli
				config_data['model_files'] = model_files
				with open(config_path, 'wb') as f:
					pickle.dump(config_data, f)
			
			with zipfile.ZipFile(save_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
				# Aggiungi configurazione
				zipf.write(config_path, 'ensemble_config.pkl')
				
				if include_models:
					for i, model_file in enumerate(model_files):
						if model_file is not None:
							model_full_path = os.path.join(models_dir, model_file)
							if os.path.exists(model_full_path):
								zipf.write(model_full_path, f'models/{model_file}')
		
		print(f"✅ Ensemble salvato con successo: {save_path}")
		return save_path
	
	def get_save_info(self):
		"""
		Restituisce informazioni utili per il salvataggio
		"""
		valid_models = len([m for m in self.models if m is not None])
		
		info = {
			'num_total_models': len(self.models),
			'num_valid_models': valid_models,
			'current_weights': self.weights.tolist(),
			'dominant_model': {
				'index': np.argmax(self.weights),
				'weight': float(np.max(self.weights)),
				'info': self.model_info[np.argmax(self.weights)] if self.model_info else None
			},
			'performance_episodes': len(self.performance_history),
			'weight_updates': len(self.weight_history)
		}
		
		return info

def load_ensemble(ensemble_path, env, load_models=True):
	"""
	Carica un ensemble da file zip
	
	Parameters:
	ensemble_path: str - Path al file zip dell'ensemble
	env: Environment - Ambiente per caricare i modelli SAC
	load_models: bool - Se caricare anche i modelli SAC (default: True)
	
	Returns:
	WeightedEnsemble - Ensemble caricato
	"""
	if not os.path.exists(ensemble_path):
		raise FileNotFoundError(f"File ensemble non trovato: {ensemble_path}")
	
	print(f"📂 Caricando ensemble da: {ensemble_path}")
	
	# Crea directory temporanea per estrazione
	with tempfile.TemporaryDirectory() as temp_dir:
		
		with zipfile.ZipFile(ensemble_path, 'r') as zipf:
			zipf.extractall(temp_dir)
		
		config_path = os.path.join(temp_dir, 'ensemble_config.pkl')
		if not os.path.exists(config_path):
			raise ValueError("File configurazione ensemble non trovato nel zip")
		
		with open(config_path, 'rb') as f:
			config_data = pickle.load(f)
		
		print(f"  📋 Configurazione caricata: {config_data['num_models']} modelli")
		
		ensemble = WeightedEnsemble.__new__(WeightedEnsemble)  # Crea senza chiamare __init__
		ensemble.env = env
		ensemble.model_paths = config_data['model_paths']
		ensemble.model_info = config_data['model_info']
		ensemble.weights = np.array(config_data['weights'])
		ensemble.performance_history = config_data['performance_history']
		ensemble.weight_history = [np.array(w) for w in config_data['weight_history']]
		ensemble.models = [None] * config_data['num_models']
		
		if load_models and 'model_files' in config_data:
			models_dir = os.path.join(temp_dir, 'models')
			
			for i, model_file in enumerate(config_data['model_files']):
				if model_file is not None:
					try:
						model_path = os.path.join(models_dir, model_file)
						if os.path.exists(model_path):
							ensemble.models[i] = SAC.load(model_path, env)
							print(f"  ✅ Modello {i} caricato")
						else:
							print(f"  ❌ File modello {i} non trovato: {model_file}")
					except Exception as e:
						print(f"  ❌ Errore caricando modello {i}: {e}")
						ensemble.models[i] = None
		
		valid_models = [m for m in ensemble.models if m is not None]
		if len(valid_models) == 0 and load_models:
			raise ValueError("Nessun modello valido caricato!")
		
		print(f"✅ Ensemble caricato: {len(valid_models)}/{config_data['num_models']} modelli validi")
		
		if 'creation_time' in config_data:
			print(f"  📅 Creato: {config_data['creation_time']}")
		
		if 'metadata' in config_data and config_data['metadata']:
			print(f"  📝 Metadata: {config_data['metadata']}")
		
		# Mostra modello dominante
		best_idx = np.argmax(ensemble.weights)
		best_weight = ensemble.weights[best_idx]
		if ensemble.model_info and best_idx < len(ensemble.model_info):
			best_info = ensemble.model_info[best_idx]
			print(f"  🏆 Modello dominante: {best_info.get('type', 'unknown')} noise={best_info.get('noise_level', 0):.3f} (peso: {best_weight:.3f})")
		
		return ensemble

def get_ensemble_info(ensemble_path):
	"""
	Ottiene informazioni su un ensemble salvato senza caricarlo completamente
	
	Parameters:
	ensemble_path: str - Path al file zip dell'ensemble
	
	Returns:
	dict - Informazioni sull'ensemble
	"""
	if not os.path.exists(ensemble_path):
		raise FileNotFoundError(f"File ensemble non trovato: {ensemble_path}")
	
	with tempfile.TemporaryDirectory() as temp_dir:
		with zipfile.ZipFile(ensemble_path, 'r') as zipf:
			zipf.extract('ensemble_config.pkl', temp_dir)
		
		config_path = os.path.join(temp_dir, 'ensemble_config.pkl')
		with open(config_path, 'rb') as f:
			config_data = pickle.load(f)
		
		info = {
			'num_models': config_data['num_models'],
			'valid_models': config_data.get('valid_models', []),
			'weights': config_data['weights'],
			'creation_time': config_data.get('creation_time', 'Unknown'),
			'metadata': config_data.get('metadata', {}),
			'performance_episodes': len(config_data.get('performance_history', [])),
			'weight_updates': len(config_data.get('weight_history', [])),
			'model_info': config_data.get('model_info', [])
		}
		
		return info

def list_saved_ensembles(directory):
	"""
	Lista tutti gli ensemble salvati in una directory
	
	Parameters:
	directory: str - Directory da cercare
	
	Returns:
	list - Lista di informazioni sugli ensemble trovati
	"""
	if not os.path.exists(directory):
		return []
	
	ensemble_files = []
	for filename in os.listdir(directory):
		if filename.endswith('.zip'):
			filepath = os.path.join(directory, filename)
			try:
				info = get_ensemble_info(filepath)
				info['filename'] = filename
				info['filepath'] = filepath
				info['file_size'] = os.path.getsize(filepath)
				ensemble_files.append(info)
			except Exception as e:
				print(f"⚠️  Errore leggendo {filename}: {e}")
	
	return ensemble_files

