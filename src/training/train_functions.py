from datetime import datetime
import numpy as np
from stable_baselines3 import SAC
from ..utils.constants import *
from ..utils.core import *
from ..models.ensemble import WeightedEnsemble
from ..evaluation.evaluate_functions import evaluate_sac_performance
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from gymnasium.wrappers import TransformObservation
from functools import partial
import os
import csv

_NOISE_ENVIRONMENTS = {}

def train_sac(env, seed, sac_model=None, n=1, batch_size=BATCH_SIZE, learning_starts=LEARNING_STARTS, episodes=EPISODES, time_steps=None, track_rewards=False, eval_freq=1000, deterministic=True, use_episode_tracker=True, n_eval_episodes=3):
	if sac_model is None:
		sac_model = SAC(
		policy='MlpPolicy',
		env=env,
		**SAC_KWARGS,
		seed=seed,
		verbose=1)

	if time_steps is None:
		time_steps = env.unwrapped.time_steps - 1
		
	callback = None
	reward_tracker = None
	
	if n == 2:
		callback = RBCPureCallback(env, verbose=0)
	
	if track_rewards:
		if use_episode_tracker:
			reward_tracker = EpisodeRewardTracker(verbose=1)
		else:
			# Crea un ambiente di valutazione pulito (senza rumore) per finetuning
			eval_env = CityLearnEnv(**ENV_CONFIG)
			eval_env = StableBaselines3Wrapper(NormalizedSpaceWrapper(eval_env))
			
			reward_tracker = SimpleRewardTracker(
				eval_env=eval_env, 
				eval_freq=eval_freq,
				verbose=1
			)
		
		if callback is not None:
			callback = CallbackList([callback, reward_tracker])
		else:
			callback = reward_tracker

	total_timesteps = episodes * time_steps
	
	env.reset()
	sac_model.learn(
		total_timesteps=total_timesteps,
		reset_num_timesteps=False,
		callback=callback,
		progress_bar=True
	)

	if track_rewards and reward_tracker:
		if use_episode_tracker:
			# Restituisce i dati dell'EpisodeRewardTracker
			return env, sac_model, reward_tracker.episode_rewards, reward_tracker.timesteps_evaluated
		else:
			# Restituisce i dati del SimpleRewardTracker
			return env, sac_model, reward_tracker.training_rewards, reward_tracker.timesteps_evaluated
	else:
		return env, sac_model

class SimpleRewardTracker(BaseCallback):
	"""
	Callback semplice per tracciare le reward durante il training
	è una valutazione veloce che si esegue ogni eval_freq timesteps sempre sui primi max_steps.    
	"""
	def __init__(self, eval_env, eval_freq=1000, verbose=0):
		super().__init__(verbose)
		self.eval_env = eval_env
		self.eval_freq = eval_freq
		self.training_rewards = []
		self.timesteps_evaluated = []
		
	def _on_step(self) -> bool:
		# Valuta ogni eval_freq timesteps
		if self.n_calls % self.eval_freq == 0:
			obs, _ = self.eval_env.reset()
			episode_reward = 0
			steps = 0
			max_steps = min(self.eval_env.unwrapped.time_steps, 100)  # Valutazione veloce
			
			while not self.eval_env.terminated and steps < max_steps:
				action, _ = self.model.predict(obs, deterministic=True)
				obs, reward, terminated, truncated, _ = self.eval_env.step(action)
				episode_reward += float(np.mean(reward) if hasattr(reward, 'shape') else reward)
				steps += 1
				if terminated or truncated:
					break
					
			self.training_rewards.append(episode_reward)
			self.timesteps_evaluated.append(self.n_calls)
			
			if self.verbose > 0:
				print(f"Timestep {self.n_calls}: Reward = {episode_reward:.2f}")
		
		return True

class EpisodeRewardTracker(BaseCallback):
	"""
	Callback semplice per tracciare le reward durante il training.
	Raccoglie le reward cumulative alla fine di ogni episodio completo.
	"""
	def __init__(self, verbose=0):
		super().__init__(verbose)
		self.episode_rewards = []
		self.timesteps_evaluated = []
		self.current_episode_reward = 0
		self.env_timestep = 0
		self.episode_length = None  # Sarà impostato dinamicamente
		
	def _on_step(self) -> bool:
		# Accumula la reward del passo corrente
		# In Stable-Baselines3, le reward sono accessibili tramite self.locals['rewards']
		try:
			if 'rewards' in self.locals and self.locals['rewards'] is not None:
				reward = self.locals['rewards']
				# Gestisci diversi formati di reward
				if hasattr(reward, '__iter__') and not isinstance(reward, str):
					reward_val = float(np.mean(reward))
				else:
					reward_val = float(reward)
				self.current_episode_reward += reward_val
			# Fallback: prova a ottenere la reward dall'info
			elif 'infos' in self.locals and self.locals['infos'] is not None:
				infos = self.locals['infos']
				if len(infos) > 0 and 'reward' in infos[0]:
					reward_val = float(infos[0]['reward'])
					self.current_episode_reward += reward_val
		except Exception as e:
			if self.verbose > 1:
				print(f"Errore nell'accesso alle reward: {e}")
		
		# Incrementa il contatore dei timestep dell'ambiente
		self.env_timestep += 1
		
		# Usa la lunghezza episodio di 719 timestep
		episode_length = 719
		
		# Alla fine di ogni episodio (ogni 719 timestep), salva la reward cumulativa
		if self.env_timestep % episode_length == 0:
			self.episode_rewards.append(self.current_episode_reward)
			self.timesteps_evaluated.append(self.n_calls)
			
			if self.verbose > 0:
				episode_num = self.env_timestep // episode_length
				print(f"Fine episodio {episode_num} (timestep {self.env_timestep}): Reward episodica = {self.current_episode_reward:.2f}")
			
			# Reset per il prossimo episodio
			self.current_episode_reward = 0
		
		return True
	
	def get_episode_data(self):
		"""Restituisce i dati degli episodi raccolti."""
		return {
			'episode_rewards': self.episode_rewards,
			'timesteps_evaluated': self.timesteps_evaluated
		}

def generate_noise_levels(n_models=10, min_noise=0.0, max_noise=0.5):
	"""
	Genera livelli di rumore crescenti per l'esperimento
	
	Parameters:
	n_models: int - Numero di modelli da generare
	min_noise: float - Rumore minimo (std)
	max_noise: float - Rumore massimo (std)
	
	Returns:
	list: Lista di livelli di rumore (std)
	"""
	if n_models == 1:
		return [min_noise]
	
	# Genera livelli di rumore distribuiti linearmente
	noise_levels = np.linspace(min_noise, max_noise, n_models)
	return noise_levels.tolist()

def train_model_with_noise(noise_std, dinamic_noise, seed, model_id):
	"""
	Allena un singolo modello SAC usando ambienti condivisi per livello di rumore
	"""
	print(f"📈 Training Modello {model_id} | Noise STD: {noise_std:.4f}")
	
	try:
		# Ottieni o crea ambiente condiviso per questo livello di rumore
		train_env, env_name = get_or_create_noise_environment(
			noise_level=noise_std,
			noise_type='gaussian',
			noise_mean=0.0,
			dinamic_noise=dinamic_noise
		)
		
		_, trained_model, training_rewards, timesteps = train_sac(
			env=train_env,
			seed=seed,
			track_rewards=True,
			eval_freq=50,
			episodes=EPISODES
		)
		
		training_data = {
			'rewards': training_rewards,
			'timesteps': timesteps,
			'noise_std': noise_std,
			'seed': seed,
			'model_id': model_id,
			'env_name': env_name  # Aggiungi nome ambiente
		}
		
		print(f"  ✅ Training completato | Ambiente: {env_name} | Episodi: {len(training_rewards)}")
		return trained_model, training_data, noise_std
		
	except Exception as e:
		print(f"  ❌ Errore durante training modello {model_id}: {e}")
		return None, None, noise_std

class ExperienceBuffer:
	"""
	Buffer per raccogliere e gestire esperienze durante il training online
	"""
	def __init__(self, max_size=10000):
		self.max_size = max_size
		self.experiences = []
		self.current_episode = []
		
	def add_step(self, obs, action, reward, next_obs, done):
		"""Aggiungi un singolo step al buffer"""
		experience = {
			'obs': obs.copy(),
			'action': action.copy(),
			'reward': reward,
			'next_obs': next_obs.copy(),
			'done': done
		}
		
		self.current_episode.append(experience)
		
		if done:
			# Episodio completato, aggiungilo al buffer
			self.experiences.append(self.current_episode.copy())
			self.current_episode = []
			
			# Rimuovi episodi vecchi se necessario
			if len(self.experiences) > self.max_size:
				self.experiences.pop(0)
	
	def get_recent_experiences(self, num_episodes=None):
		"""Ottieni episodi recenti per training"""
		if num_episodes is None:
			return self.experiences
		else:
			return self.experiences[-num_episodes:]
	
	def get_all_experiences(self):
		"""Ottieni tutte le esperienze come lista piatta"""
		all_steps = []
		for episode in self.experiences:
			all_steps.extend(episode)
		return all_steps
	
	def clear(self):
		"""Pulisci il buffer"""
		self.experiences = []
		self.current_episode = []
	
	def size(self):
		"""Restituisce numero di episodi nel buffer"""
		return len(self.experiences)
	
	def total_steps(self):
		"""Restituisce numero totale di step nel buffer"""
		return sum(len(episode) for episode in self.experiences)

def train_ensemble_online_with_experience_replay(model_paths, env, max_episodes=None, 
											   weight_update_freq=None, ensemble_method=None, 
											   learning_rate=None, update_method=None):
	"""
	Versione semplificata del training online con experience replay
	"""
	# Usa configurazione di default se parametri non forniti
	config = ENSEMBLE_CONFIG
	online_config = config.get('online_training', {})
	
	if max_episodes is None:
		max_episodes = int(config['max_episodes']/5)
	if weight_update_freq is None:
		weight_update_freq = config['weight_update_freq']
	if ensemble_method is None:
		ensemble_method = config['ensemble_method']
	if learning_rate is None:
		learning_rate = config['learning_rate']
	if update_method is None:
		update_method = config['update_method']
	
	# Buffer con configurazione corretta
	experience_buffer = ExperienceBuffer(max_size=online_config.get('buffer_size', 10000))
	
	# Training periodico con configurazione corretta
	experience_steps = online_config.get('experience_steps', 100)
	training_episodes = online_config.get('training_episodes', 5)
	final_training_episodes = online_config.get('final_training_episodes', 20)
	
	# Crea ensemble
	try:
		ensemble = WeightedEnsemble(model_paths, env)
		print(f"✅ Ensemble creato con {len(ensemble.models)} modelli")
	except Exception as e:
		print(f"❌ Errore nella creazione ensemble: {e}")
		return None
	
	# Verifica modelli validi
	valid_models = [model for model in ensemble.models if model is not None]
	if len(valid_models) == 0:
		print("❌ Nessun modello valido nell'ensemble!")
		return None
	
	# Buffer per performance
	individual_performances = [[] for _ in range(len(ensemble.models))]
	ensemble_performance = []
	
	print(f"🎯 Iniziando training...")
	
	for episode in range(max_episodes):
		try:
			obs, _ = env.reset()
			done = False
			episode_reward = 0
			step_count = 0
			max_steps = env.unwrapped.time_steps - 1
			
			print(f"\n📊 Episodio {episode}/{max_episodes}")
			
			# Esegui episodio
			while not done and step_count < max_steps:
				try:
					# Predizione ensemble
					action, _ = ensemble.predict(obs, deterministic=False, method=ensemble_method)
					
					# Step ambiente
					next_obs, reward, terminated, truncated, info = env.step(action)
					done = terminated or truncated
					
					# Calcola reward
					reward_val = np.mean(reward) if hasattr(reward, '__iter__') else reward
					episode_reward += reward_val
					
					# Aggiungi al buffer
					experience_buffer.add_step(obs, action, reward_val, next_obs, done)
					
					# Training periodico ogni experience_steps
					if step_count > 0 and step_count % experience_steps == 0:
						print(f"  🔄 Step {step_count}: Training su esperienze...")
						if experience_buffer.size() > 0:
							_train_ensemble_on_experiences(
								ensemble, 
								experience_buffer, 
								episodes=training_episodes,  # ✅ Usa config
								config=online_config,
								use_all_experiences=True
							)
					
					obs = next_obs
					step_count += 1
					
				except Exception as e:
					print(f"  ⚠️  Errore step {step_count}: {e}")
					break
			
			ensemble_performance.append(episode_reward)
			
			# Valutazione e aggiornamento pesi periodico
			if episode % weight_update_freq == 0 and episode > 0:
				print(f"  📊 Valutando modelli su esperienze...")
				
				# Usa il sistema esistente per valutare performance
				current_performances = []
				for i, model in enumerate(ensemble.models):
					if model is not None:
						model_perf = _evaluate_single_model_on_buffer(model, experience_buffer)
						current_performances.append(model_perf)
					else:
						current_performances.append(0.0)
				
				# Aggiorna pesi
				ensemble.update_weights(
					recent_performance=current_performances,
					learning_rate=learning_rate,
					method=update_method
				)

				avg_reward = np.mean(ensemble_performance[-5:]) if len(ensemble_performance) >= 5 else episode_reward
				print(f"  📈 Episodio {episode}: Reward = {episode_reward:.2f}, Media = {avg_reward:.2f}")
				print(f"  💾 Buffer: {experience_buffer.size()} episodi, {experience_buffer.total_steps()} step")
				
		except Exception as e:
			print(f"  ❌ Errore episodio {episode}: {e}")
			continue
	
	# Training finale con configurazione corretta
	if experience_buffer.size() > 0:
		print(f"\n🎯 Training finale...")
		_train_ensemble_on_experiences(
			ensemble, 
			experience_buffer, 
			episodes=final_training_episodes,  # ✅ Usa config
			config=online_config,
			use_all_experiences=True
		)
	
	print(f"\n✅ Training completato!")
	print(f"   Performance media: {np.mean(ensemble_performance):.2f}")
	print(f"   Pesi finali: {ensemble.weights}")
	
	return ensemble

def _train_ensemble_on_experiences(ensemble, experience_buffer, episodes, config, use_all_experiences=False):
	"""
	Allena l'ensemble su esperienze raccolte
	
	Parameters:
	ensemble: WeightedEnsemble - Ensemble da allenare
	experience_buffer: ExperienceBuffer - Buffer con esperienze
	episodes: int - Numero di episodi di training
	config: dict - Configurazione training
	use_all_experiences: bool - Se usare tutte le esperienze o solo le recenti
	"""
	if experience_buffer.size() == 0:
		print("  ⚠️  Buffer vuoto, saltando training")
		return
	
	if use_all_experiences:
		episodes_to_use = experience_buffer.get_all_experiences()
		print(f"  🎯 Training su {len(episodes_to_use)} step (tutte le esperienze)")
	else:
		recent_episodes = experience_buffer.get_recent_experiences(num_episodes=5)  # Usa ultimi 5 episodi
		episodes_to_use = []
		for ep in recent_episodes:
			episodes_to_use.extend(ep)
		print(f"  🎯 Training su {len(episodes_to_use)} step (esperienze recenti)")
	
	if len(episodes_to_use) == 0:
		print("  ⚠️  Nessuna esperienza disponibile")
		return
	
	for i, model in enumerate(ensemble.models):
		if model is None:
			continue
			
		try:
			model_performance = 0
			experiences_used = 0
			
			for exp in episodes_to_use:
				try:
					pred_action, _ = model.predict(exp['obs'], deterministic=True)
					
					if hasattr(exp['action'], '__len__') and hasattr(pred_action, '__len__'):
						action_diff = np.mean(np.abs(pred_action - exp['action']))
						accuracy = np.exp(-action_diff)  # Accuratezza esponenziale
					else:
						accuracy = 1.0
					
					# Peso reward per accuratezza
					weighted_reward = exp['reward'] * accuracy
					model_performance += weighted_reward
					experiences_used += 1
					
				except Exception as e:
					continue
			
			if experiences_used > 0:
				model_performance /= experiences_used
			
			print(f"    Modello {i}: performance stimata = {model_performance:.3f} su {experiences_used} esperienze")
			
		except Exception as e:
			print(f"    ❌ Errore training modello {i}: {e}")

def _evaluate_single_model_on_buffer(model, experience_buffer, num_episodes=None):
	"""
	Valuta un singolo modello basandosi sulle esperienze nel buffer
	
	Parameters:
	model: SAC - Modello da valutare
	experience_buffer: ExperienceBuffer - Buffer con esperienze raccolte
	num_episodes: int - Numero di episodi recenti da usare (None = tutti)
	
	Returns:
	float: Performance stimata del modello
	"""
	if experience_buffer.size() == 0:
		print("    ⚠️  Buffer vuoto per valutazione")
		return 0.0
	
	try:
		if num_episodes is None:
			episodes_to_use = experience_buffer.get_all_experiences()
		else:
			recent_episodes = experience_buffer.get_recent_experiences(num_episodes)
			episodes_to_use = []
			for episode in recent_episodes:
				episodes_to_use.extend(episode)
		
		if len(episodes_to_use) == 0:
			print("    ⚠️  Nessuna esperienza disponibile per valutazione")
			return 0.0
		
		total_weighted_reward = 0.0
		valid_experiences = 0
		
		for exp in episodes_to_use:
			try:
				# Predici azione con il modello
				pred_action, _ = model.predict(exp['obs'], deterministic=True)
				
				# Calcola accuratezza confrontando con azione realmente presa
				if hasattr(exp['action'], '__len__') and hasattr(pred_action, '__len__'):
					action_diff = np.mean(np.abs(pred_action - exp['action']))
					accuracy = np.exp(-action_diff)
				else:
					action_diff = abs(pred_action - exp['action'])
					accuracy = np.exp(-action_diff)
				
				# Se il modello avrebbe fatto un'azione simile e la reward era buona
				weighted_reward = exp['reward'] * accuracy
				total_weighted_reward += weighted_reward
				valid_experiences += 1
				
			except Exception as e:
				continue
		
		if valid_experiences > 0:
			performance = total_weighted_reward / valid_experiences
		else:
			performance = 0.0
		
		return performance
		
	except Exception as e:
		print(f"    ❌ Errore valutazione modello: {e}")
		return 0.0

def get_or_create_noise_environment(noise_level, noise_type='gaussian', noise_mean=0.0, 
								   dinamic_noise=False, env_name=None):
	"""
	Ottiene o crea un ambiente con rumore specifico (condiviso tra modelli)
	
	Parameters:
	noise_level: float - Livello di rumore
	noise_type: str - Tipo di rumore
	noise_mean: float - Media del rumore  
	dinamic_noise: bool - Se il rumore è dinamico
	env_name: str - Nome dell'ambiente (opzionale)
	
	Returns:
	env: Ambiente configurato con il rumore specificato
	"""
	global _NOISE_ENVIRONMENTS
	
	if env_name is None:
		env_name = f"env_{noise_type}_{noise_level:.3f}_{noise_mean:.3f}_dynamic_{dinamic_noise}"
	
	# Se l'ambiente esiste già, restituiscilo
	if env_name in _NOISE_ENVIRONMENTS:
		print(f"🔄 Riutilizzando ambiente esistente: {env_name}")
		return _NOISE_ENVIRONMENTS[env_name]['env'], env_name
	
	print(f"🏗️  Creando nuovo ambiente: {env_name}")
	
	# Crea l'ambiente base
	if noise_level == 0.0:
		env = CityLearnEnv(**ENV_CONFIG)
		env = StableBaselines3Wrapper(NormalizedSpaceWrapper(env))
		print(f"  🔹 Ambiente pulito (no noise)")
	else:
		env = CityLearnEnv(**ENV_CONFIG)
		env = StableBaselines3Wrapper(NormalizedSpaceWrapper(env))
		env = TransformObservation(env, partial(
			add_noise_to_observations,
			noise_level=noise_level,
			noise_type=noise_type,
			noise_mean=noise_mean,
			dinamic_noise=dinamic_noise,
			env_name=env_name  # Usa env_name invece di model_id
		))
		print(f"  🔸 Ambiente con rumore {noise_type} (σ={noise_level:.3f}, μ={noise_mean:.3f}, dynamic={dinamic_noise})")
	
	# Salva nell'ambiente globale
	_NOISE_ENVIRONMENTS[env_name] = {
		'env': env,
		'noise_level': noise_level,
		'noise_type': noise_type,
		'noise_mean': noise_mean,
		'dinamic_noise': dinamic_noise,
		'created_at': datetime.now().isoformat(),
		'usage_count': 0
	}
	
	return env, env_name

def clear_noise_environments():
	"""Pulisce tutti gli ambienti di rumore dalla memoria"""
	global _NOISE_ENVIRONMENTS
	count = len(_NOISE_ENVIRONMENTS)
	_NOISE_ENVIRONMENTS.clear()
	print(f"🧹 Rimossi {count} ambienti dalla memoria")

def save_noise_environments_info(filepath=None):
	"""Salva informazioni sugli ambienti di rumore in un file CSV"""
	global _NOISE_ENVIRONMENTS
	
	if filepath is None:
		timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
		filepath = os.path.join(RESULTS_DIR, f"noise_environments_{timestamp}.csv")
	
	# Prepara dati per CSV
	csv_data = []
	for name, info in _NOISE_ENVIRONMENTS.items():
		csv_data.append({
			'environment_name': name,
			'noise_level': info['noise_level'],
			'noise_type': info['noise_type'],
			'noise_mean': info['noise_mean'],
			'dinamic_noise': info['dinamic_noise'],
			'created_at': info['created_at'],
			'usage_count': info['usage_count']
		})
	
	if csv_data:
		df = pd.DataFrame(csv_data)
		
		metadata_lines = [
			f"# NOISE ENVIRONMENTS INFO - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
			f"# TOTAL ENVIRONMENTS: {len(csv_data)}",
			f"# ACTIVE ENVIRONMENTS IN MEMORY",
			"#",
			"# COLUMNS:",
			"# environment_name: Nome identificativo dell'ambiente",
			"# noise_level: Livello di rumore (standard deviation)",
			"# noise_type: Tipo di rumore (gaussian/uniform)",
			"# noise_mean: Media del rumore",
			"# dinamic_noise: Se il rumore è dinamico (True/False)",
			"# created_at: Timestamp di creazione ambiente",
			"# usage_count: Numero di volte che l'ambiente è stato utilizzato",
			"#"
		]
		
		os.makedirs(os.path.dirname(filepath), exist_ok=True)
		with open(filepath, 'w') as f:
			for line in metadata_lines:
				f.write(line + '\n')
			df.to_csv(f, index=False, float_format='%.6f')
		
		print(f"💾 Info ambienti salvate in: {filepath}")
		print(f"📊 Dati: {len(csv_data)} ambienti × {len(df.columns)} colonne")
	else:
		print("⚠️  Nessun ambiente in memoria da salvare")
		
		with open(filepath, 'w') as f:
			f.write(f"# NOISE ENVIRONMENTS INFO - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
			f.write("# NO ENVIRONMENTS IN MEMORY\n")
			f.write("#\n")
			f.write("environment_name,noise_level,noise_type,noise_mean,dinamic_noise,created_at,usage_count\n")

def get_environment_by_noise_level(noise_level, dinamic_noise=False):
	"""Ottieni ambiente esistente per livello di rumore specifico"""
	global _NOISE_ENVIRONMENTS
	
	for name, info in _NOISE_ENVIRONMENTS.items():
		if (abs(info['noise_level'] - noise_level) < 0.001 and 
			info['dinamic_noise'] == dinamic_noise):
			return info['env'], name
	
	return None, None

class AdaptiveAlphaCallback(BaseCallback):
    def __init__(self, eval_env, pretrained_model=None, pretrained_reward=None,
                eval_freq=5000, n_eval_episodes=1, tol=0.02, step_alpha=0.05,
                log_path=None, verbose=0,
                real_eval_env=None):
        # Inizializza callback: env per valutazione, modello pretrained (opzionale),
        # frequenza di valutazione, tolleranza e passo per aggiornare alpha.
        super().__init__(verbose)
        self.eval_env = eval_env
        self.pretrained_model = pretrained_model
        self.pretrained_reward = float(pretrained_reward) if pretrained_reward is not None else None
        self.eval_freq = int(eval_freq)
        self.n_eval_episodes = int(n_eval_episodes)
        self.tol = float(tol)
        self.step_alpha = float(step_alpha)
        # percorso CSV per log degli snapshot
        self.log_path = log_path or os.path.join(RESULTS_DIR, "logs", f"adaptive_alpha_cb_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        # env da usare per valutazioni "reali" (se diverso dal main eval_env)
        self.real_eval_env = real_eval_env or self.eval_env

        # crea header CSV (step, valutazione, sim/real e alpha)
        if not os.path.exists(self.log_path):
            with open(self.log_path, 'w', newline='') as f:
                w = csv.writer(f)
                w.writerow(["step", "mean_reward_eval_env", "sim_mean_reward", "real_mean_reward", "sim_to_real_gap", "alpha"])

    def _on_training_start(self) -> None:
        # Alla partenza: se non ho pretrained_reward, provo a calcolarlo dal pretrained_model
        if self.pretrained_reward is None and self.pretrained_model is not None:
            try:
                res = evaluate_sac_performance(self.eval_env, self.pretrained_model, episode_name="pretrained_cb_benchmark")
                self.pretrained_reward = float(res.get("total_reward", 0.0))
                if self.verbose:
                    print(f"[AdaptiveAlphaCallback] pretrained_reward (eval_env) = {self.pretrained_reward:.2f}")
            except Exception:
                self.pretrained_reward = None

        # calcolo opzionale di riferimento sim/real per il pretrained (usato come target_gap)
        self.pretrained_sim_reward = None
        self.pretrained_real_reward = None
        if self.pretrained_model is not None:
            try:
                r_sim = evaluate_sac_performance(self.eval_env, self.pretrained_model, episode_name="pretrained_sim")
                self.pretrained_sim_reward = float(r_sim.get("total_reward", 0.0))
            except Exception:
                self.pretrained_sim_reward = None
            try:
                r_real = evaluate_sac_performance(self.real_eval_env, self.pretrained_model, episode_name="pretrained_real")
                self.pretrained_real_reward = float(r_real.get("total_reward", 0.0))
            except Exception:
                self.pretrained_real_reward = None
            if self.verbose:
                print(f"[AdaptiveAlphaCallback] pretrained_sim={self.pretrained_sim_reward} pretrained_real={self.pretrained_real_reward}")

    def _evaluate_on_env(self, env, n_eps):
        # Esegue n_eps rollout valutativi sull'env passato e ritorna media delle total_reward
        sumr = 0.0
        for _ in range(n_eps):
            res = evaluate_sac_performance(env, self.model, episode_name="adaptive_eval")
            sumr += float(res.get("total_reward", 0.0))
        return sumr / float(n_eps)

    def _on_step(self) -> bool:
        # Esegui valutazione solo ogni eval_freq chiamate
        if self.eval_freq <= 0:
            return True
        if self.n_calls % self.eval_freq != 0:
            return True

        # valutazione principale (su eval_env)
        try:
            mean_r = self._evaluate_on_env(self.eval_env, self.n_eval_episodes)
        except Exception as e:
            if self.verbose:
                print(f"[AdaptiveAlphaCallback] evaluation failed: {e}")
            return True

        # opzionalmente valutazioni separate su "sim" e "real" env
        try:
            sim_mean = self._evaluate_on_env(self.eval_env, self.n_eval_episodes)
        except Exception:
            sim_mean = None
        try:
            real_mean = self._evaluate_on_env(self.real_eval_env, self.n_eval_episodes)
        except Exception:
            real_mean = None

        # calcolo gap sim->real se disponibili entrambe le valutazioni
        sim_to_real_gap = None
        if sim_mean is not None and real_mean is not None:
            sim_to_real_gap = float(sim_mean - real_mean)

        # leggo alpha corrente dal replay buffer (se esiste)
        buf = getattr(self.model, "replay_buffer", None)
        curr_alpha = None
        if buf is not None:
            curr_alpha = getattr(buf, "get_alpha", lambda: getattr(buf, "alpha", None))()

        # adattamento semplice basato sul confronto con pretrained_reward (se disponibile)
        if self.pretrained_reward is not None and buf is not None:
            rel_diff = (mean_r - self.pretrained_reward) / (abs(self.pretrained_reward) + 1e-8)
            if rel_diff < -self.tol:
                # performance peggiora rispetto a pretrained -> aumenta peso offline (alpha++)
                if hasattr(buf, "adjust_alpha_step"):
                    buf.adjust_alpha_step(abs(self.step_alpha))
                else:
                    buf.alpha = min(1.0, getattr(buf, "alpha", 1.0) + abs(self.step_alpha))
            elif rel_diff > self.tol:
                # performance migliora -> riduci peso offline (alpha--)
                if hasattr(buf, "adjust_alpha_step"):
                    buf.adjust_alpha_step(-abs(self.step_alpha))
                else:
                    buf.alpha = max(0.0, getattr(buf, "alpha", 0.0) - abs(self.step_alpha))
            curr_alpha = getattr(buf, "get_alpha", lambda: getattr(buf, "alpha", None))()

        # scrivo riga di log CSV con valutazioni e alpha
        with open(self.log_path, 'a', newline='') as f:
            w = csv.writer(f)
            w.writerow([int(self.n_calls), float(mean_r),
                        (float(sim_mean) if sim_mean is not None else ""),
                        (float(real_mean) if real_mean is not None else ""),
                        (float(sim_to_real_gap) if sim_to_real_gap is not None else ""),
                        (float(curr_alpha) if curr_alpha is not None else "")])

        if self.verbose:
            print(f"[AdaptiveAlphaCallback] step={self.n_calls} eval={mean_r:.2f} sim={sim_mean} real={real_mean} gap={sim_to_real_gap} alpha={curr_alpha}")

        # --- logica alternativa: adattamento basato sul gap sim->real ---
        # leggo di nuovo buffer e alpha corrente
        buf = getattr(self.model, "replay_buffer", None)
        curr_alpha = None
        if buf is not None:
            curr_alpha = getattr(buf, "get_alpha", lambda: getattr(buf, "alpha", None))()

        # calcolo target_gap se ho il pretrained valutato su sim e real
        target_gap = None
        if hasattr(self, "pretrained_sim_reward") and self.pretrained_sim_reward is not None and hasattr(self, "pretrained_real_reward") and self.pretrained_real_reward is not None:
            target_gap = float(self.pretrained_sim_reward - self.pretrained_real_reward)

        # se ho entrambe le valutazioni sim e real, applico la regola basata sul gap
        if buf is not None and sim_mean is not None and real_mean is not None:
            model_gap = float(sim_mean - real_mean)
            if target_gap is None:
                # regola semplice: se gap ridotto sotto una tolleranza relativa -> più online
                tol_gap = getattr(self, "gap_tol", 0.10)
                if abs(model_gap) <= tol_gap * (abs(sim_mean) + 1e-8):
                    # riduci contributo offline
                    if hasattr(buf, "adjust_alpha_step"):
                        buf.adjust_alpha_step(-abs(self.step_alpha))
                    else:
                        buf.alpha = max(0.0, buf.alpha - abs(self.step_alpha))
                else:
                    # aumenta contributo offline
                    if hasattr(buf, "adjust_alpha_step"):
                        buf.adjust_alpha_step(abs(self.step_alpha))
                    else:
                        buf.alpha = min(1.0, buf.alpha + abs(self.step_alpha))
            else:
                # confronto relativo rispetto al target_gap (pretrained or baseline)
                gap_rel_diff = (model_gap - target_gap) / (abs(target_gap) + 1e-8)
                if gap_rel_diff < -self.tol:
                    # gap migliore del target -> più online
                    if hasattr(buf, "adjust_alpha_step"):
                        buf.adjust_alpha_step(-abs(self.step_alpha))
                    else:
                        buf.alpha = max(0.0, buf.alpha - abs(self.step_alpha))
                elif gap_rel_diff > self.tol:
                    # gap peggiore -> più offline
                    if hasattr(buf, "adjust_alpha_step"):
                        buf.adjust_alpha_step(abs(self.step_alpha))
                    else:
                        buf.alpha = min(1.0, buf.alpha + abs(self.step_alpha))
            # aggiorno alpha per logging
            curr_alpha = getattr(buf, "get_alpha", lambda: getattr(buf, "alpha", None))()

        return True