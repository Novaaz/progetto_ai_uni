import numpy as np
from stable_baselines3 import SAC
from ..utils.constants import *
from ..utils.core import *
from ..models.ensemble import WeightedEnsemble
from ..evaluation.evaluate_functions import quick_evaluate
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from gymnasium.wrappers import TransformObservation
from functools import partial

_NOISE_MEMORY = {}

def _generate_noise_array(size, noise_type='gaussian', noise_level=0.15, noise_mean=0.0):
	"""
	Genera un array di rumore con i parametri specificati.
	
	Parametri:
	size: int - Dimensione dell'array di rumore
	noise_type: str - Tipo di rumore ('gaussian' o 'uniform')
	noise_level: float - Livello di rumore (std per gaussian, range per uniform)
	noise_mean: float - Media del rumore (solo per gaussian)
	
	Returns:
	np.ndarray - Array di rumore generato
	"""
	if noise_type == 'gaussian':
		return np.random.normal(loc=noise_mean, scale=noise_level, size=size)
	elif noise_type == 'uniform':
		return np.random.uniform(low=-noise_level, high=noise_level, size=size)
	else:
		raise ValueError(f"Tipo di rumore '{noise_type}' non supportato. Usa 'gaussian' o 'uniform'.")

def add_noise_to_observations(observations, noise_level=0.15, noise_mean=0.0, noise_type='gaussian', dinamic_noise=False, name=None):
	"""
	Aggiunge rumore alle osservazioni in base al tipo di rumore specificato.
	Se dinamic_noise=False, il rumore viene pre-generato e riutilizzato per ogni step.
	
	Parametri:
	observations: np.ndarray - Osservazioni originali
	noise_level: float - Livello di rumore da aggiungere
	noise_mean: float - Media del rumore (solo per rumore gaussiano)
	noise_type: str - Tipo di rumore ('gaussian' o 'uniform')
	dinamic_noise: bool - Se True, genera rumore nuovo ad ogni step. Se False, usa rumore pre-generato
	name: str - Nome univoco per identificare questa istanza di rumore (necessario se dinamic_noise=False)
	
	Returns:
	np.ndarray - Osservazioni con rumore aggiunto
	"""    
	global _NOISE_MEMORY
	
	obs = np.array(observations)
	noisy_observations = obs.copy()

	mask = np.zeros_like(noisy_observations, dtype=bool)
	indices = [0, 1, 2, -1, -2, -3, -4]  # day_type, hour, occupant_count, power_outage, cooling_set_point
	mask[indices] = True

	if dinamic_noise:
		 noise = _generate_noise_array(len(noisy_observations), noise_type, noise_level, noise_mean)
	else:
		if name is None:
			raise ValueError("Il parametro 'name' è obbligatorio quando dinamic_noise=False")
		
		config_key = f"{name}_{noise_type}_{noise_level}_{noise_mean}_{len(obs)}"
		
		if config_key not in _NOISE_MEMORY:
			print(f"🔊 Inizializzando memoria rumore per '{name}' - Generando 719 step di rumore")
			
			episode_noise = []
			for step in range(719):
				step_noise = _generate_noise_array(len(noisy_observations), noise_type, noise_level, noise_mean)
				episode_noise.append(step_noise)
			
			_NOISE_MEMORY[config_key] = {
				'episode_noise': episode_noise,
				'current_step': 0,
				'total_calls': 0,
				'episodes_completed': 0
			}
		
		memory_data = _NOISE_MEMORY[config_key]
		current_step = memory_data['current_step']
		noise = memory_data['episode_noise'][current_step]
		
		memory_data['total_calls'] += 1
		memory_data['current_step'] = (current_step + 1) % 719
		
		if memory_data['current_step'] == 0:
			memory_data['episodes_completed'] += 1

	noisy_observations += noise
	noisy_observations[mask] = obs[mask]
	
	return noisy_observations

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
	Allena un singolo modello SAC con un livello di rumore specificato
	
	Parameters:
	noise_std: float - Deviazione standard del rumore gaussiano
	seed: int - Seed per la riproducibilità
	model_id: int - ID del modello per il tracking
	
	Returns:
	tuple: (modello_allenato, dati_training, noise_std_usato)
	"""
	print(f"📈 Training Modello {model_id} | Noise STD: {noise_std:.4f}")
	
	try:
		if noise_std == 0.0:
			train_env = CityLearnEnv(**ENV_CONFIG)
			train_env = StableBaselines3Wrapper(NormalizedSpaceWrapper(train_env))
			print(f"  🔹 Ambiente pulito (no noise)")
		else:
			train_env = CityLearnEnv(**ENV_CONFIG)
			train_env = StableBaselines3Wrapper(NormalizedSpaceWrapper(train_env))
			train_env = TransformObservation(train_env, partial(
				add_noise_to_observations, 
				noise_type='gaussian',
				dinamic_noise=dinamic_noise,
				name=model_id,
				noise_level=noise_std,
				noise_mean=0.0
			))
			print(f"  🔸 Ambiente con rumore gaussiano (σ={noise_std:.4f}) dinamico = {dinamic_noise}")
		
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
			'model_id': model_id
		}
		
		print(f"  ✅ Training completato | Episodi: {len(training_rewards)}")
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





