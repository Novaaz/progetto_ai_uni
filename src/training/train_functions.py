import os
import numpy as np
from stable_baselines3 import SAC
from ..utils.constants import *
from ..utils.core import *
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
        sac_model = SAC(policy='MlpPolicy',
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

class OnlineFinetuningRewardTracker(BaseCallback):
    """
    Callback per tracciare le reward durante il fine-tuning online.
    Raccoglie reward step-by-step durante l'interazione con l'ambiente.
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.step_rewards = []
        self.timesteps_evaluated = []
        self.current_step = 0
        
    def _on_step(self) -> bool:
        # Raccoglie la reward del passo corrente
        try:
            if 'rewards' in self.locals and self.locals['rewards'] is not None:
                reward = self.locals['rewards']
                # Gestisci diversi formati di reward
                if hasattr(reward, '__iter__') and not isinstance(reward, str):
                    reward_val = float(np.mean(reward))
                else:
                    reward_val = float(reward)
                
                self.step_rewards.append(reward_val)
                self.timesteps_evaluated.append(self.n_calls)
                self.current_step += 1
                
                if self.verbose > 0 and self.current_step % 100 == 0:
                    print(f"Fine-tuning Step {self.current_step}: Reward = {reward_val:.2f}")
                    
        except Exception as e:
            if self.verbose > 1:
                print(f"Errore nell'accesso alle reward durante fine-tuning: {e}")
        
        return True
    
    def get_data(self):
        """Restituisce i dati raccolti durante il fine-tuning"""
        return {
            'step_rewards': self.step_rewards,
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