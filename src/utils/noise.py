import numpy as np
import sys
import os
from gymnasium.wrappers import TransformObservation
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.training.train_functions import train_sac
from src.utils.constants import *
from src.utils.core import *

_NOISE_MEMORY = {}
_NOISE_RAMPUP = 0.1
_NOISE_RAMPUP_STEP = 0.15
_NOISE_RAMPUP_EP = 4
_NOISE_CONFIG = {
	3:  0.0,   # outdoor_dry_bulb_temperature
	7:  0.0,  # carbon_intensity
	9:  0.0,   # solar_generation
	11: 0.0,  # net_electricity_consumption
	19: 0.0,  # dhw_demand
	4:  0.0,   # outdoor_relative_humidity
	15: 0.0,  # indoor_dry_bulb_temperature
	20: 0.0,  # cooling_device_efficiency
	21: 0.0,  # dhw_device_efficiency
}


def _generate_noise_array(obs, noise_type='gaussian', step=0, seed=None):
	"""
	Genera un array di rumore con i parametri specificati.

	Returns:
	np.ndarray - Array di rumore generato
	"""
	global _NOISE_CONFIG
	noise= np.zeros(obs, dtype=float)
	rng = None
	for i in range(obs):
		rng = np.random.default_rng(seed+(step+1)+i) if seed is not None else np.random.default_rng()
		if i in _NOISE_CONFIG:
			noise_level = min(_NOISE_CONFIG[i], _NOISE_RAMPUP)
		else:
			noise_level = 0.0

		if noise_type == 'gaussian':
			_noise = rng.normal(0, noise_level)
			original = noise[i]
			noise[i] = np.clip(original + _noise, 0.0, 1.0)
		elif noise_type == 'uniform':
			_noise = rng.uniform(-noise_level, noise_level)
			original = noise[i]
			noise[i] = np.clip(original + _noise, 0.0, 1.0)
		else:
			raise ValueError(f"Tipo di rumore '{noise_type}' non supportato. Usa 'gaussian' o 'uniform'.")
	return noise	

def add_noise_to_observations(observations, steps, noise_type='gaussian', dinamic_noise=False, name=None, seed=None):
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
	seed: int - Seed per la riproducibilità (opzionale)
	
	Returns:
	np.ndarray - Osservazioni con rumore aggiunto
	"""    
	global _NOISE_MEMORY, _NOISE_CONFIG, _NOISE_RAMPUP, _NOISE_RAMPUP_STEP, _NOISE_RAMPUP_EP
	
	obs = np.array(observations)
	noisy_observations = obs.copy()

	if name is None:
		raise ValueError("Il parametro 'name' è obbligatorio")

	if dinamic_noise:
		# Tieni traccia dello step per l'ambiente dinamico
		config_key = f"{name}_{noise_type}_{seed}_dynamic"
		
		if config_key not in _NOISE_MEMORY:
			_NOISE_MEMORY[config_key] = {
				'current_step': 0,
				'total_calls': 0,
				'episodes_completed': 0,
				'seed': seed
			}
			_NOISE_RAMPUP = 0.1  # Reset ramp-up per nuova istanza
		
		memory_data = _NOISE_MEMORY[config_key]
		memory_data['current_step'] = (memory_data['current_step'] + 1)

		if memory_data['current_step'] % steps == 0:
			memory_data['episodes_completed'] += 1
			if memory_data['episodes_completed'] % _NOISE_RAMPUP_EP == 0:
				_NOISE_RAMPUP = _NOISE_RAMPUP + _NOISE_RAMPUP_STEP
				print(f"🔊 [{name}] Rumore incrementato a {_NOISE_RAMPUP:.2f} dopo {memory_data['episodes_completed']} episodi")

		noise = _generate_noise_array(len(noisy_observations), noise_type, memory_data['current_step'], seed)
	else:
		
		config_key = f"{name}_{noise_type}_{seed}"
		
		if config_key not in _NOISE_MEMORY:
			print(f"🔊 Inizializzando memoria rumore per '{name}' - Generando {steps} step di rumore (seed={seed})")
			_NOISE_RAMPUP = 1.0  # così viene sempre skippato il min() alla generazione
			episode_noise = []
			for step in range(steps):
				step_noise = _generate_noise_array(len(noisy_observations), noise_type, step, seed)
				episode_noise.append(step_noise)
			
			_NOISE_MEMORY[config_key] = {
				'episode_noise': episode_noise,
				'current_step': 0,
				'episodes_completed': 0,
				'seed': seed
			}
		
		memory_data = _NOISE_MEMORY[config_key]
		current_step = memory_data['current_step']
		noise = memory_data['episode_noise'][current_step]
		
		memory_data['current_step'] = (current_step + 1) % steps
		
		if memory_data['current_step'] == 0:
			memory_data['episodes_completed'] += 1

	noisy_observations += noise
	
	return noisy_observations

def create_intelligent_env(seed=100, dinamic_noise=False, noise=None):
    """
    Crea ambiente con rumore "intelligente".
    Parametri:
    - seed: seed per riproducibilità
    - dinamic_noise: se True genera rumore dinamico ogni step
    - noise: se None usa NOISE_CONFIG; se float imposta tutti i key di NOISE_CONFIG a quel valore;
             se dict permette sovrascrivere singole variabili {idx: value}
    - persist_noise: se True scrive la nuova config in noise_mod._NOISE_CONFIG (comportamento corrente).
                     Se False non modifica la config globale ma la applica comunque all'ambiente (meno comune).
    Nota: add_noise_to_observations usa la variabile globale _NOISE_CONFIG, quindi per avere una config
          dedicata all'ambiente aggiorniamo quella globale (persist_noise=True).
    """
    global _NOISE_CONFIG, _NOISE_MEMORY, _NOISE_RAMPUP

    print("🧠 Creando ambiente con rumore intelligente...")
    env = CityLearnEnv(**ENV_CONFIG)
    env = StableBaselines3Wrapper(NormalizedSpaceWrapper(env))

    # salva config originale e costruisci nuova config in base all'argomento `noise`
    orig_config = _NOISE_CONFIG.copy() if isinstance(_NOISE_CONFIG, dict) else {}
    if noise is None:
        new_config = orig_config.copy()
    elif isinstance(noise, dict):
        new_config = orig_config.copy()
        for k, v in noise.items():
            new_config[int(k)] = float(v)
    else:
        new_config = {int(k): float(noise) for k in orig_config.keys()}

    print(new_config)

    name_token = f"intelligent_{seed}"
    if noise is not None:
        name_token += f"_{str(noise).replace(' ', '')}"

    # aggiorniamo la config globale e rimuoviamo eventuale cache collegata al name_token
    _NOISE_CONFIG = new_config.copy()
    for k in list(_NOISE_MEMORY.keys()):
        if k.startswith(f"{name_token}_") or k.startswith(f"{name_token}"):
            del _NOISE_MEMORY[k]

    def _obs_with_intelligent_noise(obs):
        arr = np.array(obs, dtype=float)
        return add_noise_to_observations(
            arr,
			steps=env.unwrapped.time_steps,
            noise_type='gaussian',
            dinamic_noise=dinamic_noise,
            name=name_token,
            seed=seed
        )

    env = TransformObservation(env, f=_obs_with_intelligent_noise)
    env.reset(seed=seed)

    print("   ✅ Ambiente con rumore intelligente creato (noise applied)")
    return env