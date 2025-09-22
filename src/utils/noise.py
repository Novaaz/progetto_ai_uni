import numpy as np

_NOISE_MEMORY = {}
_NOISE_RAMPUP = 0.1
_NOISE_RAMPUP_STEP = 0.15
_NOISE_RAMPUP_EP = 4
_NOISE_CONFIG = {
	# ALTO RUMORE (0.8) - Variabili che beneficiano
	3: 0.8,   # outdoor_dry_bulb_temperature
	7:  0.65,  # carbon_intensity 0.5 è buono 0.65 meglio
	#8: 0.8,   # non_shiftable_load 
	9: 0.8,   # solar_generation
	11: 0.8,  # net_electricity_consumption
	19: 0.8,  # dhw_demand
			
	# BASSO RUMORE (0.15) - Variabili neutre/sensibili ma non dannose
	4: 0.15,   # outdoor_relative_humidity
	15: 0.15,  # indoor_dry_bulb_temperature
	20: 0.15,  # cooling_device_efficiency
	21: 0.15,  # dhw_device_efficiency
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

def add_noise_to_observations(observations, noise_type='gaussian', dinamic_noise=False, name=None, seed=None):
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

		if memory_data['current_step'] % 719 == 0:
			memory_data['episodes_completed'] += 1
			if memory_data['episodes_completed'] % _NOISE_RAMPUP_EP == 0:
				_NOISE_RAMPUP = _NOISE_RAMPUP + _NOISE_RAMPUP_STEP
				print(f"🔊 [{name}] Rumore incrementato a {_NOISE_RAMPUP:.2f} dopo {memory_data['episodes_completed']} episodi")

		noise = _generate_noise_array(len(noisy_observations), noise_type, memory_data['current_step'], seed)
	else:
		
		config_key = f"{name}_{noise_type}_{seed}"
		
		if config_key not in _NOISE_MEMORY:
			print(f"🔊 Inizializzando memoria rumore per '{name}' - Generando 719 step di rumore (seed={seed})")
			_NOISE_RAMPUP = 1.0  # così viene sempre skippato il min() alla generazione
			episode_noise = []
			for step in range(720):
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
		
		memory_data['current_step'] = (current_step + 1) % 719
		
		if memory_data['current_step'] == 0:
			memory_data['episodes_completed'] += 1

	noisy_observations += noise
	
	return noisy_observations