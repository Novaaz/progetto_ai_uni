import csv
import os
import pandas as pd
import numpy as np
from datetime import datetime
from functools import partial
import sys
from stable_baselines3.common.buffers import ReplayBuffer, ReplayBufferSamples
from stable_baselines3.common.save_util import load_from_pkl
import torch as th
import matplotlib.pyplot as plt
import seaborn as sns
from stable_baselines3.common.callbacks import CallbackList

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.training.train_functions import train_sac, AdaptiveAlphaCallback
from src.utils.constants import *
from src.utils.core import *
from src.evaluation.evaluate_functions import evaluate_sac_performance
from gymnasium.wrappers import TransformObservation
import src.utils.noise as noise_mod
from scripts.test_var_groups import create_intelligent_env

def pretraining_clean(seed):
	"""Pre-allenamento ambiente pulito.
	per generare replay buffer per training successivo con rumore"""
	base_env = CityLearnEnv(**ENV_CONFIG)
	env = NormalizedSpaceWrapper(base_env)
	env = StableBaselines3Wrapper(env)
	_, model, _, _ = train_sac(
		env=env,
		seed=seed,
		track_rewards=True,
	)
	model.save_replay_buffer(os.path.join(MODELS_DIR, f"pretrained_replay_buffer_{seed}.pkl"))
	print(f"   ✅ Replay buffer pre-allenato salvato in {MODELS_DIR}/pretrained_replay_buffer_{seed}.pkl")
	return model

def pretraining_noise(seed):
	"""Pre-allenamento ambiente pulito.
	per generare replay buffer per training successivo con rumore"""
	env = create_intelligent_env(seed)
	_, model, _, _ = train_sac(
		env=env,
		seed=seed,
		track_rewards=True,
	)
	model.save_replay_buffer(os.path.join(MODELS_DIR, f"pretrained_replay_buffer_{seed}_noise.pkl"))
	print(f"   ✅ Replay buffer pre-allenato salvato in {MODELS_DIR}/pretrained_replay_buffer_{seed}_noise.pkl")
	return model

class HybridReplayBuffer(ReplayBuffer):
	def __init__(self, 
				 buffer_size,
				 observation_space,
				 action_space,
				 device="cpu",
				 n_envs=1,
				 alpha=0.7,  # proporzione offline
				 offline_pkl_path=None,
				 **kwargs):
		super().__init__(buffer_size, observation_space, action_space, device=device, n_envs=n_envs, **kwargs)
		self.alpha = alpha

		if offline_pkl_path is not None:
			self.offline_buffer = load_from_pkl(offline_pkl_path)
			print(f"   ✅ Replay buffer offline caricato da {offline_pkl_path}")
		else:
			self.offline_buffer = None

		self._alpha_log = []

	def sample(self, batch_size: int, env=None, **kwargs):
		if self.offline_buffer is None:
			return super().sample(batch_size, env=env, **kwargs)

		n_offline = int(self.alpha * batch_size)
		n_online = batch_size - n_offline

		offline_samples = None
		online_samples = None

		if n_offline > 0:
			offline_samples = self.offline_buffer.sample(n_offline, env=env, **kwargs)
		if n_online > 0:
			online_samples = super().sample(n_online, env=env, **kwargs)

		def _cat(a, b):
			if a is None:
				return b
			if b is None:
				return a
			return th.cat([a, b], dim=0)

		def _extract_obs_tensor(samples):
			if samples is None:
				return None
			obs = samples.observations
			if isinstance(obs, dict):
				if 'obs' in obs:
					return obs['obs']
				values = list(obs.values())
				return th.cat(values, dim=-1)
			return obs

		# estrai campioni in forma tensor
		obs_off = _extract_obs_tensor(offline_samples)
		obs_on = _extract_obs_tensor(online_samples)
		obs = _cat(obs_off, obs_on)

		next_off = None
		next_on = None
		if offline_samples is not None:
			no = getattr(offline_samples, 'next_observations', None)
			if isinstance(no, dict):
				next_off = no.get('obs') if 'obs' in no else th.cat(list(no.values()), dim=-1)
			else:
				next_off = no
		if online_samples is not None:
			no = getattr(online_samples, 'next_observations', None)
			if isinstance(no, dict):
				next_on = no.get('obs') if 'obs' in no else th.cat(list(no.values()), dim=-1)
			else:
				next_on = no
		next_obs = _cat(next_off, next_on)

		act = _cat(
			getattr(offline_samples, 'actions', None),
			getattr(online_samples, 'actions', None)
		)
		rew = _cat(
			getattr(offline_samples, 'rewards', None),
			getattr(online_samples, 'rewards', None)
		)
		don = _cat(
			getattr(offline_samples, 'dones', None),
			getattr(online_samples, 'dones', None)
		)

		return ReplayBufferSamples(
			observations=obs,
			actions=act,
			next_observations=next_obs,
			rewards=rew,
			dones=don,
		)

	def set_alpha(self, new_alpha: float):
		"""Imposta alpha (clampa in [0,1])."""
		self.alpha = float(max(0.0, min(1.0, new_alpha)))

	def get_alpha(self) -> float:
		"""Ritorna alpha corrente."""
		return float(self.alpha)

	def adjust_alpha_step(self, step: float):
		"""Aggiunge un incremento a alpha (positivo -> più offline)."""
		self.set_alpha(self.alpha + float(step))

	def log_alpha(self, step_idx: int, reward: float):
		"""Registra uno snapshot (step, alpha, reward) in memoria."""
		self._alpha_log.append({"step": int(step_idx), "alpha": float(self.alpha), "reward": float(reward)})

	def save_alpha_log_csv(self, path: str):
		"""Salva il log alpha -> CSV."""
		if not self._alpha_log:
			return
		df = pd.DataFrame(self._alpha_log)
		os.makedirs(os.path.dirname(path), exist_ok=True)
		df.to_csv(path, index=False)

def train_h20(env, seed, pretrained, isnoise=False, adaptive=False):
	"""Allena modello con rumore H2O.
	Se adaptive=True usa AdaptiveAlphaCallback.
	"""
	print("Creando ambiente con rumore H2O...")

	if isnoise:
		pretrained_path = os.path.join(MODELS_DIR, f"pretrained_replay_buffer_{seed}_noise.pkl")
	else:
		pretrained_path = os.path.join(MODELS_DIR, f"pretrained_replay_buffer_{seed}.pkl")
	if not os.path.exists(pretrained_path):
		print(f"⚠️ Replay buffer pretrained non trovato: {pretrained_path}. Continua senza offline buffer.")
		pretrained_path = None

	# crea modello con HybridReplayBuffer
	model = SAC(
		"MlpPolicy",
		env,
		replay_buffer_class=HybridReplayBuffer,
		replay_buffer_kwargs={
			"alpha": 0.7,
			"offline_pkl_path": pretrained_path,
		},
		**SAC_KWARGS,
		seed=seed,
		verbose=1,
	)

	if adaptive:
		# prepara eval env pulito
		eval_env = StableBaselines3Wrapper(NormalizedSpaceWrapper(CityLearnEnv(**REAL_ENV_CONFIG)))
		eval_env.reset(seed=seed)

		cb = AdaptiveAlphaCallback(
			eval_env=env,
			real_eval_env=eval_env,
			pretrained_model=pretrained,
			eval_freq=env.unwrapped.time_steps * 5,
			n_eval_episodes=1,
			tol=0.02,
			step_alpha=0.1,
			log_path=os.path.join(RESULTS_DIR, "logs", f"adaptive_alpha_cb_{seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"),
			verbose=1
		)
		callbacks = CallbackList([cb])

		time_steps = env.unwrapped.time_steps - 1
		total_timesteps = EPISODES * time_steps
		model.learn(total_timesteps=total_timesteps, reset_num_timesteps=True, callback=callbacks, progress_bar=True)
		print(f"✅ Adaptive (callback) training completato. Log saved in: {cb.log_path}")
		return model

	# altrimenti comportamento standard
	_, model, _, _ = train_sac(env, seed, model, track_rewards=True)
	return model

def clean_env_to_noise_sim(seed):
	pretraining_clean(seed)
	env = create_intelligent_env(seed)
	model = train_h20(env, seed)
	_, baseline_model, _, _ = train_sac(
			env=env,
			seed=seed,
			track_rewards=True,
			time_steps=719,
	)
	return model, baseline_model

def noise_env_to_real_sim(seed, low_noise_vars=None, low_noise_level=0.15):
    """
    Crea e usa un ambiente dove il rumore per alcune variabili (o tutte se low_noise_vars=None)
    è temporaneamente impostato a low_noise_level (es. 0.15). Ripristina la config originale.
    Restituisce (pretrained_model, h20_model, baseline_model)
    """
    orig_config = noise_mod._NOISE_CONFIG.copy()
    new_config = orig_config.copy()

    # pretraining + training sulla config modificata
    pretrained = pretraining_noise(seed)
    if low_noise_vars is None:
        for k in list(new_config.keys()):
            new_config[k] = float(low_noise_level)
    else:
        for k in low_noise_vars:
            new_config[int(k)] = float(low_noise_level)
    noise_mod._NOISE_CONFIG = new_config
    
    env = create_intelligent_env(seed)
    model = train_h20(env, seed, pretrained, isnoise=True, adaptive=True)
    _, baseline_model, _, _ = train_sac(
            env=env,
            seed=seed,
            track_rewards=True,
    )
    # ripristina config originale prima di ritornare
    noise_mod._NOISE_CONFIG = orig_config
    return pretrained, model, baseline_model

def plot_models_step_rewards(models, eval_env, save_dir=None, smooth_window=25):
    """
    Plotta reward per step per ogni modello.
    - colori contrastati (seaborn palette)
    - raw (trasparente) + smooth (rolling mean) + area ± std
    - ritorna lista di risultati (name, res)
    """
    if save_dir is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = os.path.join(RESULTS_DIR, "plots", f"h2o_eval_{ts}")
    os.makedirs(save_dir, exist_ok=True)

    plt.figure(figsize=(12,6))
    all_results = []
    palette = sns.color_palette("tab10", n_colors=max(3, len(models)))

    for idx, entry in enumerate(models):
        name = entry.get('name') or f"model_{idx}"
        model = entry['model']
        res = evaluate_sac_performance(eval_env, model, episode_name=f"eval_{name}")
        all_results.append((name, res))

        step_rewards = res.get('step_rewards') or res.get('episode_rewards') or []
        if isinstance(step_rewards, dict) and 'sum' in step_rewards:
            step_rewards = step_rewards['sum']
        step_rewards = np.array(step_rewards, dtype=float)
        if len(step_rewards) == 0:
            continue

        color = palette[idx % len(palette)]

        # rolling mean and std using pandas for robust edges
        s = pd.Series(step_rewards)
        ma = s.rolling(window=smooth_window, min_periods=1).mean().to_numpy()
        sd = s.rolling(window=smooth_window, min_periods=1).std().fillna(0).to_numpy()

        # plot rolling mean (thicker)
        plt.plot(ma, color=color, linewidth=2.8, label=f"{name} (ma{smooth_window})")

        # shaded area ± std
        upper = ma + sd
        lower = ma - sd
        plt.fill_between(np.arange(len(ma)), lower, upper, color=color, alpha=0.18)

    plt.xlabel("Step")
    plt.ylabel("Reward per step")
    plt.title("Confronto evoluzione reward per modello")
    plt.legend(loc='best', fontsize='small')
    plt.grid(alpha=0.3)
    plt.tight_layout()

    out_path = os.path.join(save_dir, "models_step_rewards.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    return all_results

def _eval_and_save_csv(name, model, eval_env, out_dir, n_eval_episodes=3):
    """
    Valuta `model` su `eval_env` (n_eval_episodes), salva:
    - step rewards file: {name}_steps.csv (step,reward)
    - ritorna mean_final_reward
    """
    os.makedirs(out_dir, exist_ok=True)
    try:
        # use evaluate_sac_performance; support different return formats
        res = evaluate_sac_performance(eval_env, model, episode_name=name)
    except Exception as e:
        print(f"   ⚠️ evaluate_sac_performance fallita per {name}: {e}")
        res = {}

    # extract step rewards if present
    step_rewards = res.get("step_rewards") or res.get("episode_rewards") or []
    # if step_rewards is e.g. dict or nested, try to reduce
    if isinstance(step_rewards, dict):
        # try common keys
        if 'sum' in step_rewards:
            step_rewards = step_rewards['sum']
        else:
            # flatten first numeric list found
            for v in step_rewards.values():
                if isinstance(v, (list, np.ndarray)):
                    step_rewards = v
                    break
    # ensure numpy array
    try:
        step_rewards = np.array(step_rewards, dtype=float)
    except Exception:
        step_rewards = np.array([], dtype=float)

    # save steps CSV if present
    steps_path = os.path.join(out_dir, f"{name}_steps.csv")
    if step_rewards.size > 0:
        df_steps = pd.DataFrame({"step": np.arange(len(step_rewards)), "reward": step_rewards})
        df_steps.to_csv(steps_path, index=False)
    else:
        # create minimal CSV with single empty row if no steps
        pd.DataFrame({"step": [], "reward": []}).to_csv(steps_path, index=False)

    # determine final reward: prefer total_reward, else mean of step_rewards
    final_reward = None
    if "total_reward" in res:
        try:
            final_reward = float(res.get("total_reward"))
        except Exception:
            final_reward = None
    if final_reward is None and step_rewards.size > 0:
        final_reward = float(np.sum(step_rewards)) if step_rewards.ndim == 1 else float(np.mean(step_rewards))

    # also write a tiny summary csv per model
    summary_path = os.path.join(out_dir, f"{name}_summary.csv")
    pd.DataFrame([{"model": name, "final_reward": final_reward}]).to_csv(summary_path, index=False)

    return final_reward, steps_path, summary_path

#main
seed = int(input("Inserisci seed (default 100): ") or "100")
pretrained, model, baseline_model = noise_env_to_real_sim(seed)

# crea directory esperimento
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
exp_dir = os.path.join(RESULTS_DIR, "experiments", f"exp_{ts}_seed{seed}")
models_dir = os.path.join(exp_dir, "models")
evals_dir = os.path.join(exp_dir, "evals")
os.makedirs(models_dir, exist_ok=True)
os.makedirs(evals_dir, exist_ok=True)

# salva i modelli (pretrained, baseline, h20)
pretrained.save(os.path.join(models_dir, f"pretrained_seed{seed}.zip"))
baseline_model.save(os.path.join(models_dir, f"baseline_seed{seed}.zip"))
model.save(os.path.join(models_dir, f"h20_seed{seed}.zip"))

# prepara eval env reale
eval_env = StableBaselines3Wrapper(NormalizedSpaceWrapper(CityLearnEnv(**REAL_ENV_CONFIG)))
eval_env.reset(seed=seed)

# valuta e salva csv per ogni modello; raccogli summary in una tabella
summary_rows = []
for name, mdl in [("pretrained", pretrained), ("baseline", baseline_model), ("h20", model)]:
    final_reward, steps_path, summary_path = _eval_and_save_csv(name, mdl, eval_env, evals_dir, n_eval_episodes=3)
    summary_rows.append({"model": name, "final_reward": final_reward, "steps_csv": os.path.basename(steps_path), "summary_csv": os.path.basename(summary_path)})

# salva summary globale in una riga per modello
summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(os.path.join(exp_dir, f"eval_summary_seed{seed}.csv"), index=False)

# plot reward evolution for quick visual check
plot_models_step_rewards(
    [
        {'name': 'intelligent_noise', 'model': model},
        {'name': 'baseline', 'model': baseline_model},
    ],
    eval_env,
    save_dir=os.path.join(exp_dir, "plots")
)

