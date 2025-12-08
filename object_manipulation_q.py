"""
FetchReach Q-Learning with SAC/TD3
"""

import os
import numpy as np
from dataclasses import dataclass

try:
    os.environ['MUJOCO_GL'] = 'egl'
    import mujoco
    RENDERING_AVAILABLE = True
except:
    try:
        os.environ['MUJOCO_GL'] = 'osmesa'
        import mujoco
        RENDERING_AVAILABLE = True
    except:
        RENDERING_AVAILABLE = False
    
import gymnasium as gym
from gymnasium.wrappers import FlattenObservation
import gymnasium_robotics

from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.noise import NormalActionNoise

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import imageio
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from stable_baselines3.common.callbacks import BaseCallback

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")


ENV_ID = "FetchReach-v4"

EXPERT_MODEL_PATH = "ppo_fetchreach_expert"
SAC_MODEL_PATH = "sac_fetchreach_qlearning"
TD3_MODEL_PATH = "td3_fetchreach_qlearning"
DATASET_PATH = "fetchreach_expert_dataset.npz"
DIFFUSION_MODEL_PATH = "diffusion_fetchreach_policy.pt"

Q_LEARNING_ALGO = "SAC"

TRAIN_EXPERT = False
COLLECT_DATA = False
TRAIN_DIFFUSION = False
TRAIN_Q_LEARNING = True
EVAL_Q_LEARNING = True
EVAL_DIFFUSION = False

RECORD_VIDEOS = RENDERING_AVAILABLE
VIDEO_FORMAT = "mp4"

NUM_ENVS = 8 if torch.cuda.is_available() else 4
NUM_WORKERS = 4 if torch.cuda.is_available() else 2
USE_AMP = torch.cuda.is_available()

def make_env(render_mode, rank=0, seed=0):
    def _init():
        env = gym.make(ENV_ID, reward_type="dense", render_mode=render_mode)
        env = FlattenObservation(env)
        env.reset(seed=seed + rank)
        set_random_seed(seed + rank)
        return env
    return _init

def make_vec_env(n_envs, render_mode=None, seed=0):
    if n_envs == 1:
        return DummyVecEnv([make_env(render_mode, 0, seed)])
    else:
        return SubprocVecEnv([make_env(render_mode, i, seed) for i in range(n_envs)])


def train_expert(total_timesteps: int = 2_000_000, tuning: bool = True):
    env = make_vec_env(NUM_ENVS, render_mode=None, seed=42)
    policy_kwargs = dict(net_arch=[256, 256])

    if tuning and os.path.exists(EXPERT_MODEL_PATH + ".zip"):
        model = PPO.load(EXPERT_MODEL_PATH, env=env, device=device)
        model.set_env(env)
        model.learn(total_timesteps=total_timesteps, reset_num_timesteps=False)
    else:
        model = PPO(
            "MlpPolicy", env,
            learning_rate=3e-4, n_steps=2048, batch_size=256, n_epochs=10,
            gamma=0.98, gae_lambda=0.95, ent_coef=0.0, vf_coef=0.5,
            max_grad_norm=0.5, policy_kwargs=policy_kwargs,
            verbose=1, device=device,
        )
        model.learn(total_timesteps=total_timesteps)

    model.save(EXPERT_MODEL_PATH)
    env.close()


class QLearningEvalCallback(BaseCallback):
    def __init__(self, eval_env_fn, n_eval_episodes=10, eval_freq=5000, verbose=1):
        super().__init__(verbose)
        self.eval_env_fn = eval_env_fn
        self.n_eval_episodes = n_eval_episodes
        self.eval_freq = eval_freq
        self.eval_results = []  # [(timesteps, success_rate), ...]
        self.loss_history = []  # [(timesteps, loss), ...]
        self._last_loss = 0.0
    
    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            if hasattr(self.model, 'logger') and self.model.logger is not None:
                try:
                    if hasattr(self.logger, 'name_to_value'):
                        loss = self.logger.name_to_value.get('train/critic_loss', self._last_loss)
                        if loss != 0:
                            self._last_loss = loss
                except:
                    pass
            
            self.loss_history.append((self.num_timesteps, self._last_loss))
            
            env = self.eval_env_fn()
            success = 0.0
            for _ in range(self.n_eval_episodes):
                obs, info = env.reset()
                done = False
                while not done:
                    obs_batch = obs[None, :]
                    action, _ = self.model.predict(obs_batch, deterministic=True)
                    action = action[0]
                    obs, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated
                success += info.get("is_success", 0.0)
            env.close()
            
            success_rate = success / self.n_eval_episodes
            self.eval_results.append((self.num_timesteps, success_rate))
            
            if self.verbose > 0:
                print(f"[Eval] Steps: {self.num_timesteps:,}, Success: {success_rate:.1%}, Loss: {self._last_loss:.4f}")
        
        return True
    
    def _on_rollout_end(self) -> None:
        if hasattr(self.model, 'logger') and self.model.logger is not None:
            try:
                if hasattr(self.logger, 'name_to_value'):
                    loss = self.logger.name_to_value.get('train/critic_loss', 0)
                    if loss != 0:
                        self._last_loss = loss
            except:
                pass


def train_q_learning(algo: str = "SAC", total_timesteps: int = 500_000, tuning: bool = True, eval_freq: int = 5000):
    env = make_vec_env(1, render_mode=None, seed=42)
    n_actions = env.action_space.shape[-1]
    model_path = SAC_MODEL_PATH if algo == "SAC" else TD3_MODEL_PATH
    policy_kwargs = dict(net_arch=[256, 256])
    
    if algo == "SAC":
        # ===== SAC (Soft Actor-Critic) =====
        # Best for continuous control, uses entropy regularization
        if tuning and os.path.exists(model_path + ".zip"):
            print(f"🔄 Loading existing SAC model and continuing training: {model_path}.zip")
            model = SAC.load(model_path, env=env, device=device)
            model.set_env(env)
        else:
            print("🆕 Creating new SAC model and training from scratch")
            model = SAC(
                "MlpPolicy",
                env,
                learning_rate=3e-4,           # Learning rate for all networks
                buffer_size=1_000_000,        # Replay buffer size (Q-learning needs this!)
                learning_starts=1000,         # Steps before training starts
                batch_size=256,               # Batch size for training
                tau=0.005,                    # Soft update coefficient (target network)
                gamma=0.99,                   # Discount factor
                train_freq=1,                 # Train after every step
                gradient_steps=1,             # Gradient steps per environment step
                ent_coef="auto",              # Automatic entropy tuning (SAC special)
                target_entropy="auto",        # Target entropy for automatic tuning
                policy_kwargs=policy_kwargs,
                verbose=1,
                device=device,
            )
        
    elif algo == "TD3":
        # ===== TD3 (Twin Delayed DDPG) =====
        # Uses twin Q-networks and delayed policy updates
        action_noise = NormalActionNoise(
            mean=np.zeros(n_actions),
            sigma=0.1 * np.ones(n_actions)
        )
        
        if tuning and os.path.exists(model_path + ".zip"):
            print(f"🔄 Loading existing TD3 model and continuing training: {model_path}.zip")
            model = TD3.load(model_path, env=env, device=device)
            model.set_env(env)
        else:
            print("🆕 Creating new TD3 model and training from scratch")
            model = TD3(
                "MlpPolicy",
                env,
                learning_rate=3e-4,
                buffer_size=1_000_000,
                learning_starts=1000,
                batch_size=256,
                tau=0.005,
                gamma=0.99,
                train_freq=1,
                gradient_steps=1,
                action_noise=action_noise,    # Exploration noise
                policy_delay=2,               # Delay policy updates (TD3 special)
                target_policy_noise=0.2,      # Target policy smoothing noise
                target_noise_clip=0.5,        # Clip target noise
                policy_kwargs=policy_kwargs,
                verbose=1,
                device=device,
            )
    else:
        raise ValueError(f"Unknown algorithm: {algo}. Choose 'SAC' or 'TD3'")
    
    # Create evaluation callback
    eval_callback = QLearningEvalCallback(
        eval_env_fn=make_env(render_mode=None, rank=0, seed=123),
        n_eval_episodes=10,
        eval_freq=eval_freq,
        verbose=1,
    )
    
    # Train!
    print(f"\n📚 Starting Q-Learning training for {total_timesteps:,} timesteps...")
    print(f"   This may take a while. The agent is learning by exploring the environment.")
    print(f"   Watch the ep_rew_mean (episode reward) - it should increase over time!\n")
    
    model.learn(
        total_timesteps=total_timesteps,
        reset_num_timesteps=not tuning,  # Continue from previous if tuning
        progress_bar=True,  # Show progress bar
        callback=eval_callback,
    )
    
    model.save(model_path)
    env.close()
    
    history = {
        'success_rate': eval_callback.eval_results,
        'loss': eval_callback.loss_history,
    }
    return model, history


def evaluate_q_learning(
    algo: str = "SAC",
    n_episodes: int = 50,
    save_video: bool = True,
    video_dir: str = "videos_qlearning",
):
    """
    Evaluate trained Q-Learning policy.
    """
    model_path = SAC_MODEL_PATH if algo == "SAC" else TD3_MODEL_PATH
    
    assert os.path.exists(model_path + ".zip"), \
        f"{algo} model not found. Please run train_q_learning() first."
    
    print(f"🔎 Evaluating Q-Learning ({algo}) policy...")
    
    # Load model
    env = make_vec_env(1, render_mode=None, seed=42)
    if algo == "SAC":
        model = SAC.load(model_path, env=env, device=device)
    else:
        model = TD3.load(model_path, env=env, device=device)
    
    # Evaluate without video first
    success = 0
    total_return = 0
    
    for ep in range(n_episodes):
        obs = env.reset()
        done = False
        ep_return = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, dones, infos = env.step(action)
            done = dones[0]
            ep_return += reward[0]
            
            if done:
                success += infos[0].get("is_success", 0.0)
        
        total_return += ep_return
        
        if (ep + 1) % 10 == 0:
            print(f"[Q-Learning Eval] Episode {ep+1}/{n_episodes}, "
                  f"Running success rate: {success/(ep+1):.3f}")
    
    env.close()
    
    avg_return = total_return / n_episodes
    success_rate = success / n_episodes
    
    print(f"\n📊 Q-Learning ({algo}) Results:")
    print(f"   Success rate: {success_rate:.3f} ({int(success)}/{n_episodes})")
    print(f"   Average return: {avg_return:.3f}")
    
    # Record videos if enabled
    if save_video and RECORD_VIDEOS:
        print(f"\n🎥 Recording videos...")
        record_q_learning_video(algo=algo, n_episodes=5, video_dir=video_dir)
    
    return success_rate, avg_return


def record_q_learning_video(
    algo: str = "SAC",
    n_episodes: int = 5,
    max_steps: int = 50,
    video_dir: str = "videos_qlearning",
):
    """
    Record videos of Q-Learning policy in action.
    """
    model_path = SAC_MODEL_PATH if algo == "SAC" else TD3_MODEL_PATH
    
    assert os.path.exists(model_path + ".zip"), f"{algo} model not found!"
    
    os.makedirs(video_dir, exist_ok=True)
    
    # Create rendering environment
    env_fn = make_env(render_mode="rgb_array", rank=0, seed=42)
    env = env_fn()
    
    # Load model with dummy vec env
    dummy_vec_env = make_vec_env(1, render_mode=None, seed=42)
    if algo == "SAC":
        model = SAC.load(model_path, env=dummy_vec_env, device=device)
    else:
        model = TD3.load(model_path, env=dummy_vec_env, device=device)
    dummy_vec_env.close()
    
    success = 0
    
    for ep in range(n_episodes):
        obs, info = env.reset()
        done = False
        frames = []
        step = 0
        ep_return = 0.0
        
        while not done and step < max_steps:
            # Get action from Q-learning policy
            obs_batch = obs[None, :]
            action, _ = model.predict(obs_batch, deterministic=True)
            action = action[0]
            
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_return += reward
            step += 1
            
            try:
                frame = env.render()
                if frame is not None:
                    frames.append(frame)
            except Exception as e:
                print(f"⚠️ Rendering failed: {e}")
                break
        
        success += info.get("is_success", 0.0)
        print(f"[Q-Learning Video] Episode {ep+1}/{n_episodes} return: {ep_return:.3f}, "
              f"success: {info.get('is_success', 0.0)}")
        
        # Save video
        if len(frames) > 0:
            if VIDEO_FORMAT == "mp4":
                video_path = os.path.join(video_dir, f"{algo.lower()}_ep_{ep+1:03d}.mp4")
                imageio.mimsave(video_path, frames, fps=30, codec='libx264')
            else:
                video_path = os.path.join(video_dir, f"{algo.lower()}_ep_{ep+1:03d}.gif")
                imageio.mimsave(video_path, frames, fps=15)
            
            file_size = os.path.getsize(video_path) / 1024
            print(f"  🎥 Saved: {video_path} ({file_size:.1f} KB)")
    
    print(f"\n[Q-Learning] Success rate: {success/n_episodes:.3f}")
    env.close()


def collect_expert_data(
    n_episodes: int = 200,
    max_steps_per_episode: int = 50,
):
    """
    Play FetchReach for multiple episodes with trained expert model
    and collect (obs, action) pairs to save as .npz.

    This dataset will be used for Behavior Cloning and Diffusion Policy training later.
    Both are Imitation Learning, but BC is simpler.
    IL = Strong Supervision
    RL = Weak Supervision (or Sparse Supervision)
    """
    assert os.path.exists(EXPERT_MODEL_PATH + ".zip"), \
        "Expert model not found. Please run train_expert() first."

    # Round up to nearest multiple of NUM_ENVS for efficient parallel collection
    adjusted_episodes = ((n_episodes + NUM_ENVS - 1) // NUM_ENVS) * NUM_ENVS
    if adjusted_episodes != n_episodes:
        print(f"ℹ️  Adjusted episodes from {n_episodes} to {adjusted_episodes} (multiple of {NUM_ENVS})")
    
    print(f"🚀 Collecting data with {NUM_ENVS} parallel environments")
    env = make_vec_env(NUM_ENVS, render_mode=None, seed=42)
    model = PPO.load(EXPERT_MODEL_PATH, env=env, device=device)

    observations = []
    actions = []
    
    episodes_per_env = adjusted_episodes // NUM_ENVS

    for batch_idx in range(episodes_per_env):
        obs = env.reset()
        for t in range(max_steps_per_episode):
            # Select expert action (batch prediction on GPU)
            action, _ = model.predict(obs, deterministic=True)

            observations.append(obs)
            actions.append(action)

            obs, reward, dones, infos = env.step(action)
            
        if (batch_idx + 1) % 10 == 0:
            print(f"[Expert Rollout] Collected {(batch_idx + 1) * NUM_ENVS}/{adjusted_episodes} episodes")

    env.close()

    # Flatten the batch dimension
    observations = np.concatenate(observations, axis=0).astype(np.float32)
    actions = np.concatenate(actions, axis=0).astype(np.float32)

    np.savez(DATASET_PATH, observations=observations, actions=actions)
    print(f"✅ Dataset saved to: {DATASET_PATH}")
    print(f"   observations: {observations.shape}, actions: {actions.shape}")


# ===========================
# 2. Dataset for Diffusion Policy
# ===========================
class FetchReachExpertDataset(Dataset):
    """
    Dataset for (obs, action) supervised learning.
    Using single-step action diffusion (H=1) here,
    can be extended to H>1 trajectory later.
    """
    def __init__(self, npz_path: str):
        data = np.load(npz_path)
        self.obs = data["observations"]   # shape: (N, obs_dim)
        self.actions = data["actions"]    # shape: (N, act_dim)
        assert self.obs.shape[0] == self.actions.shape[0]

    def __len__(self):
        return self.obs.shape[0]

    def __getitem__(self, idx):
        return self.obs[idx], self.actions[idx]


# ===========================
# 3. Simple DDPM-style Diffusion Model (Action Only, Cond on Obs)
# ===========================
@dataclass
class DiffusionConfig:
    action_dim: int # Size of continuous vector that diffusion model should output
    obs_dim: int # Size of input vector that diffusion model conditions on
    timesteps: int = 100       # diffusion steps T
    beta_start: float = 1e-4 # Gradually increase noise size (beta) at each step to add noise to data
    beta_end: float = 0.02
    hidden_dim: int = 256 # MLP hidden layer size


class TimeEmbedding(nn.Module): # Need to inform current timestep t. t'th denoising step. Convert t to high-dimensional vector, like positional embedding.
    # Unique representation for each t is possible.
    """
    Simple sinusoidal time embedding + MLP.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.lin = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, t: torch.Tensor): # sinusoidal positional encoding (famous)
        """
        t: (B,) integer timesteps
        """
        half_dim = self.dim // 2
        # sinusoidal embedding
        freq = torch.exp(
            torch.arange(half_dim, device=t.device) *
            -(np.log(10000.0) / (half_dim - 1))
        )
        args = t.float().unsqueeze(1) * freq.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return self.lin(emb)


class DiffusionPolicy(nn.Module):
    """
    Network that predicts epsilon_theta(x_t, t, obs).
    Input: noisy action (x_t), time embedding, obs
    Output: noise epsilon_hat (action_dim)
    """
    def __init__(self, cfg: DiffusionConfig):
        super().__init__()
        self.cfg = cfg
        self.time_mlp = TimeEmbedding(cfg.hidden_dim)

        in_dim = cfg.action_dim + cfg.obs_dim + cfg.hidden_dim
        h = cfg.hidden_dim

        self.net = nn.Sequential(
            nn.Linear(in_dim, h),
            nn.ReLU(),
            nn.Linear(h, h),
            nn.ReLU(),
            nn.Linear(h, cfg.action_dim),
        )

    def forward(self, x_t, t, obs):
        """
        x_t: (B, action_dim)
        t:   (B,) int64 timesteps
        obs: (B, obs_dim)
        """
        t_emb = self.time_mlp(t)  # (B, hidden_dim)
        x = torch.cat([x_t, obs, t_emb], dim=-1)
        eps_hat = self.net(x)
        return eps_hat # Predict the noise (eps_hat) added to original action


class ActionDiffusion:
    """
    Helper class that manages DDPM-style forward/inverse process.
    """
    def __init__(self, cfg: DiffusionConfig):
        self.cfg = cfg

        # beta schedule (linear)
        betas = torch.linspace(cfg.beta_start, cfg.beta_end, cfg.timesteps) # noise
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0) # Cumulative to t steps, how much signal (original data) remains

        self.register_buffers(betas, alphas, alphas_cumprod)

    def register_buffers(self, betas, alphas, alphas_cumprod):
        self.betas = betas.to(device)
        self.alphas = alphas.to(device)
        self.alphas_cumprod = alphas_cumprod.to(device)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

    def q_sample(self, x0, t, noise=None):
        """
        q(x_t | x_0) = sqrt(alpha_bar_t) x_0 + sqrt(1 - alpha_bar_t) noise
        """
        if noise is None:
            noise = torch.randn_like(x0)

        sqrt_alpha_bar_t = self.sqrt_alphas_cumprod[t].unsqueeze(-1)
        sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alphas_cumprod[t].unsqueeze(-1)

        return sqrt_alpha_bar_t * x0 + sqrt_one_minus_alpha_bar_t * noise

    def p_sample(self, model: DiffusionPolicy, x_t, t, obs, add_noise: bool = True):
        """
        One reverse step: p_theta(x_{t-1} | x_t)
        """
        beta_t = self.betas[t].unsqueeze(-1)          # (B, 1)
        sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alphas_cumprod[t].unsqueeze(-1)
        sqrt_recip_alpha_t = (1.0 / torch.sqrt(self.alphas[t])).unsqueeze(-1)

        # model predicts epsilon
        eps_theta = model(x_t, t, obs)

        # DDPM formula: x_{t-1} = 1/sqrt(alpha_t) * (x_t - (1-alpha_t)/sqrt(1 - alpha_bar_t) * eps_theta) + sigma_t z
        coef = beta_t / sqrt_one_minus_alpha_bar_t
        mean = sqrt_recip_alpha_t * (x_t - coef * eps_theta)
        if add_noise and (t[0] > 0):  # Only add noise when t>0 (or remove completely)
            noise = torch.randn_like(x_t)
            sigma_t = torch.sqrt(beta_t)
            x_prev = mean + sigma_t * noise
        else:
            x_prev = mean

        return x_prev

    def p_sample_loop(self, model: DiffusionPolicy, obs, n_samples=1):
        """
        Sample action conditioned on observation obs.
        Single-step action diffusion (H=1).
        GPU-accelerated batch sampling.
        """
        model.eval()
        with torch.no_grad():
            # obs: (obs_dim,) -> (B, obs_dim)
            if isinstance(obs, np.ndarray):
                obs = torch.from_numpy(obs).float()
            
            if obs.ndim == 1:
                obs_batch = obs[None, :].to(device)
            else:
                obs_batch = obs.to(device)

            B = obs_batch.size(0)
            x_t = torch.randn(B, self.cfg.action_dim, device=device)

            # Batch denoising on GPU
            for t_inv in range(self.cfg.timesteps - 1, -1, -1):
                t = torch.full((B,), t_inv, device=device, dtype=torch.long)
                x_t = self.p_sample(model, x_t, t, obs_batch, add_noise=False)

            return x_t.cpu().numpy()  # (B, action_dim)
    
    @torch.no_grad()
    def p_sample_loop_fast(self, model: DiffusionPolicy, obs_batch, ddim_steps=50):
        """
        Fast DDIM sampling for inference (fewer steps).
        Useful for real-time control.
        """
        model.eval()
        
        if isinstance(obs_batch, np.ndarray):
            obs_batch = torch.from_numpy(obs_batch).float().to(device)
        else:
            obs_batch = obs_batch.to(device)
            
        B = obs_batch.size(0)
        x_t = torch.randn(B, self.cfg.action_dim, device=device)
        
        # DDIM sampling with fewer steps
        timesteps = torch.linspace(self.cfg.timesteps - 1, 0, ddim_steps).long()
        
        for i, t_curr in enumerate(timesteps):
            t = torch.full((B,), t_curr, device=device, dtype=torch.long)
            
            # Predict noise
            eps_theta = model(x_t, t, obs_batch)
            
            # DDIM update (deterministic)
            alpha_t = self.alphas_cumprod[t].unsqueeze(-1)
            alpha_t_prev = self.alphas_cumprod[timesteps[i+1]].unsqueeze(-1) if i < len(timesteps) - 1 else torch.ones_like(alpha_t)
            
            pred_x0 = (x_t - torch.sqrt(1 - alpha_t) * eps_theta) / torch.sqrt(alpha_t)
            x_t = torch.sqrt(alpha_t_prev) * pred_x0 + torch.sqrt(1 - alpha_t_prev) * eps_theta
            
        return x_t.cpu().numpy()


# ===========================
# 4. Diffusion Policy Training Loop (GPU Accelerated)
# ===========================
def train_diffusion_policy(
    batch_size: int = 256,
    epochs: int = 20,
    lr: float = 1e-4,
    resume: bool = True,
    save_freq: int = 10,  # Save checkpoint every N epochs (set to 1 to save every epoch)
    early_stop_patience: int = 100,  # Stop if no improvement for N epochs (0 to disable)
):
    print(f"🚀 Training Diffusion Policy with GPU acceleration")
    if USE_AMP:
        print(f"   Using Automatic Mixed Precision (AMP) for faster training")
    
    dataset = FetchReachExpertDataset(DATASET_PATH)
    
    # GPU-optimized DataLoader settings
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        drop_last=True,
        num_workers=NUM_WORKERS,  # Parallel data loading
        pin_memory=True if torch.cuda.is_available() else False,  # Faster GPU transfer
        persistent_workers=True if NUM_WORKERS > 0 else False,
    )

    # ---- Load model/cfg or create new ----
    if resume and os.path.exists(DIFFUSION_MODEL_PATH):
        print(f"🔄 Loading existing diffusion policy and continuing training: {DIFFUSION_MODEL_PATH}")
        checkpoint = torch.load(DIFFUSION_MODEL_PATH, map_location=device)
        cfg_dict = checkpoint["cfg"]
        cfg = DiffusionConfig(**cfg_dict)
        model = DiffusionPolicy(cfg).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        diffusion = ActionDiffusion(cfg)
        start_epoch = checkpoint.get("epoch", 0) + 1
    else:
        print("🆕 Training new diffusion policy from scratch")
        obs_dim = dataset.obs.shape[1]
        act_dim = dataset.actions.shape[1]

        cfg = DiffusionConfig(
            action_dim=act_dim,
            obs_dim=obs_dim,
            timesteps=100,
            beta_start=1e-4,
            beta_end=0.02,
            hidden_dim=256,
        )
        diffusion = ActionDiffusion(cfg)
        model = DiffusionPolicy(cfg).to(device)
        start_epoch = 1

    # Use AdamW with weight decay for better generalization
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    
    # Learning rate scheduler for better convergence
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr * 0.01
    )
    
    mse = nn.MSELoss()
    
    # Mixed precision training scaler
    scaler = GradScaler() if USE_AMP else None

    model.train()
    
    # Early stopping tracking
    best_loss = float('inf')
    patience_counter = 0
    
    # Outer progress bar for epochs
    epoch_pbar = tqdm(range(start_epoch, start_epoch + epochs), desc="Training Diffusion Policy")
    
    for epoch in epoch_pbar:
        total_loss = 0.0
        total_samples = 0

        # Inner loop without progress bar (or minimal output)
        for obs_batch, act_batch in dataloader:
            obs_batch = obs_batch.to(device, non_blocking=True)
            act_batch = act_batch.to(device, non_blocking=True)

            B = obs_batch.size(0)
            x0 = act_batch

            t = torch.randint(0, cfg.timesteps, (B,), device=device).long()
            noise = torch.randn_like(x0)

            optimizer.zero_grad(set_to_none=True)  # Faster than zero_grad()
            
            # Mixed precision training
            if USE_AMP:
                with autocast():
                    x_t = diffusion.q_sample(x0, t, noise=noise)
                    eps_hat = model(x_t, t, obs_batch)
                    loss = mse(eps_hat, noise)
                
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                x_t = diffusion.q_sample(x0, t, noise=noise)
                eps_hat = model(x_t, t, obs_batch)
                loss = mse(eps_hat, noise)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += loss.item() * B
            total_samples += B

        scheduler.step()
        avg_loss = total_loss / total_samples
        
        # Early stopping check
        if early_stop_patience > 0:
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= early_stop_patience:
                epoch_pbar.write(f"⏹️  Early stopping at epoch {epoch} (no improvement for {early_stop_patience} epochs)")
                epoch_pbar.write(f"   Best loss: {best_loss:.4f}, Current loss: {avg_loss:.4f}")
                break
        
        # Update epoch progress bar with current metrics
        postfix_dict = {
            'loss': f'{avg_loss:.4f}',
            'lr': f'{optimizer.param_groups[0]["lr"]:.2e}'
        }
        if early_stop_patience > 0:
            postfix_dict['best'] = f'{best_loss:.4f}'
            postfix_dict['patience'] = f'{patience_counter}/{early_stop_patience}'
        
        epoch_pbar.set_postfix(postfix_dict)

        # Save checkpoint periodically (not every epoch to save time)
        if (epoch + 1) % save_freq == 0 or epoch == start_epoch + epochs - 1:
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "cfg": cfg.__dict__,
                    "epoch": epoch,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "loss": avg_loss,
                    "best_loss": best_loss,
                },
                DIFFUSION_MODEL_PATH,
            )
            epoch_pbar.write(f"💾 Checkpoint saved at epoch {epoch} (loss: {avg_loss:.4f})")

    print(f"✅ Diffusion policy saved to: {DIFFUSION_MODEL_PATH}")
    return model, diffusion, cfg


# ===========================
# 5. Diffusion Policy Evaluation
# ===========================
def load_diffusion_policy():
    """
    Load saved diffusion policy model.
    """
    checkpoint = torch.load(DIFFUSION_MODEL_PATH, map_location=device)
    cfg_dict = checkpoint["cfg"]
    cfg = DiffusionConfig(**cfg_dict)
    model = DiffusionPolicy(cfg).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    diffusion = ActionDiffusion(cfg)
    return model, diffusion, cfg

def eval_success(model, n_episodes=50):
    """
    Evaluate success rate using parallel environments for speed.
    """
    print(f"🚀 Evaluating with {NUM_ENVS} parallel environments")
    env = make_vec_env(min(NUM_ENVS, n_episodes), render_mode=None, seed=42)
    
    success = 0
    episodes_done = 0
    n_envs = env.num_envs
    
    while episodes_done < n_episodes:
        obs = env.reset()
        done = [False] * n_envs
        
        while not all(done):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, dones, infos = env.step(action)
            
            for i, (d, info) in enumerate(zip(dones, infos)):
                if d and not done[i]:
                    done[i] = True
                    if episodes_done < n_episodes:
                        success += info.get("is_success", 0.0)
                        episodes_done += 1
        
        if episodes_done >= n_episodes:
            break
    
    env.close()
    success_rate = success / n_episodes
    print(f"Success rate: {success_rate:.3f} ({success}/{n_episodes})")
    return success_rate

def evaluate_expert():
    print("🔎 Evaluating PPO Expert policy...")
    assert os.path.exists(EXPERT_MODEL_PATH + ".zip"), "Expert not trained yet!"
    env = make_vec_env(min(NUM_ENVS, 50), render_mode=None, seed=42)
    model = PPO.load(EXPERT_MODEL_PATH, env=env, device=device)
    eval_success(model, n_episodes=50)
    env.close()

def evaluate_diffusion_policy(
    model: DiffusionPolicy,
    diffusion: ActionDiffusion,
    n_episodes: int = 10,
    render: bool = False,
    save_video: bool = True,
    video_dir: str = "videos",
    use_fast_sampling: bool = True,
):
    """
    Evaluate Diffusion Policy with GPU acceleration.
    use_fast_sampling: Use DDIM for faster inference (50 steps instead of 100)
    """
    print(f"🚀 Evaluating Diffusion Policy (Fast sampling: {use_fast_sampling})")
    
    # For video recording, use single env
    env_fn = make_env(render_mode="rgb_array" if save_video else None, rank=0, seed=42)
    env = env_fn()
    
    if render:
        # gymnasium robotics render varies by backend. Just text info for now.
        print("⚠️ Render may behave differently depending on mujoco settings.")

    returns = []
    os.makedirs(video_dir, exist_ok=True)
    success = 0
    
    model.eval()
    
    for ep in range(n_episodes):
        obs, info = env.reset()
        done = False
        ep_return = 0.0
        step = 0
        frames = []

        while not done and step < 250:
            # GPU-accelerated action sampling
            obs_tensor = torch.from_numpy(obs).float()
            
            if use_fast_sampling:
                # DDIM: 50 steps (2x faster)
                action = diffusion.p_sample_loop_fast(model, obs_tensor[None, :], ddim_steps=50)[0]
            else:
                # DDPM: 100 steps (original)
                action = diffusion.p_sample_loop(model, obs_tensor, n_samples=1)[0]

            # FetchReach has action space bounds. Clipping is safer.
            action = np.clip(action, env.action_space.low, env.action_space.high)

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_return += reward
            step += 1

            if save_video:
                try:
                    frame = env.render()
                    if frame is not None:
                        frames.append(frame)
                except Exception as e:
                    if step == 1:  # Only warn once per episode
                        print(f"⚠️ Warning: Rendering disabled due to: {e}")
                    save_video = False  # Disable video saving for remaining episodes
                
        success += info.get("is_success", 0.0)

        returns.append(ep_return)
        print(f"[Diffusion Policy] Episode {ep+1}/{n_episodes} return: {ep_return:.3f}, success: {info.get('is_success', 0.0)}")

        if save_video and len(frames) > 0:
            # Save video
            if VIDEO_FORMAT == "mp4":
                video_path = os.path.join(video_dir, f"episode_{ep+1:03d}.mp4")
                imageio.mimsave(video_path, frames, fps=30, codec='libx264')
            else:
                video_path = os.path.join(video_dir, f"episode_{ep+1:03d}.gif")
                imageio.mimsave(video_path, frames, fps=15)
            
            file_size = os.path.getsize(video_path) / 1024
            print(f"  🎥 Saved video: {video_path} ({file_size:.1f} KB)")

    # print(f"✅ Avg return over {n_episodes} episodes: {np.mean(returns):.3f}")
    print(f"[Diffusion] success_rate over {n_episodes} episodes: {success / n_episodes:.3f}")
    env.close()

def record_expert_video(
    n_episodes: int = 5,
    max_steps: int = 50,
    video_dir: str = "videos_expert",
):
    """
    Save trajectories performed by PPO Expert policy as GIF like ground truth.
    """
    assert os.path.exists(EXPERT_MODEL_PATH + ".zip"), "Expert not trained yet!"

    os.makedirs(video_dir, exist_ok=True)

    # Renderable env (single environment for video recording)
    env_fn = make_env(render_mode="rgb_array", rank=0, seed=42)
    env = env_fn()
    
    # Create a dummy vec env for PPO model
    dummy_vec_env = make_vec_env(1, render_mode=None, seed=42)
    model = PPO.load(EXPERT_MODEL_PATH, env=dummy_vec_env, device=device)
    dummy_vec_env.close()

    success = 0

    for ep in range(n_episodes):
        obs, info = env.reset()
        done = False
        frames = []
        step = 0
        ep_return = 0.0

        while not done and step < max_steps:
            # Expert action (GPU-accelerated prediction)
            obs_batch = obs[None, :]  # Add batch dimension
            action, _ = model.predict(obs_batch, deterministic=True)
            action = action[0]  # Remove batch dimension
            
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_return += reward
            step += 1

            # Save render frame
            try:
                frame = env.render()
                if frame is not None:
                    frames.append(frame)
            except Exception as e:
                print(f"⚠️ Warning: Failed to render frame at step {step}: {e}")
                break

        success += info.get("is_success", 0.0)
        print(f"[Expert Video] Episode {ep+1}/{n_episodes} return: {ep_return:.3f}, "
              f"is_success: {info.get('is_success', 0.0)}")

        # Save video
        if len(frames) > 0:
            if VIDEO_FORMAT == "mp4":
                video_path = os.path.join(video_dir, f"expert_ep_{ep+1:03d}.mp4")
                imageio.mimsave(video_path, frames, fps=30, codec='libx264')
            else:
                video_path = os.path.join(video_dir, f"expert_ep_{ep+1:03d}.gif")
                imageio.mimsave(video_path, frames, fps=15)
            
            file_size = os.path.getsize(video_path) / 1024
            print(f"  🎥 Saved Expert video: {video_path} ({file_size:.1f} KB)")
        else:
            print(f"  ⚠️ No frames captured for episode {ep+1}")

    print(f"[Expert] success_rate over {n_episodes} episodes: {success / n_episodes:.3f}")
    env.close()


# ===========================
# Training Curves Plotting
# ===========================
def plot_training_curves(history=None, out_path="training_curves.png"):
    """
    Plot Q-Learning training curves (following plot_style.py style)
    - Left axis: Loss
    - Right axis: Success Rate
    """
    if history is None:
        print("⚠️ No training data to plot")
        return
    
    success_rate_data = history.get('success_rate', [])
    loss_data = history.get('loss', [])
    
    if len(success_rate_data) == 0:
        print("No success rate data to plot")
        return
    
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#E6E6E6",
        "axes.labelcolor": "#444444",
        "axes.titlecolor": "#2E2E2E",
        "xtick.color": "#555555",
        "ytick.color": "#555555",
        "grid.color": "#DDDDDD",
        "grid.linestyle": "-",
        "grid.linewidth": 0.8,
        "axes.grid": True,
        "axes.spines.top": False,
        "savefig.bbox": "tight",
        "savefig.dpi": 300,
        "font.size": 10,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
    })
    
    loss_color = "#10A37F"
    loss_fill = "#DDF5EC"
    sr_color = "#A88AE8"
    dot_edge = "white"
    
    fig, ax1 = plt.subplots(figsize=(9.0, 5.0), dpi=300)
    ax2 = ax1.twinx()
    
    sr_steps = [s / 1000 for s, _ in success_rate_data]
    sr_values = [r for _, r in success_rate_data]
    
    max_steps = max(sr_steps) if sr_steps else 50
    ax1.set_xlim(0, max_steps * 1.05)
    
    ax1.grid(True, alpha=0.28)
    ax2.grid(False)
    ax1.spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax2.spines["right"].set_visible(True)
    ax2.spines["right"].set_color("#E6E6E6")
    
    ax1.set_title("Training Curves", pad=12, weight="semibold")
    ax1.set_xlabel("Training Steps (k)")
    ax1.set_ylabel("Loss")
    ax2.set_ylabel("Success Rate")
    ax2.set_ylim(0, 1.05)
    
    ax1.tick_params(axis="both", labelsize=9, colors="#444444")
    ax2.tick_params(axis="y", labelsize=9, colors="#444444")
    
    if len(loss_data) > 0:
        loss_steps = [s / 1000 for s, _ in loss_data]
        loss_values = [l for _, l in loss_data]
        
        valid_loss = [(s, l) for s, l in zip(loss_steps, loss_values) if l > 0]
        if valid_loss:
            loss_steps, loss_values = zip(*valid_loss)
            
            if len(loss_steps) > 1:
                ax1.plot(loss_steps, loss_values, color=loss_color, linewidth=2.4, 
                        solid_capstyle="round")
                ax1.fill_between(loss_steps, loss_values, color=loss_fill, alpha=0.45)
            if len(loss_steps) > 0:
                ax1.scatter(loss_steps[-1], loss_values[-1], color=loss_color, s=64, zorder=4,
                           linewidths=1.0, edgecolors=dot_edge)
            
            ax1.set_ylim(0, max(loss_values) * 1.2 if loss_values else 1.0)
    
    if len(sr_steps) > 1:
        ax2.plot(sr_steps, sr_values, color=sr_color, linewidth=2.4,
                solid_capstyle="round", marker='o', markersize=5,
                markeredgecolor=dot_edge, markeredgewidth=1.0)
    elif len(sr_steps) == 1:
        ax2.scatter(sr_steps, sr_values, color=sr_color, s=64, zorder=4,
                   linewidths=1.0, edgecolors=dot_edge)
    
    if len(sr_steps) > 0:
        ax2.scatter(sr_steps[-1], sr_values[-1], color=sr_color, s=80, zorder=5,
                   linewidths=1.2, edgecolors=dot_edge)
        ax2.annotate(f'{sr_values[-1]:.0%}', 
                    xy=(sr_steps[-1], sr_values[-1]),
                    xytext=(8, 5), textcoords='offset points',
                    fontsize=10, fontweight='bold', color=sr_color)
    
    handles = [
        Line2D([0], [0], color=loss_color, lw=2.4, label="Loss"),
        Line2D([0], [0], color=sr_color, lw=2.4, marker='o', markersize=5, label="Success Rate"),
    ]
    ax1.legend(handles=handles, frameon=False, loc="upper right", fontsize=9)
    
    plt.tight_layout()
    plt.savefig(out_path, facecolor="white")
    plt.close()
    
    print(f"\n📈 Training curves saved to: {out_path}")


# ===========================
# Main
# ===========================
if __name__ == "__main__":
    if TRAIN_Q_LEARNING:
        _, q_learning_history = train_q_learning(
            algo=Q_LEARNING_ALGO,
            total_timesteps=30000,
            tuning=True,
            eval_freq=1000,
        )
        if q_learning_history:
            plot_training_curves(q_learning_history, out_path="training_curves.png")
    
    if EVAL_Q_LEARNING:
        evaluate_q_learning(
            algo=Q_LEARNING_ALGO,
            n_episodes=50,
            save_video=RECORD_VIDEOS,
            video_dir="videos_qlearning",
        )
    
    if TRAIN_EXPERT:
        train_expert(total_timesteps=50000, tuning=True)
        evaluate_expert()
        if RECORD_VIDEOS:
            try:
                record_expert_video(n_episodes=5, max_steps=50, video_dir="videos_expert")
            except Exception as e:
                print(f"Video recording failed: {e}")

    if COLLECT_DATA:
        collect_expert_data(n_episodes=100, max_steps_per_episode=50)

    if TRAIN_DIFFUSION:
        model, diffusion, cfg = train_diffusion_policy(
            batch_size=512 if torch.cuda.is_available() else 256,
            epochs=5000,
            lr=5e-4,
            resume=True,
            save_freq=999999,
            early_stop_patience=999999,
        )
    
    if EVAL_DIFFUSION:
        if not TRAIN_DIFFUSION:
            model, diffusion, cfg = load_diffusion_policy()
        evaluate_diffusion_policy(
            model, diffusion, 
            n_episodes=50, 
            render=False, 
            save_video=RECORD_VIDEOS,
            use_fast_sampling=True
        )