"""
FetchReach-v2 + Diffusion Policy (Continuous Action)
WSL에서도 돌아갈 수 있게 최대한 깔끔하게 작성한 예제.

필요 패키지 (예시):
    pip install gymnasium gymnasium-robotics mujoco stable-baselines3 torch numpy tqdm

주의:
- FetchReach-v2 환경이 안 뜨면, 설치한 gymnasium-robotics 버전에 맞는 env ID를 확인해야 함.
- Mujoco 라이센스/설치가 필요할 수 있음.
"""

import os
import numpy as np
from dataclasses import dataclass

import gymnasium as gym
from gymnasium.wrappers import FlattenObservation

from stable_baselines3 import PPO

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ===========================
# 0. 설정
# ===========================
ENV_ID = "FetchReach-v2"  # 설치 버전에 따라 v1/v3일 수도 있음
EXPERT_MODEL_PATH = "ppo_fetchreach_expert"
DATASET_PATH = "fetchreach_expert_dataset.npz"
DIFFUSION_MODEL_PATH = "diffusion_fetchreach_policy.pt"

# 어떤 단계까지 실행할지 플래그
TRAIN_EXPERT = True
COLLECT_DATA = True
TRAIN_DIFFUSION = True
EVAL_DIFFUSION = True

# In MuJoCo 세계: 로봇팔을 MJCF 파일(XML)로 “설계하고” 그걸 엔진이 읽어서 가상의 로봇팔을 만들어줌

def make_env(): # 로봇 팔을 책상 위에 가져다 놓고 전원을 넣는 것
    """
    FetchReach-v2 환경을 만들고, observation dict를 1D 벡터로 flatten.
    """
    env = gym.make(ENV_ID) # 로봇 시뮬레이터, MJCF 파일 (MuJoCo의 로봇/환경 정의 파일) 불러오고, 물리 엔진 초기화하고, reward function 세팅하고, action space / observation space 세팅하고
    
    # MJCF includes 로봇 base 위치, 관절(joint) 종류: revolute, prismatic 등, 각 링크(link) 크기/질량/관성, 손(gripper)의 finger geometry, collision shape, actuator (torque/force control) 설정, 카메라 위치, 목표(goal) object 위치
    
    # FetchReach-v2의 기본 observation dict: {
    # "observation": [robot_state vector...],      # 연속 벡터 (ex. length 10): joint, velocity, gripper, end-effector 위치 등
    # "desired_goal": [goal_x, goal_y, goal_z],     # 목표 좌표 벡터 (ex. length 3)
    # "achieved_goal": [current_x, current_y, current_z]  # 현재 end-effector 좌표 (ex. length 3)
    # } => FlattenObservation => obs_flat = [...16 numbers]
    env = FlattenObservation(env)  # dict(obs, desired_goal, achieved_goal) -> flat vector
    return env


# ===========================
# 1. Expert (PPO) 학습 + 데이터 수집
# ===========================
def train_expert(total_timesteps: int = 300_000):
    """
    PPO로 FetchReach expert 정책 학습.
    """
    env = make_env()
    # 관측(obs) 입력 → 행동(action) 출력하는 MLP
    # 이 로봇 환경을 보고 행동을 결정하는 뇌(MlpPolicy)를 PPO 알고리즘으로 학습시키겠다. 
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=total_timesteps) # Generate one option trajectory for goal
    model.save(EXPERT_MODEL_PATH)
    print(f"✅ Expert model saved to: {EXPERT_MODEL_PATH}.zip")


def collect_expert_data(
    n_episodes: int = 200,
    max_steps_per_episode: int = 50,
):
    """
    학습된 expert 모델로 FetchReach를 여러 episode 플레이하면서
    (obs, action) 쌍들을 모아 .npz로 저장.

    나중에 Behavior Cloning, Diffusion Policy 학습에 사용할 데이터셋.
    Both are Imitation Learning, but BC is simpler.
    IL = Strong Supervision
    RL = Weak Supervision (또는 Sparse Supervision)
    """
    assert os.path.exists(EXPERT_MODEL_PATH + ".zip"), \
        "Expert model not found. 먼저 train_expert()를 실행하세요."

    env = make_env()
    model = PPO.load(EXPERT_MODEL_PATH, env=env) # Example

    observations = []
    actions = []

    for ep in range(n_episodes):
        obs, info = env.reset()
        for t in range(max_steps_per_episode):
            # expert 행동 선택
            action, _ = model.predict(obs, deterministic=True)

            observations.append(obs)
            actions.append(action)

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            if done:
                break
        print(f"[Expert Rollout] Episode {ep+1}/{n_episodes} finished at step {t+1}")

    observations = np.array(observations, dtype=np.float32)
    actions = np.array(actions, dtype=np.float32)

    np.savez(DATASET_PATH, observations=observations, actions=actions)
    print(f"✅ Dataset saved to: {DATASET_PATH}")
    print(f"   observations: {observations.shape}, actions: {actions.shape}")


# ===========================
# 2. Diffusion Policy용 Dataset
# ===========================
class FetchReachExpertDataset(Dataset):
    """
    (obs, action) supervised 학습용 데이터셋.
    여기서는 single-step action diffusion (H=1)으로 두고,
    나중에 H>1 trajectory로 확장 가능.
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
    action_dim: int # 디퓨전 모델이 출력해야 하는 continuous vector 크기
    obs_dim: int # diffusion model이 conditioning할 입력 벡터 크기
    timesteps: int = 100       # diffusion steps T
    beta_start: float = 1e-4 # 베타(beta) 라는 노이즈 크기를 step마다 조금씩 증가시키며 데이터를 노이즈화
    beta_end: float = 0.02
    hidden_dim: int = 256 # MLP hidden layer size


class TimeEmbedding(nn.Module): # 현재 timestep t 를 알려줘야 한다. t'th denoising step. t를 고차원 벡터로 변환, like positional embedding.
    # unique representation for each t is possible.
    """
    간단한 sinusoidal time embedding + MLP.
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
    epsilon_theta(x_t, t, obs)를 예측하는 네트워크.
    입력: noisy action (x_t), time embedding, obs
    출력: noise epsilon_hat (action_dim)
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
        return eps_hat # 원래 action에 더해진 noise (eps_hat) 를 예측


class ActionDiffusion:
    """
    DDPM-style forward/inverse process를 관리하는 helper 클래스.
    """
    def __init__(self, cfg: DiffusionConfig):
        self.cfg = cfg

        # beta schedule (linear)
        betas = torch.linspace(cfg.beta_start, cfg.beta_end, cfg.timesteps) # noise
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0) # t step까지 누적해서 signal (original data) 이 얼마나 남았는지

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

    def p_sample(self, model: DiffusionPolicy, x_t, t, obs):
        """
        하나의 reverse step: p_theta(x_{t-1} | x_t)
        """
        beta_t = self.betas[t].unsqueeze(-1)          # (B, 1)
        sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alphas_cumprod[t].unsqueeze(-1)
        sqrt_recip_alpha_t = (1.0 / torch.sqrt(self.alphas[t])).unsqueeze(-1)

        # model predicts epsilon
        eps_theta = model(x_t, t, obs)

        # DDPM 공식: x_{t-1} = 1/sqrt(alpha_t) * (x_t - (1-alpha_t)/sqrt(1 - alpha_bar_t) * eps_theta) + sigma_t z
        coef = beta_t / sqrt_one_minus_alpha_bar_t
        mean = sqrt_recip_alpha_t * (x_t - coef * eps_theta)

        # t > 0 인 경우에만 noise 추가
        noise = torch.randn_like(x_t)
        # "posterior variance" 단순화해서 beta_t 사용
        sigma_t = torch.sqrt(beta_t)
        x_prev = mean + sigma_t * noise
        return x_prev

    def p_sample_loop(self, model: DiffusionPolicy, obs, n_samples=1):
        """
        관측 obs 조건에서 action 샘플링.
        single-step action diffusion (H=1).
        """
        model.eval()
        with torch.no_grad():
            # obs: (obs_dim,) -> (1, obs_dim)
            if obs.ndim == 1:
                obs_batch = obs[None, :]
            else:
                obs_batch = obs
            obs_batch = torch.tensor(obs_batch, dtype=torch.float32, device=device)

            B = obs_batch.size(0)
            x_t = torch.randn(B, self.cfg.action_dim, device=device)

            for t_inv in range(self.cfg.timesteps - 1, -1, -1):
                t = torch.full((B,), t_inv, device=device, dtype=torch.long)
                x_t = self.p_sample(model, x_t, t, obs_batch)

            return x_t.cpu().numpy()  # (B, action_dim)


# ===========================
# 4. Diffusion Policy 학습 루프
# ===========================
def train_diffusion_policy(
    batch_size: int = 256,
    epochs: int = 20,
    lr: float = 1e-4,
):
    dataset = FetchReachExpertDataset(DATASET_PATH)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

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

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mse = nn.MSELoss()

    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        total_samples = 0

        pbar = tqdm(dataloader, desc=f"[Diffusion Train] Epoch {epoch}/{epochs}")
        for obs_batch, act_batch in pbar:
            obs_batch = obs_batch.to(device)
            act_batch = act_batch.to(device)

            B = obs_batch.size(0)
            # x0 = expert action
            x0 = act_batch

            # 랜덤 timestep 샘플링
            t = torch.randint(0, cfg.timesteps, (B,), device=device).long()
            noise = torch.randn_like(x0)

            # forward q(x_t | x0)
            x_t = diffusion.q_sample(x0, t, noise=noise)

            # 모델이 noise 예측
            eps_hat = model(x_t, t, obs_batch)

            loss = mse(eps_hat, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * B
            total_samples += B

            pbar.set_postfix(loss=loss.item())

        avg_loss = total_loss / total_samples
        print(f"[Epoch {epoch}] avg loss: {avg_loss:.6f}")

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "cfg": cfg.__dict__,
        },
        DIFFUSION_MODEL_PATH,
    )
    print(f"✅ Diffusion policy saved to: {DIFFUSION_MODEL_PATH}")

    return model, diffusion, cfg


# ===========================
# 5. Diffusion Policy 평가
# ===========================
def load_diffusion_policy():
    """
    저장된 diffusion policy 모델 로드.
    """
    checkpoint = torch.load(DIFFUSION_MODEL_PATH, map_location=device)
    cfg_dict = checkpoint["cfg"]
    cfg = DiffusionConfig(**cfg_dict)
    model = DiffusionPolicy(cfg).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    diffusion = ActionDiffusion(cfg)
    return model, diffusion, cfg


def evaluate_diffusion_policy(
    model: DiffusionPolicy,
    diffusion: ActionDiffusion,
    n_episodes: int = 10,
    render: bool = False,
):
    env = make_env()
    if render:
        # gymnasium robotics render는 backend에 따라 다름. 일단 text info만.
        print("⚠️ Render는 mujoco 설정에 따라 다르게 동작할 수 있습니다.")

    returns = []

    for ep in range(n_episodes):
        obs, info = env.reset()
        done = False
        ep_return = 0.0
        step = 0

        while not done and step < 50:
            # obs를 torch tensor로 바꾸고 diffusion으로 action 샘플링
            obs_tensor = torch.tensor(obs, dtype=torch.float32)
            action = diffusion.p_sample_loop(model, obs_tensor, n_samples=1)[0]

            # FetchReach는 action space bound가 존재. clip 해주는 게 안전함.
            action = np.clip(action, env.action_space.low, env.action_space.high)

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_return += reward
            step += 1

        returns.append(ep_return)
        print(f"[Diffusion Policy] Episode {ep+1}/{n_episodes} return: {ep_return:.3f}")

    print(f"✅ Avg return over {n_episodes} episodes: {np.mean(returns):.3f}")
    env.close()


# ===========================
# Main
# ===========================
if __name__ == "__main__":
    if TRAIN_EXPERT:
        train_expert(total_timesteps=300_000)

    if COLLECT_DATA:
        collect_expert_data(n_episodes=200, max_steps_per_episode=50)

    if TRAIN_DIFFUSION:
        model, diffusion, cfg = train_diffusion_policy(
            batch_size=256,
            epochs=20,
            lr=1e-4,
        )
    else:
        model, diffusion, cfg = load_diffusion_policy()

    if EVAL_DIFFUSION:
        evaluate_diffusion_policy(model, diffusion, n_episodes=10, render=False)