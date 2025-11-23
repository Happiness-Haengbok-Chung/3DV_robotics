import os
import numpy as np
import gymnasium as gym # OpenAI Gym의 계승 버전, 강화학습에서 사용할 환경을 만드는 표준 라이브러리
from stable_baselines3 import PPO # 강화학습 알고리즘(PPO)을 불러오는 코드

ENV_ID = "Pendulum-v1"
EXPERT_MODEL_PATH = "ppo_pendulum_expert"
DATASET_PATH = "pendulum_expert_dataset.npz"

def train_expert(total_timesteps: int = 200_000):
    env = gym.make(ENV_ID)
    # 기본 “MlpPolicy”가 continuous action 지원됨
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=total_timesteps)
    model.save(EXPERT_MODEL_PATH)
    print(f"✅ Expert model saved to: {EXPERT_MODEL_PATH}.zip")

def collect_expert_data(n_episodes: int = 50, max_steps_per_episode: int = 200):
    assert os.path.exists(EXPERT_MODEL_PATH + ".zip"), "Expert model not found."
    env = gym.make(ENV_ID, render_mode=None)
    model = PPO.load(EXPERT_MODEL_PATH, env=env)

    observations = []
    actions = []

    for ep in range(n_episodes):
        obs, info = env.reset()
        for t in range(max_steps_per_episode):
            action, _ = model.predict(obs, deterministic=True)
            observations.append(obs)
            actions.append(action)

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            if done:
                print(f"Episode {ep+1}/{n_episodes} finished in {t+1} steps.")
                break

    observations = np.array(observations, dtype=np.float32)
    actions = np.array(actions, dtype=np.float32)

    np.savez(DATASET_PATH, observations=observations, actions=actions)
    print(f"✅ Dataset saved to: {DATASET_PATH}")
    print(f"   observations shape: {observations.shape}")
    print(f"   actions shape:      {actions.shape}")

if __name__ == "__main__":
    train_expert(total_timesteps=200_000)
    collect_expert_data(n_episodes=50)