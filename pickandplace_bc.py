"""
FetchPickAndPlace - Behavior Cloning
"""

import os
import numpy as np

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
import gymnasium_robotics
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import imageio
import matplotlib.pyplot as plt
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Device: {device}")

ENV_ID = "FetchPickAndPlace-v4"
MODEL_PATH = "bc_pickandplace.pt"
DATA_PATH = "expert_data.npz"


def make_env(render_mode=None):
    env = gym.make(ENV_ID, reward_type="sparse", render_mode=render_mode)
    if render_mode in ("rgb_array", "human"):
        env.reset()
        env.render()
    return env


class ScriptedExpert:
    
    def __init__(self):
        pass
    
    def reset(self):
        pass
    
    def get_action(self, obs):
        grip_pos = obs["observation"][:3]
        obj_pos = obs["achieved_goal"]
        goal_pos = obs["desired_goal"]
        fingers = obs["observation"][9:11]
        
        to_obj = obj_pos - grip_pos
        to_goal = goal_pos - grip_pos
        obj_to_goal = goal_pos - obj_pos
        
        grip_to_obj_xy = np.linalg.norm(to_obj[:2])
        grip_to_obj = np.linalg.norm(to_obj)
        grip_to_goal_xy = np.linalg.norm(to_goal[:2])
        obj_to_goal_dist = np.linalg.norm(obj_to_goal)
        gripper_closed = fingers[0] < 0.04
        obj_grasped = gripper_closed and grip_to_obj < 0.05
        
        action = np.zeros(4)
        GAIN = 15
        
        if not obj_grasped:
            if grip_to_obj_xy > 0.02:
                target = obj_pos.copy()
                target[2] = obj_pos[2] + 0.08
                action[:3] = np.clip((target - grip_pos) * GAIN, -1, 1)
                action[3] = 1.0
            elif grip_pos[2] > obj_pos[2] + 0.02:
                action[:3] = np.clip(to_obj * GAIN, -1, 1)
                action[3] = 1.0
            else:
                action[:3] = np.clip(to_obj * 5, -1, 1)
                action[3] = -1.0
        else:
            target_height = max(goal_pos[2] + 0.05, 0.5)
            if grip_pos[2] < target_height - 0.02:
                action[2] = 1.0
                action[3] = -1.0
            elif grip_to_goal_xy > 0.02:
                target = goal_pos.copy()
                target[2] = grip_pos[2]
                action[:3] = np.clip((target - grip_pos) * GAIN, -1, 1)
                action[3] = -1.0
            elif grip_pos[2] > goal_pos[2] + 0.03:
                target = goal_pos.copy()
                target[2] += 0.02
                action[:3] = np.clip((target - grip_pos) * GAIN, -1, 1)
                action[3] = -1.0
            else:
                action[3] = 1.0
                if obj_to_goal_dist < 0.05:
                    action[:3] = 0
        
        return action


def collect_expert_data(n_episodes=500, max_steps=50):
    
    env = make_env()
    expert = ScriptedExpert()
    
    observations = []
    actions = []
    successes = 0
    
    for ep in tqdm(range(n_episodes), desc="Collecting"):
        obs, _ = env.reset()
        expert.reset()
        
        for step in range(max_steps):
            action = expert.get_action(obs)
            obs_flat = np.concatenate([
                obs["observation"],
                obs["desired_goal"],
                obs["achieved_goal"]
            ])
            observations.append(obs_flat)
            actions.append(action)
            obs, _, term, trunc, info = env.step(action)
            
            if term or trunc:
                break
        
        successes += info.get("is_success", 0)
    
    env.close()
    
    observations = np.array(observations, dtype=np.float32)
    actions = np.array(actions, dtype=np.float32)
    np.savez(DATA_PATH, observations=observations, actions=actions)
    
    success_rate = successes / n_episodes
    print(f"Expert success rate: {success_rate:.1%}")
    return observations, actions, success_rate


class BCPolicy(nn.Module):
    
    def __init__(self, obs_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),
        )
    
    def forward(self, obs):
        return self.net(obs)


class ExpertDataset(Dataset):
    
    def __init__(self, data_path):
        data = np.load(data_path)
        self.observations = torch.FloatTensor(data["observations"])
        self.actions = torch.FloatTensor(data["actions"])
    
    def __len__(self):
        return len(self.observations)
    
    def __getitem__(self, idx):
        return self.observations[idx], self.actions[idx]


def train_bc(epochs=100, batch_size=256, lr=3e-4):
    dataset = ExpertDataset(DATA_PATH)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    obs_dim = dataset.observations.shape[1]
    action_dim = dataset.actions.shape[1]
    
    model = BCPolicy(obs_dim, action_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    losses = []
    best_loss = float('inf')
    
    for epoch in range(epochs):
        total_loss = 0
        for obs_batch, action_batch in dataloader:
            obs_batch = obs_batch.to(device)
            action_batch = action_batch.to(device)
            
            pred_actions = model(obs_batch)
            loss = criterion(pred_actions, action_batch)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        losses.append(avg_loss)
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), MODEL_PATH)
        
        if (epoch + 1) % 10 == 0:
            print(f"   Epoch {epoch+1}/{epochs}: loss={avg_loss:.6f}")
    
    return model, losses


def evaluate(n_episodes=50):
    env = make_env()
    obs, _ = env.reset()
    obs_flat = np.concatenate([obs["observation"], obs["desired_goal"], obs["achieved_goal"]])
    obs_dim = len(obs_flat)
    action_dim = env.action_space.shape[0]
    
    model = BCPolicy(obs_dim, action_dim).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    
    successes = 0
    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        step = 0
        
        while not done and step < 100:
            obs_flat = np.concatenate([
                obs["observation"],
                obs["desired_goal"],
                obs["achieved_goal"]
            ])
            obs_tensor = torch.FloatTensor(obs_flat).unsqueeze(0).to(device)
            with torch.no_grad():
                action = model(obs_tensor).cpu().numpy()[0]
            
            obs, _, term, trunc, info = env.step(action)
            done = term or trunc
            step += 1
        
        successes += info.get("is_success", 0)
        
        if (ep + 1) % 10 == 0:
            print(f"   Ep {ep+1}/{n_episodes}: {successes/(ep+1):.1%}")
    
    env.close()
    
    success_rate = successes / n_episodes
    print(f"BC Success Rate: {success_rate:.1%}")
    
    if RENDERING_AVAILABLE:
        record_video()
    return success_rate


def record_video(n_episodes=5, video_dir="videos_bc"):
    os.makedirs(video_dir, exist_ok=True)
    env = make_env(render_mode="rgb_array")
    obs, _ = env.reset()
    obs_flat = np.concatenate([obs["observation"], obs["desired_goal"], obs["achieved_goal"]])
    obs_dim = len(obs_flat)
    action_dim = env.action_space.shape[0]
    
    model = BCPolicy(obs_dim, action_dim).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    
    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        frames = []
        
        while not done:
            obs_flat = np.concatenate([
                obs["observation"],
                obs["desired_goal"],
                obs["achieved_goal"]
            ])
            obs_tensor = torch.FloatTensor(obs_flat).unsqueeze(0).to(device)
            
            with torch.no_grad():
                action = model(obs_tensor).cpu().numpy()[0]
            
            obs, _, term, trunc, info = env.step(action)
            done = term or trunc
            
            frame = env.render()
            if frame is not None:
                frames.append(frame)
        
        if frames:
            tag = "success" if info.get("is_success", 0) else "fail"
            path = os.path.join(video_dir, f"ep{ep+1}_{tag}.mp4")
            imageio.mimsave(path, frames, fps=30, codec='libx264')
            print(f"   {path}")
    
    env.close()


if __name__ == "__main__":
    if os.path.exists(DATA_PATH):
        os.remove(DATA_PATH)
    if os.path.exists(MODEL_PATH):
        os.remove(MODEL_PATH)
    
    collect_expert_data(n_episodes=500, max_steps=50)
    train_bc(epochs=200, batch_size=256, lr=1e-4)
    evaluate(n_episodes=50)
