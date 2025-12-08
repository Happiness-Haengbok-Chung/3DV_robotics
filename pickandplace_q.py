"""
FetchPickAndPlace - BC + Q Joint Training
Loss = BC_Loss + 0.01 * Q_Bonus
"""

import os
import numpy as np
import random
from collections import deque

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
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import imageio
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Device: {device}")

ENV_ID = "FetchPickAndPlace-v4"
MODEL_PATH = "bc_q_pickandplace.pt"
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
        GAIN = 12
        
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
    print("="*60)
    print("Collecting Expert Data")
    print("="*60)
    
    env = make_env()
    expert = ScriptedExpert()
    
    observations = []
    actions = []
    transitions = []
    successes = 0
    
    for ep in tqdm(range(n_episodes), desc="Collecting"):
        obs, _ = env.reset()
        expert.reset()
        episode_obs = []
        episode_actions = []
        
        for step in range(max_steps):
            action = expert.get_action(obs)
            
            obs_flat = np.concatenate([
                obs["observation"],
                obs["desired_goal"],
                obs["achieved_goal"]
            ])
            episode_obs.append(obs_flat)
            episode_actions.append(action)
            
            next_obs, reward, term, trunc, info = env.step(action)
            next_obs_flat = np.concatenate([
                next_obs["observation"],
                next_obs["desired_goal"],
                next_obs["achieved_goal"]
            ])
            done = term or trunc
            
            transitions.append((obs_flat, action, reward, next_obs_flat, float(done)))
            
            observations.append(obs_flat)
            actions.append(action)
            
            obs = next_obs
            if done:
                break
        
        successes += info.get("is_success", 0)
    
    env.close()
    
    observations = np.array(observations, dtype=np.float32)
    actions = np.array(actions, dtype=np.float32)
    
    np.savez(DATA_PATH, 
             observations=observations, 
             actions=actions,
             transitions_obs=np.array([t[0] for t in transitions], dtype=np.float32),
             transitions_act=np.array([t[1] for t in transitions], dtype=np.float32),
             transitions_rew=np.array([t[2] for t in transitions], dtype=np.float32),
             transitions_next=np.array([t[3] for t in transitions], dtype=np.float32),
             transitions_done=np.array([t[4] for t in transitions], dtype=np.float32))
    
    success_rate = successes / n_episodes
    print(f"\n✅ Expert success rate: {success_rate:.1%}")
    print(f"   Collected {len(observations)} transitions")
    return success_rate


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


class QNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
    
    def forward(self, obs, action):
        x = torch.cat([obs, action], dim=-1)
        return self.net(x)


def train_bc_q_joint(epochs=100, batch_size=256, lr=1e-4, q_weight=0.01):
    data = np.load(DATA_PATH)
    obs_data = torch.FloatTensor(data["observations"]).to(device)
    act_data = torch.FloatTensor(data["actions"]).to(device)
    trans_obs = torch.FloatTensor(data["transitions_obs"]).to(device)
    trans_act = torch.FloatTensor(data["transitions_act"]).to(device)
    trans_rew = torch.FloatTensor(data["transitions_rew"]).unsqueeze(1).to(device)
    trans_next = torch.FloatTensor(data["transitions_next"]).to(device)
    trans_done = torch.FloatTensor(data["transitions_done"]).unsqueeze(1).to(device)
    
    n_samples = len(obs_data)
    obs_dim = obs_data.shape[1]
    action_dim = act_data.shape[1]
    policy = BCPolicy(obs_dim, action_dim).to(device)
    q_net = QNetwork(obs_dim, action_dim).to(device)
    q_target = QNetwork(obs_dim, action_dim).to(device)
    q_target.load_state_dict(q_net.state_dict())
    
    policy_optimizer = optim.Adam(policy.parameters(), lr=lr)
    q_optimizer = optim.Adam(q_net.parameters(), lr=lr)
    
    gamma = 0.99
    tau = 0.005
    n_batches = n_samples // batch_size
    best_rate = 0.0
    history = {'bc_loss': [], 'q_loss': [], 'success_rate': []}
    
    for epoch in range(epochs):
        indices = torch.randperm(n_samples)
        epoch_bc_loss = 0
        epoch_q_loss = 0
        
        for i in range(n_batches):
            idx = indices[i * batch_size: (i + 1) * batch_size]
            obs_batch = obs_data[idx]
            act_batch = act_data[idx]
            
            q_idx = torch.randint(0, len(trans_obs), (batch_size,))
            q_obs = trans_obs[q_idx]
            q_act = trans_act[q_idx]
            q_rew = trans_rew[q_idx]
            q_next = trans_next[q_idx]
            q_done = trans_done[q_idx]
            with torch.no_grad():
                next_action = policy(q_next)
                target_q = q_rew + gamma * (1 - q_done) * q_target(q_next, next_action)
            
            current_q = q_net(q_obs, q_act)
            q_loss = F.mse_loss(current_q, target_q)
            
            q_optimizer.zero_grad()
            q_loss.backward()
            q_optimizer.step()
            
            for param, target_param in zip(q_net.parameters(), q_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
            
            pred_action = policy(obs_batch)
            bc_loss = F.mse_loss(pred_action, act_batch)
            q_value = q_net(obs_batch, pred_action)
            q_bonus = -q_value.mean()
            policy_loss = bc_loss + q_weight * q_bonus
            
            policy_optimizer.zero_grad()
            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            policy_optimizer.step()
            
            epoch_bc_loss += bc_loss.item()
            epoch_q_loss += q_loss.item()
        
        avg_bc_loss = epoch_bc_loss / n_batches
        avg_q_loss = epoch_q_loss / n_batches
        history['bc_loss'].append(avg_bc_loss)
        history['q_loss'].append(avg_q_loss)
        
        if (epoch + 1) % 5 == 0:
            rate = evaluate_policy(policy)
            history['success_rate'].append((epoch + 1, rate))
            print(f"Epoch {epoch+1}/{epochs}: BC={avg_bc_loss:.6f}, Success={rate:.1%}")
            
            if rate >= best_rate:
                best_rate = rate
                torch.save({
                    'policy': policy.state_dict(),
                    'obs_dim': obs_dim,
                    'action_dim': action_dim,
                }, MODEL_PATH)
    
    return policy, history


def evaluate_policy(policy, n_episodes=30):
    env = make_env()
    policy.eval()
    
    successes = 0
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            obs_flat = np.concatenate([obs["observation"], obs["desired_goal"], obs["achieved_goal"]])
            obs_tensor = torch.FloatTensor(obs_flat).unsqueeze(0).to(device)
            with torch.no_grad():
                action = policy(obs_tensor).cpu().numpy()[0]
            obs, _, term, trunc, info = env.step(action)
            done = term or trunc
        successes += info.get("is_success", 0)
    
    env.close()
    policy.train()
    return successes / n_episodes


def final_evaluate(n_episodes=50):
    print("="*60)
    print("🔎 Final Evaluation")
    print("="*60)
    
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    obs_dim = checkpoint['obs_dim']
    action_dim = checkpoint['action_dim']
    
    policy = BCPolicy(obs_dim, action_dim).to(device)
    policy.load_state_dict(checkpoint['policy'])
    policy.eval()
    
    env = make_env()
    successes = 0
    
    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            obs_flat = np.concatenate([obs["observation"], obs["desired_goal"], obs["achieved_goal"]])
            obs_tensor = torch.FloatTensor(obs_flat).unsqueeze(0).to(device)
            with torch.no_grad():
                action = policy(obs_tensor).cpu().numpy()[0]
            obs, _, term, trunc, info = env.step(action)
            done = term or trunc
        successes += info.get("is_success", 0)
        
        if (ep + 1) % 10 == 0:
            print(f"   Ep {ep+1}/{n_episodes}: {successes/(ep+1):.1%}")
    
    env.close()
    rate = successes / n_episodes
    print(f"\n📊 Final Success Rate: {rate:.1%}")
    
    if RENDERING_AVAILABLE:
        record_video()
    
    return rate


def record_video(n_episodes=5, video_dir="videos_bc_q"):
    print("\n🎥 Recording videos...")
    os.makedirs(video_dir, exist_ok=True)
    
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    policy = BCPolicy(checkpoint['obs_dim'], checkpoint['action_dim']).to(device)
    policy.load_state_dict(checkpoint['policy'])
    policy.eval()
    
    env = gym.make(ENV_ID, reward_type="sparse", render_mode="rgb_array")
    
    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        frames = []
        
        while not done:
            obs_flat = np.concatenate([obs["observation"], obs["desired_goal"], obs["achieved_goal"]])
            obs_tensor = torch.FloatTensor(obs_flat).unsqueeze(0).to(device)
            with torch.no_grad():
                action = policy(obs_tensor).cpu().numpy()[0]
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


def plot_training_curves(history, total_epochs, out_path="training_curves.png"):
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
    
    bc_color = "#10A37F"
    bc_fill = "#DDF5EC"
    sr_color = "#A88AE8"
    dot_edge = "white"
    
    bc_loss = history['bc_loss']
    success_rate_data = history['success_rate']
    
    epochs = list(range(1, len(bc_loss) + 1))
    sr_epochs = [ep for ep, _ in success_rate_data]
    sr_values = [rate for _, rate in success_rate_data]
    
    fig, ax1 = plt.subplots(figsize=(9.0, 5.0), dpi=300)
    ax2 = ax1.twinx()
    
    ax1.set_xlim(0, total_epochs + 1)
    ax1.set_ylim(0, max(bc_loss) * 1.2 if bc_loss else 0.5)
    ax2.set_ylim(0, 1.05)
    
    ax1.grid(True, alpha=0.28)
    ax2.grid(False)
    ax1.spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax2.spines["right"].set_visible(True)
    ax2.spines["right"].set_color("#E6E6E6")
    
    ax1.set_title("Q Learning Curves", pad=12, weight="semibold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax2.set_ylabel("Success Rate")
    
    ax1.tick_params(axis="both", labelsize=9, colors="#444444")
    ax2.tick_params(axis="y", labelsize=9, colors="#444444")
    
    if len(epochs) > 1:
        ax1.plot(epochs, bc_loss, color=bc_color, linewidth=2.4, solid_capstyle="round", label="Loss")
        ax1.fill_between(epochs, bc_loss, color=bc_fill, alpha=0.45)
    if len(epochs) > 0:
        ax1.scatter(epochs[-1], bc_loss[-1], color=bc_color, s=64, zorder=4, 
                   linewidths=1.0, edgecolors=dot_edge)
    
    if len(sr_epochs) > 1:
        ax2.plot(sr_epochs, sr_values, color=sr_color, linewidth=2.4, 
                solid_capstyle="round", marker='o', markersize=5, markeredgecolor=dot_edge,
                markeredgewidth=1.0)
    elif len(sr_epochs) == 1:
        ax2.scatter(sr_epochs, sr_values, color=sr_color, s=64, zorder=4,
                   linewidths=1.0, edgecolors=dot_edge)
    if len(sr_epochs) > 0:
        ax2.scatter(sr_epochs[-1], sr_values[-1], color=sr_color, s=80, zorder=5,
                   linewidths=1.2, edgecolors=dot_edge)
        ax2.annotate(f'{sr_values[-1]:.0%}', 
                    xy=(sr_epochs[-1], sr_values[-1]),
                    xytext=(8, 5), textcoords='offset points',
                    fontsize=10, fontweight='bold', color=sr_color)
    
    handles = [
        Line2D([0], [0], color=bc_color, lw=2.4, label="Loss"),
        Line2D([0], [0], color=sr_color, lw=2.4, marker='o', markersize=5, label="Success Rate"),
    ]
    ax1.legend(handles=handles, frameon=False, loc="upper right", fontsize=9)
    
    plt.tight_layout()
    plt.savefig(out_path, facecolor="white")
    plt.close()
    
    print(f"Training curves saved to: {out_path}")


# ===========================
# Main
# ===========================
if __name__ == "__main__":
    for f in [DATA_PATH, MODEL_PATH]:
        if os.path.exists(f):
            os.remove(f)
    
    expert_rate = collect_expert_data(n_episodes=500, max_steps=50)
    EPOCHS = 200
    policy, history = train_bc_q_joint(epochs=EPOCHS, batch_size=256, lr=1e-4, q_weight=0.01)
    plot_training_curves(history, total_epochs=EPOCHS, out_path="training_curves.png")
    final_rate = final_evaluate(n_episodes=50)
    
    print(f"Expert: {expert_rate:.1%}, BC + Q: {final_rate:.1%}")
