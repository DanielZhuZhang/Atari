"""Minimal Atari example: a random agent plays one episode of Breakout.

Run headless:          python random_agent.py
Run with a game window: python random_agent.py --render
"""

import sys

import ale_py
import gymnasium as gym

# Atari envs are not registered automatically; this makes "ALE/..." ids available.
gym.register_envs(ale_py)

render = "--render" in sys.argv
env = gym.make("ALE/Breakout-v5", render_mode="human" if render else None)

print("observation space:", env.observation_space)
print("action space:", env.action_space, env.unwrapped.get_action_meanings())

obs, info = env.reset(seed=0)

total_reward = 0.0
steps = 0
while True:
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    steps += 1
    if terminated or truncated:
        break

env.close()
print(f"episode finished after {steps} steps, total reward: {total_reward}")
