import gymnasium as gym
import panda_gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import os

# Create environment
env = make_vec_env("PandaReach-v3", n_envs=4)

model = PPO(
    "MultiInputPolicy",
    env,
    verbose=1,
    learning_rate=0.001,
    tensorboard_log="./panda_logs/"
)

print(" Starting Training... (Run for 30-60 mins)")
model.learn(total_timesteps=80000)   # You can stop early if needed

model.save("panda_reach_ppo")
print(" Training Done! Model saved as panda_reach_ppo.zip")