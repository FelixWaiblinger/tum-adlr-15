import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# from sb3_contrib import RecurrentPPO
from stable_baselines3 import PPO

import adlr_environments
from state_representation.models import CfCPolicy
from utils import create_env, linear_schedule
from utils.config import XY_POSITION_CONFIG, LOG_PATH, AGENT_PATH

# RL agent
AGENT = AGENT_PATH + "ppo_lnn"


def train(num_steps: int):
    """Train a new agent from scratch"""
    env = create_env(
        wrapper=XY_POSITION_CONFIG.wrapper,
        render=False,
        obs_type=XY_POSITION_CONFIG.observation,
        num_workers=1,
        options=XY_POSITION_CONFIG.env
    )

    model = PPO(
        CfCPolicy,
        env,
        tensorboard_log=LOG_PATH + AGENT,
        learning_rate=linear_schedule(0.001)
    )

    model.learn(total_timesteps=num_steps, progress_bar=True)
    model.save(AGENT)


def evaluate(num_steps: int=1000):
    """Evaluate a trained agent"""
    env = create_env(
        wrapper=XY_POSITION_CONFIG.wrapper,
        render=True,
        obs_type=XY_POSITION_CONFIG.observation,
        num_workers=1,
        options=XY_POSITION_CONFIG.env
    )

    model = PPO.load(AGENT)

    rewards, episodes, wins, crashes, stuck = 0, 0, 0, 0, 0
    obs = env.reset()

    for _ in range(num_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        rewards += reward
        env.render()

        if done:
            episodes += 1
            if info[0]["win"]:
                wins += 1
            elif info[0]["collision"]:
                crashes += 1
            else:
                stuck += 1

        # NOTE: uncomment to better see what's happening during evaluation 
        # time.sleep(0.2)

    env.close()

    print(f"Average reward over {episodes} episodes: {rewards / episodes}")
    print(f"Successrate: {100 * (wins / episodes):.2f}%")
    print(f"Crashrate: {100 * (crashes / episodes):.2f}%")
    print(f"Stuckrate: {100 * (stuck / episodes):.2f}%")


if __name__ == '__main__':
    print("Training policy...")
    train(10000)
    print("Training finished!")

    print("Evaluating policy...")
    evaluate()
