"""Player-controlled simulation"""

import time
from argparse import ArgumentParser

import numpy as np
import gymnasium as gym
from gymnasium.wrappers import FlattenObservation
from stable_baselines3 import SAC

import adlr_environments # pylint: disable=unused-import
from adlr_environments import LEVEL1, LEVEL2, LEVEL3
from adlr_environments.constants import Input
from adlr_environments.wrapper import PlayWrapper, RewardWrapper

AGENT = "./agents/sac_sparse"
INPUTS = ["mouse", "keyboard", "joystick", "agent"]
LEVELS = [None, LEVEL1, LEVEL2, LEVEL3]
NUM_GAMES = 5
OPTIONS = {
    "step_length": 0.05,
    "size_agent": 0.075,
    "size_target": 0.1,
    "size_static": 0.1,
    "size_dynamic": 0.075,
    "min_speed": -0.05,
    "max_speed": 0.05,
    "bps_size": 50,
}


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument("-l", "--level", type=int, default=0)
    parser.add_argument("-i", "--input", default="mouse")
    args = parser.parse_args()

    assert 0 <= args.level < len(LEVELS), f"Unrecognized level {args.level}"
    assert args.input in INPUTS, f"Unrecognized input method {args.input}"

    return LEVELS[args.level], INPUTS.index(args.input)


def create_env(level: dict, control: Input):
    """Create a specific level environment"""
    OPTIONS.update({"world": level})
    e = gym.make('World2D-Play-v0', render_mode='human', options=OPTIONS)
    e = FlattenObservation(e)
    e = RewardWrapper(e)
    e = PlayWrapper(e, control)
    return e


def print_round_start():
    print("")
    for i in range(3):
        print(f"\rRound {game+1} starts in {3 - i}...", end="")
        time.sleep(1)
    print("\n===== START =====")


def print_round_info(game: int, timestep: int, reward: float):
    print(f"\rRound: {game+1} | Time: {timestep}/500 | Reward: {reward:.3f}", end="")


def print_round_end(win: bool, reward: float):
    msg = ('You Won!' if win else 'You Lost!')
    print(f"\n{msg} | Total Reward: {reward}")


def print_game_end(wins: int, total_rewards: list):
    print("\n=== GAME OVER ===")
    print(f"Episode Rewards: {[round(r, 3) for r in total_rewards]}")
    print(f"Average Reward: {np.mean(total_rewards):.3f}")
    print(f"Success Rate: {wins}/{NUM_GAMES} = {float(wins) / NUM_GAMES}")


if __name__ == "__main__":
    # parse arguments from cli
    chosen_level, chosen_input = parse_arguments()

    # create environment
    env = create_env(chosen_level, Input(chosen_input))
    observation, info = env.reset(seed=42)
    env.render()

    # optionally load agent
    model = SAC.load(AGENT, env) if chosen_input == 3 else None

    action, wins, ep_rewards = 0, 0, []
    for game in range(NUM_GAMES):
        ep_rewards.append(0)

        print_round_start()

        for _ in range(500):
            if model is not None:
                action, _ = model.predict(observation, deterministic=True)

            observation, reward, terminated, truncated, info = env.step(action)
            ep_rewards[-1] += reward
            env.render()

            print_round_info(game, info["timestep"], reward)

            if terminated or truncated:
                print_round_end(info["win"], ep_rewards[-1])

                wins += 1 if info["win"] else 0
                observation, info = env.reset()
                break

    print_game_end(wins, ep_rewards)

    env.close()

print("\nGame finished!")
