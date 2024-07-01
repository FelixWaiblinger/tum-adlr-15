"""Player-controlled simulation"""

import time

import numpy as np
import gymnasium as gym
from gymnasium.wrappers import FlattenObservation, FrameStack
from stable_baselines3 import SAC

import adlr_environments # pylint: disable=unused-import
from adlr_environments import LEVEL1, LEVEL2, LEVEL3
from adlr_environments.constants import Input, MAX_PLAYMODE_STEPS
from adlr_environments.wrapper import BPSWrapper, PlayWrapper, RewardWrapper
from adlr_environments.utils import arg_parse


AGENT = "./agents/sac_sparse"
INPUTS = ["mouse", "keyboard", "controller", "agent"]
LEVELS = [None, LEVEL1, LEVEL2, LEVEL3]
NUM_GAMES = 5
ARGUMENTS = [
    (("-l", "--level"), int, 0),
    (("-i", "--input"), str, "mouse")
]
OPTIONS = {
    "episode_length": MAX_PLAYMODE_STEPS,
    "step_length": 0.1,
    "size_agent": 0.075,
    "size_target": 0.1,
    "size_static": 0.1,
    "size_dynamic": 0.075,
    "min_speed": -0.05,
    "max_speed": 0.05,
    "uncertainty": False
}


def create_env(level: dict, control: Input):
    """Create a specific level environment"""
    OPTIONS.update({"world": level})
    e = gym.make('World2D-Play-v0', render_mode='human', options=OPTIONS)
    e = BPSWrapper(e, num_points=50)
    e = FlattenObservation(e)
    # e = FrameStack(e, num_stack=3)
    e = RewardWrapper(e)
    e = PlayWrapper(e, control)
    return e


def print_round_start(g: int):
    """Print info at the start of each episode"""
    print("")
    for i in range(3):
        print(f"\rRound {g+1} starts in {3 - i}...", end="")
        time.sleep(1)
    print("\n===== START =====")


def print_round_info(g: int, t: int, r: float):
    """Print info about current game repeatedly"""
    print(f"\rRound: {g+1} | Time: {t}/{MAX_PLAYMODE_STEPS} | Reward: {r:.3f}", end="")


def print_round_end(w: bool, r: float):
    """Print info at the end of each episode"""
    msg = ('You Won!' if w else 'You Lost!')
    print(f"\n{msg} | Total Reward: {r}")


def print_game_end(w: int, rs: list):
    """Print info at the end of the game"""
    print("\n=== GAME OVER ===")
    print(f"Episode Rewards: {[round(r, 3) for r in rs]}")
    print(f"Average Reward: {np.mean(rs):.3f}")
    print(f"Success Rate: {w}/{NUM_GAMES} = {float(w) / NUM_GAMES}")


if __name__ == "__main__":
    # parse arguments from cli
    args = arg_parse(ARGUMENTS)
    assert 0 <= args.level < len(LEVELS), f"Unrecognized level {args.level}"
    assert args.input in INPUTS, f"Unrecognized input method {args.input}"

    chosen_level = LEVELS[args.level]
    chosen_input = INPUTS.index(args.input)

    # create environment
    env = create_env(chosen_level, Input(chosen_input))
    observation, info = env.reset(seed=42)
    env.render()

    # optionally load agent
    model = SAC.load(AGENT, env) if chosen_input == 3 else None

    action, wins, ep_rewards = 0, 0, []
    for game in range(NUM_GAMES):
        ep_rewards.append(0)

        print_round_start(game)

        for _ in range(MAX_PLAYMODE_STEPS):
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
