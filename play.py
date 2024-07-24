"""Player-controlled simulation"""

import warnings
warnings.filterwarnings("ignore")

from stable_baselines3 import SAC

import adlr_environments
from adlr_environments import LEVEL1, LEVEL2, LEVEL3
from adlr_environments.wrapper import PlayWrapper
from utils import arg_parse, create_env
from utils.printing import *
from utils.constants import Input, MAX_PLAYMODE_STEPS
from utils.config import BPS_CONFIG, AGENT_PATH


INPUTS = ["mouse", "keyboard", "controller", "agent"]
LEVELS = [None, LEVEL1, LEVEL2, LEVEL3]
ARGUMENTS = [
    (("-l", "--level"), int, 0),
    (("-i", "--input"), str, "mouse"),
    (("-u", "--uncertainty"), None, None)
]

AGENT = AGENT_PATH + "sac_bps"
NUM_GAMES = 5
CONFIG = BPS_CONFIG
CONFIG.env.update({
    "env": "World2D-Play-v0",
    "episode_length": MAX_PLAYMODE_STEPS,
})

if __name__ == "__main__":
    # parse arguments from cli
    args = arg_parse(ARGUMENTS)
    assert 0 <= args.level < len(LEVELS), f"Unrecognized level {args.level}"
    assert args.input in INPUTS, f"Unrecognized input method {args.input}"

    chosen_level = LEVELS[args.level]
    CONFIG.env.update({"world": chosen_level})

    chosen_input = INPUTS.index(args.input)
    CONFIG.wrapper.append((PlayWrapper, {"control": Input(chosen_input)}))

    CONFIG.env.update({"uncertainty": args.uncertainty})

    # create environment
    env = create_env(
        wrapper=CONFIG.wrapper,
        render=True,
        obs_type=CONFIG.observation,
        num_workers=1,
        options=CONFIG.env
    )
    observation = env.reset()
    env.render()

    # optionally load agent
    model = SAC.load(AGENT) if chosen_input == 3 else None

    action, wins, ep_rewards = [0, 0], 0, []
    for game in range(NUM_GAMES):
        ep_rewards.append(0)

        print_round_start(game)

        for _ in range(MAX_PLAYMODE_STEPS):
            if model is not None:
                action, _ = model.predict(observation, deterministic=True)

            observation, reward, done, info = env.step(action)
            ep_rewards[-1] += reward[0]
            env.render()

            print_round_info(game, info[0]["timestep"], reward[0])

            if done[0]:
                print_round_end(info[0]["win"], ep_rewards[-1])

                wins += 1 if info[0]["win"] else 0
                observation = env.reset()
                break

    print_game_end(wins, ep_rewards, NUM_GAMES)

    env.close()

print("\nGame finished!")
