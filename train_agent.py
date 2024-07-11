"""Training"""

import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
from stable_baselines3 import SAC

import adlr_environments
from utils import arg_parse, create_env, linear_schedule
from utils.config import BPS_CONFIG, AE_CONFIG, LOG_PATH, AGENT_PATH


ARGUMENTS = [
    (("-s", "--start"), int, None),
    (("-r", "--resume"), int, None),
    (("-e", "--eval"), int, None),
    (("-n", "--name"), str, None),
]

# RL agent
AGENT_TYPE = SAC
AGENT = AGENT_PATH + "sac_bps50_5M"

CONFIG = BPS_CONFIG # AE_CONFIG
CONFIG.env.update({
    # "fork": True,
    # "world": adlr_environments.LEVEL3
    "uncertainty": True
})


def start(num_steps: int):
    """Train a new agent from scratch"""
    env = create_env(
        wrapper=CONFIG.wrapper,
        render=False,
        obs_type=CONFIG.observation,
        num_workers=8,
        options=CONFIG.env
    )

    model = AGENT_TYPE(
        "MlpPolicy",
        env,
        tensorboard_log=LOG_PATH + AGENT,
        learning_rate=linear_schedule(0.001)
    )

    model.learn(total_timesteps=num_steps, progress_bar=True)
    model.save(AGENT)


def resume(num_steps: int, new_name: str=None):
    """Resume training aka. perform additional training steps and update an
    existing agent
    """
    new_name = AGENT if new_name is None else new_name

    env = create_env(
        wrapper=CONFIG.wrapper,
        render=False,
        obs_type=CONFIG.observation,
        num_workers=8,
        options=CONFIG.env
    )

    model = AGENT_TYPE.load(AGENT, env, tensorboard_log=LOG_PATH + new_name)

    model.learn(total_timesteps=num_steps, progress_bar=True)
    model.save(new_name)


def evaluate(num_steps: int=1000, slow=False):
    """Evaluate a trained agent"""
    env = create_env(
        wrapper=CONFIG.wrapper,
        render=False, #True,
        obs_type=CONFIG.observation,
        num_workers=1,
        options=CONFIG.env
    )

    model = AGENT_TYPE.load(AGENT)

    # TODO: remove before submission
    # data, progress = [], 0
    # for _ in range(100):
    #     print(f"\rData generation: {progress}%", end="")

    rewards, episodes, wins, crashes, stuck = 0, 0, 0, 0, 0
    obs = env.reset()

    # TODO: remove before submission
    # NOTE: uncomment for video recording
    # env = VecVideoRecorder(
    #     env,
    #     "./videos/",
    #     lambda x: x == 0,
    #     video_length=num_steps-1,
    #     name_prefix="static_sac_1dyn"
    # )
    # obs = env.reset()

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

        if slow:
            time.sleep(0.2)

    env.close()

    print(f"Average reward over {episodes} episodes: {rewards / episodes}")
    print(f"Successrate: {100 * (wins / episodes):.2f}%")
    print(f"Crashrate: {100 * (crashes / episodes):.2f}%")
    print(f"Stuckrate: {100 * (stuck / episodes):.2f}%")

    # TODO: remove before submission
    #     success_rate = wins / episodes
    #     crash_rate = crashes / episodes
    #     mean_reward = rewards / episodes

    #     data.append([success_rate, crash_rate, mean_reward])
    #     progress += 1

    # data = np.array(data)
    # np.savez_compressed("plots/results_bps_50_u.npz", data)

    # print(f"\rData generation: {progress}%")


if __name__ == '__main__':
    # parse arguments from cli
    args = arg_parse(ARGUMENTS)
    assert any([args.start, args.resume, args.eval])

    # record reset dataset
    if args.start is not None:
        assert args.start > 0, \
            f"Number of training steps ({args.start}) must be positive!"
        start(args.start)

    # train an autoencoder on the recorded data
    elif args.resume is not None:
        assert args.resume > 0, \
            f"Number of training steps ({args.resume}) must be positive!"
        resume(args.resume, args.name)

    # evaluate the performance of the encoder by visual inspection
    elif args.eval is not None:
        evaluate(args.eval)
