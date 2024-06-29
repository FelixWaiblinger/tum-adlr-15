"""Training"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from stable_baselines3 import SAC
# from stable_baselines3.common.vec_env import VecVideoRecorder
from gymnasium.wrappers import FlattenObservation, FrameStack

from adlr_environments.constants import Observation
from adlr_environments.wrapper import RewardWrapper, AEWrapper, BPSWrapper
from adlr_environments.utils import arg_parse, create_env, linear
from state_representation import CombineTransform, NormalizeTransform


ARGUMENTS = [
    (("-s", "--start"), int, None),
    (("-r", "--resume"), int, None),
    (("-e", "--eval"), int, None),
    (("-n", "--name"), str, None),
]

# learned state representation
AUTOENCODER = "./state_representation/autoencoder.pt"
TRANSFORM = CombineTransform([
    NormalizeTransform(start=(0, 255), end=(0, 1)),
])

# RL agent
AGENT = SAC # PPO
AGENT_NAME = "sac_ae_100"
AGENT_PATH = "./agents/"
LOG_PATH = "./logs/"

WRAPPER = [
    # (BPSWrapper, {"num_points": 100}),
    (AEWrapper, {"model_path": AUTOENCODER, "transform": TRANSFORM}),
    (FlattenObservation, {}),
    (FrameStack, {"num_stack": 3}),
    (RewardWrapper, {})
]
ENV_OPTIONS = {
    "step_length": 0.1,
    "size_agent": 0.075,
    "size_target": 0.1,
    "num_static_obstacles": 0,
    "size_static": 0.1,
    "num_dynamic_obstacles": 8,
    "size_dynamic": 0.075,
    "min_speed": -0.05,
    "max_speed": 0.05,
}
# NOTE: additional options I (Felix) use, but may not be necessary
EXTRA_OPTIONS = {
    "fork": True
}


def start(num_steps: int):
    """Train a new agent from scratch"""
    ENV_OPTIONS.update(EXTRA_OPTIONS)

    env = create_env(
        wrapper=WRAPPER,
        render=False,
        obs_type=Observation.RGB,
        num_workers=8,
        options=ENV_OPTIONS
    )

    model = AGENT(
        "MlpPolicy",
        env,
        tensorboard_log=LOG_PATH + AGENT_NAME,
        learning_rate=linear(0.001)
    )

    model.learn(total_timesteps=num_steps, progress_bar=True)
    model.save(AGENT_PATH + AGENT_NAME)


def resume(num_steps: int, new_name: str=None):
    """Resume training aka. perform additional training steps and update an
    existing agent
    """
    ENV_OPTIONS.update(EXTRA_OPTIONS)
    new_name = AGENT_NAME if new_name is None else new_name

    env = create_env(
        wrapper=WRAPPER,
        render=False,
        obs_type=Observation.RGB,
        num_workers=8,
        options=ENV_OPTIONS
    )

    model = AGENT.load(
        AGENT_PATH + AGENT_NAME,
        env,
        tensorboard_log=LOG_PATH + new_name
    )

    model.learn(total_timesteps=num_steps, progress_bar=True)
    model.save(new_name)


def evaluate(num_steps: int=1000):
    """Evaluate a trained agent"""
    env = create_env(
        wrapper=WRAPPER,
        render=True,
        obs_type=Observation.RGB,
        num_workers=1,
        options=ENV_OPTIONS
    )

    model = AGENT.load(AGENT_PATH + AGENT_NAME, env)

    rewards, episodes, wins, crashes, stuck = 0, 0, 0, 0, 0
    obs = env.reset()

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

        # time.sleep(0.2)
    env.close()

    print(f"Average reward over {episodes} episodes: {rewards / episodes}")
    print(f"Successrate: {100 * (wins / episodes):.2f}%")
    print(f"Crashrate: {100 * (crashes / episodes):.2f}%")
    print(f"Stuckrate: {100 * (stuck / episodes):.2f}%")


if __name__ == '__main__':
    # parse arguments from cli
    args = arg_parse(ARGUMENTS)
    assert not all(arg is None for arg in [args.start, args.resume, args.eval])

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
