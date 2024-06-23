"""Training"""

# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from stable_baselines3 import SAC
# from stable_baselines3.common.vec_env import VecVideoRecorder
from gymnasium.wrappers import FlattenObservation, FrameStack

from adlr_environments.wrapper import RewardWrapper
from adlr_environments.utils import create_env, linear


LOG_PATH = "./logs/"
AGENT_PATH = "./agents/"

AGENT_TYPE = SAC # PPO
AGENT = "sac_stacked"

WRAPPER = [
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
    "bps_size": 50,
    # NOTE: additional options I (Felix) use, but may not be necessary
    "fork": True
}


def start(num_steps: int):
    """Train a new agent from scratch"""
    env = create_env(
        wrapper=WRAPPER,
        render=False,
        num_workers=8,
        options=ENV_OPTIONS
    )

    model = AGENT_TYPE(
        "MlpPolicy",
        env,
        tensorboard_log=LOG_PATH + AGENT,
        learning_rate=linear(0.001)
    )

    model.learn(total_timesteps=num_steps, progress_bar=True)
    model.save(AGENT_PATH)


def resume(num_steps: int, new_name: str=None):
    """Resume training aka. perform additional training steps and update an
    existing agent
    """

    if not new_name:
        new_name = AGENT

    env = create_env(
        wrapper=WRAPPER,
        render=False,
        num_workers=8,
        options=ENV_OPTIONS
    )

    model = AGENT_TYPE.load(
        AGENT_PATH + AGENT,
        env,
        tensorboard_log=LOG_PATH + new_name
    )

    model.learn(total_timesteps=num_steps, progress_bar=True)
    model.save(new_name)


def evaluate(name: str, num_steps: int=1000):
    """Evaluate a trained agent"""
    options = ENV_OPTIONS
    options.pop("fork")

    env = create_env(
        wrapper=WRAPPER,
        render=True,
        num_workers=1,
        options=options
    )

    model = AGENT_TYPE.load(AGENT_PATH + name, env)

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
    # start(num_steps=1_000_000)

    # resume(num_steps=1_000_000)

    evaluate(AGENT)
