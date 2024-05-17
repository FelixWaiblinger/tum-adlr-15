"""Training"""

# import os
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from gymnasium.wrappers import FlattenObservation, NormalizeObservation
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
import adlr_environments  # pylint: disable=unused-import
from environment_wrappers import NormalizedObservation

scenario_name = "ppo_static_obstacles_200k_normalized"
AGENT = "./agents/" + scenario_name


def environment_creation():
    def _init():
         # create environment
        env = gym.make(id="World2D-v0", render_mode="rgb_array")
        # flatten observations
        env = FlattenObservation(env)
        # normalize observations
        env = NormalizedObservation(env)

        return env
    return _init



def continue_training(training_steps, old_name, new_name):
    env = environment_creation()
    vec_env = make_vec_env(env, n_envs=8, seed=0, vec_env_cls=DummyVecEnv)
    model = PPO.load("./agents/" + old_name, env=vec_env, tensorboard_log="./logs/" + new_name + "/")
    model.learn(total_timesteps=training_steps, progress_bar=True)
    # wrap the environment into observation wrapper
    model.save("./agents/" + new_name)


def train_environment(training_steps):
    # create environment
    env = environment_creation()
    # vectorize environment
    vec_env = make_vec_env(env, n_envs=8, seed=0, vec_env_cls=DummyVecEnv)

    # instantiate the agent
    model = PPO("MlpPolicy", vec_env, tensorboard_log="./logs/" + scenario_name + "/")

    # train the agent
    model.learn(total_timesteps=training_steps, progress_bar=True)

    # save the agents behaviour
    model.save(AGENT)


def evaluate_environment():
    env = environment_creation()
    vec_env = make_vec_env(env, n_envs=8, seed=0, vec_env_cls=DummyVecEnv)
    # load the trained agent
    model = PPO.load(AGENT, env=vec_env)

    vec_env = model.get_env()
    obs = vec_env.reset()
    for i in range(10000):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = vec_env.step(action)
        vec_env.render("human")


train_environment(training_steps=300000)
#evaluate_environment()


# continue_training(training_steps=200000, old_name="ppo_static_obstacles_200k_adapted_loss",
#                   new_name="ppo_static_obstacles_200k_adapted_loss_continued")
