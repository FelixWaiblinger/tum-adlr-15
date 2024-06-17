"""Constant values"""

MAX_EPISODE_STEPS = 200

WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
BLACK = (0, 0, 0)

# Maximum number of steps in the environment --> register to change
MAX_EPISODE_STEPS = 400

# environment and trainings options
OPTIONS = {
    "r_target": 10,
    "r_collision": -1,
    "r_time": -0.001,
    "r_distance": 0,
    "r_wall_collision": -0.1,
    "world_size": 6,
    "step_length": 0.4,
    "num_static_obstacles": 4,
    "render": False,
    "random_obstacles": True,
    "random_agent": True,
    "random_target": True
}

# path's
AGENT = "n2"
AGENT_PATH = "./agents/" + AGENT
LOG_PATH = "./logs/" + AGENT
VIDEO_PATH = "./videos/"
RESULT_PATH = "./agents/random_search_results.json"
MODEL_PATH = "./agents/random_search_model"
