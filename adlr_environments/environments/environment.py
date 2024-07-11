"""2D environment"""

from typing import Any, List, Tuple, OrderedDict
import copy

import pygame as pg
import numpy as np
import gymnasium as gym
from gymnasium.spaces import Dict, Box

from utils import eucl
from utils.constants import *
from .entity import Agent, Target, StaticObstacle, DynamicObstacle


DEFAULT_OPTIONS = {
    "world": None,
    "episode_length": MAX_EPISODE_STEPS,
    "step_length": 0.1,
    "size_agent": 0.1,
    "size_target": 0.1,
    "num_static_obstacles": 5,
    "size_static": 0.1,
    "num_dynamic_obstacles": 0,
    "size_dynamic": 0.1,
    "min_speed": -0.1,
    "max_speed": 0.1,
    "latent_size": 100,
    "uncertainty": False
}


class World2D(gym.Env):
    """Simple 2D environment including agent, target and a discrete action
    space
    """

    metadata = {"render_modes": [None, "human", "rgb_array"], "render_fps": 60}

    def __init__(self,
        render_mode: str=None,
        observation_type: Observation=Observation.POS,
        options: dict=None
    ) -> None:
        """Create new environment"""
        # pygame related stuff
        assert render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.window = None
        self.clock = None

        # fill missing options with default values
        self.options = DEFAULT_OPTIONS
        self.options.update((options if options else {}))

        self.n_obstacles = self.options["num_static_obstacles"] \
                           + self.options["num_dynamic_obstacles"]

        # stuff tracked by the environment
        self.win = False
        self.collision = False
        self.timestep = 0
        self.step_length = self.options["step_length"]
        self.agent = Agent(self.options["size_agent"])
        self.target = Target(self.options["size_target"])
        self.static_obstacles: List[StaticObstacle] = []
        self.dynamic_obstacles: List[DynamicObstacle] = []
        
        # observations include agent, target and obstacle positions (and speed)
        pos_space = OrderedDict(
            agent=Box(-1, 1, shape=(4,), dtype=DTYPE),
            target=Box(-1, 1, shape=(2,), dtype=DTYPE),
            state=Box(-1, 1, shape=(self.n_obstacles,2), dtype=DTYPE)
        )
        # observations include a top down RGB image as numpy array
        rgb_space = OrderedDict(
            image=Box(0, 255, shape=(PIXELS, PIXELS, 3), dtype=np.uint8)
        )
        
        self.observation_type = observation_type
        space = OrderedDict()
        if observation_type != Observation.POS:
            space.update(rgb_space)
        if observation_type != Observation.RGB:
            space.update(pos_space)
        self.observation_space = Dict(space)

        # actions include setting velocity in x and y direction independently
        self.action_space = Box(low=-1, high=1, shape=(2,), dtype=DTYPE)

    def _get_observations(self):
        # uncertainty is represented by noise
        if self.options["uncertainty"]:
            noise = np.random.normal(0, 0.1, size=(self.n_obstacles + 2, 2))
        else:
            noise = np.zeros((self.n_obstacles + 2, 2))

        obs = {}#OrderedDict()
        if self.observation_type != Observation.RGB:
            # add agent
            agent = self.agent.position
            obs["agent"] = np.concatenate([
                agent + noise[0],
                self.agent.speed
            ]).astype(DTYPE)

            # add target
            d = np.linalg.norm(self.target.position - agent, ord=2)
            target = self.target.position + (noise[1] if d > 0.5 else 0)
            obs["target"] = target.astype(DTYPE)

            # add obstacles (+ noise if obstacle is too far away)
            obstacle_list = self.static_obstacles + self.dynamic_obstacles
            obstacles = []
            for o, n in zip(obstacle_list, noise[2:]):
                d = np.linalg.norm(o.position - agent, ord=2)
                obstacles.append(o.position + (n if d > 0.5 else 0))
            obs["state"] = np.array(obstacles).astype(DTYPE)

        if self.observation_type != Observation.POS:
            # add image
            rm = self.render_mode
            self.render_mode = "rgb_array"
            obs["image"] = self._render_frame().astype(DTYPE)
            self.render_mode = rm

        return obs

    def _get_infos(self):
        obstacles = self.static_obstacles + self.dynamic_obstacles
        distance = eucl(self.agent.position, self.target.position)
        obs_distance = \
            np.min([eucl(o.position, self.agent.position) for o in obstacles])
        wall_collision = \
            np.any(np.abs(self.agent.position) + self.agent.size >= 1)

        return {
            "win": self.win,
            "collision": self.collision,
            "timestep": self.timestep,
            "distance": distance,
            "obs_distance": obs_distance,
            "wall_collision": wall_collision
        }

    def reset(self, *,
        seed: int=None,
        options: dict = None
    ) -> Tuple[Any, dict]:
        """Reset environment between episodes"""
        super().reset(seed=seed)
        self.options.update((options if options else {}))

        self.static_obstacles = []
        self.dynamic_obstacles = []

        # reset the world to a predefined state
        if self.options["world"]:
            self._load_world()

        # reset the world to a random state
        else:
            self._create_world()

        self.timestep = 0
        self.win = False
        self.collision = False
        return self._get_observations(), self._get_infos()

    def _load_world(self):
        """Load a predefined world"""
        world: dict = self.options["world"]
        # reset target and agent
        self.agent.position = np.array(world.get("agent"))
        self.target.position = np.array(world.get("target"))
        # reset static obstacles
        for static in world.get("static"):
            obstacle = StaticObstacle(self.options["size_static"])
            obstacle.position = np.array(static)
            self.static_obstacles.append(obstacle)
        # reset dynamic obstacles
        for dynamic in world.get("dynamic"):
            speed = self.np_random.uniform(
                self.options["min_speed"],
                self.options["max_speed"],
                size=3
            )
            speed = (speed[:2] / np.linalg.norm(speed[:2], ord=2)) * speed[2]
            obstacle = DynamicObstacle(self.options["size_dynamic"], speed)
            obstacle.position = np.array(dynamic[:2])
            self.dynamic_obstacles.append(obstacle)

    def _create_world(self):
        """Create a new random world"""
        # reset target and agent
        entities = [self.target]
        self.agent.reset(entities, self.np_random)

        # reset static obstacles
        for _ in range(self.options["num_static_obstacles"]):
            obstacle = StaticObstacle(self.options["size_static"])
            obstacle.reset(entities, self.np_random)
            self.static_obstacles.append(obstacle)

        # reset dynamic obstacles
        for _ in range(self.options["num_dynamic_obstacles"]):
            speed = self.np_random.uniform(
                self.options["min_speed"],
                self.options["max_speed"],
                size=3
            )
            speed = (speed[:2] / np.linalg.norm(speed[:2], ord=2)) * speed[2]
            obstacle = DynamicObstacle(self.options["size_dynamic"], speed)
            obstacle.reset(entities, self.np_random)
            self.dynamic_obstacles.append(obstacle)

    def step(self, action: np.ndarray):
        """Perform one time step"""
        # move agent according to chosen action
        self.agent.speed = action.astype(DTYPE)
        self.agent.position = np.clip(
            self.agent.position + self.step_length * action,
            a_min=-1, a_max=1, dtype=DTYPE
        )

        # check win condition
        self.win = self.target.collision(self.agent)

        # move dynamic obstacles
        obstacles = self.static_obstacles + self.dynamic_obstacles
        for obs in self.dynamic_obstacles:
            speed = pg.Vector2(obs.speed.tolist())
            for other in obstacles:
                # obstacles cannot collide with themselves
                if obs is other:
                    continue
                # obstacle collides with another obstacle
                if obs.collision(other):
                    normal = other.position - obs.position
                    if np.all(normal < 1e-3): # problematic if close to 0
                        normal = -obs.speed
                    speed.reflect_ip(pg.Vector2(normal.tolist()).normalize())
            # obstacle collides with an outer wall
            wall_east, wall_south = (obs.position + obs.size > 1).tolist()
            wall_west, wall_north = (obs.position - obs.size < -1).tolist()
            if wall_north:
                speed.reflect_ip(pg.Vector2([0, 1]))
            elif wall_east:
                speed.reflect_ip(pg.Vector2([-1, 0]))
            elif wall_south:
                speed.reflect_ip(pg.Vector2([0, -1]))
            elif wall_west:
                speed.reflect_ip(pg.Vector2([1, 0]))
            # set new speed direction
            obs.speed = np.array([speed.x, speed.y])

            obs.move()

        # check collisions
        self.collision = any(obs.collision(self.agent) for obs in obstacles)

        self.timestep += 1
        terminated = self.win or self.collision
        truncated = self.timestep >= self.options["episode_length"]

        observation = self._get_observations()
        info = self._get_infos()

        return observation, 0, terminated, truncated, info

    def render(self):
        if self.render_mode == "human":
            self._render_frame()
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pg.init() # pylint: disable=no-member
            pg.display.init()
            self.window = pg.display.set_mode((PIXELS, PIXELS))
        if self.clock is None and self.render_mode == "human":
            self.clock = pg.time.Clock()

        canvas = pg.Surface((PIXELS, PIXELS))
        canvas.fill(Color.WHITE.value)

        # draw static obstacles
        for obstacle in self.static_obstacles:
            obstacle.draw(canvas)

        # draw dynamic obstacles
        for obstacle in self.dynamic_obstacles:
            obstacle.draw(canvas)

        # draw the target
        self.target.draw(canvas)

        # draw the agent NOTE: 'draw_direction=False' for dataset generation
        self.agent.draw(
            canvas,
            draw_vision=self.options["uncertainty"],
            draw_direction=False
        )

        if self.render_mode == "human":
            # copy drawings from canvas to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pg.event.pump()
            pg.display.update()

            # draw image at given framerate
            self.clock.tick(self.metadata["render_fps"])

        else: # rgb_array
            return copy.deepcopy(pg.surfarray.pixels3d(canvas))

    def close(self):
        if self.window is not None:
            pg.display.quit()
            pg.quit() # pylint: disable=no-member
