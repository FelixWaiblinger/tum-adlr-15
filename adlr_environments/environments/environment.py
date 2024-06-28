"""2D environment"""

from typing import Any, List, Dict, Tuple
import copy

import torch
import pygame as pg
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import spaces

from adlr_environments.utils import eucl
from adlr_environments.constants import MAX_EPISODE_STEPS, WHITE #, GREEN, RED
from state_representation import AutoEncoder, BPS, NormalizeTransform
from .entity import Agent, Target, StaticObstacle, DynamicObstacle


AUTOENCODER = "./state_representation/autoencoder.pt"
DEFAULT_OPTIONS = {
    "world": None,
    "seed": 42,
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
}


class World2D(gym.Env):
    """Simple 2D environment including agent, target and a discrete action
    space
    """

    metadata = {"render_modes": [None, "human", "rgb_array"], "render_fps": 60}

    def __init__(self,
        render_mode: str=None,
        options: Dict[str, Any]=None
    ) -> None:
        """Create new environment"""

        # fill missing options with default values
        opts = DEFAULT_OPTIONS
        opts.update((options if options else {}))
        options = opts

        # pygame related stuff
        assert render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.window_size = 512
        self.window = None
        self.clock = None
        self.options = options
        self.plot = None

        latent = options["latent_size"]

        # observations
        self.win = False
        self.collision = False
        self.pointcloud = None
        self.timestep = 0
        self.model: AutoEncoder = torch.load(AUTOENCODER)
        self.transform = NormalizeTransform(start=(0, 255), end=(0, 1))
        self.agent = Agent(options["size_agent"])
        self.target = Target(options["size_target"])
        self.static_obstacles: List[StaticObstacle] = []
        self.dynamic_obstacles: List[DynamicObstacle] = []
        self.bps = BPS(options["seed"], latent)
        self.step_length = options["step_length"]
        self.observation_space = spaces.Dict({
            # NOTE: AutoEncoder
            "agent": spaces.Box(-1, 1, shape=(4,), dtype=np.float32),
            "target": spaces.Box(-1, 1, shape=(2,), dtype=np.float32),
            "state": spaces.Box(-1, 1, shape=(latent,), dtype=np.float32)
            # NOTE: BPS
            # "state": spaces.Box(
            #     low=0, high=2 * np.sqrt(2), shape=(latent,), dtype=np.float32
            # )
        })

        # setting "velocity" in x and y direction independently
        self.action_space = spaces.Box(
            low=-1, high=1, shape=(2,), dtype=np.float32
        )

    # NOTE: AutoEncoder
    def _get_observations(self):
        if self.pointcloud is None:
            rm = self.render_mode
            self.render_mode = "rgb_array"
            self.pointcloud = self._render_frame()
            self.render_mode = rm

        image = self.pointcloud[2::4, 2::4, :].transpose([2, 0, 1])
        image = self.transform(torch.from_numpy(np.expand_dims(image, 0)))
        state = self.model.encoder.forward(image.to(self.model.device))

        # TODO: debugging
        # reconstruction = self.model.decoder.forward(state)
        # reconstruction = reconstruction.cpu().detach().numpy()[0]
        # reconstruction = reconstruction.transpose([2, 1, 0])
        # self.plot = plt.imshow(reconstruction)
        # plt.show()

        agent: np.ndarray = np.concatenate([
            self.agent.position,
            self.agent.speed
        ])

        return {
            "agent": agent.astype(np.float32),
            "target": self.target.position.astype(np.float32),
            "state": state.cpu().detach().numpy()[0]
        }

    # NOTE: BPS
    # def _get_observations(self):
    #     obstacles = self.static_obstacles + self.dynamic_obstacles
    #     pointcloud = np.array([o.position for o in obstacles])
    #     bps_distances = self.bps.encode(pointcloud)
    #     agent: np.ndarray = np.concatenate([
    #         self.agent.position,
    #         self.agent.speed
    #     ])

    #     return {
    #         "agent": agent.astype(np.float32),
    #         "target": self.target.position.astype(np.float32),
    #         "state": bps_distances.astype(np.float32)
    #     }

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
        options: Dict[str, Any] = None
    ) -> Tuple[Any, Dict[str, Any]]:
        """Reset environment between episodes"""
        super().reset(seed=seed)
        self.options.update((options if options else {}))

        self.static_obstacles = []
        self.dynamic_obstacles = []
        static_size = self.options["size_static"]
        dynamic_size = self.options["size_dynamic"]
        min_speed = self.options["min_speed"]
        max_speed = self.options["max_speed"]

        # reset the world to a predefined state
        if self.options["world"]:
            world: Dict = self.options["world"]
            # reset target and agent
            self.agent.position = np.array(world.get("agent"))
            self.target.position = np.array(world.get("target"))
            # reset static obstacles
            for static in world.get("static"):
                obstacle = StaticObstacle(static_size)
                obstacle.position = np.array(static)
                self.static_obstacles.append(obstacle)
            # reset dynamic obstacles
            for dynamic in world.get("dynamic"):
                speed = self.np_random.uniform(min_speed, max_speed, size=3)
                speed = (speed[:2] / np.linalg.norm(speed[:2], ord=2)) * speed[2]
                obstacle = DynamicObstacle(dynamic_size, speed)
                obstacle.position = np.array(dynamic[:2])
                self.dynamic_obstacles.append(obstacle)

        # reset the world to a random state
        else:
            # reset target and agent
            entities = [self.target]
            self.agent.reset(entities, self.np_random)

            # reset static obstacles
            for _ in range(self.options["num_static_obstacles"]):
                obstacle = StaticObstacle(static_size)
                obstacle.reset(entities, self.np_random)
                self.static_obstacles.append(obstacle)

            # reset dynamic obstacles
            for _ in range(self.options["num_dynamic_obstacles"]):
                speed = self.np_random.uniform(min_speed, max_speed, size=3)
                speed = (speed[:2] / np.linalg.norm(speed[:2], ord=2)) * speed[2]
                obstacle = DynamicObstacle(dynamic_size, speed)
                obstacle.reset(entities, self.np_random)
                self.dynamic_obstacles.append(obstacle)

        self.timestep = 0
        self.win = False
        self.collision = False
        return self._get_observations(), self._get_infos()

    def step(self, action: np.ndarray):
        """Perform one time step"""

        # move agent according to chosen action
        self.agent.speed = action.astype(np.float32)
        self.agent.position = np.clip(
            self.agent.position + self.step_length * action,
            a_min=-1, a_max=1, dtype=np.float32
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
            self.window = pg.display.set_mode(
                (self.window_size, self.window_size)
            )
        if self.clock is None and self.render_mode == "human":
            self.clock = pg.time.Clock()

        canvas = pg.Surface((self.window_size, self.window_size))
        canvas.fill(WHITE)

        # draw static obstacles
        for obstacle in self.static_obstacles:
            obstacle.draw(canvas)

        # draw dynamic obstacles
        for obstacle in self.dynamic_obstacles:
            obstacle.draw(canvas)

        # draw the target
        self.target.draw(canvas)

        # draw the agent
        # NOTE: 'draw_direction=False' for dataset generation
        self.agent.draw(canvas, draw_direction=False)

        # create pointcloud from currently rendered image
        image = copy.deepcopy(pg.surfarray.pixels3d(canvas))

        # NOTE: if an actual pointcloud becomes necessary later
        # reduce resolution by taking every fourth pixel
        # self.pointcloud = img2pc(image[2::4, 2::4, :], self.window_size)

        if self.render_mode == "human":
            # copy drawings from canvas to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pg.event.pump()
            pg.display.update()

            # draw image at given framerate
            self.clock.tick(self.metadata["render_fps"])

            # NOTE: AutoEncoder
            self.pointcloud = image
        else:  # rgb_array
            return image

    def close(self):
        if self.window is not None:
            pg.display.quit()
            pg.quit() # pylint: disable=no-member
