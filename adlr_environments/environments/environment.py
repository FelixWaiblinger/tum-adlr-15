"""2D environment"""

from typing import Any, Dict, Tuple
import copy

import pygame
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from state_representation.bps import BPS, img2pc
from .entity import Agent, Target, StaticObstacle, DynamicObstacle


DEFAULT_OPTIONS = {
    "seed": 42,
    "world_size": 10,
    "step_length": 0.1,
    "num_static_obstacles": 5,
    "num_dynamic_obstacles": 0,
    "min_speed": 0.1,
    "max_speed": 0.1,
    "bps_size": 100,
}


class World2D(gym.Env):
    """Simple 2D environment including agent, target and a discrete action
    space
    """

    metadata = {"render_modes": [None, "human", "rgb_array"], "render_fps": 60}

    def __init__(self,
        render_mode=None, #,: str | None = None,
        options: Dict[str, Any] = None
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

        world_size = options["world_size"]
        obstacles = options["num_static_obstacles"]

        # observations
        self.agent = Agent()
        self.target = Target()
        self.static_obstacles: list[StaticObstacle] = []
        self.dynamic_obstacles: list[DynamicObstacle] = []
        self.pointcloud = None
        self.bps = BPS(options["seed"], options["bps_size"], world_size)
        self.timestep = 0

        self.observation_space = spaces.Dict({
            "agent": spaces.Box(0, world_size, shape=(2,), dtype=np.float32),
            "target": spaces.Box(0, world_size, shape=(2,), dtype=np.float32),
            "state": spaces.Box(
                0, world_size, shape=(obstacles * 2,), dtype=np.float32
            )
            # "state": spaces.Box(
            #     low=0, high=world_size * np.sqrt(2), # possible distances
            #     shape=(options["bps_size"],), dtype=np.float32
            # )
        })

        # setting "velocity" in x and y direction independently
        self.action_space = spaces.Box(
            low=-1, high=1, shape=(2,), dtype=np.float32
        )

    def _get_observations(self):
        # bps_distances = self.bps.encode(self.pointcloud)
        obstacles = np.array([x.position for x in self.static_obstacles])
        obstacles = obstacles.flatten().astype(np.float32)
        
        return {
            "agent": self.agent.position,
            "target": self.target.position,
            # "state": bps_distances
            "state": obstacles
        }

    def reset(self, *,
        seed=None, #: int | None = None,
        options: Dict[str, Any] = None
    ) -> Tuple[Any, Dict[str, Any]]:
        """Reset environment between episodes"""

        super().reset(seed=seed)
        self.options.update((options if options else {}))

        illegal_positions = []
        world_size = self.options["world_size"]

        # reset agent
        self.agent.reset(world_size, illegal_positions, self.np_random)

        # reset target
        self.target.reset(world_size, illegal_positions, self.np_random)

        # reset static obstacles
        self.static_obstacles = []
        num_static = self.options["num_static_obstacles"]

        for _ in range(num_static):
            obstacle = StaticObstacle()
            obstacle.reset(world_size, illegal_positions, self.np_random)
            self.static_obstacles.append(obstacle)

        # reset dynamic obstacles
        self.dynamic_obstacles = []
        num_dynamic = self.options["num_dynamic_obstacles"]
        min_speed = self.options["min_speed"]
        max_speed = self.options["max_speed"]

        for _ in range(num_dynamic):
            speed = self.np_random.uniform(min_speed, max_speed, size=2)
            obstacle = DynamicObstacle(speed)
            obstacle.reset(world_size, illegal_positions, self.np_random)
            self.dynamic_obstacles.append(obstacle)

        self.timestep = 0

        observation = self._get_observations()
        info = {
            "win": False,
            "collision": False,
            "timestep": self.timestep,
            "distance": np.linalg.norm(
                self.agent.position - self.target.position,
                ord=2
            ),
        }

        return observation, info

    def step(self, action):
        """Perform one time step"""

        world_size = self.options["world_size"]
        step_length = self.options["step_length"]

        # move agent according to chosen action
        self.agent.position = np.clip(
            self.agent.position + step_length * action,
            0, world_size, dtype=np.float32
        )

        # check win condition
        win = self.target.collision(self.agent)

        # move dynamic obstacles
        for obs in self.dynamic_obstacles:
            obs.move()

        # check collisions
        obstacles = self.static_obstacles + self.dynamic_obstacles
        collision = any(obs.collision(self.agent) for obs in obstacles)
        
        terminated = win or collision
        self.timestep += 1

        observation = self._get_observations()
        info = {
            "win": win,
            "collision": collision,
            "timestep": self.timestep,
            "distance": np.linalg.norm(
                self.agent.position - self.target.position,
                ord=2
            )
        }

        return observation, 0, terminated, False, info

    def render(self):
        if self.render_mode == "human":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init() # pylint: disable=no-member
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.window_size, self.window_size)
            )
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))

        # conversion ratio from world to pixel coordinates
        world_size = self.options["world_size"]
        world2canvas = self.window_size / world_size

        # draw static obstacles
        for obstacle in self.static_obstacles:
            obstacle.draw(canvas, world2canvas)

        # draw dynamic obstacles
        for obstacle in self.dynamic_obstacles:
            obstacle.draw(canvas, world2canvas)

        # draw the target
        self.target.draw(canvas, world2canvas)

        # draw the agent
        self.agent.draw(canvas, world2canvas)

        # create pointcloud from currently rendered image
        image = copy.deepcopy(pygame.surfarray.pixels3d(canvas))
        image = image[::4, ::4, :] # resolution = 0.25 * window_size
        self.pointcloud = img2pc(image, world_size)

        if self.render_mode == "human":
            # copy drawings from canvas to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # draw image at given framerate
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return image

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit() # pylint: disable=no-member
