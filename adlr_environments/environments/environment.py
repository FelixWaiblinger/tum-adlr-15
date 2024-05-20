"""2D environment"""

from typing import Any
# import time

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
    "min_size": 1,
    "max_size": 1,
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
        render_mode: str | None = None,
        options: dict[str, Any] = None
    ) -> None:
        """Create new environment"""

        # fill missing options with default values
        options = DEFAULT_OPTIONS | (options if options else {})

        # pygame related stuff
        assert render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.window_size = 512
        self.window = None
        self.clock = None
        self.options = options

        world_size = options["world_size"]

        # observations
        self.agent = Agent()
        self.target = Target()
        self.static_obstacles: list[StaticObstacle] = []
        self.dynamic_obstacles: list[DynamicObstacle] = []
        self.bps = BPS(options["seed"], options["bps_size"], world_size)

        self.observation_space = spaces.Dict({
            "agent": spaces.Box(0, world_size, shape=(2,), dtype=np.float32),
            "target": spaces.Box(0, world_size, shape=(2,), dtype=np.float32),
            "state": spaces.Box(
                low=0, high=world_size * np.sqrt(2), # possible distances
                shape=(options["bps_size"],), dtype=np.float32
            )
        })

        # setting "velocity" in x and y direction independently
        self.action_space = spaces.Box(
            low=-1, high=1, shape=(2,), dtype=np.float32
        )

        self._render_frame()

    def _get_observations(self):
        # start = time.time()
        bps_distances = self.bps.encode(self.pointcloud)
        # print(f"Time for one iteration : {time.time() - start}")

        return {
            "agent": self.agent.position,
            "target": self.target.position,
            "state": bps_distances
        }

    def reset(self, *,
        seed: int | None = None,
        options: dict[str, Any] = None
    ) -> tuple[Any, dict[str, Any]]:
        """Reset environment between episodes"""

        super().reset(seed=seed)
        self.options = self.options | (options if options else {})

        illegal_positions = []
        world_size = self.options.get("world_size", 10)

        # reset agent
        self.agent.reset(world_size, illegal_positions)

        # reset target
        self.target.reset(world_size, illegal_positions)

        # reset static obstacles
        self.static_obstacles.clear()
        num_static = self.options.get("num_static_obstacles", 0)
        min_size = self.options.get("min_size", 1)
        max_size = self.options.get("max_size", 1)

        for _ in range(num_static):
            size = self.np_random.uniform(min_size, max_size, size=2)
            obstacle = StaticObstacle(size)
            obstacle.reset(world_size, illegal_positions)
            self.static_obstacles.append(obstacle)

        # reset dynamic obstacles
        self.dynamic_obstacles.clear()
        num_dynamic = self.options.get("num_dynamic_obstacles", 0)
        min_speed = self.options.get("min_speed", -0.2)
        max_speed = self.options.get("max_speed", 0.2)

        for _ in range(num_dynamic):
            size = self.np_random.uniform(min_size, max_size, size=2)
            speed = self.np_random.uniform(min_speed, max_speed, size=2)
            obstacle = DynamicObstacle(size, speed)
            obstacle.reset(world_size, illegal_positions)
            self.dynamic_obstacles.append(obstacle)

        self._render_frame()

        observation = self._get_observations()
        info = {
            "win": False,
            "collision": False,
            "distance": np.linalg.norm(
                self.agent.position - self.target.position,
                ord=2
            )
        }

        # if self.render_mode == "human":
        #     self._render_frame()

        return observation, info

    def step(self, action):
        """Perform one time step"""

        world_size = self.options["world_size"]
        step_length = self.options["step_length"]

        # move agent according to chosen action
        self.agent.position = np.clip(
            self.agent.position + step_length * action, 0, world_size
        )

        # check win condition
        win = self.target.collision(self.agent)

        # move dynamic obstacles
        for obs in self.dynamic_obstacles:
            obs.move()

        # check collisions
        obstacles = self.static_obstacles + self.dynamic_obstacles
        collision = any(obs.collision(self.agent) for obs in obstacles)

        observation = self._get_observations()
        info = {
            "win": win,
            "collision": collision,
            "distance": np.linalg.norm(
                self.agent.position - self.target.position,
                ord=2
            )
        }

        terminated = win or collision

        if self.render_mode == "human":
            self._render_frame()

        return observation, 0, terminated, False, info

    def render(self):
        if self.render_mode == "rgb_array":
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
        image = np.transpose(
            np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
        )

        self.pointcloud = img2pc(image, world_size)

        if self.render_mode == "human":
            # copy drawings from canvas to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return image

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit() # pylint: disable=no-member
