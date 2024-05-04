"""2D environment"""

from typing import Any

import pygame
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .entity import Agent, Target, StaticObstacle, DynamicObstacle


class World2D(gym.Env):
    """Simple 2D environment including agent, target and a discrete action
    space
    """

    metadata = {"render_modes": [None, "human", "rgb_array"], "render_fps": 60}

    def __init__(self,
        render_mode: str | None=None,
        world_size: float=10
    ) -> None:
        """Create new environment"""

        # pygame related stuff
        assert render_mode in self.metadata["render_modes"]
        self.world_size = world_size
        self.render_mode = render_mode
        self.window_size = 512
        self.window = None
        self.clock = None

        # observations
        self.pointcloud = None
        self.agent = Agent()
        self.target = Target()
        self.static_obstacles: list[StaticObstacle] = []
        self.dynamic_obstacles: list[DynamicObstacle] = []
        self.observation_space = spaces.Dict({
            "agent": spaces.Box(0, world_size - 1, shape=(2,), dtype=float),
            "target": spaces.Box(0, world_size - 1, shape=(2,), dtype=float),
            # NOTE: spaces.Sequence prints a warning, because it is not
            # implemented in .../gymnasium/utils/passive_env_checker.py
            # see: https://github.com/DLR-RM/stable-baselines3/issues/1688
            # this seems to be no problem for sb3 though
            # (and it only warns in the first iteration)
            "static": spaces.Sequence(
                spaces.Box(0, world_size - 1, shape=(2,), dtype=float),
                stack=True
            ),
            "dynamic": spaces.Sequence(
                spaces.Box(0, world_size - 1, shape=(2,), dtype=float),
                stack=True
            )
        })

        # setting velocity in x and y direction independently
        self.action_space = spaces.Box(
            low=np.array([-1, -1]),
            high=np.array([1, 1]),
            dtype=float
        )

    def _get_observations(self):
        return {
            "agent": self.agent.position,
            "target": self.target.position,
            # currently only observe the center position of the obstacles
            "static": [obs.position for obs in self.static_obstacles],
            "dynamic": [obs.position for obs in self.dynamic_obstacles],
            # "pc": self.pointcloud
        }

    def _get_info(self):
        return {
            # l2-norm between agent and target
            "distance": np.linalg.norm(
                self.agent.position - self.target.position,
                ord=2
            )
        }

    def reset(self, *,
        seed: int | None=None,
        options: dict[str, Any]=None
    ) -> tuple[Any, dict[str, Any]]:
        """Reset environment between episodes"""

        super().reset(seed=seed)

        illegal_positions = []

        # reset agent
        self.agent.reset(self.world_size, illegal_positions)

        # reset target
        self.target.reset(self.world_size, illegal_positions)

        if options:
            # reset static obstacles
            self.static_obstacles.clear()
            num_static = options.get("num_static_obstacles", 0)
            min_size = options.get("min_size", 1)
            max_size = options.get("max_size", 1)

            for _ in range(num_static):
                size = self.np_random.uniform(min_size, max_size, size=2)
                obstacle = StaticObstacle(size)
                obstacle.reset(self.world_size, illegal_positions)
                self.static_obstacles.append(obstacle)

            # reset dynamic obstacles
            self.dynamic_obstacles.clear()
            num_dynamic = options.get("num_dynamic_obstacles", 0)
            min_speed = options.get("min_speed", -0.2)
            max_speed = options.get("max_speed", 0.2)

            for _ in range(num_dynamic):
                size = self.np_random.uniform(min_size, max_size, size=2)
                speed = self.np_random.uniform(min_speed, max_speed, size=2)
                obstacle = DynamicObstacle(size, speed)
                obstacle.reset(self.world_size, illegal_positions)
                self.dynamic_obstacles.append(obstacle)

        observation = self._get_observations()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        """Perform one time step"""

        # move agent according to chosen action
        self.agent.position = np.clip(
            self.agent.position + action, 0, self.world_size - 1
        )

        # check win condition
        win = self.target.collision(self.agent)
        if win:
            print("win")

        # move dynamic obstacles
        for obstacle in self.dynamic_obstacles:
            obstacle.move()

        # check collisions
        collision = False
        for obstacle in self.static_obstacles + self.dynamic_obstacles:
            collision = obstacle.collision(self.agent)
            if collision:
                break

        # reward is negative for each time step except if target was found
        reward = 100 if win else -1
        terminated = win or collision

        observation = self._get_observations()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info

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
        world2canvas = self.window_size / self.world_size

        # draw the target
        self.target.draw(canvas, world2canvas)

        # draw static obstacles
        for obstacle in self.static_obstacles:
            obstacle.draw(canvas, world2canvas)

        # draw dynamic obstacles
        for obstacle in self.dynamic_obstacles:
            obstacle.draw(canvas, world2canvas)

        # draw the agent
        self.agent.draw(canvas, world2canvas)

        # save current image
        # TODO convert to actual point cloud with object information
        self.pointcloud = np.transpose(
            np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
        )

        if self.render_mode == "human":
            # copy drawings from canvas to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return self.pointcloud

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit() # pylint: disable=no-member
