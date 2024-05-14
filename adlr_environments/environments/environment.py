"""2D environment"""

from typing import Any

import pygame
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .entity import Agent, Target, StaticObstacle, DynamicObstacle


DEFAULT_OPTIONS = {
    "world_size": 10,
    "step_length": 0.1,
    "num_static_obstacles": 0,
    "num_dynamic_obstacles": 0,
    "min_size": 1,
    "max_size": 1,
    "min_speed": 0.1,
    "max_speed": 0.1,
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

        world_size = options.get("world_size", 10)

        # observations
        self.pointcloud = None
        self.agent = Agent()
        self.target = Target()
        self.static_obstacles: list[StaticObstacle] = []
        self.dynamic_obstacles: list[DynamicObstacle] = []
        self.observation_space = spaces.Dict({
            "agent": spaces.Box(0, world_size, shape=(2,), dtype=float),
            "target": spaces.Box(0, world_size, shape=(2,), dtype=float),
            # OPTION 1: observe either center xy of each obstacle or pointcloud
            #           of all occupied xy's
            "static": spaces.Sequence(
                spaces.Box(0, world_size, shape=(2,), dtype=float),
                stack=True
            ),
            "dynamic": spaces.Sequence(
                spaces.Box(0, world_size, shape=(2,), dtype=float),
                stack=True
            ),
            # OPTION 2: observe point cloud of the entire environment
            # "environment": spaces.Sequence(
            #     spaces.Box(0, world_size, shape=(2,), dtype=float),
            #     stack=True
            # )
            # NOTE: spaces.Sequence prints a warning, because it is not
            #       implemented in .../gymnasium/utils/passive_env_checker.py
            #       see https://github.com/DLR-RM/stable-baselines3/issues/1688
            #       this seems to be no problem for sb3 though
            #       (and it only warns in the first iteration)
            # NOTE: MAYBE IT IS AN ISSUE! :angry-emoji:
        })

        # setting "velocity" in x and y direction independently
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=float)

    def _get_observations(self):
        # TODO: How are the observations of a sequence supposed to look like if
        #       there are no elements in the sequence???
        #       If I use None it says should be ndarray and vice versa...
        static = np.array(o.position for o in self.static_obstacles) \
            if self.static_obstacles else np.array([])

        dynamic = np.array(o.position for o in self.dynamic_obstacles) \
            if self.dynamic_obstacles else np.array([])

        return {
            "agent": self.agent.position,
            "target": self.target.position,
            # currently only observe the center position of the obstacles
            "static": static,
            "dynamic": dynamic,
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

        observation = self._get_observations()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

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
        for obstacle in self.dynamic_obstacles:
            obstacle.move()

        # check collisions
        obstacles = self.static_obstacles + self.dynamic_obstacles
        collision = any(obs.collision(self.agent) for obs in obstacles)

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
        world2canvas = self.window_size / self.options["world_size"]

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
        # pointcloud = np.transpose(pygame.surfarray.pixels3d(canvas)) #, axes=(1, 0, 2)

        if self.render_mode == "human":
            # copy drawings from canvas to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                pygame.surfarray.pixels3d(canvas), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit() # pylint: disable=no-member
