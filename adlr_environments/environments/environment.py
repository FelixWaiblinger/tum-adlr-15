"""2D environment"""

from typing import Any

import pygame
import numpy as np
import gymnasium as gym
from gymnasium import spaces


class World2D(gym.Env):
    """Simple 2D environment including agent, target and a discrete action
    space
    """

    metadata = {"render_modes": [None, "human", "rgb_array"], "render_fps": 15}

    def __init__(self, render_mode: str | None=None, size: float=10) -> None:
        """Create new environment"""

        self.size = size
        self.window_size = 512
        self.observation_space = spaces.Dict({
            "agent": spaces.Box(0, size - 1, shape=(2,), dtype=float),
            "target": spaces.Box(0, size - 1, shape=(2,), dtype=float),
            # NOTE: spaces.Sequence prints a warning, because it is not
            # implemented in .../gymnasium/utils/passive_env_checker.py
            # according to https://github.com/DLR-RM/stable-baselines3/issues/1688
            # this seems to be no problem for sb3 though
            # (and it only warns in the first iteration)
            "static": spaces.Sequence(
                spaces.Box(0, size - 1, shape=(2,), dtype=float),
                stack=True
            ),
            "dynamic": spaces.Sequence(
                spaces.Box(0, size - 1, shape=(2,), dtype=float),
                stack=True
            )
        })
        # setting velocity in x and y direction independently
        self.action_space = spaces.Box(
            low=np.array([-1, -1]),
            high=np.array([1, 1]),
            dtype=float
        )

        # pygame related stuff
        assert render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.window = None
        self.clock = None

        # observations
        self._agent_location = None
        self._target_location = None
        self._static_obstacles = []
        self._dynamic_obstacles = []

        # other variables
        self.target_size = np.ones(2)

    def _get_obs(self):
        return {
            "agent": self._agent_location,
            "target": self._target_location,
            "static": [obs[:2] for obs in self._static_obstacles],
            "dynamic": [obs[:2] for obs in self._dynamic_obstacles]
        }

    def _get_info(self):
        return {
            # l2-norm between agent and target
            "distance": np.linalg.norm(
                self._agent_location - self._target_location,
                ord=2
            )
        }

    def reset(self, *,
        seed: int | None=None,
        options: dict[str, Any]=None
    ) -> tuple[Any, dict[str, Any]]:
        """Reset environment between episodes"""

        print("Test")
        super().reset(seed=seed)

        # reset agent
        self._agent_location = self.np_random.uniform(0, self.size, size=2)

        # reset target
        self._target_location = self._agent_location
        while np.array_equal(self._target_location, self._agent_location):
            self._target_location = self.np_random.uniform(
                0, self.size, size=2
            )

        if options:
            # reset static obstacles
            self._static_obstacles.clear()
            num_static = options.get("num_static_obstacles", 0)
            min_size = options.get("min_size", 1)
            max_size = options.get("max_size", 1)

            for _ in range(num_static):
                self._static_obstacles.append(np.concatenate(
                    [self.np_random.uniform(0, self.size, size=2), # position
                    self.np_random.uniform(min_size, max_size, size=2)], # size
                    axis=0
                ))

            # reset dynamic obstacles
            self._dynamic_obstacles.clear()
            num_dynamic = options.get("num_dynamic_obstacles", 0)
            min_speed = options.get("min_speed", -0.2)
            max_speed = options.get("max_speed", 0.2)

            for _ in range(num_dynamic):
                self._dynamic_obstacles.append(np.concatenate(
                    [self.np_random.uniform(0, self.size, size=2), # position
                    self.np_random.uniform(min_size, max_size, size=2), # size
                    self.np_random.uniform(min_speed, max_speed, size=2)], # speed
                    axis=0
                ))

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        """Perform one time step"""

        # move agent according to chosen action
        self._agent_location = np.clip(
            self._agent_location + action, 0, self.size - 1
        )

        # move dynamic obstacles
        for obs in self._dynamic_obstacles:
            obs[:2] += obs[4:6]

        collision = self._check_collision()
        target = self._check_target()
        terminated = target or collision

        # reward is negative for each time step except if target was found
        reward = -1
        if target:
            reward = 100

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info

    def _check_target(self) -> bool:
        lower_bounds = self._target_location - self.target_size / 2
        upper_bounds = self._target_location + self.target_size / 2

        return np.all(np.logical_and(
            self._agent_location > lower_bounds,
            self._agent_location < upper_bounds
        ))

    def _check_collision(self) -> bool:
        obstacles = self._static_obstacles + self._dynamic_obstacles
        for obs in obstacles:
            lower_bounds = obs[:2] - obs[2:4] / 2
            upper_bounds = obs[:2] + obs[2:4] / 2

            if np.all(np.logical_and(
                self._agent_location > lower_bounds,
                self._agent_location < upper_bounds
            )):
                return True

        return False

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
        pix_square_size = (
            self.window_size / self.size
        )  # The size of a single grid square in pixels

        # First we draw the target
        pygame.draw.rect(
            canvas,
            (255, 0, 0),
            pygame.Rect(
                pix_square_size * self._target_location,
                (pix_square_size, pix_square_size),
            ),
        )
        # Now we draw the agent
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (self._agent_location + 0.5) * pix_square_size,
            pix_square_size / 3,
        )

        # Draw static obstacles
        for obs in self._static_obstacles:
            pygame.draw.rect(
                canvas,
                (0, 0, 0),
                pygame.Rect(
                    obs[:2] * pix_square_size,
                    (obs[2] * pix_square_size, obs[3] * pix_square_size)
                )
            )

        # Draw dynamic obstacles
        for obs in self._dynamic_obstacles:
            pygame.draw.rect(
                canvas,
                (0, 255, 0),
                pygame.Rect(
                    obs[:2] * pix_square_size,
                    (obs[2] * pix_square_size, obs[3] * pix_square_size)
                )
            )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit() # pylint: disable=no-member
