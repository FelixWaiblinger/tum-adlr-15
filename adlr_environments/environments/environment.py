"""2D environment"""

from typing import Any, Dict, Tuple
import copy

import pygame
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from state_representation.bps import BPS, img2pc
from .entity import Agent, Target, StaticObstacle, DynamicObstacle
from adlr_environments.utils import MAX_EPISODE_STEPS, eucl


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

FIXED_POSITIONS = {
    "agent": np.array([1, 1], dtype=np.float32),
    "target": np.array([4, 4], dtype=np.float32),
    "static_obstacle": [np.array([2.5, 2.5], dtype=np.float32), np.array([3, 3], dtype=np.float32), np.array([4, 4], dtype=np.float32), np.array([8, 7], dtype=np.float32), np.array([7, 8], dtype=np.float32)]
}


class World2D(gym.Env):
    """Simple 2D environment including agent, target and a discrete action
    space
    """

    metadata = {"render_modes": [None, "human", "rgb_array"], "render_fps": 30}

    def __init__(self,
        render_mode: str=None,
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
        # num_obstacles = options["num_static_obstacles"]

        # observations
        self.agent = Agent(position=FIXED_POSITIONS["agent"], random=False)
        self.target = Target(position=FIXED_POSITIONS["target"], random=False)
        self.static_obstacles: list[StaticObstacle] = []
        self.dynamic_obstacles: list[DynamicObstacle] = []
        self.pointcloud = None
        self.bps = BPS(options["seed"], options["bps_size"], world_size)
        self.timestep = 0

        self.observation_space = spaces.Dict({
            "agent": spaces.Box(0, world_size, shape=(2,), dtype=np.float32),
            "target": spaces.Box(0, world_size, shape=(2,), dtype=np.float32),
            "state": spaces.Box(
                0, world_size, shape=(options["num_static_obstacles"] * 2, ), dtype=np.float32
            )
            # "state": spaces.Box(
            #     low=0, high=world_size * np.sqrt(2), # possible distances
            #     shape=(options["bps_size"],), dtype=np.float32
            # )
        })

        # setting "velocity" in x and y direction independently
        self.action_space = spaces.Box(
            low=-5, high=5, shape=(2,), dtype=np.float32
        )
        self.velocity = 0

    def _get_observations(self):
        self.pointcloud = np.array([o.position for o in self.static_obstacles])
        #bps_distances = self.bps.encode(self.pointcloud)
        obstacles = np.array([x.position for x in self.static_obstacles])
        obstacles = obstacles.flatten().astype(np.float32)
        # agent = np.concatenate([
        #     self.agent.position,
        #     self.agent.speed
        # ], dtype=np.float32)
        agent = self.agent.position
        
        return {
            "agent": agent,
            "target": self.target.position,
            #"state": bps_distances
            "state": obstacles
        }
    
    def _get_infos(self, win: bool=False, collision: bool=False, wall_collision: bool=False):
        obstacles = self.static_obstacles + self.dynamic_obstacles

        return {
            "win": win,
            "collision": collision,
            "timestep": self.timestep,
            "distance": eucl(self.agent.position, self.target.position),
            # "obs_distance": np.min(
            #     [eucl(o.position, self.agent.position) for o in obstacles]
            # ),
            "wall_collision": wall_collision
        }


    def reset(self, *,
        seed: int=None,
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

        for i in range(self.options["num_static_obstacles"]):
            obstacle = StaticObstacle(position=FIXED_POSITIONS["static_obstacle"][i], random=True)
            obstacle.reset(world_size, illegal_positions, self.np_random)
            self.static_obstacles.append(obstacle)

        # reset dynamic obstacles
        self.dynamic_obstacles = []
        min_speed = self.options["min_speed"]
        max_speed = self.options["max_speed"]

        for _ in range(self.options["num_dynamic_obstacles"]):
            speed = self.np_random.uniform(min_speed, max_speed, size=2)
            obstacle = DynamicObstacle(speed)
            obstacle.reset(world_size, illegal_positions, self.np_random)
            self.dynamic_obstacles.append(obstacle)

        self.timestep = 0

        return self._get_observations(), self._get_infos()

    def step(self, action: np.ndarray):
        """Perform one time step"""

        world_size = self.options["world_size"]
        step_length = self.options["step_length"]

        # move agent according to chosen action
        action = action/np.linalg.norm(action)

        #self.velocity += action
        self.agent.speed = action.astype(np.float32)
        self.velocity = 0.6 * self.velocity + 0.4 * step_length * action

        self.agent.position = np.clip(
            self.agent.position + self.velocity,
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
        wall_collision = self.agent.wall_collision(self.options["world_size"])

        self.timestep += 1
        terminated = win or collision
        truncated = self.timestep >= MAX_EPISODE_STEPS

        observation = self._get_observations()
        info = self._get_infos(win, collision, wall_collision)

        if self.render_mode == "human":
            self.render()

        return observation, 0, terminated, truncated, info

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
