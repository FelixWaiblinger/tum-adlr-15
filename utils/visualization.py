"""Visualization and plots"""

import pygame
import numpy as np
import matplotlib.pyplot as plt

from utils.constants import Color, PIXELS


NAMES = [
    "XY",
    "XY\n+Noise",
    "BPS",
    "BPS\n+Noise",
    "AE",
    "AE\n+Noise",
    "3BP",
    "3BP\n+Noise",
    "UBPS",
    "UBPS\n+Noise"
]
COLORS = [
    "yellowgreen",
    "limegreen",
    "mediumseagreen",
    "lightseagreen",
    "steelblue",
    "slateblue",
    "mediumorchid",
    "violet",
    "salmon",
    "sandybrown",
]


def draw_uncertainty(
    canvas: pygame.Surface,
    center: pygame.Vector2,
    radius: float
) -> pygame.Surface:
    """Blur the image except the circular area around the agent (center)"""
    white = Color.WHITE.value
    low_res = PIXELS // 16

    # create mask
    mask = pygame.Surface((PIXELS, PIXELS))
    mask.fill(Color.WHITE.value)
    pygame.draw.circle(mask, Color.BLACK.value, center, radius * PIXELS / 2)
    mask = pygame.mask.from_threshold(mask, white, white)

    # blur image inside mask
    blurred = canvas.copy()
    blurred = pygame.transform.smoothscale(blurred, (low_res, low_res))
    blurred = pygame.transform.smoothscale(blurred, (PIXELS, PIXELS))

    canvas = mask.to_surface(
        surface=canvas,
        setsurface=blurred,
        unsetsurface=canvas
    )


def draw_arrow(
    surface: pygame.Surface,
    start: pygame.Vector2,
    end: pygame.Vector2,
    color: pygame.Color,
    body_width: int = 2,
    head_width: int = 4,
    head_height: int = 2,
):
    """Draw an arrow between start and end with the arrow head at the end.

    Args:
        surface (pygame.Surface): The surface to draw on
        start (pygame.Vector2): Start position
        end (pygame.Vector2): End position
        color (pygame.Color): Color of the arrow
        body_width (int, optional): Defaults to 2.
        head_width (int, optional): Defaults to 4.
        head_height (float, optional): Defaults to 2.
    """
    arrow = start - end
    angle = arrow.angle_to(pygame.Vector2(0, -1))
    body_length = arrow.length() - head_height

    # Create the triangle head around the origin
    head_verts = [
        pygame.Vector2(0, head_height / 2),  # Center
        pygame.Vector2(head_width / 2, -head_height / 2),  # Bottomright
        pygame.Vector2(-head_width / 2, -head_height / 2),  # Bottomleft
    ]
    # Rotate and translate the head into place
    translation = pygame.Vector2(0, arrow.length() - (head_height / 2))
    translation = translation.rotate(-angle)

    for vert in head_verts:
        vert.rotate_ip(-angle)
        vert += translation
        vert += start

    pygame.draw.polygon(surface, color, head_verts)

    # Stop weird shapes when the arrow is shorter than arrow head
    if arrow.length() >= head_height:
        # Calculate the body rect, rotate and translate into place
        body_verts = [
            pygame.Vector2(-body_width / 2, body_length / 2),  # Topleft
            pygame.Vector2(body_width / 2, body_length / 2),  # Topright
            pygame.Vector2(body_width / 2, -body_length / 2),  # Bottomright
            pygame.Vector2(-body_width / 2, -body_length / 2),  # Bottomleft
        ]
        translation = pygame.Vector2(0, body_length / 2).rotate(-angle)
        for vert in body_verts:
            vert.rotate_ip(-angle)
            vert += translation
            vert += start

        pygame.draw.polygon(surface, color, body_verts)


def draw_policy(
    agent,
    observation: np.ndarray,
    arrows_per_row: int=10
) -> None:
    """Plot a vector field representing the actions chosen by the policy for
    any position in the environment
    """

    # start arrows in the middle of "their box"
    offset = 2 / (arrows_per_row + 1)
    for i in np.linspace(-1 + offset, 1 - offset, arrows_per_row):
        for j in np.linspace(-1 + offset, 1 - offset, arrows_per_row):
            observation[0, :2] = np.array([i, j], dtype=np.float32)
            target, _ = agent.predict(observation, deterministic=True)
            target = 0.5 * target[0] # shorten for visibility
            plt.arrow(i, j, target[0], -target[1], head_width=0.1)
    plt.show()


def draw_bps(points: np.ndarray, pointcloud: np.ndarray) -> None:
    test = []
    plt.scatter(points[:, 0], points[:, 1], color='r')
    plt.scatter(pointcloud[:, 0], pointcloud[:, 1])
    for p in points:
        diffs = np.subtract(pointcloud, p)
        dists = np.linalg.norm(diffs, ord=2, axis=1)
        match = np.argmin(dists)
        test.append(dists[match])
        plt.arrow(p[0], p[1], diffs[match, 0], diffs[match, 1])
    plt.gca().invert_yaxis()
    plt.show()


def create_learning_curve_plot(data_paths: list):
    n_models = len(data_paths)
    mean_rewards = []

    # collect and sort saved data
    for i, path in enumerate(data_paths):
        data = np.genfromtxt(path, delimiter=',')[1:, 1:]
        mean_rewards.append(data)

    handles = []
    for i in range(n_models):
        xs = mean_rewards[i][:, 0]
        ys = mean_rewards[i][:, 1]
        ys = np.pad(ys, 10, mode="edge")[:-1]
        window = np.lib.stride_tricks.sliding_window_view(ys, 20)
        ys_mean = np.mean(window, axis=-1)
        ys_std = np.clip(np.var(window, axis=-1), 0, 5)
        plt.fill_between(xs, ys_mean-ys_std, ys_mean+ys_std, alpha=0.3, color=COLORS[2*i])
        line, = plt.plot(xs, ys_mean, color=COLORS[2*i])
        handles.append(line)
    plt.title("Learning Curve per Agent")
    plt.legend(handles, NAMES[0:8:2], fontsize="large")
    plt.xlabel("Training Steps")
    plt.ylabel("Mean Reward")
    plt.grid()
    plt.show()


def create_success_plot(data_paths: list, env: str, obs: str):
    n_models = len(data_paths)
    success_rates = np.zeros((n_models, 100))

    # collect and sort saved data
    for i, path in enumerate(data_paths):
        data = np.load(path, allow_pickle=True)['arr_0']
        success_rates[i] = data[:, 0]
    
    # success rate plot
    a = plt.subplot()
    violin = a.violinplot(
        dataset=(success_rates * 100).tolist(),
        positions=[0, 1, 2, 3],
        showmeans=False,
        showmedians=True
    )

    for part in ["cbars", "cmaxes", "cmins", "cmedians"]:
        violin[part].set_edgecolor("black")

    for i, vp in enumerate(violin['bodies']):
        vp.set_facecolor(COLORS[2*i])
        vp.set_alpha(0.7)

    a.set_xticks(ticks=[0, 1, 2, 3], labels=NAMES[0:8:2])
    plt.title("Performance in " + env + " Environment")
    plt.text(x=1.8, y=95, s=obs, fontsize="large", bbox={"facecolor": "white"})
    plt.xlabel("Agents")
    plt.ylabel("Success per Episode [%]")
    plt.grid(axis='y')
    plt.show()


def create_uncertainty_plot(data_paths: list):
    """Create violin plots for success, crash and reward of various models"""
    n_models = len(data_paths)
    mean_rewards = np.zeros((n_models, 100))

    # collect and sort saved data
    for i, path in enumerate(data_paths):
        data = np.load(path, allow_pickle=True)['arr_0']
        mean_rewards[i] = data[:, 2][0]

    # reward plot
    low = np.min(mean_rewards) - 1
    plt.bar(
        NAMES,
        height=mean_rewards[:, 0] - low,
        bottom=low,
        color=COLORS
    )
    plt.title("Effects of Noisy Observations")
    plt.xlabel("Agents")
    plt.ylabel("Mean Reward per Episode")
    plt.grid(axis="y")
    plt.show()


__all__ = [
    "draw_policy",
    "draw_arrow",
    "draw_uncertainty",
]
