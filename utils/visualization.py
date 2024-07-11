"""Visualization and plots"""

import pygame
import numpy as np
import matplotlib.pyplot as plt

from utils.constants import Color, PIXELS


def draw_uncertainty(
    canvas: pygame.Surface,
    center: pygame.Vector2,
    radius: float
) -> pygame.Surface:
    """Blur the image except the circular area around the agent (center)"""
    white = Color.WHITE.value
    low_res = PIXELS // 32

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


def create_uncertainty_plots(data_paths: list):
    """Create violin plots for success, crash and reward of various models"""
    success_rates = np.zeros((len(data_paths), 100))
    crash_rates = np.zeros((len(data_paths), 100))
    mean_rewards = np.zeros((len(data_paths), 100))

    # collect and sort saved data
    for i, path in enumerate(data_paths):
        data = np.load(path, allow_pickle=True)['arr_0']
        success_rates[i] = data[:, 0]
        crash_rates[i] = data[:, 1]
        mean_rewards[i] = data[:, 2][0]

    names = [
        "None",
        "Inference",
        "Training",
        "Training &\nInference"
    ]
    color = [
        "slateblue",
        "cornflowerblue",
        "olivedrab",
        "yellowgreen"
    ]

    # success rate plot
    a = plt.subplot()
    violin = a.violinplot(
        dataset=success_rates.tolist(),
        positions=[1, 2, 3, 4],
        showmeans=False,
        showmedians=True
    )

    for part in ["cbars", "cmaxes", "cmins", "cmedians"]:
        violin[part].set_edgecolor("black")

    for i, vp in enumerate(violin['bodies']):
        vp.set_facecolor(color[i])
        vp.set_alpha(0.7)

    a.set_xticks(ticks=[1, 2, 3, 4], labels=names)
    plt.suptitle("Effects of Uncertainty")
    plt.title("Average Success-rate per Episode")
    plt.show()

    # crash rate plot
    a = plt.subplot()
    violin = a.violinplot(
        dataset=crash_rates.tolist(),
        showmeans=False,
        showmedians=True
    )

    for part in ["cbars", "cmaxes", "cmins", "cmedians"]:
        violin[part].set_edgecolor("black")

    for i, vp in enumerate(violin['bodies']):
        vp.set_facecolor(color[i])
        vp.set_alpha(0.7)

    a.set_xticks(ticks=[1, 2, 3, 4], labels=names)
    plt.suptitle("Effects of Uncertainty")
    plt.title("Average Crash-rate per Episode")
    plt.show()

    # reward plot
    low = np.min(mean_rewards) - 1
    plt.bar(
        names,
        height=mean_rewards[:, 0] - low,
        bottom=low,
        color=color
    )
    plt.suptitle("Effects of Uncertainty")
    plt.title("Average Reward per Episode")

    plt.show()


def create_training_plots(data_paths: list):
    """Test"""
    pass


__all__ = [
    "draw_policy",
    "draw_arrow",
    "draw_uncertainty",
]
