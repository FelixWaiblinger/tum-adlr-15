"""Pretty printing"""

import time

import numpy as np

from utils.constants import MAX_PLAYMODE_STEPS


def print_round_start(g: int):
    """Print info at the start of each episode"""
    print("")
    for i in range(3):
        print(f"\rRound {g+1} starts in {3 - i}...", end="")
        time.sleep(1)
    print("\n===== START =====")


def print_round_info(g: int, t: int, r: float):
    """Print info about current game repeatedly"""
    print(f"\rRound: {g+1} | Time: {t}/{MAX_PLAYMODE_STEPS} | Reward: {r:.3f}", end="")


def print_round_end(w: bool, r: float):
    """Print info at the end of each episode"""
    msg = ('You Won!' if w else 'You Lost!')
    print(f"\n{msg} | Total Reward: {r}")


def print_game_end(w: int, rs: list, ng: int):
    """Print info at the end of the game"""
    print("\n=== GAME OVER ===")
    print(f"Episode Rewards: {[round(r, 3) for r in rs]}")
    print(f"Average Reward: {np.mean(rs):.3f}")
    print(f"Success Rate: {w}/{ng} = {float(w) / ng}")


__all__ = [
    "print_round_start",
    "print_round_info",
    "print_round_end",
    "print_game_end"
]