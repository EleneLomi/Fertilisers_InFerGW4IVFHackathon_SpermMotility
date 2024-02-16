import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation


def plot_frame(i, video, path, figax=None):
    if figax is None:
        fig, ax = plt.subplots()
    else:
        fig, ax = figax
    ax.imshow(video[i])
    ax.plot(path[:i, 0], path[:i, 1], "r-.")
    ax.scatter(path[i, 0], path[i, 1], s=10)
    return fig, ax


def animate_path(video, path):
    print("Animating path")
    T, M, N, _ = video.shape
    M // 2
    N // 2
    dists = np.linalg.norm(path[1:] - path[:-1], axis=1)
    thetas = np.arctan2(path[1:, 1] - path[:-1, 1], path[1:, 0] - path[:-1, 0])
    fig, ax = plt.subplots()
    ax.imshow(video[0])

    def update(i):
        u = int(-50 * dists[i] * np.cos(thetas[i]))
        v = int(-50 * dists[i] * np.sin(thetas[i]))
        displaced_path = path - path[i]
        displaced_path = np.array([[N // 2, M // 2]]) - displaced_path
        ax.clear()
        ax.imshow(video[i])
        ax.plot(displaced_path[:i, 0], displaced_path[:i, 1], "r")
        ax.set(xlim=(0, M), ylim=(N, 0))
        ax.axis("off")
        ax.quiver(N // 2, M // 2, u, v, angles="xy", scale_units="xy", scale=10)

    anim = FuncAnimation(fig, update, frames=T - 1, interval=30, repeat=True)
    return anim
