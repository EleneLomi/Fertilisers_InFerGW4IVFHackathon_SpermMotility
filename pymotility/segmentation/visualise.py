import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def plot_frame(i, video, path, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    ax.imshow(video[i])
    ax.plot(path[:i, 0], path[:i, 1])
    return ax


def animate_path(video, path):
    fig, ax = plt.subplots()
    plot_frame(0, video, path, ax)
    anim = FuncAnimation(
        fig, plot_frame, frames=len(video), fargs=(video, path, ax)
    )
    return anim
