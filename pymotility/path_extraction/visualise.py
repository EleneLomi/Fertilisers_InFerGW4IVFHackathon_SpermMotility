import matplotlib.pyplot as plt
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
    fig, ax = plt.subplots()
    plot_frame(0, video, path, ax)
    anim = FuncAnimation(
        fig, plot_frame, frames=len(video), fargs=(video, path, ax)
    )
    return anim
