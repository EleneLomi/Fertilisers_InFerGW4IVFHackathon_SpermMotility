import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from skvideo.io import vread

path = "tests/data/simple_video/sample1_vid1_sperm14_id17.mp4"
video = vread(path)

fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.2)
ax.imshow(video[0])


def next(event):
    global ind
    ind += 1
    ax.imshow(video[ind])
    plt.draw()


def prev(event):
    global ind
    ind -= 1
    ax.imshow(video[ind])
    plt.draw()


def on_key(event):
    if event.key == "n":
        next(event)
    elif event.key == "p":
        prev(event)


ind = 0


def main():
    axprev = plt.axes([0.7, 0.05, 0.1, 0.075])
    axnext = plt.axes([0.81, 0.05, 0.1, 0.075])
    bnext = Button(axnext, "Next")
    bnext.on_clicked(next)
    bprev = Button(axprev, "Previous")
    bprev.on_clicked(prev)
    # on key press
    plt.connect("key_press_event", lambda event: on_key(event))
    plt.show()


if __name__ == "__main__":
    main()
