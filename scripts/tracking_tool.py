import matplotlib.pyplot as plt
from skvideo.io import vread

# path = "tests/data/simple_video/sample1_vid1_sperm14_id17.mp4"
path = "tests/data/videos/sample1_vid1_sperm3_id3.mp4"
video = vread(path)

fig, ax = plt.subplots()
# plt.subplots_adjust(bottom=0.2)
ax.imshow(video[0])
ax.set(title=f"Frame 0/{len(video)}")


def next(event):
    global ind
    ind += 1
    ax.clear()
    ax.set(title=f"Frame {ind}/{len(video)}")
    ax.imshow(video[ind])
    plt.draw()


def prev(event):
    global ind
    ind -= 1
    ax.set(title=f"Frame {ind}/{len(video)}")
    ax.imshow(video[ind])
    plt.draw()


def on_key(event):
    if event.key == "n":
        next(event)
    elif event.key == "b":
        prev(event)


ind = 0


def main():
    # on key press
    plt.connect("key_press_event", lambda event: on_key(event))
    plt.show()


if __name__ == "__main__":
    main()
