from skvideo.io import vread
import matplotlib.pyplot as plt


def tracking_tool(video):
    """This function allows you navigate through the video frame by frame using p and n."""
    fig, ax = plt.subplots()

    def show_frame(i):
        print(f"Frame {i+1}/{len(video)}")
        ax.imshow(video[i])
        ax.set_title(f"Frame {i+1}/{len(video)}")
        plt.draw()

    show_frame(0)

    def on_key(event):
        global i
        if event.key == "p":
            i = max(0, i - 1)
            show_frame(i)
        elif event.key == "n":
            i = min(len(video) - 1, i + 1)
            show_frame(i)

    # connect to the event
    i = 0
    # add the argument i to the on_key function
    fig.canvas.mpl_connect("key_press_event", lambda event: on_key(event, i))

    plt.show()


if __name__ == "__main__":
    path = "tests/data/simple_video/sample1_vid1_sperm14_id17.mp4"
    video = vread(path)
    tracking_tool(video)
