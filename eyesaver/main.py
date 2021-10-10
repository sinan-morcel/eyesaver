import numpy as np
import cv2
from mss import mss
from PIL import Image
from screeninfo import get_monitors
from skimage.metrics import structural_similarity as compare_ssim
from time import sleep
from beepy import beep
import typer

app = typer.Typer()


@app.command()
def start(sensitivity: float = 0.998, post_beep_sleep: float = 1.0, inter_frame_sleep: float = 0.5):
    """
    Monitors your screen for small changes while you give your eyes a rest.
    A reasonable value for SENSITIVITY would be 0.98, but make your own tests.
    Small text changes with blue backgrounds require a sensitivity of 0.997.
    """
    # monitor info: https://stackoverflow.com/a/31171430
    monitors = get_monitors()
    if not monitors:
        typer.echo("Error, at least 1 monitor is needed")
        exit(1)

    if len(monitors) < 1:
        typer.echo("Error, at least 1 monitor is needed")
        exit(1)

    if len(monitors) > 1:
        typer.echo("Error, only supports one monitor")
        exit(1)

    monitor = monitors[0]

    # typer.echo(f"monitor: {monitor}")

    # general loop: https://stackoverflow.com/a/54246290
    bounding_box = {'top': 0, 'left': 0,
                    'width': monitor.width, 'height': monitor.height}

    sct = mss()
    try:
        typer.echo("Screen monitoring in progress...")
        mat = np.array(sct.grab(bounding_box))
        prev_image = cv2.cvtColor(
            np.array(sct.grab(bounding_box)), cv2.COLOR_BGR2GRAY)
        while True:
            curr_image = cv2.cvtColor(
                np.array(sct.grab(bounding_box)), cv2.COLOR_BGR2GRAY)

            # Image difference: https://www.pyimagesearch.com/2017/06/19/image-difference-with-opencv-and-python/
            (score, diff) = compare_ssim(curr_image, prev_image, full=True)
            # diff = (diff * 255).astype("uint8")
            # print("SSIM: {}".format(score))

            prev_image = curr_image.copy()

            if score < sensitivity:
                # https://pypi.org/project/beepy/
                beep(sound='coin')
                sleep(post_beep_sleep)
            else:
                sleep(inter_frame_sleep)

    except (KeyboardInterrupt, EOFError):
        print("\nGoodbye!")
