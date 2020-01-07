import os
import sys

from utils import sec_to_hms
from backend import Backend
from videoloader import VideoLoader, frames_to_tensor


def main():
    video_path = sys.argv[1]

    if not os.path.exists(video_path):
        return

    be = Backend()
    vl = VideoLoader(video_path)

    fps = vl.fps

    # loop through the whole video
    # todo use threading to handle it when having gpus
    while True:
        frames = vl.get_frames()

        # vl.get_frames
        if frames is None:
            return

        x = frames_to_tensor(frames)
        y = be.predict(x)

        for label in y:
            if label == 'violent':
                time = vl.pos / fps
                h, m, s = sec_to_hms(time)
                print('violent scene at time:\n%d:%d:%d' % (h, m, s))


if __name__ == '__main__':
    main()