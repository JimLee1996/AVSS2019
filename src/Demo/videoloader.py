import os
import numpy as np
import torch
import cv2
from PIL import Image


class VideoLoader:
    # can handle big video files
    def __init__(self, path):
        if os.path.exists(path):
            self.cap = cv2.VideoCapture(path)
        else:
            raise FileNotFoundError
        self.pos = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)

    def get_frames(self, n_frames=16, sample_rate=1):
        # todo more effecient
        frames = []
        for i in range(n_frames * sample_rate):
            ret, frame = self.cap.read()
            if not ret:
                return None
            if i % sample_rate == 0:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
        self.pos = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
        return np.array(frames)

    def __del__(self):
        self.cap.release()


def resize(frames, size=(112, 112), scale=1.0):

    _, h, w, _ = frames.shape

    # spatial: center crop
    crop = int(min(h, w) * scale)
    h1 = h // 2 - crop // 2
    h2 = h1 + crop
    w1 = w // 2 - crop // 2
    w2 = w1 + crop
    frames = frames[:, h1:h2, w1:w2, :]

    # todo more efficient with array not img
    clip = [
        Image.fromarray(frame).resize(size, Image.BILINEAR) for frame in frames
    ]

    return clip


def transform(pic):
    # to Tensor
    img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
    nchannel = len(pic.mode)
    img = img.view(pic.size[1], pic.size[0], nchannel)
    # put it from HWC to CHW format
    # yikes, this transpose takes 80% of the loading time/CPU
    img = img.transpose(0, 1).transpose(0, 2).contiguous()
    if isinstance(img, torch.ByteTensor):
        img = img.float().div(255)

    for t, m, s in zip(img, [0, 0, 0], [1, 1, 1]):
        t.sub_(m).div_(s)
    return img


def frames_to_tensor(frames):
    # resize frames and let them be tensors
    clip = resize(frames)
    clip = [transform(frame) for frame in clip]
    return torch.stack(clip).permute(1, 0, 2, 3)


if __name__ == '__main__':

    from backend import Backend
    # init model
    be = Backend()

    # load video
    vl = VideoLoader('video.mp4')

    batch = []
    frames = vl.get_frames()
    x = frames_to_tensor(frames)
    batch.append(x)

    # next 16 frames
    frames = vl.get_frames()
    x = frames_to_tensor(frames)
    batch.append(x)

    x = torch.stack(batch)
    y = be.predict(x)

    print(y)
