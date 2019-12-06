import random


def crop(frames, start, size, stride):
    # todo more efficient
    # padding by loop
    while start + (size - 1) * stride > len(frames) - 1:
        frames *= 2
    return frames[start:start + (size - 1) * stride + 1:stride]


class BeginCrop(object):
    def __init__(self, size, stride=1):
        self.size = size
        self.stride = stride

    def __call__(self, frames):
        return crop(frames, 0, self.size, self.stride)


class CenterCrop(object):
    def __init__(self, size, stride=1):
        self.size = size
        self.stride = stride

    def __call__(self, frames):
        start = max(0, len(frames) // 2 - self.size * self.stride // 2)
        return crop(frames, start, self.size, self.stride)


class RandomCrop(object):
    def __init__(self, size, stride=1):
        self.size = size
        self.stride = stride

    def __call__(self, frames):
        start = random.randint(
            0, max(0,
                   len(frames) - 1 - (self.size - 1) * self.stride)
        )
        return crop(frames, start, self.size, self.stride)
