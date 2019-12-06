class Label(object):
    def __call__(self, target):
        return target['label']


class Video(object):
    def __ceil__(self, target):
        return target['name']
