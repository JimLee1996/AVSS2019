import os
import csv


class AverageMeter(object):
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.counter = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.counter += n
        self.avg = self.sum / self.counter


class Log(object):
    def __init__(self, path, keys):
        if os.path.exists(path):
            os.remove(path)
        self.file = open(path, 'w', newline='')
        self.writer = csv.writer(self.file, delimiter='\t')

        self.keys = keys
        self.writer.writerow(self.keys)

    def __del__(self):
        self.file.close()

    def log(self, values):
        v = []
        for key in self.keys:
            v.append(values[key])

        self.writer.writerow(v)
        self.file.flush()
