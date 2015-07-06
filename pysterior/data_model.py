import math
import random
import numpy as np

class PosteriorSample(object):
    def __init__(self):
        self.samples = []
        self.average = None

    def get_samples(self):
        return self.samples

    def add_sample(self, new_sample):
        self.samples.append(new_sample)
        self._update_average(new_sample)

    def _update_average(self, new_sample):
        if self.average is None:
            self.average = new_sample
        else:
            self.average = self.average + ((new_sample - self.average)/len(self.samples))

    def get_mean(self):
        return self.average

    def get_median(self):
        #TODO: Make this an online calculation
        center = (len(self.samples)-1)/2
        left, right = math.floor(center), math.ceil(center)
        sorted_data = sorted(self.samples)
        return (sorted_data[left] + sorted_data[right])/2.0

    def get_random(self, n=1):
        return np.array([random.choice(self.samples) for i in range(n)])
