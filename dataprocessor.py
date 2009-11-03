import abc

import numpy as np

class DataProcessor(object):
    # Subclasses need to implement one of the following to methods
    # (the default implementation of the other will handle the other
    # case).
    def process_frame(self, frame):
        return self.process_sequence([frame]).next()

    def process_sequence(self, frames):
        for x in frames:
            yield self.process_frame(x)


class Source(object):
    # Sources only need to implement the iterator protocol
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __iter__(self):
        pass
    
    def toarray(self):
        return np.asarray([x for x in self])


class Pipeline(DataProcessor):
    def __init__(self, *dps):
        self.dps = dps

    def __iter__(self):
        return self.process_sequence()

    def process_sequence(self, frames=None):
        if frames is None:
            gen = self.dps[0]
            if issubclass(gen.__class__, Source):
                gen = iter(gen)
        else:
            gen = self.dps[0].process_sequence(frames)

        for dp in self.dps[1:]:
            gen = dp.process_sequence(gen)

        return gen

    def toarray(self, frames=None):
        return np.asarray([x for x in self.process_sequence(frames)])
