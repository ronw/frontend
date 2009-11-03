from __future__ import division

import types

import numpy as np
import scipy as sp

import scikits.samplerate as samplerate

from dataprocessor import DataProcessor, Source, Pipeline
from externaldps import *

class Resample(DataProcessor):
    """Use scikits.samplerate

    For best results len(frame)*ratio should be an integer.  Its
    probably best to do this outside of the frontend
    """
    def __init__(self, ratio=None, type='sinc_fastest', verbose=False):
        self.ratio = ratio
        self.type = type
        self.verbose = verbose

    def process_frame(self, frame):
        if self.ratio is None:
            return frame
        else:
            return samplerate.resample(frame, self.ratio, self.type,
                                       self.verbose)


class Normalize(DataProcessor):
    """Normalize each frame using a norm of the given order"""
    def __init__(self, ord=None):
        self.ord = ord

    def process_frame(self, frame):
        return frame / (np.linalg.norm(frame, self.ord) + 1e-16)


class Mono(DataProcessor):
    def process_frame(self, frame):
        if frame.ndim > 1:
            mono_frame =  frame.mean(1)
        else:
            mono_frame = frame
        return mono_frame


class Preemphasize(DataProcessor):  # or just filter()
    pass


# essentially a simple buffer - works for matrices too... (really row features)
class Framer(DataProcessor):
    """open arbitrary audio file
    
    arguments should be in second (or ms) units, not samples (as they
    are now)
    
    handles zero padding of final frames
    """
    def __init__(self, nwin, nhop=None):
        self.nwin = nwin
        if nhop is None:
            nhop = nwin
        self.nhop = nhop

    def process_sequence(self, samples):
        # Is samples a list instead of a generator?
        if not 'next' in dir(samples):
            samples = (x for x in [samples])
    
        # nhop cannot be less than 1 for normal behavior
        noverlap = self.nwin - self.nhop

        buf = samples.next().copy()
        while len(buf) < self.nwin:
            buf = np.concatenate((buf, samples.next()))
      
        frame = buf[:self.nwin]
        buf = buf[self.nwin:]

        while True:
            yield frame.copy()
            frame[:noverlap] = frame[self.nhop:]
            
            try:
                while len(buf) < self.nhop:
                    buf = np.concatenate((buf, samples.next()))
            except StopIteration:
                break
    
            frame[noverlap:] = buf[:self.nhop]
            buf = buf[self.nhop:]

        # Read remaining few samples from file and yield the remaining
        # zero padded frames.
        frame[noverlap:noverlap + len(buf)] = buf
        frame[noverlap + len(buf):] = 0
        nremaining_frames = int(np.ceil((1.0*noverlap + len(buf)) / self.nhop))

        for n in xrange(nremaining_frames):
            yield frame.copy()
            frame[:noverlap] = frame[self.nhop:]
            frame[noverlap:] = 0


class OverlapAdd(DataProcessor):
    """Perform overlap-add resynthesis

    Inverse of Framer()
    """

    def __init__(self, nwin=512, nhop=None):
        self.nwin = nwin
        if nhop is None:
            nhop = nwin
        self.nhop = nhop

    def process_sequence(self, frames):
        # nhop cannot be less than 1 for normal behavior
        noverlap = self.nwin - self.nhop

        # off by one error somewhere here
        buf = np.zeros(self.nwin)
        for frame in frames:
            buf += frame
            yield buf[:self.nhop].copy()
            buf[:noverlap] = buf[self.nhop:]
            buf[noverlap:] = 0

        nremaining_frames = int(noverlap / self.nhop) - 1
        for n in range(nremaining_frames):
            yield buf[:self.nhop].copy()
            buf[:noverlap] = buf[self.nhop:]


class Window(DataProcessor):
    def __init__(self, winfun=np.hanning):
        self.winfun = winfun

    def process_sequence(self, frames):
        win = None
        for frame in frames:
            if win is None:
                win = self.winfun(len(frame))
            yield win * frame


class RMS(DataProcessor):
    def process_frame(self, frame):
        return 20*np.log10(np.sqrt(np.mean(frame**2)))


class DB(DataProcessor):
    def __init__(self, minval=-100.0):
        self.minval = minval

    def process_frame(self, frame):
        spectrum = 20*np.log10(np.abs(frame))
        spectrum[spectrum < self.minval] = self.minval
        return spectrum


class IDB(DataProcessor):
    def process_frame(self, frame):
        return 10.0 ** (frame / 20)


class Filterbank(DataProcessor):
    def __init__(self, fb):
        self.fb = fb
        
    def process_frame(self, frame):
        return np.dot(self.fb, frame)


class Log(DataProcessor):
    def __init__(self, floor=-5.0):
        self.floor = floor

    def process_frame(self, frame):
        return np.maximum(np.log(frame), self.floor)


# compound feature extractors:

def STFT(nfft, nwin=None, nhop=None, winfun=np.hanning):
    if nwin is None:
        nwin = nfft
    return Pipeline(Framer(nwin, nhop), Window(winfun), RFFT(nfft))

def ISTFT(nfft, nwin=None, nhop=None, winfun=np.hanning):
    if nwin is None:
        nwin = nfft
    return Pipeline(IRFFT(nfft), Window(winfun), OverlapAdd(nwin, nhop))

def LogSpec(nfft, nwin=None, nhop=None, winfun=np.hanning):
    return Pipeline(STFT(nfft, nwin, nhop, winfun), DB())

def PowSpec(nfft, nwin=None, nhop=None, winfun=np.hanning):
    return Pipeline(STFT(nfft, nwin, nhop, winfun), Abs(), Square())
