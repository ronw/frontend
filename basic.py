from __future__ import division

import types

import numpy as np
import scipy as sp

import scikits.samplerate as samplerate

import decorators

# import a bunch of numpy functions directly:
external_funcs = {'abs': np.abs, 'log': np.log,
                  'real': np.real, 'imag': np.imag,
                  'diff': np.diff, 'flatten': lambda x: x.flatten(),
                  'fft': np.fft.fft, 'ifft': np.fft.ifft,
                  'rfft': np.fft.rfft, 'irfft': np.fft.irfft}
for local_name, func in external_funcs.iteritems():
    defun = "%s = decorators.generator(func)" % local_name
    exec(defun)

@decorators.generator
def resample(frame, ratio=None, type='sinc_fastest', verbose=False):
    """Use scikits.samplerate

    For best results len(frame)*ratio should be an integer.  Its
    probably best to do this outside of the frontend
    """
    if ratio is None:
        return frame
    return samplerate.resample(frame, ratio, type, verbose)

@decorators.generator
def normalize(frame, ord=None):
    """Normalize each frame using a norm of the given order"""
    return frame / np.linalg.norm(frame, ord)

# this is slower than the implementation below which does not use the
# @decorators.generator decorator
#@decorators.generator
#def tomono(frame):
#    return sum(frame) / len(frame)
#def tomono(frames):
#    for frame in frames:
#        yield sum(frame)/frame.shape[1]
@decorators.generator
def mono(frame):
    if frame.ndim > 1:
        mono_frame =  frame.mean(1)
    else:
        mono_frame = frame
    return mono_frame

@decorators.generator
def preemphasize():  # or just filter()
    pass


# essentially a simple buffer - works for matrices too... (really row features)
def framer(samples, nwin=512, nhop=None):
    """open arbitrary audio file
    
    arguments should be in second (or ms) units, not samples (as they
    are now)
    
    handles zero padding of final frames
    """
    if not issubclass(samples.__class__, types.GeneratorType):
        samples = (x for x in [samples])
    
    if not nhop:
        nhop = nwin
    # nhop cannot be less than 1 for normal behavior
    noverlap = nwin - nhop

    buf = samples.next().copy()
    while len(buf) < nwin:
        buf = np.concatenate((buf, samples.next().copy()))
      
    frame = buf[:nwin]
    buf = buf[nwin:]

    while True:
        yield frame
        frame[:noverlap] = frame[nhop:]

        try:
            while len(buf) < nhop:
                buf = np.concatenate((buf, samples.next().copy()))
        except StopIteration:
            break

        frame[noverlap:] = buf[:nhop]
        buf = buf[nhop:]

    # Read remaining few samples from file and yield the remaining
    # zero padded frames.
    frame[noverlap:noverlap + len(buf)] = buf
    frame[noverlap + len(buf):] = 0
    nremaining_frames = int(np.ceil((1.0*noverlap + len(buf)) / nhop))

    for n in range(nremaining_frames):
        yield frame
        frame[:noverlap] = frame[nhop:]
        frame[noverlap:] = 0

def overlap_add(frames, nwin=512, nhop=None):
    """Perform overlap-add resynthesis of frames

    Inverse of window()
    """
    if not nhop:
        nhop = nwin
    # nhop cannot be less than 1 for normal behavior
    noverlap = nwin - nhop

    buf = np.zeros(nwin)
    for frame in frames:
        buf += frame
        yield buf[:nhop].copy()
        buf[:noverlap] = buf[nhop:]
        buf[noverlap:] = 0

def window(frames, winfun=np.hanning):
    win = None
    for frame in frames:
        if win is None:
            win = winfun(len(frame))
        yield win * frame

# @decorators.generator
# def fft(frame, nfft=None, type='full'):
#     """ Calculates the FFT of frame

#     If type == 'full' returns the full fft including positive and
#     negative frequencies.  If 'real' only include nonnegative
#     frequencies.
#     """
#     if not nfft:
#         nfft = len(frame)
#     if type == 'full':
#         tmp = np.fft.fft(frame, nfft)
#     elif type == 'real':
#         tmp = np.fft.rfft(frame, nfft)
#     else:
#         raise ValueError, "type must be 'full' or 'real'"
#     return tmp

# @decorators.generator
# def ifft(frame, nfft=None, type='full'):
#     """ Calculates the inverse FFT of frame

#     type should match that of the fft used to create frame.  If
#     type == 'full' frame must contain 
#     """
#     if not nfft:
#         nfft = len(frame)
#     if type == 'full':
#         tmp = np.fft.ifft(frame, nfft)
#     elif type == 'real':
#         tmp = np.fft.irfft(frame, nfft)
#     else:
#         raise ValueError, "type must be 'full' or 'real'"
#     return tmp

    
# @decorators.generator
# def log(frame):
#     return np.log(frame)

# @decorators.generator
# def abs(frame):
#     return np.abs(frame)

# @decorators.generator
# def real(frame):
#     return np.real(frame)

# @decorators.generator
# def imag(frame):
#     return np.imag(frame)

@decorators.generator
def dB(frame, minval=-100.0):
    spectrum = 20*np.log10(np.abs(frame))
    spectrum[spectrum < minval] = minval
    return spectrum

@decorators.generator
def filterbank(fft_frame, fb):
    return np.dot(fb, fft_frame)


# compound feature extractors:

def stft(samples, nfft, nwin=None, nhop=None, winfun=np.hanning):
    if not nwin:
        nwin = nfft
    return rfft(window(framer(samples, nwin, nhop), winfun), nfft)

def istft(S, nfft, nwin=None, nhop=None, winfun=np.hanning):
    if not nwin:
        nwin = nfft
    return overlap_add(window(irfft(S, nfft), winfun), nwin, nhop)

def logspec(samples, nfft, nwin=None, nhop=None, winfun=np.hanning):
    return dB(stft(samples, nfft, nwin, nhop, winfun))


