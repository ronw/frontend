import functools

import numpy
import scipy

import scikits.audiolab as audiolab

from mock_sndfile import MockSndfile
import windows

def _generator(func):
    """Turn a standalone function into a generator.

    Decorator that creates a generator that loops over incoming items
    and yields the result of calling func on each one.
    """
    @functools.wraps(func)
    def gen(lst, *args, **kwargs):
        for x in lst:
            yield func(x, *args, **kwargs)
    return gen

# # Plaigiarized from David Beazley's PyCon 2009 talk:
# # http://www.dabeaz.com/coroutines/index.html
# def _coroutine(func):
#     def start(*args,**kwargs):
#         cr = func(*args,**kwargs)
#         cr.next()
#         return cr
#     return start

# @coroutine
# def _broadcast(targets):
#     """
#     Broadcast a stream onto multiple targets
#     """
#     while True:
#         item = (yield)
#         for target in targets:
#             target.send(item)

# This stuff is almost certainly broken
# @coroutine
# def coroutine_to_generator():
#     while True:
#         item = (yield)
#         yield item
# def generator_to_coroutine():

# source
def audio_source(filename, nbuf=441000, start=0, end=None):
    """open arbitrary audio file
    
    arguments should be in second (or ms) units, not samples (as they
    are now)
    """
    if isinstance(filename, str):
        f = audiolab.Sndfile(filename)
    else:
        f = MockSndfile(filename)

    if not end:
        end = f.nframes

    pos = f.seek(start)
    nremaining = end - pos
    while nremaining > nbuf:
        yield f.read_frames(nbuf)
        nremaining -= nbuf

    if nremaining > 0:
        yield f.read_frames(nremaining)
    f.close()
        
# this is slower than the implementation below which does not use the

def framer(samples, nwin=512, nhop=None):
    """open arbitrary audio file
    
    arguments should be in second (or ms) units, not samples (as they
    are now)
    
    handles zero padding of final frames
    """
    if not nhop:
        nhop = nwin/4
    # nhop cannot be less than 1 for normal behavior
    noverlap = nwin - nhop

    buf = samples.next().copy()
    while len(buf) < nwin:
        buf = numpy.concatenate((buf, samples.next().copy()))
      
    frame = buf[:nwin]
    buf = buf[nwin:]

    while True:
        yield frame
        frame[:noverlap] = frame[nhop:]

        try:
            while len(buf) < nhop:
                buf = numpy.concatenate((buf, samples.next().copy()))
        except StopIteration:
            break

        frame[noverlap:] = buf[:nhop]
        buf = buf[nhop:]

    # Read remaining few samples from file and yield the remaining
    # zero padded frames.
    frame[noverlap:noverlap + len(buf)] = buf
    frame[noverlap + len(buf):] = 0
    nremaining_frames = int(numpy.ceil((1.0*noverlap + len(buf)) / nhop))

    for n in range(nremaining_frames):
        yield frame
        frame[:noverlap] = frame[nhop:]
        frame[noverlap:] = 0

# this is slower than the implementation below which does not use the
# @_generator decorator
#@_generator
#def tomono(frame):
#    return sum(frame) / len(frame)
#def tomono(frames):
#    for frame in frames:
#        yield sum(frame)/frame.shape[1]
@_generator
def tomono(frame):
    return frame.mean(1)

@_generator
def resample():
    """Use scikits.samplerate"""
    pass

@_generator
def preemphasize():  # or just filter()
    pass

def __window_old(samples, nwin, nhop, winfun=windows.hamming):
    buf = numpy.empty(nwin)
    win = winfun(nwin)
    noverlap = nwin - nhop
    start = 0
    while True:
        for n in xrange(start, nwin):
            buf[n] = samples.next()
        # zero pad the last frame if necessary
        if n < nwin:
            for n in xrange(n, nwin):
                buf[n] = 0.0
        yield win * buf
        buf[:noverlap] = buf[nhop:] 
        start = noverlap

def window(frames, winfun=windows.hamming):
    frame = frames.next()
    win = winfun(len(frame))
    yield win * frame
    for frame in frames:
        yield win * frame

@_generator
def fft(frame, nfft=None):
    if not nfft:
        nfft = len(frame)
    tmp = numpy.fft.fft(frame, nfft)
    return tmp[:nfft/2+1]

@_generator
def log(frame):
    return numpy.log(frame)

@_generator
def dB(frame, minval=-100.0):
    spectrum = 20*numpy.log10(numpy.abs(frame))
    spectrum[spectrum < minval] = minval
    return spectrum

@_generator
def dct(frame):
    pass

def _hz_to_mel(freq, htk=False):
    if htk:
        mel = 2595.0 * numpy.log10(1 + f / 700.0)
    else:
        f_0 = 0
        f_sp = 200.0 / 3
        brkfrq = 1000.0
        brkpt = (brkfrq - f_0) / f_sp
        logstep = numpy.exp(numpy.log(6.4) / 27)
        linpts = (f < brkfrq);
        mel

def mel_filterbank(fft_frames, samplerate=8000, nfilts=40, width=1.0,
                   minfreq=0, maxfreq=None, htkmel=0, constamp=0):
    if maxfreq is None:
        maxfreq = samplerate/2
    for frame in fft_frames:
        pass

# compound feature extractors:

def stft(samples, nfft, nwin=None, nhop=None, winfun=windows.hamming):
    if not nwin:
        nwin = nfft
    return fft(window(framer(samples, nwin, nhop), winfun), nfft)

def logspec(samples, nfft, nwin=None, nhop=None, winfun=windows.hamming):
    return dB(stft(samples, nwin, nhop, nfft, winfun))

def mfcc(lifter=None):
    dct(log(mel_filterbank(stft())))

@_generator
def delta(n=1):
    pass

@_generator
def mfcc_d_a():
    s = stack(3)
    mfcc(broadcast(nop(s), delta(broadcast(nop(s), delta(s)))))

@_generator
def stack(num_incoming):
    pass

@_generator
def chroma():
    pass


