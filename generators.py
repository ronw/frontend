from __future__ import division

import functools
import itertools
import types

import numpy as np
import scipy as sp

import scikits.audiolab as audiolab
#from scikits.samplerate import resample

def _generator(func):
    """Turn a standalone function into a generator.

    Decorator that creates a generator that loops over incoming items
    and yields the result of calling func on each one.
    """
    @functools.wraps(func, assigned=('__name__', '__doc__'))
    def gen(lst, *args, **kwargs):
        for x in lst:
            yield func(x, *args, **kwargs)
    gen.__doc__ = 'Feature extraction generator: %s' % (gen.__doc__)
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

@_generator
def resample(frame, ratio=None, type='sinc_fastest', verbose=False):
    """Use scikits.samplerate

    For best results len(frame)*ratio should be an integer.  Its
    probably best to do this outside of the frontend
    """
    if ratio is None:
        return frame

    new_frame = samplerate.resample(frame, ratio, type, verbose)

# this is slower than the implementation below which does not use the
# @_generator decorator
#@_generator
#def tomono(frame):
#    return sum(frame) / len(frame)
#def tomono(frames):
#    for frame in frames:
#        yield sum(frame)/frame.shape[1]
@_generator
def mono(frame):
    return frame.mean(1)

@_generator
def preemphasize():  # or just filter()
    pass


# essentially a ring buffer! - works for matrices too... (really row features)
def framer(samples, nwin=512, nhop=None):
    """open arbitrary audio file
    
    arguments should be in second (or ms) units, not samples (as they
    are now)
    
    handles zero padding of final frames
    """
    if not type(samples) is types.GeneratorType:
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

# import a bunch of numpy functions directly:
external_funcs = {'abs': np.abs, 'log': np.log,
                  'real': np.real, 'imag': np.imag,
                  'diff': np.diff,
                  'fft': np.fft.fft, 'ifft': np.fft.ifft,
                  'rfft': np.fft.rfft, 'irfft': np.fft.irfft}
for local_name, func in external_funcs.iteritems():
    create_func = "%s = _generator(func)" % local_name
    exec(create_func)

# @_generator
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

# @_generator
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

    
# @_generator
# def log(frame):
#     return np.log(frame)

# @_generator
# def abs(frame):
#     return np.abs(frame)

# @_generator
# def real(frame):
#     return np.real(frame)

# @_generator
# def imag(frame):
#     return np.imag(frame)

@_generator
def dB(frame, minval=-100.0):
    spectrum = 20*np.log10(np.abs(frame))
    spectrum[spectrum < minval] = minval
    return spectrum


@_generator
def dct(frame):
    pass

@_generator
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

def mfcc(lifter=None):
    return dct(log(mel_filterbank(stft())))

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

def _hz2octs(freq, A440):
    return np.log2(freq / (A440 / 16.0))

@_generator
def pickpeaks(frame):
    # Keep only local maxes in freq
    #Dm = (D > D([1,[1:nr-1]],:)) & (D >= D([[2:nr],nr],:));
    lowidx  = np.concatenate(([0], range(len(frame) - 1)))
    highidx = np.concatenate((range(1, len(frame)), [len(frame) - 1]))
    localmax = np.logical_and(frame > frame[lowidx],
                              frame >= frame[highidx])
    return frame * localmax

def _create_constant_q_kernel(nbin, bins_per_octave, lowbin, highbin):
    """
    Based on Dan Ellis' logfmap.m
    """
    ratio = (highbin - 1.0) / highbin
    opr = int(round(np.log(lowbin / highbin) / np.log(ratio)))
    print opr
    ibin = lowbin * np.exp(np.arange(opr) * -np.log(ratio))

    cqmx = np.zeros((opr,nbin))

    eps = np.finfo(np.double).eps
    for i in range(opr):
        tt = np.pi * (np.arange(nbin) - ibin[i])
        cqmx[i] = (np.sin(tt) + eps) / (tt+eps)

    return cqmx

def constantq(fs, fmin, fmax, bins_per_octave=12):
    """

    Based on B. Blankertz, "The Constant Q Transform"
    http://ida.first.fhg.de/publications/drafts/Bla_constQ.pdf
    """

    Q = 1 / (2 ** (1.0 / bins_per_octave) - 1)

    K = np.ceil(bins_per_octave * np.log2(fmax / fmin))

    fftlen = 2 ** np.ceil(np.log2(np.ceil(Q * fs / fmin)))

    tempKernel = np.zeros((fftlen,1))
    sparKernel = []
    for k in xrange(K, 0, -1):
        ilen = np.ceil(Q * fs / (fmin * 2**(k / bpo)))

def _hz_to_mel(freq, htk=False):
    if htk:
        mel = 2595.0 * np.log10(1 + f / 700.0)
    else:
        f_0 = 0
        f_sp = 200.0 / 3
        brkfrq = 1000.0
        brkpt = (brkfrq - f_0) / f_sp
        logstep = np.exp(np.log(6.4) / 27)
        linpts = (f < brkfrq);
        mel

def melfb(fft_frames, samplerate=8000, nfilts=40, width=1.0,
                   minfreq=0, maxfreq=None, htkmel=0, constamp=0):
    if maxfreq is None:
        maxfreq = samplerate/2
    for frame in fft_frames:
        pass

def chromafb(nfft, nbin, samplerate, A440=440.0, ctroct=5.0, octwidth=0):
    """
    Based on dpwe's fft2chromamx.m

    Create a matrix to convert FFT to Chroma
    A440 is optional ref frq for A
    ctroct, octwidth specify a dominance window - Gaussian
    weighting centered on ctroct (in octs, re A0 = 27.5Hz) and 
    with a gaussian half-width of octwidth.  Defaults to
    halfwidth = inf i.e. flat.
    """
    wts = np.zeros((nbin, nfft))

    fftfrqbins = nbin * _hz2octs(np.arange(1, nfft, dtype='d') / nfft
                                 * samplerate, A440)

    # make up a value for the 0 Hz bin = 1.5 octaves below bin 1
    # (so chroma is 50% rotated from bin 1, and bin width is broad)
    fftfrqbins = np.concatenate(([fftfrqbins[0] - 1.5 * nbin], fftfrqbins))

    binwidthbins = np.concatenate(
        (np.maximum(fftfrqbins[1:] - fftfrqbins[:-1], 1.0), [1]))

    D = np.tile(fftfrqbins, (nbin,1))  \
        - np.tile(np.arange(0, nbin, dtype='d')[:,np.newaxis], (1,nfft))

    nbin2 = round(nbin / 2.0);

    # Project into range -nbins/2 .. nbins/2
    # add on fixed offset of 10*nbins to ensure all values passed to
    # rem are +ve
    D = np.remainder(D + nbin2 + 10*nbin, nbin) - nbin2;

    # Gaussian bumps - 2*D to make them narrower
    wts = np.exp(-0.5 * (2*D / np.tile(binwidthbins, (nbin,1)))**2)

    # normalize each column
    wts /= np.tile(np.sqrt(np.sum(wts**2, 0)), (nbin,1))

    # remove aliasing columns
    wts[:,nfft/2+1:] = 0;

    # Maybe apply scaling for fft bins
    if octwidth > 0:
        wts *= np.tile(
            np.exp(-0.5 * (((fftfrqbins/nbin - ctroct)/octwidth)**2)),
            (nbin, 1))

    return wts

def chroma(samples, fs, nfft, nwin=None, nhop=None, winfun=np.hamming, nchroma=12, center=1000, sd=1):
    A0 = 27.5  # Hz
    A440 = 440 # Hz
    f_ctr_log = np.log2(center/A0)
    CM = chromafb(nfft/2+1, nchroma, fs, A440, f_ctr_log, sd)

    return filterbank(pickpeaks(
        abs(stft(samples, nfft, nwin, nhop, winfun))), CM)

