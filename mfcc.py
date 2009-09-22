import numpy as np
import scipy as sp

import basic
import decorators

@decorators.generator
def delta(n=1):
    pass

@decorators.generator
def stack(num_incoming):
    pass

def dct(frames, ndct):
    """Compute DCT (type 3)

    Eventually this will come from scipy (ver 0.8).
    """
    frame = frames.next()
    nrow = len(frame)

    DCT = np.empty((ndct, nrow))
    for i in xrange(ndct):
        DCT[i,:] = (np.cos(i*np.arange(1, 2 * nrow, 2) / (2.0 * nrow) * np.pi)
                    * np.sqrt(2.0 / nrow))
    yield np.dot(DCT, frame)

    for frame in frames:
        yield np.dot(DCT, frame)
        

def mfcc(samples, samplerate, nfft, nwin=None, nhop=None, winfun=np.hamming,
         nmel=40, width=1.0, fmin=0, fmax=None, ndct=13):
    return dct(basic.log(melspec(samples, samplerate, nfft, nwin, nhop, 
                                 winfun, nmel, width, fmin, fmax)),  ndct)

def melspec(samples, samplerate, nfft, nwin=None, nhop=None, winfun=np.hamming,
         nmel=40, width=1.0, fmin=0, fmax=None):
    FB = melfb(samplerate, nfft, nmel, width, fmin, fmax) 
    return basic.filterbank(basic.powspec(samples, nfft, nwin, nhop, winfun),
                            FB)

@decorators.generator
def mfcc_d_a():
    s = stack(3)
    mfcc(broadcast(nop(s), delta(broadcast(nop(s), delta(s)))))

def _hz_to_mel(f):
    return 2595.0 * np.log10(1 + f / 700.0)

def _mel_to_hz(z):
    return 700.0 * (10.0**(z / 2595.0) - 1.0)

def melfb(samplerate, nfft, nfilts=40, width=1.0, fmin=0, fmax=None):
    if fmax is None:
        fmax = samplerate / 2

    wts = np.zeros((nfilts, nfft / 2 + 1))
    # Center freqs of each FFT bin
    fftfreqs = np.arange(nfft / 2 + 1, dtype=np.double) / nfft * samplerate

    # 'Center freqs' of mel bands - uniformly spaced between limits
    minmel = _hz_to_mel(fmin)
    maxmel = _hz_to_mel(fmax)
    binfreqs = _mel_to_hz(minmel
                          + np.arange((nfilts+2), dtype=np.double) / (nfilts+1)
                          * (maxmel - minmel))

    for i in xrange(nfilts):
        freqs = binfreqs[i + np.arange(3)]
        # scale by width
        freqs = freqs[1] + width * (freqs - freqs[1])
        # lower and upper slopes for all bins
        loslope = (fftfreqs - freqs[0]) / (freqs[1] - freqs[0])
        hislope = (freqs[2] - fftfreqs) / (freqs[2] - freqs[1])
        # .. then intersect them with each other and zero
        wts[i,:] = np.maximum(0, np.minimum(loslope, hislope))

    # Slaney-style mel is scaled to be approx constant E per channel
    #enorm = 2.0 / (binfreqs[2:nfilts+2] - binfreqs[:nfilts])
    #wts = np.dot(np.diag(enorm), wts)
    
    return wts
