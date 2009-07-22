import numpy as np
import scipy as sp

import basic

def _hz2octs(freq, A440):
    return np.log2(freq / (A440 / 16.0))

@decorators.generator
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

    return basic.filterbank(pickpeaks(
        basic.abs(basic.stft(samples, nfft, nwin, nhop, winfun))), CM)

@decorators.generator
def circularshift(frame, nshift=0):
    return frame[np.r_[nshift:len(frame), :nshift]]

