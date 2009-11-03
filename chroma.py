import numpy as np

from dataprocessor import DataProcessor, Pipeline
import basic

def _hz2octs(freq, A440):
    return np.log2(freq / (A440 / 16.0))

class PickPeaks(DataProcessor):
    def process_frame(self, frame):
        # Keep only local maxes in freq
        #Dm = (D > D([1,[1:nr-1]],:)) & (D >= D([[2:nr],nr],:));
        lowidx  = np.concatenate(([0], range(len(frame) - 1)))
        highidx = np.concatenate((range(1, len(frame)), [len(frame) - 1]))
        localmax = np.logical_and(frame > frame[lowidx],
                                  frame >= frame[highidx])
        return frame * localmax

def constantqfb(fs, nfft, fmin, fmax, bpo=12):
    """
    
    Based on B. Blankertz, "The Constant Q Transform"
    http://ida.first.fhg.de/publications/drafts/Bla_constQ.pdf
    """
    Q = 1 / (2 ** (1.0 / bpo) - 1)

    #nfft = 2 ** np.ceil(np.log2(np.ceil(Q * fs / fmin)))
    # Compute minimum fmin from nfft
    if fmin < Q * fs / nfft:
        fmin = Q * fs / nfft
        print 'fmin too small for nfft, increasing to %.2f' % fmin
        #log.warning('fmin too small for nfft, increasing to %d', fmin)

    K = np.ceil(bpo * np.log2(float(fmax) / fmin))
    
    tempkernel = np.zeros(nfft)
    kernel = np.zeros((K, nfft / 2 + 1), dtype=np.complex)
    for k in np.arange(K-1, -1, -1, dtype=np.float):
        ilen = np.ceil(Q * fs / (fmin * 2.0**(k / bpo)))
        if ilen % 2 == 0:
            # calculate offsets so that kernels are centered in the
            # nfftgth windows
            start = nfft / 2 - ilen / 2
        else:
            start = nfft / 2 - (ilen + 1) / 2        

        tempkernel[:] = 0
        tempkernel[start:start+ilen] = (np.hamming(ilen) / ilen
                                        * np.exp(2 * np.pi * 1j * Q
                                                 * np.r_[:ilen] / ilen))
        kernel[k] = np.fft.rfft(tempkernel)
    return kernel / nfft

class ConstantQToChroma(DataProcessor):
    def __init__(self, bpo=12):
        self.bpo = bpo
        
    def process_frame(self, cqframe):
        hpcp = np.zeros(self.bpo)
        for n in xrange(0, len(cqframe), self.bpo):
            cqoct = cqframe[n:n+self.bpo]
            hpcp[:len(cqoct)] += cqoct
        return hpcp

def CQChroma(fs, nfft, nwin=None, nhop=None, winfun=np.hamming,
             nchroma=12, fmin=55.0, fmax=587.36):
    CQ = constantqfb(fs, nfft, fmin, fmax, nchroma)
    return Pipeline(basic.STFT(nfft, nwin, nhop, winfun),
                    basic.Filterbank(CQ),
                    basic.Abs(),
                    ConstantQToChroma(nchroma))

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

    # Maybe apply scaling for fft bins
    if octwidth > 0:
        wts *= np.tile(
            np.exp(-0.5 * (((fftfrqbins/nbin - ctroct)/octwidth)**2)),
            (nbin, 1))

    # remove aliasing columns
    return wts[:,:nfft/2+1]

def Chroma(fs, nfft, nwin=None, nhop=None, winfun=np.hamming, nchroma=12,
           center=1000, sd=1):
    A0 = 27.5  # Hz
    A440 = 440 # Hz
    f_ctr_log = np.log2(center/A0)
    CM = chromafb(nfft, nchroma, fs, A440, f_ctr_log, sd)

    return Pipeline(basic.STFT(nfft, nwin, nhop, winfun),
                    basic.Abs(),
                    PickPeaks(),
                    basic.Filterbank(CM))

class CircularShift(DataProcessor):
    def __init__(self, nshift=0):
        self.nshift = nshift
        
    def process_frame(self, frame):
        return frame[np.r_[self.nshift:len(frame), :self.nshift]]

