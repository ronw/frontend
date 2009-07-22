import numpy as np
import scipy as sp

import decorators

@decorators.generator
def delta(n=1):
    pass

@decorators.generator
def stack(num_incoming):
    pass

@decorators.generator
def dct(frame):
    pass

def mfcc(lifter=None):
    return dct(log(mel_filterbank(stft())))

@decorators.generator
def mfcc_d_a():
    s = stack(3)
    mfcc(broadcast(nop(s), delta(broadcast(nop(s), delta(s)))))

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
