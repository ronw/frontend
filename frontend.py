# Sketching out a feature extraction frontend API based on
# generator/coroutine pipeline.  Like the common ASR frontend pipeline
# (e.g. sphinx)
#
# see: 
# http://www.dabeaz.com/generators-uk/
# http://www.dabeaz.com/coroutines/index.html

import audioop
import functools
import itertools
import wave

import magic
import numpy
import scipy

import scikits.audiolab as audiolab

from mock_sndfile import MockSndFile

def generator(func):
    """Turn a simple function into a generator.

    Decorator that created a generator that loops over incoming items
    and yields the result of calling func on each one.
    """
    @functools.wraps(func)
    def gen(lst, *args, **kwargs):
        for x in lst:
            yield func(x, *args, **kwargs)
    return gen

# Plaigiarized from David Beazley's PyCon 2009 talk:
# http://www.dabeaz.com/coroutines/index.html
def coroutine(func):
    def start(*args,**kwargs):
        cr = func(*args,**kwargs)
        cr.next()
        return cr
    return start

@coroutine
def broadcast(targets):
    """
    Broadcast a stream onto multiple targets
    """
    while True:
        item = (yield)
        for target in targets:
            target.send(item)

# This stuff is almost certainly broken
# @coroutine
# def coroutine_to_generator():
#     while True:
#         item = (yield)
#         yield item
# def generator_to_coroutine():
    

def generator_to_afunc(func):
    """Turn a pipeline generator into a normal function that
    accumulates into an array.

    Decorator that created a generator that loops over incoming items
    and yields the result of calling func on each one.
    """
    @functools.wraps(func)
    def gen(lst, *args, **kwargs):
        return numpy.array([func(x, *args, **kwargs) for x in lst])
    return gen    
    

def toarray(*args, **kwargs):
    pass



# Based on pipe() from
# http://paddy3118.blogspot.com/2009/05/pipe-fitting-with-python-generators.html
def pipeline(*cmds):
    """ String a set of generators together to form a pipeline.
    
    pipeline(a,b,c,d, ...) -> yield from ...d(c(b(a())))
    """
    gen = cmds[0]
    for cmd in cmds[1:]:
        gen = cmd(gen)
    for x in gen:
        yield x


# source generators
def framer(filename, *args, **kwargs):
    """open arbitrary audio file"""
    if not isinstance(filename, str):
        filename = MockSndFile(filename)
    return audio_file_framer(arg, *args, **kwargs)

def read_audio_file(filename, *args, **kwargs):
    ms = magic.open(magic.MAGIC_NONE)
    ms.load()
    filetype = ms.file(filename)
    if filetype:
        if 'WAVE' in filetype:
            return read_wav(filename, *args, **kwargs)
    print 'Unsupported file type: %s' % filetype
    return None

def read_audio_file_audiolab(filename, nwin=512, nhop=None, start=0, end=None):
    """
    arguments should be in second (or ms) units, not samples (as they
    are now)

    handles zero padding of final frames
    """
    if isinstance(filename, str):
        f = audiolab.Sndfile(filename)
    else:
        f = filename
    if not end:
        end = f.nframes
    if not nhop:
        nhop = nwin/4
    # nhop cannot be less than 1 for normal behavior
    noverlap = nwin - nhop

    pos = f.seek(start)
    nremaining = end - pos
    ninitial_read = min(nremaining, noverlap)
    frames = numpy.empty((nwin, f.channels))
    frames[:ninitial_read] = f.read_frames(ninitial_read)
    nremaining -= ninitial_read

    while nremaining > nhop:
        frames[noverlap:] = f.read_frames(nhop)
        nremaining -= nhop
        yield frames
        frames[:noverlap] = frames[nhop:]

    # Read remaining few samples from file and yield the remaining
    # zero padded frames.
    if nremaining > 0:
        frames[noverlap:noverlap + nremaining] = f.read_frames(nremaining)
        frames[noverlap + nremaining:] = 0
        nremaining_frames = (noverlap + nremaining) / nhop
    else:
        nremaining_frames = (nwin + nhop) / nhop 
    print nremaining_frames
    for n in range(nremaining_frames):
        frames[:noverlap] = frames[nhop:]
        frames[noverlap:] = 0

    f.close()

def read_wav_wave(filename, start=0, end=None, nwin=512):
    f = wave.open(filename)
    nchan, framesize, framerate, nframes, comptype, compname = f.getparams()
    if not end:
        end = nframes

    f.setpos(start)
    while True:
        pos = f.tell()
        if pos + nwin / framesize > end:
            buffer_size = max(0, end - pos)
        frames = f.readframes(nwin)
        if frames == '':
            break
        samples = [audioop.getsample(frames, framesize, x) * 1.0 / 2**(8*framesize - 1)
                   for x in xrange(len(frames)/framesize)]
        for sample in itertools.izip(samples[0::2], samples[1::2]):
            yield sample
    f.close()

# this is slower than the implementation below which does not use the
# @generator decorator
#@generator
#def tomono(frame):
#    return sum(frame) / len(frame)
#def tomono(frames):
#    for frame in frames:
#        yield sum(frame)/frame.shape[1]
@generator
def tomono(frame):
    return frame.mean(1)

def resample():
    """Use scikits.samplerate"""
    pass

def preemphasize():  # or just filter()
    pass

# Window functions
def rectangular(N):
    return numpy.ones(N)

def hamming(N):
    return 0.54 - 0.46 * numpy.cos(2 * numpy.pi * numpy.arange(N) / (N - 1))

def hann(N):
    return 0.5 * ( 1 - numpy.cos(2 * numpy.pi * numpy.arange(N) / (N - 1)))

def window_old(samples, nwin, nhop, winfun=hamming):
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

def window(frames, winfun=hamming):
    frame = frames.next()
    win = winfun(len(frame))
    yield win * frame
    for frame in frames:
        yield win * frame

@generator
def fft(frame, nfft=None):
    if not nfft:
        nfft = len(frame)
    tmp = numpy.fft.fft(frame, nfft)
    return tmp[:nfft/2+1]

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

@generator
def dB(frame, minval=-100):
    spectrum = 20*numpy.log10(numpy.abs(frame))
    spectrum[spectrum < minval] = minval
    return spectrum
@generator
def dct(frame):
    pass
# compound feature extractors:

def stft(samples, nfft, winfun=hamming):
    return fft(window(samples, winfun), nfft)

def logspec(samples, nfft, winfun=hamming):
    return dB(stft(samples, nfft, winfun))


def mfcc(lifter=None):
    dct(log(mel_filterbank(stft())))

@generator
def delta(n=1):
    pass
@generator
def mfcc_d_a():
    s = stack(3)
    mfcc(broadcast(nop(s), delta(broadcast(nop(s), delta(s)))))

@generator
def stack(num_incoming):
    pass
@generator
def chroma():
    pass

# sinks
@generator
def tolist():
    pass
@generator
def tohtkfile():
    # probably depends on tolist
    pass



if __name__ == '__main__':
    def api_samples():
        # example_pipeline_syntax (following dabeaz):
        read_audio_file(filename, mfcc(lifter=25), tohtkfile(output_filename))
        # but it has ugly parens (looks like lisp!)
        # want to have something like this instead:
        create_pipeline(read_audio_file(filename), mfcc(lifter=25), tohtkfile(output_filename))
        # but we all of our sinks optionally output something (e.g. tolist)
        feats = create_pipeline(read_audio_file(filename), mfcc(lifter=25), tolist)
        # would be even better if tolist is done by default...
