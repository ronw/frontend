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
import magic
import numpy
import wave

def generator(func):
    @functools.wraps(func)
    def gen(lst, *args, **kwargs):
        for x in lst:
            yield func(x, *args, **kwargs)
    return gen

def pipeline(*args):
  pass

# source generators
def read_audio_file(filename, *args, **kwargs):
    """ open arbitrarty audio file"""
    ms = magic.open(magic.MAGIC_NONE)
    ms.load()
    filetype = ms.file(filename)
    if filetype:
        if 'WAVE' in filetype:
            return read_wav(filename, *args, **kwargs)
    print 'Unsupported file type: %s' % filetype
    return None

def read_wav(filename, start=0, end=None, buffer_size=5000):
    f = wave.open(filename)
    nchan, framesize, framerate, nframes, comptype, compname = f.getparams()
    if not end:
        end = nframes

    f.setpos(start)
    while True:
        pos = f.tell()
        if pos + buffer_size/framesize > end:
            buffer_size = max(0, end - pos)
        frames = f.readframes(buffer_size)
        if frames == '':
            break
        samples = [audioop.getsample(frames, framesize, x) * 1.0 / (2**(8*framesize - 1) - 1)
                   for x in xrange(len(frames)/framesize)]
        for sample in itertools.izip(samples[0::2], samples[1::2]):
            yield sample
    f.close()

def read_mp3():
    pass

@generator
def tomono(frame):
    return sum(frame)/len(frame)

@generator 
def resample():
    pass

@generator
def preemphasize():  # or just filter()
    pass

def rectangular(N):
    return numpy.ones(N)

def hamming(N):
    return 0.54 - 0.46 * numpy.cos(2 * numpy.pi * numpy.arange(N) / (N - 1))

def hann(N):
    return 0.5 * ( 1 - numpy.cos(2 * numpy.pi * numpy.arange(N) / (N - 1)))

def window(samples, nwin, nhop, winfun=hamming):
    buffer = []
    win = winfun(nwin)
    while True:
        if buffer:
            buffer = buffer[nhop:] 
        buffer_has_new_samples = False
        for samp in samples:
            buffer.append(samp)
            buffer_has_new_samples = True
            if len(buffer) == nwin:
                break
        if not buffer_has_new_samples:
            break
        # zero pad if necessary
        if len(buffer) < nwin:
            for n in range(nwin - len(buffer)):
                buffer.append(0)
        yield win * numpy.array(buffer)

@generator
def fft(frame, nfft=None):
    if not nfft:
        nfft = len(frame)
    tmp = numpy.fft.fft(frame)
    return tmp[:nfft/2+1]

@generator
def mel_filterbank(fft_frame):
    pass
@generator
def dB(frame):
    spectrum = 20*numpy.log10(numpy.abs(frame))
    spectrum[spectrum < -100] = -100
    return spectrum
@generator
def dct(frame):
    pass
# compound feature extractors:

def stft(samples, nfft, nwin, nhop, winfun=hamming):
    return fft(window(samples, nwin, nhop, winfun), nfft)

def logspec(samples, nfft, nwin, nhop, winfun=hamming):
    return dB(stft(samples, nfft, nwin, nhop, winfun))

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
