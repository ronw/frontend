import numpy as np

import scikits.audiolab as audiolab
from mock_sndfile import MockSndfile

__all__ = ['audio_source']

##########
# Sources
def audio_source(filename, start=0, end=None, nbuf=None):
    """ Returns a generator that returns sequential lists of nbuf samples

    
    arguments should be in second (or ms) units, not samples (as they
    are now)
    """
    if isinstance(filename, str):
        f = audiolab.Sndfile(filename)
    elif np.iterable(filename):
        f = MockSndfile(filename)
    else:
        raise ValueError, 'Invalid filename: %s' % filename

    if not end:
        end = f.nframes
    if not nbuf:
        nbuf = 10*f.samplerate

    pos = f.seek(start)
    nremaining = end - pos
    while nremaining > nbuf:
        yield f.read_frames(nbuf)
        nremaining -= nbuf

    if nremaining > 0:
        yield f.read_frames(nremaining)
    f.close()
