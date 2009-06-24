#!/usr/bin/env python

"""Tests for frontend feature extraction pipeline."""


import numpy

from numpy.testing import *

import frontend

class TestMockSndfile(NumpyTestCase):
    def test_channels(self):
        samples = numpy.arange(200, dtype=numpy.float32).reshape((100,2))
        sndfile = frontend.MockSndfile(samples)
        self.assertEqual(sndfile.channels, 2)

        samples = numpy.arange(200, dtype=numpy.float32)
        sndfile = frontend.MockSndfile(samples)
        self.assertEqual(sndfile.channels, 1)

    def test_read_frames(self):
        samples = numpy.arange(200).reshape((100,2))
        samples = (samples  - 100) / 200
        sndfile = frontend.MockSndfile(samples)

        frames = sndfile.read_frames(10)
        assert_array_equal(samples[:10,:], frames)
        frames = sndfile.read_frames(20)
        assert_array_equal(samples[10:30,:], frames)
        frames = sndfile.read_frames(70)
        assert_array_equal(samples[30:,:], frames)

    def test_read_frames_type_conversion(self):
        samples = numpy.arange(200, dtype=numpy.float32)
        sndfile = frontend.MockSndfile(samples)
        frames = sndfile.read_frames(len(samples), dtype=numpy.int16)
        self.assert_(isinstance(frames[0], numpy.int16))
        assert_array_equal(numpy.int16(samples), frames)

    def test_read_frames_past_end(self):
        samples = numpy.arange(200)
        sndfile = frontend.MockSndfile(samples)
        self.assertRaises(RuntimeError, sndfile.read_frames, len(samples) + 1)
    
    def test_seek(self):
        # Based on TestSeek class from audiolab's test_sndfile.py.
        samples = numpy.arange(10000)
        sndfile = frontend.MockSndfile(samples)
        nframes = sndfile.nframes

        bufsize = 1024

        buf = sndfile.read_frames(bufsize)
        sndfile.seek(0)
        buf2 = sndfile.read_frames(bufsize)
        assert_array_equal(buf, buf2)

        # Now, read some frames, go back, and compare buffers
        # (check whence == 1 == SEEK_CUR)
        sndfile = frontend.MockSndfile(samples)
        sndfile.read_frames(bufsize)
        buf = sndfile.read_frames(bufsize)
        sndfile.seek(-bufsize, 1)
        buf2 = sndfile.read_frames(bufsize)
        assert_array_equal(buf, buf2)

        # Now, read some frames, go back, and compare buffers
        # (check whence == 2 == SEEK_END)
        sndfile = frontend.MockSndfile(samples)
        buf = sndfile.read_frames(nframes)
        sndfile.seek(-bufsize, 2)
        buf2 = sndfile.read_frames(bufsize)
        assert_array_equal(buf[-bufsize:], buf2)

        # Try to seek past the end.
        self.assertRaises(IOError, sndfile.seek, len(samples) + 1)


class TestAudioInput(NumpyTestCase):
    def test_read_sndfile(self):
        pass

class TestSimpleComponents(NumpyTestCase):
    def test_delta(self):
        pass

    def test_tomono(self):
        frames = numpy.random.rand((10,2), numpy.int16)

        mono_frames = numpy.array([x for x in frontend.tomono(frames)])

    def test_dB(self):
        pass
    
class TestCompoundPipelines(NumpyTestCase):
    samples = numpy.sin(numpy.arange(-10, 10, 0.01) * 2 * numpy.pi)
    
    def test_pipeline(self):
        pipe = frontend.pipeline(audio_source(samples),
                                 framer(nwin, nhop),
                                 window(hamming),
                                 fft(nfft))

    def test_stft(self):
        pass
    def test_mfcc(self):
        pass
    def test_mfcc_d_a(self):
        pass



if __name__ == '__main__':
    NumpyTest('frontend').testall()

