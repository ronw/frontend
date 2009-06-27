#!/usr/bin/env python

"""Tests for frontend feature extraction pipeline."""

import numpy

from numpy.testing import *

import frontend

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

