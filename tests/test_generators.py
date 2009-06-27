#!/usr/bin/env python

"""Tests for feature extraction generators."""

import unittest

import numpy

from numpy.testing import *

from generators import *

class TestAudioFile(unittest.TestCase):
    def test_audio_source(self):
        pass

class TestFramer(unittest.TestCase):
    def _test_framer(self, n, nbuf, nwin, nhop):
        samples = numpy.arange(n)
        frames = framer(audio_source(samples, nbuf), nwin, nhop)

        nframes = int(numpy.ceil(1.0 * len(samples) / nhop))
        for x in range(nframes):
            curr_frame = frames.next()
            
            if x * nhop + nwin < len(samples):
                curr_samples = samples[x * nhop : x * nhop + nwin].copy()
            else:
                # Make sure zero padding is correct.
                curr_samples = samples[x*nhop:].copy()
                nsamples = len(curr_samples)
                curr_samples = numpy.concatenate((curr_samples,
                                                  [0] * (nwin - nsamples)))

            assert_array_equal(curr_frame, curr_samples)
        self.assertRaises(StopIteration, frames.next)

    def test_framer_small_buf_1hop(self):
        self._test_framer(90, 5, 20, 1)
    def test_framer_large_buf_1hop(self):
        self._test_framer(90, 50, 20, 1)
    def test_framer_quarter_hop(self):
        self._test_framer(90, 50, 20, 5)
    def test_framer_half_hop(self):
        self._test_framer(90, 50, 20, 10)
    def test_framer_full_hop(self):
        self._test_framer(90, 50, 20, 20)
    def test_framer_relatively_prime(self):
        self._test_framer(500, 51, 23, 7)


class TestSimpleComponents(unittest.TestCase):
    def test_delta(self):
        pass

    def test_tomono(self):
        frames = numpy.random.rand(100, 2)

        mono_frames = [x for x in tomono(audio_source(frames, 10))]

    def test_dB(self):
        pass
    
    def test_stft(self):
        pass
    def test_mfcc(self):
        pass
    def test_mfcc_d_a(self):
        pass



if __name__ == '__main__':
    unittest.main()

