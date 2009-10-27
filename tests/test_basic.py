#!/usr/bin/env python

import unittest

import numpy as np
from numpy.testing import *

from dataprocessor import Pipeline
import basic

class TestAudioSource(unittest.TestCase):
    def test_audio_source(self):
        pass

class TestFramer(unittest.TestCase):
    def _test_framer(self, n, nbuf, nwin, nhop):
        samples = np.arange(n)
        frames = Pipeline(basic.AudioSource(samples, nbuf=nbuf),
                          basic.Framer(nwin, nhop))

        nframes = int(np.ceil(1.0 * len(samples) / nhop))
        for x in range(nframes):
            curr_frame = frames.next()
            
            if x * nhop + nwin < len(samples):
                curr_samples = samples[x * nhop : x * nhop + nwin].copy()
            else:
                # Make sure zero padding is correct.
                curr_samples = samples[x*nhop:].copy()
                nsamples = len(curr_samples)
                curr_samples = np.concatenate((curr_samples,
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
        self._test_framer(500, 101, 23, 7)


class TestSimpleComponents(unittest.TestCase):
    def test_tomono(self):
        nsamp = 1000
        nbuf = 10
        frames = np.random.rand(nsamp, 2)

        gen = audio_source(frames, nbuf=nbuf)
        stereo_frames = np.array([x for x in gen])

        gen = tomono(audio_source(frames, nbuf=nbuf))
        mono_frames = np.array([x for x in gen])

        self.assert_(stereo_frames.shape != mono_frames.shape)
        self.assertEqual(mono_frames.shape, (nsamp/nbuf, nbuf))

    def test_fft_ifft(self):
        nsamp = 1024
        nfft = 32;
        for nbuf in [8, 16, 32]:
            samples = np.random.rand(nsamp)
            frames = np.array([x for x in audio_source(samples, nbuf=nbuf)])

            gen = ifft(fft(audio_source(samples, nbuf=nbuf), nfft), nfft)
            test_frames = np.array([x for x in gen])

            assert_array_almost_equal(frames, test_frames[:,:nbuf])

    def test_delta(self):
        pass
    def test_stft(self):
        pass
    def test_mfcc(self):
        pass
    def test_mfcc_d_a(self):
        pass



if __name__ == '__main__':
    unittest.main()

