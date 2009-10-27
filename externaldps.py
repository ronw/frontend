import numpy as np
import scipy as sp

from dataprocessor import DataProcessor

# Import a bunch of numpy functions directly.
external_funcs = {'Abs': np.abs,
                  'Diff': np.diff,
                  'FFT': np.fft.fft, 'IFFT': np.fft.ifft,
                  'Flatten': lambda x: x.flatten(),
                  'RFFT': np.fft.rfft, 'IRFFT': np.fft.irfft,
                  'Real': np.real, 'Imag': np.imag,
                  'Square': np.square, 'Sqrt': np.sqrt,
                  }
                  
__all__ = external_funcs.keys()

# FIXME - figure out how to properly bind func into ExternalDP class
for local_name, func in external_funcs.iteritems():
    class ExternalDP(DataProcessor):
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

        def _call_external_func(self, frame, external_func=func):
            return external_func(frame, *self.args, **self.kwargs)

        def process_frame(self, frame):
            return self._call_external_func(frame)

    exec("ExternalDP.__name__ = '%s'" % local_name)
    exec("%s = ExternalDP" % local_name)
