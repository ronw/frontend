#! /usr/bin/env python

"""
frontend: Audio feature extraction API

Makes extensive use of generator pipelines.
Construct pipelines of feature extractors.

The most common usage of this module will be through the pipeline()
function (or one of the predefined piplelines) which is used to string
together a set of generators to create compound feature extractors as
follows:

  feat = pipeline(framer('path/to/file', nwin=100),
                  tomono(),
                  preemphasize(),
                  window(hamming),
                  fft(),
                  abs())
  
Many feature extraction function have an additional form which allows
it to be called on its own.  For example, the fun() generator might
also have a standalone form, standalone.fun().

Copyright (C) 2009 Ron J. Weiss (ronweiss@gmail.com)

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

__author__ = "Ron J. Weiss <ronweiss@gmail.com>"
__version__ = "0.1"


# requirement: Each pipeline generator must create consistent outputs
# - the dimensionality cannot change mid-stream.  Generators do not
# need to output something for every input it receives (i.e. could
# have a VAD generator that only passes through frames it thinks
# contain speech)

#from version import version as _version
#__version__ = _version

from frontend import *
from windows import *

__all__ = filter(lambda s: not s.startswith('_'), dir())

