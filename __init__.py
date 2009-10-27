import inspect

from dataprocessor import *
from sources import *
from basic import *
from chroma import *
from mfcc import *
dps = locals().copy()


# All DataProcessors will also be available as standalone functions.
#
# Naming convention:
# CamelCase for classes/generators to be used in a Pipeline lowercase
# for standalone functions.

def _dataprocessor_to_function(dpcls):
    def fun(frames, *args, **kwargs):
        dp = dpcls(*args, **kwargs)
        return np.asarray([x for x in dp.__iter__(frames)])
    return fun

def _source_to_function(dpcls):
    def fun(*args, **kwargs):
        dp = dpcls(*args, **kwargs)
        return np.asarray([x for x in dp])
    return fun

__all__ = []
for clsname, cls in dps.iteritems():
    if ((not clsname[0].isupper())
        or (not inspect.isclass(cls) and not inspect.isfunction(cls))):
        continue

    funname = clsname.lower()
    if issubclass(cls, Source):
        exec('%s = _source_to_function(%s)' % (funname, clsname))
    else:
        exec('%s = _dataprocessor_to_function(%s)' % (funname, clsname))

    __all__.append(clsname)
    __all__.append(funname)
