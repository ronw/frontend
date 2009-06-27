# Sketching out a feature extraction frontend API based on
# generator/coroutine pipeline.  Like the common ASR frontend pipeline
# (e.g. sphinx)
#
# see: 
# http://www.dabeaz.com/generators-uk/
# http://www.dabeaz.com/coroutines/index.html

#import audioop
import functools

import numpy

__all__ = ['pipeline', 'toarray', 'tohtkfile']

# Based on pipe() from
# http://paddy3118.blogspot.com/2009/05/pipe-fitting-with-python-generators.html
def pipeline(*cmds):
    """ String a set of generators together to form a pipeline.
    
    pipeline(a,b,c,d, ...) -> yield from ...d(c(b(a())))
    """
    print cmds
    gen = cmds[0]
    for cmd in cmds[1:]:
        gen = cmd(gen)
    for x in gen:
        yield x

# Add constructors for all pipeline generators to this module's namespace.
def _constructor(gen):
    """Create a constructor for a generator for use by pipeline().

    Decorated gen with a constructor function.  The constructor
    curries gen's arguments and returns a generator whose only
    argument is the input list/generator to iterate over (i.e. the
    output of the previous stage in the pipeline).

    This allows the arguments of pipeline() to be specified without
    needing lambda.
    """
    @functools.wraps(gen)
    def f(*args, **kwargs):
        print gen.__name__, kwargs
        return lambda lst: gen(lst, *args, **kwargs)
    return f

# Decorate generators using _constructor()
import generators
generators_names = [name for name in dir(generators)
                    if (callable(getattr(generators, name))
                        and not name.startswith('_'))]
for name in generators_names:
    func = getattr(generators, name)
    setattr(generators, name, _constructor(func))
from generators import *
__all__.extend(generators_names)

# Sinks
def toarray(frames):
    return numpy.array([x for x in frames])

def tohtkfile():
    # probably depends on toarray
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

