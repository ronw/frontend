import functools

def generator(func):
    """Turn a standalone function into a generator.

    Decorator that creates a generator that loops over incoming items
    and yields the result of calling func on each one.
    """
    @functools.wraps(func, assigned=('__name__', '__doc__'))
    def gen(lst, *args, **kwargs):
        for x in lst:
            yield func(x, *args, **kwargs)
    gen.__doc__ = 'Feature extraction generator: %s' % (gen.__doc__)
    return gen


# # Plaigiarized from David Beazley's PyCon 2009 talk:
# # http://www.dabeaz.com/coroutines/index.html
# def coroutine(func):
#     def start(*args,**kwargs):
#         cr = func(*args,**kwargs)
#         cr.next()
#         return cr
#     return start

# @coroutine
# def broadcast(targets):
#     """
#     Broadcast a stream onto multiple targets
#     """
#     while True:
#         item = (yield)
#         for target in targets:
#             target.send(item)

# This stuff is almost certainly broken
# @coroutine
# def coroutine_to_generator():
#     while True:
#         item = (yield)
#         yield item
# def generator_to_coroutine():

