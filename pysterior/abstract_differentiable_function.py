
class AbstractDifferentiableFunction(object):
    def eval(self, *args, **kwargs):
        raise NotImplementedError

    def gradient(self, *args, **kwargs):
        raise NotImplementedError