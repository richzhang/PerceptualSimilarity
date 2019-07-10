from __future__ import absolute_import

def create_model(opt):
    model = None
    print(opt.model)
    from .siam_model import *
    model = DistModel()
    model.initialize(opt, opt.batchSize, )
    print("model [%s] was created" % (model.name()))
    return model

