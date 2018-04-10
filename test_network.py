import sys; sys.path += ['models']
import torch
from util import util
from models import dist_model as dm
from IPython import embed
import numpy

use_gpu = True          # Whether to use GPU
spatial = False         # Whether to return a spatial map of distance of size height x width

## Initializing the model
model = dm.DistModel()

# Linearly calibrated models
#model.initialize(model='net-lin',net='squeeze',use_gpu=use_gpu,spatial=spatial)
model.initialize(model='net-lin',net='alex',use_gpu=use_gpu,spatial=spatial)
#model.initialize(model='net-lin',net='vgg',use_gpu=use_gpu,spatial=spatial)

# Off-the-shelf uncalibrated networks
#model.initialize(model='net',net='squeeze',use_gpu=use_gpu)
#model.initialize(model='net',net='alex',use_gpu=use_gpu)
#model.initialize(model='net',net='vgg',use_gpu=use_gpu)

# Low-level metrics
# model.initialize(model='l2',colorspace='Lab')
# model.initialize(model='ssim',colorspace='RGB')
print('Model [%s] initialized'%model.name())

## Example usage with dummy tensors
dummy_im0 = torch.Tensor(1,3,64,64) # image should be RGB, normalized to [-1,1]
dummy_im1 = torch.Tensor(1,3,64,64)
dist = model.forward(dummy_im0,dummy_im1)

## Example usage with images
ex_ref = util.im2tensor(util.load_image('./imgs/ex_ref.png'))
ex_p0 = util.im2tensor(util.load_image('./imgs/ex_p0.png'))
ex_p1 = util.im2tensor(util.load_image('./imgs/ex_p1.png'))
ex_d0 = model.forward(ex_ref,ex_p0)
ex_d1 = model.forward(ex_ref,ex_p1)
if not spatial:
    print('Distances: (%.6f, %.6f)'%(ex_d0, ex_d1))
else:
    print('Distances: (%.6f, %.6f)'%(ex_d0.mean(),ex_d1.mean()))            # The mean distance is the same as the non-spatial distance
    
    # Visualize a spatially-varying distance map
    import pylab
    pylab.imshow(ex_d0)
    pylab.show()

