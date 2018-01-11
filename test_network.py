import torch
from util import util
from models import dist_model as dm
from IPython import embed

## Initializing the model
model = dm.DistModel()

# Linearly calibrated models
# model.initialize(model='net-lin',net='squeeze',use_gpu=True)
model.initialize(model='net-lin',net='alex',use_gpu=True)
# model.initialize(model='net-lin',net='vgg',use_gpu=True)

# Off-the-shelf uncalibrated networks
# model.initialize(model='net',net='squeeze',use_gpu=True)
# model.initialize(model='net',net='alex',use_gpu=True)
# model.initialize(model='net',net='vgg',use_gpu=True)

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
ex_d0 = model.forward(ex_ref,ex_p0)[0]
ex_d1 = model.forward(ex_ref,ex_p1)[0]
print('Distances: (%.3f, %.3f)'%(ex_d0,ex_d1))
