import torch
from util import util
import models
from models import dist_model as dm
from IPython import embed

use_gpu = False         # Whether to use GPU
spatial = True         # Return a spatial map of perceptual distance.

# Linearly calibrated models (LPIPS)
model = models.PerceptualLoss(model='net-lin', net='alex', use_gpu=use_gpu, spatial=spatial)
	# Can also set net = 'squeeze' or 'vgg'

# Off-the-shelf uncalibrated networks
# model = models.PerceptualLoss(model='net', net='alex', use_gpu=use_gpu, spatial=spatial)
	# Can also set net = 'squeeze' or 'vgg'

# Low-level metrics
# model = models.PerceptualLoss(model='L2', colorspace='Lab', use_gpu=use_gpu)
# model = models.PerceptualLoss(model='ssim', colorspace='RGB', use_gpu=use_gpu)

## Example usage with dummy tensors
dummy_im0 = torch.zeros(1,3,64,64) # image should be RGB, normalized to [-1,1]
dummy_im1 = torch.zeros(1,3,64,64)
if(use_gpu):
	dummy_im0 = dummy_im0.cuda()
	dummy_im1 = dummy_im1.cuda()
dist = model.forward(dummy_im0,dummy_im1)

## Example usage with images
ex_ref = util.im2tensor(util.load_image('./imgs/ex_ref.png'))
ex_p0 = util.im2tensor(util.load_image('./imgs/ex_p0.png'))
ex_p1 = util.im2tensor(util.load_image('./imgs/ex_p1.png'))
if(use_gpu):
	ex_ref = ex_ref.cuda()
	ex_p0 = ex_p0.cuda()
	ex_p1 = ex_p1.cuda()

ex_d0 = model.forward(ex_ref,ex_p0)
ex_d1 = model.forward(ex_ref,ex_p1)

if not spatial:
    print('Distances: (%.3f, %.3f)'%(ex_d0, ex_d1))
else:
    print('Distances: (%.3f, %.3f)'%(ex_d0.mean(), ex_d1.mean()))            # The mean distance is approximately the same as the non-spatial distance
    
    # Visualize a spatially-varying distance map between ex_p0 and ex_ref
    import pylab
    pylab.imshow(ex_d0[0,0,...].data.cpu().numpy())
    pylab.show()
