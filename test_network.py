import argparse
import torch
from util import util
import models
from models import dist_model as dm
from IPython import embed

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-p0','--path0', type=str, default='./imgs/ex_ref.png')
parser.add_argument('-p1','--path1', type=str, default='./imgs/ex_p0.png')
parser.add_argument('--use_gpu', action='store_true', help='turn on flag to use GPU')
parser.add_argument('--model', type=str, default='net-lin')
parser.add_argument('--net', type=str, default='alex')
parser.add_argument('--spatial', action='store_true', help='outputs spatial map the same size of input images')
parser.add_argument('--spatial_out', type=str, default='./spatial_out.png')
opt = parser.parse_args()

model = models.PerceptualLoss(model=opt.model, net=opt.net, use_gpu=opt.use_gpu, spatial=opt.spatial)

img0 = util.im2tensor(util.load_image(opt.path0))
img1 = util.im2tensor(util.load_image(opt.path1))
dist = model.forward(img0, img1)

if not opt.spatial:
    print('Distances: %.3f'%(dist))
else:
    print('Distances: %.3f'%(dist.mean())) # The mean distance is approximately the same as the non-spatial distance
    
    # Visualize a spatially-varying distance map between ex_p0 and ex_ref
    import pylab
    pylab.imshow(dist[0,0,...].data.cpu().numpy())
    pylab.savefig(opt.spatial_out)
    pylab.show()

