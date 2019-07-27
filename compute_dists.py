import argparse
import models
from util import util

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-p0','--path0', type=str, default='./imgs/ex_ref.png')
parser.add_argument('-p1','--path1', type=str, default='./imgs/ex_p0.png')
parser.add_argument('--use_gpu', action='store_true', help='turn on flag to use GPU')
parser.add_argument('--model', type=str, default='net-lin')
parser.add_argument('--net', type=str, default='alex')

opt = parser.parse_args()

## Initializing the model
model = models.PerceptualLoss(model=opt.model, net=opt.net, use_gpu=opt.use_gpu, 
	dist=opt.dist, normalize=not opt.no_normalize)

# Load images
img0 = util.im2tensor(util.load_image(opt.path0)) # RGB image from [-1,1]
img1 = util.im2tensor(util.load_image(opt.path1))

# Compute distance
dist01 = model.forward(img0,img1)
print('Distance: %.3f'%dist01)
