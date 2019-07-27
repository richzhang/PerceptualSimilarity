import argparse
import os
import models
import numpy as np
from util import util

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-d','--dir', type=str, default='./imgs/ex_dir_pair')
parser.add_argument('-o','--out', type=str, default='./imgs/example_dists.txt')
parser.add_argument('--use_gpu', action='store_true', help='turn on flag to use GPU')
parser.add_argument('--model', type=str, default='net-lin')
parser.add_argument('--net', type=str, default='alex')
parser.add_argument('--no_normalize', action='store_true', help='dont normalize activations')
parser.add_argument('--dist', type=str, default='L2', help='distance function')

opt = parser.parse_args()

## Initializing the model
model = models.PerceptualLoss(model=opt.model, net=opt.net, use_gpu=opt.use_gpu, 
	dist=opt.dist, normalize=not opt.no_normalize)

# crawl directories
f = open(opt.out,'w')
files = os.listdir(opt.dir)

dists = []
for (ff,file0) in enumerate(files[:-1]):
	img0 = util.im2tensor(util.load_image(os.path.join(opt.dir,file0))) # RGB image from [-1,1]
	for (gg,file1) in enumerate(files[ff+1:]):
		img1 = util.im2tensor(util.load_image(os.path.join(opt.dir,file1)))

		# Compute distance
		dist01 = model.forward(img0,img1).item()
		dists.append(dist01)
		print('(%s, %s): %.3f'%(file0,file1,dist01))
		f.writelines('(%s, %s): %.3f'%(file0,file1,dist01))

dist_mean = np.mean(np.array(dists))
print('Mean: %.3f'%dist_mean)
f.writelines('Mean: %.3f'%dist_mean)

f.close()
