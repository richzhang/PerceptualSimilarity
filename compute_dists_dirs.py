import argparse
import os
import models
from util import util

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-d0','--dir0', type=str, default='./imgs/ex_dir0')
parser.add_argument('-d1','--dir1', type=str, default='./imgs/ex_dir1')
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
files = os.listdir(opt.dir0)

for file in files:
	if(os.path.exists(os.path.join(opt.dir1,file))):
		# Load images
		img0 = util.im2tensor(util.load_image(os.path.join(opt.dir0,file))) # RGB image from [-1,1]
		img1 = util.im2tensor(util.load_image(os.path.join(opt.dir1,file)))

		# Compute distance
		dist01 = model.forward(img0,img1)
		print('%s: %.3f'%(file,dist01))
		f.writelines('%s: %.6f\n'%(file,dist01))

f.close()
