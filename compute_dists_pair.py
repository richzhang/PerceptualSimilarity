import argparse
import os
import models
import numpy as np
from util import util

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-d','--dir', type=str, default='./imgs/ex_dir_pair')
parser.add_argument('-o','--out', type=str, default='./imgs/example_dists.txt')
parser.add_argument('--use_gpu', action='store_true', help='turn on flag to use GPU')

opt = parser.parse_args()

## Initializing the model
model = models.PerceptualLoss(model='net-lin',net='alex',use_gpu=opt.use_gpu)

# crawl directories
f = open(opt.out,'w')
files = os.listdir(opt.dir)

dists = []
for (ff,file0) in enumerate(files[:-1]):
	img0 = util.im2tensor(util.load_image(os.path.join(opt.dir,file0))) # RGB image from [-1,1]
	if(opt.use_gpu):
		img0 = img0.cuda()

	for (gg,file1) in enumerate(files[ff+1:]):
		img1 = util.im2tensor(util.load_image(os.path.join(opt.dir,file1)))
		if(opt.use_gpu):
			img1 = img1.cuda()

		# Compute distance
		dist01 = model.forward(img0,img1).item()
		dists.append(dist01)
		print('(%s, %s): %.3f'%(file0,file1,dist01))
		f.writelines('(%s, %s): %.3f'%(file0,file1,dist01))

dist_mean = np.mean(np.array(dists))
print('Mean: %.3f'%dist_mean)
f.writelines('Mean: %.3f'%dist_mean)

f.close()
