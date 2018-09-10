import argparse
import os
from IPython import embed
from util import util
import models.dist_model as dm

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dir0', type=str, default='./imgs/ex_dir0')
parser.add_argument('--dir1', type=str, default='./imgs/ex_dir1')
parser.add_argument('--out', type=str, default='./imgs/example_dists.txt')
parser.add_argument('--use_gpu', action='store_true', help='turn on flag to use GPU')
opt = parser.parse_args()

## Initializing the model
model = dm.DistModel()
model.initialize(model='net-lin',net='alex',use_gpu=opt.use_gpu)

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
