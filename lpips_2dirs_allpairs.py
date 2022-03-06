import argparse
import os
import lpips
import numpy as np

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-d0','--dir0', type=str, default='./imgs/ex_dir0')
parser.add_argument('-d1','--dir1', type=str, default='./imgs/ex_dir1')
parser.add_argument('-o','--out', type=str, default='./imgs/example_dists.txt')
parser.add_argument('-v','--version', type=str, default='0.1')
parser.add_argument('--all-pairs', action='store_true', help='turn on to test all N(N-1)/2 pairs, leave off to just do consecutive pairs (N-1)')
parser.add_argument('-N', type=int, default=None)
parser.add_argument('--use_gpu', action='store_true', help='turn on flag to use GPU')

opt = parser.parse_args()

## Initializing the model
loss_fn = lpips.LPIPS(net='alex',version=opt.version)
if(opt.use_gpu):
	loss_fn.cuda()

# crawl directories
f = open(opt.out,'w')
files0 = os.listdir(opt.dir0)
files1 = os.listdir(opt.dir1)

dists = []
for file0 in files0:
	for file1 in files1:
		img0 = lpips.im2tensor(lpips.load_image(os.path.join(opt.dir0,file0))) # RGB image from [-1,1]
		if(opt.use_gpu):
			img0 = img0.cuda()
		img1 = lpips.im2tensor(lpips.load_image(os.path.join(opt.dir1,file1)))
		if(opt.use_gpu):
			img1 = img1.cuda()

		# Compute distance
		dist = loss_fn.forward(img0,img1)
		print('(%s,%s): %.3f'%(file0,file1,dist))
		f.writelines('(%s,%s): %.6f\n'%(file0,file1,dist))

		dists.append(dist.item())

avg_dist = np.mean(np.array(dists))
stderr_dist = np.std(np.array(dists))/np.sqrt(len(dists))

print('Avg: %.5f +/- %.5f'%(avg_dist,stderr_dist))
f.writelines('Avg: %.6f +/- %.6f'%(avg_dist,stderr_dist))

f.close()
