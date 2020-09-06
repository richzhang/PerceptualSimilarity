import argparse
import os
import lpips
import numpy as np

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-d','--dir', type=str, default='./imgs/ex_dir_pair')
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
files = os.listdir(opt.dir)
if(opt.N is not None):
	files = files[:opt.N]
F = len(files)

dists = []
for (ff,file) in enumerate(files[:-1]):
	img0 = lpips.im2tensor(lpips.load_image(os.path.join(opt.dir,file))) # RGB image from [-1,1]
	if(opt.use_gpu):
		img0 = img0.cuda()

	if(opt.all_pairs):
		files1 = files[ff+1:]
	else:
		files1 = [files[ff+1],]

	for file1 in files1:
		img1 = lpips.im2tensor(lpips.load_image(os.path.join(opt.dir,file1)))

		if(opt.use_gpu):
			img1 = img1.cuda()

		# Compute distance
		dist01 = loss_fn.forward(img0,img1)
		print('(%s,%s): %.3f'%(file,file1,dist01))
		f.writelines('(%s,%s): %.6f\n'%(file,file1,dist01))

		dists.append(dist01.item())

avg_dist = np.mean(np.array(dists))
stderr_dist = np.std(np.array(dists))/np.sqrt(len(dists))

print('Avg: %.5f +/- %.5f'%(avg_dist,stderr_dist))
f.writelines('Avg: %.6f +/- %.6f'%(avg_dist,stderr_dist))

f.close()
