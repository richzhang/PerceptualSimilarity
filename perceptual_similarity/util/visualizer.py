import numpy as np
import os
import time
from . import util
from . import html
# from pdb import set_trace as st
import matplotlib.pyplot as plt
import math
# from IPython import embed

def zoom_to_res(img,res=256,order=0,axis=0):
    # img   3xXxX
    from scipy.ndimage import zoom
    zoom_factor = res/img.shape[1]
    if(axis==0):
        return zoom(img,[1,zoom_factor,zoom_factor],order=order)
    elif(axis==2):
        return zoom(img,[zoom_factor,zoom_factor,1],order=order)

class Visualizer():
    def __init__(self, opt):
        # self.opt = opt
        self.display_id = opt.display_id
        # self.use_html = opt.is_train and not opt.no_html
        self.win_size = opt.display_winsize
        self.name = opt.name
        self.display_cnt = 0 # display_current_results counter
        self.display_cnt_high = 0
        self.use_html = opt.use_html

        if self.display_id > 0:
            import visdom
            self.vis = visdom.Visdom(port = opt.display_port)

        self.web_dir = os.path.join(opt.checkpoints_dir, opt.name, 'web')
        util.mkdirs([self.web_dir,])
        if self.use_html:
            self.img_dir = os.path.join(self.web_dir, 'images')
            print('create web directory %s...' % self.web_dir)
            util.mkdirs([self.img_dir,])

    # |visuals|: dictionary of images to display or save
    def display_current_results(self, visuals, epoch, nrows=None, res=256):
        if self.display_id > 0: # show images in the browser
            title = self.name
            if(nrows is None):
                nrows = int(math.ceil(len(visuals.items()) / 2.0))
            images = []
            idx = 0
            for label, image_numpy in visuals.items():
                title += " | " if idx % nrows == 0 else ", "
                title += label
                img = image_numpy.transpose([2, 0, 1])
                img = zoom_to_res(img,res=res,order=0)
                images.append(img)
                idx += 1
            if len(visuals.items()) % 2 != 0:
                white_image = np.ones_like(image_numpy.transpose([2, 0, 1]))*255
                white_image = zoom_to_res(white_image,res=res,order=0)
                images.append(white_image)
            self.vis.images(images, nrow=nrows, win=self.display_id + 1,
                            opts=dict(title=title))

        if self.use_html: # save images to a html file
            for label, image_numpy in visuals.items():
                img_path = os.path.join(self.img_dir, 'epoch%.3d_cnt%.6d_%s.png' % (epoch, self.display_cnt, label))
                util.save_image(zoom_to_res(image_numpy, res=res, axis=2), img_path)

            self.display_cnt += 1
            self.display_cnt_high = np.maximum(self.display_cnt_high, self.display_cnt)

            # update website
            webpage = html.HTML(self.web_dir, 'Experiment name = %s' % self.name, reflesh=1)
            for n in range(epoch, 0, -1):
                webpage.add_header('epoch [%d]' % n)
                if(n==epoch):
                    high = self.display_cnt
                else:
                    high = self.display_cnt_high
                for c in range(high-1,-1,-1):
                    ims = []
                    txts = []
                    links = []

                    for label, image_numpy in visuals.items():
                        img_path = 'epoch%.3d_cnt%.6d_%s.png' % (n, c, label)
                        ims.append(os.path.join('images',img_path))
                        txts.append(label)
                        links.append(os.path.join('images',img_path))
                    webpage.add_images(ims, txts, links, width=self.win_size)
            webpage.save()

    # save errors into a directory
    def plot_current_errors_save(self, epoch, counter_ratio, opt, errors,keys='+ALL',name='loss', to_plot=False):
        if not hasattr(self, 'plot_data'):
            self.plot_data = {'X':[],'Y':[], 'legend':list(errors.keys())}
        self.plot_data['X'].append(epoch + counter_ratio)
        self.plot_data['Y'].append([errors[k] for k in self.plot_data['legend']])

        # embed()
        if(keys=='+ALL'):
            plot_keys = self.plot_data['legend']
        else:
            plot_keys = keys

        if(to_plot):
            (f,ax) = plt.subplots(1,1)
        for (k,kname) in enumerate(plot_keys):
            kk = np.where(np.array(self.plot_data['legend'])==kname)[0][0]
            x = self.plot_data['X']
            y = np.array(self.plot_data['Y'])[:,kk]
            if(to_plot):
                ax.plot(x, y, 'o-', label=kname)
            np.save(os.path.join(self.web_dir,'%s_x')%kname,x)
            np.save(os.path.join(self.web_dir,'%s_y')%kname,y)

        if(to_plot):
            plt.legend(loc=0,fontsize='small')
            plt.xlabel('epoch')
            plt.ylabel('Value')
            f.savefig(os.path.join(self.web_dir,'%s.png'%name))
            f.clf()
            plt.close()

    # errors: dictionary of error labels and values
    def plot_current_errors(self, epoch, counter_ratio, opt, errors):
        if not hasattr(self, 'plot_data'):
            self.plot_data = {'X':[],'Y':[], 'legend':list(errors.keys())}
        self.plot_data['X'].append(epoch + counter_ratio)
        self.plot_data['Y'].append([errors[k] for k in self.plot_data['legend']])
        self.vis.line(
            X=np.stack([np.array(self.plot_data['X'])]*len(self.plot_data['legend']),1),
            Y=np.array(self.plot_data['Y']),
            opts={
                'title': self.name + ' loss over time',
                'legend': self.plot_data['legend'],
                'xlabel': 'epoch',
                'ylabel': 'loss'},
            win=self.display_id)

    # errors: same format as |errors| of plotCurrentErrors
    def print_current_errors(self, epoch, i, errors, t, t2=-1, t2o=-1, fid=None):
        message = '(ep: %d, it: %d, t: %.3f[s], ept: %.2f/%.2f[h]) ' % (epoch, i, t, t2o, t2)
        message += (', ').join(['%s: %.3f' % (k, v) for k, v in errors.items()])

        print(message)
        if(fid is not None):
            fid.write('%s\n'%message)


    # save image to the disk
    def save_images_simple(self, webpage, images, names, in_txts, prefix='', res=256):
        image_dir = webpage.get_image_dir()
        ims = []
        txts = []
        links = []

        for name, image_numpy, txt in zip(names, images, in_txts):
            image_name = '%s_%s.png' % (prefix, name)
            save_path = os.path.join(image_dir, image_name)
            if(res is not None):
                util.save_image(zoom_to_res(image_numpy,res=res,axis=2), save_path)
            else:
                util.save_image(image_numpy, save_path)

            ims.append(os.path.join(webpage.img_subdir,image_name))
            # txts.append(name)
            txts.append(txt)
            links.append(os.path.join(webpage.img_subdir,image_name))
        # embed()
        webpage.add_images(ims, txts, links, width=self.win_size)

    # save image to the disk
    def save_images(self, webpage, images, names, image_path, title=''):
        image_dir = webpage.get_image_dir()
        # short_path = ntpath.basename(image_path)
        # name = os.path.splitext(short_path)[0]
        # name = short_path
        # webpage.add_header('%s, %s' % (name, title))
        ims = []
        txts = []
        links = []

        for label, image_numpy in zip(names, images):
            image_name = '%s.jpg' % (label,)
            save_path = os.path.join(image_dir, image_name)
            util.save_image(image_numpy, save_path)

            ims.append(image_name)
            txts.append(label)
            links.append(image_name)
        webpage.add_images(ims, txts, links, width=self.win_size)

    # save image to the disk
    # def save_images(self, webpage, visuals, image_path, short=False):
    #     image_dir = webpage.get_image_dir()
    #     if short:
    #         short_path = ntpath.basename(image_path)
    #         name = os.path.splitext(short_path)[0]
    #     else:
    #         name = image_path

    #     webpage.add_header(name)
    #     ims = []
    #     txts = []
    #     links = []

    #     for label, image_numpy in visuals.items():
    #         image_name = '%s_%s.png' % (name, label)
    #         save_path = os.path.join(image_dir, image_name)
    #         util.save_image(image_numpy, save_path)

    #         ims.append(image_name)
    #         txts.append(label)
    #         links.append(image_name)
    #     webpage.add_images(ims, txts, links, width=self.win_size)
