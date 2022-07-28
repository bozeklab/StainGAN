import numpy as np
import os
import ntpath
import time
from . import util
# from scipy.misc import imresize
from PIL import Image
import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (20, 16) 

class Visualizer():
    def __init__(self, opt):
        # self.opt = opt
        self.display_id = opt.display_id
        self.use_html = opt.isTrain and not opt.no_html
        self.win_size = opt.display_winsize
        self.name = opt.name
        self.opt = opt
        self.saved = False

        self.log_name = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.txt')
        with open(self.log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training Loss (%s) ================\n' % now)

    def reset(self):
        self.saved = False

    # errors: dictionary of error labels and values
    def plot_current_errors(self, epoch, counter_ratio, opt, errors):
        if not hasattr(self, 'plot_data'):
            self.plot_data = {'X': [], 'Y': [], 'legend': list(errors.keys())}
        self.plot_data['X'].append(epoch + counter_ratio)
        self.plot_data['Y'].append([errors[k].item() for k in self.plot_data['legend']])
        
        figure, axis = plt.subplots((len(self.plot_data['legend'])+1)//2,2)
        for i, (y, label) in enumerate(zip(np.array(self.plot_data['Y']).T, self.plot_data['legend'])):
            axis[i//2, i%2].plot(self.plot_data['X'], y)
            axis[i//2, i%2].set_title(label)
        plt.savefig("logs/losses.png")
        plt.cla()
        plt.close('all')

    def print_current_errors(self, epoch, i, errors, t):
        message = '(epoch: %d, iters: %d, time: %.3f) ' % (epoch, i, t)
        for k, v in errors.items():
            message += '%s: %.3f ' % (k, v)

        print(message)
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)