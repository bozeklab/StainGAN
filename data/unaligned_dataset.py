import os.path
import torchvision.transforms as transforms
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import PIL
import random
from torch.utils.data import WeightedRandomSampler
from data.sampling_weights import get_weights


class UnalignedDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_A = os.path.join(opt.dataroot, opt.phaseA)
        self.dir_B = os.path.join(opt.dataroot, opt.phaseB)
        print(self.dir_A, self.dir_B)

        self.A_paths = make_dataset(self.dir_A)
        self.B_paths = make_dataset(self.dir_B)

        self.A_paths = sorted(self.A_paths)
        self.B_paths = sorted(self.B_paths)
        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)

        self.transformA = get_transform(opt, means= [0.6770, 0.6700, 0.6744], stds = [0.1621, 0.1518, 0.1473])
        self.transformB = get_transform(opt, means= [0.8811, 0.8758, 0.9151], stds = [0.0824, 0.0732, 0.0758])

        print(self.A_size, self.B_size)

        self.samplerB = WeightedRandomSampler(get_weights(opt.csvB), opt.epoch_len, replacement=True)
        self.epoch_len = opt.epoch_len
        self.current_sample = opt.epoch_len

    def __getitem__(self, index):
        index_A = index % self.A_size
        index_B = self.next_B_index() % self.B_size

        # if self.opt.serial_batches:
        #     index_B = index % self.B_size
        # else:
        #     index_B = random.randint(0, self.B_size - 1)

        A_path = self.A_paths[index_A]
        B_path = self.B_paths[index_B]
        # print('(A, B) = (%d, %d)' % (index_A, index_B))
        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')

        A = self.transformA(A_img)
        B = self.transformB(B_img)
        if self.opt.which_direction == 'BtoA':
            input_nc = self.opt.output_nc
            output_nc = self.opt.input_nc
        else:
            input_nc = self.opt.input_nc
            output_nc = self.opt.output_nc

        if input_nc == 1:  # RGB to gray
            tmp = A[0, ...] * 0.299 + A[1, ...] * 0.587 + A[2, ...] * 0.114
            A = tmp.unsqueeze(0)

        if output_nc == 1:  # RGB to gray
            tmp = B[0, ...] * 0.299 + B[1, ...] * 0.587 + B[2, ...] * 0.114
            B = tmp.unsqueeze(0)
        return {'A': A, 'B': B,
                'A_paths': A_path, 'B_paths': B_path}

    def next_B_index(self):
        if self.current_sample >= self.epoch_len:
            self.B_samples = list(self.samplerB)
            self.current_sample = 0
        ans = self.B_samples[self.current_sample]
        self.current_sample += 1
        return ans 

    def __len__(self):
        return max(self.A_size, self.B_size)

    def name(self):
        return 'UnalignedDataset'
