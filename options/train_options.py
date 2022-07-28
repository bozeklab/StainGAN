from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    def initialize(self):

        BaseOptions.initialize(self)
        parser.add_argument("--name", default="StainNet", type=str,
                        help="name of the experiment.")
        parser.add_argument("--source_root_train", default="dataset/Cytopathology/train/trainA", type=str,
                            help="path to source images for training")
        parser.add_argument("--gt_root_train", default="dataset/Cytopathology/train/trainB", type=str,
                            help="path to ground truth images for training")
        parser.add_argument("--source_root_test", default="dataset/Cytopathology/test/testA", type=str,
                            help="path to source images for test")
        parser.add_argument("--gt_root_test", default="dataset/Cytopathology/test/testB", type=str,
                            help="path to ground truth images for test")
        parser.add_argument('--input_nc', type=int, default=3, help='# of input image channels')
        parser.add_argument('--output_nc', type=int, default=3, help='# of output image channels')
        parser.add_argument('--channels', type=int, default=32, help='# of channels in StainNet')
        parser.add_argument('--n_layer', type=int, default=3, help='# of layers in StainNet')
        parser.add_argument('--batchSize', type=int, default=10, help='input batch size')
        parser.add_argument('--nThreads', default=4, type=int, help='# threads for loading data')
        parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        parser.add_argument('--fineSize', type=int, default=256, help='crop to this size')
        parser.add_argument('--display_freq', type=int, default=50, help='frequency of showing training results on screen')
        parser.add_argument('--test_freq', type=int, default=5, help='frequency of cal')
        parser.add_argument('--lr', type=float, default=0.01, help='initial learning rate for SGD')
        parser.add_argument('--epoch', type=int, default=300, help='how many epoch to train')
        self.isTrain = True
