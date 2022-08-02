import time
import os, sys
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model

from PIL import Image
from torch import nn
import torch
from data.base_dataset import get_transform
import torchvision

opt = TestOptions().parse()
opt.nThreads = 1  # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.no_flip = True  # no flip

transforms = get_transform(opt)

BASE_PATH = "/data/shared/her2-images/test-set-external"
img_paths = [os.path.join(BASE_PATH, f) for f in os.listdir(BASE_PATH) if os.path.isfile(os.path.join(BASE_PATH, f))]

# test
start_time = time.time()
TARGET_SIZE = 256
STRIDE_SIZE = TARGET_SIZE // 2
model = create_model(opt)

RESULT_VERSION = "v8-full"
RESULT_PATH = os.path.join("/data/khusiaty/result", RESULT_VERSION)
if not os.path.exists(RESULT_PATH):
    os.makedirs(RESULT_PATH)

def remove_odd_shape(img):
    n_img = img 
    if n_img.shape[-1] % 2 == 1:
        n_img = n_img[:, :, :, 1:]

    if n_img.shape[-2] % 2 == 1:
        n_img = n_img[:, :, 1:, :]
    return n_img

def _get_padding(value):
    aux = STRIDE_SIZE - value 
    aux %= STRIDE_SIZE
    aux //= 2

    return aux

def get_padding(img):
    fst = _get_padding(img.shape[-2])
    snd = _get_padding(img.shape[-1])

    return (fst, snd)

def pass_tensor(model, tensor):
    tensor = tensor.to(torch.cuda.current_device())
    fake, cycle = model.forward_direct(tensor)

    fake = fake.cpu()
    cycle = cycle.cpu()

    return fake, cycle

for path in img_paths:
    print(path)
    img = Image.open(path).convert('RGB')
    img = transforms(img).unsqueeze(0)
    img = remove_odd_shape(img)
    padding = get_padding(img)
    
    new_path = os.path.join(RESULT_PATH, path.split("/")[-1])

    fold_params = dict(kernel_size=TARGET_SIZE, stride=STRIDE_SIZE, padding=padding)
    unfold = nn.Unfold(**fold_params)
    fold = nn.Fold(img.shape[2:], **fold_params)

    divisor = fold(unfold(torch.ones_like(img)))
    unfolded_img = unfold(img)
    unfolded_img_shape = unfolded_img.shape
    unfolded_img = unfolded_img.view(3, TARGET_SIZE, TARGET_SIZE, -1)
    unfolded_img = torch.permute(unfolded_img, (3,0,1,2))

    pairs = [pass_tensor(model, tensor) for tensor in torch.split(unfolded_img, 32, dim=0)]

    fake = [pair[0] for pair in pairs]
    fake = torch.cat(fake, dim=0)
    fake = torch.permute(fake, (1,2,3,0))
    fake = fake.view(unfolded_img_shape)
    fake = fold(fake) 
    fake = fake / divisor
    fake = fake[0]

    # for i in range(3):
    #     fake[i, ...] = fake[i, ...] * new_stds[i] + new_means[i]

    torchvision.utils.save_image(fake, new_path)

elapsed = (time.time() - start_time)
print("--- %s seconds ---" % round((elapsed ), 2))
