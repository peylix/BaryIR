import argparse
import os
import torch
import numpy as np
import time, math, glob
from PIL import Image
import cv2
from util.metrics import calculate_psnr, calculate_ssim
import torchvision
from torchvision.utils import save_image
from pytorch_fid import fid_score
from thop import profile

parser = argparse.ArgumentParser(description="PyTorch SRResNet Eval")
parser.add_argument("--cuda", action="store_true", help="use cuda?")
parser.add_argument("--degset", default="./data/test/derain/Rain100L/input/", type=str, help="degraded data")
parser.add_argument("--tarset", default="./data/test/derain/Rain100L/target/", type=str, help="target data")
# # parser.add_argument("--degset", default="./data/test/JPEG/bsds500/color/qf_20/", type=str, help="degraded data")
# parser.add_argument("--tarset", default="./data/test/refimgs/", type=str, help="target data")
# parser.add_argument("--degset", default="./data/test/dehaze/input/", type=str, help="degraded data")
# parser.add_argument("--tarset", default="./data/test/dehaze/target/", type=str, help="target data")
# parser.add_argument("--degset", default="./data/test/lowlight/eval15/low/", type=str, help="degraded data")
# parser.add_argument("--tarset", default="./data/test/lowlight/eval15/high/", type=str, help="target data")

# parser.add_argument("--degset", default="./data/test/deblur/input/", type=str, help="degraded data")
# parser.add_argument("--tarset", default="./data/test/deblur/target/", type=str, help="target data")

# parser.add_argument("--save", default="./results/lowlight/OUT/", type=str, help="savepath, Default: results")
# parser.add_argument("--savetar", default="./results/lowlight/TAR/", type=str, help="savepath, Default: targets"

parser.add_argument("--save", default="./results/derain/OUT/", type=str, help="savepath, Default: results")
parser.add_argument("--savetar", default="./results/derain/TAR/", type=str, help="savepath, Default: targets")

parser.add_argument("--model", default="./checkpoint/model_allBaryNet128__60_1.0.pth", type=str, help="model path")
# parser.add_argument("--saveres", default="./results/rain/RES/", type=str, help="savepath, Default: residual")

parser.add_argument("--gpus", default="2", type=str, help="gpu ids")

def PSNR(pred, gt, shave_border=0):
    height, width = pred.shape[:2]
    pred = pred[shave_border:height - shave_border, shave_border:width - shave_border]
    gt = gt[shave_border:height - shave_border, shave_border:width - shave_border]
    imdff = pred - gt
    rmse = math.sqrt((imdff ** 2).mean())
    if rmse == 0:
        return 100  
    return 20 * math.log10(1.0 / rmse)


opt = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpus)
cuda = True#opt.cuda

if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

os.makedirs(opt.save, exist_ok=True)
os.makedirs(opt.savetar, exist_ok=True)

BaryIR = torch.load(opt.model)["BaryIR"]
if cuda:
    BaryIR = BaryIR.cuda()
BaryIR.eval()

# ---- Parameter Count ----
total_params = sum(p.numel() for p in BaryIR.parameters())
trainable_params = sum(p.numel() for p in BaryIR.parameters() if p.requires_grad)
print("="*60)
print(f"Total parameters:     {total_params:,} ({total_params/1e6:.2f}M)")
print(f"Trainable parameters: {trainable_params:,} ({trainable_params/1e6:.2f}M)")
print("="*60)

deg_list = glob.glob(opt.degset+"*")
deg_list = sorted(deg_list)

tar_list = sorted(glob.glob(opt.tarset+"*"))
num = len(deg_list)
data_list = []
inference_times = []

with torch.no_grad():
    for img_idx, (deg_name, tar_name) in enumerate(zip(deg_list, tar_list)):
        name = tar_name.split('/')
        print("Processing ", deg_name)
        deg_img = Image.open(deg_name).convert('RGB')
        tar_img = Image.open(tar_name).convert('RGB')
        deg_img = np.array(deg_img)
        tar_img = np.array(tar_img)

        h,w = deg_img.shape[0],deg_img.shape[1]
        shape1 = deg_img.shape
        shape2 = tar_img.shape
        while (h % 8) != 0:
            h=h-1
            deg_img = deg_img[0:h, :]
            tar_img = tar_img[0:h, :]
        while (w % 8) != 0:
            w=w-1
            deg_img = deg_img[:, 0:w]
            tar_img = tar_img[:, 0:w]
        if shape1 != shape2:
            continue
        deg_img = np.transpose(deg_img, (2, 0, 1))
        deg_img = torch.from_numpy(deg_img).float() / 255
        deg_img = deg_img.unsqueeze(0)
        tar_img = np.transpose(tar_img, (2, 0, 1))
        tar_img = torch.from_numpy(tar_img).float() / 255
        tar_img = tar_img.unsqueeze(0)
        gt = tar_img

        data_degraded = deg_img
        if cuda:
            gt=gt.cuda()
            data_degraded = data_degraded.cuda()

        # Timing: use cuda events for accurate GPU timing
        if cuda:
            torch.cuda.synchronize()
        start_time = time.time()

        im_output, _ ,_,_ = BaryIR(data_degraded)

        if cuda:
            torch.cuda.synchronize()
        elapsed = time.time() - start_time

        # Skip first image (CUDA warmup)
        if img_idx > 0:
            inference_times.append(elapsed)

        res = data_degraded - im_output

        save_image(im_output.data,opt.save+'/'+name[-1])
        save_image(tar_img.data, opt.savetar+'/'+name[-1])

# ---- Inference Time Stats ----
print("="*60)
if len(inference_times) > 0:
    avg_time = sum(inference_times) / len(inference_times)
    min_time = min(inference_times)
    max_time = max(inference_times)
    total_time = sum(inference_times)
    print(f"Inference time (excluding 1st image warmup):")
    print(f"  Images timed: {len(inference_times)}")
    print(f"  Average:      {avg_time:.4f}s ({1/avg_time:.2f} FPS)")
    print(f"  Min:          {min_time:.4f}s")
    print(f"  Max:          {max_time:.4f}s")
    print(f"  Total:        {total_time:.4f}s")
else:
    print("Not enough images to compute inference time (need at least 2).")
print("="*60)

try:
    fid_value = fid_score.calculate_fid_given_paths([opt.savetar, opt.save], batch_size=1,
                                                    device='cuda', dims=2048, num_workers=4)
    print('FID value:', fid_value)
except Exception as e:
    print('FID computation failed:', e)
    print('FID value: N/A')


tar_files = sorted(os.listdir(opt.savetar))
out_files = sorted(os.listdir(opt.save))
psnr_sum, ssim_sum, n = 0.0, 0.0, 0
pmax, smax, pmin, smin = 0.0, 0.0, 100.0, 1.0
best_pname = best_sname = worst_pname = worst_sname = ''
for nt, no in zip(tar_files, out_files):
    im_t = cv2.imread(os.path.join(opt.savetar, nt))
    im_o = cv2.imread(os.path.join(opt.save, no))
    if im_t is None or im_o is None or im_t.shape != im_o.shape:
        continue
    p = calculate_psnr(im_o, im_t, test_y_channel=True)
    s = calculate_ssim(im_o, im_t, test_y_channel=True)
    psnr_sum += p; ssim_sum += s; n += 1
    if p > pmax: pmax, best_pname = p, nt
    if p < pmin: pmin, worst_pname = p, nt
    if s > smax: smax, best_sname = s, nt
    if s < smin: smin, worst_sname = s, nt

psnr_avg = psnr_sum / n
ssim_avg = ssim_sum / n
print("PSNR (Y-channel): Average {:.5f},   best {:.5f} ({}),   worst {:.5f} ({})".format(psnr_avg, pmax, best_pname, pmin, worst_pname))
print("SSIM (Y-channel): Average {:.5f},   best {:.5f} ({}),   worst {:.5f} ({})".format(ssim_avg, smax, best_sname, smin, worst_sname))
