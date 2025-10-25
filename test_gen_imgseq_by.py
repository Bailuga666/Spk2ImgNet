import argparse
import time
import os
import glob
import numpy as np
import cv2
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import torch
from torch.autograd import Variable
from dataset_by import raw_to_spike, raw_to_spike_by
from nets import SpikeNet

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description="Spike_Net_Test_256")
parser.add_argument("--num_of_layers", type=int, default=17, help="Number of total layers")
parser.add_argument("--logdir", type=str, default="./ckpt_spike/", help="path of log files")
parser.add_argument("--test_data", type=str, default="./Spk2ImgNet_test2/test2/", help="test set root")
parser.add_argument("--save_result", type=bool, default=True, help="save the reconstruction or not")
parser.add_argument("--result_dir", type=str, default="results_by/", help="path of results")
parser.add_argument("--exist_gt", default=True, help="exist ground truth or not (True/False)")
parser.add_argument("--no-gt", dest='exist_gt', action='store_false', help='shorthand to disable ground truth checking')
parser.add_argument("--model_name", type=str, default="model_061.pth", help="Name of checkpoint")
parser.add_argument("--win_r", type=int, default=80, help="window radius (half window)")
parser.add_argument("--win_step", type=int, default=80, help="window step")
parser.add_argument("--in_channels", type=int, default=161, help="input channels (window length)")
opt = parser.parse_args()

if isinstance(opt.exist_gt, str):
    opt.exist_gt = opt.exist_gt.lower() in ("1", "true", "yes", "y")

def normalize(data):
    return data / 255.0

def main():
    print("Loading model ... \n")
    net = SpikeNet(
        in_channels=opt.in_channels, features=64, out_channels=1, win_r=opt.win_r, win_step=opt.win_step
    )
    model = torch.nn.DataParallel(net).cuda()
    model.load_state_dict(torch.load(os.path.join(opt.logdir, opt.model_name)))
    model.eval()

    print("Loading data info ...\n")
    files_source = glob.glob(os.path.join(opt.test_data, "input", "*.dat"))
    files_source.sort()

    psnr_test = 0
    ssim_test = 0
    for i in range(len(files_source)):
        sub_dir = files_source[i][:-4]
        input_f = open(files_source[i], "rb+")
        video_seq = input_f.read()
        video_seq = np.fromstring(video_seq, "B")




        InSpikeArray = raw_to_spike_by(video_seq, 256, 256)  # c*h*w






        [c, h, w] = InSpikeArray.shape
        # 测试所有帧或指定帧
        for key_id in np.arange(240, 241, 1):  # 与训练一致，默认每80帧一个GT
            start_t = time.time()
            SpikeArray = InSpikeArray[key_id-240:key_id+241, :, :]
            print(f"InSpikeArray shape: {InSpikeArray.shape}")
            print(key_id-81,key_id+80)
            print("截取后的形状:", SpikeArray.shape)
            SpikeArray = np.expand_dims(SpikeArray, 0)  # n*c*h*w
            print("截取后的形状:", SpikeArray.shape)
            file_name = files_source[i].replace("\\", "/").split("/")[-1]
            
            SpikeArray = Variable(torch.Tensor(SpikeArray)).cuda()
            with torch.no_grad():
                if opt.exist_gt:
                    print("截取后的形状:", SpikeArray.shape)
                    out_rec, est0, est1, est2, est3, est4 = model(SpikeArray)
                    out_rec = torch.clamp(out_rec / 0.6, 0, 1).cpu() * 255
                else:
                    print("截取后的形状:", SpikeArray.shape)
                    out_rec, est0, est1, est2, est3, est4 = model(SpikeArray)
                    out_rec = torch.clamp(out_rec, 0, 1).cpu() ** (1 / 2.2) * 255
            out_rec = out_rec.detach().numpy().astype(np.float32)
            out_rec = np.squeeze(out_rec).astype(np.uint8)
            out_rec = out_rec[:256, :256]
            if opt.exist_gt:
                gt_path = os.path.join(opt.test_data, "gt", file_name[:-6] + str(key_id) + ".png")
                if os.path.exists(gt_path):
                    gt = cv2.imread(gt_path, 0)
                    psnr = peak_signal_noise_ratio(gt, out_rec)
                    ssim = structural_similarity(gt, out_rec)
                    print(f"{file_name} frame {key_id}: PSNR:{psnr:.2f} SSIM:{ssim:.4f}")
                    psnr_test += psnr
                    ssim_test += ssim
                else:
                    print(f"GT not found: {gt_path}")
            if opt.save_result:
                save_subdir = os.path.join(opt.result_dir, sub_dir)
                if not os.path.exists(save_subdir):
                    os.makedirs(save_subdir)
                cv2.imwrite(os.path.join(save_subdir, f"{key_id}.png"), out_rec)
                dur_time = time.time() - start_t
                print(f"dur_time:{dur_time:.2f}s")

    if opt.exist_gt and len(files_source) > 0:
        avg_psnr = psnr_test / (len(files_source) * 5)  # 5帧/文件
        avg_ssim = ssim_test / (len(files_source) * 5)
        print(f"average PSNR: {avg_psnr:.2f} average SSIM: {avg_ssim:.4f}")

if __name__ == "__main__":
    main()
