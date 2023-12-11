import os
import time
import logging
import sys
import config
import utils
import math
import argparse

import torch
from torchvision.utils import save_image
from torch.utils.data import DataLoader

# from check_psnr_ssim import check_psnr_ssim_overall
from pytorch_msssim import ssim_matlab as calc_ssim

parser = argparse.ArgumentParser(description='Video Frame Interpolation Testing')
parser.add_argument('--random_seed', default=0, type=int)

parser.add_argument('--datasetName', type=str, default='Vimeo_90K',
                    choices=['UCF101', 'Vimeo_90K', 'VimeoSepTuplet', 'Snufilm'])
parser.add_argument('--datasetPath',
                    default='')
parser.add_argument('--test_batch_size', default=256, type=int)
parser.add_argument('--num_workers', default=4, type=int)

parser.add_argument('--modelName', type=str, default='RSTSCANet',
                    choices=['RSTSCANet', 'CAIN', 'VFIT_B', 'VFIformer'])
parser.add_argument('--loss', type=str, default='1*L1')

parser.add_argument('--checkpoint_dir', type=str,
                    default='F:\\Pycharm Projects\\RSTSCANet_VFI_Kien1\\checkpoints\\RSTCANet_best.pth')
parser.add_argument('--save_folder', default='./test_results', type=str)
parser.add_argument('--save_images', default=False, type=bool)


##### Parse CmdLine Arguments #####
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
args = parser.parse_args()
cwd = os.getcwd()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.manual_seed(args.random_seed)

#### Dataset loading ####
if args.datasetName == 'UCF101':
    from datasets.ucf101 import ucf101
    args.datasetPath = 'D:\\KIEN\Dataset\\UCF101\\UCF101_Dataset\\ucf101_interp'
    test_set = ucf101.UCF101(root=args.datasetPath, is_training=False)
elif args.datasetName == 'Vimeo_90K':
    from datasets.vimeo_90K.vimeo_90K import Vimeo_90K
    args.datasetPath = 'F:\\Pycharm Projects\\RSTSCANet_VFI_Kienv1\\data\\vimeo_triplet'
    test_set = Vimeo_90K(root=args.datasetPath, is_training=False)
elif args.datasetName == 'VimeoSepTuplet':
    from datasets.vimeo_90K.vimeo_90K import VimeoSepTuplet
    args.datasetPath = 'D:\\KIEN\\Dataset\\vimeo_septuplet'
    test_set = VimeoSepTuplet(root=args.datasetPath, is_training=False, mode='full')
elif args.datasetName == 'Snufilm':
    args.datasetPath = 'D:\\KIEN\\Dataset\\SNUFILM'
    from datasets.snu_film.snufilm import SNUFILM
    test_set = SNUFILM(data_root=args.datasetPath, mode='hard')

test_loader = torch.utils.data.DataLoader(test_set, pin_memory=True,
                                             batch_size=args.test_batch_size,
                                             num_workers=args.num_workers,
                                             shuffle=False, drop_last=False)

print("\nBuilding model: %s"%args.modelName)
args_model, unparsed = config.get_args()
# if args.modelName == 'VFIT_B':
#     from my_models.Sep_STS.VFIT_B import UNet_3D_3D
#     model = UNet_3D_3D(n_inputs=args_model.nb_frame, joinType=args_model.joinType)
# elif args.modelName == 'VFIT_S':
#     from my_models.Sep_STS.VFIT_S import UNet_3D_3D
#     model = UNet_3D_3D(n_inputs=args_model, joinType=args_model.joinType)
# elif args.modelName == 'RSTSCANet':
#     from rstsca_model import RSTSCANet
#     model = RSTSCANet(args_model)
from model.rstsca_model import RSTSCANet
model = RSTSCANet.RSTSCANet(args_model)
# elif args.modelName == 'VFIformer':
#     from Other_Models.VFIformer.VFIformer_models.trainer import Trainer
#     model = Trainer(args)
#     model.test()
# elif args.modelName == 'CAIN':
#     from Other_Models.CAIN.CAIN_model.cain import CAIN
#     model = CAIN()
 # model = torch.nn.DataParallel(model).to(device)
model = model.to(device)
# print(model)
print("#params", sum([p.numel() for p in model.parameters()]))

def save_batch_images(ims_pred, ims_gt):
    save_images_path = os.path.join(args.save_folder, args.modelName, args.datasetName)
    if not os.path.exists(os.path.join(save_images_path)):
        os.makedirs(os.path.join(save_images_path))
    # Save every image in batch to indicated location
    for j in range(ims_pred.size(0)):
        pred_name = str(args_model.out_counter) + '_out.png'
        gt_name = str(args_model.out_counter) + '_gt.png'

        save_image(ims_pred[j, :, :, :], os.path.join(save_images_path, pred_name))
        save_image(ims_gt[j, :, :, :], os.path.join(save_images_path, gt_name))

        args_model.out_counter += 1

def calc_psnr(pred, gt):
    diff = (pred - gt).pow(2).mean() + 1e-8
    return -10 * math.log10(diff)

def eval_metrics(im_pred, im_gt, psnrs, ssims, folder):
    # PSNR should be calculated for each image, since sum(log) =/= log(sum).
    for i in range(im_gt.size()[0]):
        psnr = calc_psnr(im_pred[i], im_gt[i])
        psnrs.update(psnr)

        ssim = calc_ssim(im_pred[i].unsqueeze(0).clamp(0, 1), im_gt[i].unsqueeze(0).clamp(0, 1),
                         val_range=1.)
        ssims.update(ssim)

        logging.info('testing on: %s    psnr: %.6f  ssim: %.6f' % (folder[i], psnr, ssim))


def setup_logger(log_file_path):
    log_formatter = logging.Formatter("%(asctime)s [%(levelname)-5.5s] %(message)s")
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    log_file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
    log_file_handler.setFormatter(log_formatter)
    root_logger.addHandler(log_file_handler)

    log_stream_handler = logging.StreamHandler(sys.stdout)
    log_stream_handler.setFormatter(log_formatter)
    root_logger.addHandler(log_stream_handler)

    logging.info('Logging file is %s' % log_file_path)

def test(args):
    time_taken = []
    losses, psnrs, ssims = utils.init_meters(args.loss, reset_loss=True)
    model.eval()
    args_model.out_counter = 0

    save_log_path = os.path.join(args.save_folder, args.modelName)
    if not os.path.exists(save_log_path):
        os.makedirs(save_log_path)
    log_file_path = save_log_path + '/' + time.strftime('%Y%m%d_%H%M%S') + '_' + args.datasetName + '.log'
    setup_logger(log_file_path)

    for arg in vars(args):
        logging.info(arg + ':%s' % getattr(args, arg))
    logging.info('parameters: %s' % (sum([p.numel() for p in model.parameters()])))

    logging.info('------Start Testing------')
    logging.info('%d testing sample' % (test_set.__len__()))
    start = time.time()
    with torch.no_grad():
        for i, (images, gt_image, folders) in enumerate(test_loader):

            images = [img_.cuda() for img_ in images]
            gt = gt_image.cuda()

            torch.cuda.synchronize()

            out = model(images)

            # torch.cuda.synchronize()
            time_taken.append(time.time() - start)

            eval_metrics(out, gt, psnrs, ssims, folders)
            if args.save_images:
                save_batch_images(out, gt)

    logging.info('--------- average PSNR: %.6f,   SSIM: %.6f,   Time: %s' % (psnrs.avg, ssims.avg.item(), sum(time_taken)/len(time_taken)))


""" Entry Point """
def main(args):
    assert args.checkpoint_dir is not None
    checkpoint = torch.load(args.checkpoint_dir)

    model.load_state_dict(checkpoint["state_dict"])
    test(args)


if __name__ == "__main__":
    main(args)
    # check_psnr_ssim_overall(data_path='./test_results/')
