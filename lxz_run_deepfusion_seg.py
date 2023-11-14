import datetime
import os
import time
import sys
import numpy as np
from openpyxl import Workbook, load_workbook
import torch
import torch.utils.data
from torch import nn
import torchvision
from mosh.config import SMPLX_MODEL_NEUTRAL_PATH
from nn.p4t.tools import copy2cpu as c2c

from nn.p4t.tools import rotation6d_2_rot_mat, rodrigues_2_rot_mat
from nn.p4t import utils
from nn.p4t.modules.geodesic_loss import GeodesicLoss
from nn.p4t.modules.seg_loss import CrossEntropyLoss, FocalLoss, SmoothClsLoss, DiceLoss
from nn.p4t.scheduler import WarmupMultiStepLR
from nn.datasets.seg_dataset import SegDataset
import nn.fusion_model.modules.deepfusion as Models
# from visualization.mesh_plot import MoshEvaluateStreamPlot, pcl2box
from nn.p4t.modules.loss import LossManager


def train_one_epoch(args, model, losses, criterions, loss_weight, optimizer, lr_scheduler, data_loader, device, epoch):

    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value}"))
    metric_logger.add_meter("clips/s", utils.SmoothedValue(window_size=10, fmt="{value:.3f}"))

    header = "Epoch: [{}]".format(epoch)

    for input in metric_logger.log_every(data_loader, args.print_freq, header):
        start_time = time.time()
        label = ['img_label', 'pcd_label']
        if isinstance(input, dict):
            for k, v in input.items():
                if k in label:
                    input[k] = v.to(args.device, dtype=torch.int64)
                else:
                    if isinstance(v, torch.Tensor):
                        input[k] = v.to(args.device, dtype=torch.float32)
        else:
            input = input.to(device, dtype=torch.float32)
        img_seg_label, pcd_seg_label = input['img_label'], input['pcd_label']
        img_seg_logit, img_seg_pred, point_seg_logit = model(input)
        
        batch_size = img_seg_label.shape[0]
        img_seg_logit = img_seg_logit.reshape(batch_size, 6, -1)
        img_seg_label = img_seg_label.reshape(batch_size, -1)

        ### bce loss
        losses.update_loss("img_ce_loss", loss_weight[0] * criterions["ce"](img_seg_logit, img_seg_label))
        losses.update_loss("pcd_ce_loss", loss_weight[1] * criterions["ce"](point_seg_logit, pcd_seg_label))
        loss = losses.calculate_total_loss()
        optimizer.zero_grad()
        # with torch.autograd.detect_anomaly():
        loss.backward()
        nn.utils.clip_grad_value_(model.parameters(), 5)
        optimizer.step()

        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters["clips/s"].update(batch_size / (time.time() - start_time))
        lr_scheduler.step()
        sys.stdout.flush()

def main(args):
    output_dir = args.output_dir
    if output_dir and not os.path.exists(os.path.join(output_dir, "pth")):
        utils.mkdir(os.path.join(output_dir, "pth"))

    print(args)
    print("torch version: ", torch.__version__)
    print("torchvision version: ", torchvision.__version__)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    torch.cuda.set_device(args.device)
    device = torch.device("cuda")

    args.inputs = args.inputs.replace(" ", "").split(",")

    # Data loading code
    dataset = SegDataset(args)

    # train_size = int(0.9 * len(dataset))
    # eval_size = len(dataset) - train_size
    # dataset_train, dataset_eval = torch.utils.data.random_split(
    #     dataset, [train_size, eval_size]
    # )
    dataset_train = dataset
    print("Creating data loaders")

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True
    )
    # data_loader_eval = torch.utils.data.DataLoader(
    #     dataset_eval,
    #     batch_size=args.batch_size,
    #     num_workers=args.workers,
    #     pin_memory=True,
    # )

    print("Creating model")
    Model = getattr(Models, args.model)
    model = Model(args)

    # if torch.cuda.device_count() > 1:
    #     model = nn.DataParallel(model)
    model.to(device)

    # bot = TimerBot()
    losses = LossManager()

    cross_entropy_criterion = nn.CrossEntropyLoss()
    smooth_criterion = SmoothClsLoss()
    focal_criterion = FocalLoss()
    dice_criterion = DiceLoss()

    criterions = dict(ce=cross_entropy_criterion)

    lr = args.lr
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # convert scheduler to be per iteration, not per epoch, for warmup that lasts
    # between different epochs
    warmup_iters = args.lr_warmup_epochs * len(data_loader_train)
    lr_milestones = [len(data_loader_train) * m for m in args.lr_milestones]
    lr_scheduler = WarmupMultiStepLR(
        optimizer,
        milestones=lr_milestones,
        gamma=args.lr_gamma,
        warmup_iters=warmup_iters,
        warmup_factor=1e-5,
    )

    model_without_ddp = model

    if args.resume:
        resume = os.path.join(args.resume, "pth", "checkpoint.pth")
        checkpoint = torch.load(resume, map_location="cpu")
        model_without_ddp.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        args.start_epoch = checkpoint["epoch"] + 1

    if args.train:
        print("Start training")
        start_time = time.time()

        loss_weight = list(map(float, args.loss_weight.split(",")))

        for epoch in range(args.start_epoch, args.epochs):
            train_one_epoch(args, model, losses, criterions, loss_weight, optimizer, lr_scheduler, data_loader_train, device, epoch)
            losses.calculate_epoch_loss(os.path.join(output_dir, "loss/train"), epoch)
            # list(evaluate(args, model, losses, criterions, data_loader_eval, device))
            # losses.calculate_epoch_loss(
            #     os.path.join(output_dir, "loss/eval"), epoch)

            if output_dir:
                checkpoint = {
                    "model": model_without_ddp.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "epoch": epoch,
                    "args": args,
                }
                utils.save_on_master(checkpoint, os.path.join(output_dir, "pth", "checkpoint.pth"))
                if (epoch + 5) % 5 == 0:
                    utils.save_on_master(checkpoint, os.path.join(output_dir, "pth", "epoch{}.pth".format(epoch)))
                    os.system("cp -r {}/loss {}/backup".format(output_dir, output_dir))

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print("Training time {}".format(total_time_str))

    else:
        loss_weight = list(map(float, args.loss_weight.split(",")))
        data_loader_test = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True,
        )
        # print("Start testing")
        # save_path = os.path.join(output_dir, "error", args.test_scene)
        # gen = evaluate(
        #     args, model, losses, criterions, data_loader_test, device, save_path
        # )
        # plot = MoshEvaluateStreamPlot()
        # if args.save_snapshot:
        #     snapshot_path = os.path.join(args.output_dir, "snapshot", args.test_scene)
        #     plot.show(gen, fps=30, save_path=snapshot_path)
        # else:
        #     plot.show(gen, fps=30)


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="DeepFusion Model Training")

    parser.add_argument("--seed", default=35, type=int, help="random seed")
    parser.add_argument("--model", default="DeepFusionSeg", type=str, help="model")
    # input
    parser.add_argument("--data_path", default="/remote-home/linxinzhuo/code/labeled_data", type=str, help="dataset")
    parser.add_argument("--seq_idxes", type=str, default="")
    parser.add_argument("--num_points", default=1024, type=int, help="number of points per frame")
    parser.add_argument("--normal_scale", default=1, type=int, help="normal scale of labels")
    parser.add_argument("--skip_head", default=0, type=int, help="number of skip frames")
    parser.add_argument("--output_dim", default=6, type=int, help="output dim")
    parser.add_argument("--feature_type", default="none", type=str, help="type of features")
    parser.add_argument("--features", default=3, type=int, help="dim of features")
    parser.add_argument("--inputs", type=str, default="lidar,image", help="input data")
    # P4D
    parser.add_argument("--radius", default=0.7, type=float, help="radius for the ball query")
    parser.add_argument("--nsample", default=32, type=int, help="number of neighbors for the ball query")
    parser.add_argument("--npoint", default=49, type=int, help="number of points for FPS")
    # embedding
    parser.add_argument("--emb_relu", default=False, action="store_true")
    # transformer
    parser.add_argument("--dim", default=1024, type=int, help="transformer dim")
    parser.add_argument("--depth", default=5, type=int, help="transformer depth")
    parser.add_argument("--heads", default=8, type=int, help="transformer head")
    parser.add_argument("--dim_head", default=128, type=int, help="transformer dim for each head")
    parser.add_argument("--mlp_dim", default=2048, type=int, help="transformer mlp dim")
    # training
    parser.add_argument("-b", "--batch_size", default=16, type=int)
    parser.add_argument("--epochs", default=1000, type=int, metavar="N", help="number of total epochs to run")
    parser.add_argument("-j", "--workers", default=4, type=int, metavar="N", help="number of data loading workers (default: 16)")
    parser.add_argument("--lr", default=0.0001, type=float, help="initial learning rate")
    parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")
    parser.add_argument(
        "--wd",
        "--weight_decay",
        default=1e-4,
        type=float,
        metavar="W",
        help="weight decay (default: 1e-4)",
        dest="weight_decay",
    )
    parser.add_argument(
        "--lr_milestones",
        nargs="+",
        default=[100, 200],
        type=int,
        help="decrease lr on milestones",
    )
    parser.add_argument(
        "--lr_gamma",
        default=0.1,
        type=float,
        help="decrease lr by a factor of lr-gamma",
    )
    parser.add_argument("--lr_warmup_epochs", default=10, type=int, help="number of warmup epochs")
    parser.add_argument("--loss_weight", default="1,1", type=str, help="weight of loss")  #####要改
    parser.add_argument("--use_gender", default=0, type=int, help="use gender")
    parser.add_argument("--device", default=0, type=int, help="cuda device")
    # output
    parser.add_argument("--print_freq", default=100, type=int, help="print frequency")
    parser.add_argument("--num_frames", default=10000, type=int, help="number of test frames")
    parser.add_argument("--output_dir", default="/remote-home/linxinzhuo/code/lxz_3DSVC", type=str, help="path where to save")
    # resume
    parser.add_argument("--resume", default="", help="resume from checkpoint")
    parser.add_argument("--start_epoch", default=0, type=int, metavar="N", help="start epoch")
    parser.add_argument("--train", dest="train", default=True, help="train or test")
    # parser.add_argument("--visual", dest="visual", action="store_true", help="visual")
    parser.add_argument("--create_pkl", dest="create_pkl", action="store_true", help="create pkl data")
    parser.add_argument("--use_pkl", dest="use_pkl", action="store_true", help="use pkl data")
    parser.add_argument(
        "--save_snapshot",
        dest="save_snapshot",
        action="store_true",
        help="save snapshot",
    )
    parser.add_argument("--read_orig_img", action="store_true")

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    # torch.autograd.set_detect_anomaly(True)
    main(args)
