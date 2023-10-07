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
from nn.p4t.scheduler import WarmupMultiStepLR
from nn.datasets.fusion_dataset import FusionDataset
from nn.SMPL.mosh_loss import MoshLoss, SMPLXModel
import nn.fusion_model.modules.deepfusion as Models
from message.dingtalk import TimerBot
from visualization.mesh_plot import MoshEvaluateStreamPlot, pcl2box
from nn.p4t.modules.loss import LossManager


def train_one_epoch(args, model, losses, criterions, loss_weight, optimizer, lr_scheduler, data_loader, device, epoch):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter(
        'lr', utils.SmoothedValue(window_size=1, fmt='{value}'))
    metric_logger.add_meter(
        'clips/s', utils.SmoothedValue(window_size=10, fmt='{value:.3f}'))

    header = 'Epoch: [{}]'.format(epoch)

    for input, label in metric_logger.log_every(data_loader, args.print_freq, header):
        start_time = time.time()
        if isinstance(input, dict):
            for k, v in input.items():
                if isinstance(v, torch.Tensor):
                    input[k] = v.to(args.device, dtype=torch.float32)
        else:
            input = input.to(device, dtype=torch.float32)
        label = label.to(device, dtype=torch.float32)
        output = model(input, True)
        batch_size = label.shape[0]
        # translation loss
        losses.update_loss(
            "trans_loss", loss_weight[0]*criterions["mse"](output[:, 0:3], label[:, 0:3]))
        # pose loss
        if args.use_6d_pose:
            output_mat = rotation6d_2_rot_mat(output[:, 3:-16])
            target_mat = rodrigues_2_rot_mat(label[:, 3:-16])
            losses.update_loss(
                "pose_loss", loss_weight[1]*criterions["rot_mat"](output_mat, target_mat))
            v_loss, j_loss = criterions["smpl"](torch.cat((output[:, :3], output_mat, output[:, -16:]), -1),
                                                torch.cat((label[:, :3], target_mat, label[:, -16:]), -1), args.use_gender)
        else:
            losses.update_loss(
                "pose_loss", loss_weight[1]*criterions["mse"](output[:, 3:-16], label[:, 3:-16]))
            v_loss, j_loss = criterions["smpl"](output, label, args.use_gender)
        # shape loss
        losses.update_loss(
            "shape_loss", loss_weight[2]*criterions["mse"](output[:, -16:], label[:, -16:]))
        # joints loss
        losses.update_loss("joints_loss", loss_weight[3]*j_loss)
        # vertices loss
        losses.update_loss("vertices_loss", loss_weight[4]*v_loss)
        # gender loss
        if args.use_gender:
            losses.update_loss(
                "gender_loss", loss_weight[5]*criterions["entropy"](output[:, -1], label[:, -1]))

        loss = losses.calculate_total_loss()
        optimizer.zero_grad()
        #with torch.autograd.detect_anomaly():
        loss.backward()
        nn.utils.clip_grad_value_(model.parameters(), 5)
        optimizer.step()

        metric_logger.update(
            loss=loss.item(), lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters['clips/s'].update(
            batch_size / (time.time() - start_time))
        lr_scheduler.step()
        sys.stdout.flush()


def evaluate(args, model, losses, criterions, data_loader, device, save_path=''):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    per_joint_err = []
    per_vertex_err = []
    shape_err = []
    colors = dict(mmWave=[179, 230, 213], Depth=[208, 163, 230], RGBD=[
                  229, 195, 161], RGB=[159, 175, 216])
    frames = 0

    body_model = SMPLXModel(bm_fname=SMPLX_MODEL_NEUTRAL_PATH,
                            num_betas=16, num_expressions=0, device=device)
    with torch.no_grad():
        for data_dict, target in metric_logger.log_every(data_loader, 100, header):
            if isinstance(data_dict, dict):
                for k, v in data_dict.items():
                    if isinstance(v, torch.Tensor):
                        data_dict[k] = v.to(
                            device, non_blocking=True, dtype=torch.float32)
            else:
                data_dict = data_dict.to(
                    device, non_blocking=True, dtype=torch.float32)
            target = target.to(device, non_blocking=True, dtype=torch.float32)
            output = model(data_dict)
            # translation loss
            losses.update_loss("trans_loss", criterions["mse"](
                output[:, 0:3], target[:, 0:3]))
            # pose loss
            if args.use_6d_pose:
                output_mat = rotation6d_2_rot_mat(output[:, 3:-16])
                target_mat = rodrigues_2_rot_mat(target[:, 3:-16])
                losses.update_loss(
                    "pose_loss", criterions["rot_mat"](output_mat, target_mat))
                output = torch.cat(
                    (output[:, :3], output_mat, output[:, -16:]), -1)
                target = torch.cat(
                    (target[:, :3], target_mat, target[:, -16:]), -1)
                v_loss, j_loss, per_err = criterions["smpl"](
                    output, target, args.use_gender, train=False)
            else:
                losses.update_loss("pose_loss", criterions["mse"](
                    output[:, 3:-16], target[:, 3:-16]))
                v_loss, j_loss, per_err = criterions["smpl"](
                    output, target, args.use_gender, train=False)
            per_joint_err.append(per_err[0])
            per_vertex_err.append(per_err[1])
            shape_err.append(abs(output[:, -16:] - target[:, -16:]))
            # shape loss
            losses.update_loss("shape_loss", criterions["mse"](
                output[:, -16:], target[:, -16:]))
            # joints loss
            losses.update_loss("joints_loss", j_loss)
            # vertices loss
            losses.update_loss("vertices_loss", v_loss)
            # gender loss
            if args.use_gender:
                losses.update_loss("gender_loss", criterions["entropy"](
                    output[:, -1], target[:, -1]))

            loss = losses.calculate_total_loss()

            # could have been padded in distributed setup

            batch_size = target.shape[0]
            metric_logger.update(loss=loss.item())
            metric_logger.meters['loss'].update(loss, n=batch_size)

            if args.visual:
                pred_mesh = body_model(
                    output[:, :3], output[:, 3:-16], output[:, -16:])
                label_mesh = body_model(
                    target[:, :3], target[:, 3:-16], target[:, -16:])
                for i in range(len(target)):
                    yield dict(
                        radar_pcl=dict(
                            # mesh = pcl2box(data_dict['radar'][i][:,:3]) if data_dict['radar'].shape[1] else None,
                            pcl=c2c(
                                data_dict['radar'][i][:, :3]) if data_dict['radar'].shape[1] else None,
                            color=[0, 0.8, 0],
                        ),
                        master_pcl=dict(
                            # mesh = pcl2box(data_dict['master_depth'][i][:,:3]) if data_dict['master_depth'].shape[1] else None,
                            pcl=c2c(data_dict['master_depth'][i][:, :3]
                                    ) if data_dict['master_depth'].shape[1] else None,
                            color=np.asarray([59, 170, 235]) / 255,
                        ),
                        sub_pcl=dict(
                            # mesh = pcl2box(data_dict['sub_depth'][i][:,:3]) if data_dict['sub_depth'].shape[1] else None,
                            pcl=c2c(
                                data_dict['sub_depth'][i][:, :3]) if data_dict['sub_depth'].shape[1] else None,
                            color=np.asarray([245, 157, 86]) / 255,
                        ),
                        pred_smpl=dict(
                            mesh=[c2c(pred_mesh['verts'][i]),
                                  c2c(pred_mesh['faces'])],
                            # color = np.asarray(colors[input_data]) /255
                            color=np.asarray([208, 163, 230]) / 255,
                        ),
                        # label_smpl = dict(
                        #     mesh = [c2c(label_mesh['verts'][i]), c2c(label_mesh['faces'])],
                        #     color = np.asarray([235, 189, 191]) / 255,
                        # )
                    )
                    frames += 1
                    if frames > args.num_frames:
                        return

        print("joints loss:", np.average(
            torch.tensor(losses.loss_dict["joints_loss"])))
        print("vertices loss:", np.average(
            torch.tensor(losses.loss_dict["vertices_loss"])))

        if save_path:
            if not os.path.isdir(save_path):
                os.makedirs(save_path)
            # 22 means the number of joints
            j_err = c2c(torch.vstack(per_joint_err)[:, :22])
            v_err = c2c(torch.vstack(per_vertex_err))
            s_err = c2c(torch.vstack(shape_err))
            np.save(os.path.join(save_path, "per_joint_err"), j_err)
            np.save(os.path.join(save_path, "per_vertex_err"), v_err)
            np.save(os.path.join(save_path, "shape_err"), s_err)
            print("mean joint err (cm):", np.mean(j_err)*100)
            print("mean vertex err (cm):", np.mean(v_err)*100)
            print("max joint err (cm):", np.mean(
                np.max(j_err, axis=1), axis=0)*100)
            print("max vertex err (cm):", np.mean(
                np.max(v_err, axis=1), axis=0)*100)
            print("mean shape err (cm):", np.mean(s_err))
            with open(os.path.join(save_path, "error.txt"), 'w') as f:
                f.write("mean joint error: " + str(np.mean(j_err)*100))
                f.write("\nmean vertex error: " + str(np.mean(v_err)*100))
                f.write("\nmax joint error: " +
                        str(np.mean(np.max(j_err, axis=1), axis=0)*100))
                f.write("\nmax vertex error: " +
                        str(np.mean(np.max(v_err, axis=1), axis=0)*100))
                f.write("\nmean shape error: " + str(np.mean(s_err)))
            # write errors to the excel
            sheet_loc_dict = {
                'lab1': ['A1', 'B1', 'A2', 'B2'],
                'lab2': ['C1', 'D1', 'C2', 'D2'],
                'furnished': ['E1', 'F1', 'E2', 'F2'],
                'rain': ['G1', 'H1', 'G2', 'H2'],
                'smoke': ['I1', 'J1', 'I2', 'J2'],
                'poor_lighting': ['K1', 'L1', 'K2', 'L2'],
                'occlusion': ['M1', 'N1', 'M2', 'N2'],
            }
            sheet_locs = sheet_loc_dict[args.test_scene]
            excel_path = os.path.join(args.output_dir, 'error', 'error.xlsx')
            if not os.path.exists(excel_path):
                work_book = Workbook()
                work_sheet = work_book.active
                work_sheet.title = "Sheet1"
            else:
                work_book = load_workbook(filename=excel_path)
                work_sheet = work_book['Sheet1']
            work_sheet[sheet_locs[0]] = str(np.mean(j_err)*100)
            work_sheet[sheet_locs[1]] = str(np.mean(v_err)*100)
            work_sheet[sheet_locs[2]] = str(
                np.mean(np.max(j_err, axis=1), axis=0)*100)
            work_sheet[sheet_locs[3]] = str(
                np.mean(np.max(v_err, axis=1), axis=0)*100)
            work_book.save(excel_path)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()


def main(args):
    output_dir = args.output_dir
    if output_dir and not os.path.exists(os.path.join(output_dir, 'pth')):
        utils.mkdir(os.path.join(output_dir, 'pth'))

    print(args)
    print("torch version: ", torch.__version__)
    print("torchvision version: ", torchvision.__version__)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    torch.cuda.set_device(args.device)
    device = torch.device('cuda')

    args.inputs = args.inputs.replace(' ', '').split(',')
    args.input_dict = {}
    for m in ['image', 'depth', 'radar']:
        args.input_dict[m] = [i for i in args.inputs if m in i]

    # Data loading code
    dataset = FusionDataset(args)

    train_size = int(0.9 * len(dataset))
    eval_size = len(dataset) - train_size
    dataset_train, dataset_eval = torch.utils.data.random_split(
        dataset, [train_size, eval_size])

    print("Creating data loaders")

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
    data_loader_eval = torch.utils.data.DataLoader(
        dataset_eval, batch_size=args.batch_size, num_workers=args.workers, pin_memory=True)

    print("Creating model")
    Model = getattr(Models, args.model)
    model = Model(args)

    # if torch.cuda.device_count() > 1:
    #     model = nn.DataParallel(model)
    model.to(device)

    bot = TimerBot()
    losses = LossManager(bot)

    mse_criterion = nn.MSELoss()
    smpl_criterion = MoshLoss(device=device, scale=args.normal_scale)
    rot_mat_criterion = GeodesicLoss()
    entropy_criterion = nn.BCEWithLogitsLoss()
    criterions = dict(mse=mse_criterion, smpl=smpl_criterion,
                      rot_mat=rot_mat_criterion, entropy=entropy_criterion)

    lr = args.lr
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # convert scheduler to be per iteration, not per epoch, for warmup that lasts
    # between different epochs
    warmup_iters = args.lr_warmup_epochs * len(data_loader_train)
    lr_milestones = [len(data_loader_train) * m for m in args.lr_milestones]
    lr_scheduler = WarmupMultiStepLR(optimizer, milestones=lr_milestones,
                                     gamma=args.lr_gamma, warmup_iters=warmup_iters, warmup_factor=1e-5)

    model_without_ddp = model

    if args.resume:
        resume = os.path.join(args.resume, 'pth', 'checkpoint.pth')
        checkpoint = torch.load(resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1

    if args.train:
        print("Start training")
        start_time = time.time()

        loss_weight = list(map(float, args.loss_weight.split(",")))

        for epoch in range(args.start_epoch, args.epochs):
            train_one_epoch(args, model, losses, criterions, loss_weight,
                            optimizer, lr_scheduler, data_loader_train, device, epoch)
            losses.calculate_epoch_loss(
                os.path.join(output_dir, "loss/train"), epoch)
            list(evaluate(args, model, losses, criterions, data_loader_eval, device))
            losses.calculate_epoch_loss(
                os.path.join(output_dir, "loss/eval"), epoch)

            if output_dir:
                checkpoint = {
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args}
                utils.save_on_master(checkpoint, os.path.join(
                    output_dir, 'pth', 'checkpoint.pth'))
                if (epoch + 5) % 5 == 0:
                    utils.save_on_master(checkpoint, os.path.join(
                        output_dir, 'pth', 'epoch{}.pth'.format(epoch)))
                    os.system(
                        "cp -r {}/loss {}/backup".format(output_dir, output_dir))

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))

    else:
        loss_weight = list(map(float, args.loss_weight.split(",")))
        data_loader_test = torch.utils.data.DataLoader(
            dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
        print("Start testing")
        save_path = os.path.join(output_dir, "error", args.test_scene)
        gen = evaluate(args, model, losses, criterions,
                       data_loader_test, device, save_path)
        plot = MoshEvaluateStreamPlot()
        if args.save_snapshot:
            snapshot_path = os.path.join(
                args.output_dir, 'snapshot', args.test_scene)
            plot.show(gen, fps=30, save_path=snapshot_path)
        else:
            plot.show(gen, fps=30)


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='DeepFusion Model Training')

    parser.add_argument('--seed', default=35, type=int, help='random seed')
    parser.add_argument('--model', default='DeepFusion',
                        type=str, help='model')
    # input
    parser.add_argument(
        '--data_path', default='/home/nesc525/drivers/7/mmBody', type=str, help='dataset')
    parser.add_argument("--seq_idxes", type=str, default='')
    parser.add_argument('--num_points', default=1024,
                        type=int, help='number of points per frame')
    parser.add_argument('--normal_scale', default=1,
                        type=int, help='normal scale of labels')
    parser.add_argument('--skip_head', default=0, type=int,
                        help='number of skip frames')
    parser.add_argument('--output_dim', default=6, type=int, help='output dim')
    parser.add_argument('--use_6d_pose', default=1,
                        type=int, help='use 6d pose')
    parser.add_argument('--feature_type', default='none',
                        type=str, help='type of features')
    parser.add_argument('--features', default=3,
                        type=int, help='dim of features')
    parser.add_argument('--dataset', default="mmWave", type=str,
                        help='type of input data, mmWave, Depth or mmFusion')
    parser.add_argument('--test_scene', default="lab1", type=str,
                        help='type of test data, test, rain, smoke, night, occlusion, confusion')
    parser.add_argument(
        '--inputs', type=str, default='radar,master_image,master_depth,sub_image,sub_depth', help='input data')
    # P4D
    parser.add_argument('--radius', default=0.7, type=float,
                        help='radius for the ball query')
    parser.add_argument('--nsample', default=32, type=int,
                        help='number of neighbors for the ball query')
    parser.add_argument('--npoint', default=49, type=int,
                        help='number of points for FPS')
    # embedding
    parser.add_argument('--emb_relu', default=False, action='store_true')
    # transformer
    parser.add_argument('--dim', default=1024, type=int,
                        help='transformer dim')
    parser.add_argument('--depth', default=5, type=int,
                        help='transformer depth')
    parser.add_argument('--heads', default=8, type=int,
                        help='transformer head')
    parser.add_argument('--dim_head', default=128, type=int,
                        help='transformer dim for each head')
    parser.add_argument('--mlp_dim', default=2048,
                        type=int, help='transformer mlp dim')
    # training
    parser.add_argument('-b', '--batch_size', default=32, type=int)
    parser.add_argument('--epochs', default=50, type=int,
                        metavar='N', help='number of total epochs to run')
    parser.add_argument('-j', '--workers', default=4, type=int,
                        metavar='N', help='number of data loading workers (default: 16)')
    parser.add_argument('--lr', default=0.0001, type=float,
                        help='initial learning rate')
    parser.add_argument('--momentum', default=0.9,
                        type=float, metavar='M', help='momentum')
    parser.add_argument('--wd', '--weight_decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)', dest='weight_decay')
    parser.add_argument('--lr_milestones', nargs='+',
                        default=[100, 200], type=int, help='decrease lr on milestones')
    parser.add_argument('--lr_gamma', default=0.1, type=float,
                        help='decrease lr by a factor of lr-gamma')
    parser.add_argument('--lr_warmup_epochs', default=10,
                        type=int, help='number of warmup epochs')
    parser.add_argument(
        '--loss_weight', default="1,0.001,0.1,1,1,1", type=str, help='weight of loss')
    parser.add_argument('--use_gender', default=0, type=int, help='use gender')
    parser.add_argument('--device', default=0, type=int, help='cuda device')
    # output
    parser.add_argument('--print_freq', default=100,
                        type=int, help='print frequency')
    parser.add_argument('--num_frames', default=10000,
                        type=int, help='number of test frames')
    parser.add_argument('--output_dir', default='',
                        type=str, help='path where to save')
    # resume
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int,
                        metavar='N', help='start epoch')
    parser.add_argument('--train', dest="train",
                        action="store_true", help='train or test')
    parser.add_argument('--visual', dest="visual",
                        action="store_true", help='visual')
    parser.add_argument('--create_pkl', dest="create_pkl",
                        action="store_true", help='create pkl data')
    parser.add_argument('--use_pkl', dest="use_pkl",
                        action="store_true", help='use pkl data')
    parser.add_argument('--save_snapshot', dest="save_snapshot",
                        action="store_true", help='save snapshot')
    parser.add_argument('--read_orig_img', action="store_true")

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    torch.autograd.set_detect_anomaly(True)
    main(args)
