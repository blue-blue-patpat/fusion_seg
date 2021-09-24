import datetime
import os
import time
import sys
import numpy as np
import torch
import torch.utils.data
from torch import nn
import torchvision
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import cv2

from nn.p4t import utils
from nn.p4t.scheduler import WarmupMultiStepLR
from nn.p4t.datasets.mmbody import MMBody3D
import nn.p4t.modules.model as Models
from message.dingtalk import TimerBot
from visualization.o3d_plot import NNPredLabelStreamPlot
from optitrack.config import marker_lines


def train_one_epoch(model, criterion, optimizer, lr_scheduler, data_loader, device, epoch, print_freq):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value}'))
    metric_logger.add_meter('clips/s', utils.SmoothedValue(window_size=10, fmt='{value:.3f}'))

    header = 'Epoch: [{}]'.format(epoch)
    for xyz_clip, feature_clip, target, _ in metric_logger.log_every(data_loader, print_freq, header):
        start_time = time.time()
        xyz_clip, feature_clip, target = xyz_clip.to(device), feature_clip.to(device), target.to(device)
        output = model(xyz_clip, feature_clip)
        loss = criterion(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_size = xyz_clip.shape[0]
        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters['clips/s'].update(batch_size / (time.time() - start_time))
        lr_scheduler.step()
        sys.stdout.flush()
    return list(metric_logger.meters['loss'].deque)


def evaluate(model, criterion, data_loader, device, visual=False, scale=1):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    rmse_list = []
    with torch.no_grad():
        for xyz_clip, feature_clip, target, _ in metric_logger.log_every(data_loader, 100, header):
            xyz_clip = xyz_clip.to(device, non_blocking=True)
            feature_clip = feature_clip.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            output = model(xyz_clip, feature_clip)
            loss = criterion(output, target)

            # FIXME need to take into account that the datasets
            # could have been padded in distributed setup
            xyz_clip = xyz_clip.cpu().numpy()
            output = output.cpu().numpy()
            target = target.cpu().numpy()
            rmse = np.sqrt(mean_squared_error(target, output)) * scale

            batch_size = xyz_clip.shape[0]
            metric_logger.update(loss=loss.item())
            metric_logger.meters['rmse'].update(rmse, n=batch_size)
            torch.cuda.empty_cache()

            if visual:
                for b, batch in enumerate(xyz_clip):
                    arbe_frame = batch[-1] * scale
                    pred = output[b].reshape(-1, 3) * scale
                    label = target[b].reshape(-1, 3) * scale
                    yield dict(
                        arbe_pcl = dict(
                            pcl = arbe_frame,
                            color = [0,1,0]
                        ),
                        pred = dict(
                            skeleton = pred,
                            lines = marker_lines,
                            colors = np.asarray([[1,0,0]] * len(marker_lines))
                        ),
                        label = dict(
                            skeleton = label,
                            lines = marker_lines,
                            colors = np.asarray([[0,0,1]] * len(marker_lines))
                        )
                    )
            else:
                yield rmse
            print("batch rmse:", rmse)
            rmse_list.append(rmse)
        print("RMSE:", np.mean(rmse_list))

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()


def main(args):
    if args.output_dir:
        utils.mkdir(args.output_dir)

    print(args)
    print("torch version: ", torch.__version__)
    print("torchvision version: ", torchvision.__version__)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device('cuda')

    # Data loading code
    print("Loading data")

    dataset_all = MMBody3D(
            root_path=args.data_path,
            frames_per_clip=args.clip_len,
            step_between_clips=1,
            num_points=args.num_points,
            train=args.train
    )

    train_size = int(0.9 * len(dataset_all))
    test_size = len(dataset_all) - train_size
    dataset_train, dataset_eval = torch.utils.data.random_split(dataset_all, [train_size, test_size])

    print("Creating data loaders")

    data_loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
    data_loader_eval = torch.utils.data.DataLoader(dataset_eval, batch_size=args.batch_size, num_workers=args.workers, pin_memory=True)

    print("Creating model")
    Model = getattr(Models, args.model)
    model = Model(radius=args.radius, nsamples=args.nsamples, spatial_stride=args.spatial_stride,
                  temporal_kernel_size=args.temporal_kernel_size, temporal_stride=args.temporal_stride,
                  emb_relu=args.emb_relu,
                  dim=args.dim, depth=args.depth, heads=args.heads, dim_head=args.dim_head,
                  mlp_dim=args.mlp_dim, num_classes=dataset_all.output_dim)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(device)

    criterion = nn.MSELoss()

    lr = args.lr
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # convert scheduler to be per iteration, not per epoch, for warmup that lasts
    # between different epochs
    warmup_iters = args.lr_warmup_epochs * len(data_loader_train)
    lr_milestones = [len(data_loader_train) * m for m in args.lr_milestones]
    lr_scheduler = WarmupMultiStepLR(optimizer, milestones=lr_milestones, gamma=args.lr_gamma, warmup_iters=warmup_iters, warmup_factor=1e-5)

    model_without_ddp = model

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1

    if args.train:
        print("Start training")
        start_time = time.time()
        fig = plt.figure()
        loss_list = []
        rmse_list = []
        bot =TimerBot()
        dingbot = True
        for epoch in range(args.start_epoch, args.epochs):
            loss = train_one_epoch(model, criterion, optimizer, lr_scheduler, data_loader_train, device, epoch, args.print_freq)
            loss_list += loss

            rmse_gen = evaluate(model, criterion, data_loader_eval, device)
            rmse_list.append(np.mean(list(rmse_gen)))

            fig.add_subplot(1, 1, 1).plot(loss_list)
            fig.canvas.draw()
            img = cv2.cvtColor(np.asarray(fig.canvas.buffer_rgba()), cv2.COLOR_RGBA2BGR)
            if args.output_dir:
                cv2.imwrite(os.path.join(args.output_dir, 'loss.png'), img)

            if dingbot:
                bot.add_md("tran_mmbody", "【LOSS】 \n ![img]({}) \n 【RMSE】\n epoch={}, rmse={}".format(bot.img2b64(img), epoch, rmse_list[-1]))
                bot.enable()
            
            if args.output_dir:
                checkpoint = {
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args}
                utils.save_on_master(
                    checkpoint,
                    os.path.join(args.output_dir, 'model_{}.pth'.format(epoch)))
                utils.save_on_master(
                    checkpoint,
                    os.path.join(args.output_dir, 'checkpoint.pth'))

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))
        print('RMSE {}'.format(rmse_list))
        
    else:
        data_loader_test = torch.utils.data.DataLoader(dataset_all, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
        plot = NNPredLabelStreamPlot()
        print("Start testing")
        gen = evaluate(model, criterion, data_loader_test, device, visual=True, scale=dataset_all.normal_scale)
        plot.show(gen, fps=15)

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='P4Transformer Model Training')

    parser.add_argument('--data_path', default='/media/nesc525/perple', type=str, help='dataset')
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--model', default='P4Transformer', type=str, help='model')
    # input
    parser.add_argument('--clip_len', default=5, type=int, metavar='N', help='number of frames per clip')
    parser.add_argument('--num_points', default=1024, type=int, metavar='N', help='number of points per frame')
    # P4D
    parser.add_argument('--radius', default=0.7, type=float, help='radius for the ball query')
    parser.add_argument('--nsamples', default=32, type=int, help='number of neighbors for the ball query')
    parser.add_argument('--spatial_stride', default=32, type=int, help='spatial subsampling rate')
    parser.add_argument('--temporal_kernel_size', default=3, type=int, help='temporal kernel size')
    parser.add_argument('--temporal_stride', default=1, type=int, help='temporal stride')
    # embedding
    parser.add_argument('--emb_relu', default=False, action='store_true')
    # transformer
    parser.add_argument('--dim', default=1024, type=int, help='transformer dim')
    parser.add_argument('--depth', default=5, type=int, help='transformer depth')
    parser.add_argument('--heads', default=8, type=int, help='transformer head')
    parser.add_argument('--dim_head', default=128, type=int, help='transformer dim for each head')
    parser.add_argument('--mlp_dim', default=2048, type=int, help='transformer mlp dim')
    # training
    parser.add_argument('-b', '--batch_size', default=14, type=int)
    parser.add_argument('--epochs', default=50, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('-j', '--workers', default=10, type=int, metavar='N', help='number of data loading workers (default: 16)')
    parser.add_argument('--lr', default=0.01, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('--wd', '--weight_decay', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)', dest='weight_decay')
    parser.add_argument('--lr_milestones', nargs='+', default=[20, 30], type=int, help='decrease lr on milestones')
    parser.add_argument('--lr_gamma', default=0.1, type=float, help='decrease lr by a factor of lr-gamma')
    parser.add_argument('--lr_warmup_epochs', default=10, type=int, help='number of warmup epochs')
    # output
    parser.add_argument('--print_freq', default=10, type=int, help='print frequency')
    parser.add_argument('--output_dir', default='', type=str, help='path where to save')
    # resume
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='start epoch')
    parser.add_argument('--train', dest="train", action="store_true", help='train or test')

    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)
