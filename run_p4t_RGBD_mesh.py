import datetime
from json import dump
import os
import time
import sys
import numpy as np
from numpy.lib import average
import torch
import torch.utils.data
from torch import nn
import torchvision
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import cv2
from human_body_prior.models.vposer_model import VPoser
from minimal.models import KinematicModel, KinematicPCAWrapper
from minimal.config import SMPL_MODEL_1_0_MALE_PATH, SMPL_MODEL_1_0_PATH, SMPL_MODEL_1_0_NEUTRAL_PATH, VPOSER_DIR
from human_body_prior.tools.model_loader import load_model
from nn.p4t import tools
from nn.p4t import utils
from nn.p4t.modules.geodesic_loss import GeodesicLoss
from nn.p4t.scheduler import WarmupMultiStepLR
from nn.datasets.depth_mesh import DepthMesh3D2,DepthMesh3D
from nn.SMPL.smpl_layer import SMPLLoss
import nn.p4t.modules.model as Models
from message.dingtalk import TimerBot
from visualization.mesh_plot import MeshEvaluateStreamPlot
from visualization.utils import o3d_mesh, o3d_pcl, o3d_plot, o3d_smpl_mesh
from nn.p4t.modules.loss import LossManager
from visualization.mesh_plot import pcl2box

torch.cuda.set_device(0)
device = torch.device('cuda')

def train_one_epoch(model, losses, criterions, loss_weight, optimizer, lr_scheduler, data_loader, device, epoch, print_freq, output_dim, use_gender):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value}'))
    metric_logger.add_meter('clips/s', utils.SmoothedValue(window_size=10, fmt='{value:.3f}'))

    header = 'Epoch: [{}]'.format(epoch)

    for clip, target in metric_logger.log_every(data_loader, print_freq, header):
        start_time = time.time()
        clip, target = clip.to(device), target.to(device)
        output = model(clip)
        batch_size = clip.shape[0]
        # translation loss
        losses.update_loss("trans_loss", loss_weight[0]*criterions["mse"](output[:,0:3], target[:,0:3]))
        # pose loss
        if output_dim >= 157:
            output_mat = tools.rotation6d_2_rot_mat(output[:,3:-11])
            target_mat = tools.rodrigues_2_rot_mat(target[:,3:-11])
            losses.update_loss("pose_loss", loss_weight[1]*criterions["rot_mat"](output_mat, target_mat))
            v_loss, j_loss = criterions["smpl"](torch.cat((output[:,:3], output_mat, output[:,-11:]), -1), torch.cat((target[:,:3], target_mat, target[:,-11:]), -1), use_gender)
        else:
            losses.update_loss("pose_loss", loss_weight[1]*criterions["mse"](output[:,3:-11],target[:,3:-11]))
            v_loss, j_loss = criterions["smpl"](output, target, use_gender)
        # shape loss
        losses.update_loss("shape_loss", loss_weight[2]*criterions["mse"](output[:,-11:-1], target[:,-11:-1]))
        # vertices loss
        losses.update_loss("vertices_loss", loss_weight[3]*v_loss)
        # joints loss
        losses.update_loss("joints_loss", loss_weight[4]*j_loss)
        # gender loss
        if use_gender:
            losses.update_loss("gender_loss", loss_weight[5]*criterions["entropy"](output[:,-1], target[:,-1]))

        loss = losses.calculate_total_loss()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters['clips/s'].update(batch_size / (time.time() - start_time))
        lr_scheduler.step()
        sys.stdout.flush()

def evaluate(model, losses, criterions, loss_weight, data_loader, device, output_dim, use_gender, visual=False, output_path=''):
    model.eval()
    vp, _ = load_model(VPOSER_DIR, model_code=VPoser,
                        remove_words_in_model_weights='vp_model.',
                        disable_grad=True)
    vp = vp.to('cuda')
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    gender_acc = []
    per_joint_err = []
    per_vertex_err = []
    # aim_joints_loss = 0.2187
    # aim_vertices_loss = 0.241
    if use_gender:
        smpl_m_model = KinematicPCAWrapper(KinematicModel().init_from_file(SMPL_MODEL_1_0_MALE_PATH, compute_mesh=False))
        smpl_f_model = KinematicPCAWrapper(KinematicModel().init_from_file(SMPL_MODEL_1_0_PATH, compute_mesh=False))
    else:
        smpl_m_model = smpl_f_model = KinematicPCAWrapper(KinematicModel().init_from_file(SMPL_MODEL_1_0_NEUTRAL_PATH, compute_mesh=False))
    with torch.no_grad():
        for clip, target, _ in metric_logger.log_every(data_loader, 100, header):
            clip = clip.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            output = model(clip)
            
            # translation loss
            losses.update_loss("trans_loss", loss_weight[0]*criterions["mse"](output[:,0:3], target[:,0:3]))
            # pose loss
            if output_dim >= 157:
                output_mat = tools.rotation6d_2_rot_mat(output[:,3:-11])
                target_mat = tools.rodrigues_2_rot_mat(target[:,3:-11])
                losses.update_loss("pose_loss", loss_weight[1]*criterions["rot_mat"](output_mat, target_mat))
                v_loss, j_loss, per_loss = criterions["smpl"](torch.cat((output[:,:3], output_mat, output[:,-11:]), -1), torch.cat((target[:,:3], target_mat, target[:,-11:]), -1), use_gender, train=False)
            else:
                losses.update_loss("pose_loss", loss_weight[1]*criterions["mse"](output[:,3:-11],target[:,3:-11]))
                v_loss, j_loss, per_loss = criterions["smpl"](output, target, use_gender, train=False)
            per_joint_err.append(per_loss[0])
            per_vertex_err.append(per_loss[1])
            # shape loss
            losses.update_loss("shape_loss", loss_weight[2]*criterions["mse"](output[:,-11:-1], target[:,-11:-1]))
            # joints loss
            losses.update_loss("joints_loss", loss_weight[3]*j_loss)
            # vertices loss
            losses.update_loss("vertices_loss", loss_weight[4]*v_loss)
            # gender loss
            if use_gender:
                losses.update_loss("gender_loss", loss_weight[5]*criterions["entropy"](output[:,-1], target[:,-1]))

            loss = losses.calculate_total_loss()

            if output_dim >= 158:
                output = tools.rotation6d_2_rodrigues(output)

            # could have been padded in distributed setup
            clip = clip.cpu().numpy()
            output = output.cpu().numpy()
            target = target.cpu().numpy()

            batch_size = clip.shape[0]
            metric_logger.update(loss=loss.item())
            metric_logger.meters['loss'].update(loss, n=batch_size)

            gender_pred = np.where(output[:,-1]>0.5, 1, 0)
            acc = np.mean(np.equal(gender_pred, target[:,-1]))
            gender_acc.append(acc)
            # if losses.loss_dict["joints_loss"][-1].cpu().numpy()>aim_joints_loss:
            #     time.sleep(20)
            # elif losses.loss_dict["vertices_loss"][-1].cpu().numpy() > aim_vertices_loss:
            #     time.sleep(20)
            if visual:    
                print("batch gender acc: {}%".format(acc * 100))
                print("batch joints loss:", losses.loss_dict["joints_loss"][-1].cpu().numpy())
                print("batch vertices loss:", losses.loss_dict["vertices_loss"][-1].cpu().numpy())
                for b, batch in enumerate(clip):
                    arbe_frame = batch[-1][:,:3]
                    pred = output[b]
                    pred[3+22*3:3+24*3] = 0
                    label = target[b]
                    print(_[2])
                    poseSMPL = torch.from_numpy(np.array([pred[6:69]])).type(torch.float).to('cuda')
                    poZ = vp.encode(poseSMPL).mean
                    pred[6:69] = vp.decode(poZ)['pose_body'].contiguous().reshape(poseSMPL.shape[1]).cpu().numpy()
                    pred[3+22*3:3+24*3] = 0
                    label[3+22*3:3+24*3] = 0
                    # restore to origin size, except gender
                    yield dict(
                        radar_pcl = dict(
                            mesh = pcl2box(arbe_frame),
                            color = np.asarray([255, 181, 74]) / 255,
                        ),
                        pred_smpl = dict(
                            params = pred[:-1],
                            color = np.asarray([229, 195, 161]) / 255,
                            #model = smpl_f_model,
                            model = smpl_m_model,
                        ),
                        label_smpl = dict(
                            params = label[:-1],
                            color = np.asarray([235, 189, 191]) / 255,
                            model=smpl_m_model if label[-1] > 0.5 else smpl_f_model,
                        )
                    )
        print("gender acc:", np.mean(gender_acc))
        print("joints loss:", np.average(torch.tensor(losses.loss_dict["joints_loss"])))
        print("vertices loss:", np.average(torch.tensor(losses.loss_dict["vertices_loss"])))
        if not os.path.isdir(os.path.join(output_path, "loss/test")):
            os.makedirs(os.path.join(output_path, "loss/test"))
        j_err = torch.stack(per_joint_err).cpu().numpy()[:,:24]
        v_err = torch.stack(per_vertex_err).cpu().numpy()
        np.save(os.path.join(output_path, "loss/test/per_joint_err"), j_err)
        np.save(os.path.join(output_path, "loss/test/per_vertex_err"), v_err)
        print("mean joint err:", np.mean(j_err)*100)
        print("mean vertex err:", np.mean(v_err)*100)
        print("max joint err:", np.mean(np.max(j_err, axis=1), axis=0)*100)
        print("max vertex err:", np.mean(np.max(v_err, axis=1), axis=0)*100)
        with open(os.path.join(output_path, "loss/test/error.txt"), 'w') as f:
            f.write("gender_acc: " + str(np.mean(gender_acc)))
            f.write("\nmean joint error: " + str(np.mean(j_err)*100))
            f.write("\nmean vertex error: " + str(np.mean(v_err)*100))
            f.write("\nmax joint error: " + str(np.mean(np.max(j_err, axis=1), axis=0)*100))
            f.write("\nmax vertex error: " + str(np.mean(np.max(v_err, axis=1), axis=0)*100))

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
    

    # Data loading code
    print("Loading data")

    dataset = DepthMesh3D2(
            root_path=args.data_path,
            frames_per_clip=args.clip_len,
            step_between_clips=1,
            num_points=args.num_points,
            normal_scale=args.normal_scale,
            skip_head=args.skip_head,
            output_dim=args.output_dim,
            train=args.train,
            device=args.device
    )

    train_size = int(0.9 * len(dataset))
    eval_size = len(dataset) - train_size
    dataset_train, dataset_eval = torch.utils.data.random_split(dataset, [train_size, eval_size])

    print("Creating data loaders")

    data_loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
    data_loader_eval = torch.utils.data.DataLoader(dataset_eval, batch_size=args.batch_size, num_workers=args.workers, pin_memory=True)

    print("Creating model")
    Model = getattr(Models, args.model)
    model = Model(radius=args.radius, nsamples=args.nsamples, spatial_stride=args.spatial_stride,
                  temporal_kernel_size=args.temporal_kernel_size, temporal_stride=args.temporal_stride,
                  emb_relu=args.emb_relu,
                  dim=args.dim, depth=args.depth, heads=args.heads, dim_head=args.dim_head,
                  mlp_dim=args.mlp_dim, output_dim=dataset.output_dim)

    #if torch.cuda.device_count() > 1:
        #model = nn.DataParallel(model)
    model.to(device)
    
    losses = LossManager()

    mse_criterion = nn.MSELoss()
    smpl_criterion = SMPLLoss(device=device, scale=args.normal_scale)
    rm_criterion = GeodesicLoss()
    entropy_criterion = nn.BCEWithLogitsLoss()
    criterions = dict(mse=mse_criterion, smpl=smpl_criterion, rot_mat=rm_criterion, entropy=entropy_criterion)

    lr = args.lr
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

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

        bot =TimerBot()
        loss_weight = list(map(float, args.loss_weight.split(",")))

        for epoch in range(args.start_epoch, args.epochs):
            train_one_epoch(model, losses, criterions, loss_weight, optimizer, lr_scheduler, data_loader_train, device, epoch, args.print_freq, args.output_dim, args.use_gender)
            losses.calculate_epoch_loss(os.path.join(args.output_dir,"loss/train"), epoch, bot)
            list(evaluate(model, losses, criterions, loss_weight, data_loader_eval, device, args.output_dim, use_gender=args.use_gender))
            losses.calculate_epoch_loss(os.path.join(args.output_dir,"loss/eval"), epoch, bot)

            if args.output_dir:
                checkpoint = {
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args}
                utils.save_on_master(
                    checkpoint,
                    os.path.join(args.output_dir, 'checkpoint.pth'))
                if (epoch + 5) % 5 == 0:
                    utils.save_on_master(
                        checkpoint,
                        os.path.join(args.output_dir, 'epoch{}.pth'.format(epoch)))
                    os.system("cp -r {}/loss {}/backup".format(args.output_dir, args.output_dir))

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))
        
    else:
        loss_weight = list(map(float, args.loss_weight.split(",")))
        data_loader_test = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
        print("Start testing")
        gen = evaluate(model, losses, criterions, loss_weight, data_loader_test, device, output_dim=args.output_dim, visual=True, use_gender=args.use_gender, output_path=args.output_dir)
        plot = MeshEvaluateStreamPlot(save_path="/home/nesc525/drivers/4/rgbd_obj")
        plot.show(gen, fps=100)
        losses.calculate_test_loss(os.path.join(args.output_dir,"loss/test"))
        single = False
        if single:
            data = next(gen)
            o3d_plot([o3d_pcl(**data["radar_pcl"]), o3d_smpl_mesh(**data["pred_smpl"]), o3d_smpl_mesh(**data["label_smpl"])])

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='P4Transformer Model Training')

    parser.add_argument('--data_path', default='/home/nesc525/drivers/2,/home/nesc525/drivers/1,/home/nesc525/drivers/3', type=str, help='dataset')
    parser.add_argument('--seed', default=35, type=int, help='random seed')
    parser.add_argument('--model', default='P4Transformer', type=str, help='model')
    # input
    parser.add_argument('--clip_len', default=5, type=int, metavar='N', help='number of frames per clip')
    parser.add_argument('--num_points', default=4096, type=int, metavar='N', help='number of points per frame')
    parser.add_argument('--normal_scale', default=1, type=int, metavar='N', help='normal scale of labels')
    parser.add_argument('--skip_head', default=0, type=int, metavar='N', help='number of skip frames')
    parser.add_argument('--new_gmm', action='store_true', help='new gmm')
    parser.add_argument('--output_dim', default=158, type=int, metavar='N', help='output dim')
    parser.add_argument('--device', default='sub1', type=str, help='device of kinect')
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
    parser.add_argument('-b', '--batch_size', default=32, type=int)
    parser.add_argument('--epochs', default=100, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('-j', '--workers', default=10, type=int, metavar='N', help='number of data loading workers (default: 16)')
    parser.add_argument('--lr', default=0.0001, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('--wd', '--weight_decay', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)', dest='weight_decay')
    parser.add_argument('--lr_milestones', nargs='+', default=[100,200], type=int, help='decrease lr on milestones')
    parser.add_argument('--lr_gamma', default=0.1, type=float, help='decrease lr by a factor of lr-gamma')
    parser.add_argument('--lr_warmup_epochs', default=10, type=int, help='number of warmup epochs')
    parser.add_argument('--loss_weight', default="1,0.001,0.1,1,1,1", type=str, help='weight of loss')
    parser.add_argument('--use_gender', default=1, type=int, help='use gender')
    # output
    parser.add_argument('--print_freq', default=10, type=int, help='print frequency')
    parser.add_argument('--output_dir', default='', type=str, help='path where to save')
    # resume
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='start epoch')
    parser.add_argument('--train', default=False, dest="train", action="store_true", help='train or test')

    args = parser.parse_args()

    return args

if __name__ == "__main__":

    args = parse_args()
    main(args)