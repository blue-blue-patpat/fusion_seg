from __future__ import print_function
import os
import argparse
import torch
import sys
import matplotlib.pyplot as plt
import cv2
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR,'nn/paconv'))
from util.util import cal_loss, IOStream, load_cfg_from_cfg_file, merge_cfg_from_list
import sklearn.metrics as metrics
from tensorboardX import SummaryWriter
import random
from mmbody2 import MMBody3D
from optitrack.config import marker_lines
from visualization.o3d_plot import NNPredLabelStreamPlot


def get_parser():
    parser = argparse.ArgumentParser(description='3D Object Classification')
    parser.add_argument('--config', type=str, default='config/dgcnn_paconv.yaml', help='config file')
    parser.add_argument('opts', help='see config/dgcnn_paconv.yaml for all options', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    assert args.config is not None
    cfg = load_cfg_from_cfg_file(args.config)
    if args.opts is not None:
        cfg = merge_cfg_from_list(cfg, args.opts)
    cfg['manual_seed'] = cfg.get('manual_seed', 0)
    cfg['workers'] = cfg.get('workers', 6)
    return cfg


def _init_():
    if not os.path.exists('ignoredata/paconv/checkpoints'):
        os.makedirs('ignoredata/paconv/checkpoints')
    if not os.path.exists('ignoredata/paconv/checkpoints/'+args.exp_name):
        os.makedirs('ignoredata/paconv/checkpoints/'+args.exp_name)

    if not args.eval:  # backup the running files
        os.system('cp main.py ignoredata/paconv/checkpoints' + '/' + args.exp_name + '/' + 'main.py.backup')
        os.system('cp util/PAConv_util.py ignoredata/paconv/checkpoints' + '/' + args.exp_name + '/' + 'PAConv_util.py.backup')
        os.system('cp util/data_util.py ignoredata/paconv/checkpoints' + '/' + args.exp_name + '/' + 'data_util.py.backup')
        if args.arch == 'dgcnn':
            os.system('cp model/DGCNN_PAConv.py ignoredata/paconv/checkpoints' + '/' + args.exp_name + '/' + 'DGCNN_PAConv.py.backup')
        elif args.arch == 'pointnet':
            os.system('cp model/PointNet_PAConv.py ignoredata/paconv/checkpoints' + '/' + args.exp_name + '/' + 'PointNet_PAConv.py.backup')

    global writer
    writer = SummaryWriter('ignoredata/paconv/checkpoints/' + args.exp_name)


# weight initialization:
def weight_init(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
    elif isinstance(m, torch.nn.Conv2d):
        torch.nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
    elif isinstance(m, torch.nn.Conv1d):
        torch.nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
    elif isinstance(m, torch.nn.BatchNorm2d):
        torch.nn.init.constant_(m.weight, 1)
        torch.nn.init.constant_(m.bias, 0)
    elif isinstance(m, torch.nn.BatchNorm1d):
        torch.nn.init.constant_(m.weight, 1)
        torch.nn.init.constant_(m.bias, 0)


def train(args, io):
    dataset_all = MMBody3D(root_path='/media/nesc525/perple2',frames_per_clip=2,step_between_clips=1,num_points=args.num_points,train=True)
    train_size = int(0.9 * len(dataset_all))
    test_size = len(dataset_all) - train_size
    dataset_train, dataset_test = torch.utils.data.random_split(dataset_all, [train_size, test_size])
    train_loader2 = DataLoader(dataset_train,num_workers=args.workers, batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader2 = DataLoader(dataset_test,num_workers=args.workers, batch_size=args.batch_size, shuffle=True, drop_last=True)

    device = torch.device("cuda" if args.cuda else "cpu")
    if args.arch == 'dgcnn':
        from model.DGCNN_PAConv_copy_wF import PAConv
        model = PAConv(args).to(device)
    elif args.arch == 'pointnet':
        from model.PointNet_PAConv import PAConv
        model = PAConv(args).to(device)
    else:
        raise Exception("Not implemented")

    io.cprint(str(model))

    model.apply(weight_init)
    model = nn.DataParallel(model)
    print("Let's use", torch.cuda.device_count(), "GPUs!")

    print("Use SGD")
    opt = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(opt, args.epochs, eta_min=args.lr/100)

    #criterion = cal_loss
    criterion = nn.MSELoss()
    best_test_rmse = 1
    fig = plt.figure()
    fig_r = plt.figure()
    f = open('ignoredata/paconv/checkpoints/%s/LOSS.txt' % args.exp_name,'w')
    f_r = open('ignoredata/paconv/checkpoints/%s/RMSE.txt' % args.exp_name,'w')
    loss_list = []
    RMSE_L = []
    for epoch in range(args.epochs):
        scheduler.step()
        ####################
        # Train
        ####################
        train_loss = 0.0
        count = 0.0
        model.train()
        train_pred = []
        train_true = []
        for data, label in train_loader2:
            #print(data.shape)[16, 1024, 3]
            #print(label.shape)[16, 1]
            data, label = data.to(device), label.to(device).squeeze()
            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]
            opt.zero_grad()
            logits = model(data)
            #print(logits.shape)[16,40]
            #label = torch.zeros_like(logits).scatter(1, label.view(-1, 1), 1)
            logits = logits.float()
            label = label.float()
            loss = criterion(logits, label)
            #print(loss):tensor(4.6178, device='cuda:0', grad_fn=<NegBackward>
            loss.backward()
            opt.step()
            preds = logits
            count += batch_size
            train_loss += loss.item() * batch_size
            train_true.append(label.cpu().numpy())
            train_pred.append(preds.detach().cpu().numpy())
            loss_list.append(loss.item())
        train_true = np.concatenate(train_true)
        train_pred = np.concatenate(train_pred)
        #train_acc = metrics.accuracy_score(train_true, train_pred)
        outstr = 'Train %d, loss: %.6f, ' % (epoch, train_loss * 1.0 / count)
        io.cprint(outstr)
        LOSSS = train_loss * 1.0 / count
        fig.add_subplot(1,1,1).plot(loss_list)
        fig.canvas.draw()
        img = cv2.cvtColor(np.asarray(fig.canvas.buffer_rgba()),cv2.COLOR_RGBA2BGR)
        cv2.imwrite('ignoredata/paconv/checkpoints/%s/loss.png' % args.exp_name,img)
        f.write(str(LOSSS)+'\n')
        writer.add_scalar('loss_train', train_loss * 1.0 / count, epoch + 1)
        torch.cuda.empty_cache()

        ####################
        # Test
        ####################
        test_loss = 0.0
        count = 0.0
        model.eval()
        test_pred = []
        test_true = []
        rmse_list = []
        for data, label in test_loader2:
            data, label = data.to(device), label.to(device).squeeze()
            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]
            logits = model(data)           
            #label = torch.zeros_like(logits).scatter(1, label.view(-1, 1), 1)
            logits = logits.float()
            label = label.float()
            loss = criterion(logits, label)
            preds = logits
            count += batch_size
            test_loss += loss.item() * batch_size
            output = preds.cpu().detach().numpy()
            target = label.cpu().numpy()
            rmse = np.sqrt(mean_squared_error(target,output))*1.2
            rmse_list.append(rmse)
            test_true.append(label.cpu().numpy())
            test_pred.append(preds.detach().cpu().numpy())
        test_true = np.concatenate(test_true)
        test_pred = np.concatenate(test_pred)
        #test_acc = metrics.accuracy_score(test_true, test_pred)
        outstr = 'Test %d, loss: %.6f,' % (epoch, test_loss * 1.0 / count)
        io.cprint(outstr)
        RMSE_L.append(np.mean(rmse_list))
        fig_r.add_subplot(1,1,1).plot(RMSE_L)
        fig_r.canvas.draw()
        img_r = cv2.cvtColor(np.asarray(fig_r.canvas.buffer_rgba()),cv2.COLOR_RGBA2BGR)
        cv2.imwrite('ignoredata/paconv/checkpoints/%s/RMSE.png' % args.exp_name,img_r)
        f_r.write(str(np.mean(rmse_list))+'\n')
        writer.add_scalar('loss_test', test_loss * 1.0 / count, epoch + 1)

        if np.mean(rmse_list) <= best_test_rmse:
            best_test_rmse = np.mean(rmse_list)
            io.cprint('Best RMSE:%.6f' % best_test_rmse)
            torch.save(model.state_dict(), 'ignoredata/paconv/checkpoints/%s/best_model.t7' % args.exp_name)


def test(args, io):
    dataset_test = MMBody3D(root_path='/media/nesc525/perple2',frames_per_clip=2,step_between_clips=1,num_points=args.num_points,train=False)
    test_loader2 = DataLoader(dataset_test,num_workers=args.workers, batch_size=args.batch_size, shuffle=True, drop_last=True)
    criterion = nn.MSELoss()

    
    device = torch.device("cuda" if args.cuda else "cpu")

    # Try to load models:
    if args.arch == 'dgcnn':
        from model.DGCNN_PAConv_copy_wF import PAConv
        model = PAConv(args).to(device)
    elif args.arch == 'pointnet':
        from model.PointNet_PAConv import PAConv
        model = PAConv(args).to(device)
    else:
        raise Exception("Not implemented")

    io.cprint(str(model))

    model = nn.DataParallel(model)
    #需要修改模型位置
    model.load_state_dict(torch.load("ignoredata/paconv/checkpoints/20211003_wF6/best_model.t7"))
    #model.load_state_dict(torch.load("/home/nesc525/chen/3DSVC/nn/paconv/20210919-11:13/best_model.t7"))
    model = model.eval()
    count = 0.0
    test_true = []
    test_pred = []
    rmse_list = []
    test_loss = 0.0
    for data, label in test_loader2:
        data, label = data.to(device), label.to(device).squeeze()
        data = data.permute(0, 2, 1)
        batch_size = data.size()[0]
        with torch.no_grad():
            logits = model(data)
        logits = logits.float()
        label = label.float()
        loss = criterion(logits, label)
        count += batch_size
        test_loss += loss.item() * batch_size            
        preds = logits.detach().cpu().numpy()
        rmse = np.sqrt(mean_squared_error(label.cpu(),preds))
        rmse_list.append(rmse)
        labels = label.cpu().numpy()
        test_true.append(label)
        test_pred.append(preds)
        data = data.cpu().numpy()
        

        scale=1.2
        for b, batch in enumerate(data):
            arbe_frame = batch[:3,:].transpose(1,0) * scale
            pred = preds[b].reshape(-1, 3) * scale
            label = labels[b].reshape(-1, 3) * scale
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

    outstr = 'Test %d, loss: %.6f, test RMSE: %.6f,' % (1, test_loss * 1.0 / count, np.mean(rmse_list))
    io.cprint(outstr)

def test2(args, io):
    dataset_test = MMBody3D(root_path='/media/nesc525/perple2',frames_per_clip=2,step_between_clips=1,num_points=args.num_points,train=False)
    test_loader2 = DataLoader(dataset_test,num_workers=args.workers, batch_size=args.batch_size, shuffle=True, drop_last=True)
    criterion = nn.MSELoss()

    
    device = torch.device("cuda" if args.cuda else "cpu")

    # Try to load models:
    if args.arch == 'dgcnn':
        from model.DGCNN_PAConv_copy_wF import PAConv
        model = PAConv(args).to(device)
    elif args.arch == 'pointnet':
        from model.PointNet_PAConv import PAConv
        model = PAConv(args).to(device)
    else:
        raise Exception("Not implemented")

    io.cprint(str(model))

    model = nn.DataParallel(model)
    #需要修改模型位置
    model.load_state_dict(torch.load("ignoredata/paconv/checkpoints/20211003-wF6/best_model.t7"))
    #model.load_state_dict(torch.load("/home/nesc525/chen/3DSVC/nn/paconv/20210919-11:13/best_model.t7"))
    model = model.eval()
    count = 0.0
    test_true = []
    test_pred = []
    rmse_list = []
    test_loss = 0.0
    for data, label in test_loader2:
        data, label = data.to(device), label.to(device).squeeze()
        data = data.permute(0, 2, 1)
        batch_size = data.size()[0]
        with torch.no_grad():
            logits = model(data)
        logits = logits.float()
        label = label.float()
        loss = criterion(logits, label)
        count += batch_size
        test_loss += loss.item() * batch_size            
        preds = logits.detach().cpu().numpy()
        rmse = np.sqrt(mean_squared_error(label.cpu(),preds))
        rmse_list.append(rmse)
        labels = label.cpu().numpy()
        test_true.append(label)
        test_pred.append(preds)
        data = data.cpu().numpy()
    outstr = 'Test %d, loss: %.6f, test RMSE: %.6f,' % (1, test_loss * 1.0 / count, np.mean(rmse_list))
    io.cprint(outstr)

if __name__ == "__main__":
    args = get_parser()
    _init_()
    if not args.eval:
        io = IOStream('ignoredata/paconv/checkpoints/' + args.exp_name + '/%s_train.log' % (args.exp_name))
    else:
        io = IOStream('ignoredata/paconv/checkpoints/' + args.exp_name + '/%s_test.log' % (args.exp_name))
    io.cprint(str(args))

    if args.manual_seed is not None:
        random.seed(args.manual_seed)
        np.random.seed(args.manual_seed)
        torch.manual_seed(args.manual_seed)

    args.cuda = not args.no_cuda and torch.cuda.is_available()

    if args.cuda:
        io.cprint('Using GPU')
        if args.manual_seed is not None:
            torch.cuda.manual_seed(args.manual_seed)
            torch.cuda.manual_seed_all(args.manual_seed)
    else:
        io.cprint('Using CPU')

    if not args.eval:
        train(args, io)
    else:
        io.cprint('Start Testing')
        if args.get('visual',True):
            plot = NNPredLabelStreamPlot()
            gen = test(args, io)
            plot.show(gen, fps=15)
        else:
            test2(args, io)
