import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import json

from nn.p4t import tools

class Loss():
    def __init__(self, criterion, weight) -> None:
        super(Loss).__init__()
        self.criterion = criterion
        self.weight = weight
        self.train_loss = []
        self.eval_loss = []
        
    def _update_loss(self, pred, label, train):
        if train:
            self.train_loss.append(self.weight * self.criterion(pred, label))
        else:
            self.eval_loss.append(self.weight * self.criterion(pred, label))

class TransLoss(Loss):
    def __init__(self, criterion, weight, *args) -> None:
        super(Loss).__init__(criterion, weight)

    def update_loss(self, output, target, train):
        self._update_loss(output[:,3:-11], target[:,3:-11], train)

class PoseLoss(Loss):
    def __init__(self, criterion, weight, *args) -> None:
        super(Loss).__init__(criterion, weight)

    def update_loss(self, output, target, train):
        output_mat = tools.rotation6d_2_rot_mat(output[:,3:-11])
        target_mat = tools.rodrigues_2_rot_mat(target[:,3:-11])
        self._update_loss(output_mat, target_mat, train)

class ShapeLoss(Loss):
    def __init__(self, criterion, weight, *args) -> None:
        super(Loss).__init__(criterion, weight)

    def update_loss(self, output, target, train):
        self._update_loss(output[:,-11:-1], target[:,-11:-1], train)

class VerticesLoss(Loss):
    def __init__(self, criterion, weight, use_gender) -> None:
        super(Loss).__init__(criterion, weight)
        self.use_gender = use_gender

    # def update_loss(self, output, target, train):
    #     output_mat = tools.rotation6d_2_rot_mat(output[:,3:-11])
    #     target_mat = tools.rodrigues_2_rot_mat(target[:,3:-11])
    #     v_loss, j_loss = self.criterion[1](torch.cat((output[:,:3], output_mat, output[:,-11:]), -1), torch.cat((target[:,:3], target_mat, target[:,-11:]), -1), self.use_gender)
    #     self._update_loss(output[:,-11:-1], target[:,-11:-1], train)
        
class LossManager():
    def __init__(self) -> None:
        super(LossManager).__init__()
        self.loss_dict = {}
        self.batch_loss = []
<<<<<<< HEAD
=======
        self.epoch_loss = []
        self.batch_size = 0
        self.len_train_batch = 0
>>>>>>> a5cd3d0f9a5d98072aa5d624318605e6cc6b8a6f

    def update_loss(self, name, loss):
        if name not in self.loss_dict:
            self.loss_dict.update({name:[loss]})
        else:
            self.loss_dict[name].append(loss)

    def calculate_total_loss(self):
        batch_loss = []
        for loss in self.loss_dict.values():
            batch_loss.append(loss[-1])
        total_loss = torch.sum(torch.stack(batch_loss))
        self.batch_loss.append(total_loss)
        return total_loss

<<<<<<< HEAD
    def calculate_epoch_loss(self, output_path, epoch, ding_bot=None):
        fig = plt.figure()
        loss_json = os.path.join(output_path, "loss.json")
        if not os.path.isdir(output_path):
            os.makedirs(output_path)
            open(loss_json, "w")
        with open(loss_json, "r") as f:
            losses = json.load(f) if epoch else dict()
        with open(loss_json, "w") as f:
            losses.update({"epoch":epoch})
            for i, (name, loss) in enumerate(self.loss_dict.items()):
                epoch_loss = np.hstack((losses.get(name, []), np.average(torch.tensor(loss))))
                losses.update({name:list(epoch_loss)})
                fig.add_subplot(3, 3, i+1, title=name).plot(epoch_loss)
            total_loss = np.hstack((losses.get("total_loss", []), np.average(torch.tensor(self.batch_loss))))
            losses.update({"total_loss":list(total_loss)})
            json.dump(losses, f)
        fig.add_subplot(3, 3, i+2, title="total_loss").plot(total_loss)
=======
    def calculate_epoch_loss(self, start_token, output_path, ding_bot=None):
        fig = plt.figure()
        
        if start_token:
            self.len_train_batch = self.batch_size // len(self.loss_dict)

        for i, name in enumerate(self.loss_dict):
            epoch_loss = torch.mean(torch.tensor(self.loss_dict[name]["batch_loss"][-self.len_train_batch:]))
            test_loss = torch.mean(torch.tensor(self.loss_dict[name]["test_loss"][-self.len_train_batch:]))
            self.loss_dict[name]["epoch_loss"].append(epoch_loss)
            fig.add_subplot(3, 3, i+1, title=name).plot(self.loss_dict[name]["epoch_loss"])
        self.epoch_loss.append(torch.mean(torch.tensor(self.batch_loss[-self.len_train_batch:])))
        fig.add_subplot(3, 3, i+2, title="total_loss").plot(self.epoch_loss)
>>>>>>> a5cd3d0f9a5d98072aa5d624318605e6cc6b8a6f
        fig.tight_layout()
        fig.canvas.draw()
        img = cv2.cvtColor(np.asarray(fig.canvas.buffer_rgba()), cv2.COLOR_RGBA2BGR)
        cv2.imwrite(os.path.join(output_path, 'loss.png'), img)
        if ding_bot:
            ding_bot.add_md("tran_mmbody", "【LOSS】 \n ![img]({}) \n 【RMSE】\n epoch={}, rmse={}".format(bot.img2b64(img), epoch, rmse))
            ding_bot.enable()
        plt.close()
<<<<<<< HEAD
        cv2.imwrite(os.path.join(output_path, 'loss.png'), img)
        if ding_bot:
            ding_bot.add_md("tran_mmbody", "【IMG】 \n ![img]({}) \n 【LOSS】\n epoch={}, loss={}".format(ding_bot.img2b64(img), epoch, total_loss[-1]))
            ding_bot.enable()
        
        self.loss_dict = {}
        self.batch_loss = []
=======
        return img, self.epoch_loss
>>>>>>> a5cd3d0f9a5d98072aa5d624318605e6cc6b8a6f
