import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2

class LossManager():
    def __init__(self) -> None:
        super(LossManager).__init__()
        self.loss_dict = {}
        self.batch_loss = []
        self.epoch_loss = []
        self.batch_size = 0
        self.len_train_batch = 0

    def update_loss(self, name, loss):
        if name not in self.loss_dict:
            self.loss_dict.update({name:dict(batch_loss=[loss], epoch_loss=[])})
        else:
            self.loss_dict[name]["batch_loss"].append(loss)
        self.batch_size += 1

    def calculate_total_loss(self):
        batch_loss = []
        for loss in self.loss_dict.values():
            batch_loss.append(loss["batch_loss"][-1])
        total_loss = torch.sum(torch.stack(batch_loss))
        self.batch_loss.append(total_loss)
        return total_loss

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
        fig.tight_layout()
        fig.canvas.draw()
        img = cv2.cvtColor(np.asarray(fig.canvas.buffer_rgba()), cv2.COLOR_RGBA2BGR)
        cv2.imwrite(os.path.join(output_path, 'loss.png'), img)
        if ding_bot:
            ding_bot.add_md("tran_mmbody", "【LOSS】 \n ![img]({}) \n 【RMSE】\n epoch={}, rmse={}".format(bot.img2b64(img), epoch, rmse))
            ding_bot.enable()
        plt.close()
        return img, self.epoch_loss
