import matplotlib.pyplot as plt
from numpy.random import f


class WeightLoss(list):
    def __init__(self, weight=1) -> None:
        super().__init__()
        self.weight = weight

    def delta(self, idx=-1, absolute=True):
        if absolute:
            return abs(self[idx-1] - self[idx])
        else:
            return self[idx-1] - self[idx]
    
    def __getitem__(self, i):
        try:
            ret = super(WeightLoss, self).__getitem__(i) * self.weight
        except:
            ret = 0.
        return ret

class LossManager():
    def __init__(self, losses_with_weights={}, mse_threshold=1e-8, loss_threshold=1e-8) -> None:
        self.losses = {}
        self.epoch = 0
        self.mse_threshold = mse_threshold
        self.loss_threshold = loss_threshold
        self.add_loss(losses_with_weights)

        plt.ion()
        self.fig = plt.figure()
        self.fig.set_size_inches(12, 6)
        self.draw_flag = True

        self.arts = []

        # self.plotlosses = PlotLosses(groups={'losses': list(self.losses.keys())+['loss'], '_exclude': ['mesh', 'pcls', 'kpts_target', 'kpts_updated']},
        #                             outputs=[MeshPlot(cell_size=[10, 3]), ExtremaPrinterWithExclude()])

    def add_loss(self, losses_with_weights: dict):
        for loss_name, weight in losses_with_weights.items():
            self.losses[loss_name] = WeightLoss(weight)

    def update_loss(self, loss_name: str, value: float):
        if loss_name in self.losses.keys():
            self.losses[loss_name].append(value)

    def update_losses(self, losses_value: dict):
        for loss_name, value in losses_value.items():
            self.update_loss(loss_name, value)
        self.epoch += 1

    def delta(self, idx=-1, absolute=True):
        return sum([loss.delta(idx, absolute) for loss in self.losses.values()])

    def check_losses(self):
        return self[-1] < self.loss_threshold or self.delta() < self.mse_threshold

    def str_losses(self, idx=-1):
        return "\t".join(["{}={:.4f}".format(loss_name, loss[idx]) for loss_name, loss in self.losses.items()])

    def clear_plt(self):
        self.fig.clf()

    def show_losses(self):
        ax = self.fig.add_subplot(1, 2, 1)
        for loss_name, loss in self.losses.items():
            ax.plot(loss, label=loss_name)
        if "loss" not in self.arts:
            self.arts.append("loss")
            ax.legend()
            ax.set_xlabel("Iteration")
            ax.set_ylabel("Loss")
            ax.set_title("Loss vs iterations")

    def show_output(self, mesh, pcls=None, kpts_target=None, kpts_updated=None):
        x, y, z = mesh[0].T
        # ax = Axes3D(fig)
        ax = self.fig.add_subplot(1, 2, 2, projection='3d')
        if pcls is not None:
            pcl_x, pxl_y, pxl_z = pcls.T
            ax.scatter(pcl_x, -pxl_z, pxl_y, color=[(0.62,0.92,0.42,0.3)], depthshade=False)
            
        if kpts_target is not None:
            kpts_target_x, kpts_target_y, kpts_target_z = kpts_target.T
            ax.scatter(kpts_target_x, -kpts_target_z, kpts_target_y, color=[(0,1,0,1)], depthshade=False)
        
        if kpts_updated is not None:
            kpts_updated_x, kpts_updated_y, kpts_updated_z = kpts_updated.T
            ax.scatter(kpts_updated_x, -kpts_updated_z, kpts_updated_y, color=[(0,0,1,1)], depthshade=False)

        if "mesh" not in self.arts:
            self.arts.append("mesh")
            ax.set_xlabel('x')
            ax.set_ylabel('z')
            ax.set_zlabel('y')
            ax.set_title("mesh")
            ax.set_xlim(-1.2, 1.2)
            ax.set_ylim(-1.2, 1.2)
            ax.set_zlim(-1.2, 1.2)
            # 仰角 方位角
            ax.view_init(elev=0, azim=75)
        
        ax.plot_trisurf(x, -z, y, triangles=mesh[1], cmap=plt.cm.Spectral)

        if self.draw_flag:
            plt.show()
            self.draw_flag = False

    def __len__(self):
        return min([len(loss) for loss in self.losses.values()])

    def __getitem__(self, i):
        return sum([loss[i] for loss in self.losses.values()])
