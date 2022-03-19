import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.ticker import FuncFormatter

from matplotlib import rcParams

config = {
    "font.family":'Times New Roman',
    "font.size": 40,
    "mathtext.fontset":'stix',
    # "font.serif": ['SimSun'],
}
rcParams.update(config)


def per_joint_err_plot():
    per_joint_loss_in_lab = np.load("/home/nesc525/drivers/4/p4tmesh/large2/6dwithgender/loss/test/per_joint_loss.npy")
    xs=np.linspace(0,23,24)
    plt.bar(x=xs, height=per_joint_loss_in_lab[:24],width=0.8,align="center")
    plt.xticks(ticks=xs, labels=('Pelvis','L_Hip','R_Hip','Spine1','L_Knee','R_Knee','Spine2','L_Ankle','R_Ankle','Spine3','L_Foot','R_Foot','Neck','L_Collar'
                                    ,'R_Collar','Head','L_Shoulder','R_Shoulder','L_Elbow','R_Elbow','L_Wrist','R_Wrist','L_Hand','R_Hand'))
    plt.xlim(-0.5,23.5)
    plt.ylabel("error(m)")
    plt.show()

def plot_cdf(arr, color, label):
    hist, bin_edges = np.histogram(arr*1000, bins=100, range=[0,200])
    cdf = np.cumsum(hist)/arr.size
    plt.plot(bin_edges[:-1], cdf, color=color, linewidth=3.0, label=label)

    # model=make_interp_spline(bin_edges[:-1], cdf, bc_type="natural")
    # xs=np.linspace(0, 0.2, 100)
    # ys=model(xs)
    # plt.plot(xs, ys, color=color, label=label)

def cdf_plot():
    device = "mmWave"
    path = "/home/nesc525/drivers/4/p4tmesh/large4/6dwithgender/loss"
    # path = "/media/nesc525/disk4/p4t_rgbd/{}/loss".format(device)
    joint_loss_in_lab = np.load(os.path.join(path, "indoor_test/joints_loss.npy"))
    joint_loss_corridor = np.load(os.path.join(path, "corridor_test/joints_loss.npy"))
    joint_loss_night = np.load(os.path.join(path, "night_test/joints_loss.npy"))
    joint_loss_outdoor = np.load(os.path.join(path, "outdoor_test/joints_loss.npy"))
    joint_loss_rain = np.load(os.path.join(path, "rain_test/joints_loss.npy"))
    joint_loss_smoke = np.load(os.path.join(path, "smoke_test/joints_loss.npy"))
    joint_loss_occlusion = np.load(os.path.join(path, "mask_test/joints_loss.npy"))

    # vertices_loss_in_lab = np.load(os.path.join(path, "vertices_loss.npy"))
    # rgbd_vertices_loss_in_lab = np.load(os.path.join(path2, "vertices_loss.npy"))
    plt.figure(figsize=(15, 8), dpi=80)

    plot_cdf(joint_loss_in_lab, 'red', 'In_lab')
    plot_cdf(joint_loss_occlusion, 'violet', 'Occlusion')
    plot_cdf(joint_loss_corridor, 'orange', 'Corridor')
    plot_cdf(joint_loss_outdoor, 'green', 'Daytime')
    plot_cdf(joint_loss_night, 'yellow', 'Night')
    plot_cdf(joint_loss_rain, 'indigo', 'Rain')
    plot_cdf(joint_loss_smoke, 'blue', 'Smoke')
    # plot_rmse(p4t_rmse_rt_02, 'pink', 'r_t=2')
    # plot_rmse(paconv_rmse_wF3, 'cyan', 'f=3')
    # plot_rmse(paconv_rmse_wF6, 'black', 'f=6')
    # plot_rmse(paconv_rmse_woF, 'grey', 'f=0')
    # plot_rmse(plstm_rmse, 'gold', 'plstm')

    def to_percent(temp, position):
        return '%1.0f'%(100*temp) + '%'

    plt.xlim((0,200))
    plt.ylim((0,1))
    ax = plt.gca()
    ax.xaxis.set_major_locator(plt.MultipleLocator(25))
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.1))
    ax.yaxis.set_major_formatter(FuncFormatter(to_percent))
    # plt.xticks(np.arange(0,200,10))
    plt.grid()
    plt.xlabel("error (mm)")
    plt.ylabel("percentage")
    # plt.tick_params(labelsize=16)
    plt.legend(loc="lower right", fontsize=30)
    plt.savefig(fname=os.path.join(path,device+"_cdf"), dpi=300)
    plt.show()

if __name__ == "__main__":
    cdf_plot()