from calib import coord_trans
import numpy as np


def data_aling_test():
    from dataloader import align
    df = align('./dataloader/__test__/gt', './dataloader/__test__/camera', './dataloader/__test__/radar', './dataloader/__test__/output/test.csv', abspath=False)
    print(df)


def coord_trans_test():
    from calib import coord_trans
    coords = coord_trans.read_static_ts('./calib/__test__/radar1.ts')
    # set or view board params here
    print(coord_trans.set_params())

    # compute transform matrix, print T, R param
    R, t = coord_trans.trans_matrix(coords)
    print(R, t)
    orig_coord = np.array([1000, 1500, 900])

    # transform a coordinate
    trans_coord = coord_trans.trans_coord(orig_coord, R, t)
    print(trans_coord)


if __name__ == '__main__':
    coord_trans_test()
