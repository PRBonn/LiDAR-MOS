import os

import matplotlib
import numpy as np
import pykitti

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import yaml

basedir = ''
sequence = ''
uncerts = ''
preds = ''
gt = ''
img = ''
lidar = ''
projected_uncert = ''
projected_preds = ''

dataset = pykitti.odometry(basedir, sequence)

EXTENSIONS_LABEL = ['.label']
EXTENSIONS_LIDAR = ['.bin']
EXTENSIONS_IMG = ['.png']


def is_label(filename):
    return any(filename.endswith(ext) for ext in EXTENSIONS_LABEL)


def is_lidar(filename):
    return any(filename.endswith(ext) for ext in EXTENSIONS_LIDAR)


def is_img(filename):
    return any(filename.endswith(ext) for ext in EXTENSIONS_IMG)


def get_mpl_colormap(cmap_name):
    cmap = plt.get_cmap(cmap_name)

    # Initialize the matplotlib color map
    sm = plt.cm.ScalarMappable(cmap=cmap)

    # Obtain linear color range
    color_range = sm.to_rgba(np.linspace(0, 1, 256), bytes=True)[:, 2::-1]

    return color_range.reshape(256, 3).astype(np.float32) / 255.0


path = os.path.join(basedir + 'sequences/' + sequence + uncerts)

scan_uncert = [os.path.join(dp, f) for dp, dn, fn in os.walk(
    os.path.expanduser(path)) for f in fn if is_label(f)]
scan_uncert.sort()
path = os.path.join(basedir + 'sequences/' + sequence + preds)
scan_preds = [os.path.join(dp, f) for dp, dn, fn in os.walk(
    os.path.expanduser(path)) for f in fn if is_label(f)]
scan_preds.sort()

path = os.path.join(basedir + 'sequences/' + sequence + gt)
scan_gt = [os.path.join(dp, f) for dp, dn, fn in os.walk(
    os.path.expanduser(path)) for f in fn if is_label(f)]
scan_gt.sort()

color_map_dict = yaml.safe_load(open("color_map.yml"))['color_map']
learning_map = yaml.safe_load(open("color_map.yml"))['learning_map']
color_map = {}
uncert_mean = np.zeros(20)
total_points_per_class = np.zeros(20)
for key, value in color_map_dict.items():
    color_map[key] = np.array(value, np.float32) / 255.0


def plot_and_save(label_uncert, label_name, lidar_name, cam2_image_name):
    labels = np.fromfile(label_name, dtype=np.int32).reshape((-1))
    uncerts = np.fromfile(label_uncert, dtype=np.float32).reshape((-1))
    velo_points = np.fromfile(lidar_name, dtype=np.float32).reshape(-1, 4)
    try:
        cam2_image = plt.imread(cam2_image_name)
    except IOError:
        print('detect error img %s' % label_name)

    plt.imshow(cam2_image)

    if True:

        # Project points to camera.
        cam2_points = dataset.calib.T_cam2_velo.dot(velo_points.T).T

        # Filter out points behind camera
        idx = cam2_points[:, 2] > 0
        print(idx)
        # velo_points_projected = velo_points[idx]
        cam2_points = cam2_points[idx]
        labels_projected = labels[idx]
        uncert_projected = uncerts[idx]

        # Remove homogeneous z.
        cam2_points = cam2_points[:, :3] / cam2_points[:, 2:3]

        # Apply instrinsics.
        intrinsic_cam2 = dataset.calib.K_cam2
        cam2_points = intrinsic_cam2.dot(cam2_points.T).T[:, [1, 0]]
        cam2_points = cam2_points.astype(int)

        for i in range(0, cam2_points.shape[0]):
            u, v = cam2_points[i, :]
            label = labels_projected[i]
            uncert = uncert_projected[i]
            if label > 0 and v > 0 and v < 1241 and u > 0 and u < 376:
                uncert_mean[learning_map[label]] += uncert
                total_points_per_class[learning_map[label]] += 1
                m_circle = plt.Circle((v, u), 1,
                                      color=matplotlib.cm.viridis(uncert),
                                      alpha=0.4,
                                      # color=color_map[label][..., ::-1]
                                      )
                plt.gcf().gca().add_artist(m_circle)

    plt.axis('off')
    path = os.path.join(basedir + 'sequences/' + sequence + projected_uncert)
    plt.savefig(path + label_name.split('/')[-1].split('.')[0] + '.png', bbox_inches='tight', transparent=True,
                pad_inches=0)


# with futures.ProcessPoolExecutor() as pool:
for label_uncert, label_name, lidar_name, cam2_image_name in zip(scan_uncert, scan_preds, dataset.velo_files,
                                                                 dataset.cam2_files):
    print(label_name.split('/')[-1])
    # if label_name == '/SPACE/DATA/SemanticKITTI/dataset/sequences/13/predictions/preds/001032.label':
    plot_and_save(label_uncert, label_name, lidar_name, cam2_image_name)
print(total_points_per_class)
print(uncert_mean)
if __name__ == "__main__":
    pass
