import numpy as np
import torch
import einops as ein
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def equalize_3D_aspect(ax, X, Y, Z):
    max_range = np.array([np.nanmax(X)-np.nanmin(X), np.nanmax(Y)-np.nanmin(Y), np.nanmax(Z)-np.nanmin(Z)]).max()
    Xb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][0].flatten() + 0.5*(np.nanmax(X)+np.nanmin(X))
    Yb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][1].flatten() + 0.5*(np.nanmax(Y)+np.nanmin(Y))
    Zb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][2].flatten() + 0.5*(np.nanmax(Z)+np.nanmin(Z))
    # Comment or uncomment following both lines to test the fake bounding box:
    for xb, yb, zb in zip(Xb, Yb, Zb):
       ax.plot([xb], [yb], [zb], 'w')

def plot_views(pix_to_coordinates, num_angles=5, save_path=None, cmap=plt.cm.tab10):
    fig = plt.figure(figsize=(25,10), dpi=300)
    for idx, angle in enumerate(np.linspace(0,90,num_angles)):
        ax = fig.add_subplot(1, num_angles, idx+1, projection='3d')
        ax.view_init(30, angle)
        for c in range(len(pix_to_coordinates)):
            temp = ein.rearrange(pix_to_coordinates[c], 'h w c -> (h w) c').detach().cpu().numpy()
            ax.scatter(temp[:, 0], temp[:, 1], zs=temp[:, 2], s=0.01, color=cmap(c))
        equalize_3D_aspect(ax, temp[:, 0], temp[:, 1], temp[:, 2])
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_agg(face_labels, color, num_angles=5, save_path=None, cmap='inferno'):
    valid_idxs = torch.where(~torch.isnan(color))
    face_labels = face_labels[valid_idxs].cpu()
    color = color[valid_idxs].cpu()
    color = torch.clamp(color, 0, 1)
    
    fig = plt.figure(figsize=(25,10))
    for idx, angle in enumerate(np.linspace(0,90,num_angles)):
        ax = fig.add_subplot(1, num_angles, idx+1, projection='3d')
        ax.view_init(30, angle)
        im = ax.scatter(face_labels[:, 0], face_labels[:, 1], zs=face_labels[:, 2], s=0.01, c=color, cmap=cmap, marker='o')
        equalize_3D_aspect(ax, face_labels[:, 0], face_labels[:, 1], face_labels[:, 2])
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()