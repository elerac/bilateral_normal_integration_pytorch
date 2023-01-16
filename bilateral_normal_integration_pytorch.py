import argparse
from pathlib import Path
import time
from tqdm import trange
import numpy as np
import cv2
import torch
import pyvista as pv
from utils import map_depth_map_to_point_clouds, construct_facets_from


def sigmoid(x: torch.Tensor, k: float = 1.0) -> torch.Tensor:
    return 1 / (1 + torch.exp(-k * x))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", type=Path, required=True)
    parser.add_argument("-k", type=float, default=2.0)
    parser.add_argument("-i", "--iter", type=int, default=500)
    parser.add_argument("-t", "--tol", type=float, default=1e-5)
    parser.add_argument("-i2", "--iter2", type=int, default=5000)
    parser.add_argument("-t2", "--tol2", type=float, default=1e-5)
    parser.add_argument("-l", "--lr", type=float, default=0.01)
    args = parser.parse_args()
    path_dir: Path = args.path
    k: float = args.k
    max_iter: int = args.iter
    tol: float = args.tol
    lr: float = args.lr
    max_iter_sub: int = args.iter2
    tol_sub: float = args.tol2

    if not path_dir.is_dir():
        raise FileNotFoundError(f"{path_dir} must be a directory")

    # Normal map
    path_normal = path_dir / "normal_map.png"
    normal_map = cv2.imread(str(path_normal), -1)[:, :, :3]
    height, width, _ = normal_map.shape
    if normal_map.dtype == np.uint16:
        normal_map = normal_map / 65535.0 * 2.0 - 1.0
    else:
        normal_map = normal_map / 255.0 * 2.0 - 1.0

    nz = -normal_map[..., 0]  # transfer the normal map from the normal coordinates to the camera coordinates
    ny = normal_map[..., 1]
    nx = normal_map[..., 2]

    # Mask
    path_mask = path_dir / "mask.png"
    if path_mask.exists():
        mask = cv2.imread(str(path_mask), 0) != 0
    else:
        mask = np.full((height, width), True)

    # K
    path_K = path_dir / "K.txt"
    K = np.loadtxt(path_K) if path_K.exists() else None

    if K is not None:  # perspective
        yy, xx = np.meshgrid(range(width), range(height))
        xx = np.flip(xx, axis=0)

        cx = K[0, 2]
        cy = K[1, 2]
        fx = K[0, 0]
        fy = K[1, 1]

        uu = xx - cx
        vv = yy - cy

        nz_u = uu * nx + vv * ny + fx * nz
        nz_v = uu * nx + vv * ny + fy * nz
        del xx, yy, uu, vv
    else:  # orthographic
        nz_u = nz.copy()
        nz_v = nz.copy()

    # Select PyTorch backend
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    # Transfer data to device
    normal_map = torch.Tensor(normal_map).to(device)
    nx = torch.Tensor(nx).to(device)
    ny = torch.Tensor(ny).to(device)
    nz_u = torch.Tensor(nz_u).to(device)
    nz_v = torch.Tensor(nz_v).to(device)
    mask = torch.Tensor(mask).to(device)

    # Kernels for the partial derivatives
    kernel_u_posi = torch.Tensor([[0, 0, 0], [0, -1, 1], [0, 0, 0]]).to(device)
    kernel_u_nega = torch.Tensor([[0, 0, 0], [-1, 1, 0], [0, 0, 0]]).to(device)
    kernel_v_posi = torch.Tensor([[0, 1, 0], [0, -1, 0], [0, 0, 0]]).to(device)
    kernel_v_nega = torch.Tensor([[0, 0, 0], [0, 1, 0], [0, -1, 0]]).to(device)
    kernels_u = torch.stack([kernel_u_posi, kernel_u_nega], dim=0)[:, None, :, :]
    kernels_v = torch.stack([kernel_v_posi, kernel_v_nega], dim=0)[:, None, :, :]

    # PyTorch oprimizer and initialization
    z_initial = torch.zeros((height, width)).to(device)
    z = torch.nn.Parameter(z_initial)
    optimizer = torch.optim.Adam([z], lr=lr)

    wu = torch.full((height, width), 0.5).to(device)
    wv = torch.full((height, width), 0.5).to(device)

    num_normals = int(torch.sum(mask).item())
    projection = "orthographic" if K is None else "perspective"
    print(f"Running bilateral normal integration with k={args.k} in the {projection} case.")
    print(f"The number of normal vectors is {num_normals}.")

    tic = time.time()

    loss_list = []
    loss_old = 100000
    pbar1 = trange(max_iter)
    for i in pbar1:
        # fix weights and solve for depths
        loss_sub_old = 100000
        pbar_sub = trange(max_iter_sub, leave=False)
        for j in pbar_sub:
            optimizer.zero_grad()

            dzu_posi, dzu_nega = nz_u * torch.nn.functional.conv2d(z[None, ...], kernels_u, padding="same")
            dzv_posi, dzv_nega = nz_v * torch.nn.functional.conv2d(z[None, ...], kernels_v, padding="same")

            loss = (
                torch.sum(
                    mask * (wu * (dzu_posi + nx) ** 2 + (1 - wu) * (dzu_nega + nx) ** 2 + wv * (dzv_posi + ny) ** 2 + (1 - wv) * (dzv_nega + ny) ** 2)
                )
                / num_normals
            )

            loss.backward()
            optimizer.step()

            loss_sub = loss.item()
            relative_loss_sub = abs(loss_sub_old - loss_sub) / loss_sub
            loss_sub_old = loss_sub
            pbar_sub.set_description(f" Sub loop: step {j + 1}/{max_iter_sub} loss: {loss_sub:.5f} relative loss: {relative_loss_sub:.3e}")
            if relative_loss_sub < tol_sub:
                break

        # update weights
        with torch.no_grad():
            wu = sigmoid(dzu_nega**2 - dzu_posi**2, k)
            wv = sigmoid(dzv_nega**2 - dzv_posi**2, k)

        # compute relative loss to judge whether the iteration should be terminated
        loss = loss_sub
        loss_list.append(loss)
        relative_loss = abs(loss_old - loss) / loss_old
        loss_old = loss
        pbar1.set_description(f"Main loop: step {i + 1}/{max_iter} loss: {loss_sub:.5f} relative loss: {relative_loss:.3e}")
        if relative_loss < tol:
            break

    toc = time.time()
    print(f"Total time: {toc - tic:.3f} sec")

    # Depth
    z_np = z.cpu().detach().numpy()
    if K is not None:  # perspective
        depth_map = np.exp(z_np)
    else:  # orthographic
        depth_map = z_np

    mask_np = mask.cpu().detach().numpy() != 0
    d_min = np.min(depth_map[mask_np])
    d_max = np.max(depth_map[mask_np])
    d_min, d_max = d_max, d_min  # Flip colormap
    depth_map_normalized = np.clip(255 * (depth_map - d_min) / (d_max - d_min), 0, 255).astype(np.uint8)
    depth_map_colred = cv2.applyColorMap(depth_map_normalized, cv2.COLORMAP_VIRIDIS)
    depth_map_colred[~mask_np] = 0
    cv2.imwrite(str(path_dir / f"depthmap_k_{args.k}.png"), depth_map_colred)

    # Mesh
    vertices = map_depth_map_to_point_clouds(depth_map, mask_np, K=K)
    facets = construct_facets_from(mask_np)
    surface = pv.PolyData(vertices, facets)
    surface.save(str(path_dir / f"mesh_k_{args.k}.ply"), binary=False)

    # wu and wv
    wu_map = wu.cpu().detach().numpy()
    wv_map = wv.cpu().detach().numpy()
    wu_map_colored = cv2.applyColorMap((255 * wu_map).astype(np.uint8), cv2.COLORMAP_JET)
    wv_map_colored = cv2.applyColorMap((255 * wv_map).astype(np.uint8), cv2.COLORMAP_JET)
    wu_map_colored[~mask_np] = 255
    wv_map_colored[~mask_np] = 255
    cv2.imwrite(str(path_dir / f"wu_k_{args.k}.png"), wu_map_colored)
    cv2.imwrite(str(path_dir / f"wv_k_{args.k}.png"), wv_map_colored)

    # Loss
    np.save(path_dir / f"loss_k_{args.k}", np.array(loss_list))
    np.savetxt(path_dir / f"loss_k_{args.k}.csv", np.array(loss_list), delimiter=",")

    print(f"saved {path_dir}")


if __name__ == "__main__":
    main()
