import numpy as np


def move_left(mask):
    return np.pad(mask, ((0, 0), (0, 1)), "constant", constant_values=0)[:, 1:]


def move_right(mask):
    return np.pad(mask, ((0, 0), (1, 0)), "constant", constant_values=0)[:, :-1]


def move_top(mask):
    return np.pad(mask, ((0, 1), (0, 0)), "constant", constant_values=0)[1:, :]


def move_bottom(mask):
    return np.pad(mask, ((1, 0), (0, 0)), "constant", constant_values=0)[:-1, :]


def move_top_left(mask):
    return np.pad(mask, ((0, 1), (0, 1)), "constant", constant_values=0)[1:, 1:]


def move_top_right(mask):
    return np.pad(mask, ((0, 1), (1, 0)), "constant", constant_values=0)[1:, :-1]


def move_bottom_left(mask):
    return np.pad(mask, ((1, 0), (0, 1)), "constant", constant_values=0)[:-1, 1:]


def move_bottom_right(mask):
    return np.pad(mask, ((1, 0), (1, 0)), "constant", constant_values=0)[:-1, :-1]


def generate_reference_sphere(width: int) -> np.ndarray:
    """Generate normal map image of reference sphere

    Parameters
    ----------
    width : int
        Width (also height) of image

    Returns
    -------
    img_sphere : np.ndarray
        Genrated sphere image (nz, ny, nx)
    """

    def i2n(i: float, num: float) -> float:
        """Index to normal"""
        n = ((i - 1) / num) * 2.0 - 1.0
        return n

    # Set nx and ny
    img_sphere = np.fromfunction(lambda iy, ix, iz: -i2n(iy, width) * (iz == 1) + i2n(ix, width) * (iz == 2), (width, width, 3), dtype=float)

    img_norm = np.linalg.norm(img_sphere, axis=-1)  # nx**2 + ny**2
    img_mask = img_norm > 1.0  # Out of sphere area
    img_mask_not = np.bitwise_not(img_mask)

    # Set nz
    img_sphere[img_mask] = np.array([1, 0, 0])
    img_sphere[img_mask_not, 0] = np.sqrt(1 - img_norm[img_mask_not])

    return img_sphere


def construct_facets_from(mask):
    idx = np.zeros_like(mask, dtype=int)
    idx[mask] = np.arange(np.sum(mask))

    facet_move_top_mask = move_top(mask)
    facet_move_left_mask = move_left(mask)
    facet_move_top_left_mask = move_top_left(mask)

    facet_top_left_mask = np.logical_and.reduce((facet_move_top_mask, facet_move_left_mask, facet_move_top_left_mask, mask))
    facet_top_right_mask = move_right(facet_top_left_mask)
    facet_bottom_left_mask = move_bottom(facet_top_left_mask)
    facet_bottom_right_mask = move_bottom_right(facet_top_left_mask)

    return np.stack(
        (
            4 * np.ones(np.sum(facet_top_left_mask)),
            idx[facet_top_left_mask],
            idx[facet_bottom_left_mask],
            idx[facet_bottom_right_mask],
            idx[facet_top_right_mask],
        ),
        axis=-1,
    ).astype(int)


def map_depth_map_to_point_clouds(depth_map, mask, K=None, step_size=1):
    # y
    # |  z
    # | /
    # |/
    # o ---x
    H, W = mask.shape
    yy, xx = np.meshgrid(range(W), range(H))
    xx = np.flip(xx, axis=0)

    if K is None:
        vertices = np.zeros((H, W, 3))
        vertices[..., 0] = xx * step_size
        vertices[..., 1] = yy * step_size
        vertices[..., 2] = depth_map
        vertices = vertices[mask]
    else:
        u = np.zeros((H, W, 3))
        u[..., 0] = xx
        u[..., 1] = yy
        u[..., 2] = 1
        u = u[mask].T  # 3 x m
        vertices = (np.linalg.inv(K) @ u).T * depth_map[mask, np.newaxis]  # m x 3

    return vertices
