try:
    import matplotlib as mpl
    from matplotlib import colors as mcolors
    from matplotlib.colors import Normalize, LightSource
    import matplotlib.pyplot as plt
    from matplotlib import get_backend
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from mpl_toolkits.mplot3d import Axes3D
    plt.switch_backend('agg')
except:
    pass

import numpy as np
from numpy import pi


def config_colors(x, y, z, dx, dy, dz, color):
    cuboid = np.array([
        # -z
        (
            (0, 0, 0),
            (0, 1, 0),
            (1, 1, 0),
            (1, 0, 0),
        ),
        # +z
        (
            (0, 0, 1),
            (1, 0, 1),
            (1, 1, 1),
            (0, 1, 1),
        ),
        # -y
        (
            (0, 0, 0),
            (1, 0, 0),
            (1, 0, 1),
            (0, 0, 1),
        ),
        # +y
        (
            (0, 1, 0),
            (0, 1, 1),
            (1, 1, 1),
            (1, 1, 0),
        ),
        # -x
        (
            (0, 0, 0),
            (0, 0, 1),
            (0, 1, 1),
            (0, 1, 0),
        ),
        # +x
        (
            (1, 0, 0),
            (1, 1, 0),
            (1, 1, 1),
            (1, 0, 1),
        ),
    ])

    polys = np.empty(x.shape + cuboid.shape)
    for i, p, dp in [(0, x, dx), (1, y, dy), (2, z, dz)]:
        p = p[..., np.newaxis, np.newaxis]
        dp = dp[..., np.newaxis, np.newaxis]
        polys[..., i] = p + dp * cuboid[..., i]

    polys = polys.reshape((-1,) + polys.shape[2:])

    facecolors = []
    if len(color) == len(x):
        for c in color:
            facecolors.extend([c] * 6)
    else:
        facecolors = list(mcolors.to_rgba_array(color))
        if len(facecolors) < len(x):
            facecolors *= (6 * len(x))

    normals = config_normals(polys)
    return config_shade_colors(facecolors, normals)


def config_normals(polygons):
    if isinstance(polygons, np.ndarray):
        n = polygons.shape[-2]
        i1, i2, i3 = 0, n//3, 2*n//3
        v1 = polygons[..., i1, :] - polygons[..., i2, :]
        v2 = polygons[..., i2, :] - polygons[..., i3, :]
    else:
        v1 = np.empty((len(polygons), 3))
        v2 = np.empty((len(polygons), 3))
        for poly_i, ps in enumerate(polygons):
            n = len(ps)
            i1, i2, i3 = 0, n//3, 2*n//3
            v1[poly_i, :] = ps[i1, :] - ps[i2, :]
            v2[poly_i, :] = ps[i2, :] - ps[i3, :]

    return np.cross(v1, v2)


def config_shade_colors(color, normals, lightsource=None):
    if lightsource is None:
        lightsource = LightSource(azdeg=225, altdeg=19.4712)

    def mod(v):
        return np.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)

    shade = np.array([np.dot(n / mod(n), lightsource.direction)
                      if mod(n) else np.nan for n in normals])
    mask = ~np.isnan(shade)

    if mask.any():
        norm = Normalize(min(shade[mask]), max(shade[mask]))
        shade[~mask] = min(shade[mask])
        color = mcolors.to_rgba_array(color)
        alpha = color[:, 3]
        colors = (0.5 + norm(shade)[:, np.newaxis] * 0.5) * color
        colors[:, 3] = alpha
    else:
        colors = np.asanyarray(color).copy()

    return colors


def state_to_density_matrix(quantum_state):
    """ convert quantum state to density matrix

    Args:
        quantum state: complex list

    Returns:
        density matrix

    Raises:
        RuntimeError: if input is not a valid quantum state.
    """
    rho = np.asarray(quantum_state)
    if rho.ndim == 1:
        rho = np.outer(rho, np.conj(rho))
    shape = np.shape(rho)
    if len(shape) != 2 or shape[0] != shape[1]:
        raise RuntimeError("Input is not a valid quantum state.")
    num = int(np.log2(rho.shape[0]))
    if 2 ** num != rho.shape[0]:
        raise RuntimeError("Input is not a multi-qubit quantum state.")
    return rho


def complex_phase_cmap():
    cdict = {'blue': ((0.00, 0.0, 0.0),
                      (0.25, 0.0, 0.0),
                      (0.50, 1.0, 1.0),
                      (0.75, 1.0, 1.0),
                      (1.00, 0.0, 0.0)),
             'green': ((0.00, 0.0, 0.0),
                       (0.25, 1.0, 1.0),
                       (0.50, 0.0, 0.0),
                       (0.75, 1.0, 1.0),
                       (1.00, 0.0, 0.0)),
             'red': ((0.00, 1.0, 1.0),
                     (0.25, 0.5, 0.5),
                     (0.50, 0.0, 0.0),
                     (0.75, 0.0, 0.0),
                     (1.00, 1.0, 1.0))}

    cmap = mpl.colors.LinearSegmentedColormap('phase_colormap', cdict, 256)

    return cmap


def config_color_array(color):
    if color is None:
        color = ["#648fff", "#648fff"]
    else:
        if len(color) != 2:
            raise RuntimeError("'color' must be a list of len=2.")
        if color[0] is None:
            color[0] = "#648fff"
        if color[1] is None:
            color[1] = "#648fff"
    return color


def plot_state_city(state, title="", figsize=None, color=None, ax_real=None, ax_imag=None):
    """ plot quantum state city 

    Args:
        quantum state: complex list
        title : string for figure
        color : color for figure

    Returns:
        matplot figure

    Raises:
        RuntimeError: if input is not a valid quantum state.
    """
    alpha = 1
    rho = state_to_density_matrix(state)

    num = int(np.log2(len(rho)))
    real_matrix = np.real(rho)
    imag_matrix = np.imag(rho)

    # get the labels
    column_names = [bin(i)[2:].zfill(num) for i in range(2**num)]
    row_names = [bin(i)[2:].zfill(num) for i in range(2**num)]

    length_x = len(real_matrix[0])            # Work out matrix dimensions
    length_y = len(real_matrix[:, 0])
    position_x = np.arange(0, length_x, 1)    # Set up a mesh of positions
    position_y = np.arange(0, length_y, 1)
    position_x, position_y = np.meshgrid(position_x+0.25, position_y+0.25)

    position_x = position_x.flatten()
    position_y = position_y.flatten()
    zpos = np.zeros(length_x*length_y)

    dx = 0.5 * np.ones_like(zpos)  # width of bars
    dy = dx.copy()
    dzr = real_matrix.flatten()
    dzi = imag_matrix.flatten()

    color = config_color_array(color)
    if ax_real is None and ax_imag is None:
        if figsize is None:
            figsize = (15, 5)

        fig = plt.figure(figsize=figsize)
        axia_1 = fig.add_subplot(1, 2, 1, projection='3d')
        axia_2 = fig.add_subplot(1, 2, 2, projection='3d')
    elif ax_real is not None:
        fig = ax_real.get_figure()
        axia_1 = ax_real
        if ax_imag is not None:
            axia_2 = ax_imag
    else:
        fig = ax_imag.get_figure()
        axia_1 = None
        axia_2 = ax_imag

    max_dzr = max(dzr)
    min_dzr = min(dzr)
    min_dzi = np.min(dzi)
    max_dzi = np.max(dzi)

    if axia_1 is not None:
        fc1 = config_colors(position_x, position_y,
                            zpos, dx, dy, dzr, color[0])
        for idx, cur_zpos in enumerate(zpos):
            if dzr[idx] > 0:
                zorder = 2
            else:
                zorder = 0
            b1 = axia_1.bar3d(position_x[idx], position_y[idx], cur_zpos,
                              dx[idx], dy[idx], dzr[idx],
                              alpha=alpha, zorder=zorder)
            b1.set_facecolors(fc1[6*idx:6*idx+6])

        xlim, ylim = axia_1.get_xlim(), axia_1.get_ylim()
        x = [xlim[0], xlim[1], xlim[1], xlim[0]]
        y = [ylim[0], ylim[0], ylim[1], ylim[1]]
        z = [0, 0, 0, 0]
        verts = [list(zip(x, y, z))]

        pc1 = Poly3DCollection(verts, alpha=0.15, facecolor='k',
                               linewidths=1, zorder=1)

        if min(dzr) < 0 < max(dzr):
            axia_1.add_collection3d(pc1)
        axia_1.set_xticks(np.arange(0.5, length_x+0.5, 1))
        axia_1.set_yticks(np.arange(0.5, length_y+0.5, 1))
        if max_dzr != min_dzr:
            axia_1.axes.set_zlim3d(
                np.min(dzr), max(np.max(dzr) + 1e-9, max_dzi))
        else:
            if min_dzr == 0:
                axia_1.axes.set_zlim3d(np.min(dzr), max(
                    np.max(dzr)+1e-9, np.max(dzi)))
            else:
                axia_1.axes.set_zlim3d(auto=True)
        axia_1.get_autoscalez_on()
        axia_1.w_xaxis.set_ticklabels(row_names, fontsize=14, rotation=45,
                                      ha='right', va='top')
        axia_1.w_yaxis.set_ticklabels(column_names, fontsize=14, rotation=-22.5,
                                      ha='left', va='center')
        axia_1.set_zlabel('Re[$\\rho$]', fontsize=14)
        for tick in axia_1.zaxis.get_major_ticks():
            tick.label.set_fontsize(14)

    if axia_2 is not None:
        fc2 = config_colors(position_x, position_y,
                            zpos, dx, dy, dzi, color[1])
        for idx, cur_zpos in enumerate(zpos):
            if dzi[idx] > 0:
                zorder = 2
            else:
                zorder = 0
            b2 = axia_2.bar3d(position_x[idx], position_y[idx], cur_zpos,
                              dx[idx], dy[idx], dzi[idx],
                              alpha=alpha, zorder=zorder)
            b2.set_facecolors(fc2[6*idx:6*idx+6])

        xlim, ylim = axia_2.get_xlim(), axia_2.get_ylim()
        x = [xlim[0], xlim[1], xlim[1], xlim[0]]
        y = [ylim[0], ylim[0], ylim[1], ylim[1]]
        z = [0, 0, 0, 0]
        verts = [list(zip(x, y, z))]

        pc2 = Poly3DCollection(verts, alpha=0.2, facecolor='k',
                               linewidths=1, zorder=1)

        if min(dzi) < 0 < max(dzi):
            axia_2.add_collection3d(pc2)
        axia_2.set_xticks(np.arange(0.5, length_x+0.5, 1))
        axia_2.set_yticks(np.arange(0.5, length_y+0.5, 1))
        if min_dzi != max_dzi:
            eps = 0
            axia_2.axes.set_zlim3d(np.min(dzi), max(
                np.max(dzr)+1e-9, np.max(dzi)+eps))
        else:
            if min_dzi == 0:
                axia_2.set_zticks([0])
                eps = 1e-9
                axia_2.axes.set_zlim3d(np.min(dzi), max(
                    np.max(dzr)+1e-9, np.max(dzi)+eps))
            else:
                axia_2.axes.set_zlim3d(auto=True)

        axia_2.w_xaxis.set_ticklabels(row_names, fontsize=14, rotation=45,
                                      ha='right', va='top')
        axia_2.w_yaxis.set_ticklabels(column_names, fontsize=14, rotation=-22.5,
                                      ha='left', va='center')
        axia_2.set_zlabel('Im[$\\rho$]', fontsize=14)
        for tick in axia_2.zaxis.get_major_ticks():
            tick.label.set_fontsize(14)
        axia_2.get_autoscalez_on()

    fig.suptitle(title, fontsize=16)
    if ax_real is None and ax_imag is None:
        if get_backend() in ['module://ipykernel.pylab.backend_inline', 'nbAgg']:
            plt.close(fig)
        plt.show()
        return fig


def plot_density_matrix(M, xlabels=None, ylabels=None,
                        title=None, limits=None, phase_limits=None, fig=None, axis_vals=None,
                        threshold=None):
    """ plot quantum state density matrix 

    Args:
        quantum state: complex list
        title : string for figure
        color : color for figure

    Returns:
        matplot figure

    Raises:
        RuntimeError: if input is not a valid quantum state.
    """

    # if isinstance(M, Qobj):
    # extract matrix data from Qobj
    # M = M.full()
    # M = M.toarray(order='C')

    index_array = [0.2, 0.4, 0.6, 0.8, 1.0]

    key = 0.0
    for matrix in M:
        for value in matrix:
            if (key < abs(value)):
                key = abs(value)

    z_axis_limit = index_array[(int)(key / 0.2)]

    n = np.size(M)
    position_x, position_y = np.meshgrid(range(M.shape[0]), range(M.shape[1]))
    position_x = position_x.T.flatten() - 0.5
    position_y = position_y.T.flatten() - 0.5
    zpos = np.zeros(n)
    dx = dy = 0.8 * np.ones(n)
    vectors = M.flatten()
    dz = abs(vectors)

    # make small numbers real, to avoid random colors
    idx, = np.where(abs(vectors) < 0.001)
    vectors[idx] = abs(vectors[idx])

    if phase_limits:  # check that limits is a list type
        phase_min = phase_limits[0]
        phase_max = phase_limits[1]
    else:
        phase_min = -pi
        phase_max = pi

    norm = mpl.colors.Normalize(phase_min, phase_max)
    cmap = complex_phase_cmap()

    colors = cmap(norm(np.angle(vectors)))
    if threshold is not None:
        colors[:, 3] = 1 * (dz > threshold)

    if axis_vals is None:
        fig = plt.figure()
        axis_vals = fig.add_subplot(projection="3d")
        # axis_vals = Axes3D(fig, azim=-35, elev=35)

    axis_vals.bar3d(position_x, position_y, zpos, dx, dy, dz, color=colors)

    if title and fig:
        axis_vals.set_title(title)

    # x axis
    axis_vals.axes.w_xaxis.set_major_locator(plt.IndexLocator(1, -0.5))
    if xlabels:
        axis_vals.set_xticklabels(xlabels)
    axis_vals.tick_params(axis='x', labelsize=12)

    # y axis
    axis_vals.axes.w_yaxis.set_major_locator(plt.IndexLocator(1, -0.5))
    if ylabels:
        axis_vals.set_yticklabels(ylabels)
    axis_vals.tick_params(axis='y', labelsize=12)

    # z axis
    if limits and isinstance(limits, list):
        axis_vals.set_zlim3d(limits)
    else:
        axis_vals.set_zlim3d([0, z_axis_limit])  # use min/max
    # axis_vals.set_zlabel('abs')

    cax, kw = mpl.colorbar.make_axes(axis_vals, shrink=.75, pad=.0)
    cb = mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm)
    cb.set_ticks([-pi, -pi / 2, 0, pi / 2, pi])
    cb.set_ticklabels(
        (r'$-\pi$', r'$-\pi/2$', r'$0$', r'$\pi/2$', r'$\pi$'))
    cb.set_label('arg', rotation='horizontal')

    plt.show()
    # plt.ylabel('arg',rotation=)
    return fig, axis_vals
