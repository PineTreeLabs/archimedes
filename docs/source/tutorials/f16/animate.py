"""Animate a steady level turn maneuver of the F-16.

Requires a low-poly F-16 STL file, available here:
https://www.printables.com/model/840061-low-poly-f-16-falcon-aka-viper-jet-fighter/files

For additional dependencies: `uv pip install meshio`
"""

# ruff: noqa: N803, N806, N816
from __future__ import annotations

from pprint import pprint

import matplotlib.pyplot as plt
import meshio
import numpy as np
from f16 import SubsonicF16
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import archimedes as arc


def traj_data(xs_flat, unravel, stride=5):
    xs = arc.vmap(unravel)(xs_flat.T)

    positions = xs.pos.copy()
    positions[:, 2] *= -1  # Use altitude as Z for visualization
    positions *= 1 / 1000  # Scale to kft

    orientations = []
    for i in range(len(ts)):
        x = unravel(xs_flat[:, i])
        R_BN = x.att.as_matrix()
        orientations.append(R_BN)

    def att_as_rpy(x_flat):
        x = unravel(x_flat)
        rpy = np.mod(x.att.as_euler("xyz"), 2 * np.pi)
        return np.rad2deg(rpy)

    rpy = arc.vmap(att_as_rpy, in_axes=1)(xs_flat)

    return {
        "positions": positions[::stride],
        "orientations": orientations[::stride],
        "rpy": rpy[::stride],
        "time": ts[::stride],
    }


def plot_f16_trajectory(traj_data, mesh_file):
    positions = traj_data["positions"]
    orientations = traj_data["orientations"]
    rpy = traj_data["rpy"]

    fig = plt.figure(figsize=(10, 8), layout="constrained")
    ax = fig.add_subplot(111, projection="3d")

    # Read the surface mesh
    mesh = meshio.read(mesh_file)
    vertices = mesh.points
    faces = mesh.cells[0].data

    vertices = (vertices - vertices.mean(axis=0)) / 50  # center and scale

    vertices = vertices[:, [2, 0, 1]]
    vertices[:, 2] *= -1  # Flip Z axis
    vertices[:, 1] *= -1  # Flip Y axis

    # Shift to origin at approximate CG
    vertices[:, 0] += 0.2
    vertices[:, 2] -= 0.1

    # Calculate bounds from trajectory
    x_min, x_max = positions[:, 0].min(), positions[:, 0].max()
    y_min, y_max = positions[:, 1].min(), positions[:, 1].max()
    z_min, z_max = positions[:, 2].min(), positions[:, 2].max()

    # Add some padding
    padding = 0.1
    x_pad = (x_max - x_min) * padding
    y_pad = (y_max - y_min) * padding
    z_pad = 2

    # # Set up the plot
    ax.set_xlim(x_min - x_pad, x_max + x_pad)
    ax.set_ylim(y_min - y_pad, y_max + y_pad)
    ax.set_zlim(z_min - z_pad, z_max + z_pad)
    ax.set_xlabel("N [kft]")
    ax.set_ylabel("E [kft]")
    ax.set_zlabel("alt [kft]")
    ax.set_aspect("equal")

    # Trail line
    (trail_line,) = ax.plot([], [], [], linewidth=2)

    # Text
    time_text = ax.text2D(0.05, 0.75, "", transform=ax.transAxes)

    limits = np.array([getattr(ax, f"get_{axis}lim")() for axis in "xyz"])
    ax.set_box_aspect(np.ptp(limits, axis=1))

    def update(frame):
        for coll in ax.collections:
            coll.remove()

        # Transform vertices
        transformed = vertices @ orientations[frame].T
        transformed[:, 2] *= -1  # Flip z back to altitude coordinates
        transformed += positions[frame]

        # Add aircraft mesh
        mesh = Poly3DCollection(
            transformed[faces],
            alpha=0.8,
            facecolor="dimgray",
            edgecolor="k",
            linewidth=0.2,
        )
        ax.add_collection3d(mesh)

        # Update trail
        trail_line.set_data(positions[: frame + 1, 0], positions[: frame + 1, 1])
        trail_line.set_3d_properties(positions[: frame + 1, 2])
        # trail_line.set_color()

        # Update text
        time_text.set_text(
            f"Roll: {rpy[frame, 0]:.1f}°\nPitch: {rpy[frame, 1]:.1f}°\n"
            f"Yaw: {rpy[frame, 2]:.1f}°"
        )

        return mesh, trail_line, time_text

    anim = FuncAnimation(
        fig,
        update,
        frames=len(positions),
        interval=50,
        blit=False,
    )
    plt.close()
    return anim


if __name__ == "__main__":
    # Steady turn rate
    model = SubsonicF16(xcg=0.3)
    result = model.trim(vt=500, turn_rate=0.1, alt=10000.0, gamma=0.0)
    pprint(result.variables)

    x0 = result.state
    u0 = result.inputs

    x0_flat, unravel = arc.tree.ravel(x0)

    def ode_rhs(t, x_flat):
        x = unravel(x_flat)
        x_t = model.dynamics(t, x, u0)
        x_t_flat, _ = arc.tree.ravel(x_t)
        return x_t_flat

    t0, tf = 0.0, 60.0
    ts = np.arange(t0, tf, 0.1)
    xs_flat = arc.odeint(ode_rhs, (t0, tf), x0_flat, t_eval=ts)
    xs = arc.vmap(unravel)(xs_flat.T)

    data = traj_data(xs_flat, unravel, stride=5)
    for mode in ("light", "dark"):
        arc.theme.set_theme(mode)
        anim = plot_f16_trajectory(data, "_static/f16.stl")
        anim.save(f"_static/f16_turn_{mode}.gif", writer=PillowWriter(fps=20))
