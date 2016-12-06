import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np


class Plotter(object):
    def __init__(self, results):
        self.data = results  # x, r, z

    def animate(self, N, steps=None):
        """Plot NxN grid
        """
        rates = self.data["r"]
        N2 = pow(N, 2)

        fig = plt.figure()

        frames = (rate[:N2].reshape(N, N) for rate in rates.T)
        im = plt.imshow(next(frames), interpolation="none", cmap="Greys")

        def updatefig(dat):
            im.set_array(dat)
            return im,

        anim = animation.FuncAnimation(fig, updatefig, frames=frames, interval=1, blit=True,
            repeat_delay=1000, save_count=steps or 1000)

        return anim

    def plot_outputs(self, ax, N=None):
        if not N:
            N = self.data["z"].shape[1]
        ax.plot(self.data["z"][:, :N].T)

    def plot_hidden(self, ax, N=None):
        if not N:
            N = self.data["r"].shape[1]
        ax.plot(self.data["r"][:, :N].T)


def plot_n_units(r, n, steps=1000):
    plt.figure(figsize=(8, 8 * n / 10))
    offset = np.repeat(np.arange(1, 2 * n + 1, 2)[None, :], steps, axis=0)
    plt.plot(r[:, :n] + offset, color="blue")
    plt.ylim(0, 2*n)
    plt.hlines(np.arange(0, 2 * n, 2), 0, steps, color="Grey", linestyle=":")
