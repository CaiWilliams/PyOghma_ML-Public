import matplotlib.pyplot as plt
import numpy as np


class Figures:
    def __init__(self):
        self.fig, self.ax = plt.subplots()

    def initialise_figure(self, *args, **kwargs):
        self.fig, self.ax = plt.subplots(*args, **kwargs)
        if np.shape(self.ax) == ():
            self.active_ax = self.ax

    def set_subfigure(self, row, cols):
        if np.shape(self.ax) == ():
            raise IndexError("This figure does not have sub-figures")
        if len(np.shape(self.ax)) == 1:
            if row == 0 and cols != 0:
                self.active_ax = self.ax[cols]
            elif row != 0 and cols == 0:
                self.active_ax = self.ax[row]
            elif row == 0 and cols == 0:
                self.active_ax = self.ax[cols]
        else:
            self.active_ax = self.ax[row, cols]

    # Base Plots
    def scatter(self, *args, **kwargs):
        self.active_plot = self.active_ax.scatter(*args, **kwargs)

    def plot(self, *args, **kwargs):
        self.active_plot = self.active_ax.plot(*args, **kwargs)

    def hist(self, *args, **kwargs):
        self.active_plot = self.active_ax.hist(*args, **kwargs)

    def hist2d(self, *args, **kwargs):
        self.active_plot = self.active_ax.hist2d(*args, **kwargs)

    # Axes Manipulation
    def set_x_limits(self, **kwargs):
        self.active_ax.set_xlim(**kwargs)

    def set_y_limits(self, **kwargs):
        self.active_ax.set_ylim(**kwargs)

    def set_x_ticks(self, *args, **kwargs):
        self.active_ax.set_xticks(*args, **kwargs)

    def set_y_ticks(self, *args, **kwargs):
        self.active_ax.set_yticks(*args, **kwargs)

    def set_x_label(self, *args, **kwargs):
        self.active_ax.set_xlabel(*args, **kwargs)

    def set_y_label(self, *args, **kwargs):
        self.active_ax.set_ylabel(*args, **kwargs)

    # Extras
    def colorbar(self, *args, **kwargs):
        self.fig.colorbar(self.active_plot[3], *args, **kwargs)

    def set_title(self, *args, **kwargs):
        self.active_ax.set_title(*args, **kwargs)

    def save_to_disk(self, *args, **kwargs):
        self.fig.savefig(*args, **kwargs)

    def show(self):
        plt.show()


if __name__ == "__main__":
    F = Figures()
    F.initialise_figure(figsize=(5, 5))

    #F.set_subfigure(0, 0)
    F.hist2d(np.random.normal(0,1,1000),np.random.normal(0,1,1000))
    F.set_x_limits(left=-2, right=2)
    F.set_x_ticks(np.linspace(-2,2,6))
    F.set_y_label('y-axis')
    F.set_x_label('x-axis')
    F.colorbar(label='colorbar')
    F.set_title('Title')
    F.save_to_disk('fig.png', dpi=300)
