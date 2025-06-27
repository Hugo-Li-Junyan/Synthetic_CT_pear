import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button


def plot_volume(volume, plane='XY'):
    assert plane in ['XY', 'xy', 'XZ', 'xz', 'YZ', 'yz'], 'Must specify the plane to visualize'
    # Initial setup
    third_axs_idx = 0
    fig, ax = plt.subplots()
    label = None
    valmax = None
    if plane in ['XY', 'xy']:
        label = 'Z'
        valmax = volume.shape[0] - 1
        ax.imshow(volume[third_axs_idx, :, :], cmap='gray', vmin=volume.min(), vmax=volume.max())
    elif plane in ['XZ', 'xz']:
        label = 'Y'
        valmax = volume.shape[1] - 1
        ax.imshow(volume[:, third_axs_idx, :], cmap='gray', vmin=volume.min(), vmax=volume.max())
    elif plane in ['YZ', 'yz']:
        label = 'X'
        valmax = volume.shape[2] - 1
        ax.imshow(volume[:, :, third_axs_idx], cmap='gray', vmin=volume.min(), vmax=volume.max())

    # adjust the main plot to make room for the sliders
    fig.subplots_adjust(left=0.25, bottom=0.25)

    # Make a horizontal slider for Z axis.
    ax_third = fig.add_axes([0.25, 0.1, 0.65, 0.03])
    allowed_z = range(volume.shape[0])
    third_axs_slider = Slider(
        ax=ax_third,
        label=label,
        valmin=0,
        valmax=valmax,
        valstep=allowed_z,
        valinit=third_axs_idx,
    )

    def update(val):
        ax.clear()
        if plane in ['XY', 'xy']:
            ax.imshow(volume[third_axs_slider.val, :, :], cmap='gray', vmin=volume.min(), vmax=volume.max())
        elif plane in ['XZ', 'xz']:
            ax.imshow(volume[:, third_axs_slider.val, :], cmap='gray', vmin=volume.min(),
                      vmax=volume.max())
        elif plane in ['YZ', 'yz']:
            ax.imshow(volume[:, :, third_axs_slider.val], cmap='gray', vmin=volume.min(),
                      vmax=volume.max())

        fig.canvas.draw_idle()

    third_axs_slider.on_changed(update)

    resetax = fig.add_axes([0.8, 0.025, 0.1, 0.04])
    button = Button(resetax, 'Reset', hovercolor='0.975')

    def reset(event):
        third_axs_slider.reset()
    button.on_clicked(reset)
    plt.show()


def plot_spectral_volume(volume, plane='XY'):
    assert plane in ['XY', 'xy', 'XZ', 'xz', 'YZ', 'yz'], 'Must specify the plane to visualize'
    # Initial setup
    energy_idx = 0
    third_axs_idx = 0
    fig, ax = plt.subplots()
    label = None
    valmax = None
    if plane in ['XY', 'xy']:
        label = 'Z'
        valmax = volume.shape[1] - 1
        ax.imshow(volume[energy_idx, third_axs_idx, :, :], cmap='gray', vmin=volume.min(), vmax=volume.max())
    elif plane in ['XZ', 'xz']:
        label = 'Y'
        valmax = volume.shape[2] - 1
        ax.imshow(volume[energy_idx, :, third_axs_idx, :], cmap='gray', vmin=volume.min(), vmax=volume.max())
    elif plane in ['YZ', 'yz']:
        label = 'X'
        valmax = volume.shape[3] - 1
        ax.imshow(volume[energy_idx, :, :, third_axs_idx], cmap='gray', vmin=volume.min(), vmax=volume.max())

    # adjust the main plot to make room for the sliders
    fig.subplots_adjust(left=0.25, bottom=0.25)

    # Make a horizontal slider for Z axis.
    ax_third = fig.add_axes([0.25, 0.1, 0.65, 0.03])
    allowed_z = range(volume.shape[1])
    third_axs_slider = Slider(
        ax=ax_third,
        label=label,
        valmin=0,
        valmax=valmax,
        valstep=allowed_z,
        valinit=third_axs_idx,
    )

    # Make a vertical slider for energy level axis
    ax_energy = fig.add_axes([0.1, 0.25, 0.0225, 0.63])
    allowed_energy = range(volume.shape[0])
    energy_slider = Slider(
        ax=ax_energy,
        label="Energy (KeV)",
        valmin=0,
        valmax=volume.shape[0]-1,
        valstep=allowed_energy,
        valinit=energy_idx,
        orientation="vertical"
    )

    def update(val):
        ax.clear()
        if plane in ['XY', 'xy']:
            ax.imshow(volume[energy_slider.val, third_axs_slider.val, :, :], cmap='gray', vmin=volume.min(), vmax=volume.max())
        elif plane in ['XZ', 'xz']:
            ax.imshow(volume[energy_slider.val, :, third_axs_slider.val, :], cmap='gray', vmin=volume.min(),
                      vmax=volume.max())
        elif plane in ['YZ', 'yz']:
            ax.imshow(volume[energy_slider.val, :, :, third_axs_slider.val], cmap='gray', vmin=volume.min(),
                      vmax=volume.max())

        fig.canvas.draw_idle()

    third_axs_slider.on_changed(update)
    energy_slider.on_changed(update)

    resetax = fig.add_axes([0.8, 0.025, 0.1, 0.04])
    button = Button(resetax, 'Reset', hovercolor='0.975')

    def reset(event):
        third_axs_slider.reset()
        energy_slider.reset()
    button.on_clicked(reset)
    plt.show()


def plot_spectral_projection(projection):
    energy_idx = 0
    fig, ax = plt.subplots()
    ax.imshow(projection[energy_idx, :, :], cmap='gray', vmin=projection.min(), vmax=projection.max())

    # adjust the main plot to make room for the sliders
    fig.subplots_adjust(left=0.25)

    # Make a vertical slider for energy level axis
    ax_energy = fig.add_axes([0.1, 0.25, 0.0225, 0.63])
    allowed_energy = range(projection.shape[0])
    energy_slider = Slider(
        ax=ax_energy,
        label="Energy (KeV)",
        valmin=0,
        valmax=projection.shape[0]-1,
        valstep=allowed_energy,
        valinit=energy_idx,
        orientation="vertical"
    )

    def update(val):
        ax.clear()
        ax.imshow(projection[energy_slider.val, :, :], cmap='gray', vmin=projection.min(), vmax=projection.max())
        fig.canvas.draw_idle()

    energy_slider.on_changed(update)

    resetax = fig.add_axes([0.8, 0.025, 0.1, 0.04])
    button = Button(resetax, 'Reset', hovercolor='0.975')

    def reset(event):
        energy_slider.reset()
    button.on_clicked(reset)
    plt.show()


if __name__ == '__main__':
    pass
