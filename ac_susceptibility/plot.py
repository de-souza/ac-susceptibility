#
#    Copyright 2018 Léo De Souza
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
#
import numpy as np
import matplotlib.pyplot as plt

from .load import load
from .xyfit import xyfit


def plot(data_path, skip_voltage, calibration_data):
    """Plot magnetization and/or voltage from measurement files."""
    init_matplotlib()

    input_folder = data_path / "input"
    output_folder = data_path / "output"

    for measurement_folder in input_folder.iterdir():

        magnetization_data = []

        for temperature_folder in list_subfolders(measurement_folder):

            freqs_and_files = list_freqs_and_files(temperature_folder)
            temperature_data = np.empty((len(freqs_and_files), 4), dtype=np.complex_)

            for i, (freq, voltage_file) in enumerate(freqs_and_files):

                data = load(voltage_file)
                fit, pfit = xyfit(data, calibration_data)
                temperature_data[i] = freq, *pfit[:3]

                if not skip_voltage:
                    voltage_plot_path = (
                        output_folder
                        / "voltage"
                        / measurement_folder.name
                        / temperature_folder.name
                        / (voltage_file.stem + ".pdf")
                    )
                    make_voltage_plot(data, fit, voltage_plot_path)

            magnetization_data.append([temperature_folder.name, temperature_data])

        magnetization_plot_xy_path = (
            output_folder / "magnetization" / (measurement_folder.name + "_xy.pdf")
        )
        magnetization_plot_polar_path = (
            output_folder / "magnetization" / (measurement_folder.name + "_polar.pdf")
        )
        make_magnetization_plot_xy(magnetization_data, magnetization_plot_xy_path)
        make_magnetization_plot_polar(magnetization_data, magnetization_plot_polar_path)


def init_matplotlib():
    """Set plot style."""
    plt.style.use("seaborn-notebook")
    plt.rc("lines", linewidth=1)
    plt.rc("grid", linewidth=0.1)


def list_subfolders(folder):
    """Return list of subfolders sorted by temperature.

    Args:
        folder: A parent folder as a "Path" object.

    Returns:
        A list of "Path" objects corresponding to the sorted temperature
        subfolders.

    """
    subfolders = [entry for entry in folder.iterdir() if entry.is_dir()]
    subfolders = sorted(subfolders, key=lambda x: int(x.name.split("K")[0]))
    return subfolders


def list_freqs_and_files(folder):
    """Return list of "fequency, file" tuples within a folder.

    Args:
        folder: A parent folder as a "Path" object.

    Returns:
        A list of (freq, file) tuples, where 'freq' is a string of the
        frequency and 'file' is the file's 'Path' object.

    """
    files = [entry for entry in folder.iterdir() if entry.name.endswith("txt")]
    freqs = [float(entry.stem[:-2]) for entry in files]
    return sorted(zip(freqs, files))


def make_voltage_plot(data, fit, path):
    """Plot amplitude and phase of the signal versus sample position.

    Args:
        data: An array containing the position of the sample, the
            amplitude of the voltage measured by the lock-in amplifier,
            and its phase.
        fit: Same object as "data" with fitted values.
        path: The path of the saved plot as a "Path" object.

    """
    print(f'Saving plot "{path}"...')

    path.parent.mkdir(parents=True, exist_ok=True)

    pos = data[:, 0]
    if pos[-1] < 0:
        pos = -pos

    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(11, 4))

    _draw_ax(
        ax1,
        pos,
        data[:, 1] * 1000,
        fit[:, 1] * 1000,
        title="X Channel",
        xlabel="Position (mm)",
        ylabel="Voltage (mV)",
        legend=["Fit", "Data"],
        ncol=1,
    )

    _draw_ax(
        ax2,
        pos,
        data[:, 2] * 1000,
        fit[:, 2] * 1000,
        title="Y Channel",
        xlabel="Position (mm)",
        ylabel="Voltage (mV)",
        legend=["Fit", "Data"],
        ncol=1,
    )

    fig.tight_layout()
    fig.savefig(path.as_posix(), dpi=300)
    plt.close()


def make_magnetization_plot_xy(data, path):
    """Plot amplitude and phase of the signal versus frequency.

    Args:
        data: An array or a list of arrays containing tuples of:
            - A string of the temperature.
            - The position of the sample, the amplitude of the baseline
                and each peak, and their phase.
        path: The path of the saved plot as a "Path" object.

    """
    print(f'Saving plot "{path.name}"...')

    path.parent.mkdir(parents=True, exist_ok=True)

    temp = []
    curves = [[], [], [], [], [], [], []]

    for i, j in data:
        temp.append(i)
        curves[0].append(j[:, 0].real)
        curves[1].append((j[:, 1] / j[:, 0]).real * 1e6)
        curves[2].append((j[:, 2] / j[:, 0]).real * 1e6)
        curves[3].append((j[:, 3] / j[:, 0]).real * 1e6)
        curves[4].append(j[:, 1].imag / j[:, 0].real * 1e6)
        curves[5].append(j[:, 2].imag / j[:, 0].real * 1e6)
        curves[6].append(j[:, 3].imag / j[:, 0].real * 1e6)

    fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(12, 12))

    _draw_ax(
        ax1,
        curves[0],
        curves[1],
        title="X Baseline",
        xlabel="Frequency (Hz)",
        ylabel="Voltage / Frequency (mV / kHz)",
        xscale="log",
        legend=temp,
    )

    _draw_ax(
        ax2,
        curves[0],
        curves[4],
        title="Y Baseline",
        xlabel="Frequency (Hz)",
        ylabel="Voltage / Frequency (mV / kHz)",
        xscale="log",
        legend=temp,
    )

    _draw_ax(
        ax3,
        curves[0],
        curves[2],
        title="X Peak #1",
        xlabel="Frequency (Hz)",
        ylabel="Voltage / Frequency (mV / kHz)",
        xscale="log",
        legend=temp,
    )

    _draw_ax(
        ax4,
        curves[0],
        curves[5],
        title="Y Peak #1",
        xlabel="Frequency (Hz)",
        ylabel="Voltage / Frequency (mV / kHz)",
        xscale="log",
        legend=temp,
    )

    _draw_ax(
        ax5,
        curves[0],
        curves[3],
        title="X Peak #2",
        xlabel="Frequency (Hz)",
        ylabel="Voltage / Frequency (mV / kHz)",
        xscale="log",
        legend=temp,
    )

    _draw_ax(
        ax6,
        curves[0],
        curves[6],
        title="Y Peak #2",
        xlabel="Frequency (Hz)",
        ylabel="Voltage / Frequency (mV / kHz)",
        xscale="log",
        legend=temp,
    )

    fig.suptitle(path.name, weight="bold", size="x-large")
    fig.tight_layout(h_pad=3, w_pad=3, rect=(0, 0, 1, 0.95))
    fig.savefig(path.as_posix(), dpi=300)
    plt.close()


def make_magnetization_plot_polar(data, path):
    """Plot amplitude and phase of the signal versus frequency.

    Args:
        data: An array or a list of arrays containing tuples of:
            - A string of the temperature.
            - The position of the sample, the amplitude of the baseline
                and each peak, and their phase.
        path: The path of the saved plot as a "Path" object.

    """
    print(f'Saving plot "{path.name}"...')

    temp = []
    curves = [[], [], [], [], [], [], []]

    for i, j in data:
        temp.append(i)
        curves[0].append(j[:, 0].real)
        curves[1].append(np.abs(j[:, 1] / j[:, 0]) * 1e6)
        curves[2].append(np.abs(j[:, 2] / j[:, 0]) * 1e6)
        curves[3].append(np.abs(j[:, 3] / j[:, 0]) * 1e6)
        curves[4].append(np.unwrap(np.angle(j[:, 1], deg=True), 180))
        curves[5].append(np.unwrap(np.angle(j[:, 2], deg=True), 180))
        curves[6].append(np.unwrap(np.angle(j[:, 3], deg=True), 180))

    fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(12, 12))

    _draw_ax(
        ax1,
        curves[0],
        curves[1],
        title="Amplitude Baseline",
        xlabel="Frequency (Hz)",
        ylabel="Voltage / Frequency (mV / kHz)",
        xscale="log",
        legend=temp,
    )

    _draw_ax(
        ax2,
        curves[0],
        curves[4],
        title="Phase Baseline",
        xlabel="Frequency (Hz)",
        ylabel="Phase (°)",
        xscale="log",
        legend=temp,
    )

    _draw_ax(
        ax3,
        curves[0],
        curves[2],
        title="Amplitude Peak #1",
        xlabel="Frequency (Hz)",
        ylabel="Voltage / Frequency (mV / kHz)",
        xscale="log",
        legend=temp,
    )

    _draw_ax(
        ax4,
        curves[0],
        curves[5],
        title="Phase Peak #1",
        xlabel="Frequency (Hz)",
        ylabel="Phase (°)",
        xscale="log",
        legend=temp,
    )

    _draw_ax(
        ax5,
        curves[0],
        curves[3],
        title="Amplitude Peak #2",
        xlabel="Frequency (Hz)",
        ylabel="Voltage / Frequency (mV / kHz)",
        xscale="log",
        legend=temp,
    )

    _draw_ax(
        ax6,
        curves[0],
        curves[6],
        title="Phase Peak #2",
        xlabel="Frequency (Hz)",
        ylabel="Phase (°)",
        xscale="log",
        legend=temp,
    )

    fig.suptitle(path.name, weight="bold", size="x-large")
    fig.tight_layout(h_pad=3, w_pad=3, rect=(0, 0, 1, 0.95))
    fig.savefig(path.as_posix(), dpi=300)
    plt.close()


def _draw_ax(axes, freq, data=None, fit=None, **kwargs):
    """Draws plots using the matplotlib.axes.Axes class.

    Args:
        axes: The "Axes" object where the plots are drawn.
        freq: An array or list of arrays of the x values.
        data: An array or list of arrays of y values plotted with
            markers.
        fit: An array or list of arrays of y values plotted with lines.
        kwargs: Optional keyword arguments:
            - title: The title string.
            - xlabel: The xlabel strings.
            - ylabel: The ylabel strings.
            - xscale: ['linear' | 'log' | 'logit' | 'symlog']
            - legend: An iterable of strings for the legend.
            - legend_ncol: The number of columns in the legend.

    """
    if not isinstance(freq, list):
        freq = [freq]

    if fit is not None:
        if not isinstance(fit, list):
            fit = [fit]
        for i, j in zip(freq, fit):
            axes.plot(i, j)

    if data is not None:
        if not isinstance(data, list):
            data = [data]
        for i, j in zip(freq, data):
            axes.plot(i, j, marker=".")

    if "legend" in kwargs:
        axes.legend(kwargs["legend"], ncol=kwargs.get("ncol", 2))

    axes.set_xlabel(kwargs.get("xlabel", ""))
    axes.set_ylabel(kwargs.get("ylabel", ""))

    axes.set_xscale(kwargs.get("xscale", "linear"))
    axes.set_xticklabels([f"{i:g}" for i in axes.get_xticks()])

    axes.set_title(kwargs.get("title", ""))
    axes.grid(True)
