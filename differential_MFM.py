import gwyfile
from gwyfile.objects import GwyContainer, GwyDataField, GwySIUnit

import numpy as np
from skimage.registration import phase_cross_correlation as pcc
from scipy.ndimage import map_coordinates
from scipy.signal import fftconvolve

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib_scalebar.scalebar import ScaleBar

import warnings

from registration import *


class MFMData:
    """
    Takes a gwyddion file (.gwy) as an input and creates a new `MFMData` object.
        
    Parameters
    ----------
    file_path: str
        The file_path to the gwyddion file.
    """
    def __init__(self, file_path: str) -> None:
        self._file_path = file_path
        self.gwy_obj: gwyfile.objects.GwyObject = gwyfile.load(file_path)
        self.channels = gwyfile.util.get_datafields(self.gwy_obj)

        self.topography: np.ndarray = self.channels['ZSensor'].data
        self.phase: np.ndarray = self.channels['Phase'].data

        self.x_extents = (0, self.channels['ZSensor'].xreal)
        self.y_extents = (0, self.channels['ZSensor'].yreal)
        
        self.x_px_size = self.channels['ZSensor'].xreal/self.channels['ZSensor']['xres']
        self.y_px_size = self.channels['ZSensor'].yreal/self.channels['ZSensor']['yres']

    def reset(self) -> None:
        """
        Resets the object to its original state using the `file_path`,
        useful if the MFMData object has been externally modified.
        """
        if self._file_path is None:
            raise ValueError(
                "Cannot reset: this MFMData instance was not created from a file."
            )
        self.__init__(self._file_path)

    def plot(
            self, 
            channel: str, 
            ax: Axes, 
            color_range: tuple[float, float] | str | None = None, 
            color_map=None,
    ) -> mpl.image.AxesImage:
        """
        Plots the selected channel in the provided matplotlib Axes.\\
        The `Axes` can be obtained by calling `fig, ax = plt.subplots()`, where ax is the axes.

        Parameters
        ----------
        channel : str
            The channel to be plotted. Can be either "topography", or "phase".
        ax : mpl.axes.Axes
            The matplotlib Axes in which the image has to be plotted.
        color_range : tuple[float, float] | str, optional
            The range of values that are used for color mapping, given as `(vmin, vamx)` where vmin is the lowest, 
            and vmax is the highest.\\
            If "equal" is provided, then the minimum and maximum of the range is set such that `vmin == vmax`
            and encompasses all the values in the dataset.\\
            Defaults to None, i.e. default imshow behaviour of matplotlib.
        color_map : mpl.colors.Colormap | str
            Which colormap to use for displaying the image. Defaults to "afmhot".

        Returns
        -------
        img_ax : mpl.image.AxesImage
            The image object that is created by drawing the image in the axes.
        """
        match channel:
            case "topo" | "topography" | "t":   img_data = self.topography
            case "phase"| 'p':                  img_data = self.phase
            case _:
                raise Exception('Invalid channel!')

        if color_range == "equal":
            img_max = np.max(np.abs([np.min(img_data), np.max(img_data)]))
            color_range = (-img_max, img_max)
            print(f"Using color range (vmin, vmax): {color_range[0]: .4e}, {color_range[1]: .4e}")

        img_ax = ax.imshow(
            img_data,
            cmap = mpl.colormaps['afmhot'] if color_map is None else color_map,
            vmin = None if color_range is None else color_range[0],
            vmax = None if color_range is None else color_range[1],
            interpolation='none',
            extent=self.x_extents + self.y_extents,
        )

        ax.add_artist(
            ScaleBar(1, location='lower right', pad=0.25, border_pad=0.5, length_fraction=0.21,box_alpha=0.5)
        )

        ax.axis('off')

        return img_ax

class DifferentialMFMData:
    """
    Registers and subsequently performs differential MFM on the two datasets provided. 
    Registration is performed using the specified method.

    Parameters
    ----------
    init_data : MFMData
        MFMData object that contains data of the MFM scan with the tip in its initial magnetization state. phi_1
    aftr_data : MFMData
        MFMData object that contains data of the MFM scan with the tip in its reversed magnetization state. phi_2
    displacement_guess : tuple[int, int], optional
        A guess for the translational displacement between the two images, in pixels. Default is `(0, 0)`.
    method : str, optional
        The method used to register the two images. Default is `"hill descent"`.\\
            The list of available methods are "hill descent", "phase cross correlation".
    
    Raises
    ------
    ValueError
        If `init_data` and `aftr_data` have unequal pixel sizes.
    NotImplementedError
        If the specified `method` is not implemented.
    """
    def __init__(
        self,
        init_data: "MFMData",
        aftr_data: "MFMData",
        displacement_guess: tuple[int, int] = (0, 0),
        method: str = "hill descent",
    ) -> None:

        if (init_data.x_px_size != aftr_data.x_px_size) or (init_data.y_px_size != aftr_data.y_px_size):
            raise ValueError(
                "Cannot perform differential MFM on datasets which have unequal pixel sizes."
            )
        
        self._file_path = None
        self.gwy_obj = None
        self.channels = None
        self.init_data = init_data
        self.aftr_data = aftr_data
        self.cost_map = None

        match method:
            case "hill descent" | "hd":
                self.method = "hill descent"
                self._max_itr = (init_data.topography.shape[0] + init_data.topography.shape[1])//4
                
                self.registration_result = match_image_by_hill_descent(
                    displacement_guess,
                    init_data.topography, aftr_data.topography,
                    iterations=self._max_itr
                )
        
                init_slice, aftr_slice = generate_slices(self.registration_result['displacement'])
                self.init_slice, self.aftr_slice = init_slice, aftr_slice

                self.topography = np.copy(init_data.topography[init_slice])
                self.diff_topo = init_data.topography[init_slice] - aftr_data.topography[aftr_slice]

                self.phi_1 = init_data.phase[init_slice]
                self.phi_2 = aftr_data.phase[aftr_slice]
                self.phase = self.phi_1 - self.phi_2
                self.sum_phase = self.phi_1 + self.phi_2
            
            case "phase cross correlation" | "pcc":
                self.method = "phase cross correlation"
                pcc_result = pcc(init_data.topography, aftr_data.topography, upsample_factor=100)
                self.registration_result = {
                    "displacement": np.array([pcc_result[0][1], pcc_result[0][0]]),
                    "pcc_error": pcc_result[1],
                    "pcc_phase_diff": pcc_result[2],
                }

                dy, dx = pcc_result[0]
                m, n = init_data.topography.shape

                y, x = np.meshgrid(np.arange(m), np.arange(n), indexing='ij')
                coords = np.array([y - dy, x - dx])
                aftr_topo_registered = map_coordinates(aftr_data.topography, coords, order=3, mode='nearest')
                aftr_phase_registered = map_coordinates(aftr_data.phase, coords, order=3, mode='nearest')

                shift = (round(dx), round(dy))
                init_slice, aftr_slice = generate_slices(shift)
                self.init_slice, self.aftr_slice = init_slice, None

                self.topography = np.copy(init_data.topography[init_slice])

                # Since aftr_data has been registered, after_slice is not needed, only init_slice.
                self.diff_topo = init_data.topography[init_slice] - aftr_topo_registered[init_slice]

                self.phi_1 = init_data.phase[init_slice]
                self.phi_2 = aftr_phase_registered[init_slice]
                self.phase = self.phi_1 - self.phi_2
                self.sum_phase = self.phi_1 + self.phi_2

            case _:
                raise NotImplementedError(
                    f"Method \"{method}\" has not been implemented."
                )

        self.registration_result['rms_residual'] = np.sqrt(np.mean(self.diff_topo**2))

        # Sanity check
        init_roughness = np.sqrt(np.mean((init_data.topography - init_data.topography.mean())**2))
        if self.registration_result['rms_residual'] > init_roughness:
            warnings.warn(
                "RMS residual of the registration is greater than the roughness of the `init_data`. Please check diff_topo!!",
                UserWarning,
                stacklevel=2
            )

        self.x_extents = (0, init_data.x_px_size * self.topography.shape[1])
        self.y_extents = (0, init_data.y_px_size * self.topography.shape[0])

        self.x_px_size = init_data.x_px_size
        self.y_px_size = init_data.y_px_size

    def save_gwy(self, filename: str) -> None:
        """
        Creates and saves the differential MFM data in a gwyddion (.gwy) file
        with the given filename.

        Parameters
        ----------
        filename : str
            Given filename. The `.gwy` extension is appended automatically if
            not already present.

        Returns
        -------
        None
        """
        if not filename.endswith('.gwy'):
            filename += ".gwy"

        gwy_obj = GwyContainer()

        gwy_obj['/0/data/title'] = 'Topography'
        gwy_obj['/0/data'] = GwyDataField(
            np.copy(self.topography), 
            xreal=self.x_extents[1],
            yreal=self.y_extents[1],
            si_unit_xy=GwySIUnit(unitstr="m"),
            si_unit_z=GwySIUnit(unitstr="m")
        )

        gwy_obj['/1/data/title'] = 'Differential Phase'
        gwy_obj['/1/data'] = GwyDataField(
            np.copy(self.phase), 
            xreal=self.x_extents[1],
            yreal=self.y_extents[1],
            si_unit_xy=GwySIUnit(unitstr="m"),
            si_unit_z=GwySIUnit(unitstr="°"),
        )

        gwy_obj['/2/data/title'] = 'Summed Phase'
        gwy_obj['/2/data'] = GwyDataField(
            np.copy(self.sum_phase), 
            xreal=self.x_extents[1],
            yreal=self.y_extents[1],
            si_unit_xy=GwySIUnit(unitstr="m"),
            si_unit_z=GwySIUnit(unitstr="°"),
        )

        gwy_obj['/3/data/title'] = 'Original Phase - 1'
        gwy_obj['/3/data'] = GwyDataField(
            np.copy(self.phi_1), 
            xreal=self.x_extents[1],
            yreal=self.y_extents[1],
            si_unit_xy=GwySIUnit(unitstr="m"),
            si_unit_z=GwySIUnit(unitstr="°"),
        )

        gwy_obj['/4/data/title'] = 'Original Phase - 2'
        gwy_obj['/4/data'] = GwyDataField(
            np.copy(self.phi_2), 
            xreal=self.x_extents[1],
            yreal=self.y_extents[1],
            si_unit_xy=GwySIUnit(unitstr="m"),
            si_unit_z=GwySIUnit(unitstr="°"),
        )

        gwy_obj.tofile(filename)

    def plot(
            self,
            channel: str,
            ax : Axes,
            color_range: tuple[float, float] | str | None = None,
            color_map=None
        ) -> mpl.image.AxesImage:
        """
        Plots the selected channel in the provided matplotlib Axes.\\
        The `Axes` can be obtained by calling `fig, ax = plt.subplots()`, where ax is the axes.

        Parameters
        ----------
        channel : str
            The channel to be plotted. Can be either "topography", "phase", or "diff_phase".
        ax : mpl.axes.Axes
            The matplotlib Axes in which the image has to be plotted.
        color_range : tuple[float, float] | str, optional
            The range of values that are used for color mapping, given as `(vmin, vamx)` where vmin is the lowest, 
            and vmax is the highest.\\
            If "equal" is provided, then the minimum and maximum of the range is set such that `vmin == vmax`
            and encompasses all the values in the dataset.\\
            Defaults to None, i.e. default imshow behaviour of matplotlib.
        color_map : mpl.colors.Colormap | str
            Which colormap to use for displaying the image. Defaults to "afmhot".

        Returns
        -------
        img_ax : mpl.image.AxesImage
            The image object that is created by drawing the image in the axes.
        """
        match channel:
            case "topo" | "topography" | "t":   img_data = self.topography
            case "phase"| 'p':                  img_data = self.phase
            case "diff_topo" | "dtopo" | "dt":  img_data = self.diff_topo 
            case _:
                raise Exception('Invalid channel!')

        if color_range == "equal":
            img_max = np.max(np.abs([np.min(img_data), np.max(img_data)]))
            color_range = (-img_max, img_max)
            print(f"Using color range (vmin, vmax): {color_range[0]: .4e}, {color_range[1]: .4e}")

        img_ax = ax.imshow(
            img_data,
            cmap = mpl.colormaps['afmhot'] if color_map is None else color_map,
            vmin = None if color_range is None else color_range[0],
            vmax = None if color_range is None else color_range[1],
            interpolation='none',
            extent=self.x_extents + self.y_extents,
        )

        ax.add_artist(
            ScaleBar(1, location='lower right', pad=0.25, border_pad=0.5, length_fraction=0.21,box_alpha=0.5)
        )

        ax.axis('off')

        return img_ax

    def gen_cost_map(self, del_px: int = 50) -> None:
        """
        Generates a 2D cost map of the topography registration cost function
        over a range of pixel displacements, storing the result in
        `self.cost_map`. Computed via FFT cross-correlation rather than a
        per-pixel loop.

        Parameters
        ----------
        del_px : int, optional
            The maximum pixel displacement (in both x and y) to evaluate around
            the origin. The resulting cost map will have shape\\
            `(2 * del_px + 1, 2 * del_px + 1)`. Default is `50`.

        Raises
        ------
        IndexError
            If `del_px` exceeds either dimension of `self.topography`.

        Returns
        -------
        None
        """
        print('Generating cost map.')
        if del_px > self.topography.shape[0]:
            raise IndexError(
                f"del_px: {del_px} > vertical image resolution: {self.topography.shape[0]}"
            )
        if del_px > self.topography.shape[1]:
            raise IndexError(
                f"del_px: {del_px} > horizontal image resolution: {self.topography.shape[1]}"
            )

        I = self.init_data.topography
        A = self.aftr_data.topography
        m, n = I.shape

        ones_I = np.ones_like(I)
        ones_A = np.ones_like(A)

        # Cross-correlation via fftconvolve: correlate(a, b) = convolve(a, flip(b))
        S_II = fftconvolve(I**2, ones_A[::-1, ::-1], mode='full')
        S_AA = fftconvolve(ones_I, (A**2)[::-1, ::-1], mode='full')
        S_IA = fftconvolve(I, A[::-1, ::-1], mode='full')
        N = fftconvolve(ones_I, ones_A[::-1, ::-1], mode='full')

        cost_full = (S_II + S_AA - 2 * S_IA) / N

        # Zero displacement sits at (m-1, n-1) in 'full' correlation output.
        cy, cx = m - 1, n - 1
        y_slice = slice(cy - del_px, cy + del_px + 1)
        x_slice = slice(cx - del_px, cx + del_px + 1)

        # Original loop indexed cost_map as [x_disp_idx, y_disp_idx]; transpose
        # to match that convention.
        self.cost_map = cost_full[y_slice, x_slice].T

    def plot_cost_map(
            self,
            ax: Axes,
            del_px: int = 50,
            color_range: tuple[float, float] | str | None = None,
            color_map=None,
        ) -> mpl.image.AxesImage:
        """
        Plots the registration cost map in the provided matplotlib `Axes`,
        with contour lines overlaid. Generates the cost map first via
        `gen_cost_map` if it hasn't been computed yet, or if it was computed
        with a different `del_px`.

        Parameters
        ----------
        ax : mpl.axes.Axes
            The matplotlib `Axes` in which the cost map has to be plotted.
        del_px : int, optional
            The maximum pixel displacement (in both x and y) to evaluate around
            the origin. Default is `50`.
        color_range : tuple[float, float] | str, optional
            The range of values that are used for color mapping, given as
            `(vmin, vmax)` where vmin is the lowest, and vmax is the highest.
            Defaults to `None`, i.e. default imshow behaviour of matplotlib.
        color_map : mpl.colors.Colormap | str, optional
            Which colormap to use for displaying the image. Defaults to
            `mpl.colormaps['gist_earth']`.

        Returns
        -------
        img_ax : mpl.image.AxesImage
            The image object that is created by drawing the cost map in the
            axes.
        """
        if (self.cost_map is None):
            self.gen_cost_map(del_px)
        elif (self.cost_map.shape[0] != 2 * del_px + 1):
            self.gen_cost_map(del_px)
            
        plot_extents = (-del_px, del_px, -del_px, del_px)
        img_ax = ax.imshow(
            self.cost_map,
            cmap = mpl.colormaps['gist_earth'] if color_map is None else color_map,
            vmin = None if color_range is None else color_range[0],
            vmax = None if color_range is None else color_range[1],
            interpolation='none',
            extent=plot_extents,
        )

        contour_levels = np.linspace(self.cost_map.min(), self.cost_map.max(), 10)
        ax.contour(
            self.cost_map,
            extent=plot_extents,
            levels=contour_levels,
            alpha=0.5,
            origin='upper',
            colors='black', linewidths=1
        )
        
        return img_ax
    