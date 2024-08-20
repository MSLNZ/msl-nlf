"""
Load a **.nlf** file.
"""
import math
from struct import unpack

import numpy as np

ansi = {'encoding': 'ansi', 'errors': 'replace'}


class Loader:

    def __init__(self, path: str) -> None:
        """Helper class to read a **.nlf** file.

        Parameters
        ----------
        path
            The path to a **.nlf** file.
        """
        with open(path, mode='rb') as fp:
            self._data = fp.read()
        self._offset = 0
        self._view = memoryview(self._data)
        self.version = float(self.read_string_padded(10))

    def _read(self, fmt: str, size: int):
        i = self._offset
        j = i + size
        value, = unpack(fmt, self._view[i:j])
        self._offset = j
        return value

    def read_boolean(self) -> bool:
        """Read a boolean."""
        return self._read('?', 1)

    def read_byte(self) -> int:
        """Read a byte."""
        return self._read('b', 1)

    def read_extended(self) -> float:
        """Read a Delphi 10-byte extended float."""
        # 80-bit (10-byte), floating-point value according to the IEEE-754 specification
        # https://en.wikipedia.org/wiki/Extended_precision#x86_extended_precision_format
        #   sign=bit[79]          1 bit
        #   exponent=bit[64:79]  15 bit
        #   integer=bit[63]       1 bit
        #   fraction=bit[:63]    63 bit
        # read integer+fraction as uint64
        # read sign+exponent as uint16
        uint64 = self._read('Q', 8)
        uint16 = self._read('H', 2)
        sign = uint16 >> 15
        exponent = uint16 & 0x7FFF
        integer = uint64 >> 63
        fraction = uint64 & 0x7FFFFFFFFFFFFFFF
        if exponent < 0x7FFF:
            m = integer + fraction / (1 << 63)
            return (-1) ** sign * m * 2.0 ** (exponent - 16383)
        if fraction == 0:
            return (-1) ** sign * math.inf
        return math.nan

    def read_integer(self) -> int:
        """Read an unsigned integer."""
        return self._read('I', 4)

    def read_string(self) -> str:
        """Read a string."""
        length = self.read_integer()
        return self._read(f'{length}s', length).decode(**ansi)

    def read_string_padded(self, length: int) -> str:
        """Read a string that is null padded.

        Parameters
        ----------
        length
            The total length of the null-padded string.
        """
        n = self.read_byte()
        buffer = self._read(f'{length}s', length)
        return buffer[:n].decode(**ansi)

    def read_word(self) -> int:
        """Read an unsigned short."""
        return self._read('H', 2)


def load_graph(loader: Loader) -> dict:
    """Load a *TGraphWindow*.

    Parameters
    ----------
    loader
        The class helper.

    Returns
    -------
    dict
        The settings of the *TGraphWindow*.
    """
    # See the repository "Nonlinear-Fitting/NLFGraph.pas"
    # procedure TGraphWindow.LoadFile(TheStream:TStream);
    graph = dict()
    graph['win_left'] = loader.read_integer()
    graph['left'] = graph['win_left']
    graph['win_top'] = loader.read_integer()
    graph['top'] = graph['win_top']
    graph['win_width'] = loader.read_integer()
    graph['width'] = graph['win_width']
    graph['win_height'] = loader.read_integer()
    graph['height'] = graph['win_height']
    graph['the_state'] = loader.read_byte()
    graph['num_curves'] = loader.read_integer()
    graph['x_axis_scale_to_window'] = loader.read_boolean()
    graph['y_axis_scale_to_window'] = loader.read_boolean()
    graph['x_axis_length'] = loader.read_string_padded(20)
    graph['y_axis_length'] = loader.read_string_padded(20)
    graph['x_min'] = loader.read_extended()
    graph['x_max'] = loader.read_extended()
    graph['y_min'] = loader.read_extended()
    graph['y_max'] = loader.read_extended()
    graph['x_interval'] = loader.read_extended()
    graph['y_interval'] = loader.read_extended()
    graph['x_num_places'] = loader.read_integer()
    graph['y_num_places'] = loader.read_integer()
    graph['x_num_minor_ticks'] = loader.read_integer()
    graph['y_num_minor_ticks'] = loader.read_integer()
    graph['x_major_tick_size'] = loader.read_integer()
    graph['x_minor_tick_size'] = loader.read_integer()
    graph['y_major_tick_size'] = loader.read_integer()
    graph['y_minor_tick_size'] = loader.read_integer()
    graph['x_max_auto'] = loader.read_word()
    graph['x_min_auto'] = loader.read_word()
    graph['x_inc_auto'] = loader.read_word()
    graph['x_ticks_auto'] = loader.read_word()
    graph['x_dec_auto'] = loader.read_word()
    graph['y_max_auto'] = loader.read_word()
    graph['y_min_auto'] = loader.read_word()
    graph['y_inc_auto'] = loader.read_word()
    graph['y_ticks_auto'] = loader.read_word()
    graph['y_dec_auto'] = loader.read_word()
    graph['x_major_grid'] = loader.read_word()
    graph['x_minor_grid'] = loader.read_word()
    graph['y_major_grid'] = loader.read_word()
    graph['y_minor_grid'] = loader.read_word()
    graph['x_max_disp_places'] = loader.read_integer()
    graph['x_min_disp_places'] = loader.read_integer()
    graph['x_inc_disp_places'] = loader.read_integer()
    graph['y_max_disp_places'] = loader.read_integer()
    graph['y_min_disp_places'] = loader.read_integer()
    graph['y_inc_disp_places'] = loader.read_integer()
    graph['empty'] = loader.read_boolean()
    graph['axis_max_x'] = loader.read_integer()  # TPoint.x
    graph['axis_max_y'] = loader.read_integer()  # TPoint.y
    graph['axis_min_x'] = loader.read_integer()  # TPoint.x
    graph['axis_min_y'] = loader.read_integer()  # TPoint.y
    if loader.version >= 4.1:
        graph['x_title_auto'] = loader.read_boolean()
        graph['y_title_auto'] = loader.read_boolean()
    else:
        graph['x_title_auto'] = True
        graph['y_title_auto'] = True
    graph['x_title'] = loader.read_string_padded(255)
    graph['y_title'] = loader.read_string_padded(255)
    graph['x_title_font_size'] = loader.read_integer()
    graph['x_title_font_style'] = loader.read_byte()
    graph['x_title_font_pitch'] = loader.read_byte()
    graph['x_title_font_name'] = loader.read_string_padded(255)
    graph['y_title_font_size'] = loader.read_integer()
    graph['y_title_font_style'] = loader.read_byte()
    graph['y_title_font_pitch'] = loader.read_byte()
    graph['y_title_font_name'] = loader.read_string_padded(255)
    graph['x_number_font_size'] = loader.read_integer()
    graph['x_number_font_style'] = loader.read_byte()
    graph['x_number_font_pitch'] = loader.read_byte()
    graph['x_number_font_name'] = loader.read_string_padded(255)
    graph['y_number_font_size'] = loader.read_integer()
    graph['y_number_font_style'] = loader.read_byte()
    graph['y_number_font_pitch'] = loader.read_byte()
    graph['y_number_font_name'] = loader.read_string_padded(255)
    plot_types = [None, None]  # MaxCurves = 2
    curve_colour = [None, None]
    num_points = [None, None]
    x_data = [None, None]
    y_data = [None, None]
    for i in range(graph['num_curves']):
        plot_types[i] = loader.read_byte()
        curve_colour[i] = loader.read_integer()
        num_points[i] = loader.read_integer()
        x_data[i] = [loader.read_extended() for _ in range(num_points[i])]
        y_data[i] = [loader.read_extended() for _ in range(num_points[i])]
    graph['plot_types'] = plot_types
    graph['curve_colour'] = curve_colour
    graph['num_points'] = num_points
    graph['x_data'] = x_data
    graph['y_data'] = y_data
    if loader.version >= 3.1:
        graph['min_x_value'] = loader.read_extended()
        graph['max_x_value'] = loader.read_extended()
    return graph


def load_form(loader: Loader) -> dict:
    """Load a *TDataForm*.

    Parameters
    ----------
    loader
        The class helper.

    Returns
    -------
    dict
        The settings of the *TDataForm*.
    """
    # See the repository "Nonlinear-Fitting/NLFDataForm.pas"
    # procedure TDataForm.LoadFile(TheStream:TStream; FromDisk:Boolean);
    form = dict()
    form['win_left'] = loader.read_integer()
    form['left'] = form['win_left']
    form['win_top'] = loader.read_integer()
    form['top'] = form['win_top']
    form['win_width'] = loader.read_integer()
    form['width'] = form['win_width']
    form['win_height'] = loader.read_integer()
    form['height'] = form['win_height']
    form['the_state'] = loader.read_byte()
    form['the_size'] = loader.read_integer()

    max_count = -1
    col_widths = [0] * form['the_size']
    the_count = [0] * form['the_size']
    for i in range(form['the_size']):
        col_widths[i] = loader.read_integer()
        the_count[i] = loader.read_integer()
        if the_count[i] > max_count:
            max_count = the_count[i]
    form['max_count'] = max_count
    form['col_widths'] = col_widths
    form['the_count'] = the_count

    strings = [[None] * max_count] * form['the_size']
    cell_format = [[None] * max_count] * form['the_size']
    cell_places = [[None] * max_count] * form['the_size']
    for i in range(form['the_size']):
        for j in range(the_count[i]):
            strings[i][j] = loader.read_string()  # noqa
            cell_format[i][j] = loader.read_byte()  # noqa
            cell_places[i][j] = loader.read_byte()  # noqa
    form['strings'] = strings
    form['cell_format'] = cell_format
    form['cell_places'] = cell_places
    return form


def _load(path: str) -> dict:
    """Load a **.nlf** file.

    Parameters
    ----------
    path
        The path to a **.nlf** file.

    Returns
    -------
    dict
        The settings of the file.
    """
    # See the repository "Nonlinear-Fitting/NLFMain.pas"
    # procedure TNLFMainForm.LoadFile(FromDisk:Boolean; FileName: string);
    file = dict()
    loader = Loader(path)
    version = loader.version
    file['version'] = version
    file['equation'] = loader.read_string()
    file['mma'] = loader.read_integer()  # num parameters
    file['npt'] = loader.read_integer()  # num points
    file['weighted'] = loader.read_boolean()
    if version < 3.1:
        file['min_x_value'] = loader.read_extended()
        file['max_x_value'] = loader.read_extended()
    file['extrapolate_calc_graphs'] = loader.read_boolean()  # Options -> Calculated Curves...
    file['curve_fitted'] = loader.read_boolean()  # whether the "Calculate" button has been clicked
    if version >= 4.3:
        file['plot_uncert_curve'] = loader.read_boolean()  # Graph -> Plot Uncertainty Curve
        if loader.version >= 4.32:
            # TCalcCurveType = (ccXData:0, ccXFitData:1, ccEvenData:2)
            file['calc_curve_type'] = loader.read_byte()  # Options -> Calculated Curves...
        else:
            if loader.read_boolean():
                file['calc_curve_type'] = 0  # ccXData
            else:
                file['calc_curve_type'] = 2  # ccEvenData
        file['num_calc_points'] = loader.read_integer()  # Options -> Calculated Curves...

    # Options -> Plot Absolute or Relative Residuals
    if version < 4.2:
        file['absolute_res'] = True
    else:
        file['absolute_res'] = loader.read_boolean()

    # Enum(dxVx:0, dyVx:1, dxVy:2, dyVy:3)
    if version < 1.1:
        file['residual_type'] = 1
    else:
        file['residual_type'] = loader.read_byte()

    # Options -> Fitting Options...
    if version < 4.0:
        # Enum(LM:0, AmLS:1, AmMD:2, AmMM:3, PwLS:4, PwMD:5, PwMM:6)
        file['fitting_method'] = 0
    else:
        file['fitting_method'] = loader.read_byte()
    if version < 5.2:
        file['max_iterations'] = 999
        file['tolerance_str'] = '1E-20'
        file['tolerance'] = 1e-20
        file['delta_str'] = '0.1'
        file['delta'] = 0.1
        file['randomise_str'] = '0'
        file['randomise_fraction'] = 0
        file['num_param_sig_figs'] = 17
        file['second_derivs_H'] = True
        file['second_derivs_B'] = True
        file['unweighted_uncert'] = 0  # Enum(Data:0, Parameters:1, Both:2)
    else:
        file['max_iterations'] = loader.read_integer()
        file['num_param_sig_figs'] = loader.read_integer()
        file['tolerance_str'] = loader.read_string()
        file['tolerance'] = float(file['tolerance_str'])
        file['delta_str'] = loader.read_string()
        file['delta'] = float(file['delta_str'])
        file['randomise_str'] = loader.read_string()
        file['randomise_fraction'] = float(file['randomise_str'])/100.
        file['second_derivs_H'] = loader.read_boolean()
        file['second_derivs_B'] = loader.read_boolean()
        if version < 5.3:
            file['unweighted_uncert'] = 0  # Enum(Data:0, Parameters:1, Both:2)
        else:
            file['unweighted_uncert'] = loader.read_byte()
    if version < 1.2:
        file['uy_weights_only'] = True
    else:
        file['uy_weights_only'] = loader.read_boolean()

    a = np.zeros(file['mma'])
    constant = np.zeros(file['mma'], dtype=bool)
    for i in range(file['mma']):
        _ = loader.read_string()  # string representation of 'a'
        constant[i] = loader.read_boolean()
        a[i] = loader.read_extended()
    file['a'] = a
    file['constant'] = constant

    if version < 3.0:
        file['variable_number'] = 1
        file['plot_variable'] = 1
    else:
        file['variable_number'] = loader.read_integer()  # num variables
        file['plot_variable'] = loader.read_integer()  # Calculated Curves Options: x-variable

    x = np.zeros((file['variable_number'], file['npt']))
    ux = np.zeros((file['variable_number'], file['npt']))
    y = np.zeros(file['npt'])
    uy = np.zeros(file['npt'])
    for i in range(file['npt']):
        for j in range(file['variable_number']):
            x[j, i] = loader.read_extended()
        y[i] = loader.read_extended()
        for j in range(file['variable_number']):
            ux[j, i] = loader.read_extended()
        uy[i] = loader.read_extended()
    file['x'] = x
    file['ux'] = ux
    file['y'] = y
    file['uy'] = uy

    sig = np.ones(file['npt'])
    if version >= 3.0:
        for i in range(file['npt']):
            sig[i] = loader.read_extended()
    file['sig'] = sig

    nvar_1 = file['variable_number'] + 1
    valid_correlations = np.zeros((nvar_1, nvar_1), dtype=bool)
    is_correlated = np.zeros((nvar_1, nvar_1), dtype=bool)
    num_corr_points = np.zeros((nvar_1, nvar_1), dtype=int)
    corr_coeff = np.zeros((nvar_1, nvar_1, file['npt'], file['npt']), dtype=float)
    w_matrix = np.empty(0)
    if version < 5.0:
        file['correlated_data'] = False
        file['uncorrelated_fit'] = True
    else:
        file['correlated_data'] = loader.read_boolean()
        file['uncorrelated_fit'] = loader.read_boolean()
        if file['correlated_data']:
            for i in range(nvar_1):
                for j in range(nvar_1):
                    is_corr = loader.read_boolean()
                    is_correlated[i, j] = is_corr
                    if is_corr:
                        valid_correlations[i, j] = loader.read_boolean()
                        ncp = loader.read_integer()
                        num_corr_points[i, j] = ncp
                        for k1 in range(ncp):
                            for k2 in range(ncp):
                                corr_coeff[i, j, k1, k2] = loader.read_extended()
                    else:
                        valid_correlations[i, j] = False
                        num_corr_points[i, j] = 0
            the_size = loader.read_integer()
            if the_size > 0:
                w_matrix = np.zeros((the_size, the_size), dtype=float)
                for i in range(the_size):
                    for j in range(the_size):
                        w_matrix[i, j] = loader.read_extended()
    file['valid_correlations'] = valid_correlations
    file['is_correlated'] = is_correlated
    file['num_corr_points'] = num_corr_points
    file['corr_coeff'] = corr_coeff
    file['w_matrix'] = w_matrix

    # Comments
    if version >= 4.3:
        file['left'] = loader.read_integer()
        file['top'] = loader.read_integer()
        file['width'] = loader.read_integer()
        file['height'] = loader.read_integer()
        file['the_state'] = loader.read_byte()
    if version >= 4.2:
        file['comments'] = loader.read_string()

    file['fit_graph'] = load_graph(loader)
    file['residuals_graph'] = load_graph(loader)
    file['uncert_graph'] = load_graph(loader)
    file['results_form'] = load_form(loader)
    file['covar_form'] = load_form(loader)
    file['data_form'] = load_form(loader)

    if loader.version > 5.42:
        max_row = loader.read_integer()
        include_row = [loader.read_boolean() for _ in range(max_row)]
    else:
        include_row = [True] * file['npt']
    file['include_row'] = include_row

    return file
