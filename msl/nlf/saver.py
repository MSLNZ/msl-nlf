"""
Save a **.nlf** file.
"""
import math
import os
from struct import pack

import numpy as np

from .datatypes import FitMethod
from .datatypes import Input

ansi = {'encoding': 'ansi'}


class Saver:

    def __init__(self, version: str) -> None:
        """Helper class to create a **.nlf** file.

        Parameters
        ----------
        version
            The DLL version number.
        """
        self._buffer = bytearray()
        self.write_string_padded(version, 10)

    def save(self, path: str) -> None:
        """Save the buffer to a **.nlf** file.

        Parameters
        ----------
        path
            The **.nlf** file path.
        """
        with open(path, mode='wb') as fp:
            fp.write(self._buffer)

    def write_boolean(self, value: bool) -> None:
        """Write a boolean.

        Parameters
        ----------
        value
            Write `value` to the buffer.
        """
        self._buffer.extend(pack('?', value))

    def write_byte(self, value: int) -> None:
        """Write a byte.

        Parameters
        ----------
        value
            Write `value` to the buffer.
        """
        self._buffer.extend(pack('b', value))

    def write_extended(self, value: float) -> None:
        """Write a Delphi 10-byte extended float.

        Parameters
        ----------
        value
            Write `value` to the buffer.
        """
        # See Loader.read_extended for structure of an 80-bit float
        if math.isfinite(value):
            mantissa, exponent = math.frexp(abs(value))
            uint16 = exponent + 16382
            if value < 0:
                uint16 |= 0x8000
            uint64 = round(mantissa * (2 << 63))
        elif math.isnan(value):
            uint16, uint64 = 0x7FFF, 1
        else:  # +-Inf
            uint16, uint64 = 0x7FFF, 0
            if value < 0:
                uint16 |= 0x8000
        self._buffer.extend(pack('QH', uint64, uint16))

    def write_integer(self, value: int) -> None:
        """Write an unsigned integer.

        Parameters
        ----------
        value
            Write `value` to the buffer.
        """
        self._buffer.extend(pack('I', value))

    def write_string(self, value: str) -> None:
        """Write a string.

        Parameters
        ----------
        value
            Write `value` to the buffer.
        """
        length = len(value)
        self.write_integer(length)
        self._buffer.extend(pack(f'{length}s', value.encode(**ansi)))

    def write_string_padded(self, value: str, pad: int) -> None:
        """Write a string with null padding at the end.

        Parameters
        ----------
        value
            Write `value` to the buffer.
        pad
            The total number of bytes the string must be. Pads
            null bytes until the number of bytes written is
            the appropriate length.
        """
        length = len(value)
        self.write_byte(length)
        self._buffer.extend(pack(f'{length}s', value.encode(**ansi)))
        n = pad - length
        if n > 0:
            self._buffer.extend(pack(f'{n}x'))

    def write_word(self, value: int) -> None:
        """Write an unsigned short.

        Parameters
        ----------
        value
            Write `value` to the buffer.
        """
        self._buffer.extend(pack('H', value))


def save_graph(saver: Saver, name: str) -> None:
    """Save a *TGraphWindow*.

    Parameters
    ----------
    saver
        The class helper.
    name
        The name of the graph window.
    """
    if name == 'fit_graph':
        left = 1
        top = 300
        width = 380
        height = 300
    elif name == 'residuals_graph':
        left = 400
        top = 300
        width = 380
        height = 300
    elif name == 'uncert_graph':
        left = 300
        top = 50
        width = 200
        height = 100
    else:
        assert False, f'Unsupported TGraphWindow name {name}'

    num_curves = 1
    saver.write_integer(left)  # win_left=left
    saver.write_integer(top)  # win_top=top
    saver.write_integer(width)  # win_width=width
    saver.write_integer(height)  # win_height=height
    saver.write_byte(1)  # the_state
    saver.write_integer(num_curves)  # num_curves
    saver.write_boolean(True)  # x_axis_scale_to_window
    saver.write_boolean(True)  # y_axis_scale_to_window
    saver.write_string_padded('76.994', 20)  # x_axis_length
    saver.write_string_padded('51.065', 20)  # y_axis_length
    saver.write_extended(0)  # x_min
    saver.write_extended(0)  # x_max
    saver.write_extended(0)  # y_min
    saver.write_extended(0)  # y_max
    saver.write_extended(0)  # x_interval
    saver.write_extended(0)  # y_interval
    saver.write_integer(0)  # x_num_places
    saver.write_integer(0)  # y_num_places
    saver.write_integer(0)  # x_num_minor_ticks
    saver.write_integer(0)  # y_num_minor_ticks
    saver.write_integer(6)  # x_major_tick_size
    saver.write_integer(3)  # x_minor_tick_size
    saver.write_integer(6)  # y_major_tick_size
    saver.write_integer(3)  # y_minor_tick_size
    saver.write_word(1)  # x_max_auto
    saver.write_word(1)  # x_min_auto
    saver.write_word(1)  # x_inc_auto
    saver.write_word(1)  # x_ticks_auto
    saver.write_word(1)  # x_dec_auto
    saver.write_word(1)  # y_max_auto
    saver.write_word(1)  # y_min_auto
    saver.write_word(1)  # y_inc_auto
    saver.write_word(1)  # y_ticks_auto
    saver.write_word(1)  # y_dec_auto
    saver.write_word(0)  # x_major_grid
    saver.write_word(0)  # x_minor_grid
    saver.write_word(0)  # y_major_grid
    saver.write_word(0)  # y_minor_grid
    saver.write_integer(0)  # x_max_disp_places
    saver.write_integer(0)  # x_min_disp_places
    saver.write_integer(0)  # x_inc_disp_places
    saver.write_integer(0)  # y_max_disp_places
    saver.write_integer(0)  # y_min_disp_places
    saver.write_integer(0)  # y_inc_disp_places
    saver.write_boolean(True)  # empty
    saver.write_integer(351)  # axis_max_x
    saver.write_integer(20)  # axis_max_y
    saver.write_integer(60)  # axis_min_x
    saver.write_integer(213)  # axis_min_y
    saver.write_boolean(True)  # x_title_auto
    saver.write_boolean(True)  # y_title_auto
    saver.write_string_padded('x', 255)  # x_title
    saver.write_string_padded('y', 255)  # y_title
    saver.write_integer(8)  # x_title_font_size
    saver.write_byte(0)  # x_title_font_style
    saver.write_byte(0)  # x_title_font_pitch
    saver.write_string_padded('Arial', 255)  # x_title_font_name
    saver.write_integer(8)  # y_title_font_size
    saver.write_byte(0)  # y_title_font_style
    saver.write_byte(0)  # y_title_font_pitch
    saver.write_string_padded('Arial', 255)  # y_title_font_name
    saver.write_integer(8)  # x_number_font_size
    saver.write_byte(0)  # x_number_font_style
    saver.write_byte(0)  # x_number_font_pitch
    saver.write_string_padded('Arial', 255)  # x_number_font_name
    saver.write_integer(8)  # y_number_font_size
    saver.write_byte(0)  # y_number_font_style
    saver.write_byte(0)  # y_number_font_pitch
    saver.write_string_padded('Arial', 255)  # y_number_font_name
    for i in range(num_curves):
        saver.write_byte(1)  # plot_types
        saver.write_integer(16711680)  # curve_colour
        saver.write_integer(0)  # num_points
        # do not write x_data (num_points=0)
        # do not write y_data (num_points=0)
    saver.write_extended(0)  # min_x_value
    saver.write_extended(0)  # max_x_value


def save_form(saver: Saver, data: dict) -> None:
    """Save a *TDataForm*.

    Parameters
    ----------
    saver
        The class helper.
    data
        The data to add to the form. The x, y, ux, ux
        arrays should be added to the Data form. The Results
        form should have the appropriate number of columns
        and number of rows. Covariance forms is not populated.
    """
    saver.write_integer(data['left'])  # win_left=left
    saver.write_integer(data['top'])  # win_top=top
    saver.write_integer(data['width'])  # win_width=width
    saver.write_integer(data['height'])  # win_height=height
    saver.write_byte(data['the_state'])

    def write_cell(cell):
        saver.write_string(str(cell))
        saver.write_byte(0)  # cell_format
        saver.write_byte(0)  # cell_places

    # Add items to table in column-wise order
    if 'covar_form' in data:
        # this Form gets automatically populated when
        # the Calculate button is clicked on the GUI,
        # msl-nlf does not use Form information
        saver.write_integer(0)  # the_size
        return

    if 'results_form' in data:
        # must create the layout for this Form for the GUI,
        # msl-nlf does not use Form information
        header = ['Parameter', 'Value', 'Uncertainty', 't-Ratio']
        the_size = len(header)
        saver.write_integer(the_size)
        for i in range(the_size):
            col_width = 75 if i == 0 else 128
            saver.write_integer(col_width)  # col_widths
            saver.write_integer(data['nparams']+3)  # the_count

        # add Parameter column
        write_cell(header[0])
        for i in range(data['nparams']):
            write_cell(f'a{i+1}')
        write_cell('Chi Squared')
        write_cell('Error Of Fit')

        # add remaining columns as empty cells
        for h in header[1:]:
            write_cell(h)
            for i in range(data['nparams']+2):
                write_cell('')
        return

    # Create the Data Form
    nvars, npts = data['x'].shape
    header = ['']
    if nvars == 1:
        header.extend(['x', 'y', 'ux'])
    else:
        for i in range(nvars):
            header.append(f'x{i+1}')
        header.append('y')
        for i in range(nvars):
            header.append(f'ux{i+1}')
    header.extend(['uy', 'x Fit', 'y Fit', 'x Res', 'y Res', 'x Uncert', 'y Uncert'])

    the_size = len(header)
    saver.write_integer(the_size)
    for i in range(the_size):
        col_width = 36 if i == 0 else 64
        saver.write_integer(col_width)  # col_widths
        saver.write_integer(npts + 1)  # the_count, +1 for the header row

    # add row numbers
    for i in range(npts+1):
        write_cell(header[i] if i == 0 else i)

    # add x data
    for i, row in enumerate(data['x'], start=1):
        write_cell(header[i])
        for value in row:
            write_cell(value)

    # add y data
    write_cell(header[nvars+1])
    for value in data['y']:
        write_cell(value)

    # add ux data
    for i, row in enumerate(data['ux'], start=nvars+2):
        write_cell(header[i])
        for value in row:
            write_cell(value)

    # add uy data
    write_cell(header[2*nvars+2])
    for value in data['uy']:
        write_cell(value)

    # fill the rest with empty cells
    for item in header[-6:]:
        write_cell(item)
        for j in range(npts):
            write_cell('')


def save(*,
         path: str,
         comments: str,
         overwrite: bool,
         data: Input) -> None:
    """Save a **.nlf** file.

    The file can be opened in the Delphi GUI application or loaded via
    the :func:`~msl.nlf.load` function.

    Parameters
    ----------
    path
        The **.nlf** file path.
    comments
        Additional comments to add to the file. This text will appear in
        the *Comments* window in the Delphi GUI application.
    overwrite
        Whether to overwrite the file if it already exists. If the file
        exists, and this value is :data:`False`, then an error is raised.
    data
        The input data to the fit model.
    """
    if not overwrite and os.path.isfile(path):
        raise FileExistsError(f'Will not overwrite {path!r}')

    from . import version_info
    version = f'{version_info.major}.{version_info.minor}'

    # Nonlinear-Fitting/NLF DLL/NLFDLLMaths.pas
    # TFittingMethod=(LM,AmLS,AmMD,AmMM,PwLS,PwMD,PwMM);
    methods = {
        FitMethod.LM: 0,
        FitMethod.AMOEBA_LS: 1,
        FitMethod.AMOEBA_MD: 2,
        FitMethod.AMOEBA_MM: 3,
        FitMethod.POWELL_LS: 4,
        FitMethod.POWELL_MD: 5,
        FitMethod.POWELL_MM: 6,
    }

    nvars, npts = data.x.shape

    saver = Saver(version)
    saver.write_string(data.equation)
    saver.write_integer(len(data.params))
    saver.write_integer(npts)
    saver.write_boolean(data.weighted)
    saver.write_boolean(False)  # extrapolate_calc_graphs
    saver.write_boolean(False)  # curve_fitted
    saver.write_boolean(False)  # plot_uncert_curve
    saver.write_byte(2)  # TCalcCurveType (ccXData:0, ccXFitData:1, ccEvenData:2)
    saver.write_integer(201)  # num_calc_points
    saver.write_boolean(True)  # absolute_res
    saver.write_byte(1)  # residual_type (dxVx:0, dyVx:1, dxVy:2, dyVy:3)
    saver.write_byte(methods[data.fit_method])  # fitting_method
    saver.write_integer(data.max_iterations)  # max_iterations
    saver.write_integer(17)  # num_param_sig_figs
    saver.write_string(str(data.tolerance))  # tolerance_str
    saver.write_string(str(data.delta))  # delta_str
    saver.write_string('0')  # randomise_str
    saver.write_boolean(data.second_derivs_H)  # second_derivs_H
    saver.write_boolean(data.second_derivs_B)  # second_derivs_B
    saver.write_byte(0)  # unweighted_uncert
    saver.write_boolean(data.uy_weights_only)  # uy_weights_only
    for param in data.params:
        saver.write_string(str(param.value))
        saver.write_boolean(param.constant)
        saver.write_extended(param.value)
    saver.write_integer(nvars)  # variable_number
    saver.write_integer(1)  # plot_variable
    x, y, ux, uy = data.x, data.y, data.ux, data.uy
    for i in range(npts):
        for j in range(nvars):
            saver.write_extended(x[j, i])
        saver.write_extended(y[i])
        for j in range(nvars):
            saver.write_extended(ux[j, i])
        saver.write_extended(uy[i])
    for i in range(npts):
        saver.write_extended(0)  # sig

    saver.write_boolean(data.correlated)  # correlated_data
    saver.write_boolean(not data.correlated)  # uncorrelated_fit
    if data.correlated:
        for i in range(nvars+1):
            for j in range(nvars+1):
                is_corr = data.correlations.is_correlated[i, j]
                saver.write_boolean(is_corr)
                if is_corr:
                    saver.write_boolean(is_corr)  # valid_correlations

                    # find the Correlation object based on i, j values
                    n1 = 'Y' if i == 0 else f'X{i}'
                    n2 = 'Y' if j == 0 else f'X{j}'
                    coeffs = None
                    for corr in data.correlations.data:
                        if corr.path.endswith(f'{n1}-{n2}.txt'):
                            coeffs = corr.coefficients
                            break
                    if coeffs is None:
                        coeffs = np.zeros((npts, npts))

                    ncp = len(coeffs)  # num_corr_points
                    saver.write_integer(ncp)
                    for k1 in range(npts):
                        for k2 in range(npts):
                            saver.write_extended(coeffs[k1, k2])

        saver.write_integer(0)  # size of w_matrix (don't specify)

    # Comments
    saver.write_integer(750)  # left
    saver.write_integer(1)  # top
    saver.write_integer(250)  # width
    saver.write_integer(100)  # height
    saver.write_byte(0 if comments else 1)  # the_state
    prefix = '{\\rtf1\\ansi\\ansicpg1252\\deff0\\deflang5129{' \
             '\\fonttbl{\\f0\\fnil\\fcharset0 Times New Roman;}}\r\n' \
             '\\viewkind4\\uc1\\pard\\f0\\fs20 '
    suffix = '\\par\r\n}\r\n\x00'
    comments = comments.replace('\n', '\\par\r\n')
    comments = comments.replace('{', '\\{')
    comments = comments.replace('}', '\\}')
    saver.write_string(prefix + comments + suffix)

    save_graph(saver, 'fit_graph')
    save_graph(saver, 'residuals_graph')
    save_graph(saver, 'uncert_graph')

    form = dict(
        results_form=True,
        left=500,
        top=100,
        width=500,
        height=200,
        the_state=0,
        nparams=len(data.params),
    )
    save_form(saver, form)  # results_form

    form = dict(
        covar_form=True,
        left=300,
        top=400,
        width=100,
        height=100,
        the_state=1,
    )
    save_form(saver, form)  # covar_form

    form = dict(
        data_form=True,
        left=1,
        top=1,
        width=750,
        height=500,
        the_state=0,
        x=data.x,
        y=data.y,
        ux=data.ux,
        uy=data.uy
    )
    save_form(saver, form)  # data_form

    saver.save(path)
