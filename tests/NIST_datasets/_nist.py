"""
Information about NIST datasets.
"""
import os.path
from urllib.request import urlretrieve

import numpy as np

from msl.nlf import Model

datasets = {
    # Dataset name: (equation, number of parameters)
    'Misra1a': ('a1*(1-exp(-a2*x))', 2),
    'Chwirut2': ('exp(-a1*x)/(a2+a3*x)', 3),
    'Chwirut1': ('exp(-a1*x)/(a2+a3*x)', 3),
    'Lanczos3': ('a1*exp(-a2*x) + a3*exp(-a4*x) + a5*exp(-a6*x)', 6),
    'Gauss1': ('a1*exp( -a2*x ) + a3*exp( -(x-a4)^2 / a5^2 ) + a6*exp( -(x-a7)^2 / a8^2 )', 8),
    'Gauss2': ('a1*exp( -a2*x ) + a3*exp( -(x-a4)^2 / a5^2 ) + a6*exp( -(x-a7)^2 / a8^2 )', 8),
    'DanWood': ('a1*x^a2', 2),
    'Misra1b': ('a1 * (1-(1+a2*x/2)^(-2))', 2),
    'Kirby2': ('(a1 + a2*x + a3*x^2) / (1 + a4*x + a5*x^2)', 5),
    'Hahn1': ('(a1+a2*x+a3*x^2+a4*x^3) / (1+a5*x+a6*x^2+a7*x^3)', 7),
    'Nelson': ('a1 - a2*x1 * exp(-a3*x2)', 3),
    'MGH17': ('a1 + a2*exp(-x*a4) + a3*exp(-x*a5)', 5),
    'Lanczos1': ('a1*exp(-a2*x) + a3*exp(-a4*x) + a5*exp(-a6*x)', 6),
    'Lanczos2': ('a1*exp(-a2*x) + a3*exp(-a4*x) + a5*exp(-a6*x)', 6),
    'Gauss3': ('a1*exp( -a2*x ) + a3*exp( -(x-a4)^2 / a5^2 ) + a6*exp( -(x-a7)^2 / a8^2 )', 8),
    'Misra1c': ('a1 * (1-(1+2*a2*x)^(-.5))', 2),
    'Misra1d': ('a1*a2*x*((1+a2*x)^(-1))', 2),
    'Roszman1': ('a1 - a2*x - arctan(a3/(x-a4))/pi', 4),
    'ENSO': ('a1 + a2*cos( 2*pi*x/12 ) + a3*sin( 2*pi*x/12 ) + a5*cos( 2*pi*x/a4 ) + '
             'a6*sin( 2*pi*x/a4 ) + a8*cos( 2*pi*x/a7 ) + a9*sin( 2*pi*x/a7 )', 9),
    'MGH09': ('a1*(x^2+x*a2) / (x^2+x*a3+a4)', 4),
    'Thurber': ('(a1 + a2*x + a3*x^2 + a4*x^3) / (1 + a5*x + a6*x^2 + a7*x^3)', 7),
    'BoxBOD': ('a1*(1-exp(-a2*x))', 2),
    'Rat42': ('a1 / (1+exp(a2-a3*x))', 3),
    'MGH10': ('a1 * exp(a2/(x+a3))', 3),
    'Eckerle4': ('(a1/a2) * exp(-0.5*((x-a3)/a2)^2)', 3),
    'Rat43': ('a1 / ((1+exp(a2-a3*x))^(1/a4))', 4),
    'Bennett5': ('a1 * (a2+x)^(-1/a3)', 3),
}


class NIST:

    def __init__(self, name: str) -> None:
        """Load a NIST dataset."""
        path = os.path.join(os.path.dirname(__file__), f'{name}.dat')
        with open(path) as fp:
            lines = [line.strip() for line in fp.readlines()]

        # whether log(y) must be taken before applying the fit
        self.log_y = name == 'Nelson'

        self.equation, num_params = datasets[name]
        if name in ('ENSO', 'Roszman1'):
            # defined in Roszman1, used in ENSO and Roszman1
            pi = '3.141592653589793238462643383279'
            self.equation = self.equation.replace('pi', pi)
        assert 'b' not in self.equation
        assert '[' not in self.equation
        assert ']' not in self.equation
        assert '**' not in self.equation
        assert 'pi' not in self.equation

        # read the Start values and Certified values
        self.guess1, self.guess2, self.certified = [], [], {}
        for i, line in enumerate(lines[40:40+num_params], start=1):
            assert line.startswith(f'b{i} =')
            start1, start2, value, uncert = map(float, line[4:].split())
            self.guess1.append(start1)
            self.guess2.append(start2)
            self.certified[f'a{i}'] = {'value': value, 'uncert': uncert}

        # read the other four Certified values
        assert lines[41 + num_params].startswith('Residual Sum of Squares:')
        self.chisqr = float(lines[41 + num_params][29:])
        self.eof = float(lines[42 + num_params][29:])
        if name == 'Rat43':
            self.dof = 11  # should be 11 = 15 - 4 (not 9)
        else:
            self.dof = int(lines[43 + num_params][29:])
        self.npts = int(lines[44 + num_params][29:])

        # read the data
        row = 59
        assert lines[row].startswith('Data:')
        if name == 'Nelson':
            dtype = [('y', float), ('x1', float), ('x2', float)]
        else:
            dtype = [('y', float), ('x', float)]
        data = np.loadtxt(path, skiprows=row+1, dtype=dtype)
        self.y = data['y']
        assert len(self.y) == self.npts
        if name == 'Nelson':
            self.x = np.vstack((data['x1'], data['x2']))
            assert self.x.shape == (2, self.npts)
        else:
            self.x = data['x']
            assert len(self.x) == self.npts


def download():
    """Download all datasets."""
    url_root = 'https://www.itl.nist.gov/div898/strd/nls/data/LINKS/DATA'
    for i, dataset in enumerate(datasets, start=1):
        print(f'Downloading {dataset} [{i} of {len(datasets)}]')
        filename = f'{dataset}.dat'
        urlretrieve(f'{url_root}/{filename}', filename=filename)


if __name__ == '__main__':
    # download()

    # for d in datasets:
    #     NIST(d)

    dset = 'Bennett5'
    nist = NIST(dset)
    with Model(nist.equation) as model:
        file = os.path.join(os.path.expanduser('~'), 'Desktop', f'{dset}.nlf')
        model.save(file, x=nist.x, y=nist.y, params=nist.guess1)
