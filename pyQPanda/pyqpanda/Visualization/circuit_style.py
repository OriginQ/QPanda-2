# -*- coding: utf-8 -*-

# This code is part of PyQpanda.
#
# (C) Copyright Origin Quantum 2018-2019\n
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

from copy import copy
from warnings import warn

class DefaultStyle:
    def __init__(self):
        # Set colors
        basis_color = '#FA74A6'
        clifford_color = '#6FA4FF'
        iswap_color = '#FF33FF'
        non_gate_color = '#000000'
        other_color = '#BB8BFF'
        pauli_color = '#05BAB6'
        iden_color = '#05BAB6'

        self.name = 'iqx'
        self.tc = '#000000'
        self.sc = '#000000'
        self.lc = '#000000'
        self.not_gate_lc = '#ffffff'
        self.cc = '#778899'
        self.gc = '#ffffff'
        self.gt = '#000000'
        self.bc = '#bdbdbd'
        self.bg = '#ffffff'
        self.edge_color = None
        self.math_fs = 15
        self.fs = 13
        self.sfs = 8
        self.colored_add_width = 0.2
        self.disptex = {
            'I': 'I',
            'U1': 'U1',
            'U2': 'U2',
            'U3': 'U3',
            'U4': 'U4',
            'X': 'X',
            'Y': 'Y',
            'Z': 'Z',
            'X1': 'X1',
            'Y1': 'Y1',
            'Z1': 'Z1',
            'H': 'H',
            'S': 'S',
            'S.dag': 'S^\\dagger',
            'T': 'T',
            'T.dag': 'T^\\dagger',
            'r': 'R',
            'RX': 'RX',
            'RY': 'RY',
            'RZ': 'RZ',
            'CPHASE': 'CRz',
            'CU': 'CU',
            'RESET': '\\left|0\\right\\rangle'
        }
        self.dispcol = {
            'U1': basis_color,
            'U2': basis_color,
            'U3': basis_color,
            'U4': basis_color,
            'I': iden_color,
            'X': pauli_color,
            'Y': pauli_color,
            'Z': pauli_color,
            'X1': pauli_color,
            'Y1': pauli_color,
            'Z1': pauli_color,
            'H': clifford_color,
            'CNOT': clifford_color,
            'CPHASE': clifford_color,
            'CZ': clifford_color,
            'SWAP': clifford_color,
            'S': clifford_color,
            'S.dag': clifford_color,
            'dcx': clifford_color,
            'ISWAP': iswap_color,
            'ISWAPTheta': iswap_color,
            'SQISWAP': iswap_color,
            'T': other_color,
            'T.dag': other_color,
            'r': other_color,
            'RX': other_color,
            'RY': other_color,
            'RZ': other_color,
            'RESET': non_gate_color,
            'target': '#ffffff',
            'multi': other_color,
            'meas': non_gate_color
        }
        self.latexmode = False
        self.bundle = True
        self.index = False
        self.figwidth = -1
        self.dpi = 150
        self.margin = [2.0, 0.1, 0.1, 0.3]
        self.cline = 'doublet'

    def set_style(self, style_dic):
        dic = copy(style_dic)
        self.tc = dic.pop('textcolor', self.tc)
        self.sc = dic.pop('subtextcolor', self.sc)
        self.lc = dic.pop('linecolor', self.lc)
        self.cc = dic.pop('creglinecolor', self.cc)
        self.gt = dic.pop('gatetextcolor', self.tc)
        self.gc = dic.pop('gatefacecolor', self.gc)
        self.bc = dic.pop('barrierfacecolor', self.bc)
        self.bg = dic.pop('backgroundcolor', self.bg)
        self.fs = dic.pop('fontsize', self.fs)
        self.sfs = dic.pop('subfontsize', self.sfs)
        self.disptex = dic.pop('displaytext', self.disptex)
        self.dispcol = dic.pop('displaycolor', self.dispcol)
        self.latexmode = dic.pop('latexdrawerstyle', self.latexmode)
        self.bundle = dic.pop('cregbundle', self.bundle)
        self.index = dic.pop('showindex', self.index)
        self.figwidth = dic.pop('figwidth', self.figwidth)
        self.dpi = dic.pop('dpi', self.dpi)
        self.margin = dic.pop('margin', self.margin)
        self.cline = dic.pop('creglinestyle', self.cline)

        if dic:
            warn('style option/s ({}) is/are not supported'.format(', '.join(dic.keys())),
                 DeprecationWarning, 2)


class BWStyle:
    def __init__(self):
        self.name = 'bw'
        self.tc = '#000000'
        self.sc = '#000000'
        self.lc = '#000000'
        self.not_gate_lc = '#000000'
        self.cc = '#778899'
        self.gc = '#ffffff'
        self.gt = '#000000'
        self.bc = '#bdbdbd'
        self.bg = '#ffffff'
        self.edge_color = '#000000'
        self.fs = 13
        self.math_fs = 15
        self.colored_add_width = 0.2
        self.sfs = 8
        self.disptex = {
            'I': 'I',
            'U1': 'U_0',
            'U2': 'U_1',
            'U3': 'U_2',
            'U4': 'U_3',
            'X': 'X',
            'Y': 'Y',
            'Z': 'Z',
            'H': 'H',
            'S': 'S',
            'S.dag': 'S^\\dagger',
            'T': 'T',
            'T.dag': 'T^\\dagger',
            'r': 'R',
            'RX': 'R_x',
            'RY': 'R_y',
            'RZ': 'R_z',
            'RESET': '\\left|0\\right\\rangle'
        }
        self.dispcol = {
            'I': '#ffffff',
            'U1': '#ffffff',
            'U2': '#ffffff',
            'U3': '#ffffff',
            'U4': '#ffffff',
            'X': '#ffffff',
            'Y': '#ffffff',
            'Z': '#ffffff',
            'H': '#ffffff',
            'CNOT': '#000000',
            'S': '#ffffff',
            'S.dag': '#ffffff',
            'T': '#ffffff',
            'T.dag': '#ffffff',
            'r': '#ffffff',
            'RX': '#ffffff',
            'RY': '#ffffff',
            'RZ': '#ffffff',
            'RESET': '#ffffff',
            'target': '#ffffff',
            'meas': '#ffffff',
            'SWAP': '#000000',
            'multi': '#000000'
        }
        self.latexmode = False
        self.bundle = True
        self.index = False
        self.figwidth = -1
        self.dpi = 150
        self.margin = [2.0, 0.0, 0.0, 0.3]
        self.cline = 'doublet'

    def set_style(self, style_dic):
        dic = copy(style_dic)
        self.tc = dic.pop('textcolor', self.tc)
        self.sc = dic.pop('subtextcolor', self.sc)
        self.lc = dic.pop('linecolor', self.lc)
        self.cc = dic.pop('creglinecolor', self.cc)
        self.gt = dic.pop('gatetextcolor', self.tc)
        self.gc = dic.pop('gatefacecolor', self.gc)
        self.bc = dic.pop('barrierfacecolor', self.bc)
        self.bg = dic.pop('backgroundcolor', self.bg)
        self.fs = dic.pop('fontsize', self.fs)
        self.sfs = dic.pop('subfontsize', self.sfs)
        self.disptex = dic.pop('displaytext', self.disptex)
        for key in self.dispcol.keys():
            self.dispcol[key] = self.gc
        self.dispcol = dic.pop('displaycolor', self.dispcol)
        self.latexmode = dic.pop('latexdrawerstyle', self.latexmode)
        self.bundle = dic.pop('cregbundle', self.bundle)
        self.index = dic.pop('showindex', self.index)
        self.figwidth = dic.pop('figwidth', self.figwidth)
        self.dpi = dic.pop('dpi', self.dpi)
        self.margin = dic.pop('margin', self.margin)
        self.cline = dic.pop('creglinestyle', self.cline)

        if dic:
            warn('style option/s ({}) is/are not supported'.format(', '.join(dic.keys())),
                 DeprecationWarning, 2)
