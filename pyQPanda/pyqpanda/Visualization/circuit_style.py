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
        basis_color = '#38A7C3'
        single_color = '#4FBDF1'
        rotating_single_color = '#40AEC9'
        swap_color = '#25C485'
        multi_qubit_color = '#16A48C'
        other_color = '#BB8BFF'
        pauli_color = '#3F9EEF'
        iden_color = '#05BAB6'
        non_gate_color = '#000000'

        self.name = 'original_default'
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
            'U1.dag': 'U1^\\dagger',
            'U2': 'U2',
            'U2.dag': 'U2^\\dagger',
            'U3': 'U3',
            'U3.dag': 'U3^\\dagger',
            'U4': 'U4',
            'U4.dag': 'U4^\\dagger',
            'X': 'X',
            'Y': 'Y',
            'Z': 'Z',
            'X1': 'X1',
            'X1.dag': 'X1^\\dagger',
            'Y1': 'Y1',
            'Y1.dag': 'Y1^\\dagger',
            'Z1': 'Z1',
            'Z1.dag': 'Z1^\\dagger',
            'H': 'H',
            'S': 'S',
            'S.dag': 'S^\\dagger',
            'T': 'T',
            'T.dag': 'T^\\dagger',
            'RX': 'RX',
            'RX.dag': 'RX^\\dagger',
            'RY': 'RY',
            'RY.dag': 'RY^\\dagger',
            'RZ': 'RZ',
            'RZ.dag': 'RZ^\\dagger',
            'CPHASE': 'RZ',
            'CPHASE.dag': 'RZ^\\dagger',
            'CZ': 'Z',
            'CU': 'U',
            'CU.dag': 'U^\\dagger',
            'iSWAP' : 'i',
            'iSWAP.dag' : 'i^\\dagger',
            'SqiSWAP' : 'Sqi',
            'SqiSWAP.dag' : 'Sqi^\\dagger',
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
            'X1': rotating_single_color,
            'Y1': rotating_single_color,
            'Z1': rotating_single_color,
            'H': single_color,
            'CNOT': multi_qubit_color,
            'CPHASE': multi_qubit_color,
            'CZ': multi_qubit_color,
            'SWAP': swap_color,
            'S': single_color,
            'S.dag': single_color,
            'ISWAP': swap_color,
            'ISWAPTheta': swap_color,
            'SQISWAP': swap_color,
            'T': single_color,
            'T.dag': single_color,
            'RX': rotating_single_color,
            'RY': rotating_single_color,
            'RZ': rotating_single_color,
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
