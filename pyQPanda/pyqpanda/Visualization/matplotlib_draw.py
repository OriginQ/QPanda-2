# -*- coding: utf-8 -*-

# This code is part of PyQpanda.
#
# (C) Copyright Origin Quantum 2018-2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""circuit visualization backend."""

import collections
import fractions
import itertools
import logging
import math

import numpy as np
from .circuit_style import *
from pyqpanda.pyQPanda import GateType
from pyqpanda.pyQPanda import NodeType

try:
    from matplotlib import get_backend
    from matplotlib import patches
    from matplotlib import pyplot as plt
    plt.switch_backend('agg')
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from .pi_check import *

logger = logging.getLogger(__name__)

WID = 0.65
HIG = 0.65
DEFAULT_SCALE = 4.3
PORDER_GATE = 5
PORDER_LINE = 3
PORDER_REGLINE = 2
PORDER_GRAY = 3
PORDER_TEXT = 6
PORDER_SUBP = 4

my_qubit = 'qubit'
my_cbit = 'cbit'


class Anchor:
    def __init__(self, reg_num, yind, fold):
        self.__yind = yind
        #self.__fold = fold
        self.__fold = 0.0
        self.__reg_num = reg_num
        self.__gate_placed = []
        self.gate_anchor = 0
        self.b_fold = False
        self.last_h_pos = 0.0
        self.max_x_pos = 30.0
        self.__flod_cnt = 0

    def get_fold_info(self):
        return self.__fold, self.__flod_cnt

    def set_fold_info(self, target_fold, fold_cnt):
        self.__fold = target_fold
        self.__flod_cnt = fold_cnt

    def update_fold_info(self, index, gate_width, x_offset, layer_offset):
        tmp_x_pos = index - self.__fold + 1 + 0.5 * \
            (gate_width - 1) + x_offset + layer_offset
        if tmp_x_pos > self.max_x_pos:
            #self.b_fold = True
            self.__fold = index
            self.__flod_cnt += 1
        # else:
        #     self.b_fold = False

    def plot_coord(self, index, gate_width, x_offset, layer_offset, last_h_pos):
        test_x_pos = index - self.__fold + 1 + 0.5 * \
            (gate_width - 1) + x_offset + layer_offset
        h_pos = index - self.__fold + 1

        # if self.__fold == h_pos:
        # if last_h_pos > (h_pos + gate_width * 3):
        if test_x_pos > self.max_x_pos:
            self.b_fold = True
            self.__fold = index
            self.__flod_cnt += 1

        # if last_h_pos > h_pos:
        #     self.b_fold = True

        # check folding
        # if self.__fold > 0:
        if True:
            # test_pos = index % self.__fold + 1 + 0.5 * ((2 * gate_width) - 1) + x_offset
            # if test_pos > 29:
            # if h_pos + (gate_width - 1) > self.__fold:
            # if last_h_pos > (h_pos + gate_width * 3):
            if self.b_fold:
                # if self.b_fold:
                # index += self.__fold - (h_pos - 1)
                tmp_offset = x_offset
            else:
                tmp_offset = x_offset + layer_offset
            x_pos = index - self.__fold + 1 + 0.5 * (gate_width - 1)
            #y_pos = self.__yind - (index // self.__fold) * (self.__reg_num + 1)
            y_pos = self.__yind - (self.__flod_cnt) * (self.__reg_num + 1)
        else:
            x_pos = index + 1 + 0.5 * (gate_width - 1)
            y_pos = self.__yind

        # could have been updated, so need to store
        self.gate_anchor = index
        # self.last_h_pos = h_pos

        return x_pos + tmp_offset, y_pos

    def is_locatable(self, index, gate_width):
        hold = [index + i for i in range(gate_width)]
        for p in hold:
            if p in self.__gate_placed:
                return False
        return True

    def set_index(self, index, gate_width):
        # h_pos = index - self.__fold + 1
        # if h_pos + (gate_width - 1) > self.__fold:
        #     _index = index + self.__fold - (h_pos - 1)
        # else:
        _index = index
        #_index = index - self.__fold
        for ii in range(math.floor(gate_width)):
            if _index + ii not in self.__gate_placed:
                self.__gate_placed.append(_index + ii)
        self.__gate_placed.sort()

    def get_index(self):
        if self.__gate_placed:
            return self.__gate_placed[-1] + 1
        return 0


class MatplotlibDrawer:
    def __init__(self, qregs, cregs, ops,
                 scale=1.0, style=None, plot_barriers=True,
                 reverse_bits=False, layout=None, fold=25, ax=None):

        if not HAS_MATPLOTLIB:
            raise ImportError('The class MatplotlibDrawer needs matplotlib. '
                              'To install, run "pip install matplotlib".')

        self._ast = None
        self._scale = DEFAULT_SCALE * scale
        self._creg = []
        self._qreg = []
        self._registers(cregs, qregs)
        self._ops = ops

        self._qreg_dict = collections.OrderedDict()
        self._creg_dict = collections.OrderedDict()
        self._cond = {
            'n_lines': 0,
            'xmax': 0,
            'ymax': 0,
        }
        '''
        config = user_config.get_config()
        if config and (style is None):
            config_style = config.get('circuit_mpl_style', 'default')
            if config_style == 'default':
                self._style = DefaultStyle()
            elif config_style == 'Q1':
                self._style = BWStyle()
        elif style is False:
            self._style = BWStyle()
        else:
            self._style = DefaultStyle()
        '''
        self._style = DefaultStyle()

        self.plot_barriers = plot_barriers
        self.reverse_bits = reverse_bits
        self.layout = layout
        self.layer_offset = 0.0
        self.layer_offset_recode = []
        '''
        if style:
            if isinstance(style, dict):
                self._style.set_style(style)
            elif isinstance(style, str):
                with open(style, 'r') as infile:
                    dic = json.load(infile)
                self._style.set_style(dic)
        '''
        if ax is None:
            self.return_fig = True
            self.figure = plt.figure()
            self.figure.patch.set_facecolor(color=self._style.bg)
            self.ax = self.figure.add_subplot(111)
        '''else:
            self.return_fig = False
            self.ax = ax
            self.figure = ax.get_figure()
        '''

        self.fold = fold
        if self.fold < 2:
            self.fold = -1

        self.ax.axis('off')
        self.ax.set_aspect('equal')
        self.ax.tick_params(labelbottom=False, labeltop=False,
                            labelleft=False, labelright=False)

        self.x_offset = 0

    def _registers(self, creg, qreg):
        self._creg = []
        for r in creg:
            self._creg.append(r)
        self._qreg = []
        for r in qreg:
            self._qreg.append(r)

    @property
    def ast(self):
        return self._ast

    def _custom_multiqubit_gate(self, xy, cxy=None, fc=None, wide=True, text=None,
                                subtext=None):
        xpos = min([x[0] for x in xy])
        ypos = min([y[1] for y in xy])
        ypos_max = max([y[1] for y in xy])

        if cxy:
            ypos = min([y[1] for y in cxy])
        if wide:
            if subtext:
                boxes_length = round(max([len(text), len(subtext)]) / 7) or 1
            else:
                boxes_length = math.ceil(len(text) / 7) or 1
            wid = WID * 2.5 * boxes_length
        else:
            wid = WID

        if fc:
            _fc = fc
        else:
            if self._style.name != 'Q1':
                if self._style.gc != DefaultStyle().gc:
                    _fc = self._style.gc
                else:
                    _fc = self._style.dispcol['multi']
                _ec = self._style.dispcol['multi']
            else:
                _fc = self._style.gc

        qubit_span = abs(ypos) - abs(ypos_max) + 1
        height = HIG + (qubit_span - 1)
        box = patches.Rectangle(
            xy=(xpos - 0.5 * wid, ypos - .5 * HIG),
            width=wid, height=height,
            fc=_fc,
            ec=self._style.dispcol['multi'],
            linewidth=1.5, zorder=PORDER_GATE)
        self.ax.add_patch(box)
        # Annotate inputs
        for bit, y in enumerate([x[1] for x in xy]):
            self.ax.text(xpos - 0.45 * wid, y, str(bit), ha='left', va='center',
                         fontsize=self._style.fs, color=self._style.gt,
                         clip_on=True, zorder=PORDER_TEXT)

        if text:

            disp_text = text
            if subtext:
                self.ax.text(xpos, ypos + 0.4 * height, disp_text, ha='center',
                             va='center', fontsize=self._style.fs,
                             color=self._style.gt, clip_on=True,
                             zorder=PORDER_TEXT)
                self.ax.text(xpos, ypos + 0.2 * height, subtext, ha='center',
                             va='center', fontsize=self._style.sfs,
                             color=self._style.sc, clip_on=True,
                             zorder=PORDER_TEXT)
            else:
                self.ax.text(xpos, ypos + .5 * (qubit_span - 1), disp_text,
                             ha='center',
                             va='center',
                             fontsize=self._style.fs,
                             color=self._style.gt,
                             clip_on=True,
                             zorder=PORDER_TEXT,
                             wrap=True)

    def _gate(self, xy, fc=None, wide=False, text=None, subtext=None):
        xpos, ypos = xy

        tmp_text = text
        if (text is not None) and (len(tmp_text) > 4) and (tmp_text[-4:] == '.dag'):
            tmp_text = tmp_text[:-4]
        if wide:
            if subtext:
                subtext_len = len(subtext)
                if '$\\pi$' in subtext:
                    pi_count = subtext.count('pi')
                    subtext_len = subtext_len - (4 * pi_count)

                boxes_wide = round(max(subtext_len, len(text)) / 10, 1) or 1
                wid = WID * 1.5 * boxes_wide
            else:
                boxes_wide = round(len(text) / 10) or 1
                wid = WID * 2.2 * boxes_wide
            if wid < WID:
                wid = WID
        else:
            wid = WID
        if fc:
            _fc = fc
        elif self._style.gc != DefaultStyle().gc:
            _fc = self._style.gc
        elif tmp_text and tmp_text in self._style.dispcol:
            _fc = self._style.dispcol[tmp_text]
        else:
            _fc = self._style.gc

        box = patches.Rectangle(
            xy=(xpos - 0.5 * wid, ypos - 0.5 * HIG), width=wid, height=HIG,
            fc=_fc, ec=self._style.edge_color, linewidth=1.5, zorder=PORDER_GATE)
        self.ax.add_patch(box)

        if text:
            font_size = self._style.fs
            sub_font_size = self._style.sfs
            # check if gate is not unitary
            if text in ['RESET']:
                disp_color = self._style.not_gate_lc
                sub_color = self._style.not_gate_lc
                font_size = self._style.math_fs

            else:
                disp_color = self._style.gt
                sub_color = self._style.sc

            if text in self._style.disptex:
                disp_text = "${}$".format(self._style.disptex[text])
            else:
                disp_text = text
            if subtext:
                self.ax.text(xpos, ypos + 0.15 * HIG, disp_text, ha='center',
                             va='center', fontsize=font_size,
                             color=disp_color, clip_on=True,
                             zorder=PORDER_TEXT)
                self.ax.text(xpos, ypos - 0.3 * HIG, subtext, ha='center',
                             va='center', fontsize=sub_font_size,
                             color=sub_color, clip_on=True,
                             zorder=PORDER_TEXT)
            else:
                self.ax.text(xpos, ypos, disp_text, ha='center', va='center',
                             fontsize=font_size,
                             color=disp_color,
                             clip_on=True,
                             zorder=PORDER_TEXT)

    def _subtext(self, xy, text):
        xpos, ypos = xy

        self.ax.text(xpos, ypos - 0.3 * HIG, text, ha='center', va='top',
                     fontsize=self._style.sfs,
                     color=self._style.tc,
                     clip_on=True,
                     zorder=PORDER_TEXT)

    def _sidetext(self, xy, text):
        xpos, ypos = xy

        # 0.15 = the initial gap, each char means it needs to move
        # another 0.0375 over
        xp = xpos + 0.15 + (0.0375 * len(text))
        self.ax.text(xp, ypos + HIG, text, ha='center', va='top',
                     fontsize=self._style.sfs,
                     color=self._style.tc,
                     clip_on=True,
                     zorder=PORDER_TEXT)

    def _line(self, xy0, xy1, lc=None, ls=None, zorder=PORDER_LINE):
        x0, y0 = xy0
        x1, y1 = xy1
        if lc is None:
            linecolor = self._style.lc
        else:
            linecolor = lc
        if ls is None:
            linestyle = 'solid'
        else:
            linestyle = ls

        if linestyle == 'doublet':
            theta = np.arctan2(np.abs(x1 - x0), np.abs(y1 - y0))
            dx = 0.05 * WID * np.cos(theta)
            dy = 0.05 * WID * np.sin(theta)
            self.ax.plot([x0 + dx, x1 + dx], [y0 + dy, y1 + dy],
                         color=linecolor,
                         linewidth=2,
                         linestyle='solid',
                         zorder=zorder)
            self.ax.plot([x0 - dx, x1 - dx], [y0 - dy, y1 - dy],
                         color=linecolor,
                         linewidth=2,
                         linestyle='solid',
                         zorder=zorder)
        else:
            self.ax.plot([x0, x1], [y0, y1],
                         color=linecolor,
                         linewidth=2,
                         linestyle=linestyle,
                         zorder=zorder)

    def _reset(self, qxy):
        qx, qy = qxy
        self._gate(qxy, fc=self._style.dispcol['RESET'])

        # add measure symbol
        arc = patches.Arc(xy=(qx, qy - 0.15 * HIG), width=WID * 0.7,
                          height=HIG * 0.7, theta1=0, theta2=180, fill=False,
                          ec=self._style.not_gate_lc, linewidth=2,
                          zorder=PORDER_GATE)
        self.ax.add_patch(arc)
        self.ax.plot([qx, qx + 0.35 * WID],
                     [qy - 0.15 * HIG, qy + 0.20 * HIG],
                     color=self._style.not_gate_lc, linewidth=2, zorder=PORDER_GATE)

    def _measure(self, qxy, cxy, cid):
        qx, qy = qxy
        cx, cy = cxy

        self._gate(qxy, fc=self._style.dispcol['meas'])

        # add measure symbol
        arc = patches.Arc(xy=(qx, qy - 0.15 * HIG), width=WID * 0.7,
                          height=HIG * 0.7, theta1=0, theta2=180, fill=False,
                          ec=self._style.not_gate_lc, linewidth=2,
                          zorder=PORDER_GATE)
        self.ax.add_patch(arc)
        self.ax.plot([qx, qx + 0.35 * WID],
                     [qy - 0.15 * HIG, qy + 0.20 * HIG],
                     color=self._style.not_gate_lc, linewidth=2, zorder=PORDER_GATE)
        # arrow
        self._line(qxy, [cx, cy + 0.35 * WID], lc=self._style.cc,
                   ls=self._style.cline)
        arrowhead = patches.Polygon(((cx - 0.20 * WID, cy + 0.35 * WID),
                                     (cx + 0.20 * WID, cy + 0.35 * WID),
                                     (cx, cy)),
                                    fc=self._style.cc,
                                    ec=None)
        self.ax.add_artist(arrowhead)
        # target
        if self._style.bundle:
            self.ax.text(cx + .15, cy + .1, str(cid), ha='left', va='bottom',
                         fontsize=0.8 * self._style.fs,
                         color=self._style.tc,
                         clip_on=True,
                         zorder=PORDER_TEXT)

    def _conds(self, xy, istrue=False):
        xpos, ypos = xy

        if istrue:
            _fc = self._style.lc
        else:
            _fc = self._style.gc

        box = patches.Circle(xy=(xpos, ypos), radius=WID * 0.15,
                             fc=_fc, ec=self._style.lc,
                             linewidth=1.5, zorder=PORDER_GATE)
        self.ax.add_patch(box)

    def _ctrl_qubit(self, xy, fc=None, ec=None):
        if self._style.gc != DefaultStyle().gc:
            fc = self._style.gc
            ec = self._style.gc
        if fc is None:
            fc = self._style.lc
        if ec is None:
            ec = self._style.lc
        xpos, ypos = xy
        box = patches.Circle(xy=(xpos, ypos), radius=WID * 0.15,
                             fc=fc, ec=ec,
                             linewidth=1.5, zorder=PORDER_GATE)
        self.ax.add_patch(box)

    def get_op_color(self, op_name):
        tmp_op_name = op_name
        if (tmp_op_name is not None) and (len(tmp_op_name) > 4) and (tmp_op_name[-4:] == '.dag'):
            tmp_op_name = tmp_op_name[:-4]

        if tmp_op_name in ['U1', 'U2', 'U3', 'U4', 'CU']:
            color_str = 'U4'
        elif tmp_op_name in ['X', 'Y', 'Z']:
            color_str = 'X'
        elif tmp_op_name in ['I']:
            color_str = 'I'
        elif tmp_op_name in ['X1', 'Y1', 'Z1', 'RX', 'RY', 'RZ']:
            color_str = 'X1'
        elif tmp_op_name in ['CNOT', 'CPHASE', 'CZ']:
            color_str = 'CNOT'
        elif tmp_op_name in ['H', 'S', 'T']:
            color_str = 'H'
        elif tmp_op_name in ['SWAP', 'ISWAP', 'ISWAPTheta', 'SQISWAP']:
            color_str = 'SWAP'
        else:
            color_str = 'multi'

        return color_str

    def set_multi_ctrl_bits(self, ctrl_state, num_ctrl_qubits, qbit, color_str):
        # convert op.ctrl_state to bit string and reverse
        cstate = "{0:b}".format(ctrl_state).rjust(num_ctrl_qubits, '0')[::-1]
        for i in range(num_ctrl_qubits):
            # Make facecolor of ctrl bit the box color if closed and bkgrnd if open
            fc_open_close = (self._style.dispcol[color_str] if cstate[i] == '1'
                             else self._style.bg)
            self._ctrl_qubit(qbit[i], fc=fc_open_close,
                             ec=self._style.dispcol[color_str])

    def _tgt_qubit(self, xy, fc=None, ec=None, ac=None,
                   add_width=None):
        if self._style.gc != DefaultStyle().gc:
            fc = self._style.gc
            ec = self._style.gc
        if fc is None:
            fc = self._style.dispcol['target']
        if ec is None:
            ec = self._style.lc
        if ac is None:
            ac = self._style.lc
        if add_width is None:
            add_width = 0.35

        linewidth = 2

        if self._style.dispcol['target'] == '#ffffff':
            add_width = self._style.colored_add_width

        xpos, ypos = xy

        box = patches.Circle(xy=(xpos, ypos), radius=HIG * 0.35,
                             fc=fc, ec=ec, linewidth=linewidth,
                             zorder=PORDER_GATE)
        self.ax.add_patch(box)
        # add '+' symbol
        self.ax.plot([xpos, xpos], [ypos - add_width * HIG,
                                    ypos + add_width * HIG],
                     color=ac, linewidth=linewidth, zorder=PORDER_GATE + 1)

        self.ax.plot([xpos - add_width * HIG, xpos + add_width * HIG],
                     [ypos, ypos], color=ac, linewidth=linewidth,
                     zorder=PORDER_GATE + 1)

    def _swap(self, xy, color, fc=None, ec=None, ac=None, add_width=None):
        xpos, ypos = xy

        self.ax.plot([xpos - 0.20 * WID, xpos + 0.20 * WID],
                     [ypos - 0.20 * WID, ypos + 0.20 * WID],
                     color=color, linewidth=2, zorder=PORDER_LINE + 1)
        self.ax.plot([xpos - 0.20 * WID, xpos + 0.20 * WID],
                     [ypos + 0.20 * WID, ypos - 0.20 * WID],
                     color=color, linewidth=2, zorder=PORDER_LINE + 1)

    def _swap_gate(self, q_xy, gate_type, param, dagger, ctrl_qubits=0):
        font_size = self._style.sfs + 1
        if gate_type == GateType.ISWAP_THETA_GATE:
            swap_sub_text = '{}'.format(param)
            #font_size = self._style.sfs
        elif gate_type == GateType.ISWAP_GATE:
            swap_sub_text = 'iSWAP'
        elif gate_type == GateType.SQISWAP_GATE:
            swap_sub_text = 'SqiSWAP'

        if dagger:
            if gate_type == GateType.ISWAP_THETA_GATE:
                if swap_sub_text[0] == '-':
                    swap_sub_text = swap_sub_text[1:]
                else:
                    swap_sub_text = '-' + swap_sub_text
            else:
                swap_sub_text = swap_sub_text + '.dag'

        if gate_type == GateType.ISWAP_THETA_GATE:
            swap_sub_text = "{}".format(swap_sub_text)
        else:
            swap_sub_text = "${}$".format(self._style.disptex[swap_sub_text])

        self._iswap(q_xy[ctrl_qubits], self._style.dispcol['ISWAP'],
                    fc=self._style.dispcol['ISWAP'], ec=self._style.dispcol['ISWAP'],
                    subtext=swap_sub_text, font_size=font_size)
        self._iswap(q_xy[ctrl_qubits + 1], self._style.dispcol['ISWAP'],
                    fc=self._style.dispcol['ISWAP'], ec=self._style.dispcol['ISWAP'],
                    subtext=swap_sub_text, font_size=font_size)

    def _iswap(self, xy, color, subtext, fc=None, ec=None, ac=None, add_width=None, font_size=None):
        xpos, ypos = xy

        if self._style.gc != DefaultStyle().gc:
            fc = self._style.gc
            ec = self._style.gc
        if fc is None:
            fc = self._style.dispcol['target']
        if ec is None:
            ec = self._style.lc
        if ac is None:
            ac = self._style.lc
        if add_width is None:
            add_width = 0.35

        linewidth = 2

        if self._style.dispcol['target'] == '#ffffff':
            add_width = self._style.colored_add_width
        box = patches.Circle(xy=(xpos, ypos), radius=HIG * 0.5,
                             fc=fc, ec=ec, linewidth=linewidth,
                             zorder=PORDER_GATE)
        self.ax.add_patch(box)

        arc_up = patches.Arc(xy=(xpos, ypos), width=WID * 0.8,
                             height=HIG * 0.8, theta1=30, theta2=150, fill=False,
                             ec=self._style.not_gate_lc, linewidth=1.5,
                             zorder=PORDER_GATE)
        self.ax.add_patch(arc_up)
        arc_down = patches.Arc(xy=(xpos, ypos), width=WID * 0.8,
                               height=HIG * 0.8, theta1=210, theta2=330, fill=False,
                               ec=self._style.not_gate_lc, linewidth=1.5,
                               zorder=PORDER_GATE)
        self.ax.add_patch(arc_down)
        self.ax.plot([xpos - 0.22, xpos - 0.20],
                     [ypos + 0.15, ypos + 0.22],
                     color=self._style.not_gate_lc, linewidth=1, zorder=PORDER_GATE)
        self.ax.plot([xpos + 0.20, xpos + 0.22],
                     [ypos - 0.22, ypos - 0.15],
                     color=self._style.not_gate_lc, linewidth=1, zorder=PORDER_GATE)

        # angle
        if font_size == None:
            font_size = self._style.fs
        sub_color = self._style.sc
        self.ax.text(xpos, ypos-0.03, subtext, ha='center',
                     va='center', fontsize=font_size,
                     color=sub_color, clip_on=True,
                     zorder=PORDER_TEXT)

    def _barrier(self, config, anc):
        xys = config['coord']
        group = config['group']
        for xy in xys:
            xpos, ypos = xy
            self.ax.plot(
                [xpos, xpos],
                [ypos + 0.5, ypos - 0.5],
                linewidth=1,
                linestyle="dashed",
                color=self._style.lc,
                zorder=PORDER_TEXT,
            )
            box = patches.Rectangle(
                xy=(xpos - (0.3 * WID), ypos - 0.5),
                width=0.6 * WID,
                height=1,
                fc=self._style.bc,
                ec=None,
                alpha=0.6,
                linewidth=1.5,
                zorder=PORDER_GRAY,
            )
            self.ax.add_patch(box)

    def _linefeed_mark(self, xy):
        xpos, ypos = xy

        self.ax.plot([xpos - .1, xpos - .1],
                     [ypos, ypos - self._cond['n_lines'] + 1],
                     color=self._style.lc, zorder=PORDER_LINE)
        self.ax.plot([xpos + .1, xpos + .1],
                     [ypos, ypos - self._cond['n_lines'] + 1],
                     color=self._style.lc, zorder=PORDER_LINE)

    def draw(self, filename=None, verbose=False):
        self._draw_regs()
        self._draw_ops(verbose)

        _xl = - self._style.margin[0]
        _xr = self._cond['xmax'] + self._style.margin[1]
        _yb = - self._cond['ymax'] - self._style.margin[2] + 1 - 0.5
        _yt = self._style.margin[3] + 0.5
        self.ax.set_xlim(_xl, _xr)
        self.ax.set_ylim(_yb, _yt)

        # update figure size
        fig_w = _xr - _xl
        fig_h = _yt - _yb
        if self._style.figwidth < 0.0:
            self._style.figwidth = fig_w * self._scale * self._style.fs / 72 / WID
        self.figure.set_size_inches(
            self._style.figwidth, self._style.figwidth * fig_h / fig_w)

        if filename:
            self.figure.savefig(filename, dpi=self._style.dpi,
                                bbox_inches='tight')
            plt.close(self.figure)

        if self.return_fig:
            if get_backend() in ['module://ipykernel.pylab.backend_inline',
                                 'nbAgg']:
                plt.close(self.figure)
            return self.figure

    def _draw_regs(self):

        len_longest_label = 0
        # quantum register
        for ii, reg in enumerate(self._qreg):
            if len(self._qreg) > 1:
                if self.layout is None:
                    label = '${{{name}}}_{{{index}}}$'.format(name='q',
                                                              index=reg)
                else:
                    label = '${{{name}}}_{{{index}}} \\mapsto {{{physical}}}$'.format(
                        name=self.layout[reg.index].register.name,
                        index=self.layout[reg.index].index,
                        physical=reg.index)
            else:
                label = '${{{name}}}_{{{index}}}$'.format(name='q', index=reg)

            if len(label) > len_longest_label:
                len_longest_label = len(label)

            pos = -ii
            self._qreg_dict[reg] = {
                'y': pos,
                'label': label,
                'index': reg,
                'group': my_qubit
                # 'group': reg.register
            }
            self._cond['n_lines'] += 1
        # classical register
        if self._creg:
            n_creg = self._creg.copy()
            n_creg.pop(0)
            idx = 0
            y_off = -len(self._qreg)
            for ii, (reg, nreg) in enumerate(itertools.zip_longest(
                    self._creg, n_creg)):
                pos = y_off - idx
                if self._style.bundle:
                    label = '${}$'.format('c')
                    self._creg_dict[ii] = {
                        'y': pos,
                        'label': label,
                        'index': reg,
                        'group': my_cbit
                        # 'group': reg.register
                    }
                    if (reg != nreg and nreg):
                        continue
                else:
                    label = '${}_{{{}}}$'.format('c', reg)
                    self._creg_dict[ii] = {
                        'y': pos,
                        'label': label,
                        'index': reg,
                        'group': my_cbit
                        # 'group': reg.register
                    }
                if len(label) > len_longest_label:
                    len_longest_label = len(label)

                self._cond['n_lines'] += 1
                idx += 1

        # 7 is the length of the smallest possible label
        self.x_offset = -.5 + 0.18 * (len_longest_label - 7)

    def _draw_regs_sub(self, n_fold, feedline_l=False, feedline_r=False):
        if n_fold < len(self.layer_offset_recode):
            #self._cond['xmax'] = self.fold + self.x_offset + 1 - 0.1 + self.layer_offset_recode[n_fold] + 1.5
            self._cond['xmax'] = 30.5

        # quantum register
        for qreg in self._qreg_dict.values():
            if n_fold == 0:
                label = qreg['label']
            else:
                label = qreg['label']
            y = qreg['y'] - n_fold * (self._cond['n_lines'] + 1)
            self.ax.text(self.x_offset - 0.2, y, label, ha='right', va='center',
                         fontsize=1.25 * self._style.fs,
                         color=self._style.tc,
                         clip_on=True,
                         zorder=PORDER_TEXT)
            self._line([self.x_offset + 0.2, y], [self._cond['xmax'], y],
                       zorder=PORDER_REGLINE)
            self._line([self.x_offset + 1.6, y], [self._cond['xmax'] + 1, y],
                       zorder=PORDER_REGLINE)
        # classical register
        this_creg_dict = {}
        for creg in self._creg_dict.values():
            if n_fold == 0:
                label = creg['label']
            else:
                label = creg['label']
            y = creg['y'] - n_fold * (self._cond['n_lines'] + 1)
            if y not in this_creg_dict.keys():
                this_creg_dict[y] = {'val': 1, 'label': label}
            else:
                this_creg_dict[y]['val'] += 1
        for y, this_creg in this_creg_dict.items():
            # bundle
            if this_creg['val'] > 1:
                self.ax.plot([self.x_offset + 0.4, self.x_offset + 0.5], [y - .1, y + .1],
                             color=self._style.cc,
                             zorder=PORDER_LINE)
                '''
                self.ax.text(self.x_offset + 1.0, y + .1, str(this_creg['val']), ha='left',
                             va='bottom',
                             fontsize=0.8 * self._style.fs,
                             color=self._style.tc,
                             clip_on=True,
                             zorder=PORDER_TEXT)
                '''
                self.ax.text(self.x_offset + 0.4, y + .1, str(this_creg['val']), ha='left',
                             va='bottom',
                             fontsize=0.8 * self._style.fs,
                             color=self._style.tc,
                             clip_on=True,
                             zorder=PORDER_TEXT)
            self.ax.text(self.x_offset - 0.2, y, this_creg['label'], ha='right', va='center',
                         fontsize=1.5 * self._style.fs,
                         color=self._style.tc,
                         clip_on=True,
                         zorder=PORDER_TEXT)
            self._line([self.x_offset + 0.2, y], [self._cond['xmax'], y], lc=self._style.cc,
                       ls=self._style.cline, zorder=PORDER_REGLINE)
            self._line([self.x_offset + 1.5, y], [self._cond['xmax'] + 2, y], lc=self._style.cc,
                       ls=self._style.cline, zorder=PORDER_REGLINE)

        # lf line
        if feedline_r:
            # self._linefeed_mark((self.fold + self.x_offset + 1 - 0.1 + self.layer_offset_recode[n_fold],
            #                      - n_fold * (self._cond['n_lines'] + 1)))
            self._linefeed_mark((self._cond['xmax'],
                                 - n_fold * (self._cond['n_lines'] + 1)))
        if feedline_l:
            self._linefeed_mark((self.x_offset + 0.3,
                                 - n_fold * (self._cond['n_lines'] + 1)))

    def _rzz(self, qxy, param, qreg_b, qreg_t):
        color = self._style.dispcol['multi']
        self._ctrl_qubit(qxy[0], fc=color, ec=color)
        self._ctrl_qubit(qxy[1], fc=color, ec=color)
        self._sidetext(qreg_b, text='zz({})'.format(param))
        # add qubit-qubit wiring
        self._line(qreg_b, qreg_t, lc=color)

    def _cu1(self, qxy, param, qreg_b, qreg_t):
        color = self._style.dispcol['multi']
        self._ctrl_qubit(qxy[0], fc=color, ec=color)
        self._ctrl_qubit(qxy[1], fc=color, ec=color)
        self._sidetext(qreg_b, text='U1 ({})'.format(param))

        # add qubit-qubit wiring
        self._line(qreg_b, qreg_t, lc=color)

    def _cnot(self, q_xy, qreg_b, qreg_t, num_ctrl_qubits=1):
        if self._style.dispcol['CNOT'] != '#ffffff':
            add_width = self._style.colored_add_width
        else:
            add_width = None

        # if 0 == num_ctrl_qubits:
        #     self.set_multi_ctrl_bits(1, num_ctrl_qubits, q_xy, 'CNOT')
        # else:
        #     self.set_multi_ctrl_bits((2**num_ctrl_qubits) - 1, num_ctrl_qubits, q_xy, 'CNOT')
        self.set_multi_ctrl_bits(
            (2**num_ctrl_qubits) - 1, num_ctrl_qubits, q_xy, 'CNOT')

        if self._style.name != 'Q1':
            self._tgt_qubit(q_xy[num_ctrl_qubits], fc=self._style.dispcol['CNOT'],
                            ec=self._style.dispcol['CNOT'],
                            ac=self._style.dispcol['target'],
                            add_width=add_width)
        else:
            self._tgt_qubit(q_xy[num_ctrl_qubits], fc=self._style.dispcol['target'],
                            ec=self._style.dispcol['CNOT'],
                            ac=self._style.dispcol['CNOT'],
                            add_width=add_width)
        # add qubit-qubit wiring
        self._line(qreg_b, qreg_t, lc=self._style.dispcol['CNOT'])

    def _cz(self, q_xy, qreg_b, qreg_t):
        disp = 'CZ'
        if self._style.name != 'Q1':
            color = self._style.dispcol['CZ']
            self._ctrl_qubit(q_xy[0], fc=color, ec=color)
            self._gate(q_xy[1], wide=False, text=disp, fc=color)
        else:
            self._ctrl_qubit(q_xy[0])
            self._gate(q_xy[1], wide=False, text=disp, fc=color)

        # add qubit-qubit wiring
        if self._style.name != 'Q1':
            self._line(qreg_b, qreg_t, lc=color)
        else:
            self._line(qreg_b, qreg_t, zorder=PORDER_LINE + 1)

    def _draw_ops(self, verbose=False):
        _wide_gate = [GateType.RX_GATE, GateType.RY_GATE, GateType.RZ_GATE, GateType.U1_GATE, GateType.U2_GATE,
                      GateType.U3_GATE, GateType.U4_GATE, GateType.CU_GATE, GateType.CPHASE_GATE, GateType.ISWAP_THETA_GATE,
                      GateType.U3_GATE, GateType.U4_GATE, GateType.CU_GATE, GateType.CPHASE_GATE, GateType.ISWAP_THETA_GATE,
                      GateType.RXX_GATE, GateType.RYY_GATE, GateType.RZZ_GATE, GateType.RZX_GATE]
        _barriers = {'coord': [], 'group': []}

        #
        # generate coordinate manager
        #
        q_anchors = {}
        for key, qreg in self._qreg_dict.items():
            key = qreg['index']
            q_anchors[key] = Anchor(reg_num=self._cond['n_lines'],
                                    yind=qreg['y'],
                                    fold=self.fold)
        c_anchors = {}
        for key, creg in self._creg_dict.items():
            c_anchors[key] = Anchor(reg_num=self._cond['n_lines'],
                                    yind=creg['y'],
                                    fold=self.fold)
        #
        # draw gates
        #
        n_fold = 0
        my_offset = 0.0
        prev_anc = -1
        for layer in self._ops:
            layer_width = 1
            my_offset_tmp = 0.0
            occupied_qubits = []

            for op in layer:
                # If one of the standard wide gates
                if op.m_gate_type in _wide_gate:
                    if layer_width < 2:
                        layer_width = 2
                    if int(op.m_gate_type) >= 0:
                        param = self.param_parse(op.m_params)
                        if '$\\pi$' in param:
                            pi_count = param.count('pi')
                            len_param = len(param) - (4 * pi_count)
                        else:
                            len_param = len(param)
                        if len_param > len(op.m_name):
                            box_width = math.floor(len(param) / 10) + 0.25
                            #box_width = math.ceil(len(param) / 10)
                            if op.m_name == 'unitary':
                                box_width = 2
                            # If more than 4 characters min width is 2
                            if box_width <= 1:
                                box_width = 2
                            if layer_width < box_width:
                                if box_width > 2:
                                    layer_width = box_width
                                else:
                                    layer_width = 2
                            continue

                # If custom ControlledGate
                elif op.m_gate_type in [GateType.RPHI_GATE, GateType.CPHASE_GATE]:
                    # if op.type == 'op' and hasattr(op.op, 'params'):
                    if int(op.m_gate_type) >= 0 and len(op.m_params) > 0:
                        param = self.param_parse(op.m_params)
                        if '$\\pi$' in param:
                            pi_count = param.count('pi')
                            len_param = len(param) - (4 * pi_count)
                        else:
                            len_param = len(param)
                        if len_param > len(op.m_name):
                            box_width = math.floor(len_param / 5.5)
                            layer_width = box_width
                            continue

                # if custom gate with a longer than standard name determine
                # width
                '''
                elif op.m_gate_type not in [34, 26, -1, -2] and len(op.m_name) >= 4:
                    box_width = math.ceil(len(op.m_name) / 6)

                    # handle params/subtext longer than op names
                    if op.type == 'op' and hasattr(op.op, 'params'):
                        param = self.param_parse(op.op.params)
                        if '$\\pi$' in param:
                            pi_count = param.count('pi')
                            len_param = len(param) - (4 * pi_count)
                        else:
                            len_param = len(param)
                        if len_param > len(op.m_name):
                            box_width = math.floor(len(param) / 8)
                            # If more than 4 characters min width is 2
                            if box_width <= 1:
                                box_width = 2
                            if layer_width < box_width:
                                if box_width > 2:
                                    layer_width = box_width * 2
                                else:
                                    layer_width = 2
                            continue
                    # If more than 4 characters min width is 2
                    layer_width = math.ceil(box_width * WID * 2.5)
                '''
            this_anc = prev_anc + 1

            cur_layer_max_x_pos = 0.0
            for op in layer:
                _iswide = op.m_gate_type in _wide_gate
                # if op.m_gate_type not in [34, 26, -1, -2]:
                #    _iswide = True

                # get qreg index
                q_idxs = []
                for qarg in (op.m_control_qubits.to_list()):
                    for index, reg in self._qreg_dict.items():
                        qubit_index = qarg.get_phy_addr()
                        if (reg['index'] == qubit_index):
                            q_idxs.append(qubit_index)
                            break

                for qarg in op.m_target_qubits.to_list():
                    for index, reg in self._qreg_dict.items():
                        qubit_index = qarg.get_phy_addr()
                        if (reg['index'] == qubit_index):
                            q_idxs.append(qubit_index)
                            break

                # get creg index
                c_idxs = []
                for carg in op.m_cbits:
                    for index, reg in self._creg_dict.items():
                        if (reg['index'] == carg):
                            c_idxs.append(index)
                            break

                # Only add the gate to the anchors if it is going to be plotted.
                # This prevents additional blank wires at the end of the line if
                # the last instruction is a barrier type
                if self.plot_barriers or \
                        op.m_gate_type not in [GateType.BARRIER_GATE]:
                    for ii in q_idxs:
                        q_anchors[ii].set_index(this_anc, layer_width)

                # append new occupied-qubit
                list = q_idxs
                lmin = min(list)
                lmax = max(list) + 1
                if op.m_node_type == NodeType.MEASURE_GATE:
                    lmax = len(q_anchors)
                if op.m_gate_type in [GateType.ORACLE_GATE, GateType.TWO_QUBIT_GATE]:
                    layer_width = 2

                for hh in range(lmin, lmax):
                    if hh in occupied_qubits:
                        my_offset_tmp += 0.9*layer_width
                        my_offset += my_offset_tmp
                        break

                for q in range(lmin, lmax):
                    occupied_qubits += [q]

                # if op.m_node_type == NodeType.MEASURE_GATE:
                #     lmax = len(q_anchors)
                #     for q in range(lmin,lmax):
                #         occupied_qubits += [q]

                # for ii in q_idxs:
                #     if q_anchors[ii].b_fold:
                #         my_offset_tmp = 0.0
                #         self.layer_offset = 0.0
                #         q_anchors[ii].b_fold = False
                #         break

                # qreg coordinate
                q_xy = [q_anchors[ii].plot_coord(this_anc, layer_width, self.x_offset + my_offset_tmp, self.layer_offset, q_anchors[ii].last_h_pos)
                        for ii in q_idxs]
                if len(q_xy) > 0:
                    x0, y0 = q_xy[0]
                    if x0 > cur_layer_max_x_pos:
                        cur_layer_max_x_pos = x0
                    # for ii in range(0, len(q_anchors)):
                    #     q_anchors[ii].last_h_pos = x0

                # for index, reg in self._creg_dict.items():
                #     c_anchors[index].update_fold_info(this_anc, layer_width, self.x_offset + my_offset_tmp, self.layer_offset)

                # creg coordinate
                c_xy = [c_anchors[ii].plot_coord(this_anc, layer_width, self.x_offset + my_offset_tmp, self.layer_offset, c_anchors[ii].last_h_pos)
                        for ii in c_idxs]

                # if len(c_xy) > 0:
                #     x0, y0 = c_xy[0]
                #     for ii in range(0, len(c_anchors)):
                #         c_anchors[ii].last_h_pos = x0

                # tmp_b_first_time = True
                # for ii in q_anchors:
                #     if q_anchors[ii].b_fold:
                #         if tmp_b_first_time:
                #             tmp_b_first_time = False
                #             self.layer_offset_recode.append(self.layer_offset)
                #             self.layer_offset = 0.0

                #         q_anchors[ii].b_fold = False

                # bottom and top point of qreg
                qreg_b = min(q_xy, key=lambda xy: xy[1])
                qreg_t = max(q_xy, key=lambda xy: xy[1])

                # update index based on the value from plotting
                this_anc = q_anchors[q_idxs[0]].gate_anchor

                # if verbose:
                #     print(op)

                if int(op.m_gate_type) >= 0 and len(op.m_params) > 0:
                    param = self.param_parse(op.m_params)
                else:
                    param = None
                # conditional gate
                '''
                if op.condition:
                    c_xy = [c_anchors[ii].plot_coord(this_anc, layer_width, self.x_offset) for
                            ii in self._creg_dict]
                    mask = 0
                    for index, cbit in enumerate(self._creg):
                        if cbit.register == op.condition[0]:
                            mask |= (1 << index)
                    val = op.condition[1]
                    # cbit list to consider
                    fmt_c = '{{:0{}b}}'.format(len(c_xy))
                    cmask = list(fmt_c.format(mask))[::-1]
                    # value
                    fmt_v = '{{:0{}b}}'.format(cmask.count('1'))
                    vlist = list(fmt_v.format(val))[::-1]
                    # plot conditionals
                    v_ind = 0
                    xy_plot = []
                    for xy, m in zip(c_xy, cmask):
                        if m == '1':
                            if xy not in xy_plot:
                                if vlist[v_ind] == '1' or self._style.bundle:
                                    self._conds(xy, istrue=True)
                                else:
                                    self._conds(xy, istrue=False)
                                xy_plot.append(xy)
                            v_ind += 1
                    creg_b = sorted(xy_plot, key=lambda xy: xy[1])[0]
                    self._subtext(creg_b, hex(val))
                    self._line(qreg_t, creg_b, lc=self._style.cc,
                               ls=self._style.cline)
                '''
                #
                # draw special gates
                #
                if op.m_node_type == NodeType.MEASURE_GATE:
                    vv = self._creg_dict[c_idxs[0]]['index']
                    self._measure(q_xy[0], c_xy[0], vv)
                    #this_anc = this_anc + 1
                elif op.m_node_type == NodeType.RESET_NODE:
                    self._gate(q_xy[0], text='RESET')
                elif op.m_gate_type in [GateType.BARRIER_GATE]:
                    _barriers = {'coord': [], 'group': []}
                    for index, qbit in enumerate(q_idxs):
                        q_group = self._qreg_dict[qbit]['group']

                        if q_group not in _barriers['group']:
                            _barriers['group'].append(q_group)
                        _barriers['coord'].append(q_xy[index])
                    if self.plot_barriers:
                        self._barrier(_barriers, this_anc)
                elif len(op.m_control_qubits.to_list()) > 0:
                    disp = op.m_name
                    num_ctrl_qubits = len(op.m_control_qubits.to_list())
                    num_qargs = len(q_xy) - num_ctrl_qubits
                    # set the ctrl qbits to open or closed
                    color_str = self.get_op_color(op.m_name)
                    self.set_multi_ctrl_bits(
                        (2**num_ctrl_qubits) - 1, num_ctrl_qubits, q_xy, color_str)

                    # add qubit-qubit wiring
                    self._line(qreg_b, qreg_t,
                               lc=self._style.dispcol[color_str])
                    if num_qargs == 1:
                        if param:
                            self._gate(q_xy[num_ctrl_qubits], wide=_iswide,
                                       text=disp,
                                       fc=self._style.dispcol[color_str],
                                       subtext='{}'.format(param))
                        else:
                            fcx = op.m_name if op.m_name in self._style.dispcol else 'multi'
                            self._gate(q_xy[num_ctrl_qubits], wide=_iswide, text=disp,
                                       fc=self._style.dispcol[fcx])
                    elif num_qargs == 2:
                        # CNOT
                        if op.m_gate_type == GateType.CNOT_GATE:
                            #self._cnot(q_xy, qreg_b, qreg_t, num_ctrl_qubits+1)
                            color = self._style.dispcol['CNOT']
                            if self._style.name != 'Q1':
                                self._line(qreg_b, qreg_t,
                                           lc=color)
                            else:
                                self._line(qreg_b, qreg_t,
                                           zorder=PORDER_LINE + 1)
                            cx_xy = q_xy[num_ctrl_qubits:]
                            self._cnot(cx_xy, qreg_b, qreg_t)

                        # cz for latexmode
                        elif op.m_gate_type == GateType.CZ_GATE:
                            disp = op.m_name
                            if self._style.name != 'Q1':
                                color = self._style.dispcol['CZ']
                                self._ctrl_qubit(q_xy[num_ctrl_qubits],
                                                 fc=color,
                                                 ec=color)
                                self._ctrl_qubit(q_xy[num_ctrl_qubits + 1],
                                                 fc=color,
                                                 ec=color)
                            else:
                                self._ctrl_qubit(q_xy[num_ctrl_qubits])
                                self._ctrl_qubit(q_xy[num_ctrl_qubits + 1])
                            # add qubit-qubit wiring
                            if self._style.name != 'Q1':
                                self._line(qreg_b, qreg_t,
                                           lc=color)
                            else:
                                self._line(qreg_b, qreg_t,
                                           zorder=PORDER_LINE + 1)

                            cz_xy = q_xy[num_ctrl_qubits:]
                            self._cz(cz_xy, qreg_b, qreg_t)
                        # control gate
                        elif op.m_gate_type in [GateType.CU_GATE, GateType.CPHASE_GATE]:
                            disp = op.m_name

                            color = None
                            if self._style.name != 'Q1':
                                if op.m_name == 'CPHASE':
                                    color = self._style.dispcol['CPHASE']
                                else:
                                    color = self._style.dispcol['U4']

                            self._ctrl_qubit(
                                q_xy[num_ctrl_qubits], fc=color, ec=color)
                            if param:
                                self._gate(q_xy[num_ctrl_qubits + 1], wide=_iswide,
                                           text=disp,
                                           fc=color,
                                           subtext='{}'.format(param))
                            else:
                                self._gate(q_xy[num_ctrl_qubits + 1], wide=_iswide, text=disp,
                                           fc=color)
                            # add qubit-qubit wiring
                            self._line(qreg_b, qreg_t, lc=color)
                        # swap gate
                        elif op.m_gate_type == GateType.SWAP_GATE:
                            self._swap(q_xy[num_ctrl_qubits],
                                       self._style.dispcol['SWAP'])
                            self._swap(q_xy[num_ctrl_qubits + 1],
                                       self._style.dispcol['SWAP'])
                            # add qubit-qubit wiring
                            self._line(qreg_b, qreg_t,
                                       lc=self._style.dispcol['SWAP'])

                        # iswap gate
                        elif op.m_gate_type in [GateType.ISWAP_THETA_GATE, GateType.ISWAP_GATE, GateType.SQISWAP_GATE]:
                            self._swap_gate(
                                q_xy, op.m_gate_type, param, op.m_is_dagger, ctrl_qubits=num_ctrl_qubits)
                            # add qubit-qubit wiring
                            self._line(qreg_b, qreg_t,
                                       lc=self._style.dispcol['ISWAP'])
                            # self._custom_multiqubit_gate(q_xy, c_xy, wide=_iswide,
                            #                              fc=self._style.dispcol[op.m_name],
                            #                              text=op.m_name)

                        # Custom gate
                        elif op.m_gate_type in [GateType.RXX_GATE, GateType.RYY_GATE, GateType.RZZ_GATE, GateType.RZX_GATE]:
                            self._custom_multiqubit_gate(q_xy, c_xy, wide=True,
                                                         text=op.m_gate_type, subtext=f"{param}")
                        else:
                            self._custom_multiqubit_gate(q_xy, c_xy, wide=True,
                                                         text='Unitary')
                    else:
                        self._custom_multiqubit_gate(
                            q_xy[num_ctrl_qubits:], wide=True, fc=self._style.dispcol['Unitary'],
                            text=disp)

                #
                # draw single qubit gates
                #
                elif len(q_xy) == 1:
                    disp = op.m_name
                    if param:
                        self._gate(q_xy[0], wide=_iswide, text=disp,
                                   subtext=str(param))
                    else:
                        self._gate(q_xy[0], wide=_iswide, text=disp)
                #
                # draw multi-qubit gates (n=2)
                #
                elif len(q_xy) == 2:
                    # CNOT
                    if op.m_gate_type == GateType.CNOT_GATE:
                        self._cnot(q_xy, qreg_b, qreg_t, 1)
                    # cz for latexmode
                    elif op.m_gate_type == GateType.CZ_GATE:
                        self._cz(q_xy, qreg_b, qreg_t)
                    # control gate
                    elif op.m_gate_type in [GateType.CU_GATE, GateType.CPHASE_GATE]:
                        disp = op.m_name

                        color = None
                        if self._style.name != 'Q1':
                            if op.m_name == 'CPHASE':
                                color = self._style.dispcol['CPHASE']
                            else:
                                color = self._style.dispcol['U4']

                        self._ctrl_qubit(q_xy[0], fc=color, ec=color)
                        if param:
                            self._gate(q_xy[1], wide=_iswide,
                                       text=disp,
                                       fc=color,
                                       subtext='{}'.format(param))
                        else:
                            self._gate(q_xy[1], wide=_iswide, text=disp,
                                       fc=color)
                        # add qubit-qubit wiring
                        self._line(qreg_b, qreg_t, lc=color)
                    # swap gate
                    elif op.m_gate_type == GateType.SWAP_GATE:
                        self._swap(q_xy[0], self._style.dispcol['SWAP'])
                        self._swap(q_xy[1], self._style.dispcol['SWAP'])
                        # add qubit-qubit wiring
                        self._line(qreg_b, qreg_t,
                                   lc=self._style.dispcol['SWAP'])

                    # iswap gate
                    elif op.m_gate_type in [GateType.ISWAP_THETA_GATE, GateType.ISWAP_GATE, GateType.SQISWAP_GATE]:
                        self._swap_gate(q_xy, op.m_gate_type,
                                        param, op.m_is_dagger)
                        # add qubit-qubit wiring
                        self._line(qreg_b, qreg_t,
                                   lc=self._style.dispcol['ISWAP'])

                    # Custom gate
                    elif op.m_gate_type in [GateType.RXX_GATE, GateType.RYY_GATE, GateType.RZZ_GATE, GateType.RZX_GATE]:
                        disp = op.m_name
                        self._custom_multiqubit_gate(q_xy, c_xy, wide=True,
                                                     text=disp, subtext=str(param))

                    else:
                        self._custom_multiqubit_gate(q_xy, c_xy, wide=True,
                                                     text='Unitary')
                #
                # draw multi-qubit gates (n=3)
                #
                elif len(q_xy) in range(3, 6):
                    # Unitary
                    self._custom_multiqubit_gate(q_xy, c_xy, wide=True,
                                                 text='Unitary')
                    '''
                    if op.m_name == 'cswap':
                        self._ctrl_qubit(q_xy[0],
                                         fc=self._style.dispcol['multi'],
                                         ec=self._style.dispcol['multi'])
                        self._swap(q_xy[1], self._style.dispcol['multi'])
                        self._swap(q_xy[2], self._style.dispcol['multi'])
                        # add qubit-qubit wiring
                        self._line(qreg_b, qreg_t, lc=self._style.dispcol['multi'])
                    # ccx gate
                    elif op.m_name == 'ccx' or op.m_name == 'c3x' or op.m_name == 'c4x':
                        num_ctrl_qubits = op.op.num_ctrl_qubits
                        # set the ctrl qbits to open or closed
                        self.set_multi_ctrl_bits(op.op.ctrl_state, num_ctrl_qubits, q_xy, 'multi')
                        if self._style.name != 'Q1':
                            self._tgt_qubit(q_xy[num_ctrl_qubits], fc=self._style.dispcol['multi'],
                                            ec=self._style.dispcol['multi'],
                                            ac=self._style.dispcol['target'])
                        else:
                            self._tgt_qubit(q_xy[num_ctrl_qubits], fc=self._style.dispcol['target'],
                                            ec=self._style.dispcol['multi'],
                                            ac=self._style.dispcol['multi'])
                        # add qubit-qubit wiring
                        self._line(qreg_b, qreg_t, lc=self._style.dispcol['multi'])
                    # custom gate
                    else:
                        self._custom_multiqubit_gate(q_xy, c_xy, wide=_iswide,
                                                     text=getattr(op.op, 'label', None) or op.m_name)
                    '''
                # draw custom multi-qubit gate
                elif len(q_xy) > 5:
                    self._custom_multiqubit_gate(q_xy, c_xy, wide=True,
                                                 text='Unitary')
                else:
                    print('Invalid gate %s', op.m_name)

            # layer end
            tmp_b_first_time = True
            cur_fold = 0
            cur_fold_cnt = 0
            bb_fold = False
            for ii in q_anchors:
                if q_anchors[ii].b_fold:
                    if tmp_b_first_time:
                        tmp_b_first_time = False
                        self.layer_offset_recode.append(self.layer_offset)
                        self.layer_offset = 0.0
                        n_fold += 1
                        cur_fold, cur_fold_cnt = q_anchors[ii].get_fold_info()
                        bb_fold = True
                    q_anchors[ii].b_fold = False

            if bb_fold:
                for index, reg in self._qreg_dict.items():
                    q_anchors[index].set_fold_info(cur_fold, cur_fold_cnt)

                for index, reg in self._creg_dict.items():
                    c_anchors[index].set_fold_info(cur_fold, cur_fold_cnt)
                    c_anchors[index].b_fold = False
                    # c_anchors[index].update_fold_info(this_anc, layer_width, self.x_offset + my_offset_tmp, self.layer_offset)

            self.layer_offset += my_offset_tmp

            # for ii in range(0, len(q_anchors)):
            for ii in (q_anchors):
                q_anchors[ii].last_h_pos = cur_layer_max_x_pos

            # for ii in range(0, len(c_anchors)):
            for ii in (c_anchors):
                c_anchors[ii].last_h_pos = cur_layer_max_x_pos

            # layer end
            # tmp_b_first_time = True
            # for ii in q_anchors:
            #     if q_anchors[ii].b_fold:
            #         # my_offset_tmp = 0.0
            #         if tmp_b_first_time:
            #             tmp_b_first_time = False
            #             # self.layer_offset_recode.append(self.layer_offset)
            #             # self.layer_offset = 0.0
            #         q_anchors[ii].b_fold = False
            #         # break

            # adjust the column if there have been barriers encountered, but not plotted
            barrier_offset = 0
            if not self.plot_barriers:
                # only adjust if everything in the layer wasn't plotted
                barrier_offset = -1 if all([op.m_gate_type in
                                            [GateType.BARRIER_GATE]
                                            for op in layer]) else 0
            prev_anc = this_anc + layer_width + barrier_offset - 1
        #
        # adjust window size and draw horizontal lines
        #
        anchors = [q_anchors[ii].get_index() for ii in self._qreg_dict]
        if anchors:
            max_anc = max(anchors)
        else:
            max_anc = 0
        # n_fold = max(0, max_anc - 1) // self.fold
        # window size
        if max_anc > self.fold > 0:
            #self._cond['xmax'] = self.fold + 1 + self.x_offset + my_offset
            self._cond['xmax'] = 30.0
            self._cond['ymax'] = (n_fold + 1) * (self._cond['n_lines'] + 1) - 1
        else:
            self._cond['xmax'] = max_anc + 1 + self.x_offset + my_offset
            self._cond['ymax'] = self._cond['n_lines']
        # add horizontal lines
        for ii in range(n_fold + 1):
            feedline_r = (n_fold > 0 and n_fold > ii)
            feedline_l = (ii > 0)
            self._draw_regs_sub(ii, feedline_l, feedline_r)
        # draw gate number
        if self._style.index:
            for ii in range(max_anc):
                if self.fold > 0:
                    x_coord = ii % self.fold + 1
                    y_coord = - (ii // self.fold) * \
                        (self._cond['n_lines'] + 1) + 0.7
                else:
                    x_coord = ii + 1
                    y_coord = 0.7
                self.ax.text(x_coord, y_coord, str(ii + 1), ha='center',
                             va='center', fontsize=self._style.sfs,
                             color=self._style.tc, clip_on=True,
                             zorder=PORDER_TEXT)

    @staticmethod
    def param_parse(v):
        # create an empty list to store the parameters in
        param_parts = [None] * len(v)
        for i, e in enumerate(v):
            try:
                param_parts[i] = pi_check(e, output='mpl', ndigits=3)
            except TypeError:
                param_parts[i] = str(e)

            if param_parts[i].startswith('-'):
                param_parts[i] = '$-$' + param_parts[i][1:]

        param_parts = ', '.join(param_parts)
        return param_parts

    @staticmethod
    def format_numeric(val, tol=1e-5):
        if isinstance(val, complex):
            return str(val)
        elif complex(val).imag != 0:
            val = complex(val)
        abs_val = abs(val)
        if math.isclose(abs_val, 0.0, abs_tol=1e-100):
            return '0'
        if math.isclose(math.fmod(abs_val, 1.0),
                        0.0, abs_tol=tol) and 0.5 < abs_val < 9999.5:
            return str(int(val))
        if 0.1 <= abs_val < 100.0:
            return '{:.2f}'.format(val)
        return '{:.1e}'.format(val)

    @staticmethod
    def fraction(val, base=np.pi, n=100, tol=1e-5):
        abs_val = abs(val)
        for i in range(1, n):
            for j in range(1, n):
                if math.isclose(abs_val, i / j * base, rel_tol=tol):
                    if val < 0:
                        i *= -1
                    return fractions.Fraction(i, j)
        return None
