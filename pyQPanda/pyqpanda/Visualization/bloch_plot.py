try:
    from matplotlib import animation
    import matplotlib.pyplot as plt
    from matplotlib import get_backend
    from mpl_toolkits.mplot3d import Axes3D
    plt.switch_backend('TKAgg')
    HAS_MATPLOTLIB = True
except:
    HAS_MATPLOTLIB = False
    pass

import numpy as np
from numpy import pi
from scipy import sparse
from .quantum_state_plot import state_to_density_matrix
from pyqpanda import circuit_layer
from math import sin, cos, acos, sqrt
from .bloch import Bloch
import matplotlib

def count_pauli(i):

    i = i - ((i >> 1) & 0x55555555)
    i = (i & 0x33333333) + ((i >> 2) & 0x33333333)
    return (((i + (i >> 4) & 0xF0F0F0F) * 0x1010101) & 0xffffffff) >> 24


class pauli(object):

    def __init__(self, z, x):
        self.z = z
        self.x = x

    def to_matrix(self):

        _x, _z = self.x, self.z
        n = 2**len(_x)
        twos_array = 1 << np.arange(len(_x))
        xs = np.array(_x).dot(twos_array)
        zs = np.array(_z).dot(twos_array)
        rows = np.arange(n+1, dtype=np.uint)
        columns = rows ^ xs
        global_factor = (-1j)**np.dot(np.array(_x, dtype=np.uint), _z)
        data = global_factor*(-1)**np.mod(count_pauli(zs & rows), 2)
        matrix = sparse.csr_matrix((data, columns, rows), shape=(n, n))
        return matrix.toarray()


def get_single_paulis(num_qubits, index):
    labels = ['X', 'Y', 'Z']

    single_pauli = []
    for label in labels:

        z = np.zeros(len(label), dtype=np.bool)
        x = np.zeros(len(label), dtype=np.bool)
        for i, char in enumerate(label):
            if char == 'X':
                x[-i - 1] = True
            elif char == 'Z':
                z[-i - 1] = True
            elif char == 'Y':
                z[-i - 1] = True
                x[-i - 1] = True
            elif char != 'I':
                raise RuntimeError("Pauli error")

        tmp = pauli(z, x)

        pauli_z = np.zeros(num_qubits, dtype=np.bool)
        pauli_x = np.zeros(num_qubits, dtype=np.bool)

        pauli_z[index] = tmp.z[0]
        pauli_x[index] = tmp.x[0]
        single_pauli.append(pauli(pauli_z, pauli_x))
    return single_pauli


def plot_bloch_vector(bloch, title="bloch", axis_obj=None, fig_size=None):
    """Draw a quantum state bloch view

    Args:
        state : the quantum state 
        title : the figure title 
        fig_size : the figure size 

    Returns: 
        bloch figure shows quantum state

    """
    if not HAS_MATPLOTLIB:
        raise ImportError('Must have Matplotlib installed')
    if fig_size is None:
        fig_size = (5, 5)
    bloch_obj = Bloch(axes=axis_obj)
    bloch_obj.add_vectors(bloch)
    bloch_obj.render(title=title)
    if axis_obj is None:
        fig = bloch_obj.fig
        fig.set_size_inches(fig_size[0], fig_size[1])
        if get_backend() in ['module://ipykernel.pylab.backend_inline',
                             'nbAgg']:
            plt .close(fig)
        plt.show()
        return fig
    return None


def plot_bloch_multivector(state, title='', fig_size=None):
    """Draw a quantum state bloch view

    Args:
        state : the quantum state 
        title : the figure title 
        fig_size : the figure size 

    Returns: 
        bloch figure shows quantum state

    """

    if not HAS_MATPLOTLIB:
        raise ImportError('Must have Matplotlib installed')

    # get density matrix from quantum state vector
    state = state_to_density_matrix(state)
    print(state)
    num = int(np.log2(len(state)))

    width, height = plt.figaspect(1/num)
    fig = plt.figure(figsize=(width, height))
    for qubit in range(num):
        axis_obj = fig.add_subplot(1, num, qubit + 1, projection='3d')
        pauli_singles = get_single_paulis(num, qubit)

        bloch_state = list(
            map(lambda x: np.real(np.trace(np.dot(x.to_matrix(), state))),
                pauli_singles))
        plot_bloch_vector(bloch_state, "qubit " + str(qubit), axis_obj=axis_obj,
                          fig_size=fig_size)
    fig.suptitle(title, fontsize=16)
    if get_backend() in ['module://ipykernel.pylab.backend_inline',
                         'nbAgg']:
        plt.close(fig)
    plt.show()
    return fig


def normalize(v, tolerance=0.00001):
    mag2 = sum(n * n for n in v)
    if abs(mag2 - 1.0) > tolerance:
        mag = sqrt(mag2)
        v = tuple(n / mag for n in v)
    return np.array(v)


class gate_node(object):

    def __init__(self, name, theta):
        self.name = name
        self.theta = theta


class PlotVector:

    def __init__(self):
        self._val = None

    @staticmethod
    def from_axisangle(theta, v):
        v = normalize(v)

        new_quaternion = PlotVector()
        new_quaternion._axisangle_to_q(theta, v)
        return new_quaternion

    @staticmethod
    def from_value(value):
        new_quaternion = PlotVector()
        new_quaternion._val = value
        return new_quaternion

    def _axisangle_to_q(self, theta, v):
        x = v[0]
        y = v[1]
        z = v[2]

        w = cos(theta/2.)
        x = x * sin(theta/2.)
        y = y * sin(theta/2.)
        z = z * sin(theta/2.)

        self._val = np.array([w, x, y, z])

    def __mul__(self, b):

        if isinstance(b, PlotVector):
            return self._multiply_with_quaternion(b)
        elif isinstance(b, (list, tuple, np.ndarray)):
            if len(b) != 3:
                raise RuntimeError(
                    "Input vector has invalid length {0}".format(len(b)))
            return self._multiply_with_vector(b)
        else:
            raise RuntimeError(
                "Multiplication with unknown type {0}".format(type(b)))

    def _multiply_with_quaternion(self, q_2):
        w_1, x_1, y_1, z_1 = self._val
        w_2, x_2, y_2, z_2 = q_2._val
        w = w_1 * w_2 - x_1 * x_2 - y_1 * y_2 - z_1 * z_2
        x = w_1 * x_2 + x_1 * w_2 + y_1 * z_2 - z_1 * y_2
        y = w_1 * y_2 + y_1 * w_2 + z_1 * x_2 - x_1 * z_2
        z = w_1 * z_2 + z_1 * w_2 + x_1 * y_2 - y_1 * x_2

        result = PlotVector.from_value(np.array((w, x, y, z)))
        return result

    def _multiply_with_vector(self, v):
        q_2 = PlotVector.from_value(np.append((0.0), v))
        return (self * q_2 * self.get_conjugate())._val[1:]

    def get_conjugate(self):
        w, x, y, z = self._val
        result = PlotVector.from_value(np.array((w, -x, -y, -z)))
        return result

    def __repr__(self):
        theta, v = self.get_axisangle()
        return "(({0}; {1}, {2}, {3}))".format(theta, v[0], v[1], v[2])

    def get_axisangle(self):
        w, v = self._val[0], self._val[1:]
        theta = acos(w) * 2.0

        return theta, normalize(v)

    def tolist(self):
        return self._val.tolist()

    def vector_norm(self):
        _, v = self.get_axisangle()
        return np.linalg.norm(v)


def bloch_plot_dict(frames_per_gate):
    bloch_dict = dict()
    bloch_dict['x'] = ('x', PlotVector.from_axisangle(
        np.pi / frames_per_gate, [1, 0, 0]), '#1a8bbc')
    bloch_dict['y'] = ('y', PlotVector.from_axisangle(
        np.pi / frames_per_gate, [0, 1, 0]), '#c02ecc')
    bloch_dict['z'] = ('z', PlotVector.from_axisangle(
        np.pi / frames_per_gate, [0, 0, 1]), '#db7734')
    bloch_dict['s'] = ('s', PlotVector.from_axisangle(np.pi / 2 / frames_per_gate,
                                                      [0, 0, 1]), '#9b59b6')
    bloch_dict['sdg'] = ('sdg', PlotVector.from_axisangle(-np.pi / 2 / frames_per_gate, [0, 0, 1]),
                         '#8e44ad')
    bloch_dict['h'] = ('h', PlotVector.from_axisangle(np.pi / frames_per_gate, normalize([1, 0, 1])),
                       '#a96386')
    bloch_dict['t'] = ('t', PlotVector.from_axisangle(np.pi / 4 / frames_per_gate, [0, 0, 1]),
                       '#e74c3c')
    bloch_dict['tdg'] = ('tdg', PlotVector.from_axisangle(-np.pi / 4 / frames_per_gate, [0, 0, 1]),
                         '#c0392b')
    return bloch_dict


def traversal_circuit(circuit):

    origin_gates = ['H', 'X', 'Y', 'Z', 'RX', 'RY', 'RZ', 'S', 'T', 'U1']

    layers = circuit_layer(circuit)[0]

    if 0 == len(layers):
        raise RuntimeError("empty circuit")

    def get_layer_qubit_addr(node):
        return node.m_target_qubits[0].get_phy_addr()

    trans = 180. / pi

    cir = []
    tar_qubit = layers[0][0].m_target_qubits[0].get_phy_addr()
    for layer in layers:

        if get_layer_qubit_addr(layer[0]) != tar_qubit:
            raise RuntimeError("only one qubit circuits are supported")
        if layer[0].m_name not in origin_gates:
            raise RuntimeError("un supported gate".format(gate[0].name))
        if layer[0].m_name == 'H':
            cir.append(gate_node('h', 0))
        if layer[0].m_name == 'X':
            cir.append(gate_node('x', 0))
        if layer[0].m_name == 'Y':
            cir.append(gate_node('y', 0))
        if layer[0].m_name == 'Z':
            cir.append(gate_node('z', 0))
        if layer[0].m_name == 'U1':
            cir.append(gate_node('u1', np.round(layer[0].m_params[0] * trans)))
        if layer[0].m_name == 'RX':
            cir.append(gate_node('rx', np.round(layer[0].m_params[0] * trans)))
        if layer[0].m_name == 'RY':
            cir.append(gate_node('ry', np.round(layer[0].m_params[0] * trans)))
        if layer[0].m_name == 'RZ':
            cir.append(gate_node('rz', np.round(layer[0].m_params[0] * trans)))
        if layer[0].m_name == 'T' and (1 - layer[0].m_is_dagger):
            cir.append(gate_node('t', 0))
        if layer[0].m_name == 'T' and layer[0].m_is_dagger:
            cir.append(gate_node('tdg', 0))
        if layer[0].m_name == 'S' and (1 - layer[0].m_is_dagger):
            cir.append(gate_node('s', 0))
        if layer[0].m_name == 'S' and layer[0].m_is_dagger:
            cir.append(gate_node('sdg', 0))

    return cir


def plot_bloch_circuit(circuit,
                       trace=True,
                       saveas=None,
                       fps=20,
                       secs_per_gate=1):
    """Draw a quantum circuit bloch view , only support one qubit

    Args:
        circuit : the quantum circuit 
        trace : whether shows the trace 
        fps : flash fps 

    Returns: 
        bloch figure shows quantum circuit

    """

    if not HAS_MATPLOTLIB:
        raise ImportError("Must have Matplotlib installed.")

    frames_per_gate = fps
    time_between_frames = (secs_per_gate * 1000) / fps

    plot_cir = traversal_circuit(circuit)
    plot_dict = bloch_plot_dict(frames_per_gate)

    simple_gates = ['h', 'x', 'y', 'z', 's', 'sdg', 't', 'tdg']
    list_of_circuit_gates = []

    for gate in plot_cir:
        if gate.name in simple_gates:
            list_of_circuit_gates.append(plot_dict[gate.name])
        else:
            theta = gate.theta
            rad = np.deg2rad(theta)
            if gate.name == 'rx':
                quaternion = PlotVector.from_axisangle(
                    rad / frames_per_gate, [1, 0, 0])
                list_of_circuit_gates.append(
                    ('rx:'+str(theta), quaternion, '#f39c12'))
            elif gate.name == 'ry':
                quaternion = PlotVector.from_axisangle(
                    rad / frames_per_gate, [0, 1, 0])
                list_of_circuit_gates.append(
                    ('ry:'+str(theta), quaternion, '#0c93bf'))
            elif gate.name == 'rz':
                quaternion = PlotVector.from_axisangle(
                    rad / frames_per_gate, [0, 0, 1])
                list_of_circuit_gates.append(
                    ('rz:'+str(theta), quaternion, '#a06aad'))
            elif gate.name == 'u1':
                quaternion = PlotVector.from_axisangle(
                    rad / frames_per_gate, [0, 0, 1])
                list_of_circuit_gates.append(
                    ('u1:'+str(theta), quaternion, '#69a45a'))

    if len(list_of_circuit_gates) == 0:
        raise RuntimeError("Nothing to visualize.")

    starting_pos = normalize(np.array([0, 0, 1]))
    view=[-60,30]
    fig = plt.figure(figsize=(6, 6))
    if tuple(int(x) for x in matplotlib.__version__.split(".")) >= (3, 4, 0):
        _ax = Axes3D(
            fig, azim=view[0], elev=view[1], auto_add_to_figure=False
        )
        fig.add_axes(_ax)
    else:
        _ax = Axes3D(
            fig,
            azim=view[0],
            elev=view[1],
        )

    sphere = Bloch(axes=_ax)

    class PlotParams:

        def __init__(self):
            self.new_vec = []
            self.last_gate = -2
            self.colors = []
            self.pnts = []

    plot_parms = PlotParams()
    plot_parms.new_vec = starting_pos

    def plot_flash(i):
        sphere.clear()

        gate_counter = (i-1) // frames_per_gate
        if gate_counter != plot_parms.last_gate:
            plot_parms.pnts.append([[], [], []])
            plot_parms.colors.append(list_of_circuit_gates[gate_counter][2])

        if i == 0:
            sphere.add_vectors(plot_parms.new_vec)
            plot_parms.pnts[0][0].append(plot_parms.new_vec[0])
            plot_parms.pnts[0][1].append(plot_parms.new_vec[1])
            plot_parms.pnts[0][2].append(plot_parms.new_vec[2])
            plot_parms.colors[0] = 'r'
            sphere.make_sphere()
            return _ax

        plot_parms.new_vec = list_of_circuit_gates[gate_counter][1] * \
            plot_parms.new_vec

        plot_parms.pnts[gate_counter+1][0].append(plot_parms.new_vec[0])
        plot_parms.pnts[gate_counter+1][1].append(plot_parms.new_vec[1])
        plot_parms.pnts[gate_counter+1][2].append(plot_parms.new_vec[2])

        sphere.add_vectors(plot_parms.new_vec)
        if trace:
            for point_set in plot_parms.pnts:
                sphere.add_points([point_set[0], point_set[1], point_set[2]])

        sphere.vector_color = [list_of_circuit_gates[gate_counter][2]]
        sphere.point_color = plot_parms.colors
        sphere.point_marker = 'o'

        annotation_text = list_of_circuit_gates[gate_counter][0]
        annotationvector = [1.4, -0.45, 1.7]
        sphere.add_annotation(annotationvector,
                              annotation_text,
                              color=list_of_circuit_gates[gate_counter][2],
                              fontsize=30,
                              horizontalalignment='left')

        sphere.make_sphere()

        plot_parms.last_gate = gate_counter
        return _ax

    def init():
        sphere.vector_color = ['r']
        return _ax

    ani = animation.FuncAnimation(fig, plot_flash,
                                  range(frames_per_gate *
                                        len(list_of_circuit_gates)+1),
                                  init_func=init,
                                  blit=False,
                                  repeat=False,
                                  interval=time_between_frames)

    if saveas:
        ani.save(saveas, fps=30)
    plt.show()
    plt.close(fig)
    return None
