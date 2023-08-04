import numpy
import pyqpanda as pq
from typing import List
from pyqpanda.Visualization.matplotlib_draw import MatplotlibDrawer
from pyqpanda import circuit_layer

corners_str = ['┌', '┐', '└', '┘']
left_str, right_str = '┤', '├'
horizontal_str, vertical_str = '─', '│'
space = ' '


class CircuitComposer(pq.QCircuit):
    """CircuitComposer for representing quantum circuit using ascii art.

    Args
    ----------
    n_qubits: int
        Number of qubits of quantum circuit.

    circuit_layers : List[List[str]]
        List of each layer of quantum circuit, each layer of a list of string, 
        representing each line of ascii art.
    
    Examples
    --------
    >>> n_qubits = 6
    >>> qvm = pq.CPUQVM()
    >>> qvm.init_qvm()
    >>> q = qvm.qAlloc_many(n_qubits)

    >>> circ1 = CircuitComposer(n_qubits)
    >>> circuit = pq.QCircuit()
    >>> circuit << pq.H(q[0]) << pq.CNOT(q[0], q[1]) << pq.CNOT(q[1], q[2])

    >>> circ1.append(circuit)
    >>> circ1 << pq.BARRIER(q)
    >>> circ1.append(pq.QFT(q[4:]), "QFT")
    >>> print(circ1)
               ┌─┐                    !         
    q_0:  |0>─ ┤H├ ───■── ──────  ────! ────────
               └─┘ ┌──┴─┐             !         
    q_1:  |0>─ ─── ┤CNOT├ ───■──  ────! ────────
                   └────┘ ┌──┴─┐      !         
    q_2:  |0>─ ─── ────── ┤CNOT├  ────! ────────
                          └────┘      !         
    q_3:  |0>─ ─── ────── ──────  ────! ────────
                                      ! ┌──────┐
    q_4:  |0>─ ─── ────── ──────  ────! ┤4     ├
                                      ! │  QFT │
    q_5:  |0>─ ─── ────── ──────  ────! ┤5     ├
                                      ! └──────┘
    
    >>> print(circ1.circuit)
              ┌─┐                   !                         
    q_0:  |0>─┤H├ ───■── ────── ────! ─── ────────────── ─── ─
              └─┘ ┌──┴─┐            !                         
    q_1:  |0>──── ┤CNOT├ ───■── ────! ─── ────────────── ─── ─
                  └────┘ ┌──┴─┐     !                         
    q_2:  |0>──── ────── ┤CNOT├ ────! ─── ────────────── ─── ─
                         └────┘     !                         
    q_3:  |0>──── ────── ────── ────! ─── ────────────── ─── ─
                                    !                    ┌─┐  
    q_4:  |0>──── ────── ────── ────! ─── ───────■────── ┤H├ X
                                    ! ┌─┐ ┌──────┴─────┐ └─┘ │
    q_5:  |0>──── ────── ────── ────! ┤H├ ┤CR(1.570796)├ ─── X
                                    ! └─┘ └────────────┘      
    """

    def __init__(self, n_qubits: int) -> None:
        super().__init__()
        self.n_qubits = n_qubits

        # Saving all circuit layers, each layer is a list of string
        self.circuit_layers = [self.__get_prefix_circuit()]

        self.circuit_latex_layers = self.__get_latex_prefix_circuit()

        self.circuit_pic_layers = []

    def __lshift__(self, circ: pq.QCircuit):
        return self.insert(circ)

    def append(self, circ: pq.QCircuit, name: str = '') -> None:
        """Append a quantum circuit.

        Args
        ----------
        circ : pq.QCircuit
            Quantum circuit to be appended.
        name : str, optional
            Name of the appended quantum circuit, by default None

        Notes
        -----
        If `name` is empty string, then append the circuit with its string 
        representation not boxed.

        The name must be in full English and cannot contain numbers or symbols

        Even when a circuit is appended with a name, the composed circuit 
        string can be get by using `circuit` property.

        Examples
        --------
        >>> circ = CircuitComposer(2)
        >>> circuit = pq.QCircuit()
        >>> circuit << pq.H(qubits[0]) << pq.CNOT(qubits[0], qubits[1])
        >>> circ.append(circuit)
        >>> print(circ)
                   ┌─┐        
        q_0:  |0>─ ┤H├ ───■──
                   └─┘ ┌──┴─┐
        q_1:  |0>─ ─── ┤CNOT├
                       └────┘
        
        >>> circ.append(circuit, "GHZ")
        >>> print(circ)
                   ┌─┐        ┌──────┐
        q_0:  |0>─ ┤H├ ───■── ┤0     ├
                   └─┘ ┌──┴─┐ │  GHZ │
        q_1:  |0>─ ─── ┤CNOT├ ┤1     ├
                       └────┘ └──────┘

        >>> print(circ.circuit)
                  ┌─┐        ┌─┐       
        q_0:  |0>─┤H├ ───■── ┤H├ ───■──
                  └─┘ ┌──┴─┐ └─┘ ┌──┴─┐
        q_1:  |0>──── ┤CNOT├ ─── ┤CNOT├
                      └────┘     └────┘
        """
        if name:
            self.__insert_circuit(circ, name)
        else:
            self.insert(circ)

    def insert(self, circ: pq.QCircuit):
        if self == circ:
            raise NotImplementedError("Circuit cannot insert on itself.")

        self = super().insert(circ)
        # append ASII
        self.circuit_layers.append(self.__get_circuit_strings(circ))

        # append latex
        circuit_latex = self.__get_circuit_latex(circ)
        for i in range(len(circuit_latex)):
            self.circuit_latex_layers[i + 1] = (self.circuit_latex_layers[i + 1][:-12]
                                                + circuit_latex[i])  # remove "&\nghost{}\\"

        # append pic
        self.circuit_pic_layers.append([circ, None])
        return self

    def __insert_circuit(self, circ: pq.QCircuit, name, gateType=None):

        self = super().insert(circ)

        # append ASII
        self.circuit_layers.append(self.__get_circuit_box_string(circ, name))

        # append latex
        circuit_latex = self.__get_circuit_box_latex(circ, name)
        for i in range(len(circuit_latex)):
            self.circuit_latex_layers[i + 1] = (self.circuit_latex_layers[i + 1][:-12]
                                                + circuit_latex[i])  # remove "&\nghost{}\\"

        # append pic
        circuit_pic = self.__get_circuit_box_pic(circ, name)
        self.circuit_pic_layers.append([circuit_pic, name])


    def __str__(self) -> str:
        result = ''

        circuit_text =[]
        for row in range(len(self.circuit_layers[0])):
            line =''
            for col in range(len(self.circuit_layers)):
                line+=self.circuit_layers[col][row]
            circuit_text.append(line)

        #

        # num = int(len(circuit_text)/100+1)
        # for i in range(num):
        for line in circuit_text:
            result += line + '\n'


        return result

    def draw_circuit(self, output=None, filename=None) -> str:
        """ draw a quantum circuit.

        Parameters
        ----------
        output : str，Output type: text、latex、pic

        filename : str, optional
           When the output type is pic,
           this parameter can be used to specify the path to save the picture

        Returns
        -------
        str, if 'text' return ASII of  circuit,
             if 'latex' return latex of  circuit,
             if 'pic' return picture save path  of  circuit,

        Notes
        -----
        If `output` is empty string, use text
        if filename is empty string,use 'QCircuit_pic.png'
        """
        if output is None:
            output = "text"

        if output == "text":
            return self.__str__()
        # output latex
        elif output == 'latex':
            result = ''
            for i in range(len(self.circuit_latex_layers)):
                result += self.circuit_latex_layers[i]
                result += '\n'
            return result

        elif output == 'pic':
            if filename is None:
                filename = 'QCircuit_pic.png'

            node_infos = []

            for i in range(len(self.circuit_pic_layers)):
                if self.circuit_pic_layers[i][1] is not None:

                    infos = circuit_layer(self.circuit_pic_layers[i][0])[0]

                    infos[0][0].m_name = self.circuit_pic_layers[i][1]
                    node_infos.extend(infos)
                else:
                    infos = circuit_layer(self.circuit_pic_layers[i][0])[0]
                    node_infos.extend(infos)

            layer_info = circuit_layer(self)
            # print(info.m_node_type)
            qcd = MatplotlibDrawer(qregs=layer_info[1], cregs=layer_info[2], ops=node_infos, scale=0.7, fold=30)
            qcd.draw(filename,False)
            return filename

    @property
    def circuit(self):
        """Return original quantum circuit representation."""
        circ_text = pq.draw_qprog_text(self, output_file='')
        return circ_text

    def __get_circuit_string_lists(self, circ: pq.QCircuit = None) -> List[str]:
        """Get quantum circuit string list.

        Parameters
        ----------
        circ : pq.QCircuit, optional
            Quantum circuit, by default None.

        Returns
        -------
        List[str]
            string representation of `circ`.
        """
        qvm = pq.CPUQVM()
        qvm.init_qvm()
        qubits = qvm.qAlloc_many(self.n_qubits)

        circuit = pq.QCircuit()
        for q in qubits:
            circuit << pq.I(q)

        if circ:
            circuit << circ

        circ_text = pq.draw_qprog_text(circuit, output_file='')
        circ_text = circ_text.strip('\n')
        circ_text = circ_text.split('\n')

        # destroy qvm
        qvm.finalize()
        return circ_text

    def __get_circuit_latex_lists(self, circ: pq.QCircuit = None) -> List[str]:
        """Get quantum circuit latex list.

        Parameters
        ----------
        circ : pq.QCircuit, optional
            Quantum circuit, by default None.

        Returns
        -------
        List[str]
            latex string representation of `circ`.
        """
        qvm = pq.CPUQVM()
        qvm.init_qvm()
        qubits = qvm.qAlloc_many(self.n_qubits)

        circuit = pq.QCircuit()

        for q in qubits:
            circuit << pq.I(q)

        if circ:
            circuit << circ

        circ_text = pq.draw_qprog_latex(circuit, output_file='')
        circ_text = circ_text.strip('\n')
        circ_text = circ_text.split('\n')

        # Remove the head and tail
        circ_text = circ_text[8:-2]

        # Remove I gate
        new_lines = []
        for line in circ_text:
            line = line[line.find("I") + 3:]
            new_lines.append(line)
        # destroy qvm
        qvm.finalize()
        return new_lines

    def __get_prefix_circuit(self):
        circ_text = self.__get_circuit_string_lists()

        # Note: Only 1000 qubits are considered, i.e., qubit width is 3 by 
        # default.
        pos_1 = circ_text[0].find('┌')
        if pos_1 == -1:
            raise ValueError("Unable to generate circuit prefix.")

        new_lines = [circ[0:pos_1] for circ in circ_text]
        return new_lines

    def __get_latex_prefix_circuit(self):
        circ_text = []

        # latex header
        circ_text.append("\\documentclass[border=2px]{standalone}\n"
                         "\n"
                         "\\usepackage[braket, qm]{qcircuit}\n"
                         "\\usepackage{graphicx}\n"
                         "\n"
                         "\\begin{document}\n"
                         "\\scalebox{1.0}{\n"
                         "\\Qcircuit @C = 1.0em @R = 0.5em @!R{ \\\\")

        # latex mid
        for i in range(self.n_qubits):
            circ_text.append(r"\nghost{q_{%d}\ket{0}}&\lstick{q_{%d}\ket{0}}&\nghost{}\\" % (i, i))

        # latex end
        circ_text.append("\\\\ }} \n"
                         "\\end{document}")
        return circ_text

    def __get_circuit_strings(self, circ: pq.QCircuit) -> List[str]:
        circ_text = self.__get_circuit_string_lists(circ)
        pos = circ_text[0].find('┐')
        pos2 = 8 # circ_text[1].find('>')
        if pos == -1:
            raise ValueError("Unable to generate circuit text representation.")

        new_lines = [circuit[pos + 1:] for circuit in circ_text[:2*self.n_qubits+1]]
        new_lines.extend([circuit[ pos2 + 1:] for circuit in circ_text[2*self.n_qubits+1:]])

        # remove '>'
        for row in range(len(new_lines)):
            new_lines[row] = new_lines[row].rstrip('>')

        # join
        join_row = len(self.circuit_layers[0])
        while join_row < len(new_lines):
            join_row += 1
            for i in range(len(self.circuit_layers[0])):
                new_lines[i] += new_lines[join_row]
                join_row += 1
        new_lines = new_lines[:len(self.circuit_layers[0])]
        return new_lines

    def __get_circuit_latex(self, circ: pq.QCircuit) -> List[str]:
        circ_text = self.__get_circuit_latex_lists(circ)

        return circ_text

    def __get_circuit_box_string(self,
                                 circ: pq.QCircuit, name: str) -> List[str]:
        """Get box string representation of quantum circuit.

        Parameters
        ----------
        circ : pq.QCircuit
            Quantum circuit.
        name : str
            Name of quantum circuit.

        Returns
        -------
        List[str]
            Box string representaion of `circ`.
        """
        qubits = pq.get_all_used_qubits_to_int(circ)
        qubits.sort()

        if not qubits:
            return ['' for _ in range(2 * self.n_qubits + 1)]

        # two 1's represent two spaces
        qubit_witdh = len(str(qubits[-1]))
        width = 1 + qubit_witdh + 1 + len(name) + 1 + 1

        all_qubits = list(range(self.n_qubits))
        qubit_index = [all_qubits.index(element) for element in qubits]
        min_index = qubit_index[0]
        max_index = qubit_index[-1]

        lines = []

        # Empty qubits
        for i in range(min_index):
            lines.append(space * width)
            lines.append(horizontal_str * width)

        first_line = corners_str[0] + horizontal_str * (width - 2) + corners_str[1]
        lines.append(first_line)

        # middle lines
        middle_lines = 2 * (max_index - min_index) + 1
        for i in range(middle_lines):
            if i % 2 == 0:
                q = all_qubits[min_index + i // 2]
                str_qubit = str(q) if q in qubits else space * qubit_witdh
                if i != (max_index - min_index):
                    line = '{}{}{}{}'.format(left_str,
                                             str_qubit,
                                             space * (width - 2 - len(str_qubit)),
                                             right_str)
                else:
                    # name line
                    line = '{}{}{}{}{}{}'.format(left_str,
                                                 str_qubit,
                                                 space,
                                                 name,
                                                 space,
                                                 right_str
                                                 )
            elif i == max_index - min_index:
                # circuit name
                line = '{}{}{}{}{}'.format(vertical_str,
                                           space * (qubit_witdh + 1),
                                           name,
                                           space,
                                           vertical_str)
            else:
                line = vertical_str + space * (width - 2) + vertical_str

            lines.append(line)

        last_line = corners_str[2] + horizontal_str * (width - 2) + corners_str[3]
        lines.append(last_line)

        # Empty qubits
        for i in range(max_index, self.n_qubits - 1):
            lines.append(horizontal_str * width)
            lines.append(space * width)


        return lines

    def __get_circuit_box_latex(self, circ: pq.QCircuit, name: str) -> List[str]:
        """Get box string of latex type for the quantum circuit.

           Parameters
           ----------
           circ : pq.QCircuit
               Quantum circuit.
           name : str
               Name of quantum circuit.

           Returns
           -------
           List[str]
               Box string  of `circ`.
       """
        qubits = pq.get_all_used_qubits_to_int(circ)
        qubits.sort()

        if not qubits:
            return [r'&\nghost{}\\' for _ in range(self.n_qubits)]

        # two 1's represent two spaces
        qubit_witdh = len(str(qubits[-1]))
        width = 1 + qubit_witdh + 1 + len(name) + 1 + 1

        all_qubits = list(range(self.n_qubits))
        qubit_index = [all_qubits.index(element) for element in qubits]
        min_index = qubit_index[0]
        max_index = qubit_index[-1]

        lines = []
        # Empty qubits
        for i in range(min_index):
            lines.append(r"&\qw &\qw &\nghost{}\\")

        # middle lines
        middle_lines = (max_index - min_index)
        lines.append(r"&\multigate {%d}{\mathcal{%s}}_<<<{%d} &\qw &\nghost{}\\" % (middle_lines, name, min_index))
        for i in range(middle_lines):
            lines.append(r"&\ghost{\mathcal{%s}}_<<<{%d} &\qw  &\nghost{}\\" % (name, min_index + i + 1))

        # Empty qubits
        for i in range(max_index, self.n_qubits - 1):
            lines.append(r"&\qw &\qw &\nghost{}\\")

        return lines

    def __get_circuit_box_pic(self, circ: pq.QCircuit, name: str) -> List[str]:
        """Packaging a circuit as a QOracle ,
           removing detailed information about the circuit to improve efficiency

           Parameters
           ----------
           circ : pq.QCircuit
               Quantum circuit.
           name : str
               Name of quantum circuit.

           Returns
           -------
           pq.QCircuit,A circuit containing a QOracle

       """
        q = pq.get_all_used_qubits(circ)

        if not q:
            return pq.QCircuit()

        new_circ = pq.QCircuit()

        new_circ << pq.QOracle(q, numpy.eye((2 ** len(q))))
        return new_circ
