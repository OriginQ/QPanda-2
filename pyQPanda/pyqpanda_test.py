import numpy as np
from pyqpanda import *

online_api_key = "2302e020100301006072a8648ce3d020106052b8104001c041730150201010410b6d33ad8772eb9705e844394453a3c8a/6327"
# online_api_key = "302e020100301006072a8648ce3d020106052b8104001c041730150201010410b6d33ad8772eb9705e844394453a3c8a/6327"
test_api_key = "2302e020100301006072a8648ce3d020106052b8104001c0417301502010104104615aad5ebb390b75b8cac9311b97cac/10139"

def test_qcloud_device_error_mitigation():
    """
    Execute a series of quantum error mitigation protocols on a simulated quantum circuit.

    This function initializes a QCloud machine, configures it, and allocates qubits and
    classical bits. It constructs quantum programs for error mitigation and performs
    the following mitigation steps:

    - Zero-Noise Error (ZNE) mitigation
    - Permutation Error Correction (PEC) mitigation
    - Read-out Error mitigation

    Each mitigation step is executed with a specified number of iterations and a set of
    experimental parameters. The results of these mitigations are printed to the console.

    Args:
        None

    Returns:
        None

    Raises:
        No exceptions are raised by this function, but errors in the QCloud machine
        or the quantum programs may result in runtime errors that are not handled by
        this function.

    Note:
        This function is intended for testing and demonstration purposes within the
        pyQPanda package and assumes that the QCloud machine and quantum programs
        are correctly set up and configured for the quantum virtual machine or quantum
        cloud service environment.
    """

    machine = QCloud()
    machine.set_configure(72,72)

    machine.init_qvm(online_api_key,True)

    qlist = machine.qAlloc_many(72)
    clist = machine.cAlloc_many(72)

    measure_prog = QProg()
    measure_prog << CZ(qlist[52], qlist[53])\
                << Measure(qlist[53], clist[0])\

    pmeasure_prog = QProg()
    pmeasure_prog << CZ(qlist[52], qlist[53])\

    exps = ["XX"]
    noise_length = [1,1.2,1.4,1.5,1.6]

    zne_em_result = machine.zne_error_mitigation(pmeasure_prog, 1000, exps, noise_length);
    pec_em_result = machine.pec_error_mitigation(pmeasure_prog, 1000, exps);
    read_out_em_result = machine.read_out_error_mitigation(measure_prog, 1000, exps);

    print(zne_em_result)
    print(pec_em_result)
    print(read_out_em_result)

def test_qcloud_async():
    """
    Simulates quantum computations and measures outcomes on a quantum virtual machine or cloud service using the pyQPanda framework.

    This function initializes a quantum machine, configures it, allocates quantum and classical bits, constructs quantum circuits,
    converts them to OriginIR, and submits them for execution. It also queries and prints the state and results of the submitted tasks.

    Parameters:
        None

    Returns:
        None

    Steps performed:
        1. Instantiate a QCloud object to interface with quantum resources.
        2. Configure the quantum machine with specific parameters.
        3. Initialize the quantum virtual machine with an online API key.
        4. Allocate quantum and classical bits for measurements.
        5. Construct a quantum program with a series of quantum gates and measurements.
        6. Create a batch of quantum programs for parallel execution.
        7. Convert the quantum programs to OriginIR format.
        8. Submit the batch for execution on the real quantum chip and wait for the batch ID.
        9. Query the state and result of the batch task using the batch ID.
        10. Optionally, submit a single quantum program for execution and wait for its task ID.
        11. Query the state and result of the single task using the task ID.
        12. Print a completion message.

    Note:
        The function is designed to be a test routine and includes commented-out sections that can be enabled for testing specific scenarios.
    """
    machine = QCloud()
    machine.set_configure(72,72);

    # online
    machine.init_qvm(online_api_key,True)

    # test
    # machine.init_qvm(test_api_key,True)
    # machine.set_qcloud_url('http://oqcs.originqc.com')

    qlist = machine.qAlloc_many(6)
    clist = machine.cAlloc_many(6)

    # 构建量子程序，可以手动输入，也可以来自OriginIR或QASM语法文件等
    measure_prog = QProg()
    measure_prog << hadamard_circuit(qlist)\
                 << CZ(qlist[0], qlist[1])\
                << Measure(qlist[0], clist[0])\
                << Measure(qlist[1], clist[1])\
                << Measure(qlist[2], clist[2])

    batch_prog = [measure_prog for _ in range (6)]

    pmeasure_prog = QProg()
    pmeasure_prog  << hadamard_circuit(qlist)\
                   << CZ(qlist[0], qlist[1])
    
    originir_list = [convert_qprog_to_originir(prog, machine) for prog in batch_prog]
    
    batch_id = machine.async_batch_real_chip_measure(originir_list, 1000 ,real_chip_type.origin_72)
    print(batch_id)
    time.sleep(10)
    # batch_id = '8C3C5BDDA616E1A094B76A85473F3557'
    state, result = machine.query_batch_task_state_result(batch_id)
    print(state, result)

    # result = machine.real_chip_measure(measure_prog, 1000 ,real_chip_type.origin_72)
    # print(result)

    task_id = machine.async_real_chip_measure(measure_prog, 1000 ,real_chip_type.origin_72)
    time.sleep(10)

    # task_id = "633C6A6119AD631AE61DB7B1ADC5A17C"
    state, result = machine.query_task_state_result(task_id)
    print(state, result)

    print("test_qcloud_async finished.")

def test_qcloud_cluster_online():
    """
    Simulates and measures quantum circuits on a quantum virtual machine or quantum cloud service.

    This function initializes a QCloud instance with a specified configuration and performs a series of quantum operations.
    It constructs a quantum program that includes Hadamard gates, controlled-Z gates, and measurements on multiple qubits.
    The quantum program is executed multiple times in batch mode, and the results are printed.

    Parameters:
        None

    Returns:
        None

    The function performs the following steps:
    1. Sets up a QCloud instance with a specific configuration.
    2. Initializes the quantum virtual machine with the online API key.
    3. Allocates qubits and classical bits for measurements.
    4. Constructs a quantum program that applies a Hadamard gate, a CZ gate, and measurements on multiple qubits.
    5. Repeats the quantum program six times to create a batch of programs.
    6. Measures the results of the batch program execution on the quantum virtual machine.
    7. Prints the batch measurement results.

    Note: The function uses internal helper functions such as `hadamard_circuit`, `CZ`, `Measure`, and `batch_real_chip_measure`
          which are assumed to be defined within the same package.
    """
    machine = QCloud()
    machine.set_configure(72,72);

    # online
    machine.init_qvm(online_api_key,True)

    # # test
    # machine.init_qvm(test_api_key,True)
    # machine.set_qcloud_url('http://oqcs.originqc.com')

    qlist = machine.qAlloc_many(6)
    clist = machine.cAlloc_many(6)

    measure_prog = QProg()
    measure_prog << hadamard_circuit(qlist)\
                 << CZ(qlist[0], qlist[1])\
                << Measure(qlist[0], clist[0])\
                << Measure(qlist[1], clist[1])\
                << Measure(qlist[2], clist[2])

    batch_prog = [measure_prog for _ in range (6)]

    pmeasure_prog = QProg()
    pmeasure_prog  << hadamard_circuit(qlist)\
                   << CZ(qlist[0], qlist[1])

    # estimate_price_result = machine.estimate_price(10, 1000)
    batch_measure_result = machine.batch_real_chip_measure(batch_prog, 1000, real_chip_type.origin_72)
    # measure_result = machine.full_amplitude_measure(measure_prog, 1000);
    # pmeasure_result = machine.full_amplitude_pmeasure(pmeasure_prog, qlist);

    # print(estimate_price_result)
    print(batch_measure_result)
    # print(measure_result)
    # print(pmeasure_result)

def test_sparse_state_init():
    """
    Tests the initialization of a sparse state on a quantum virtual machine (QVM).

    This function creates a CPUQVM instance, configures it with a specified number of qubits,
    initializes the QVM, allocates qubits and classical bits, initializes a sparse state,
    and runs a simple quantum circuit. It then prints the probability distribution of the
    final quantum state.

    The test is conducted within the context of the pyQPanda package, which is designed for
    programming quantum computers using quantum circuits and gates. The function operates on a
    quantum circuit simulator (quantum virtual machine) or quantum cloud service.

    Parameters:
        None

    Returns:
        None

    Raises:
        None

    Example Usage:
        >>> test_sparse_state_init()
        Output: Probability distribution of the final quantum state
    """
    machine = CPUQVM()
    machine.set_configure(72,72)

    machine.init_qvm()

    qlist = machine.qAlloc_many(6)
    clist = machine.cAlloc_many(6)

    sparse_state = {'000000' : 0.5 + 0.5j, '000001' : 0.5 + 0.5j}
    machine.init_sparse_state(sparse_state, qlist)

    prog = QProg()
    prog << I(qlist[0])

    machine.directly_run(prog)  
    probs = machine.get_qstate();

    print(probs)

def test_qcloud_originir():
    """
    Tests the integration of QCloud service with a quantum virtual machine using an OriginIR quantum program.
    
    This function initializes a QCloud instance, configures it for a 72x72 quantum system, and initializes the quantum virtual machine.
    It then converts an OriginIR quantum program into a quantum program suitable for the QCloud's quantum virtual machine.
    Subsequently, it measures the results of the quantum program execution on the virtual machine and prints the outcome.
    
    Parameters:
    None
    
    Returns:
    None
    
    The function is intended for use within the pyQPanda package, specifically in the context of running quantum circuits
    and gates on a quantum circuit simulator or a quantum cloud service.
    """
    machine = QCloud()
    machine.set_configure(72,72);

    machine.init_qvm(online_api_key,True)

    # q = machine.qAlloc_many(5)
    # c = machine.cAlloc_many(5)

    grover_prog_value = convert_originir_to_qprog("./grover-2.txt", machine)

    result = machine.real_chip_measure(grover_prog_value[0], 1000, real_chip_type.origin_72)
    print(result)

def test_qcloud_pqc_encryption():
    """
    Tests the quantum permutation cipher (PQC) encryption capabilities of the QCloud quantum computing service.

    This function initializes a QCloud instance, configures it, and performs a series of operations to
    demonstrate the encryption and decryption process using PQC. It includes setting up a quantum virtual
    machine, allocating quantum and classical bits, constructing quantum circuits, converting them to
    OriginIR format, and executing batch and single task measurements on the quantum cloud.

    Key steps include:
    - Initializing the QCloud service with user authentication and configuration settings.
    - Setting up the quantum virtual machine with specific encryption parameters.
    - Allocating quantum and classical bits for quantum circuit operations.
    - Constructing and converting quantum circuits into OriginIR format.
    - Running batch and single task measurements on the quantum cloud service.
    - Querying the status of batch and single task executions and printing results.

    The function prints a message indicating the successful completion of the PQC encryption test.
    """
    machine = QCloud()
    machine.set_configure(72,72)

    # online
    # machine.init_qvm(online_api_key, True, True, 100)

    # test : http://qcloud4test.originqc.com/zh
    machine.init_qvm(user_token=online_api_key,
                     enable_logging=True,
                     log_to_console=True,
                     use_bin_or_hex=True,
                     enable_pqc_encryption=True)
    
    # machine.set_qcloud_url('http://oqcs.originqc.com')
    
    sym_iv, sym_kys = machine.get_pqc_encryption()
    machine.update_pqc_keys(sym_iv, sym_kys)

    qlist = machine.qAlloc_many(6)
    clist = machine.cAlloc_many(6)

    # 构建量子程序，可以手动输入，也可以来自OriginIR或QASM语法文件等
    measure_prog = QProg()
    measure_prog << H(qlist[0])\
                 << CNOT(qlist[0], qlist[1])\
                 << CNOT(qlist[1], qlist[2])\
                 << Measure(qlist[0], clist[0])\
                 << Measure(qlist[1], clist[1])

    batch_prog = [measure_prog for _ in range (2)]

    pmeasure_prog = QProg()
    pmeasure_prog << H(qlist[0])\
                  << CNOT(qlist[0], qlist[1])\
                  << CNOT(qlist[1], qlist[2])
    
    prog_string = convert_qprog_to_originir(measure_prog, machine)
    originir_list = [convert_qprog_to_originir(prog, machine) for prog in batch_prog]

    # 同步批量
    real_chip_measure_batch_result = machine.batch_real_chip_measure(batch_prog, 
                                                                     1000, 
                                                                     real_chip_type.origin_72, 
                                                                     False)
    print(real_chip_measure_batch_result)

    # 异步批量
    batch_id = machine.async_batch_real_chip_measure(batch_prog, 
                                                     1000 ,
                                                     real_chip_type.origin_72)
    
    print(batch_id) # batch_id = '9FC340C1C82A3351B8A08930A22B6909'
    
    # batch_id = 'DD65ECDB1431207491D1473BDF51305F';
    
    while(True):
        time.sleep(1)
        status, result = machine.query_batch_task_state_result(batch_id)
        print(status, result)

        if status == QCloud.TaskStatus.FINISHED.value:
            print(result)
            break

    # 同步单任务
    result = machine.real_chip_measure(measure_prog, 1000 ,real_chip_type.origin_72)
    print(result)

    # 异步单任务
    task_id = machine.async_real_chip_measure(measure_prog, 1000 ,real_chip_type.origin_72)
    print(task_id) # batch_id = '0225207844C8EAD60BF219227D8CCF50'

    while(True):
        time.sleep(2)

        status, result = machine.query_task_state_result(task_id)
        print(status, result)

        if status == QCloud.TaskStatus.FINISHED.value:
            print(result)
            break

    print("qcloud pqc_encryption passed.")

def test_qcloud_query():
    """
    Verifies the functionality of the QCloud query system for batch tasks.

    This function initializes a QCloud machine with specific configuration settings,
    initiates a quantum virtual machine with provided API keys, and queries the status
    of a batch task repeatedly until completion. It prints the task status and result,
    and upon completion, confirms the successful execution of the query.

    Parameters:
        None

    Returns:
        None

    Raises:
        None

    Usage:
        To test the QCloud query system, simply call this function. It is typically used
        within the pyQPanda package for debugging and ensuring the integrity of the
        quantum computation processes on the quantum circuit simulator or quantum cloud service.
    """
    machine = QCloud()
    machine.set_configure(72,72)

    # online
    machine.init_qvm(online_api_key, True, True)
    
    # batch_id = 'DD65ECDB1431207491D1473BDF51305F';

    batch_id = '591F1F60700F8527E5FF474123155603'
    
    while(True):
        time.sleep(1)
        status, result = machine.query_batch_task_state_result(batch_id)
        print(status, result)

        if status == QCloud.TaskStatus.FINISHED.value:
            print(result)
            break

    print("qcloud query passed.")

def test_hex_to_bin_result():
    """
    Executes a series of quantum operations and measurements on a quantum virtual machine or cloud service.

    This function initializes a quantum machine with specific configuration, allocates quantum and classical
    registers, constructs quantum circuits, and performs measurements. It also converts quantum programs to
    OriginIR format and measures the results on a real chip. The function prints the measurement results
    and confirms the completion of the test by printing a success message.

    The function performs the following steps:
    - Configures the quantum machine with the specified settings.
    - Initializes the quantum virtual machine with API keys.
    - Sets the URL for the quantum cloud service.
    - Allocates quantum and classical registers for the experiments.
    - Constructs a quantum program with Hadamard, CNOT, and Measure gates.
    - Creates a batch of identical quantum programs.
    - Converts the quantum programs to OriginIR format.
    - Measures the results on a real chip using the quantum cloud service.
    - Prints the batch measurement results.
    - Prints the OriginIR measurement results.
    - Confirms the successful execution of the test.

    Parameters:
        None

    Returns:
        None

    Raises:
        None

    Examples:
        The function is typically called as follows:
        >>> test_hex_to_bin_result()
    """
    machine = QCloud()
    machine.set_configure(72,72);

    # online
    # machine.init_qvm(online_api_key, True, True, 100)

    # test : http://qcloud4test.originqc.com/zh
    machine.init_qvm(test_api_key,
                     True,
                     True)
    
    machine.set_qcloud_url('http://oqcs.originqc.com')

    qlist = machine.qAlloc_many(6)
    clist = machine.cAlloc_many(6)

    # 构建量子程序，可以手动输入，也可以来自OriginIR或QASM语法文件等
    measure_prog = QProg()
    measure_prog << H(qlist[0])\
                 << CNOT(qlist[0], qlist[1])\
                 << CNOT(qlist[1], qlist[2])\
                << Measure(qlist[0], clist[0])\
                << Measure(qlist[1], clist[1])\
                << Measure(qlist[2], clist[2])

    batch_prog = [measure_prog for _ in range (6)]

    pmeasure_prog = QProg()
    pmeasure_prog << H(qlist[0])\
                  << CNOT(qlist[0], qlist[1])\
                  << CNOT(qlist[1], qlist[2])
    
    prog_string = convert_qprog_to_originir(measure_prog, machine)
    originir_list = [convert_qprog_to_originir(prog, machine) for prog in batch_prog]

    # real_chip_measure_result = machine.real_chip_measure(measure_prog, 1000, real_chip_type.origin_72, task_from=6)
    real_chip_measure_batch_result = machine.batch_real_chip_measure(batch_prog, 1000, real_chip_type.origin_72, False,task_from=6)
    
    # originir_result =  machine.real_chip_measure(prog_string, 1000, task_from=6)
    originir_list_result = machine.batch_real_chip_measure(originir_list, 1000, real_chip_type.origin_72, task_from=6)

    # fidelity_result = machine.get_state_fidelity(measure_prog, 1000, real_chip_type.origin_72)
    # qst_result = machine.get_state_tomography_density(measure_prog, 1000, real_chip_type.origin_72)
    # estimate_price_result = machine.estimate_price(10, 1000)
    # measure_result = machine.full_amplitude_measure(measure_prog, 1000);
    # pmeasure_result = machine.full_amplitude_pmeasure(pmeasure_prog, qlist);

    # print(real_chip_measure_result)
    print(real_chip_measure_batch_result)

    # print(originir_result)
    print(originir_list_result)

    print("qcloud test passed.")
    
    # print(qst_result)
    # print(fidelity_result)
    # print(estimate_price_result)

    # print(measure_result)
    # print(pmeasure_result)

def test_big_data_batch_result():
    """
    Tests the batch measurement functionality of the quantum circuit simulator or quantum cloud service within the pyQPanda package.
    
    The function initializes a quantum circuit with specific gates and measurements, and submits it for batch processing.
    It then queries the status of the batch task and prints the results once completed.

    Parameters:
        None

    Returns:
        None

    The function performs the following steps:
        - Connects to the quantum circuit simulator or quantum cloud service.
        - Configures the machine settings.
        - Initializes the quantum virtual machine.
        - Allocates quantum and classical bits.
        - Constructs a quantum program with gates and measurements.
        - Creates a batch of quantum programs for batch processing.
        - Submits the batch for real-chip measurement.
        - Polls the batch task status until completion.
        - Prints the batch result.
        - Submits another batch for non-real-chip measurement.
        - Prints the batch ID for non-real-chip measurement.
        - Confirms the test has passed.
    """
    machine = QCloud()
    machine.set_configure(72,72);

    # online
    # machine.init_qvm(online_api_key, True, True, 100)

    # test
    machine.init_qvm(test_api_key,
                     True,
                     True)
    
    machine.set_qcloud_url('http://oqcs.originqc.com')

    qlist = machine.qAlloc_many(6)
    clist = machine.cAlloc_many(6)

    # 构建量子程序，可以手动输入，也可以来自OriginIR或QASM语法文件等
    measure_prog = QProg()
    measure_prog << H(qlist[0])\
                 << CNOT(qlist[0], qlist[1])\
                 << CNOT(qlist[1], qlist[2])\
                << Measure(qlist[0], clist[0])\
                << Measure(qlist[1], clist[1])\
                << Measure(qlist[2], clist[2])
    
    big_measure_prog = QProg()
    for i in range(1024 * 64):
        big_measure_prog  << H(qlist[0])\
                          << CNOT(qlist[0], qlist[1])\
                          << CNOT(qlist[1], qlist[2])\
                          
    big_measure_prog << Measure(qlist[0], clist[0])\
                     << Measure(qlist[1], clist[1])\
                     << Measure(qlist[2], clist[2])
    
    batch_prog = [measure_prog for _ in range (6)]
    big_batch_prog = [big_measure_prog for _ in range (9)]

    # real_chip_measure_result = machine.real_chip_measure(measure_prog, 1000, real_chip_type.origin_72)
    # big_real_chip_measure_batch_result = machine.batch_real_chip_measure(big_batch_prog, 1000, real_chip_type.origin_72, True)
    # print(big_real_chip_measure_batch_result)
    
    # real_chip_measure_batch_result = machine.batch_real_chip_measure(batch_prog, 1000, real_chip_type.origin_72, True)
    # print(real_chip_measure_batch_result)
    
    big_batch_id = machine.async_batch_real_chip_measure(big_batch_prog, 1000, real_chip_type.origin_72, True)
    print(big_batch_id)
    
    while(True):
        time.sleep(5)
        status, result = machine.query_batch_task_state_result(big_batch_id)

        if status == QCloud.TaskStatus.FINISHED.value:
            print(result)
            break
    
    batch_id = machine.async_batch_real_chip_measure(batch_prog, 1000, real_chip_type.origin_72, False)
    print(batch_id)

    print('test_big_data_batch_result passed.')


def test_density():
    """
    Simulates a quantum circuit, converts the quantum state to a density matrix, and plots it.

    This function allocates quantum and classical registers, constructs a quantum
    circuit with various gates, runs the circuit on a quantum virtual machine (QVM),
    retrieves the quantum state, converts it to a density matrix, and plots the matrix.

    Parameters:
        None

    Returns:
        None

    Raises:
        RuntimeError: If the quantum circuit configuration or state is invalid.

    Note:
        This function utilizes the pyQPanda package for quantum computing operations.
        It assumes the presence of the CPUQVM class, QProg class, and related quantum
        gate and state manipulation functions, as well as the necessary libraries for
        running the QVM and plotting the density matrix.
    """

    machine = CPUQVM()
    machine.set_configure(50, 50)
    machine.init_qvm()

    q = machine.qAlloc_many(4)
    c = machine.cAlloc_many(4)

    prog = QProg()
    prog.insert(X(q[1]))\
        .insert(H(q[0]))\
        .insert(H(q[1]))\
        .insert(T(q[2]))\
        .insert(RX(q[1], 1))\
        .insert(RX(q[2], 2))\
        .insert(RX(q[3], 3))\
        .insert(RZ(q[1], np.pi))\
        .insert(RZ(q[2], np.pi))\
        .insert(RZ(q[3], np.pi))\
        .insert(random_qcircuit(q, 10))
    
    machine.directly_run(prog)
    result = machine.get_qstate()
    rho = state_to_density_matrix(result)
    plot_density_matrix(rho)
    machine.finalize()

def test_qcloud_multi_control_limit():
    """
    Simulates a quantum circuit on a QCloud quantum virtual machine using the pyQPanda package.
    
    This function initializes a QCloud instance with a specific configuration, sets up a quantum
    program with multiple control gates, and measures the outcomes on a real chip of the specified type.
    
    The quantum program includes a sequence of quantum gates such as X, H, and CNOT, and is
    decomposed using the LDD (Linear Discriminant Decomposition) method. The measured results are
    then printed out.
    
    Parameters:
        None
    
    Returns:
        None
    
    Notes:
        - This function uses `test_api_key` for initialization, which is typically used for testing purposes.
        - The URL for the QCloud service is set to 'http://oqcs.originqc.com'.
        - The function assumes that the necessary classes (`QCloud`, `QProg`, `X`, `H`, `CNOT`, `Measure`,
          `ldd_decompose`, and `real_chip_type`) are defined and available in the current scope.
    """
    machine = QCloud()
    machine.set_configure(72,72);

    # online
    # machine.init_qvm(online_api_key,True)

    # # test
    machine.init_qvm(test_api_key,True)
    machine.set_qcloud_url('http://oqcs.originqc.com')

    q = machine.qAlloc_many(6)
    c = machine.cAlloc_many(6)

    measure_prog = QProg()
    measure_prog << X(q[1])\
                 << X(q[2])\
                 << H(q[0]).control([q[1], q[2], q[3]])\
                 << CNOT(q[0], q[1])\
                 << CNOT(q[1], q[2]).control([q[1], q[2], q[3]])\
                 << Measure(q[0], c[0])
    
    decomposed_prog = ldd_decompose(measure_prog)
    
    measure_result = machine.real_chip_measure(decomposed_prog, 1000, real_chip_type.origin_72)

    print(measure_result)

def test_comm_protocol_encode_decode():
    """
    Test the communication protocol encoding and decoding for quantum circuits within the pyQPanda framework.

    This function performs the following steps:
    1. Initializes a CPUQVM instance and sets the configuration parameters.
    2. Allocates quantum and classical registers.
    3. Constructs a quantum circuit with various quantum gates.
    4. Creates a quantum program by appending the constructed circuit and additional quantum gates.
    5. Configures the communication protocol settings.
    6. Encodes the quantum program into a data format suitable for communication.
    7. Decodes the encoded data back into a quantum program using a quantum virtual machine.
    8. Prints the original and decoded quantum programs.
    9. Retrieves the unitary matrices for the original and decoded programs.
    10. Compares the unitary matrices to verify the integrity of the encoding and decoding process.
    11. Prints the result of the integrity check.

    This function is intended to be used for testing the robustness and correctness of the communication protocol in the pyQPanda package.
    """
    machine = CPUQVM()
    machine.set_configure(72,72)

    q = machine.qAlloc_many(8)
    c = machine.cAlloc_many(8)

    circuit = QCircuit()
    circuit << RXX(q[0], q[1], 2)
    circuit << RYY(q[0], q[1], 3)
    circuit << RZZ(q[0], q[1], 3)
    circuit << RZX(q[0], q[1], 4)

    prog = QProg()
    prog << random_qcircuit(q, 6)
    prog << H(q[0])
    prog << circuit.dagger()
    prog << CR(q[0], q[1], 4)
    prog << U1(q[0], 2)
    prog << U2(q[0], 2, 3)
    prog << U3(q[0], 2, 3, 4)
    prog << U4(q[0], 2, 3, 4, 5)
    prog << BARRIER(q)
    prog << Toffoli(q[0], q[1], q[2])

    config = CommProtocolConfig()

    config.open_mapping = True
    config.open_error_mitigation = False
    config.optimization_level = 2
    config.circuits_num = 5
    config.shots = 1000

    print(prog)

    encode_data = comm_protocol_encode(prog, config)

    decode_config = CommProtocolConfig()
    decode_progs, decode_config  = comm_protocol_decode(encode_data, machine)

    for prog in decode_progs:
        print(prog)

    encode_matrix = get_unitary(prog);
    decode_matrix = get_unitary(decode_progs[0]);

    import numpy as np
    def are_lists_equal(list1, list2, tolerance=1e-6):
        array1 = np.array(list1, dtype=complex)
        array2 = np.array(list2, dtype=complex)
        return np.allclose(array1, array2, atol=tolerance)

    if(are_lists_equal(encode_matrix, decode_matrix)):
        print("test_comm_protocol_encode_data passed.")
    else:
        print("test_comm_protocol_encode_data failed.")

def test_multi_control_decompose_and_convert_to_u3():
    """
    Tests the decomposition and conversion of a multi-control quantum circuit to the U3 basis,
    followed by a transformation to a standard basis and comparison of unitary matrices.

    This function allocates qubits and classical bits, constructs a multi-control quantum
    circuit with various gates, and then performs the following steps:

    1. Allocates resources for a quantum virtual machine (QVM) and sets the configuration.
    2. Allocates a number of qubits and classical bits.
    3. Constructs a quantum program (QProg) with multi-control gates.
    4. Decomposes the multi-control gates into a standard basis using a specified set of gates.
    5. Prints the decomposed quantum program.
    6. Constructs a standard basis quantum program for comparison.
    7. Converts the standard basis quantum program to the U3 basis.
    8. Prints the converted quantum program.
    9. Retrieves the unitary matrices of both the original and decomposed programs.
    10. Compares the unitary matrices for equality within a specified tolerance.
    11. Outputs the result of the unitary matrix comparison.

    Parameters:
    None

    Returns:
    None
    """
    machine = CPUQVM()
    machine.set_configure(72,72)

    q = machine.qAlloc_many(4)
    c = machine.cAlloc_many(4)

    multi_control_prog = QProg()
    multi_control_prog << H(q[1])
    multi_control_prog << H(q[2])
    multi_control_prog << H(q[3])
    multi_control_prog << U1(q[0], 2).control([q[1], q[2], q[3]])
    multi_control_prog << U2(q[0], 2, 3).control([q[1], q[2], q[3]])
    multi_control_prog << U3(q[0], 2, 3, 4).control([q[1], q[2], q[3]])
    multi_control_prog << U4(q[0], 2, 3, 4, 5)

    convert_single_gates = ["H","U3"]
    convert_double_gates = ["CZ","CNOT"]

    decomposed_prog = decompose_multiple_control_qgate(multi_control_prog, machine, convert_single_gates, convert_double_gates, False)

    print(decomposed_prog)

    prog = QProg()
    # prog << CR(q[0], q[1], 4)
    prog << RX(q[0], 1)
    prog << U1(q[0], 2)
    prog << U2(q[0], 2, 3)
    prog << U3(q[0], 2, 3, 4)
    prog << U4(q[0], 2, 3, 4, 5)

    convert_prog = transform_to_base_qgate(prog, machine, convert_single_gates, convert_double_gates)
    print(convert_prog)

    src_matrix = get_unitary(multi_control_prog);
    dst_matrix = get_unitary(decomposed_prog)

    print(src_matrix)
    print(dst_matrix)

    import numpy as np
    def are_lists_equal(list1, list2, tolerance=1e-6):
        array1 = np.array(list1, dtype=complex)
        array2 = np.array(list2, dtype=complex)
        return np.allclose(array1, array2, atol=tolerance)

    if(are_lists_equal(src_matrix, dst_matrix)):
        print("transform_to_base_qgate passed.")
    else:
        print("transform_to_base_qgate failed.")

def test_benchmark():
    """
    Execute a series of benchmark tests for quantum computing algorithms and configurations.

    The function includes tests for the Quantum Volume (qv), Entanglement Entropy Benchmark (xeb), and Random Benchmark (rb) of quantum circuits.

    Tests performed:
    - `test_qv`: Measures the Quantum Volume for specified qubit combinations using a fixed number of trials and configuration parameters.
    - `test_xeb`: Evaluates the entanglement entropy for various gate layer configurations using double-gate operations.
    - `test_rb`: Measures the random benchmark results for single and double qubit systems with varying Clifford gate sets.

    The function prints the results of each test. These tests are designed to provide insights into the performance and capabilities of quantum circuits and their simulation.

    Parameters:
    None

    Returns:
    None

    Notes:
    - The function uses QCloudTaskConfig and related functions from the pyQPanda package to set up and execute quantum tasks.
    - The configuration parameters, such as cloud token, chip ID, and shots, are set to predefined values.
    - The test results are printed to the console, and their interpretation should be done within the context of quantum computing principles.
    """
    def test_qv():
    
        #构建待测量的量子比特组合， 这里比特组合为2组，其中 量子比特3、4为一组；量子比特2，3，5为一组
        qubit_lists = [[31, 32], [0, 6]] 

        #设置随机迭代次数
        ntrials = 2
        
        # 配置量子计算任务参数
        config = QCloudTaskConfig()
        config.cloud_token=online_api_key
        config.chip_id = origin_72
        config.open_amend = False
        config.open_mapping = False
        config.open_optimization = False
        config.shots = 1000

        qv_result = calculate_quantum_volume(config, qubit_lists, ntrials)
        print("Quantum Volume : ", qv_result)

        #运行结果：
        # Quantum Volume ： 8

    def test_xeb():

        # 设置不同层数组合
        range = [2,4,6,8,10]
        # 现在可测试双门类型主要为 CZ CNOT SWAP ISWAP SQISWAP
        
        # 配置量子计算任务参数
        config = QCloudTaskConfig()
        config.cloud_token=online_api_key
        config.chip_id = origin_72
        config.open_amend = False
        config.open_mapping = False
        config.open_optimization = False
        config.shots = 1000

        res = double_gate_xeb(config, 0, 1, range, 2, GateType.CZ_GATE)
        # 对应的数值随噪声影响，噪声数值越大，所得结果越小，且层数增多，结果数值越小。
        print(res)
        # 运行结果：
        # 2: 0.9922736287117004, 4: 0.9303175806999207, 6: 0.7203856110572815, 8: 0.7342230677604675, 10: 0.7967881560325623

    def test_rb():

        # 设置随机线路中clifford门集数量
        range = [ 5,10,15 ]

        # 配置量子计算任务参数
        config = QCloudTaskConfig()
        config.cloud_token=online_api_key
        config.chip_id = origin_72
        config.open_amend = False
        config.open_mapping = False
        config.open_optimization = False
        config.shots = 1000

        # 测量单比特随机基准
        single_rb_result = single_qubit_rb(config, 0, range, 2)

        # 同样可以测量两比特随机基准
        double_rb_result = double_qubit_rb(config, 0, 1, range, 2)
        
        # 对应的数值随噪声影响，噪声数值越大，所得结果越小，且随clifford门集数量增多，结果数值越小。
        print(single_rb_result)
        print(double_rb_result)

        # 运行结果：
        # 5: 0.9996, 10: 0.9999, 15: 0.9993000000000001

    test_qv()
    # test_xeb()
    # test_rb()

def test_qcloud_qst():
    """
    Executes a quantum circuit simulation using the QCloud service to generate a state tomography density.

    The function initializes a quantum virtual machine (QVM) with a specified configuration, allocates qubits and
    classical bits, constructs a quantum circuit, and performs a tomography measurement. The results are then
    printed.

    Parameters:
        None

    Returns:
        None

    The quantum circuit performed includes a Hadamard gate, RX gates, controlled-Z (CZ) gates, and measurement
    operations. The circuit's state is analyzed using tomography, and the density matrix is returned.
    """
    PI=3.14159

    machine = QCloud()
    machine.set_configure(72,72);

    machine.init_qvm(online_api_key,True)

    q = machine.qAlloc_many(6)
    c = machine.cAlloc_many(6)

    prog = QProg()
    prog << hadamard_circuit(q)\
        << RX(q[1], PI / 4)\
        << RX(q[2], PI / 4)\
        << RX(q[1], PI / 4)\
        << CZ(q[0], q[1])\
        << CZ(q[1], q[2])\
        << Measure(q[0], c[0])\
        << Measure(q[1], c[1])

    result = machine.get_state_tomography_density(prog, 1000,real_chip_type.origin_72)
    print(result)
    machine.finalize()
    
def test_cpu():
    
    machine = FullAmplitudeQVM()
    machine.set_configure(72,72)
    # machine.init_qvm("CPU")
    machine.init_qvm("GPU")

    q = machine.qAlloc_many(4)
    c = machine.cAlloc_many(4)

    prog = QProg()
    prog << RX(q[0], 1)
    prog << U1(q[0], 2)
    prog << U2(q[0], 2, 3)
    prog << measure_all(q,c)
    
    prog1 = QProg()
    prog1 << RX(q[0], 1)
    prog1 << U1(q[0], 2)
    
    measure_result = machine.run_with_configuration(prog, 1000)
    prob_result = machine.prob_run_dict(prog,q)
    
    opt = PauliOperator({"Z0 Z3": 2, "X0 Y3": 3})
    hamiltonian = opt.to_hamiltonian(False)
    
    exp = machine.get_expectation(prog, hamiltonian,q)
    
    print(measure_result)
    print(prob_result)
    print(exp)

if __name__ == "__main__":


    test_cpu()
    # test_qcloud_query()
    # test_benchmark()
    # test_qcloud_qst()
    # test_qcloud_originir()
    # test_density()
    # test_request()
    # test_qcloud_async()
    # test_qcloud_cluster()
    # test_sparse_state_init()
    # test_hex_to_bin_result()
    # test_qcloud_pqc_encryption()
    # test_big_data_batch_result()
    # test_comm_protocol_encode_decode()
    # test_multi_control_decompose_and_convert_to_u3()