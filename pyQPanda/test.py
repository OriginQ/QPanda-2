from pyqpanda import *

if __name__ == "__main__":

    QCM = QCloud()
    QCM.init_qvm("A0D08FA558CE45B2AF4F1DC122CF2589", True)

    QCM.set_qcloud_api("http://www.72bit.com")

    q = QCM.qAlloc_many(4)
    c = QCM.cAlloc_many(4)

    measure_prog_array = []
    for i in range(3):
        measure_prog_array.append(QProg())

    measure_prog_array[0] << H(q[0]) << H(q[1]) << measure_all(q, c)
    measure_prog_array[1] << H(q[0]) << CNOT(q[0], q[1]) << measure_all(q, c)
    measure_prog_array[2] << H(q[0]) << H(
        q[1]) << CNOT(q[0], q[1]) << measure_all(q, c)

    result = QCM.full_amplitude_measure_batch(measure_prog_array, 1000)
    print(result)

    QCM.finalize()
