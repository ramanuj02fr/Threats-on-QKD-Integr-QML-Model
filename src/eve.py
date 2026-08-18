import numpy as np


def generate_bases(num_qubits):
    return np.random.randint(2, size=num_qubits)


def intercept(qc, num_qubits, bases):
    for i in range(num_qubits):
        if bases[i] == 1:
            qc.h(i)
        qc.measure(i, i)
        