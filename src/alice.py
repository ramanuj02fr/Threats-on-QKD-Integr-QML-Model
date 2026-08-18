import numpy as np
from qiskit import QuantumCircuit


def generate_bits(num_qubits):
    return np.random.randint(2, size=num_qubits)

def generate_bases(num_qubits):
    return np.random.randint(2, size=num_qubits)

def encode(num_qubits, bits, bases):
    qc = QuantumCircuit(num_qubits, num_qubits)

    for i in range(num_qubits):
        if bits[i] == 1:
            qc.x(i)
    for i in range(num_qubits):
        if bases[i] == 1:
            qc.h(i)    

    return qc
