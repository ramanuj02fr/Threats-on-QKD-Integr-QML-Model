import numpy as np
from qiskit_aer import AerSimulator


def generate_bases(num_qubits):
    return np.random.randint(2, size=num_qubits)


def measure(qc, num_qubits, bases):
    for i in range(num_qubits):
        if bases[i] == 1:
            qc.h(i)
        qc.measure(i, i)

    simulator = AerSimulator()
    result = simulator.run(qc, shots=1).result()
    counts = result.get_counts()

    result_string = list(counts.keys())[0]
    bob_result = result_string[::-1]

    return [int(b) for b in bob_result]
