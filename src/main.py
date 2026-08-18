from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

import alice
import bob
import eve
import qber

num_qubits = 32

alice_bits = alice.generate_bits(num_qubits)
alice_bases = alice.generate_bases(num_qubits)

bob_bases_1 = bob.generate_bases(num_qubits)

qc1 = alice.encode(num_qubits, alice_bits, alice_bases)
bob_result_1 = bob.measure(qc1, num_qubits, bob_bases_1)

sifted_a1, sifted_b1, qber1 = qber.compute_qber(
    alice_bits, bob_result_1, alice_bases, bob_bases_1
)

print("---- No Eve Case ----")
print("Sifted key length:", len(sifted_a1))
print("QBER:", qber1 * 100, "%")

eve_bases = eve.generate_bases(num_qubits)
bob_bases_2 = bob.generate_bases(num_qubits)

qc2 = alice.encode(num_qubits, alice_bits, alice_bases)
eve.intercept(qc2, num_qubits, eve_bases)

simulator = AerSimulator()
result = simulator.run(qc2, shots=1).result()
counts = result.get_counts()
result_string = list(counts.keys())[0]
eve_result = [int(b) for b in result_string[::-1]]

qc3 = alice.encode(num_qubits, eve_result, eve_bases)
bob_result_2 = bob.measure(qc3, num_qubits, bob_bases_2)

sifted_a2, sifted_b2, qber2 = qber.compute_qber(
    alice_bits, bob_result_2, alice_bases, bob_bases_2
)

print("\n---- With Eve Case ----")
print("Sifted key length:", len(sifted_a2))
print("QBER:", qber2 * 100, "%")
