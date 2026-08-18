def compute_qber(alice_bits, bob_bits, alice_bases, bob_bases):
    sifted_alice = []
    sifted_bob = []

    for i in range(len(alice_bits)):
        if alice_bases[i] == bob_bases[i]:
            sifted_alice.append(alice_bits[i])
            sifted_bob.append(bob_bits[i])

    errors = sum(1 for a, b in zip(sifted_alice, sifted_bob) if a != b)

    if len(sifted_alice) == 0:
        return sifted_alice, sifted_bob, 0.0

    qber = errors / len(sifted_alice)
    return sifted_alice, sifted_bob, qber
