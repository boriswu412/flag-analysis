"""
Example: Using the flexible protocol builder

Shows how to create protocols with custom conditions between rounds.
"""

from protocol import build_flexible_protocol, visualize_protocol_from_file

# Example 1: Simple protocol with syndrome comparison
print("Example 1: Check if syndromes converge")
print("-" * 50)

protocol1 = build_flexible_protocol(
    circuits=["stab_X", "stab_Z"],
    num_rounds=3,
    between_round_conditions={
        1: "s0 == s1",  # After round 1, check if s0 equals s1
        2: "s1 == s2"   # After round 2, check if s1 equals s2
    },
    save_path="./protocols/example1_syndrome_convergence.json"
)

print("✓ Created protocol with syndrome convergence checks")
print(f"  Nodes: {list(protocol1.nodes.keys())}")

# Example 2: Protocol with flag checking
print("\n\nExample 2: Check if syndrome flagged")
print("-" * 50)

protocol2 = build_flexible_protocol(
    circuits=["circuit_A", "circuit_B", "circuit_C"],
    num_rounds=2,
    between_round_conditions={
        1: "s1.flag"  # Check if syndrome 1 has flag
    },
    save_path="./protocols/example2_flag_check.json"
)

print("✓ Created protocol with flag checking")
print(f"  Nodes: {list(protocol2.nodes.keys())}")

# Example 3: Protocol with not equal
print("\n\nExample 3: Check if syndromes different")
print("-" * 50)

protocol3 = build_flexible_protocol(
    circuits=["measure_1", "measure_2"],
    num_rounds=3,
    between_round_conditions={
        1: "s0 != s1",  # Continue if syndromes are different
        2: "s1 == s2"
    },
    save_path="./protocols/example3_not_equal.json"
)

print("✓ Created protocol with != condition")
print(f"  Nodes: {list(protocol3.nodes.keys())}")

# Example 4: No conditions - just run all rounds
print("\n\nExample 4: Simple sequential execution")
print("-" * 50)

protocol4 = build_flexible_protocol(
    circuits=["circ1", "circ2", "circ3"],
    num_rounds=2,
    between_round_conditions=None,  # No conditions
    decode_method="raw_syndrome",
    save_path="./protocols/example4_sequential.json"
)

print("✓ Created simple sequential protocol")
print(f"  Nodes: {list(protocol4.nodes.keys())}")

# Example 5: Use raw syndrome circuit at end
print("\n\nExample 5: Decode with raw syndrome circuit")
print("-" * 50)

protocol5 = build_flexible_protocol(
    circuits=["stab_X", "stab_Z"],
    num_rounds=3,
    between_round_conditions={1: "s0 == s1"},
    decode_method="raw_syndrome",  # Run raw syndrome circuit
    save_path="./protocols/example5_raw_syndrome.json"
)

print("✓ Protocol will run raw syndrome circuit at end")
print("  Instructions at 'complete' node:")
for instr in protocol5.nodes["complete"].instructions:
    print(f"    - {instr}")

# Example 6: Use multiple rounds for LUT
print("\n\nExample 6: Decode with multi-round LUT")
print("-" * 50)

protocol6 = build_flexible_protocol(
    circuits=["measure_1", "measure_2"],
    num_rounds=4,
    between_round_conditions={2: "s1 == s2"},
    decode_method="multi_round_lut",  # Use syndrome from multiple rounds
    rounds_for_lut=[1, 2, 3],  # Use last 3 rounds
    save_path="./protocols/example6_multi_round_lut.json"
)

print("✓ Protocol will use syndromes from rounds 1, 2, 3")
print("  Instructions at 'complete' node:")
for instr in protocol6.nodes["complete"].instructions:
    print(f"    - {instr}")

# Example 7: Use all rounds for LUT (default)
print("\n\nExample 7: Decode with all rounds")
print("-" * 50)

protocol7 = build_flexible_protocol(
    circuits=["circ_A", "circ_B"],
    num_rounds=3,
    decode_method="multi_round_lut",  # rounds_for_lut=None means use all
    save_path="./protocols/example7_all_rounds_lut.json"
)

print("✓ Protocol will use syndromes from all rounds (default)")
print("  Instructions at 'complete' node:")
for instr in protocol7.nodes["complete"].instructions:
    print(f"    - {instr}")

# Show protocol structure
print("\n\nProtocol Execution Flow:")
print("=" * 50)
print("""
Round 0:
  r0_c0 (run circuit_A)
    ├─ flag raised → r1_c0 (skip to next round)
    └─ no flag → r0_c1
  r0_c1 (run circuit_B)
    ├─ flag raised → r1_c0
    └─ no flag → r0_c2
  r0_c2 (run circuit_C, last circuit)
    └─ → check_after_r0 (condition check)

check_after_r0:
  Check condition "s0 == s1"
    ├─ TRUE → converged (early exit)
    └─ FALSE → r1_c0 (continue to round 1)

Round 1:
  [same structure...]
""")

print("\n✓ All examples created successfully!")
print("\nSupported conditions:")
print("  - s1 == s2  (syndromes equal)")
print("  - s1 != s2  (syndromes not equal)")
print("  - s1.flag   (syndrome 1 flagged)")
print("\nDecode methods:")
print("  - raw_syndrome: Run raw syndrome circuit and use LUT")
print("  - multi_round_lut: Use syndromes from multiple rounds with LUT_s_i_s_j")

