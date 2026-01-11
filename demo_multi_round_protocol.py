"""
Demo: Multi-round circuit protocol with bit-by-bit syndrome measurement

This demonstrates the new protocol where:
- Each circuit produces ONE syndrome bit
- Multiple circuits run sequentially in each round
- If a flag is raised, the round stops immediately (incomplete syndrome)
- Syndromes can be compared between rounds for convergence detection
"""

from protocol import (
    build_multi_round_circuit_protocol,
    build_custom_circuit_protocol,
    visualize_protocol_from_file
)

# Example 1: Simple multi-round protocol
print("="*60)
print("Example 1: Simple Multi-Round Protocol")
print("="*60)

# Create a protocol with 3 circuits per round, running 3 rounds
protocol1 = build_multi_round_circuit_protocol(
    circuit_names=["stab_X_1", "stab_X_2", "stab_Z"],
    num_rounds=3,
    save_path="./protocols/demo_multi_round.json"
)

print("\n✓ Protocol saved to: ./protocols/demo_multi_round.json")
print("\nProtocol structure:")
print(f"  - Start node: {protocol1.start_node}")
print(f"  - Total nodes: {len(protocol1.nodes)}")
print(f"  - Nodes: {list(protocol1.nodes.keys())[:5]}...")

# Example 2: Custom circuit protocol with advanced features
print("\n" + "="*60)
print("Example 2: Custom Protocol with Convergence Detection")
print("="*60)

config = {
    "circuits": ["circuit_A", "circuit_B", "circuit_C", "circuit_D"],
    "rounds": 4,
    "stop_on_flag": True,
    "check_syndrome_convergence": True
}

protocol2 = build_custom_circuit_protocol(
    circuit_config=config,
    save_path="./protocols/demo_custom_protocol.json"
)

print("\n✓ Protocol saved to: ./protocols/demo_custom_protocol.json")
print("\nProtocol features:")
print(f"  - Circuits per round: {len(config['circuits'])}")
print(f"  - Number of rounds: {config['rounds']}")
print(f"  - Stop on flag: {config['stop_on_flag']}")
print(f"  - Syndrome convergence check: {config['check_syndrome_convergence']}")

# Example 3: Protocol without stopping on flags
print("\n" + "="*60)
print("Example 3: Protocol that Doesn't Stop on Flags")
print("="*60)

config3 = {
    "circuits": ["measure_1", "measure_2"],
    "rounds": 2,
    "stop_on_flag": False,  # Run all circuits even if flag raised
    "check_syndrome_convergence": False
}

protocol3 = build_custom_circuit_protocol(
    circuit_config=config3,
    save_path="./protocols/demo_no_stop_on_flag.json"
)

print("\n✓ Protocol saved to: ./protocols/demo_no_stop_on_flag.json")
print("  This protocol completes all circuits regardless of flags")

# Visualize one of the protocols
print("\n" + "="*60)
print("Generating Visualization")
print("="*60)

try:
    visualize_protocol_from_file(
        "./protocols/demo_multi_round.json",
        outname="demo_multi_round_graph"
    )
    print("\n✓ Visualization saved to: demo_multi_round_graph.svg")
except Exception as e:
    print(f"\n✗ Visualization failed: {e}")
    print("  (Install graphviz: pip install graphviz)")

print("\n" + "="*60)
print("Protocol Execution Flow Example")
print("="*60)

print("""
Example execution with 3 circuits ["C1", "C2", "C3"] and 2 rounds:

Round 0:
  1. Run C1 → get bit s_0[0], check flag f_0[0]
     - If f_0[0] = True → STOP, incomplete syndrome
     - If f_0[0] = False → continue
  2. Run C2 → get bit s_0[1], check flag f_0[1]
     - If f_0[1] = True → STOP, incomplete syndrome
     - If f_0[1] = False → continue
  3. Run C3 → get bit s_0[2], check flag f_0[2]
     - Now we have complete syndrome s_0 = [bit0, bit1, bit2]
     - If f_0[2] = True → flag detected with complete syndrome
     - If f_0[2] = False → proceed to Round 1

Round 1:
  [Same process...]
  - After completion, compare s_1 vs s_0
  - If s_1 == s_0 and no flags → converged!
  - Otherwise continue to next round
""")

print("\n" + "="*60)
print("To use in your quantum error correction:")
print("="*60)
print("""
from protocol import load_protocol, all_paths_with_conditions_and_instructions

# Load the protocol
start_node, protocol_dict = load_protocol("./protocols/demo_multi_round.json")

# Get all possible execution paths
for branch_steps, conditions, instructions in all_paths_with_conditions_and_instructions(protocol_dict):
    print("Path:", [step[0] for step in branch_steps])
    print("Conditions:", conditions)
    print("Instructions:", instructions)
""")

print("\nDone! 🎉")
