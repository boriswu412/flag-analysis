"""
Simple test: Create a protocol with conditions
"""

from sequential_protocol import build_sequential_protocol
from protocol import Protocol

# Example: 4 circuits, 4 rounds, check if syndromes converge
protocol = Protocol()
build_sequential_protocol(
    circuits=[ "XZZXI_f", "IXZZX_f", "XIXZZ_f" ,"ZXIXZ_f"],
    num_rounds=5,
    between_round_conditions={
        1: "s0 == s1",  # After round 1, check if s0 equals s1
        2: "s1 != s2",   # After round 2, check if s1 equals s2
        3: "s2 == s3",
        4: "s3 != s4"
    },
    decode_method="raw_syndrome",
    rounds_for_lut=None,
    save_path="./protocols/test_protocol.json",
    protocol=protocol
)

print("✓ Protocol created!")
print(f"\nStart node: {protocol.start_node}")
print(f"\nAll nodes ({len(protocol.nodes)}):")
for node_id in sorted(protocol.nodes.keys()):
    node = protocol.nodes[node_id]
    print(f"\n  {node_id}:")
    if node.instructions:
        print(f"    Instructions: {node.instructions}")
    if node.branches:
        print(f"    Branches ({len(node.branches)}):")
        for i, branch in enumerate(node.branches):
            target = branch.target if branch.target else "TERMINAL"
            cond = branch.condition.to_dict() if branch.condition else "unconditional"
            print(f"      {i+1}. → {target}")
            if branch.condition:
                print(f"         if: {cond}")

print("\n" + "="*60)
print("How it works:")
print("="*60)
print("""
Round 0: Run circuits stab_X, stab_Z, stab_Y
  r0_c0 → r0_c1 → r0_c2 → check_after_r0

check_after_r0: Check "s0 == s1"
  If TRUE → converged (done!)
  If FALSE → r1_c0 (continue)

Round 1: Run circuits again
  r1_c0 → r1_c1 → r1_c2 → check_after_r1

check_after_r1: Check "s1 == s2"
  If TRUE → converged (done!)
  If FALSE → r2_c0 (continue)

Round 2: Final round
  r2_c0 → r2_c1 → r2_c2 → complete

complete: Run raw syndrome circuit and decode with LUT
""")

print("\n✓ Saved to: ./protocols/test_protocol.json")
