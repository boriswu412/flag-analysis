"""
Sequential Protocol Builder
Builds protocols without flag branching - assumes clean execution with only syndrome comparisons.
"""

from typing import List, Dict, Optional
from protocol import Protocol, Node, Branch, Condition, parse_condition_string


def build_sequential_protocol(circuits: List[str], num_rounds: int,
                              between_round_conditions: Optional[Dict[int, str]],
                              decode_method: str, rounds_for_lut: Optional[List[int]],
                              save_path: Optional[str], protocol: Protocol) -> Protocol:
    """
    Build a sequential protocol without flag branching.
    Circuits run sequentially, only syndrome comparisons cause branching.
    
    Args:
        circuits: List of circuit names to run
        num_rounds: Number of rounds to execute
        between_round_conditions: Dict mapping round index to condition string (e.g., {1: "s0 == s1"})
        decode_method: Decoding method (e.g., "raw_syndrome")
        rounds_for_lut: Optional list of rounds for LUT (not used in this implementation)
        save_path: Optional path to save the protocol JSON
        protocol: Protocol object to build into
        
    Returns:
        Protocol object with sequential execution graph
        
    Condition strings:
        - "s0 == s1": All syndrome bits from round 0 match round 1
        - "s1 != s2": Not all syndrome bits from round 1 match round 2
        
    The function expands round-level comparisons into bit-wise comparisons:
        s0 == s1 → (s_0 == s_4) AND (s_1 == s_5) AND (s_2 == s_6) AND (s_3 == s_7)
        (assuming 4 circuits per round)
    """
    num_circuits = len(circuits)
    
    # Create all nodes first
    circuit_nodes = []  # List of (node_id, path_index, round, circuit)
    path_index = 0
    
    for r in range(num_rounds):
        for c in range(num_circuits):
            if r == 0 and c == 0:
                node_id = "r_0_c_0#0"
            else:
                node_id = f"r_{r}_c_{c}#_{path_index}"
            circuit_nodes.append((node_id, path_index, r, c))
            path_index += 1
    
    # Track the next available index for terminal nodes
    next_terminal_idx = path_index
    
    # Now create nodes with proper connections
    for i, (node_id, idx, r, c) in enumerate(circuit_nodes):
        instructions = [circuits[c]]
        
        # Flag condition: f_{idx} == False (we assume no flags raised)
        flag_false = Condition(cond_type="equal", left=f"f_{idx}", right=False)
        
        is_last_circuit = (c == num_circuits - 1)
        is_last_round = (r == num_rounds - 1)
        
        if is_last_circuit and not is_last_round:
            # Check for condition after this round
            if between_round_conditions and r in between_round_conditions:
                # Next circuit node (first of next round)
                next_node_id = circuit_nodes[i + 1][0] if i + 1 < len(circuit_nodes) else None
                
                # Parse syndrome comparison condition like "s0 == s1" or "s0 != s1" to compare all bits
                condition_str = between_round_conditions[r]
                
                # Check for == or !=
                is_not_equal = "!=" in condition_str
                is_equal = "==" in condition_str
                
                if is_equal or is_not_equal:
                    parts = condition_str.split("!=" if is_not_equal else "==")
                    left = parts[0].strip()
                    right = parts[1].strip()
                    
                    # Only expand if it's a round comparison (sX == sY or sX != sY)
                    if left.startswith('s') and left[1:].isdigit() and right.startswith('s') and right[1:].isdigit():
                        left_round = int(left[1:])
                        right_round = int(right[1:])
                        
                        # Build AND condition for all syndrome bits
                        left_start = left_round * num_circuits
                        right_start = right_round * num_circuits
                        
                        bit_conditions = []
                        for bit in range(num_circuits):
                            bit_cond = Condition(
                                cond_type="equal",
                                left=f"s_{left_start + bit}",
                                right=f"s_{right_start + bit}"
                            )
                            bit_conditions.append(bit_cond)
                        
                        # Create AND of all bit comparisons
                        if len(bit_conditions) == 1:
                            all_bits_match = bit_conditions[0]
                        else:
                            all_bits_match = Condition(cond_type="and", operands=bit_conditions)
                        
                        # If !=, negate the condition
                        syndrome_match = Condition(cond_type="not", operand=all_bits_match) if is_not_equal else all_bits_match
                    else:
                        # Not a round comparison, use fallback
                        syndrome_match = parse_condition_string(condition_str)
                else:
                    # Fallback to simple parsing
                    syndrome_match = parse_condition_string(condition_str)
                
                # Combine flag condition AND syndrome match condition
                combined_condition = Condition(
                    cond_type="and",
                    operands=[flag_false, syndrome_match]
                )
                
                # Continue to next round when condition is met
                branches = [
                    Branch(target=next_node_id, condition=combined_condition)
                ]
                protocol.add_node(Node(node_id=node_id, instructions=instructions, branches=branches))
            else:
                # No condition, go directly to next round
                next_node_id = circuit_nodes[i + 1][0] if i + 1 < len(circuit_nodes) else None
                branches = [Branch(target=next_node_id, condition=flag_false)]
                protocol.add_node(Node(node_id=node_id, instructions=instructions, branches=branches))
                
        elif is_last_circuit and is_last_round:
            # Last circuit of last round
            terminal_node_id = f"terminal#_{next_terminal_idx}"
            lut_node_id = f"lut#_{next_terminal_idx + 1}"
            
            # Check if there's a condition for this round
            if between_round_conditions and r in between_round_conditions:
                # Parse syndrome comparison condition
                condition_str = between_round_conditions[r]
                
                # Check for == or !=
                is_not_equal = "!=" in condition_str
                is_equal = "==" in condition_str
                
                if is_equal or is_not_equal:
                    parts = condition_str.split("!=" if is_not_equal else "==")
                    left = parts[0].strip()
                    right = parts[1].strip()
                    
                    # Only expand if it's a round comparison (sX == sY or sX != sY)
                    if left.startswith('s') and left[1:].isdigit() and right.startswith('s') and right[1:].isdigit():
                        left_round = int(left[1:])
                        right_round = int(right[1:])
                    
                        left_start = left_round * num_circuits
                        right_start = right_round * num_circuits
                    
                        bit_conditions = []
                        for bit in range(num_circuits):
                            bit_cond = Condition(
                                cond_type="equal",
                                left=f"s_{left_start + bit}",
                                right=f"s_{right_start + bit}"
                            )
                            bit_conditions.append(bit_cond)
                    
                        # Create AND of all bit comparisons
                        if len(bit_conditions) == 1:
                            all_bits_match = bit_conditions[0]
                        else:
                            all_bits_match = Condition(cond_type="and", operands=bit_conditions)
                        
                        # If !=, negate the condition
                        syndrome_match = Condition(cond_type="not", operand=all_bits_match) if is_not_equal else all_bits_match
                    else:
                        # Not a round comparison, use fallback
                        syndrome_match = parse_condition_string(condition_str)
                else:
                    syndrome_match = parse_condition_string(condition_str)
                    
                # Combine flag condition AND syndrome match condition
                combined_condition = Condition(
                    cond_type="and",
                    operands=[flag_false, syndrome_match]
                )
                
                branches = [Branch(target=terminal_node_id, condition=combined_condition)]
                protocol.add_node(Node(node_id=node_id, instructions=instructions, branches=branches))
            else:
                # No condition, just check flag
                branches = [Branch(target=terminal_node_id, condition=flag_false)]
                protocol.add_node(Node(node_id=node_id, instructions=instructions, branches=branches))
            
            # Terminal nodes
            if decode_method == "raw_syndrome":
                # The terminal node index corresponds to the syndrome it produces
                terminal_syndrome_idx = next_terminal_idx
                # LUT uses the terminal syndrome index (the last syndrome including terminal)
                protocol.add_node(Node(
                    node_id=terminal_node_id,
                    instructions=["raw_syndrome"],
                    branches=[Branch(target=lut_node_id, condition=None)]
                ))
                protocol.add_node(Node(
                    node_id=lut_node_id,
                    instructions=[f"LUT_s_{terminal_syndrome_idx}"],
                    branches=[]
                ))
        else:
            # Not last circuit - go to next circuit
            next_node_id = circuit_nodes[i + 1][0] if i + 1 < len(circuit_nodes) else None
            branches = [Branch(target=next_node_id, condition=flag_false)]
            protocol.add_node(Node(node_id=node_id, instructions=instructions, branches=branches))
    
    if save_path:
        protocol.save_to_file(save_path)
    
    return protocol
