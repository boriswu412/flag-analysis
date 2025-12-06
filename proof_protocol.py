from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
from pathlib  import Path

from circuit_op import *
from protocol import *

from qiskit import QuantumCircuit


from z3 import BoolVal, Xor, Bool,simplify,substitute, And, Not,Or, PbLe, AtMost,ForAll, Implies, Exists, PbGe, AtLeast

from z3 import Solver, unsat, sat

from flag_analysis import *
from circuit_op import *
from protocol import *

from copy import deepcopy


from dataclasses import dataclass
from typing import List, Dict, Any

@dataclass
class PathRecord:
    nodes: List[str]                   # node IDs from root → leaf
    conditions: List[Dict]             # branch conditions taken
    instr_steps: List[Dict[str, Any]]  # [{"instr": name}, ...]
    state: Dict[str, List[Any]]        # final symbolic state {"data":..., "ancX":..., ...}


from copy import deepcopy

from typing import List, Dict, Any

def collect_paths_with_states(protocol,
                              start_node: str,
                              init_state,
                              config: Dict) -> List[List[List[Any]]]:
    """
    Returns:
        paths: List of paths.
        Each path is a list of steps:
            [round, input_id, output_id, instruction, condition_dict, state]

        - round: depth along that path (0,1,2,...)
        - input_id: current node id
        - output_id: branch target node id (or None at leaf)
        - instruction: instruction name at input_id (or None)
        - condition_dict: branch condition dict (or None)
        - state: CircuitXZ state *after* executing the instruction at input_id
                 (or the incoming state if there is no instruction)
    """
    all_paths: List[List[List[Any]]] = []

    def dfs(round_idx: int,
            node_id: str,
            cur_state,
            cur_path: List[List[Any]]):
        node = protocol[node_id]

        # --- execute instruction at this node (if any) ---
        instr = node.instructions[0] if node.instructions else None
        state_after = cur_state

        if instr is not None:
            # lookup QASM file in config
            if instr not in config:
                raise KeyError(f"Instruction '{instr}' not found in config")
            qasm_path = config[instr]

            # you already have these helpers
            qc = load_qasm(qasm_path)
            gate_list = get_gate_only_indices(qc)
            groups = detect_qubit_groups(qc)
            
            res = symbolic_execution_of_state(
                qasm_path,
                cur_state,
                round_idx,
                fault_gate=gate_list,
                track_steps=False,   # so here: (state, sites_info)
            )
            state_after = res[0]
            
        # --- leaf node: no branches ---
        if not node.branches:
            step = [round_idx, node_id, None, instr, None, state_after]
            all_paths.append(cur_path + [step])
            return

        # --- internal node: follow each branch ---
        for br in node.branches:
            cond_dict = br.condition.to_dict() if br.condition is not None else None
            
            print("current path:", cur_path)
            full_state = [ (state[5], )  for state in  cur_path  if cur_path != [] ] +[state_after]

            z3_condition = condition_to_z3(cond_dict, full_state,groups)
            step = [round_idx, node_id, br.target, instr,z3_condition, state_after]
            dfs(
                round_idx + 1,
                br.target,
                state_after,
                cur_path + [step],
            )
    
    dfs(0, start_node, init_state, [])
    return all_paths



from z3 import BoolVal, And, Or, Not

# -----------------------------
# Parse "s_1", "f_3", etc.
# -----------------------------

def parse_var(name: str):
    """
    Convert 's_1' -> ('s', 1)
            'f_3' -> ('f', 3)
    """
    if "_" not in name:
        raise ValueError(f"Bad variable format (expected like 's_1'): {name}")
    group, idx_str = name.split("_", 1)
    return group, int(idx_str)


def read_state_variable(q_type: str, index: int, state: dict, groups : Dict):
    """
    Map (group, index) into your protocol state structure.

    Expect state like:
        state["syn"]  : List[BoolRef]   # syndrome ancilla bits
        state["flag"] : List[BoolRef]   # flag qubit bits

    's_i' -> state["syn"][i]
    'f_i' -> state["flag"][i]
    """

    state =  state_to_raw_expr_dict(state, groups)
    print("Reading state variable:", q_type, index, "from state:", state)
    
    if q_type == "s":
        syn = [ q.z for q in state["ancX"]] + [q.z for q in state["ancZ"]]
        if index < 0 or index >= len(syn):
            raise IndexError(f"s_{index} out of range (len syn = {len(syn)})")
        return syn[index]

    if q_type == "f":
       
        flags = [ q.z for q in state["flagX"]] + [q.z for q in state["flagZ"]]
        if index < 0 or index >= len(flags):
            raise IndexError(f"f_{index} out of range (len flag = {len(flags)})")
        return flags[index]

    raise ValueError(f"Unknown variable group in condition: {q_type!r}")


def read_operand(x, full_state: dict , groups:Dict):
    """
    Turn an operand from the condition dict into a z3 Bool expression.

    Supports:
      - 0, 1              → False / True
      - 's_i', 'f_j'      → read from state via (group, index)
    """
    # numeric constants
    if x == 0:
        return False
    if x == 1:
        return True

    if isinstance(x, str):
        q_type, idx = parse_var(x)

        print("from full_state:", full_state)

        if q_type == 's' :
            return [ q.z for q in state_to_raw_expr_dict(full_state[idx-1],groups)["ancX"]] + [q.z for q in state_to_raw_expr_dict(full_state[idx-1], groups)["ancZ"]]
         
        elif q_type == 'f' :   
            return [ q.z for q in state_to_raw_expr_dict(full_state[idx-1], groups)["flagX"]] + [q.z for q in state_to_raw_expr_dict(full_state[idx-1], groups)["flagZ"]]

    raise ValueError(f"Unsupported operand in condition: {x!r}")


# -----------------------------------
# Main translator: condition → z3 expr
# -----------------------------------

def condition_to_z3(cond: dict | None, full_state: dict, groups:Dict) -> Bool:
    """
    Convert a protocol condition dict into a z3 BoolRef
    using the given `state`, where:

        state["syn"]  : List[BoolRef]  # syndromes (ancilla-based)
        state["flag"] : List[BoolRef]  # flags

    cond JSON forms:
      - {"type":"and", "operands":[cond1, cond2, ...]}
      - {"type":"or",  "operands":[cond1, cond2, ...]}
      - {"type":"not", "operands":[cond1]}
      - {"type":"equal",     "left":..., "right":...}
      - {"type":"not_equal", "left":..., "right":...}
    """
    if cond is None:
        return BoolVal(True)  # no condition → always true

    print("Translating condition:", cond)
    t = cond["type"]

    if t == "and":
        return And(*(condition_to_z3(c, full_state,  groups) for c in cond["operands"]))

    if t == "or":
        return Or(*(condition_to_z3(c,full_state , groups) for c in cond["operands"]))

    if t == "not":
        sub = cond["operands"][0]  # your JSON uses 'operands' even for NOT
        return Not(condition_to_z3(sub, full_state , groups))

    if t in ("equal"):
        L = read_operand(cond["left"],  full_state , groups)
        R = read_operand(cond["right"], full_state , groups)
        print("Equal operands:", len(L),  R)
        if  isinstance(R, bool):
            return And(*[L[i] == Bool(False) for i in range(len(L))])
        else :
            return And(*[L[i] == R[i] for i in range(len(L))])  

    raise ValueError(f"Unknown condition type: {t}")
