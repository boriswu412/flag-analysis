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



from typing import List, Dict
from typing import Dict, List

from typing import Dict, List

from typing import Dict, List

def proof_protocol(protocol,
                  start_node: str,
                  init_state,
                  config: Dict,
                  t: int,
                  stab_txt_path: str):

    all_paths = []

    def dfs(round_idx: int,
            node_id: str,
            cur_state,          # CircuitXZ
            cur_groups,         # dict or None
            cur_path: List[Dict]):

        node = protocol[node_id]

        # -------------------------------
        # Execute instruction (if exists)
        # -------------------------------
        instr = node.instructions[0] if node.instructions else None
        state_after = cur_state
        site_info = []
        groups = cur_groups  # default: carry from parent

        if instr is not None and node.branches:
            # This is a circuit instruction (e.g. flag_syndrome, raw_syndrome)
            if instr not in config:
                raise KeyError(f"Instruction '{instr}' not found in config")

            qasm_path = config[instr]
            qc = load_qasm(qasm_path)
            gate_list = get_gate_only_indices(qc)
            groups = detect_qubit_groups(qc)   # new groups for this circuit

            print("round:", round_idx, "node:", node_id, "instr:", instr)
            state_after, site_info = symbolic_execution_of_state(
                qasm_path,
                cur_state,
                round_idx,
                fault_gate=gate_list,
                track_steps=False
            )

        # -------------------------------
        # Build dict-view of state_after
        # -------------------------------
        if groups is not None:
            state_dict = state_to_raw_expr_dict(state_after, groups)
        else:
            # no anc/flag structure yet; keep only data as a list
            state_dict = {
                "data": list(state_after.qubits),
                "ancX": [],
                "ancZ": [],
                "flagX": [],
                "flagZ": [],
            }

        # -------------------------------
        # Leaf node (no branches)
        # -------------------------------
        if not node.branches:
            step = {
                "round": round_idx,
                "node": node_id,
                "next": None,
                "instruction": instr,
                "condition": None,
                "state": state_dict,   # always dict
                "site_info": site_info
            }
            full_path = cur_path + [step]
            
            all_paths.append(full_path)
            print("len full_path:", len(all_paths))
            # Leaf behavior
            if instr == 'Break':
                return
            elif instr and instr.startswith("LUT_"):  # FIX: Added null check
                print("LUT", instr)
                all_condition = [
                    s["condition"] for s in full_path
                    if s["condition"] is not None
                ]
                gen_syn = parse_lut_instr(instr)
              
                

                
                proof_path(full_path, t, gen_syn, all_condition, stab_txt_path)
                
                return
            else:
                # leaf with some other instruction, but nothing to prove
                return

        # -------------------------------
        # Branching
        # -------------------------------
        for br in node.branches:
            cond_dict = br.condition.to_dict() if br.condition is not None else None

            # all states in full_state are dicts now
            full_state = [s["state"] for s in cur_path] + [state_dict]
            
            z3_condition = condition_to_z3(cond_dict, full_state, groups)

            step = {
                "round": round_idx,
                "node": node_id,
                "next": br.target,
                "instruction": instr,
                "condition": z3_condition,
                "state": state_dict,       # dict
                "site_info": site_info
            }

            dfs(
                round_idx + 1,
                br.target,
                state_after,   # still CircuitXZ
                groups,        # carry the same groups forward
                cur_path + [step]
            )

    # initial call: no groups yet, clean data state
    dfs(0, start_node, init_state, None, [])
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


from z3 import BoolVal, BoolRef

from z3 import BoolVal, BoolRef

from z3 import BoolRef, BoolVal

from z3 import BoolRef, BoolVal

def read_operand(x, full_state, groups):
    """
    Interpret operands:
      - int / bool
      - 's_k' (syndrome bit from round k)
      - 'f_k' (flag bit from round k)
    """
    # ints / bools
    if isinstance(x, int) or isinstance(x, bool):
        return x

    # already Z3
    if isinstance(x, BoolRef):
        return x

    # strings like 's_0', 'f_1', ...
    if isinstance(x, str):
        parts = x.split("_")
        if len(parts) != 2:
            raise ValueError(f"Bad variable name in condition: {x!r}")
        kind, idx_str = parts
        round_idx = int(idx_str)  # This is the round index, not array index

        if not full_state:
            raise ValueError("full_state is empty in read_operand")

        # Check if we have enough rounds
        if round_idx >= len(full_state):
            raise IndexError(f"{kind}_{round_idx} refers to round {round_idx}, but we only have {len(full_state)} rounds")

        state_dict = full_state[round_idx]   # Get state from the specific round
        ancX_list = state_dict.get("ancX", [])
        ancZ_list = state_dict.get("ancZ", [])
        flagX_list = state_dict.get("flagX", [])
        flagZ_list = state_dict.get("flagZ", [])

        if kind == "s":
            # ancX.z followed by ancZ.x - take all syndrome bits from this round
            syn_bits = [q.z for q in ancX_list] + [q.x for q in ancZ_list]
            if not syn_bits:
                raise IndexError(f"s_{round_idx} refers to round {round_idx}, but no syndrome bits found in that round")
            
            # Return the OR of all syndrome bits (this represents "any syndrome fired")
            
            return syn_bits

        elif kind == "f":
            # flagX.z followed by flagZ.x - take all flag bits from this round  
            flag_bits = [q.z for q in flagX_list] + [q.x for q in flagZ_list]
            if not flag_bits:
                raise IndexError(f"f_{round_idx} refers to round {round_idx}, but no flag bits found in that round")
            
            # Return the OR of all flag bits (this represents "any flag fired")
            
            return flag_bits

        else:
            raise ValueError(f"Unknown condition variable kind: {kind!r}")

    raise ValueError(f"Unsupported operand in condition: {x!r}")

def parse_lut_instr(name: str):
    """
    Parse something like 'LUT_s_0_f_0_s_1' into:
      [('s', 0), ('f', 0), ('s', 1)]
    """
    if not name.startswith("LUT_"):
        raise ValueError(f"Not a LUT instruction: {name}")

    tokens = name[4:].split("_")  # drop 'LUT_' and split
    if len(tokens) % 2 != 0:
        raise ValueError(f"Bad LUT format: {name}")

    pairs = []
    print("pairs:", pairs)
    for kind, idx_str in zip(tokens[0::2], tokens[1::2]):
        if kind not in ("s", "f"):
            raise ValueError(f"Unknown LUT kind '{kind}' in {name}")
        pairs.append((kind, int(idx_str)))
    return pairs
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

    t = cond["type"]

    if t == "and":
        return And(*(condition_to_z3(c, full_state,  groups) for c in cond["operands"]))

    if t == "or":
        return Or(*(condition_to_z3(c,full_state , groups) for c in cond["operands"]))

    if t == "not":
        sub = cond["operands"][0]  # your JSON uses 'operands' even for NOT
        return Not(condition_to_z3(sub, full_state , groups))

    if t == "equal":
        L = read_operand(cond["left"],  full_state, groups)
        R = read_operand(cond["right"], full_state, groups)

        # Normalize Python bools to Z3
        

        # --- list vs bool (e.g. s_0 == 0) ---
        if isinstance(L, list) and isinstance(R, bool):
           
            if R:
                # every bit in L must be true
                return And(*[li == BoolVal(True) for li in L])
            else:
                # every bit in L must be false
                return And(*[li == BoolVal(False) for li in L])
               
        if isinstance(R, list) and isinstance(L, bool):
            if L:
                # every bit in L must be true
                return And(*[ri == BoolVal(True) for ri in L])
            else:
                # every bit in L must be false
                return And(*[ri == BoolVal(False) for ri in L])

        if isinstance(L, list) and isinstance(R, int):
            if R < 0:
                raise ValueError(f"Negative integer in equality: {R}")
            if R > len(L):
                raise ValueError(f"Integer in equality exceeds list length: {R} > {len(L)}")
            
            else:
                from z3 import PbEq
                return PbEq([(li, 1) for li in L], R)
            
        if isinstance(R, list) and isinstance(L, int):
            if L < 0:
                raise ValueError(f"Negative integer in equality: {L}")
            if L > len(R):
                raise ValueError(f"Integer in equality exceeds list length: {L} > {len(R)}")
            
            else:
                from z3 import PbEq
                return PbEq([(ri, 1) for ri in R], L)
      
        

        

        # --- list vs list: pairwise equality ---
        if isinstance(L, list) and isinstance(R, list):
            if len(L) != len(R):
                raise ValueError(
                    f"List lengths differ in equality: {len(L)} vs {len(R)}"
                )
            return And(*[li == ri for li, ri in zip(L, R)])

        # --- scalar vs scalar (Z3 or int/bool) ---
        # By here, L and R should both be scalar Z3 expressions or ints.
        # If ints 0/1 sneak in, you can map them to BoolVal as well:
        if isinstance(L, int):
            L = BoolVal(bool(L))
        if isinstance(R, int):
            R = BoolVal(bool(R))

        # Now they are both scalar Z3 expressions → simple equality.
        return L == R


def proof_path(path : list[dict], t : int , gen_syn : list ,all_condtion : list, stab_txt_path: str) :
    """
    Given a path (list of steps with conditions and states), build a z3 formula
    that encodes the conditions along the path.

    Each step in the path is a dict with keys:
      - "round": int
      - "node": str
      - "next": str | None
      - "instruction": str | None
      - "condition": z3 BoolRef | None
      - "state": dict

    The resulting formula is the AND of all step conditions.
    """
    
    vars = [v for step in path for info in step["site_info"]  for v in info["vars"].values()]
    faults =  [info["act"] for step in path for info in step["site_info"]]
    gen_syn_z3 = []
    for type, idx in gen_syn:
        #print("type, idx:", type, idx)
        if type == 's' :
           
            syn =  [ anc.z for anc in path[idx]["state"]["ancX"]] + [ anc.x for anc in path[idx]["state"]["ancZ"]]
            gen_syn_z3 += syn 
        elif type == 'f' :
            flag =  [ flag.z for flag in path[idx]["state"]["flagX"]] + [ flag.x for flag in path[idx]["state"]["flagZ"] ]
            gen_syn_z3 += flag 
    #    print("gen_syn_z3:", gen_syn_z3)
    at_most_t_faults = [AtMost( *faults , t)]

    return uniqness_proof(vars, at_most_t_faults,all_condtion,  gen_syn_z3, path[-1]["state"]["data"],stab_txt_path)
    
    
    

    # Add fault constraints if needed
    

    return simplify(path_formula)