# flag_analysis.py
# Minimal Pauli-flow utilities focused on the FLAG qubit.
# Tested with Qiskit 2.x (with compatibility shims).
# Requires: pip install qiskit z3-solver

from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional


# --- Qiskit imports + loader shim ---
from qiskit import QuantumCircuit
try:
    # Some 2.x installs expose a dedicated qasm2 loader
    from qiskit.qasm2 import loads as qasm2_loads  # type: ignore
    _HAS_QASM2 = True
except Exception:
    _HAS_QASM2 = False

from z3 import BoolVal, Xor, Bool,simplify,substitute
from z3 import Solver, unsat, sat
# ---------------------------
# Pauli-flow data structures
# ---------------------------

@dataclass
class QubitXZ:
    x: object  # z3 BoolRef (or BoolVal)
    z: object

@dataclass
class CircuitXZ:
    qubits: List[QubitXZ]

# ---------------------------
# Boolean helpers
# ---------------------------

def bfalse(): return BoolVal(False)
def bxor(a, b): return Xor(a, b)

# ---------- Qiskit 2.2 regmap helper ----------
def _regmap_indices(qc: QuantumCircuit):
    """Return dict: global_qubit_index -> (qreg_name, local_index)."""
    regmap = {}
    idx = 0
    for qreg in qc.qregs:
        for j in range(qreg.size):
            regmap[idx] = (qreg.name, j)
            idx += 1
    return regmap

# ---------------------------
# Init
# ---------------------------

def new_clean_circuit_state(n_qubits: int) -> CircuitXZ:
    """Start with no errors anywhere: X=0, Z=0 per qubit."""
    return CircuitXZ([QubitXZ(bfalse(), bfalse()) for _ in range(n_qubits)])


def new_variable_circuit_state(qc: QuantumCircuit) -> CircuitXZ:
    """
    Initialize each qubit with named Bool variables based on qreg name + local index.
    Produces variables like: q0_x, q0_z, ancX1_x, ancX1_z, flagZ2_x, ...
    """
    regmap = _regmap_indices(qc)
    qubits: List[QubitXZ] = []
    for i in range(qc.num_qubits):
        regname, j = regmap[i]
        prefix = f"{regname}{j}"
        qubits.append(QubitXZ(x=Bool(f"{prefix}_x"), z=Bool(f"{prefix}_z")))
    return CircuitXZ(qubits)



# ---------------------------
# Group detection by register name
# ---------------------------

def detect_qubit_groups(qc: QuantumCircuit) -> Dict[str, List[int]]:
    """
    Group qubits by their register names for Qiskit 2.2 style.
    Each qreg has a name (like 'ancX', 'ancZ', 'flagX', 'flagZ', 'q'),
    and qubits are numbered globally across all registers.
    """
    groups = {'data': [], 'ancX': [], 'ancZ': [], 'flagX': [], 'flagZ': []}

    # Build mapping: global index → (register name, local index)
    idx = 0
    regmap = {}
    for qreg in qc.qregs:
        for j in range(qreg.size):
            regmap[idx] = qreg.name
            idx += 1

    # Classify each qubit index
    for i, reg in regmap.items():
        reg_l = reg.lower()
        if   reg_l.startswith("ancx"):  groups["ancX"].append(i)
        elif reg_l.startswith("ancz"):  groups["ancZ"].append(i)
        elif reg_l.startswith("flagx"): groups["flagX"].append(i)
        elif reg_l.startswith("flagz"): groups["flagZ"].append(i)
        else: groups["data"].append(i)

    return groups

# ---------------------------
# Clifford update rules
# ---------------------------

def apply_h(state: CircuitXZ, q: int) -> None:
    """Hadamard on qubit q: (x,z) <- (z,x)."""
    state.qubits[q].x, state.qubits[q].z = state.qubits[q].z, state.qubits[q].x

def apply_s(state: CircuitXZ, q: int) -> None:
    """Phase S on qubit q: (x,z) <- (x, x xor z)."""
    x, z = state.qubits[q].x, state.qubits[q].z
    state.qubits[q].z = bxor(x, z)

def apply_sdg(state: CircuitXZ, q: int) -> None:
    """S† on qubit q: (x,z) <- (x, z xor x).  (inverse of S)"""
    x, z = state.qubits[q].x, state.qubits[q].z
    state.qubits[q].z = bxor(z, x)

def apply_cnot(state: CircuitXZ, ctrl: int, targ: int) -> None:
    """
    CNOT(c->t):
      x_c' = x_c
      z_c' = z_c xor z_t
      x_t' = x_t xor x_c
      z_t' = z_t
    """
    xc, zc = state.qubits[ctrl].x, state.qubits[ctrl].z
    xt, zt = state.qubits[targ].x, state.qubits[targ].z
    state.qubits[ctrl].x = xc
    state.qubits[ctrl].z = bxor(zc, zt)
    state.qubits[targ].x = bxor(xt, xc)
    state.qubits[targ].z = zt

# ---------------------------
# Fault injection on FLAG
# ---------------------------

def inject_flag_error(state: CircuitXZ, flag_idx: int, kind: str) -> None:
    """
    Insert a Pauli error on the flag qubit at the *current time*.
    kind ∈ {'I','X','Z','Y'}.
    """
    k = kind.upper()
    if k == 'I':
        return
    if k in ('X', 'Y'):
        state.qubits[flag_idx].x = bxor(state.qubits[flag_idx].x, BoolVal(True))
    if k in ('Z', 'Y'):
        state.qubits[flag_idx].z = bxor(state.qubits[flag_idx].z, BoolVal(True))

# ---------------------------
# QASM → Pauli-flow
# ---------------------------

def _qiskit_qubit_index(qc: QuantumCircuit, qobj) -> int:
    """
    Robustly get integer index from a Qiskit Qubit object in 1.x/2.x.
    """
    # find_bit returns a BitLocations with .index
    return qc.find_bit(qobj).index

def apply_qasm_gate_into_state(state: CircuitXZ, name: str, qidxs: List[int]) -> None:
    """Apply supported (Clifford) gates to Pauli-flow state."""
    if name == 'h':
        apply_h(state, qidxs[0])
    elif name == 's':
        apply_s(state, qidxs[0])
    elif name in ('sdg', 'sxdg'):  # sdg is the usual name; include alias just in case
        apply_sdg(state, qidxs[0])
    elif name in ('cx', 'cnot'):
        apply_cnot(state, qidxs[0], qidxs[1])
    elif name in ('id', 'barrier', 'reset', 'measure'):
        # ignored here; measurement is read via final Z/X bits directly
        pass
    else:
        raise NotImplementedError(f"Unsupported gate in Pauli-flow: {name}")

def _load_qasm(qasm_path: str) -> QuantumCircuit:
    """
    Loader shim for Qiskit 2.x: try the classic from_qasm_file first;
    if your build expects qasm2 loader, use it as a fallback.
    """
    try:
        return QuantumCircuit.from_qasm_file(qasm_path)
    except Exception:
        if _HAS_QASM2:
            with open(qasm_path, "r", encoding="utf-8") as f:
                txt = f.read()
            return qasm2_loads(txt)
        raise  # rethrow if no fallback available

def build_state_from_qasm(qasm_path: str) -> Tuple[CircuitXZ, QuantumCircuit]:
    """Load QASM and walk the gates to produce final Pauli-flow state (no faults)."""
    qc = _load_qasm(qasm_path)
    st = new_clean_circuit_state(qc.num_qubits)
    for instr, qargs, _ in qc.data:
        name = instr.name
        qidxs = [_qiskit_qubit_index(qc, q) for q in qargs]
        apply_qasm_gate_into_state(st, name, qidxs)
    return st, qc

def build_variable_state_from_qasm(qasm_path: str) -> Tuple[CircuitXZ, QuantumCircuit, Dict[str, object]]:
    """
    Load QASM, build variable-initialized Pauli-flow state, propagate gates,
    and return (state, qc, varenv) where varenv maps variable names to z3 Bools.
    """
    qc = _load_qasm(qasm_path)

    # 1) variable-initialized state
    state = new_variable_circuit_state(qc)

    # 2) also return a varenv for easy substitutions/evaluation
    regmap = _regmap_indices(qc)
    varenv: Dict[str, object] = {}
    for i in range(qc.num_qubits):
        regname, j = regmap[i]
        prefix = f"{regname}{j}"
        varenv[f"{prefix}_x"] = state.qubits[i].x
        varenv[f"{prefix}_z"] = state.qubits[i].z

    # 3) walk the circuit (Clifford updates)
    for instr, qargs, _ in qc.data:
        name = instr.name
        qidxs = [_qiskit_qubit_index(qc, q) for q in qargs]
        apply_qasm_gate_into_state(state, name, qidxs)

    return state, qc, varenv
# ---------------------------
# Outputs we care about
# ---------------------------


def ancillas_Z(state: CircuitXZ, anc_idxs: List[int]):
    """Syndrome bits if ancillas are measured in Z basis (flips if X on ancilla)."""
    return [state.qubits[i].x for i in anc_idxs]

def ancillas_X(state: CircuitXZ, anc_idxs: List[int]):
    """Syndrome bits if ancillas are measured in X basis (flips if Z on ancilla)."""
    return [state.qubits[i].z for i in anc_idxs]

def flags_Z(state: CircuitXZ, flag_idxs: List[int]):
    """Flag measured in Z basis (flips if X on flag)."""
    return [state.qubits[i].x for i in flag_idxs]

def flags_X(state: CircuitXZ, flag_idxs: List[int]):
    """Flag measured in X basis (flips if Z on flag)."""
    return [state.qubits[i].z for i in flag_idxs]

def data_qubits(state: CircuitXZ, data_idxs: List[int]):
    """
    Return the residual Pauli error components (x,z) for each data qubit.
    Example:
      (False, True)  -> Z error
      (True, False)  -> X error
      (True, True)   -> Y error
      (False, False) -> no error
    """
    return [(state.qubits[i].x, state.qubits[i].z) for i in data_idxs]


def eval_under(boolexpr, assignment: dict, varenv: dict):
    """
    Evaluate a z3 Bool expression under a *partial assignment*.
    assignment: dict like {"q3_x": True, "ancX0_z": False}
    - Only these vars are substituted.
    - Any var not listed stays symbolic (instead of defaulting to False).
    """
    subs = []
    for name, val in assignment.items():
        if name in varenv:
            subs.append((varenv[name], BoolVal(val)))
    return simplify(substitute(boolexpr, subs))

def project_data_only(expr, varenv: dict):
    """
    Substitute anc/flag variables to False, keep all data variables symbolic.
    """
    subs = []
    for name, sym in varenv.items():
        if name.startswith(("ancX","ancZ","flagX","flagZ")):
            subs.append((sym, BoolVal(False)))
    return simplify(substitute(expr, subs))
# ---------------------------
# High-level helper:
#   What happens if the flag has an X/Z/Y error just before measurement?
# ---------------------------

def analyze_flag_errors_multi(
    qasm_path: str,
    anc_idxs: List[int],
    flag_idxs: List[int],
    flag_error_kinds: Dict[int, str],  # mapping: flag_idx -> 'I'|'X'|'Z'|'Y'
):
    """
    Build the Pauli-flow state from QASM, inject specified Pauli errors
    on one or more flag qubits (just before measurement), and return:
      - syn_flips:  List[BoolExpr]  (Z on each ancilla in anc_idxs)
      - flag_flips: List[BoolExpr]  (X-basis flip formula for each flag in flag_idxs)
    """
    state, _ = build_state_from_qasm(qasm_path)

    # Inject errors on the chosen flag qubits
    for fidx, kind in flag_error_kinds.items():
        inject_flag_error(state, fidx, kind)

    syn_flips  = ancillas_Z(state, anc_idxs)
    flag_flips = flags_X(state, flag_idxs)  # X-basis measurement assumed
    return syn_flips, flag_flips

# ---------------------------
# Stabilizers:
#  
# ---------------------------
def load_symplectic_txt(path: str):
    """
    Each line has 'XXXXXXX ZZZZZZZ' (0/1).
    Returns list of (Sx, Sz) where each is a list of 0/1.
    """
    gens = []
    with open(path, 'r') as f:
        for ln in f:
            if not ln.strip():
                continue
            xs, zs = ln.split()
            Sx = [int(c) for c in xs.strip()]
            Sz = [int(c) for c in zs.strip()]
            gens.append((Sx, Sz))
    return gens

def anticomm_formula(Sx, Sz, varenv):
    """
    Build z3 Bool formula:
      ⊕_i ( E_x[i]*S_z[i]  ⊕  E_z[i]*S_x[i] )
    where error vars are q{i}_x, q{i}_z (or data{i}_x/z).
    """
    acc = BoolVal(False)
    for i in range(len(Sx)):
        if Sz[i]:  # stabilizer has Z → anticommutes with X error
            acc = Xor(acc, varenv.get(f"q{i}_x", varenv.get(f"data{i}_x")))
        if Sx[i]:  # stabilizer has X → anticommutes with Z error
            acc = Xor(acc, varenv.get(f"q{i}_z", varenv.get(f"data{i}_z")))
    return acc

# ---------------------------
# Ordered solver check: ancilla[i] ≡ stabilizer[i]
# ---------------------------

def _equiv(a, b) -> bool:
    """True iff a and b are logically equivalent (UNSAT of XOR)."""
    s = Solver()
    s.add(Xor(a, b))          # SAT means there exists an assignment where they differ
    return s.check() == unsat # UNSAT ⇒ no such assignment ⇒ equivalent

def _counterexample(a, b):
    """Return a counterexample model if a ≢ b, else None."""
    s = Solver()
    s.add(Xor(a, b))
    return s.model() if s.check() == sat else None

def check_ancillas_match_symplectic_ordered(qasm_path: str,
                                            stab_txt_path: str,
                                            order: str = "X-then-Z"):
    """
    Pairwise, ordered equivalence:
      ancilla[i]  ≡  anticommute_formula_from_txt_line[i]

    - Ancilla formulas are taken from the circuit and projected to data-only
      (anc/flag vars set False; data vars symbolic).
    - Stabilizer formulas come from the txt (file order preserved).
    - `order` tells how to concatenate ancillas from QASM registers:
         "X-then-Z" (default) means ancX first, then ancZ (both in QASM order).
         "Z-then-X" means ancZ first, then ancX.
      Choose the one that matches the line order in your .txt.
    """
    # Build circuit (symbolic) and detect groups
    state, qc, varenv = build_variable_state_from_qasm(qasm_path)
    groups = detect_qubit_groups(qc)

    # Ancilla flip formulas from circuit → project to data-only
    ancX = [project_data_only(e, varenv) for e in ancillas_X(state, groups["ancX"])]
    ancZ = [project_data_only(e, varenv) for e in ancillas_Z(state, groups["ancZ"])]

    ancillas = (ancX + ancZ) if order == "X-then-Z" else (ancZ + ancX)

    # Stabilizer anticommute formulas from txt (exact line order)
    gens = load_symplectic_txt(stab_txt_path)
    stabs = [anticomm_formula(Sx, Sz, varenv) for (Sx, Sz) in gens]

    if len(ancillas) != len(stabs):
        print(f"[COUNT MISMATCH] ancillas={len(ancillas)} vs stabs={len(stabs)}")
        return {"ok": False, "pairs": [], "mismatches": list(range(min(len(ancillas), len(stabs))))}

    pairs = []
    mismatches = []
    ok_all = True

    for i, (a, s) in enumerate(zip(ancillas, stabs)):
        same = _equiv(a, s)
        pairs.append((i, same))
        if not same:
            ok_all = False
            print(f"[PAIR {i}] NOT EQUIVALENT")
            print("  ancilla expr:", simplify(a))
            print("  stabilizer :", simplify(s))
            m = _counterexample(a, s)
            if m:
                print("  counterexample:", m)

    print("Overall ordered match:", ok_all)
    return {"ok": ok_all, "pairs": pairs, "mismatches": mismatches}

# ---------------------------
# Example CLI usage (optional)
# ---------------------------
if __name__ == "__main__":
    qasm_file = "my_flagged_round.qasm"

    # Suppose your layout has:
    # - ancillas at indices [8, 9, 10]  (three stabilizer measurements)
    # - flags at    indices [11, 12]    (two shared/parallel flags)
    anc_idxs  = [8, 9, 10]
    flag_idxs = [11, 12]

    # Inject a Z fault on flag 11 and a Y fault on flag 12 just before measurement
    errors = {11: "Z", 12: "Y"}

    syn, flg = analyze_flag_errors_multi(qasm_file, anc_idxs, flag_idxs, errors)

    print("Syndrome flips (per ancilla):")
    for i, expr in zip(anc_idxs, syn):
        print(f"  anc[{i}] Z -> {expr}")

    print("Flag X-basis flips (per flag):")
    for i, expr in zip(flag_idxs, flg):
        print(f"  flag[{i}] X-meas flip -> {expr}")