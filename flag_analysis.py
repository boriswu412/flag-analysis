# flag_analysis.py
# Minimal Pauli-flow utilities focused on the FLAG qubit.
# Tested with Qiskit 2.x (with compatibility shims).
# Requires: pip install qiskit z3-solver

from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
from pathlib  import Path

def read_config(path="config.txt"):
    """Read the QASM path from a simple key=value text config."""
    config_file = Path(path)
    if not config_file.exists():
        raise FileNotFoundError("Missing config.txt file!")

    config = {}
    for line in config_file.read_text().splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            config[key.strip()] = value.strip()
    return config



# --- Qiskit imports + loader shim ---
from qiskit import QuantumCircuit

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
try:
    # Some 2.x installs expose a dedicated qasm2 loader
    from qiskit.qasm2 import loads as qasm2_loads  # type: ignore
    _HAS_QASM2 = True
except Exception:
    _HAS_QASM2 = False

from z3 import BoolVal, Xor, Bool,simplify,substitute, And, Not,Or, PbLe, AtMost,ForAll, Implies, Exists, PbGe, AtLeast

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

def xor_list(lst):
    acc = BoolVal(False)
    for v in lst:
        acc = Xor(acc, v)
    return acc
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
# Split circuit 
# ---------------------------
from qiskit import QuantumCircuit, QuantumRegister



def _build_bit_lookup(qc: QuantumCircuit):
    """
    Return dict: { QubitObject -> (register_name, local_index) }
    Works across Qiskit 1.x/2.x because it iterates the circuit's qregs directly.
    """
    m = {}
    for reg in qc.qregs:
        # reg is a QuantumRegister; iterating gives Bit objects in order
        for j, bit in enumerate(reg):
            m[bit] = (reg.name, j)
    return m


def split_circuit_full_q_compact_flags(qc: QuantumCircuit):
    """
    Split `qc` by 'barrier'. In each slice:
      - keep full data register 'q' (same size),
      - create compact ancilla/flag regs containing only used bits for that slice,
      - ignore 'measure' ops; no classical registers created.
      - if the original slice ended at a barrier, append a barrier in the subcircuit.
    Returns: List[QuantumCircuit]
    """
    # 1) collect ops between barriers (skip measures), and remember if a slice ended at a barrier
    slices = []  # list of (ops, ended_by_barrier)
    cur = []
    for instr, qargs, cargs in qc.data:
        if instr.name == "barrier":
            if cur:
                slices.append((cur, True))  # this slice ended due to a barrier
                cur = []
        elif instr.name == "measure":
            continue
        else:
            cur.append((instr, qargs))
    if cur:
        slices.append((cur, False))  # last slice (no trailing barrier)

    # 2) map each Qubit object -> (register_name, local_index)
    bit_lookup = _build_bit_lookup(qc)

    # 3) original data register (kept full size)
    q_reg_orig = next((r for r in qc.qregs if r.name == "q"), None)
    if q_reg_orig is None:
        raise ValueError("Expected a data register named 'q'.")
    q_size = q_reg_orig.size

    subcircuits = []
    for k, (ops, ended_by_barrier) in enumerate(slices):
        # which anc/flag locals are used in this slice
        used_by_reg = {"ancX": set(), "ancZ": set(), "flagX": set(), "flagZ": set()}

        for instr, qargs in ops:
            for qb in qargs:
                rname, lidx = bit_lookup[qb]
                if rname in used_by_reg:
                    used_by_reg[rname].add(lidx)

        # build new regs: full 'q', compact anc/flag only if used
        q_new = QuantumRegister(q_size, "q")
        regs = [q_new]
        remap = {("q", j): q_new[j] for j in range(q_size)}

        def add_compact_reg(reg_name):
            idxs = sorted(used_by_reg[reg_name])
            if not idxs:
                return
            R = QuantumRegister(len(idxs), reg_name)
            regs.append(R)
            for new_i, old_i in enumerate(idxs):
                remap[(reg_name, old_i)] = R[new_i]

        for rn in ("ancX", "ancZ", "flagX", "flagZ"):
            add_compact_reg(rn)

        # build subcircuit and append remapped ops
        sub = QuantumCircuit(*regs, name=f"stab_{k}_fullq_compactflags")
        for instr, qargs in ops:
            new_qargs = []
            for qb in qargs:
                rname, lidx = bit_lookup[qb]
                key = (rname, lidx)
                if key not in remap:
                    raise KeyError(f"Unmapped bit {rname}[{lidx}] in slice {k}")
                new_qargs.append(remap[key])
            sub.append(instr, new_qargs, [])

        # append a barrier if the original block ended at a barrier
        if ended_by_barrier:
            sub.barrier(*sub.qubits)

        subcircuits.append(sub)

    return subcircuits


from qiskit.qasm2 import dumps as qasm2_dumps  # for Qiskit 2.x

def save_qasm_full_slices(qc, prefix="stab"):
    """Split by barriers and save each stabilizer as a .qasm file (Qiskit 2.x compatible)."""
    subs = split_circuit_full_q_compact_flags(qc)
    for i, sub in enumerate(subs):
        filename = f"{prefix}_{i}.qasm"
        try:
            # Preferred (Qiskit 2.x)
            qasm_str = qasm2_dumps(sub)
        except Exception:
            # Fallback for Qiskit 1.x
            qasm_str = sub.qasm()
        with open(filename, "w", encoding="utf-8") as f:
            f.write(qasm_str)
        print(f"Saved: {filename}")
def remove_flag_gates(qasm_path: str, save_path: str = None):
    """
    Load a QASM file, remove all gates that act on flag qubits (flagX[...] or flagZ[...]),
    but preserve the original `barrier` gates.
    """
    qc = QuantumCircuit.from_qasm_file(qasm_path)
    new_qc = QuantumCircuit(*qc.qregs, *qc.cregs)

    # Map: global index -> register name
    regmap = {}
    idx = 0
    for qreg in qc.qregs:
        for _ in range(qreg.size):
            regmap[idx] = qreg.name.lower()
            idx += 1

    def is_flag_qubit(qbit):
        """Check if the given Qubit belongs to a flag register."""
        loc = qc.find_bit(qbit)
        reg_name = regmap.get(loc.index, "")
        return reg_name.startswith("flagx") or reg_name.startswith("flagz")

    # Filter out gates acting on flag qubits, but keep barriers
    for instr, qargs, cargs in qc.data:
        if instr.name == "barrier":
            # Always include barrier gates
            new_qc.append(instr, qargs, cargs)
        elif any(is_flag_qubit(q) for q in qargs):
            # Skip gates that act on flag qubits
            continue
        else:
            # Include all other gates
            new_qc.append(instr, qargs, cargs)

    # Optionally save the modified circuit
    if save_path:
        with open(save_path, "w") as f:
            f.write(qasm2_dumps(new_qc))  # Use qasm2_dumps to generate QASM string

    return new_qc
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


def apply_cz(state: CircuitXZ, ctrl: int, targ: int) -> None:
    """
    CZ(c->t):
      x_c' = x_c
      z_c' = z_c  xor x_t
      x_t' = x_t 
      z_t' = z_t xor x_c 
    """
    xc, zc = state.qubits[ctrl].x, state.qubits[ctrl].z
    xt, zt = state.qubits[targ].x, state.qubits[targ].z
    state.qubits[ctrl].x = xc
    state.qubits[ctrl].z = bxor(xt ,  zc)
    state.qubits[targ].x = xt
    state.qubits[targ].z =  bxor(xc, zt)


def apply_notnot(state: CircuitXZ, ctrl: int, targ: int) -> None:
    """
    NOTNOT(c->t):
      x_c' = x_c xor z_t
      z_c' = z_c 
      x_t' = x_t  xor  z_c
      z_t' = z_t
    """
    xc, zc = state.qubits[ctrl].x, state.qubits[ctrl].z
    xt, zt = state.qubits[targ].x, state.qubits[targ].z
    state.qubits[ctrl].x = bxor(xc, zt)
    state.qubits[ctrl].z = zc
    state.qubits[targ].x = bxor(xt, zc)
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

def inject_flag_symbolic_one_axis(state, fidx: int, axis="z", prefix="ferr"):
    """
    Add a symbolic error variable on a flag qubit, restricted to X *or* Z axis.

    axis: "x" or "z"
    """
    var = Bool(f"{prefix}{fidx}_{axis}")

    if axis == "x":
        # Only X part of error
        state.qubits[fidx].x = Xor(state.qubits[fidx].x, var)
    elif axis == "z":
        # Only Z part of error
        state.qubits[fidx].z = Xor(state.qubits[fidx].z, var)
    else:
        raise ValueError("axis must be 'x' or 'z'")

    return var

def inject_symbolic_one_axis_many(state, idxs, axis="z", prefix="ferr"):
    """
    Inject one symbolic Boolean on the chosen axis for EACH qubit in `idxs`.
    axis: 'x' or 'z'
    Returns: list of created Bool vars (same order as idxs).
    """
    if axis not in ("x", "z"):
        raise ValueError("axis must be 'x' or 'z'")
    vars_created = []
    for i in idxs:
        v = Bool(f"{prefix}{i}_{axis}")
        if axis == "x":
            state.qubits[i].x = Xor(state.qubits[i].x, v)
        else:
            state.qubits[i].z = Xor(state.qubits[i].z, v)
        vars_created.append(v)
    return vars_created

def inject_on_ancillas(state, anc_idxs, axis="z", prefix="ancFault"):
    """Inject symbolic faults on ancillas only (one axis)."""
    return inject_symbolic_one_axis_many(state, anc_idxs, axis=axis, prefix=prefix)

def inject_on_flags(state, flag_idxs, axis="z", prefix="flagFault"):
    """Inject symbolic faults on flags only (one axis)."""
    return inject_symbolic_one_axis_many(state, flag_idxs, axis=axis, prefix=prefix)

def _inject_1q_fault_after(state, q, fault_kind=None, prefix="f"):
    """
    One-qubit Pauli on wire q after a 1q gate (or on ONE wire of a 2q gate):
      fault_kind: None -> symbolic; 'I'|'X'|'Z'|'Y' -> concrete.
    Returns dict {'fx','fz','act'}.
    """
    if fault_kind is None:
        fx = Bool(f"{prefix}_x"); fz = Bool(f"{prefix}_z")
    else:
        k = fault_kind.upper()
        fx = BoolVal(k in ("X","Y")); fz = BoolVal(k in ("Z","Y"))
    state.qubits[q].x = Xor(state.qubits[q].x, fx)
    state.qubits[q].z = Xor(state.qubits[q].z, fz)
    return {"fx": fx, "fz": fz, "act": Or(fx, fz)}

def _inject_2q_fault_after(state, q0: int, q1: int, fault_kind=None, prefix="f"):
    """
    Gate-agnostic 2-qubit Pauli injection on wires (q0, q1) *after* a 2q gate.
    It does not depend on the specific 2q gate; it simply toggles X/Z components.

    Args:
      state: CircuitXZ (your Pauli-flow state)
      q0, q1: qubit indices (in the circuit's global indexing)
      fault_kind:
        - None               -> symbolic on both wires
        - (k0, k1)           -> concrete per-wire, each in {'I','X','Z','Y'} (case-insensitive)
          e.g. ('Y','X') means inject Y on q0 and X on q1
      prefix: name prefix for z3 symbols (when symbolic)

    Returns:
      info: dict with z3 literals and helpers:
        {
          'fx0','fz0','fx1','fz1',   # per-wire Pauli indicator bits
          'act0','act1',             # per-wire activity (X or Z non-identity)
          'act'                      # any activity on either wire
        }
    """
    if fault_kind is None:
        fx0 = Bool(f"{prefix}_x0"); fz0 = Bool(f"{prefix}_z0")
        fx1 = Bool(f"{prefix}_x1"); fz1 = Bool(f"{prefix}_z1")
    else:
        k0, k1 = fault_kind
        k0 = k0.upper(); k1 = k1.upper()
        fx0 = BoolVal(k0 in ("X","Y")); fz0 = BoolVal(k0 in ("Z","Y"))
        fx1 = BoolVal(k1 in ("X","Y")); fz1 = BoolVal(k1 in ("Z","Y"))

    # Inject on both wires (gate-agnostic)
    state.qubits[q0].x = Xor(state.qubits[q0].x, fx0)
    state.qubits[q0].z = Xor(state.qubits[q0].z, fz0)
    state.qubits[q1].x = Xor(state.qubits[q1].x, fx1)
    state.qubits[q1].z = Xor(state.qubits[q1].z, fz1)

    act0 = Or(fx0, fz0); act1 = Or(fx1, fz1)
    info = {
        "fx0": fx0, "fz0": fz0, "fx1": fx1, "fz1": fz1,
        "act0": act0, "act1": act1,
        "act": Or(act0, act1),
        
    }
    return info

def add_fault_mode_constraints(solver, info, fault_mode="2q", fault_kind=None):
    """
    Add constraints to the solver according to the declared fault_mode.

    - "1q": at most one of {act_c, act_t} is true
    - "2q": both act_c and act_t are true
    - "either": no extra restriction
    """
    if fault_mode == "1q":
        solver.add(AtMost(info["act_c"], info["act_t"], 1))
    elif fault_mode == "2q":
        solver.add(Or(info["act_c"], info["act_t"]))
    # "either": do nothing
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

    elif name in ('notnot'):  # notnot is a common alias for ccx
        apply_notnot(state, qidxs[0], qidxs[1])

    elif name in ('cz'):
        apply_cz(state, qidxs[0], qidxs[1])

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


def build_state_with_fault_after_gate(qasm_path: str, gate_index: int, fault_mode="either", fault_kind=None):
    """
    Run circuit ideally; inject a fault right AFTER gate #gate_index only.
    fault_mode: '1q' | '2q' | 'either'  (for CNOTs)
    fault_kind:
      - None                -> symbolic
      - 'I'|'X'|'Z'|'Y'     -> for 1q gates, or for CNOT in mode='1q' (applied to one wire)
      - (kc,kt)             -> for CNOT in mode='2q'/'either' (concrete per-wire)
    Returns: (state, qc, site_info, groups)
    """
    qc = _load_qasm(qasm_path)
    state = new_clean_circuit_state(qc.num_qubits)
    groups = detect_qubit_groups(qc)
    site_info = None

    for i, (instr, qargs, _) in enumerate(qc.data):
        name = instr.name
        qidxs = [_qiskit_qubit_index(qc, q) for q in qargs]

        if name in ("h","s","sdg"):
            apply_qasm_gate_into_state(state, name, qidxs)
            if i == gate_index:
                info = _inject_1q_fault_after(
                    state, qidxs[0],
                    fault_kind=None if fault_kind is None else fault_kind,
                    prefix=f"f_site{i}"
                )
                site_info = {
                    "gate_index": i, "gate_name": name,
                    "qubits": (qidxs[0],),
                    "vars": info, "act": info["act"], "fault_mode": "1q"
                }

        elif name in ("cx","cnot", "notnot", "cz"):
            c, t = qidxs
            if (name == "cx"  or name == "cnot"):apply_cnot(state, c, t)
            elif (name == "notnot"): apply_notnot(state, c, t)
            elif (name == "cz"): apply_cz(state, c, t)
            if i == gate_index:
                info = _inject_2q_fault_after(
                    state, c, t, 
                    fault_kind=None if fault_kind is None else fault_kind,
                    prefix=f"f_gate{i}"
                )
                site_info = {
                    "gate_index": i, "gate_name": "cx",
                    "qubits": (c, t),
                    "vars": info, "act": info["act"], "fault_mode": fault_mode
                }

        elif name in ("barrier","id","reset","measure"):
            if i == gate_index:
                site_info = {
                    "gate_index": i, "gate_name": name,
                    "qubits": tuple(qidxs),
                    "vars": {}, "act": BoolVal(False), "fault_mode": "none"
                }
        else:
            raise NotImplementedError(f"Unsupported gate: {name}")

    if site_info is None:
        raise IndexError(f"gate_index {gate_index} out of range (len={len(qc.data)})")

    return state, qc, site_info, groups


from copy import deepcopy
def symbolic_propagate_state_checked(qasm_path: str, init_state, *, track_steps=False):
    """
    Propagate a CircuitXZ `init_state` through a Qiskit QuantumCircuit `qc`
    using the Pauli-flow update rules in `apply_qasm_gate_into_state`.

    - Verifies qubit-count match (circuit vs. state).
    - Ignores non-evolution ops: barrier/reset/measure/id.
    - If track_steps=True, also returns a list of (gate_index, name, qidxs, state_snapshot).

    Returns:
        final_state                    if track_steps == False
        (final_state, step_snapshots)  if track_steps == True
    """
    qc = _load_qasm(qasm_path)
    # --- consistency check ---
    n_circ = qc.num_qubits
    n_state = len(init_state.qubits)
    if n_circ != n_state:
        raise ValueError(f"Qubit count mismatch: circuit={n_circ}, state={n_state}")

    # we won’t mutate caller’s state
    state = deepcopy(init_state)

    # Qiskit 2.x: map Qubit -> global index
    def _qidx(qbit):
        return qc.find_bit(qbit).index

    snapshots = []  # (i, name, qidxs, deepcopy(state))

    
    # Walk gates
    for i, (instr, qargs, _cargs) in enumerate(qc.data):
        name = instr.name.lower()
        qidxs = [_qidx(q) for q in qargs]

        if name in ('barrier', 'reset', 'measure', 'id'):
            # no evolution needed
            continue

        # delegate the actual Clifford update to your centralized function
        apply_qasm_gate_into_state(state, name, qidxs)

        if track_steps:
            snapshots.append((i, name, tuple(qidxs), deepcopy(state)))
        
        print(f"After gate {i}: {name} on qubits {qidxs}")
        print("state:")
        print(state.qubits[1].z)

    return (state, snapshots) if track_steps else state


def _reset_qubits(state: CircuitXZ, idxs):
    """Set (x,z) = (False, False) for each qubit index in idxs."""
    for i in idxs:
        state.qubits[i].x = BoolVal(False)
        state.qubits[i].z = BoolVal(False)

def _reset_qubit_x(state: CircuitXZ, idxs):
    """Set x = False for x part qubit index idx."""
    for i in idxs:
        state.qubits[i].x = BoolVal(False)

def _reset_qubit_z(state: CircuitXZ, idxs):
    """Set z = False for z part qubit index idx."""
    for i in idxs:
        state.qubits[i].z = BoolVal(False)

def symbolic_propagate_with_resets(
    qc: QuantumCircuit,
    init_state: CircuitXZ,
    *,
    track_steps: bool = False,
    reset_groups=("ancX", "ancZ", "flagX", "flagZ"),
    reset_at_start: bool = True,
    reset_on_barrier: bool = True,
    reset_on_measure: bool = False,
):
    """
    Propagate a CircuitXZ `init_state` through QASM at `qasm_path` with Pauli-flow
    updates, and optionally reset ancilla/flag groups to clean (X=Z=False):

      - `reset_groups`: tuple of group names to reset (default all anc/flag).
      - `reset_at_start`: reset selected groups before the first gate.
      - `reset_on_barrier`: reset selected groups whenever a 'barrier' is seen.
      - `reset_on_measure`: reset a qubit if it is measured *and* belongs to selected groups.

    Returns:
        final_state
        or (final_state, snapshots) if track_steps=True, where snapshots is
        a list of (gate_index, name, qidxs, deep_copied_state).
    """
    #qc = _load_qasm(qasm_path)

    # --- consistency check ---
    n_circ = qc.num_qubits
    n_state = len(init_state.qubits)
    if n_circ != n_state:
        raise ValueError(f"Qubit count mismatch: circuit={n_circ}, state={n_state}")

    # copy so we don't mutate caller's state
    state = deepcopy(init_state)

    # Build groups from register names
    groups = detect_qubit_groups(qc)   # expects keys: 'data','ancX','ancZ','flagX','flagZ'
    group_idxs = {g: groups.get(g, []) for g in ("data","ancX","ancZ","flagX","flagZ")}

    print("groups detected:", group_idxs)

    # Which indices are selected for bulk resets?
    selected_reset_idxs = []
    for g in reset_groups:
        selected_reset_idxs.extend(group_idxs.get(g, []))
    selected_reset_idxs = sorted(set(selected_reset_idxs))

    # Qubit → global index (Qiskit 2.x)
    def _qidx(qbit):
        return qc.find_bit(qbit).index

    #print("Selected qubits for resets:", selected_reset_idxs)
    # optional initial reset
    if reset_at_start and selected_reset_idxs:
        #print("Performing initial reset on selected qubits.")
        _reset_qubits(state, selected_reset_idxs)

    snapshots = []

    for i, (instr, qargs, cargs) in enumerate(qc.data):
        name = instr.name.lower()
        qidxs = [_qidx(q) for q in qargs]

        print(f"Processing gate {i}: {name} on qubits {qidxs}")

        if name in ('id', 'reset'):
            continue

        if name == 'barrier':
            
            if track_steps:
                snapshots.append((i, name, tuple(qidxs), deepcopy(state)))
            continue

        if name == 'measure':
            if reset_on_measure:
                to_reset = [q for q in qidxs if q in selected_reset_idxs]
                if to_reset:
                    _reset_qubits(state, to_reset)
            if track_steps:
                snapshots.append((i, name, tuple(qidxs), deepcopy(state)))
            continue

        # Apply the actual Clifford update
        apply_qasm_gate_into_state(state, name, qidxs)

        if track_steps:
            snapshots.append((i, name, tuple(qidxs), deepcopy(state)))

        
        #_reset_qubit_x(state, groups.get("ancX", []))

        #_reset_qubit_z(state, groups.get("ancZ", []))

        #_reset_qubit_x(state, groups.get("flagX", []))

        #_reset_qubit_z(state, groups.get("flagZ", []))

    return (state, snapshots) if track_steps else state

def build_state_with_faults_after_gates(qasm_path: str, gate_indices: list, fault_mode="either", fault_kind=None):
    """
    Run circuit ideally; inject faults right AFTER all gates in `gate_indices`.
    
    fault_mode: '1q' | '2q' | 'either'  (for 2-qubit gates)
    fault_kind:
      - None                -> symbolic
      - 'I'|'X'|'Z'|'Y'     -> for 1q gates, or for CNOT in mode='1q' (applied to one wire)
      - (kc,kt)             -> for 2q gates (concrete per-wire)
    
    Returns:
      state, qc, sites_info, groups
      where `sites_info` is a list of site_info dicts (one per injected fault)
    """
    qc = _load_qasm(qasm_path)
    state = new_clean_circuit_state(qc.num_qubits)
    groups = detect_qubit_groups(qc)
    print("groups detected:", groups)
    sites_info = []

    gate_indices = set(gate_indices)  # so we can check membership quickly

    for i, (instr, qargs, _) in enumerate(qc.data):
        name = instr.name
        qidxs = [_qiskit_qubit_index(qc, q) for q in qargs]
        print(f"Processing gate {i}: {name} on qubits {qidxs}")
        if name in ("h","s","sdg"):
            apply_qasm_gate_into_state(state, name, qidxs)
            if i in gate_indices:
                info = _inject_1q_fault_after(
                    state, qidxs[0],
                    fault_kind=None if fault_kind is None else fault_kind,
                    prefix=f"f_site{i}"
                )
                sites_info.append({
                    "gate_index": i, "gate_name": name,
                    "qubits": (qidxs[0],),
                    "vars": info, "act": info["act"], "fault_mode": "1q"
                })

        elif name in ("cx","cnot","notnot","cz"):
            c, t = qidxs
            if name in ("cx","cnot"):
                apply_cnot(state, c, t)
                #print("CNOT",c, t )

            elif name == "notnot":
                apply_notnot(state, c, t)
            
            elif name == "cz":
                apply_cz(state, c, t)

            if i in gate_indices:

                
                info = _inject_2q_fault_after(
                    state, c, t,
                    fault_kind=None if fault_kind is None else fault_kind,
                    prefix=f"faulty_gate{i}"
                )
                sites_info.append({
                    "gate_index": i, "gate_name": name,
                    "qubits": (c, t),
                    "vars": info, "act": info["act"], "fault_mode": fault_mode
                })

        elif name in ("barrier","id","reset","measure"):
            
            if i in gate_indices:
                sites_info.append({
                    "gate_index": i, "gate_name": name,
                    "qubits": tuple(qidxs),
                    "vars": {}, "act": BoolVal(False), "fault_mode": "none"
                })
        
        else:
            raise NotImplementedError(f"Unsupported gate: {name}")

        #_reset_qubit_x(state, groups.get("ancX", []))

        #_reset_qubit_z(state, groups.get("ancZ", []))

        #_reset_qubit_x(state, groups.get("flagX", []))

        #_reset_qubit_z(state, groups.get("flagZ", []))
      
    if not sites_info:
        raise IndexError(f"gate_indices {gate_indices} produced no injections (len={len(qc.data)})")

    return state, qc, sites_info, groups

def build_stab_equiv_errors(E_x, E_z, stab_txt_path, prefix="g"):
    """
    Construct stabilizer-equivalent errors.

    Args:
      E_x, E_z: lists of z3 Bool formulas for data qubits
      stab_txt_path: path to stabilizer .txt file
      prefix: name prefix for generator selector variables (default "g")

    Returns:
      (Epx, Epz, gsel)
        - Epx, Epz: new error expressions after applying all possible generator products
        - gsel: list of selector Bool variables, one per generator
    """
    # Load stabilizers from file
    gens = load_symplectic_txt(stab_txt_path)
    m = len(gens)     # number of generators
    n = len(E_x)      # number of data qubits

    # Create selector vars g0..g{m-1}
    gsel = [Bool(f"{prefix}{j}") for j in range(m)]

    # Collect which generators flip which qubit components
    addX = [[] for _ in range(n)]
    addZ = [[] for _ in range(n)]
    for j, (Sx, Sz) in enumerate(gens):
        gj = gsel[j]
        for i in range(n):
            if Sx[i]: addX[i].append(gj)   # Z on generator anticommutes with X error
            if Sz[i]: addZ[i].append(gj)   # X on generator anticommutes with Z error

    # Apply XOR modifications to each data qubit
    Epx, Epz = [], []
    for i in range(n):
        xi = E_x[i]
        for t in addX[i]:
            xi = Xor(xi, t)
        zi = E_z[i]
        for t in addZ[i]:
            zi = Xor(zi, t)
        Epx.append(xi)
        Epz.append(zi)

    return Epx, Epz, gsel
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

def data_error_weight_literals(state, data_idxs):
    """
    Return a list of literals indicating whether each data qubit carries
    any non-trivial Pauli error (X or Z).
    Useful for counting error weight with PB constraints.
    """
    return [Or(state.qubits[i].x, state.qubits[i].z) for i in data_idxs]


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
            #print("Adding Z term for qubit", i)
            acc = Xor(acc, varenv.get(f"q{i}_z", varenv.get(f"data{i}_z")))
        #print(f"Step {i}: acc = {acc}")
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
    print(f"Total ancillas considered: {len(ancillas)}")
    for a in ancillas:
        print("Ancilla formula:", a)
    # Stabilizer anticommute formulas from txt (exact line order)
    gens = load_symplectic_txt(stab_txt_path)
    stabs = [anticomm_formula(Sx, Sz, varenv) for (Sx, Sz) in gens]

    

    if len(ancillas) != len(stabs):
        print(f"[COUNT MISMATCH] ancillas={len(ancillas)} vs stabs={len(stabs)}")
        return {"ok": False, "mismatches": list(range(min(len(ancillas), len(stabs))))}

    # Combine all equivalence checks into a single AND condition
    combined_condition = And(*[Xor(a, s) == False for a, s in zip(ancillas, stabs)])

    # Check if the combined condition is satisfied
    s = Solver()
    s.add(Not(combined_condition))  # Check if there exists a counterexample
    if s.check() == unsat:
        #print("Overall ordered match: True")
        return {"ok": True, "mismatches": []}
    else:
        #print("Overall ordered match: False")
        #print("Counterexample:", s.model())
        mismatches = [i for i, (a, s) in enumerate(zip(ancillas, stabs)) if not _equiv(a, s)]
        return {"ok": False, "mismatches": mismatches}


##For checking the general syndromes are unique
def exists_stab_equiv(E1_x, E1_z, E2_x, E2_z, stab_txt_path):
    """
    gens: [(Sx, Sz), ...] each Sx,Sz list[int] length n
    Returns: (constraint, selectors)
      constraint encodes: ∃ gsel . E2 == E1 ⊕ (sum of selected generators)
    """

    gens = load_symplectic_txt(stab_txt_path)
    m = len(gens); n = len(E1_x)
    gsel = [Bool(f"gsel_{j}") for j in range(m)]

    # per-qubit XOR sums of selected generator columns
    addX = [BoolVal(False) for _ in range(n)]  # contributes to X part from Sz
    addZ = [BoolVal(False) for _ in range(n)]  # contributes to Z part from Sx
    for j, (Sx, Sz) in enumerate(gens):
        gj = gsel[j]
        for i in range(n):
            if Sz[i]: addX[i] = Xor(addX[i], gj)
            if Sx[i]: addZ[i] = Xor(addZ[i], gj)

    eqs = []
    for i in range(n):
        eqs.append(E2_x[i] == Xor(E1_x[i], addX[i]))
        eqs.append(E2_z[i] == Xor(E1_z[i], addZ[i]))
    return And(eqs), gsel

def check_gate_k_with_fault(
    qasm_path: str,
    gate_index: int,
    fault_mode: str = "2q",   # '1q' or '2q' or 'either'
    fault_kind = None,            # None | 'X'|'Z'|'Y' | (kc,kt)
    w_min: int = 2
):
    """
    Inject ONE fault after gate k with the given fault_mode/kind.
    Return (ok, info) where ok=True if UNSAT (i.e., weight ≥ w_min for all assignments).
    """
    state, qc, site, groups = build_state_with_fault_after_gate(
        qasm_path, gate_index, fault_mode=fault_mode, fault_kind=fault_kind
    )
    b = data_error_weight_literals(state, groups["data"])

    s = Solver()
    # Enforce the chosen fault structure if it's a CNOT site
    add_fault_mode_constraints(s, site, fault_mode, fault_kind)

    # Look for a counterexample: data-weight ≤ w_min-1
    s.add(PbLe([(bi,1) for bi in b], w_min-1))

    if s.check() == unsat:
        return True, {"gate": qc.data[gate_index][0].name, "index": gate_index}
    mdl = s.model()
    # Report which wires (and which Pauli) got chosen under the model
    if site["gate_name"] == "cx":
        v = site["vars"]
        ctrl = (
            "Y" if (bool(mdl.eval(v["fxc"], True)) and bool(mdl.eval(v["fzc"], True)))
            else ("X" if bool(mdl.eval(v["fxc"], True))
            else ("Z" if bool(mdl.eval(v["fzc"], True)) else "I"))
        )
        targ = (
            "Y" if (bool(mdl.eval(v["fxt"], True)) and bool(mdl.eval(v["fzt"], True)))
            else ("X" if bool(mdl.eval(v["fxt"], True))
            else ("Z" if bool(mdl.eval(v["fzt"], True)) else "I"))
        )
        where = f"cx c={site['qubits'][0]}, t={site['qubits'][1]}  (ctrl={ctrl}, targ={targ})"
    else:
        v = site["vars"]
        k = "Y" if (bool(mdl.eval(v["fx"], True)) and bool(mdl.eval(v["fz"], True))) \
            else ("X" if bool(mdl.eval(v["fx"], True)) else ("Z" if bool(mdl.eval(v["fz"], True)) else "I"))
        where = f"{site['gate_name']} q={site['qubits'][0]}  ({k})"

    bvals = [bool(mdl.eval(e, True)) for e in b]
    return False, {
        "gate": qc.data[gate_index][0].name,
        "index": gate_index,
        "fault": where,
        "data_weight": sum(bvals),
        "data_bits_true": [groups["data"][i] for i,v in enumerate(bvals) if v],
    }
from z3 import Solver, ForAll, Exists, Or, Xor, PbLe

def forall_fault_exists_low_weight_per_gate(
    qasm_path: str,
    stab_txt_path: str,
    gate_indices=None,          # e.g. range(10) or [0,1,2]
    fault_mode: str = "2q",     # "2q" | "1q" | "either" (whatever your builder accepts)
    flag_axis: str = "z",       # inject flag error on this axis ("x" or "z")
    flag_prefix: str = "flagErr",
):
    """
    This is for checking 'bad loocation'
    For each gate in `gate_indices`:
      state, qc, site_info, groups = build_state_with_fault_after_gate(...)
      fault_vars = all 'f*' vars from site_info['vars'] (universally quantified)
      E' = stabilizer-equivalent data error
      b  = per-qubit error indicators
      Check:  ∀ fault_vars. ( Or(fault_vars) → ∃ gsel.  sum(b) ≤ 1 )
      And also: Xor(site_info['act'], flag_var)   (your extra constraint)

    Returns: dict {gate_index: {"result": sat/unsat, "num_fault_vars": int, "num_gens": int}}
    """
    results = {}
    unsat_gates = []
    # If user didn't pass indices, default to all gates
    if gate_indices is None:
        # Peek the circuit once to know how many gates
        _, qc, _, _ = build_state_with_fault_after_gate(qasm_path, gate_index=0, fault_mode=fault_mode)
        gate_indices = range(len(qc.data))

    for i in gate_indices:
        # 1) Build state with *symbolic* fault inserted after gate i
        state, qc, site_info, groups = build_state_with_fault_after_gate(
            qasm_path, gate_index=i, fault_mode=fault_mode
        )

        # 2) Collect the fault variables at this site (universally quantified)
        fault_vars = [v for k, v in site_info["vars"].items() if k.startswith("f")]
        if not fault_vars:
            # No fault DOFs at this site (e.g. a barrier/measure) → skip
            results[i] = {"result": "no-fault-vars", "num_fault_vars": 0, "num_gens": 0}
            continue

        # 3) Extract data error (E_x, E_z)
        data_idxs = groups["data"]
        E_x = [state.qubits[j].x for j in data_idxs]
        E_z = [state.qubits[j].z for j in data_idxs]

        

        # 4) Build stabilizer-equivalent errors E' using selector Booleans gsel
        Epx, Epz, gsel = build_stab_equiv_errors(E_x, E_z, stab_txt_path, prefix=f"g")

        # 5) Weight ≤ 1 predicate: sum over per-qubit indicators b_i = Or(E′x_i, E′z_i)
        b = [Or(xi, zi) for xi, zi in zip(Epx, Epz)]

        # 6) ∀ fault_vars: Or(fault_vars) → ∃ gsel: sum(b) ≤ 1
        s = Solver()
        body = Exists(gsel, PbLe([(bi, 1) for bi in b], 1))
        phi  = ForAll(fault_vars, Or(fault_vars) == False)  # placeholder replaced below

        # Rebuild phi cleanly (the line above avoids z3py “no quantifier vars” edge cases if empty)
        phi = ForAll(fault_vars, 
                     Or(  # (¬any_fault) ∨ (∃ gsel: weight ≤ 1)
                        Or([v for v in fault_vars]) == False,
                        body
                     ))

        # 7) Add both the quantified property and your extra XOR constraint
        s.add(phi)
        

        res = s.check()
        results[i] = {
            "result": str(res),
            "num_fault_vars": len(fault_vars),
            "num_gens": len(gsel),
        }
        if str(res) == "unsat":
            unsat_gates.append(i)

    return results, unsat_gates

# ---------------------------
# Rename symbol
# ---------------------------
def primed_copy(exprs: list, rename: dict):
    """Return [ substitute(e, rename) for e in exprs ]."""
    return [substitute(e, [(k, v) for k, v in rename.items()]) for e in exprs]

def make_renamer_from_symbols(symbols: list, suffix= "_p"):
    """
    Given a list of z3 symbols (BoolRef) that appear in E_x/E_z etc.,
    build a rename map sym -> fresh Bool with a suffix.
    """
    ren = {}
    for s in symbols:
        # s.decl().name() gets 'f_gate3_x0' etc.
        ren[s] = Bool(s.decl().name() + suffix)
    return ren

# ---------------------------
# For evakluation
# ---------------------------

from z3 import Bool, BoolVal, substitute, simplify, is_true, Z3_OP_UNINTERPRETED, is_bool

def collect_bool_symbols(expr):
    """Recursively collect all uninterpreted Bool symbols in expr."""
    syms = set()
    def _walk(e):
        if is_bool(e) and e.decl().kind() == Z3_OP_UNINTERPRETED and e.num_args() == 0:
            syms.add(e)
        for ch in e.children():
            _walk(ch)
    _walk(expr)
    return syms

def eval_with_values(exprs, assignment, default_false=True):
    """
    Evaluate Z3 Bool expr(s) under a dict of variable→bool values.
    Missing vars default to False if default_false=True.
    Works for a single expr or an iterable of exprs.
    """
    def _eval_one(e):
        syms = collect_bool_symbols(e)
        # Map names of symbols that actually appear in e
        name2sym = {s.decl().name(): s for s in syms}

        subs = []
        # Apply provided assignments *only for symbols that appear in e*
        for name, val in assignment.items():
            if name in name2sym:
                subs.append((name2sym[name], BoolVal(val)))

        # Default any remaining symbols (that appear in e) to False if requested
        if default_false:
            for name, s in name2sym.items():
                if name not in assignment:
                    subs.append((s, BoolVal(False)))

        return is_true(simplify(substitute(e, subs)))

    if isinstance(exprs, (list, tuple, set)):
        return [_eval_one(e) for e in exprs]
    else:
        return _eval_one(exprs)

# ---------------------------
# Example CLI usage (optional)
# ---------------------------
def prove_syndrome_extractions(qasm_path: str, stab_txt_path: str):
    state, qc, varenv = build_variable_state_from_qasm(qasm_path)
    groups = detect_qubit_groups(qc)

    # Flip predicates (basis-aware)
    synX_exprs = ancillas_X(state, groups["ancX"])   # X-type syndromes (check .z)
    synZ_exprs = ancillas_Z(state, groups["ancZ"])   # Z-type syndromes (check .x)
    flgX_exprs = flags_X(state, groups["flagX"])     # flags measured in X (check .z)
    flgZ_exprs = flags_Z(state, groups["flagZ"])     # flags measured in Z (check .x)

    
    # Build an assignment:
    # - allow arbitrary data errors via named vars (you can set a subset True)
    # - force all anc/flag variables to False to model "no circuit faults"
    asgmt = {}

    # Force all anc/flag vars False:
    for name in varenv:
        if name.startswith("ancX") or name.startswith("ancZ") or name.startswith("flagX") or name.startswith("flagZ"):
            asgmt[name] = False

    # Example 1: single X error on q[3]  (Steane’s first X-stabilizer should click)

    # (all other q*_x/z default to False)

    # Evaluate syndromes/flags
    synX_vals = [eval_under(e, asgmt, varenv) for e in synX_exprs]
    synZ_vals = [eval_under(e, asgmt, varenv) for e in synZ_exprs]
    flgX_vals = [eval_under(e, asgmt, varenv) for e in flgX_exprs]
    flgZ_vals = [eval_under(e, asgmt, varenv) for e in flgZ_exprs]

    #print("AncX (X-type) syndromes:", synX_vals)
    #print("AncZ (Z-type) syndromes:", synZ_vals)
    #print("Flags X-basis:", flgX_vals)
    #print("Flags Z-basis:", flgZ_vals)

    # Load stabilizers
    stabs = load_symplectic_txt( stab_txt_path)
    #print("Stabilizers:", stabs)
    # Get Boolean formulas
    stab_exprs = [anticomm_formula(Sx, Sz, varenv) for Sx,Sz in stabs]

    for i, e in enumerate(stab_exprs):
        print(f"Stabilizer {i} formula:", e)

    report = check_ancillas_match_symplectic_ordered(
    qasm_path,
    stab_txt_path,
    order="X-then-Z"   # change to "Z-then-X" if your .txt lists Z-first
    )

    print("Result of ordered ancilla vs stabilizer check:") 
    if  report["ok"]:
        print("Success : ancilla measurements match stabilizers in order.")
        return True
    else:
      for mi in report["mismatches"]:
          print(f"  Mismatch at stabilizer index {mi}")
      return False



def find_bad_locations(qasm_path: str, stab_txt_path: str,num_gates: int):
    results, unsat_gates = forall_fault_exists_low_weight_per_gate(
        qasm_path,
        stab_txt_path,
        fault_mode="2q",
        flag_axis="z",
        flag_prefix="flagErr",
    )

    bad_locations_dict = [] # List to store bad locations for the current circuit
    qc = QuantumCircuit.from_qasm_file(qasm_path)



    for i in range(num_gates):  # Iterate over gates in the subcircuit
        
        circuit_bad_locations = []
        if qc.data[i].name  in ["barier", "measure", "reset"]:
            print(f"Gate index {i}: " , qc.data[i].name )
            continue  # Skip non-unitary gates

        
        state, qc, site_info, groups = build_state_with_fault_after_gate(
            qasm_path,
            gate_index=i,
            fault_mode="2q"
        )
        
        # Extract fault variables
        fault_var = [v for k, v in site_info["vars"].items() if k.startswith("f")]

        # Extract qubit groups
        data_idxs = groups["data"]
        ancz_idxs = groups["ancZ"]
        flagx_idxs = groups["flagX"]
        ancx_idxs = groups["ancX"]
        flagz_idxs = groups["flagZ"]

        # Extract error components
        E_x = [state.qubits[i].x for i in data_idxs]
        E_z = [state.qubits[i].z for i in data_idxs]

        
        # Build stabilizer-equivalent errors
        Epx, Epz, gsel = build_stab_equiv_errors(E_x, E_z, stab_txt_path)

        # Build per-qubit error indicators
        b = [Or(xi, zi) for xi, zi in zip(Epx, Epz)]

        # Create a Z3 solver
        
  
        s = Solver()
        #s.add(ForAll(fault_var, Implies(Or(fault_var), Exists(gsel, PbLe([(bi, 1) for bi in b], 1)))))
        s.add(ForAll(gsel, PbGe([(bi, 1) for bi in b], 2)  ) )
        s.add(Or(fault_var))  # At most one fault
        

    
        
        
        

        # Check satisfiability
        if s.check() == sat : print(f"Gate index {i}: Bad location " )
        else :print(f"Gate index {i}: Safe" )
        if s.check() == sat:
            #print("Bad location found at gate index:", i)
            #print("qc.instructions ", qc.data[i].name, qc.data[i].qubits)
        
            # Store bad locations and gate numbers for the current subcircuit
            bad_locations_dict.append(i)

    # Update the gate count for the next subcircuit

    # Print the results
    if bad_locations_dict != []:

        print("Success : index of bad locations :")
        print(bad_locations_dict)
    else :print("There is no bad locaiton")



    return bad_locations_dict

def check_flag_raised(qasm_path: str, stab_txt_path: str,num_gates: int, bad_locations_dict: List[int]):
    
    
    state, qc, sites_info, groups = build_state_with_faults_after_gates(qasm_path ,bad_locations_dict, fault_mode="2q")    
    #print(sites_info)

    data_idxs = groups["data"]
    ancz_idxs = groups["ancZ"]
    flagx_idxs = groups["flagX"]
    ancx_idxs = groups["ancX"]
    flagz_idxs = groups["flagZ"]


    fault_var = [[v for k,v in s["vars"].items() if k.startswith("f")] for s in sites_info]
    #print("Injected fault variables:", fault_var)


    all_fault_vars = [f for sublist in fault_var for f in sublist]
    #print("All fault variables:", all_fault_vars)
    acts = [s["act"] for s in sites_info]
    
    #print("Data qubits:", data_idxs)
    #print("Ancilla qubits (Z-basis):", anc_idxs)
    #print("Flag qubits (X-basis):", flag_idxs)

    E_x = [state.qubits[i].x for i in data_idxs]
    E_z = [state.qubits[i].z for i in data_idxs]


    F = []
    if groups["flagX"] != []: F.extend([state.qubits[i].z for i in groups["flagX"]])
    if groups["flagZ"] != []: F.extend([state.qubits[i].x for i in groups["flagZ"]])
    
    if  groups["flagX"] == [] and groups["flagZ"] == [] : print("No flag qubits found.")


    

    Epx, Epz, gsel = build_stab_equiv_errors(E_x, E_z,stab_txt_path)
    b= [Or(xi, zi) for xi, zi in zip(Epx, Epz)]

    s = Solver()
    # Add the constraint to the solver
    #s.add(AtMost(*acts,1 )  )
    #s.add(Xor(acts[0],acts[1]))
    #s.add(acts[0] == True)
    #s.add(Exists(all_fault_vars,Implies( Not(Exists(gsel, PbLe([(bi, 1) for bi in b], 1))), Not(Or(F[0],F[1]) )  )))
    
    #s.add(ForAll(all_fault_vars, Implies(Not(Exists(gsel, PbLe([(bi, 1) for bi in b], 1))),Or(*F)  )))# Combine F_z into a single condition if it's a list
    '''
    s.add(ForAll(gsel, Implies(
        And(
            Not(Exists(fault_var, PbLe([(bi, 1) for bi in b], 1))),
            AtMost(*acts,1 ),
        ),
        Or(F)  # Combine F_z into a single condition if it's a list
    )))
    '''
    #s.add(ForAll(gsel, And( PbGe([(bi, 1) for bi in b], 2), AtMost(*acts,1 ), And( Not(Or(F) ))  )  ) )
    #s.add(Exists(all_fault_vars,  ForAll(gsel,And(PbGe( [(bi, 1) for bi in b], 2), And([Not(f) for f in F]))) ) )
    s.add(
        And(
            ForAll(gsel, PbGe([(bi,1) for bi in b], 2)),
            Not(Or(F))
        )
    )
    s.add(AtMost(*acts,1 ))  # At most two faults
   

    #print(s.check())
    if s.check() == unsat : 
        print("Result:")
        print("Success : when high-weight error happens, at least one of the flag qubits raised")

        return True
    if s.check() == sat:
        print("Result:")
        print("Failure : there exists a high-weight error where none of the flag qubits raised ")
        print("Counterexample model:")
        for d in s.model().decls(): 
        
            val = s.model()[d]
            if str(val)  == "True": 
                print(f"{d.name()} = {val}")
                
        return False

def check_generalised_syndrome_uniqueness(
    qasm_path: str,
    stab_txt_path: str,
    bad_locations : List[int]
    ):
    clean_qc = remove_flag_gates(qasm_path, save_path=None)
    #print("Original gates:", len(clean_qc.data))
    print("Processing the flag circuit")
    state, qc, sites_info, groups = build_state_with_faults_after_gates( qasm_path,bad_locations, fault_mode="2q")   

    print("#######################################")
    #print(sites_info)
    data_idxs = groups["data"]
    ancz_idxs = groups["ancZ"]
    flagx_idxs = groups["flagX"]
    ancx_idxs = groups["ancX"]
    flagz_idxs = groups["flagZ"]

    

    after_flag_state_X = [state.qubits[i].x for i in data_idxs]
    after_flag_state_Z = [state.qubits[i].z for i in data_idxs]

    fault_var = [[v for k,v in s["vars"].items() if k.startswith("f")] for s in sites_info]
    gate_fault_constr = [Or(f) for f in fault_var if f != []]



    flag_err_var =  []

    flag_err_var.extend(inject_on_flags(state, flagx_idxs, axis="z", prefix="flagErr"))


    flag_err_var.extend(inject_on_flags(state, flagz_idxs, axis="x", prefix="flagErr"))
    #print("flagz_idxs", flagz_idxs)
    #print("flag_err_var", flag_err_var)
    anc_err_var = [] 

    anc_err_var.extend(inject_on_flags(state, ancx_idxs, axis="z", prefix="ancErr"))
    anc_err_var.extend(inject_on_flags(state, ancz_idxs, axis="x", prefix="ancErr"))

    A = [state.qubits[i].z for i in ancx_idxs] + [state.qubits[i].x for i in ancz_idxs]
    F = [state.qubits[i].z for i in flagx_idxs] + [state.qubits[i].x for i in flagz_idxs]


    
    after_raw_state, snap = symbolic_propagate_with_resets( clean_qc ,state, track_steps= True)

    raw_anc_err_var = []

    raw_anc_err_var.extend(inject_on_flags(state, ancx_idxs, axis="z", prefix="raw_ancErr"))
    raw_anc_err_var.extend(inject_on_flags(state, ancz_idxs, axis="x", prefix="raw_ancErr"))

    all_fault = gate_fault_constr+ flag_err_var + anc_err_var+ raw_anc_err_var

    one_fault_constr = [ And (PbGe( [(f,1) for f in all_fault], 1), PbLe( [(f,1) for f in all_fault], 1))]



    var = [sub for sub in fault_var for sub in sub] + flag_err_var + anc_err_var + raw_anc_err_var

    ren_1 = make_renamer_from_symbols(var, "_p1")
    ren_2 = make_renamer_from_symbols(var, "_p2")

    one_fault_constr_p1 = primed_copy(one_fault_constr, ren_1)
    one_fault_constr_p2 = primed_copy(one_fault_constr, ren_2)

    #print("one_fault_constr_p1", one_fault_constr_p1)
    #print("one_fault_constr_p2", one_fault_constr_p2)

    A_1 = primed_copy(A, ren_1)
    A_2 = primed_copy(A, ren_2)
    F_1 = primed_copy(F, ren_1)
    F_2 = primed_copy(F, ren_2)

    for a in F:
        print("Flag syndrome formula:", simplify(a))
    for i in range(len(F_1)) : 
        print(f"{i}Flag syndrome formula p1:", eval_with_values(F_1[i], {"faulty_gate24_x1_p1" : True, "faulty_gate24_z0_p1" : True}, default_false=True) )
    
    for i in range(len(F_2)) : 
        print(f"{i}Flag syndrome formula p2:", eval_with_values(F_1[i], {"faulty_gate27_z0_p2" : True}, default_false=True))
    
    print("Processing the no flag circuit")

    

    E_x = [after_raw_state.qubits[i].x for i in data_idxs]
    E_z = [after_raw_state.qubits[i].z for i in data_idxs]

    raw_A = [after_raw_state.qubits[i].z for i in ancx_idxs] + [after_raw_state.qubits[i].x for i in ancz_idxs]

    E_x_1 = primed_copy(E_x, ren_1)
    E_z_1 = primed_copy(E_z, ren_1)
    E_x_2 = primed_copy(E_x, ren_2)
    E_z_2 = primed_copy(E_z, ren_2)

    raw_A_1 = primed_copy(raw_A, ren_1)
    raw_A_2 = primed_copy(raw_A, ren_2)

    gen_syn_1 = A_1 + F_1 + raw_A_1
    gen_syn_2 = A_2 + F_2 + raw_A_2
    
    p1 = {"faulty_gate5_z0_p1" : True, "faulty_gate5_x1_p1" : True,  "faulty_gate5_x0_p1" : True}
    p2 = {"faulty_gate11_z0_p2" : True, "faulty_gate11_x0_p2" : True}
    for i in range(len(A_1)) :
        print(f"{i} Ancilla syndrome formula p1:", eval_with_values(A_1[i], p1, default_false=True) )

    for i in range((len(F_1))) :
        print(f"{i} Flag syndrome formula p1:", eval_with_values( F_1[i], p1, default_false=True) )

    for i in range(len(raw_A_1)) :
        print(f"{i} Raw Ancilla syndrome formula p1:", eval_with_values( raw_A_1[i], p1, default_false=True) )

    for i in range(len(A_2)) :
        print(f"{i} Ancilla syndrome formula p2:", eval_with_values(A_2[i], p2, default_false=True) )
    for i in range((len(F_2))) :
        print(f"{i} Flag syndrome formula p2:", eval_with_values( F_2[i], p2, default_false=True) )

    for i in range(len(raw_A_2)) :
        print(f"{i} Raw Ancilla syndrome formula p2:", eval_with_values( raw_A_2[i], p2, default_false=True) )

    
    stab_eq , gsel = exists_stab_equiv(E_x_1, E_z_1, E_x_2, E_z_2, stab_txt_path)




    same_syn =  And( *[x == y for x, y in zip(gen_syn_1, gen_syn_2)] )

    s = Solver()    
    s.add(same_syn, Not(Exists(gsel, stab_eq)))
    s.add(one_fault_constr_p1)
    s.add(one_fault_constr_p2)


    print("Result:")
    if s.check() == unsat :
       
        print("Success: every error maps to different generalised syndrome")

        return True 

    if s.check() == sat:
        print("Failure: there exists two different errors that map to the same generalised syndrome")
        print("The model that would cause different errors map to the same generalised syndrome:")
        for d in s.model().decls(): 
    
            val = s.model()[d]
            if str(val)  == "True": 
                print(f"{d.name()} = {val}")

        return False
    


def main():
    config = read_config()
    qasm_path = Path(config["qasm_path"])
    stab_txt_path = Path(config["stab_txt_path"])

    qc = QuantumCircuit.from_qasm_file(str(qasm_path))
    num_gates = sum(1 for inst in qc.data )
    #print(f"Number of gates in the circuit: {num_gates}")

    print("Step 1: Proving the circuit is etracts syndrome correctly when no fault in the circuit")
    step_1 = prove_syndrome_extractions(str(qasm_path), str(stab_txt_path))

    print("\nStep 2: Finding bad locations in the circuit")

    bad_location = find_bad_locations(str(qasm_path), str(stab_txt_path),num_gates)

    
    print("\nStep 3: Checking if flag is raised when high weight error occurs")
    step_3 = check_flag_raised(str(qasm_path), str(stab_txt_path),num_gates,bad_location)

    print("\nStep 4: Checking two non-degenerate error dont map to same generalised syndrome")
    step_4 = check_generalised_syndrome_uniqueness(str(qasm_path), str(stab_txt_path),bad_location)

    if step_1 and step_3 and step_4 :
        print("\nOverall Result: The flag circuit passes all the checks!")
        return True
    else :
        print("\nOverall Result: The flag circuit fails one or more checks.")
        return False


    


if __name__ == "__main__":
    main()
       