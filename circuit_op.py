from qiskit import QuantumCircuit
from qiskit.qasm2 import dumps as qasm2_dumps



import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
try:
    # Some 2.x installs expose a dedicated qasm2 loader
    from qiskit.qasm2 import loads as qasm2_loads  # type: ignore
    _HAS_QASM2 = True
except Exception:
    _HAS_QASM2 = False



def load_qasm(qasm_path: str) -> QuantumCircuit:
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

def get_gate_only_indices(qc):
    """
    Return a list of indices in qc.data corresponding ONLY to real gates.
    Skips: measure, barrier, reset, id.
    """
    skip = {"measure", "barrier", "reset", "id"}
    gate_indices = []

    for i, inst in enumerate(qc.data):
        if inst.operation.name.lower() not in skip:
            gate_indices.append(i)

    return gate_indices


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

from qiskit import QuantumCircuit, QuantumRegister  # make sure QuantumRegister is imported
from qiskit.qasm2 import dumps as qasm2_dumps

def remove_flag_gates(qasm_path: str, save_path: str = None):
    """
    Load a QASM file and produce a new circuit where:
      - All flag qubit registers (flagX*, flagZ*) are removed.
      - All non-barrier gates that act on flag qubits are removed.
      - Barriers are kept, but their flag-qubit arguments are removed.

    If a barrier only had flag qubits, it is dropped.
    """
    qc = QuantumCircuit.from_qasm_file(qasm_path)

    # Helper: which qregs are flag registers?
    def is_flag_reg(qreg):
        n = qreg.name.lower()
        return n.startswith("flagx") or n.startswith("flagz")

    # Keep only non-flag qregs in new circuit
    kept_qregs = [q for q in qc.qregs if not is_flag_reg(q)]
    new_qc = QuantumCircuit(*kept_qregs, *qc.cregs)

    # Old qubit -> (reg_name, local_index)
    bit_lookup = _build_bit_lookup(qc)

    # Map reg_name -> qreg in the new circuit
    new_reg_by_name = {qreg.name: qreg for qreg in new_qc.qregs}

    # Helper: is a given original qubit a flag qubit?
    def is_flag_qubit(qbit):
        reg_name, _ = bit_lookup[qbit]
        reg_name_l = reg_name.lower()
        return reg_name_l.startswith("flagx") or reg_name_l.startswith("flagz")

    # Walk instructions in the original circuit
    for instr, qargs, cargs in qc.data:
        name = instr.name

        if name == "barrier":
            # Keep barriers, but strip out flag qubits
            kept_qargs = []
            for q in qargs:
                if is_flag_qubit(q):
                    continue  # drop flag qubit from barrier
                reg_name, local_idx = bit_lookup[q]
                new_qreg = new_reg_by_name[reg_name]
                kept_qargs.append(new_qreg[local_idx])

            # If there are remaining non-flag qubits, add a barrier on them
            if kept_qargs:
                new_qc.barrier(*kept_qargs)
            # If no non-flag qubits remain, drop this barrier entirely
            continue

        # For non-barrier gates: if they touch any flag qubit, drop the gate
        if any(is_flag_qubit(q) for q in qargs):
            continue

        # Otherwise, remap qubits and keep the gate
        new_qargs = []
        for q in qargs:
            reg_name, local_idx = bit_lookup[q]
            new_qreg = new_reg_by_name[reg_name]
            new_qargs.append(new_qreg[local_idx])

        new_qc.append(instr, new_qargs, cargs)

    # Optionally save the modified circuit
    if save_path:
        with open(save_path, "w") as f:
            f.write(qasm2_dumps(new_qc))

    return new_qc


from pathlib import Path

def paulistring_to_qasm(pauli: str,
                        anc_name: str = "ancX",
                        anc_idx: int = 0,
                        data_name: str = "q",
                        barrier_after: bool = True,
                        save_path: str | None = None) -> str:
    """
    Generate OPENQASM 2.0 that couples one ancilla to each data qubit according
    to a Pauli string (I/X/Y/Z). Uses cx/cy/cz with the ancilla as the first operand.
    Optionally saves the output to `save_path`.
    """
    pauli = pauli.strip().upper()
    n = len(pauli)
    if n != 5:
        raise ValueError(f"expected length-5 pauli string, got length {n}: {pauli}")
    if any(c not in "IXYZ" for c in pauli):
        raise ValueError(f"pauli string must use only I/X/Y/Z, got: {pauli}")

    gate_for = {"X": "cx", "Y": "cy", "Z": "cz"}  # I is skipped

    lines = []
    lines += [
        "OPENQASM 2.0;",
        'include "qelib1.inc";',
        "",
        f"qreg {data_name}[{n}];          // data qubits",
        f"qreg {anc_name}[1];       // ancilla",
        "",
        "// =========================",
        f"// Stabilizer : ( {pauli} )",
        f"// ancilla = {anc_name}[{anc_idx}]",
        "// =========================",
    ]

    for i, p in enumerate(pauli):
        if p == "I":
            continue
        g = gate_for[p]
        lines.append(f"{g}  {anc_name}[{anc_idx}], {data_name}[{i}];")

    if barrier_after:
        lines.append("")
        lines.append("barrier;")

    qasm = "\n".join(lines)

    if save_path is not None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(qasm)

    return qasm
