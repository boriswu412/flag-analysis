# SAT‑based Flag Qubit Circuit Verification Tool

A tool to verify flag‑based stabilizer extraction circuits using SAT checks.

---

## Installation

Clone the repository and install requirements:

```bash
git clone https://github.com/boriswu412/flag-analysis.git
cd flag-analysis
pip install -r requirements.txt
```

(Optional) create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
```

> Requires Python 3.9+ (tested with 3.10).

---

## Usage

Edit `config.txt` with two rows:
```
qasm_path=path/to/circuit.qasm
stab_txt_path=path/to/stab.txt
```

Then run:
```bash
python flag_analysis.py
```

### QASM file
**Qubits:**
- `q` — data qubits
- `ancX`, `ancZ` — ancilla qubits in X or Z basis
- `flagX`, `flagZ` — flag qubits in X or Z basis

**Gates:**
- Supported: `cx` (CNOT), `cz`
- Hadamard gates on ancillas or flags are replaced by basis changes (X ↔ Z)
- Measurements are omitted from QASM
- Reused qubits in flag circuits are treated as distinct  qubits

### `stab.txt`
Defines the parity‑check matrix (first X, then Z rows). Row order must match ancilla order.

---

## Running and Output

The verification includes **four steps**:

1. **Syndrome correctness:** verify when no faults in the flag circuit the circuit is a syndrome extraction circuit.
2. **Fault detection:** find gates that cause high‑weight errors → stored in `bad_location`.
3. **Flag check:** verify faults on bad gates trigger at least one flag.
4. **Syndrome uniqueness:** check after a round of flag circuit and a round of no flag circuit the generalized syndrome is unique up to degenracy.

Each step should output **“Success.”**

If Step 3 or 4 fails, a counterexample is printed showing which variables are `True`.

Example (Step 3):
```
faulty_gate4_z1 = True
faulty_gate4_x0 = True
```
Gate 4 (`cx [2,7]`): fault with X on 2 and Z on 7 causes a high‑weight error but no flag.

Example (Step 4):
```
faulty_gate19_z1_p2 = True
faulty_gate14_z1_p1 = True
faulty_gate19_z0_p2 = True
faulty_gate19_x0_p2 = True
faulty_gate14_x0_p1 = True
```
Gates 19 (`cx [8,1]`) YZ error  and 14 (`cx [7,2]`) XZ error: these faults yield identical generalized syndromes, violating uniqueness.

