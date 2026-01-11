"""
dimacs_bridge.py

A small, practical bridge for:
  Z3 Bool formula  ->  CNF (Tseitin via Z3)  ->  DIMACS CNF
  CryptoMiniSat model (DIMACS ints)          ->  assignment on *your original Z3 vars*

Key idea:
  DIMACS export introduces auxiliary variables (Tseitin). That's fine.
  To interpret a SAT model back in Z3 terms, you MUST keep a mapping:
      DIMACS variable id  <->  Z3 BoolRef

This file provides:
  - build_dimacs(phi): returns (dimacs_str, var2id, id2var, orig_vars)
  - parse_dimacs_model(text): returns set of signed ints from a SAT solver output
  - model_to_assignment(lits, id2var, orig_vars): returns dict {BoolRef: bool} restricted to orig_vars
"""

from __future__ import annotations
from typing import Dict, List, Tuple, Set, Iterable, Optional
import os, re, shutil, subprocess


from z3 import (
    BoolRef, BoolVal, Not, is_true, is_false, simplify, substitute,
    is_not, is_or, is_and, is_const, Goal, Tactic, Then
)

# ---------------------------
# Symbol collection
# ---------------------------

def _is_bool_var(e: BoolRef) -> bool:
    """
    True iff e looks like a boolean variable (uninterpreted 0-arity Bool).
    This excludes compounds like And(...), Xor(...), etc.
    """
    if not isinstance(e, BoolRef):
        return False
    if not is_const(e):
        return False
    # is_const includes True/False; exclude those
    if is_true(e) or is_false(e):
        return False
    return True


def collect_bool_symbols(expr: BoolRef) -> Set[BoolRef]:
    """Collect boolean variables appearing in expr."""
    seen: Set[BoolRef] = set()
    stack = [expr]
    while stack:
        e = stack.pop()
        if isinstance(e, BoolRef):
            if _is_bool_var(e):
                seen.add(e)
            for c in e.children():
                stack.append(c)
    return seen

def is_user_var(name: str) -> bool:
    return (
        name.startswith("r_") or
        name.startswith("s_") or
        name.startswith("f_")
    )

# ---------------------------
# CNF extraction (via Z3)
# ---------------------------

def _flatten_or(e: BoolRef) -> List[BoolRef]:
    """Return a flat list of literals from an Or tree (or [e] if not Or)."""
    if is_or(e):
        out = []
        for c in e.children():
            out.extend(_flatten_or(c))
        return out
    return [e]


def _is_lit(e: BoolRef) -> bool:
    """A CNF literal: v or Not(v) where v is a boolean var, or True/False."""
    if is_true(e) or is_false(e):
        return True
    if _is_bool_var(e):
        return True
    if is_not(e) and _is_bool_var(e.children()[0]):
        return True
    return False


def _clause_to_lits(cl: BoolRef) -> List[BoolRef]:
    """
    Convert a CNF clause into list of literals.
    Clause can be:
      - Or(lit, lit, ...)
      - lit
    """
    cl = simplify(cl)
    if is_true(cl):
        return [BoolVal(True)]  # tautology clause
    if is_false(cl):
        return [BoolVal(False)]  # empty/unsat clause representation

    if is_or(cl):
        lits = _flatten_or(cl)
    else:
        lits = [cl]

    # sanity check: Z3's tseitin-cnf should give us literals here
    for l in lits:
        if not _is_lit(l):
            raise ValueError(f"Non-literal in CNF clause: {l}")
    return lits


def to_cnf_clauses(
    phi: BoolRef,
    *,
    use_pb2bv: bool = False,
    use_card2bv: bool = False,
) -> List[List[BoolRef]]:
    """
    Use Z3 tactics to produce CNF clauses for `phi`.

    If your formula uses pseudo-Boolean constraints (PbEq/PbLe/PbGe) or
    cardinality constraints (AtMost/AtLeast), Z3 may print DIMACS that
    depends on those higher-level constructs.

    Setting:
      - use_pb2bv=True  enables the `pb2bv` lowering pass (handles PbEq/PbLe/PbGe)
      - use_card2bv=True enables the `card2bv` lowering pass (handles AtMost/AtLeast)

    Returns:
      list of CNF clauses; each clause is a list of literals (BoolRef or Not(BoolRef)).
    """
    g = Goal()
    g.add(phi)

    # Build tactic pipeline.
    steps = [
        "simplify",
        "propagate-values",
        "solve-eqs",
        "elim-uncnstr",
    ]
    if use_pb2bv:
        steps.append("pb2bv")
    if use_card2bv:
        steps.append("card2bv")

    # `bit-blast` is harmless for pure Bool, and helpful if pb/card lowering
    # introduces bit-vectors.
    steps.extend(["bit-blast", "tseitin-cnf"])

    cnf_goal = Then(*steps)(g)

    # cnf_goal is a Goal; turn it into a list of clause expressions
    e = simplify(cnf_goal.as_expr())
    fs = e.children() if is_and(e) else [e]

    clauses: List[List[BoolRef]] = []
    for f in fs:
        f = simplify(f)
        if is_true(f):
            continue
        clauses.append(_clause_to_lits(f))
    return clauses


# ---------------------------
# DIMACS printing
# ---------------------------

def _lit_to_dimacs_int(lit: BoolRef, var2id: Dict[BoolRef, int]) -> int:
    """Convert a literal to signed DIMACS int using var2id."""
    lit = simplify(lit)
    if is_true(lit):
        # Clause is satisfied; caller should typically drop whole clause, but keep safe:
        return 0
    if is_false(lit):
        # False literal: represent impossible literal by 0 sentinel (caller will handle)
        return 0

    if _is_bool_var(lit):
        return var2id[lit]
    if is_not(lit) and _is_bool_var(lit.children()[0]):
        return -var2id[lit.children()[0]]

    raise ValueError(f"Not a DIMACS literal: {lit}")



# ---------------------------
# SAT solver output parsing
# ---------------------------

def parse_dimacs_model(text: str) -> Set[int]:
    """
    Parse CryptoMiniSat-style output.
    Returns a set of signed ints (literals) that are assigned True in the model.
    Typical formats:
      - lines starting with 'v ' followed by ints ending with 0
      - may include 's SATISFIABLE'
    """
    lits: Set[int] = set()
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("v") or line.startswith("V"):
            parts = line[1:].strip().split()
            for p in parts:
                try:
                    k = int(p)
                except ValueError:
                    continue
                if k == 0:
                    continue
                lits.add(k)
    return lits


def model_to_assignment(
    lits_true: Set[int],
    id2var: Dict[int, BoolRef],
    orig_vars: Optional[Set[BoolRef]] = None,
    default_false: bool = True,
) -> Dict[BoolRef, bool]:
    """
    Turn DIMACS model literals into an assignment on Z3 BoolRefs.

    - lits_true: set of signed ints (positive means var=True, negative means var=False)
    - id2var: DIMACS id -> BoolRef mapping from build_dimacs
    - orig_vars: if provided, restrict returned assignment to these vars only

    If a variable is missing from the SAT output:
      - default_false=True  -> treat as False (common)
      - default_false=False -> omit it
    """
    # Build a map id -> bool
    id_val: Dict[int, bool] = {}
    for lit in lits_true:
        vid = abs(lit)
        val = (lit > 0)
        # If both +x and -x appear, that's inconsistent solver output; last wins.
        id_val[vid] = val

    out: Dict[BoolRef, bool] = {}
    for vid, var in id2var.items():
        if orig_vars is not None and var not in orig_vars:
            continue
        if vid in id_val:
            out[var] = id_val[vid]
        else:
            if default_false:
                out[var] = False
    return out


# ---------------------------
# Convenience: apply assignment to Z3 formula
# ---------------------------

def eval_z3_bool(expr: BoolRef, assignment: Dict[BoolRef, bool], default_false: bool = True) -> bool:
    """
    Evaluate a Z3 Bool expr under a BoolRef->bool assignment.
    Missing vars default to False if default_false=True.
    """
    syms = collect_bool_symbols(expr)
    subs = []
    for s in syms:
        if s in assignment:
            subs.append((s, BoolVal(assignment[s])))
        elif default_false:
            subs.append((s, BoolVal(False)))
    return is_true(simplify(substitute(expr, subs)))


__all__ = [
    "collect_bool_symbols",
    "to_cnf_clauses",
    "build_dimacs",
    "parse_dimacs_model",
    "model_to_assignment",
    "eval_z3_bool",
]

# dimacs_bridge.py
# ----------------
# Z3 -> DIMACS -> CryptoMiniSat -> back to Z3 vars

import os, re, shutil, subprocess
from typing import Dict, List, Optional, Tuple
from z3 import BoolRef, Goal, Then




# --------------------------------------------------
# Z3 Goal → DIMACS (with var map)
# --------------------------------------------------
def build_dimacs(goal: Goal, use_card2bv: bool):
    t = Then(
        "simplify",
        "propagate-values",
        "solve-eqs",
        "elim-uncnstr",
        "pb2bv",
        "card2bv" if use_card2bv else "skip",
        "bit-blast",
        "tseitin-cnf",
    )
    subgoals = list(t(goal))

    dimacs_files = []
    var_maps = []

    for i, sg in enumerate(subgoals):
        dimacs = sg.dimacs()
        path = f"subgoal_{i}.cnf"
        with open(path, "w") as f:
            f.write(dimacs)

        # extract variable map from comments
        var_map = {}
        for line in dimacs.splitlines():
            if line.startswith("c var"):
                _, _, name, num = line.split()
                var_map[int(num)] = name

        dimacs_files.append(path)
        var_maps.append(var_map)

    return dimacs_files, var_maps


def resolve_cryptominisat_binary(cms_bin: str = "cryptominisat5") -> str:
    """Resolve CryptoMiniSat binary path or raise a helpful error."""
    cms_exec = cms_bin
    if os.path.sep not in cms_exec:
        found = shutil.which(cms_exec)
        if found is None and cms_exec == "cryptominisat5":
            for cand in ("/opt/homebrew/bin/cryptominisat5", "/usr/local/bin/cryptominisat5"):
                if os.path.exists(cand) and os.access(cand, os.X_OK):
                    found = cand
                    break
        if found is None:
            raise FileNotFoundError(
                "CryptoMiniSat binary not found: 'cryptominisat5'.\n"
                "Install it on macOS with: brew install cryptominisat\n"
                "Then restart your terminal/Jupyter so PATH updates, and verify with: which cryptominisat5\n"
                "Or pass cms_bin='/opt/homebrew/bin/cryptominisat5' (Apple Silicon) or '/usr/local/bin/cryptominisat5' (Intel)."
            )
        return found
    if not (os.path.exists(cms_exec) and os.access(cms_exec, os.X_OK)):
        raise FileNotFoundError(f"CryptoMiniSat binary path is not executable: {cms_exec}")
    return cms_exec


import subprocess, re
from typing import Optional, List, Tuple

def run_cryptominisat(
    cnf_path: str,
    cms_exec: str,
    timeout_s: Optional[float] = None,
    cms_extra_args: Optional[list] = None,
) -> Tuple[str, Optional[List[int]], str]:
    """
    Returns (status, model_lits, out_str) with out_str CLEAN:
      - only 's ...' and 'v ...' lines (drops all 'c ...' stats)
    """
    cmd = [cms_exec]
    if cms_extra_args:
        cmd += list(map(str, cms_extra_args))
    cmd.append(cnf_path)

    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        return "unknown", None, "s UNKNOWN\n"

    raw = (p.stdout or "") + "\n" + (p.stderr or "")

    kept_lines = []
    model_lits: List[int] = []

    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("c "):
            continue  # DROP stats/comments
        if line.startswith("s "):
            kept_lines.append(line)
        if line.startswith("v "):
            kept_lines.append(line)
            for tok in line.split()[1:]:
                if tok == "0":
                    continue
                if re.fullmatch(r"-?\d+", tok):
                    model_lits.append(int(tok))

    status = "unknown"
    if any("UNSAT" in ln for ln in kept_lines):
        status = "unsat"
        model_lits = []
    elif any("SAT" in ln for ln in kept_lines):
        status = "sat"

    clean_out = "\n".join(kept_lines) + ("\n" if kept_lines else "")
    return status, (model_lits if model_lits else None), clean_out


def _parse_z3_dimacs_varmap(dimacs_text: str) -> Dict[int, str]:
    """Parse Z3's DIMACS comments of the form: 'c <id> <name>' (best-effort)."""
    vmap: Dict[int, str] = {}
    for line in dimacs_text.splitlines():
        line = line.strip()
        if not line.startswith("c"):
            continue
        parts = line.split()
        if len(parts) < 3:
            continue
        # common format: c <int> <name>
        try:
            idx = int(parts[1])
        except ValueError:
            continue
        name = parts[2]
        vmap[idx] = name
    return vmap

from z3 import Goal, Then, Bool
def build_dimacs(goal: Goal, use_card2bv: bool = True) -> Tuple[List[str], List[Dict[int, object]]]:
    """Convert a Z3 Goal to one-or-more CNF subgoals, write DIMACS files, and return per-subgoal var maps.

    Returns:
      cnf_files: list of written CNF paths
      var_maps:  list of {dimacs_int -> z3.BoolRef} maps (best-effort)
    """
    # tactic pipeline
    t = Then(
        "simplify",
        "propagate-values",
        "solve-eqs",
        "elim-uncnstr",
        "pb2bv",
        "card2bv",
        "bit-blast",
        "tseitin-cnf",
    ) if use_card2bv else Then(
        "simplify",
        "propagate-values",
        "solve-eqs",
        "elim-uncnstr",
        "pb2bv",
        "bit-blast",
        "tseitin-cnf",
    )

    subgoals = list(t(goal))
    if not subgoals:
        return [], []

    cnf_files: List[str] = []
    var_maps: List[Dict[int, object]] = []

    for i, sg in enumerate(subgoals):
        # Z3 Goal supports .dimacs() once in CNF form (best-effort)
        dimacs = sg.dimacs()
        cnf_path = f"uniq_sub{i}.cnf"
        with open(cnf_path, "w") as f:
            f.write(dimacs)
        cnf_files.append(cnf_path)

        # build a best-effort mapping dimacs_int -> Bool(name)
        name_map = _parse_z3_dimacs_varmap(dimacs)
        vmap: Dict[int, object] = {k: Bool(v) for k, v in name_map.items()}
        var_maps.append(vmap)

    return cnf_files, var_maps


def model_to_z3_assignment(model_lits: List[int], var_map: Dict[int, object]) -> Dict[str, bool]:
    """Map DIMACS model literals back to Z3 Bool names using a dimacs->Bool map."""
    out: Dict[str, bool] = {}
    for lit in model_lits:
        v = abs(int(lit))
        if v not in var_map:
            continue
        b = var_map[v]
        try:
            name = b.decl().name()
        except Exception:
            name = str(b)
        out[name] = (lit > 0)
    return out


def pretty_print_z3_assignment(assign: Dict[str, bool], *, only_prefixes: Optional[Tuple[str, ...]] = None) -> str:
    """Pretty print mapped assignment; optionally filter by variable-name prefixes."""
    items = sorted(assign.items(), key=lambda kv: kv[0])
    if only_prefixes:
        items = [(k, v) for (k, v) in items if k.startswith(only_prefixes)]
    return "\n".join(f"{k} = {v}" for k, v in items) + ("\n" if items else "")

def pretty_print_true_z3_vars(model_lits, var_map, *, strip_suffix=True, do_print= False):
    """
    From DIMACS model literals + var_map, print only user vars that are True,
    and return two dicts:
        p1_dict: {base_name: True}
        p2_dict: {base_name: True}

    Assumes:
      - model_to_z3_true_vars(model_lits, var_map) -> iterable[str] or iterable[z3 Bool] names
      - is_user_var(name: str) -> bool  (filters out k!, and!, at-most-..., etc.)

    strip_suffix:
      - If True, store keys without the '_p1'/'_p2' suffix.
      - If False, store full var names as keys.
    """
    true_vars = model_to_z3_true_vars(model_lits, var_map) or []

    p1_dict = {}
    p2_dict = {}

    for v in true_vars:
        # normalize to string name
        name = v.decl().name() if hasattr(v, "decl") else str(v)

        # filter internals
        if not is_user_var(name):
            continue

        if name.endswith("_p1"):
            key = name[:-3] if strip_suffix else name
            p1_dict[key] = True
        elif name.endswith("_p2"):
            key = name[:-3] if strip_suffix else name
            p2_dict[key] = True
        else:
            # if you ever want "other" vars, you can add a third dict here
            pass

    if do_print:
        if not p1_dict and not p2_dict:
            print("(no user variables are True)")
        else:
            if p1_dict:
                print("=== p1 True vars ===")
                for k in sorted(p1_dict):
                    print(k if strip_suffix else f"{k}")
            else:
                print("=== p1 True vars ===")
                print("(none)")

            if p2_dict:
                print("=== p2 True vars ===")
                for k in sorted(p2_dict):
                    print(k if strip_suffix else f"{k}")
            else:
                print("=== p2 True vars ===")
                print("(none)")

    return p1_dict, p2_dict
def model_to_z3_true_vars(model_lits, var_map):
    """
    model_lits : list[int]      # e.g. [1, -2, 3, -4]
    var_map    : dict[int, BoolRef]  # DIMACS id -> Z3 Bool

    Returns:
        dict[str, bool] mapping variable name -> True
        (only includes variables assigned True)
    """
    true_vars = {}

    for lit in model_lits:
        if lit > 0:  # TRUE in DIMACS
            v = var_map.get(lit)
            if v is not None:
                true_vars[v.decl().name()] = True

    return true_vars

