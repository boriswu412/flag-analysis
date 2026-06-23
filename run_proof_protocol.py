#!/usr/bin/env python3
"""Run proof_protocol from the command line (no Jupyter)."""

import argparse
import sys
from pathlib import Path

from flag_analysis import load_symplectic_txt, new_clean_circuit_state, read_config, set_quiet
from protocol import load_protocol
from proof_protocol import proof_protocol


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run multi-path protocol verification via proof_protocol.",
    )
    parser.add_argument("--protocol", required=True, help="Path to protocol JSON")
    parser.add_argument("--config", required=True, help="Path to protocol config .txt")
    parser.add_argument(
        "--t",
        type=int,
        default=1,
        help="Max faults per path (default: 1)",
    )
    parser.add_argument(
        "--metrics-dir",
        default=None,
        help="Directory for proof metrics TXT (default: next to config file)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose proof_protocol output",
    )
    args = parser.parse_args()

    try:
        set_quiet(args.quiet)
        start_node, protocol = load_protocol(args.protocol)
        config = read_config(args.config)
        if args.metrics_dir:
            config["metrics_dir"] = str(Path(args.metrics_dir).resolve())
        if args.quiet:
            config["__quiet__"] = True
        gen = load_symplectic_txt(str(config["stab_txt_path"]))
        init_state = new_clean_circuit_state(len(gen[0][0]))
        _, stats = proof_protocol(protocol, start_node, init_state, config, args.t)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    sat_paths = [row for row in stats if row.get("status") == "sat"]
    if sat_paths:
        if not args.quiet:
            print(
                f"FAILED: {len(sat_paths)} path(s) returned SAT (counterexample found).",
                file=sys.stderr,
            )
        return 1

    if not args.quiet:
        verified = [row for row in stats if row.get("status") == "unsat"]
        print(f"SUCCESS: all {len(verified)} verified path(s) are UNSAT.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
