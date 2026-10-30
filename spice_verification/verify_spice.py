#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# Numerical experiments for
#   A. Ashrafi, "A Minimum-Self-Conductance Characterization of Kron Reduction
#   in Resistive Networks," IEEE Transactions on Circuits and Systems I:
#   Regular Papers (accepted, 2026).
#
# Author:    Ashkan Ashrafi, Department of Electrical and Computer Engineering,
#            San Diego State University, San Diego, CA, USA (ashrafi@sdsu.edu)
# Copyright: (c) 2026 Ashkan Ashrafi. Released under the MIT License; see the
#            LICENSE file in this repository.
# -----------------------------------------------------------------------------
"""Run four saved ngspice netlists and check the paper's terminal equivalence.

Python standard library only. Run: python3 verify_spice.py
The conductance matrices printed in the manuscript (Eqs. (42)-(45) of the
revised version) are embedded below. The netlists are read, never rewritten.
Generated simulator outputs, logs, and JSON are replaced in spice_results on
each run.
"""

from datetime import datetime, timezone
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "spice_results"
CASES = {
    **{f"basis{k + 1}": [int(j == k) for j in range(5)] for k in range(5)},
    "mixed": [1, -0.5, 2, 0.25, -1],
    "common": [1, 1, 1, 1, 1],
}
TOL = 1e-10  # Absolute tolerances: A for current, S for matrix, W for power.

# Matrices as printed in the manuscript. Kron matrices are integer numerators
# over the denominators 46 (shunt-free) and 70 (shunted).
PAPER_MATRICES = {
    "eq:Gexample": [[3,-2,0,0,0,-1,0,0], [-2,7,-3,0,0,0,-2,0], [0,-3,4,-1,0,0,0,0],
                    [0,0,-1,4,-2,0,0,-1], [0,0,0,-2,5,0,0,-3], [-1,0,0,0,0,2,-1,0],
                    [0,-2,0,0,0,-1,5,-2], [0,0,0,-1,-3,0,-2,6]],
    "eq:Gshunted": [[3,-2,0,0,0,-1,0,0], [-2,8,-3,0,0,0,-2,0], [0,-3,4,-1,0,0,0,0],
                    [0,0,-1,4,-2,0,0,-1], [0,0,0,-2,5,0,0,-3], [-1,0,0,0,0,2,-1,0],
                    [0,-2,0,0,0,-1,7,-2], [0,0,0,-1,-3,0,-2,6]],
    "eq:GammaExample": [[112,-104,0,-2,-6], [-104,274,-138,-8,-24], [0,-138,184,-46,0],
                        [-2,-8,-46,175,-119], [-6,-24,0,-119,149]],
    "eq:GammaShunted": [[172,-152,0,-2,-6], [-152,512,-210,-8,-24], [0,-210,280,-70,0],
                        [-2,-8,-70,267,-179], [-6,-24,0,-179,233]],
}


def matrix_after(source, label, denominator=1):
    tail = source.split(r"\label{" + label + "}", 1)[1]
    body = tail.split(r"\begin{bmatrix}", 1)[1].split(r"\end{bmatrix}", 1)[0]
    body = re.sub(r"\\phantom\{[^}]*\}", "", body)
    return [[F(x.strip()) / denominator for x in row.split("&")]
            for row in body.split(r"\\")]


def inverse(a):
    n = len(a)
    x = [list(row) + [F(i == j) for j in range(n)] for i, row in enumerate(a)]
    for k in range(n):
        pivot = next(j for j in range(k, n) if x[j][k])
        x[k], x[pivot] = x[pivot], x[k]
        scale = x[k][k]
        x[k] = [v / scale for v in x[k]]
        for j in range(n):
            if j != k:
                scale = x[j][k]
                x[j] = [v - scale * w for v, w in zip(x[j], x[k])]
    return [row[n:] for row in x]


def schur(g):
    ui = inverse([row[5:] for row in g[5:]])
    return [[g[i][j] - sum(g[i][5 + k] * ui[k][l] * g[5 + l][j]
                          for k in range(3) for l in range(3))
             for j in range(5)] for i in range(5)]


def read_resistors(path, n):
    """Reconstruct conductances from actual saved SPICE resistor values."""
    g = [[F(0) for _ in range(n)] for _ in range(n)]
    resistors = []
    for line in path.read_text().splitlines()[1:]:
        if not line or line[0].lower() != "r":
            continue
        name, a, b, value = line.split()
        a, b = int(a), int(b)
        resistance = F(value.strip("{}"))
        if resistance <= 0:
            raise ValueError(f"Nonpositive resistor: {name}")
        conductance = 1 / resistance
        resistors.append((name, a, b, float(resistance)))
        for node in [a, b]:
            if node:
                g[node - 1][node - 1] += conductance
        if a and b:
            g[a - 1][b - 1] -= conductance
            g[b - 1][a - 1] -= conductance
    return g, resistors


def read_op(path):
    text = path.read_text()
    if "Flags: real" not in text or "No. Points: 1\n" not in text:
        raise ValueError(f"Expected one real operating point: {path}")
    variables = text.split("Variables:\n", 1)[1].split("Values:\n", 1)[0]
    names = [line.split()[1].lower() for line in variables.splitlines() if line.strip()]
    lines = [line.split() for line in text.split("Values:\n", 1)[1].splitlines()
             if line.strip()]
    values = [float(line[-1]) for line in lines]
    if len(names) != len(values):
        raise ValueError(f"Raw variable/value mismatch: {path}")
    return dict(zip(names, values))


def maximum_difference(a, b):
    return max(abs(x - y) for x, y in zip(a, b))


def floats(a):
    return [[float(x) for x in row] for row in a]


def main():
    ngspice = shutil.which("ngspice")
    if not ngspice:
        raise SystemExit("ngspice is not on PATH. Install it, then rerun this script.")
    version = subprocess.check_output([ngspice, "--version"], text=True).strip()
    OUT.mkdir(exist_ok=True)
    result = {
        "run_utc": datetime.now(timezone.utc).isoformat(),
        "ngspice_executable": ngspice,
        "ngspice_version": version,
        "reference_matrices": "embedded from the manuscript, Eqs. (42)-(45)",
        "absolute_tolerance": TOL,
        "current_convention": "q[k] = -i(Vdrivek), source positive terminal at node k",
        "variants": {},
    }
    all_pass = True
    for variant, full_label, kron_label, denominator in [
        ("shunt_free", "eq:Gexample", "eq:GammaExample", 46),
        ("shunted", "eq:Gshunted", "eq:GammaShunted", 70),
    ]:
        full = [[F(x) for x in row] for row in PAPER_MATRICES[full_label]]
        gamma = [[F(x, denominator) for x in row] for row in PAPER_MATRICES[kron_label]]
        if schur(full) != gamma:
            raise AssertionError(f"Printed Schur complement mismatch: {variant}")
        observed = {}
        netlist_hashes = {}
        for model, reference in [("original", full), ("kron", gamma)]:
            stem = f"eight_node_{variant}_{model}"
            netlist = ROOT / (stem + ".cir")
            g, resistors = read_resistors(netlist, len(reference))
            if g != reference:
                raise AssertionError(f"Netlist does not match every printed matrix entry: {netlist}")
            netlist_hashes[netlist.name] = hashlib.sha256(netlist.read_bytes()).hexdigest()
            for case in CASES:
                (OUT / f"{stem}_{case}.raw").unlink(missing_ok=True)
            cmd = [ngspice, "-n", "-b", "-o", str(OUT / (stem + ".log")), str(netlist)]
            run = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=60)
            (OUT / (stem + ".stdout.txt")).write_text(run.stdout + run.stderr)
            log = (OUT / (stem + ".log")).read_text()
            if run.returncode or re.search(r"(?im)^.*\b(error|fatal|singular matrix|warning)\b", log):
                raise RuntimeError(f"ngspice failed or warned: {netlist.name}\n{log}")
            observed[model] = {}
            for case, imposed in CASES.items():
                raw = read_op(OUT / f"{stem}_{case}.raw")
                v = [raw[f"v({i})"] for i in range(1, len(reference) + 1)]
                q = [-raw[f"i(vdrive{i})"] for i in range(1, 6)]
                expected = [sum(float(gamma[i][j]) * imposed[j] for j in range(5))
                            for i in range(5)]
                vg = {0: 0.0, **{i: value for i, value in enumerate(v, 1)}}
                resistor_power = sum((vg[a] - vg[b]) ** 2 / r for _, a, b, r in resistors)
                input_power = sum(v[i] * q[i] for i in range(5))
                kcl = [sum(float(g[i][j]) * v[j] for j in range(len(g))) -
                       (q[i] if i < 5 else 0.0) for i in range(len(g))]
                observed[model][case] = {
                    "node_voltages_V": v,
                    "terminal_injections_A": q,
                    "paper_prediction_A": expected,
                    "max_current_error_vs_paper_A": maximum_difference(q, expected),
                    "max_voltage_clamp_error_V": maximum_difference(v[:5], imposed),
                    "max_kcl_residual_A": max(abs(x) for x in kcl),
                    "resistor_dissipation_W": resistor_power,
                    "input_power_W": input_power,
                    "power_balance_error_W": abs(resistor_power - input_power),
                }
        comparisons = {}
        for case in CASES:
            a, b = observed["original"][case], observed["kron"][case]
            comparisons[case] = {
                "max_terminal_current_difference_A": maximum_difference(
                    a["terminal_injections_A"], b["terminal_injections_A"]),
                "total_power_difference_W": abs(a["resistor_dissipation_W"] -
                                                b["resistor_dissipation_W"]),
            }
        measured = {model: [[observed[model][f"basis{j + 1}"]["terminal_injections_A"][i]
                             for j in range(5)] for i in range(5)]
                    for model in observed}
        max_matrix_error = max(abs(measured[model][i][j] - float(gamma[i][j]))
                               for model in measured for i in range(5) for j in range(5))
        errors = [max_matrix_error]
        errors += [x for c in comparisons.values() for x in c.values()]
        errors += [r[key] for model in observed.values() for r in model.values()
                   for key in ["max_current_error_vs_paper_A", "max_voltage_clamp_error_V",
                               "max_kcl_residual_A", "power_balance_error_W"]]
        passed = all(x <= TOL for x in errors)
        all_pass &= passed
        result["variants"][variant] = {
            "pass": passed, "all_netlist_entries_match_paper_exactly": True,
            "printed_schur_complement_matches_exactly": True,
            "netlist_sha256": netlist_hashes,
            "paper_kron_matrix_S": floats(gamma),
            "measured_terminal_matrices_S": measured,
            "max_matrix_error_vs_paper_S": max_matrix_error,
            "max_terminal_current_difference_A": max(
                c["max_terminal_current_difference_A"] for c in comparisons.values()),
            "effective_shunt_conductances_S": [float(sum(row)) for row in gamma],
            "cases": observed, "comparisons": comparisons,
        }
    result["all_pass"] = all_pass
    (OUT / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
    report = ["# SPICE verification of the eight-node circuits", "",
              f"Result: **{'PASS' if all_pass else 'FAIL'}**. Run UTC: {result['run_utc']}.", "",
              f"Simulator: `{ngspice}`; `{version.splitlines()[1].strip()}`.", "",
              "## Scope and method", "",
              "Four resistor netlists implement Eqs. (41)-(44). Each is simulated at seven DC "
              "operating points: five unit terminal-voltage vectors, one mixed vector "
              "[1, -0.5, 2, 0.25, -1] V, and the common vector [1, 1, 1, 1, 1] V. "
              "This is 28 ngspice operating-point solutions. All five terminal currents are compared.", "",
              "The five basis tests recover every entry of the terminal conductance matrix. "
              "Because these circuits contain linear resistors, this checks the full terminal map; "
              "the mixed and common tests are additional checks. Internal-node voltages are not "
              "claimed to be preserved by the reduced circuit.", "",
              "Every resistor value in each saved netlist is independently converted back to a "
              "conductance matrix and compared to the manuscript using exact rational arithmetic. "
              "The printed Schur complements are checked separately. Measured currents come from "
              "ngspice raw outputs, not from the Schur-complement calculation.", "",
              "The tests also check voltage clamps, KCL at every node, and equality of resistor "
              "dissipation and input power. Absolute acceptance tolerance is 1e-10 in the "
              "applicable units (A, V, S, W); this is a numerical check of ideal circuits.", "",
              "## Results", "",
              "| Network | Maximum full/reduced current difference (A) | Maximum matrix error vs paper (S) | Result |",
              "|---|---:|---:|---|"]
    for name, data in result["variants"].items():
        report.append(f"| {name.replace('_', ' ')} | {data['max_terminal_current_difference_A']:.3e} | "
                      f"{data['max_matrix_error_vs_paper_S']:.3e} | {'PASS' if data['pass'] else 'FAIL'} |")
    for name, data in result["variants"].items():
        report += ["", f"### {name.replace('_', ' ').capitalize()} mixed-voltage test", "",
                   "| Terminal | Full-network injection (A) | Reduced-network injection (A) | Absolute difference (A) |",
                   "|---|---:|---:|---:|"]
        a = data["cases"]["original"]["mixed"]["terminal_injections_A"]
        b = data["cases"]["kron"]["mixed"]["terminal_injections_A"]
        for i, (x, y) in enumerate(zip(a, b), 1):
            report.append(f"| {i} | {x:.12g} | {y:.12g} | {abs(x-y):.3e} |")
    report += ["", "## Grounding, shunts, and current signs", "",
               "- Paper edge labels are conductances in siemens. SPICE resistor cards require "
               "resistances in ohms, R = 1/g.",
               "- In the original shunted network, the 1 S shunt at node 2 is 1 ohm, and "
               "the 2 S shunt at node 7 is 0.5 ohm.",
               "- The reduced shunted network needs ground resistances 35/6, 35/59, 35/4, "
               "and 35/12 ohms at nodes 1, 2, 4, and 5, respectively. Node 3 has no shunt.",
               "- A diagonal matrix entry is the total incident conductance. The ground "
               "conductance is the row sum, not the diagonal entry.",
               "- Voltage sources have their positive terminal at the circuit node and "
               "negative terminal at ground. The paper's positive current injection equals "
               "minus the SPICE current through that voltage source.",
               "- Nodes 6-8 in the original circuit have zero external injection. In the "
               "reduced circuit they are omitted. No artificial resistive grounding is added "
               "to the shunt-free circuit; the terminal voltage sources fix its reference.", "",
               "With all terminal voltages at 1 V, the reduced shunted injections equal "
               "[12, 118, 0, 8, 24]/70 A, directly testing Eq. (45).", "",
               "These runs verify equivalence for the correctly constructed circuits. They "
               "do not identify the cause of Reviewer 4's discrepancy without that reviewer's netlist.", "",
               "## Reproduce", "", "From this directory:", "", "```sh",
               "python3 verify_spice.py", "```", "",
               "The verification script uses only the Python standard library and ngspice. "
               "It reruns all four saved netlists and replaces generated outputs in `spice_results/`. "
               "It does not modify manuscript files or regenerate the netlists.", "",
               "To run one netlist directly:", "", "```sh", "mkdir -p spice_results",
               "ngspice -n -b -o spice_results/eight_node_shunted_original.log eight_node_shunted_original.cir",
               "```", "",
               "`-n` ignores personal .spiceinit settings for reproducibility. Raw outputs are "
               "ASCII; `spice_results/verification.json` contains all measured currents, voltages, "
               "matrix entries, residuals, and netlist hashes.", "",
               "Reference: [official ngspice control-language tutorial]"
               "(https://ngspice.sourceforge.io/ngspice-control-language-tutorial.html).", ""]
    (ROOT / "SPICE_VERIFICATION.md").write_text("\n".join(report))
    print("PASS" if all_pass else "FAIL")
    for name, data in result["variants"].items():
        print(name, "max current difference (A):", data["max_terminal_current_difference_A"],
              "max matrix error (S):", data["max_matrix_error_vs_paper_S"])
    if not all_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
