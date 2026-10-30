<!--
Copyright (c) 2026 Ashkan Ashrafi, Department of Electrical and Computer
Engineering, San Diego State University (ashrafi@sdsu.edu).
Code for "A Minimum-Self-Conductance Characterization of Kron Reduction in
Resistive Networks," IEEE Trans. Circuits Syst. I (2026). MIT License.
-->
# SPICE verification of the eight-node circuits

Result: **PASS**. Run UTC: 2026-09-27T21:46:52.940569+00:00.

Simulator: `/opt/homebrew/bin/ngspice`; `** ngspice-47 : Circuit level simulation program`.

## Scope and method

Four resistor netlists implement Eqs. (41)-(44). Each is simulated at seven DC operating points: five unit terminal-voltage vectors, one mixed vector [1, -0.5, 2, 0.25, -1] V, and the common vector [1, 1, 1, 1, 1] V. This is 28 ngspice operating-point solutions. All five terminal currents are compared.

The five basis tests recover every entry of the terminal conductance matrix. Because these circuits contain linear resistors, this checks the full terminal map; the mixed and common tests are additional checks. Internal-node voltages are not claimed to be preserved by the reduced circuit.

Every resistor value in each saved netlist is independently converted back to a conductance matrix and compared to the manuscript using exact rational arithmetic. The printed Schur complements are checked separately. Measured currents come from ngspice raw outputs, not from the Schur-complement calculation.

The tests also check voltage clamps, KCL at every node, and equality of resistor dissipation and input power. Absolute acceptance tolerance is 1e-10 in the applicable units (A, V, S, W); this is a numerical check of ideal circuits.

## Results

| Network | Maximum full/reduced current difference (A) | Maximum matrix error vs paper (S) | Result |
|---|---:|---:|---|
| shunt free | 8.882e-16 | 8.882e-16 | PASS |
| shunted | 8.882e-16 | 8.882e-16 | PASS |

### Shunt free mixed-voltage test

| Terminal | Full-network injection (A) | Reduced-network injection (A) | Absolute difference (A) |
|---|---:|---:|---:|
| 1 | 3.6847826087 | 3.6847826087 | 0.000e+00 |
| 2 | -10.7608695652 | -10.7608695652 | 0.000e+00 |
| 3 | 9.25 | 9.25 | 0.000e+00 |
| 4 | 1.58152173913 | 1.58152173913 | 0.000e+00 |
| 5 | -3.75543478261 | -3.75543478261 | 0.000e+00 |

### Shunted mixed-voltage test

| Terminal | Full-network injection (A) | Reduced-network injection (A) | Absolute difference (A) |
|---|---:|---:|---:|
| 1 | 3.62142857143 | 3.62142857143 | 0.000e+00 |
| 2 | -11.5142857143 | -11.5142857143 | 0.000e+00 |
| 3 | 9.25 | 9.25 | 0.000e+00 |
| 4 | 1.53928571429 | 1.53928571429 | 0.000e+00 |
| 5 | -3.88214285714 | -3.88214285714 | 0.000e+00 |

## Grounding, shunts, and current signs

- Paper edge labels are conductances in siemens. SPICE resistor cards require resistances in ohms, R = 1/g.
- In the original shunted network, the 1 S shunt at node 2 is 1 ohm, and the 2 S shunt at node 7 is 0.5 ohm.
- The reduced shunted network needs ground resistances 35/6, 35/59, 35/4, and 35/12 ohms at nodes 1, 2, 4, and 5, respectively. Node 3 has no shunt.
- A diagonal matrix entry is the total incident conductance. The ground conductance is the row sum, not the diagonal entry.
- Voltage sources have their positive terminal at the circuit node and negative terminal at ground. The paper's positive current injection equals minus the SPICE current through that voltage source.
- Nodes 6-8 in the original circuit have zero external injection. In the reduced circuit they are omitted. No artificial resistive grounding is added to the shunt-free circuit; the terminal voltage sources fix its reference.

With all terminal voltages at 1 V, the reduced shunted injections equal [12, 118, 0, 8, 24]/70 A, directly testing Eq. (45).

These runs verify equivalence for the correctly constructed circuits. They do not identify the cause of Reviewer 4's discrepancy without that reviewer's netlist.

## Reproduce

From this directory:

```sh
python3 verify_spice.py
```

The verification script uses only the Python standard library and ngspice. It reruns all four saved netlists and replaces generated outputs in `spice_results/`. It does not modify manuscript files or regenerate the netlists.

To run one netlist directly:

```sh
mkdir -p spice_results
ngspice -n -b -o spice_results/eight_node_shunted_original.log eight_node_shunted_original.cir
```

`-n` ignores personal .spiceinit settings for reproducibility. Raw outputs are ASCII; `spice_results/verification.json` contains all measured currents, voltages, matrix entries, residuals, and netlist hashes.

Reference: [official ngspice control-language tutorial](https://ngspice.sourceforge.io/ngspice-control-language-tutorial.html).
