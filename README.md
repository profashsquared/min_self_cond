Copyright (c) 2026 Ashkan Ashrafi, Department of Electrical and Computer
Engineering, San Diego State University (ashrafi@sdsu.edu).
Code for "A Minimum-Self-Conductance Characterization of Kron Reduction in
Resistive Networks," IEEE Trans. Circuits Syst. I (2026). MIT License.


# Code for "A Minimum-Self-Conductance Characterization of Kron Reduction in Resistive Networks"


# What the script does

The script embeds the eight-node branch list (Fig. 1) and the two shunts of
Fig. 3, reconstructs both conductance matrices with exact rational arithmetic,
and audits every matrix printed in the paper (the audit table is in the response
to Reviewer 1). It then, for each network, sample count M in {10, 1000}, and
prescribed condition number kappa_2(V_s) in {1, 50, 100}:

1. Solves the equality-constrained joint program (40) from voltage-current data
   alone and compares the result with the zero-padded Kron matrix
   (`M*/exact_recovery.csv`).
2. Solves the five row-wise programs (22) and the optimal-face uniqueness
   certificate of Remark 3 (`M*/uniqueness.csv`).
3. Runs the finite-SNR Monte Carlo comparison of Section VI-E: the regularized
   joint program (41) with 10-fold cross-validation (leave-one-out at M = 10)
   over the 13-value grid eta in {0} U {10^(-6+0.75k), k = 0..10} U {100},
   lambda = eta * max|entry of Q V^T / M|, against the direct pseudoinverse, on the same 100 noise realizations per condition, with
   independent noise in all voltage and sourced-current measurements at 0-50 dB
   (`--include-voltage-only` adds a voltage-noise-only experiment, not used in the paper). Metric (48), physical-feasibility
   diagnostics, timings, and a paired bootstrap are written to
   `M*/comparison_summary.csv`, `M*/nodewise_rmse.csv`, and `M*/paired_comparison.csv`.

Where to find the numbers quoted in the paper:

- 65/72 and 48/48 accuracy conditions, Table I counts and timings:
  `M*/paired_comparison.csv` and `M*/comparison_summary.csv`.
- RRMSE percentages at 20 and 50 dB: `sourced_block_relative_frobenius_rmse`
  in `M*/comparison_summary.csv` (fractions; multiply by 100).
- eta = 0 selection rates: `smallest_eta_fraction`, same file.
- 4.28e-13 S / 8.71e-13 A exact-data errors: `M*/exact_recovery.csv`, rows `joint_exact`.
- 60 unique rows, maximum width 2.98e-14 S: `M*/uniqueness.csv`.
- Table II (alternative exact-fit network): `M*/exact_results.json`,
  keys `alternative_diagonal_S` and `alternative_max_residual_A`.
- Exact-arithmetic audit (table in the response to Reviewer 1): `exact_arithmetic_audit.csv`.

## Running

The finite-SNR quadratic programs and the exact LPs are solved with Gurobi
(the reported run used Gurobi 11.0.3, single thread). Without a Gurobi license
you can still run the solver-free checks, which use SciPy/HiGHS for the row LPs
and the uniqueness certificate:

    python3 min_self_conductance_kron.py --self-test

With Gurobi available:

    python3 min_self_conductance_kron.py --check          # small solver test
    python3 min_self_conductance_kron.py --samples 10 1000 --exact-only --output exact_only
    python3 min_self_conductance_kron.py --samples 10 1000 --trials 100 --output rerun   # full study, ~30 min on an Apple M-series laptop

`--folds K` changes the number of cross-validation folds (default 10). To
regenerate the summary files and figures from the stored run without
re-solving anything:

    python3 min_self_conductance_kron.py --samples 10 1000 --trials 100 --output results/revision_kfold_vq --summarize-only

Random streams are seeded per condition and trial (seed 20260921), so a rerun
reproduces the same noisy inputs; fitted matrices agree with the stored ones up
to solver tolerance. The QPs are ordinary convex QPs; stored trials have also
been reproduced with SciPy's SLSQP to about 1e-10 in objective value.

## Notes on interpretation

- The direct estimator uses terminal voltages and currents only; the joint
  estimator uses all nodal voltages, terminal currents, and the known zero
  interior injections. The comparison is between the two procedures as
  described in the paper, not an equal-information comparison.
- The matrix metric compares the estimated sourced block with Gamma; it is not
  the Schur complement of the noisy full-order estimate.
- Uniqueness results are numerical checks at a stated tolerance (1e-7 S), not
  exact-arithmetic certificates. The rational audit covers only the printed matrices.

## SPICE verification

`spice_verification/verify_spice.py` needs only the Python standard library and
ngspice on the PATH; the printed conductance matrices it checks against are
embedded in the script. It reruns the four saved netlists (original and reduced,
shunt-free and shunted) at seven operating points, compares terminal currents,
and rebuilds the conductance matrices from the resistor values in each netlist.
`SPICE_VERIFICATION.md` is its generated report (its equation numbers refer to
the originally submitted manuscript: (41)-(44) there are (42)-(45) in the
revision). The `*_nonzero.cir` netlists are the simultaneous nonzero-excitation
circuits listed in the response to Reviewer 4; each can be run directly with
`ngspice -b <file>`, and `spice_results/nonzero_verification.json` holds their
results.
