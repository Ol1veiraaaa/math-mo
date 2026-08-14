# End-to-End Demo Status

## Implemented Artifacts

- Q1: 560-row temporal training pipeline, three expanding validation windows,
  model comparison, 140 hidden-test predictions in millions, and 72 future
  match predictions in persons. Ridge has the lowest observed temporal-CV MSE
  (225.0601) among the tested models; this is not a hidden-test or universal
  optimality claim.
- Q2: deterministic feasible baseline plus a restricted-candidate COPT MIP.
  The COPT schedule passes an independent six-group hard-constraint validator
  and improves full P2 score from 0.3187898738 to 0.4581636504. The COPT
  objective and independent evaluator agree to numerical precision.
- Q3: official-formula state update, simultaneous 20,000-draw Poisson
  simulation, 32-place advancement accounting, risk calculation, and matched
  static/dynamic COPT resource models. The dynamic global objective improves
  0.29247% under common updated-environment bounds.
- Q4: reproducible parser for the completed 2026 FIFA World Cup OpenFootball
  schedule, row-level source provenance, 72-match structural validation, and a
  same-scale comparison against the Q2 schedule.
- Analysis campaign: 10,000 Q2 weight perturbations, 5k/10k/20k/50k Q3
  simulation convergence, and static/dynamic Q3 decision accounting.

## Explicit Boundaries

- Q2 is integer-optimal only inside its documented restricted candidate set,
  required by the 2,000-variable COPT evaluation limit.
- Q1 selection is based only on internal temporal validation because official
  hidden-test labels are unavailable.
- Q4 compares structural indicators that can be reconstructed on both
  tournaments. It does not fabricate historical attendance, ticket revenue,
  broadcast value, golden-slot scores, fairness or risk inputs that the public
  source does not contain.
- Actual 2026 and the simulated 2026 competition share scale but differ in
  regional organization. Per-team, per-transition and concentration metrics are
  therefore used; conclusions are multi-objective trade-offs rather than a
  claim that one schedule is uniformly superior.

## Current Verification

The complete Python test suite and `scripts/validate_deliverables.py` are the
authoritative executable checks. Paper, reproduction instructions and final
submission packaging are tracked separately and must pass their own build and
visual-review gates before the project is considered finished.
