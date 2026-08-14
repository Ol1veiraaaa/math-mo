# COPT Solver Evidence

## Q2 Restricted-Candidate Venue/Time MIP

`src/c_contest_q1/q2_copt.py` is the executed Q2 COPT model. The installed
COPT 8.0.6 non-commercial evaluation license limits a MIP to 2,000 variables
and 2,000 constraints. To keep the model within that limit, each match retains
its feasibility-first incumbent venue, the two largest security-eligible venue
alternatives, and all time slots on its incumbent reference date. This is an
explicit candidate decomposition and is not an unrestricted 72 x 16 x 80
global-optimality claim.

The model chooses one venue/time pair for each of 72 matches and enforces venue
totals and local-day limits, time-slot broadcast capacity, venue/UTC collision,
golden-slot fairness, third-round high-security daily capacity, and security
eligibility. Candidate travel burdens use the official prior-round
eligible-venue averaging definition. The selected schedule is independently
checked by `validate_q2_schedule` after optimization.

Executed evidence under `outputs/q2/copt/`:

- `solver.log`: 824 assignment candidates, 844 variables in total (840 binary),
  865 constraints, 0.0000% gap and zero bound/row/integrality violations;
- `result_2_group_schedule.csv`: the 72-row independently validated schedule;
- `solve_metadata.json`: COPT objective and independent full P2 score.

The restricted model linearizes the full official `Z2`, including venue
activation cost and both fairness ranges. Its COPT objective is
0.45816365037316586; the independent evaluator returns 0.458163650373169
(absolute difference below 4e-15). The fixed feasibility-first baseline scores
0.3187898737709609. This proves improvement and restricted-set optimality, not
unrestricted global optimality.

## Q3 Static and Dynamic Resource MIPs

`src/c_contest_q1/q3_copt.py` builds one common candidate set for both plans:
three broadcast levels, every permitted security level, three transport levels,
and five price points spanning each day's supplied bounds. Candidates violating
the official 88% same-transport attendance-protection rule are removed.

The static model optimizes the pre-feedback environment. Its four decisions are
then frozen and re-evaluated in the updated environment. The dynamic model uses
the same candidates, capacities, daily budgets, and updated-environment
normalization bounds, but re-optimizes after feedback. This makes the reported
static/dynamic comparison identifiable.

Executed evidence under `outputs/q3/copt/`:

- `solver.log`: COPT logs for the static, dynamic and feasible-lower-bound MIPs;
- `static_selection.csv`, `dynamic_selection.csv` and `feasible_lower_bound_selection.csv`: the selected candidates;
- `result_3_dynamic_strategy.csv`: the official 24-row Q3 output;
- `simulation_summary.csv`: conditional probabilities and risk intermediates;
- `solve_metadata.json`: model sizes, statuses and objective accounting.

All three models use 1,623 binary candidates and 48 constraints and terminate at a
0.0000% gap. The third model minimizes the same updated-environment objective
under the same daily constraints, providing a feasible lower endpoint for the
reported $Z_3$ range. The static pre-information objective is 6.679789185102986. With
its decisions held fixed, its updated-environment value is 6.6327957655273675.
The dynamic updated objective is 6.652194498235981, a total improvement of
0.0029246690828978374 (0.29247%). Ten of 24 matches change at least one decision.
Seven per-match contributions rise and three fall because daily capacities and
budgets couple the matches; the valid comparison is the positive global sum.
The matching minimization model returns 4.79629815250905, so the feasible
updated-environment objective range under the same candidates and daily
constraints is [4.79629815250905, 6.652194498235981].

## Reproducible Verification

Run Q2 and Q3 with the Python 3.12 COPT environment:

```powershell
.\.venv\Scripts\python.exe scripts\run_q2_copt.py
.\.venv\Scripts\python.exe scripts\run_q3_copt.py
```

Then independently check every official deliverable with:

```powershell
.\.venv-demo\Scripts\python.exe scripts\validate_deliverables.py
```

The exact solver logs and `outputs/validation/deliverable_validation.json` are
the controlling evidence for model status, feasibility and artifact hashes.
