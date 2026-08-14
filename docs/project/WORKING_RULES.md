# Contest Working Rules

## Delivery Flow

1. Work on exactly one contest question at a time.
2. First provide a complete end-to-end runnable Demo for the current question. A Demo may use a deliberately limited but valid baseline model; it must still read original data, validate inputs, train, predict, generate required outputs, and report its evidence.
3. Do not stop automatically after a question. Continue to the next question once its required upstream artifact is frozen for the current iteration, while preserving the ability to reopen and improve earlier questions.
4. Present model effects, baseline comparisons, diagnostics, robustness checks, and limitations after each substantial modeling iteration, but a user review is an evidence checkpoint rather than an implementation stop gate.
5. Do not let downstream work silently change an already accepted upstream model. Any reopening must create a new versioned artifact and record downstream impact.
6. The immediate delivery priority is one complete end-to-end Demo for all four questions. Q1 Demo v0 is the current upstream prediction artifact; higher-capacity Q1 model upgrades remain versioned follow-up work and must not block the full-demo chain.

## Modeling Preferences

1. Prefer models with stronger expected empirical performance when they can be installed and reproduced locally.
2. A complex model is accepted only when the common evaluation protocol demonstrates an improvement; model count or novelty is not evidence.
3. Keep feature engineering concise. Every derived feature must have a stated theoretical or factual rationale and must be available at the decision time.
4. Do not introduce unrelated or opaque variables beyond what the problem requires.
5. Keep runnable code, locked dependencies, seeds, model artifacts, metrics, plots, and outputs in the local project.

## COPT Competition Track

1. COPT is the preferred mathematical-programming solver for the Q2 venue-time assignment model and, where the Q3 resource decision can be formulated linearly or as a mixed-integer program, for the Q3 static and dynamic resource models.
2. CP-SAT, local search, and heuristics may supply feasible warm starts, nonlinear objective refinement, or robustness comparators, but must not obscure the COPT formulation and result evidence.
3. Preserve each COPT model source file, data interface, run command, solver version, parameter file, license-neutral log, solver status, objective/bound/gap, wall-clock time, and independent constraint-check report.
4. Capture a local COPT run screenshot after the model completes. Include the screenshot/log reference and the exact COPT role in the paper body or appendix and in supporting materials.
5. The COPT formulation must have a practical modeling reason: it must expose meaningful binary/continuous decisions, hard contest constraints, and an objective compatible with the stated evaluation framework.

## Local Execution

1. The project is opened and inspected in Visual Studio Code.
2. Source editing is performed through workspace tools. Windows UI automation may operate the VS Code editor, but it must not automate VS Code terminals or Codex/ChatGPT extensions.
3. Commands, environment setup, training, and tests run through the workspace command tools so that their outputs remain auditable.

## Model Routing

1. The primary task performs all implementation, testing, reasoning, and evidence integration directly; do not use subagents.
2. A high-reasoning model switch still requires user approval; routine work continues in the current primary task.
