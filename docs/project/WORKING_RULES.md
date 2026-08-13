# Contest Working Rules

## Question Gates

1. Work on exactly one contest question at a time.
2. After completing a question, stop before starting the next question.
3. Present model effects, baseline comparisons, diagnostics, robustness checks, and limitations at the stop gate.
4. Continue revising the current question until the user and Codex explicitly agree that the selected model is the best supported choice under the available evidence.
5. Do not let downstream work silently change an already accepted upstream model. Any reopening requires a new review gate.

## Modeling Preferences

1. Prefer models with stronger expected empirical performance when they can be installed and reproduced locally.
2. A complex model is accepted only when the common evaluation protocol demonstrates an improvement; model count or novelty is not evidence.
3. Keep feature engineering concise. Every derived feature must have a stated theoretical or factual rationale and must be available at the decision time.
4. Do not introduce unrelated or opaque variables beyond what the problem requires.
5. Keep runnable code, locked dependencies, seeds, model artifacts, metrics, plots, and outputs in the local project.

## Local Execution

1. The project is opened and inspected in Visual Studio Code.
2. Source editing is performed through workspace tools. Windows UI automation may operate the VS Code editor, but it must not automate VS Code terminals or Codex/ChatGPT extensions.
3. Commands, environment setup, training, and tests run through the workspace command tools so that their outputs remain auditable.

