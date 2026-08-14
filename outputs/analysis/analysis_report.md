# Modeling Analysis Campaign

Parent claims: the restricted-candidate Q2 COPT schedule improves the fixed feasibility baseline, and Q3 feedback re-optimization improves the static resource plan under the same updated evaluation bounds.

## Q2 Weight Robustness

Across 10,000 independent +/-20% weight perturbations normalized to sum to one, the COPT schedule win rate was 100.00%. The 5th percentile objective advantage was 0.117677.

## Q3 Monte Carlo Convergence

Against 50,000 simulations, the official 20,000-draw run had probability MAE 0.001416 and maximum absolute difference 0.008530.

## Q3 Static vs Dynamic Resources

Feedback changed at least one resource or price decision for 10 of 24 matches. The globally constrained dynamic objective improved by 0.292% over the fixed static decisions evaluated in the same updated environment.

Comparability: Q2 changes weights only; Q3 convergence changes simulation count only; static/dynamic Q3 uses one candidate set, identical capacities and budgets, and common updated-environment normalization bounds.

Next route: use these results in the paper and proceed to Q4 external schedule comparison.
