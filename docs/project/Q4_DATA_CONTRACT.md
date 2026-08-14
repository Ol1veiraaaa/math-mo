# Q4 Actual-Schedule Data Contract

The comparison uses the completed 2026 FIFA World Cup group stage (Canada,
Mexico, United States). The raw OpenFootball schedule and stadium snapshot are
retained under `research/q4/`, with provenance recorded in
`research/q4/SOURCE.md` and cryptographic hashes in
`outputs/q4/source_hashes.json`.

`scripts/collect_q4_openfootball.py` parses the source into
`outputs/q4/actual_schedule.csv` and `outputs/q4/actual_stadiums.csv` without
hand-entering match rows. The output follows the organizer's
`actual_schedule_template.csv` column order and keeps a source URL and retrieval
date on every row. The parser/validator confirms:

- 72 group matches, 12 groups and 48 teams;
- 6 matches per group and 3 matches per team;
- 16 named venues across 3 countries;
- local kickoff times with per-line UTC offsets converted to UTC+00:00;
- unique match identifiers and finite score fields.

`scripts/run_q4_comparison.py` compares the actual tournament with the
72-match, 48-team Q2 COPT schedule. Because both are 72-match, 48-team,
16-venue competitions, the scale indicators are directly comparable; the
comparison additionally uses per-team travel, per-transition travel, rest
distributions, timezone crossings, venue-load CV/HHI and mean capacity. Travel
measures venue-to-venue transitions only; it excludes team origin to first
venue because the public schedule does not provide an equivalent origin
mapping.

The comparison is same-scale but geographically distinct: FIFA's actual plan
groups teams regionally (western/central/eastern), which cuts per-team travel
(2,058.63 versus 4,257.17 km) and timezone crossings (0.625 versus 1.458), while
the simulated optimized plan has longer rest (167.81 versus 133.42 mean hours,
159.00 versus 91.00 minimum hours) and slightly higher mean capacity (71,353
versus 68,292). No historical ticket, broadcast, attendance, golden-slot,
fairness or risk value is invented. Those dimensions are discussed as
unavailable rather than silently scored.
