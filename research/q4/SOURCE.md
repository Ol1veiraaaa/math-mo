# Q4 Source Snapshot

The 2026 FIFA World Cup group-stage schedule and stadium data were retrieved on
2026-08-14 from the public OpenFootball World Cup repository:

- schedule: https://github.com/openfootball/worldcup/blob/master/2026--canada-usa-mexico/cup.txt
- stadiums: https://github.com/openfootball/worldcup/blob/master/2026--canada-usa-mexico/stadiums.csv

The tournament ended on 2026-07-19, so the schedule contains final results.
The parser uses only the group-stage section (72 matches, 12 groups); the
knockout stages in `cup_finals.txt` are out of scope because the Q2 optimized
schedule covers the group stage only.

The parser retains the row-level schedule page URL in `source_url` and the
retrieval date in every generated actual-schedule record. It does not use goal
scorer annotations in the comparison model. Stadium coordinates are parsed
from the DMS strings in `stadiums.csv`; capacities come from the same file.
