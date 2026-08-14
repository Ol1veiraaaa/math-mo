from __future__ import annotations

import hashlib
import json
import re
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from c_contest_q1.q4_demo import ACTUAL_REQUIRED, validate_actual_schedule


SOURCE_URL = "https://github.com/openfootball/worldcup/blob/master/2026--canada-usa-mexico/cup.txt"
STADIUMS_URL = "https://github.com/openfootball/worldcup/blob/master/2026--canada-usa-mexico/stadiums.csv"
RETRIEVAL_DATE = "2026-08-14"
YEAR = 2026
COUNTRY_NAMES = {"ca": "Canada", "us": "United States", "mx": "Mexico"}
DATE_PATTERN = re.compile(r"^(?:Sun|Mon|Tue|Wed|Thu|Fri|Sat)\s+June\s+(\d{1,2})\s*$")
MATCH_PATTERN = re.compile(
    r"^\s*(?:(\d{1,2}:\d{2})\s+UTC([+-]\d+)\s+)?(.+?)\s+(\d+)-(\d+)"
    r"(?:\s+\(\d+-\d+\))?\s+(.+?)\s+@\s+(.+?)\s*$"
)
COORD_PATTERN = re.compile(
    r'''(\d+(?:\.\d+)?)°(?:(?:(\d+)')?(?:(\d+(?:\.\d+)?)")?)?\s*([NSEW])'''
)


def _dms_to_decimal(text: str) -> float:
    match = COORD_PATTERN.fullmatch(text.strip())
    if not match:
        raise ValueError(f"unparsed coordinate: {text!r}")
    degrees, minutes, seconds, hemisphere = match.groups()
    value = float(degrees) + float(minutes or 0) / 60 + float(seconds or 0) / 3600
    return value if hemisphere in ("N", "E") else -value


def parse_stadiums(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path, comment="#", skipinitialspace=True)
    stadiums = raw.iloc[:, :5].copy()
    stadiums.columns = ["city", "timezone", "cc", "venue", "capacity"]
    stadiums["city"] = stadiums.city.str.strip()
    stadiums["venue"] = stadiums.venue.str.strip()
    coordinates = raw["coords"].str.split(n=1, expand=True)
    coordinates.columns = ["latitude_text", "longitude_text"]
    stadiums["latitude"] = coordinates.latitude_text.map(_dms_to_decimal)
    stadiums["longitude"] = coordinates.longitude_text.map(_dms_to_decimal)
    stadiums["utc_offset"] = stadiums.timezone.str.extract(r"UTC([+-]\d+)", expand=False).astype(int)
    stadiums["country"] = stadiums.cc.map(COUNTRY_NAMES)
    columns = ["city", "venue", "country", "utc_offset", "capacity", "latitude", "longitude"]
    result = stadiums[columns].dropna()
    if len(result) != 16:
        raise ValueError(f"expected 16 stadiums, got {len(result)}")
    return result


def parse_schedule(path: Path, stadiums: pd.DataFrame) -> pd.DataFrame:
    text = path.read_text(encoding="utf-8-sig")
    groups: dict[str, list[str]] = {}
    for line in text.splitlines():
        roster_match = re.fullmatch(r"Group\s+([A-L])\s*\|\s*(.+)", line.strip())
        if roster_match:
            teams = re.split(r"\s{2,}", roster_match.group(2).strip())
            if len(teams) != 4:
                raise ValueError(f"group {roster_match.group(1)} does not list four teams: {teams!r}")
            groups[roster_match.group(1)] = teams
    if len(groups) != 12:
        raise ValueError(f"expected 12 groups, got {len(groups)}")
    venue_by_city = dict(zip(stadiums.city, stadiums.venue))
    country_by_city = dict(zip(stadiums.city, stadiums.country))
    group: str | None = None
    current_date: datetime | None = None
    current_time: str | None = None
    current_offset: int | None = None
    group_match_count = {key: 0 for key in groups}
    rows: list[dict[str, object]] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        group_match = re.fullmatch(r"\u25aa\s*Group\s+([A-L])", line)
        if group_match:
            group = group_match.group(1)
            current_time = None
            current_offset = None
            continue
        date_match = DATE_PATTERN.fullmatch(line)
        if date_match and group:
            current_date = datetime(YEAR, 6, int(date_match.group(1)))
            current_time = None
            current_offset = None
            continue
        if not group or current_date is None or "@" not in line:
            continue
        match = MATCH_PATTERN.match(raw_line)
        if not match:
            raise ValueError(f"unparsed match line: {raw_line!r}")
        explicit_time, explicit_offset, team_a, goals_a, goals_b, team_b, city = match.groups()
        if explicit_time is not None:
            current_time, current_offset = explicit_time, int(explicit_offset)
        if current_time is None or current_offset is None:
            raise ValueError(f"match without explicit or inherited time: {raw_line!r}")
        team_a, team_b = team_a.strip(), team_b.strip()
        if team_a not in groups[group] or team_b not in groups[group]:
            raise ValueError(f"unexpected teams in group {group}: {team_a}, {team_b}")
        if city.strip() not in venue_by_city:
            raise ValueError(f"city not in stadiums table: {city!r}")
        group_match_count[group] += 1
        round_number = (group_match_count[group] - 1) // 2 + 1
        hour, minute = map(int, current_time.split(":"))
        local = current_date.replace(
            hour=hour, minute=minute, tzinfo=timezone(timedelta(hours=current_offset))
        )
        rows.append({
            "match_id": f"WC26-{group}{group_match_count[group]:02d}",
            "competition": "2026 FIFA World Cup Canada, Mexico, United States",
            "stage": "Group Stage",
            "group_id": group,
            "round_in_group": round_number,
            "team_a": team_a,
            "team_b": team_b,
            "venue": venue_by_city[city.strip()],
            "city": city.strip(),
            "country": country_by_city[city.strip()],
            "date": local.date().isoformat(),
            "local_kickoff_time": current_time,
            "utc_datetime": local.astimezone(timezone.utc).isoformat(),
            "goals_a": int(goals_a),
            "goals_b": int(goals_b),
            "source_url": SOURCE_URL,
            "retrieval_date": RETRIEVAL_DATE,
        })
    result = pd.DataFrame(rows, columns=ACTUAL_REQUIRED)
    if len(result) != 72 or group_match_count != {key: 6 for key in groups}:
        raise ValueError(f"expected 72 matches and six per group, got {len(result)} / {group_match_count}")
    appearances = pd.concat([result.team_a, result.team_b]).value_counts()
    if len(appearances) != 48 or not appearances.eq(3).all():
        raise ValueError("each of the 48 actual teams must appear exactly three times")
    errors = validate_actual_schedule(result)
    if errors:
        raise ValueError("; ".join(errors))
    return result


def main() -> None:
    source = ROOT / "research" / "q4" / "openfootball_2026_cup.txt"
    stadiums_path = ROOT / "research" / "q4" / "openfootball_2026_stadiums.csv"
    out = ROOT / "outputs" / "q4"
    out.mkdir(parents=True, exist_ok=True)
    stadiums = parse_stadiums(stadiums_path)
    schedule = parse_schedule(source, stadiums)
    schedule.to_csv(out / "actual_schedule.csv", index=False)
    stadiums.to_csv(out / "actual_stadiums.csv", index=False)
    hashes = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (source, stadiums_path)
    }
    (out / "source_hashes.json").write_text(json.dumps(hashes, indent=2), encoding="utf-8")
    print(schedule.head().to_string(index=False))
    print({"rows": len(schedule), "groups": schedule.group_id.nunique(), "teams": pd.concat([schedule.team_a, schedule.team_b]).nunique(), "venues": schedule.venue.nunique()})


if __name__ == "__main__":
    main()
