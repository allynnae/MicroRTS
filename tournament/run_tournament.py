#!/usr/bin/env python3
"""
MicroRTS Tournament Runner

Discovers agent submissions, validates them, installs into the source tree,
compiles once, and runs a single-elimination tournament against reference AIs.

Uses the same scoring system as benchmark_arena.py for direct comparability.

Usage:
    python3 tournament/run_tournament.py [--games N] [--skip-h2h] [--submissions-dir DIR]
"""

import subprocess
import json
import os
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path

# Import validation
sys.path.insert(0, str(Path(__file__).parent))
from validate_submission import validate_submission, find_all_submissions

# Configuration - matches benchmark_arena.py exactly
CONFIG_FILE = "resources/config.properties"
RESULTS_DIR = "tournament_results"
MAX_CYCLES = 1500
MAP = "maps/8x8/basesWorkers8x8.xml"
GAME_TIMEOUT = 900  # 15 minutes

# Reference AI anchors - identical to benchmark_arena.py
ANCHORS = {
    "ai.RandomBiasedAI": {
        "name": "RandomBiasedAI",
        "weight": 10,
        "tier": "easy"
    },
    "ai.abstraction.HeavyRush": {
        "name": "HeavyRush",
        "weight": 20,
        "tier": "medium-hard"
    },
    "ai.abstraction.LightRush": {
        "name": "LightRush",
        "weight": 15,
        "tier": "medium"
    },
    "ai.abstraction.WorkerRush": {
        "name": "WorkerRush",
        "weight": 15,
        "tier": "medium"
    },
    "ai.competition.tiamat.Tiamat": {
        "name": "Tiamat",
        "weight": 20,
        "tier": "hard"
    },
    "ai.coac.CoacAI": {
        "name": "CoacAI",
        "weight": 20,
        "tier": "hard"
    },
}

def update_config(ai1, ai2):
    """Update config.properties with AI settings."""
    with open(CONFIG_FILE, 'r') as f:
        content = f.read()

    content = re.sub(r'^AI1=.*$', f'AI1={ai1}', content, flags=re.MULTILINE)
    content = re.sub(r'^AI2=.*$', f'AI2={ai2}', content, flags=re.MULTILINE)
    content = re.sub(r'^max_cycles=.*$', f'max_cycles={MAX_CYCLES}', content, flags=re.MULTILINE)
    content = re.sub(r'^headless=.*$', 'headless=true', content, flags=re.MULTILINE)

    with open(CONFIG_FILE, 'w') as f:
        f.write(content)


def run_game(ai1, ai2, ai1_name="", ai2_name=""):
    """Run a single game and return result."""
    update_config(ai1, ai2)

    env = os.environ.copy()
    display1 = ai1_name or ai1.split(".")[-1]
    display2 = ai2_name or ai2.split(".")[-1]

    print(f"  {display1} vs {display2}...", end=" ", flush=True)

    try:
        result = subprocess.run(
            ["java", "-cp", "lib/*:lib/bots/*:bin", "rts.MicroRTS", "-f", CONFIG_FILE],
            capture_output=True,
            text=True,
            timeout=GAME_TIMEOUT,
            env=env
        )
        output = result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        print("TIMEOUT")
        return {"result": "timeout", "ticks": MAX_CYCLES}
    except Exception as e:
        print(f"ERROR: {e}")
        return {"result": "error", "ticks": 0, "error": str(e)}

    winner = None
    ticks = MAX_CYCLES

    winner_match = re.search(r'WINNER:\s*(-?\d+)', output)
    if winner_match:
        winner = int(winner_match.group(1))

    tick_match = re.search(r'FINAL_TICK:\s*(\d+)', output)
    if tick_match:
        ticks = int(tick_match.group(1))

    if winner is None:
        if "Player 0 wins" in output:
            winner = 0
        elif "Player 1 wins" in output:
            winner = 1

    if winner == 0:
        print(f"WIN ({ticks} ticks)")
        return {"result": "win", "ticks": ticks}
    elif winner == 1:
        print(f"LOSS ({ticks} ticks)")
        return {"result": "loss", "ticks": ticks}
    else:
        print(f"DRAW ({ticks} ticks)")
        return {"result": "draw", "ticks": ticks}


def calculate_game_score(result, ticks):
    """Calculate score for a single game (same as benchmark_arena.py)."""
    if result == "win":
        base = 1.0
        if ticks < MAX_CYCLES * 0.5:
            bonus = 0.2
        elif ticks < MAX_CYCLES * 0.75:
            bonus = 0.1
        else:
            bonus = 0.0
        return min(1.2, base + bonus)
    elif result == "draw":
        return 0.5
    else:
        return 0.0


def calculate_benchmark_score(results):
    """Calculate final benchmark score 0-100 (same as benchmark_arena.py)."""
    total_score = 0.0
    for anchor_class, anchor_info in ANCHORS.items():
        if anchor_class in results:
            games = results[anchor_class]
            if games:
                avg_score = sum(
                    calculate_game_score(g["result"], g["ticks"])
                    for g in games
                ) / len(games)
                total_score += avg_score * anchor_info["weight"]
    return round(total_score, 1)


def score_to_grade(score):
    """Convert benchmark score to letter grade."""
    if score >= 90:
        return "A+"
    elif score >= 80:
        return "A"
    elif score >= 70:
        return "B"
    elif score >= 60:
        return "C"
    elif score >= 40:
        return "D"
    else:
        return "F"


def install_submission(submission_dir):
    """
    Copy a submission's Java file into the source tree.
    Reads the package declaration from the Java source to determine
    the correct install directory (supports ai.abstraction.submissions.*
    and ai.mcts.submissions.*).
    Returns (fully_qualified_class, display_name, metadata) or raises on failure.
    """
    submission_dir = Path(submission_dir)
    metadata_path = submission_dir / "metadata.json"

    with open(metadata_path) as f:
        metadata = json.load(f)

    agent_class = metadata["agent_class"]
    agent_file = metadata["agent_file"]

    # Read package from Java source
    java_source = (submission_dir / agent_file).read_text()
    package_match = re.search(r'^\s*package\s+([\w.]+)\s*;', java_source, re.MULTILINE)
    if not package_match:
        raise ValueError(f"No package declaration in {agent_file}")

    package = package_match.group(1)
    package_path = package.replace(".", "/")

    # Create package directory under src/
    target_dir = Path("src") / package_path
    target_dir.mkdir(parents=True, exist_ok=True)

    # Copy Java file
    src = submission_dir / agent_file
    dst = target_dir / agent_file
    shutil.copy2(src, dst)

    fqcn = f"{package}.{agent_class}"
    display_name = metadata.get("display_name", metadata["team_name"])
    return fqcn, display_name, metadata


def compile_all():
    """Compile the entire project including submissions."""
    print("Compiling project...", flush=True)

    # Use ant if available, fallback to manual javac compilation
    try:
        result = subprocess.run(
            ["ant", "build"],
            capture_output=True,
            text=True,
            timeout=120
        )
        if result.returncode == 0:
            print("Compilation successful (ant).")
            return True
        print(f"ant build failed:\n{result.stderr}")
        return False
    except FileNotFoundError:
        print("ant not found, falling back to javac...")

    # Fallback: find all Java sources and compile with javac
    sources_result = subprocess.run(
        ["find", "src", "-name", "*.java"],
        capture_output=True, text=True
    )
    if sources_result.returncode != 0:
        print("Failed to find Java sources.")
        return False

    sources_file = "sources.list"
    with open(sources_file, 'w') as f:
        f.write(sources_result.stdout)

    result = subprocess.run(
        ["javac", "-cp", "lib/*:bin", "-d", "bin", "@" + sources_file],
        capture_output=True,
        text=True,
        timeout=120
    )

    if result.returncode != 0:
        print(f"Compilation failed:\n{result.stderr}")
        return False

    print("Compilation successful (javac).")
    return True


def opponent_breakdown(reference_games):
    """Build per-opponent stats dict."""
    breakdown = {}
    for anchor_class, games in reference_games.items():
        if anchor_class not in ANCHORS:
            continue
        anchor_info = ANCHORS[anchor_class]
        wins = sum(1 for g in games if g["result"] == "win")
        draws = sum(1 for g in games if g["result"] == "draw")
        losses = sum(1 for g in games if g["result"] not in ("win", "draw"))
        avg_score = sum(
            calculate_game_score(g["result"], g["ticks"]) for g in games
        ) / len(games) if games else 0.0
        weighted_pts = round(avg_score * anchor_info["weight"], 1)
        breakdown[anchor_info["name"]] = {
            "wins": wins,
            "draws": draws,
            "losses": losses,
            "avg_game_score": round(avg_score, 3),
            "weighted_points": weighted_pts
        }
    return breakdown


def run_tournament(games_per_pair=1, skip_h2h=False, submissions_dir="submissions"):
    """Run the full tournament."""
    print("=" * 60)
    print("MicroRTS Tournament Runner")
    print("=" * 60)
    now = datetime.now()
    timestamp = now.strftime('%Y-%m-%d_%H-%M')
    print(f"Date: {now.strftime('%Y-%m-%d %H:%M')}")
    print(f"Map: {MAP}")
    print(f"Max Cycles: {MAX_CYCLES}")
    print(f"Games per matchup: {games_per_pair}")
    print()

    # Discover and validate submissions
    print("DISCOVERING SUBMISSIONS")
    print("-" * 40)
    submissions = find_all_submissions(submissions_dir)

    if not submissions:
        print("No submissions found.")
        sys.exit(1)

    valid_submissions = []
    for sub_dir in submissions:
        ok, errors = validate_submission(sub_dir)
        if ok:
            print(f"  [PASS] {sub_dir.name}")
            valid_submissions.append(sub_dir)
        else:
            print(f"  [FAIL] {sub_dir.name}")
            for err in errors:
                print(f"         {err}")

    if not valid_submissions:
        print("\nNo valid submissions found.")
        sys.exit(1)

    print(f"\n{len(valid_submissions)} valid submission(s)")
    print()

    # Install submissions
    print("INSTALLING SUBMISSIONS")
    print("-" * 40)
    contestants = {}  # fqcn -> {display_name, metadata, team_name}

    for sub_dir in valid_submissions:
        fqcn, display_name, metadata = install_submission(sub_dir)
        if fqcn:
            contestants[fqcn] = {
                "display_name": display_name,
                "metadata": metadata,
                "team_name": metadata["team_name"]
            }
            print(f"  Installed: {display_name} ({fqcn})")

    print()

    # Compile
    if not compile_all():
        print("Build failed. Aborting tournament.")
        sys.exit(1)
    print()

    # Phase 1: Each submission vs Reference AIs (single-elimination)
    print("TOURNAMENT GAMES (single-elimination)")
    print("-" * 40)
    print(f"  Elimination order ({len(ANCHORS)} opponents, 100 pts total):")
    for i, (_, info) in enumerate(ANCHORS.items(), 1):
        print(f"    {i}. {info['name']} ({info['tier']}): {info['weight']} pts max")
    print()

    all_results = {}
    benchmark_scores = {}
    eliminated_at = {}

    for fqcn, info in contestants.items():
        display_name = info["display_name"]
        all_results[display_name] = {"reference_games": {}}
        eliminated_at[display_name] = None
        eliminated = False

        print(f"\n{display_name}:")

        for anchor_class, anchor_info in ANCHORS.items():
            if eliminated:
                break

            all_results[display_name]["reference_games"][anchor_class] = []

            for game_num in range(games_per_pair):
                result = run_game(fqcn, anchor_class, display_name, anchor_info["name"])
                result["game_num"] = game_num + 1
                result["opponent"] = anchor_info["name"]
                all_results[display_name]["reference_games"][anchor_class].append(result)

            games = all_results[display_name]["reference_games"][anchor_class]
            has_win = any(g["result"] == "win" for g in games)
            if not has_win:
                eliminated = True
                eliminated_at[display_name] = anchor_info["name"]
                print(f"  ** ELIMINATED at {anchor_info['name']} (no win) **")

        if not eliminated:
            print(f"  ** CLEARED ALL OPPONENTS **")

        benchmark_scores[display_name] = calculate_benchmark_score(
            all_results[display_name]["reference_games"]
        )

    print()

    # Phase 2: Head-to-head (optional)
    h2h_results = []
    if not skip_h2h and len(contestants) > 1:
        print("HEAD-TO-HEAD GAMES (supplementary)")
        print("-" * 40)

        fqcn_list = list(contestants.keys())
        for i, fqcn1 in enumerate(fqcn_list):
            for fqcn2 in fqcn_list[i+1:]:
                name1 = contestants[fqcn1]["display_name"]
                name2 = contestants[fqcn2]["display_name"]
                for _ in range(games_per_pair):
                    result = run_game(fqcn1, fqcn2, name1, name2)
                    result["player0"] = name1
                    result["player1"] = name2
                    h2h_results.append(result)
        print()

    # Display results
    print("=" * 60)
    print("TOURNAMENT RESULTS")
    print("=" * 60)
    print()

    sorted_scores = sorted(benchmark_scores.items(), key=lambda x: x[1], reverse=True)

    print(f"{'Rank':<6}{'Team':<30}{'Score':<10}{'Grade':<8}{'Eliminated at'}")
    print("-" * 70)

    for rank, (name, score) in enumerate(sorted_scores, 1):
        grade = score_to_grade(score)
        elim = eliminated_at.get(name)
        elim_str = elim if elim else "-- cleared all --"
        print(f"{rank:<6}{name:<30}{score:<10}{grade:<8}{elim_str}")

    print()

    # Build output
    Path(RESULTS_DIR).mkdir(exist_ok=True)

    # Per-team breakdown
    team_results = []
    for name, score in sorted_scores:
        bd = opponent_breakdown(all_results[name]["reference_games"])
        # Find metadata
        meta = None
        for fqcn, info in contestants.items():
            if info["display_name"] == name:
                meta = info["metadata"]
                break

        # Find the fqcn for this team
        team_fqcn = ""
        for fqcn, info in contestants.items():
            if info["display_name"] == name:
                team_fqcn = fqcn
                break

        team_results.append({
            "team_name": meta["team_name"] if meta else name,
            "display_name": name,
            "agent_class": team_fqcn,
            "model_provider": meta.get("model_provider", "unknown") if meta else "unknown",
            "model_name": meta.get("model_name", "unknown") if meta else "unknown",
            "score": score,
            "grade": score_to_grade(score),
            "eliminated_at": eliminated_at.get(name),
            "opponents": bd,
            "date": now.isoformat(),
            "map": MAP,
            "games_per_matchup": games_per_pair
        })

    tournament_data = {
        "version": "1.0",
        "format": "single-elimination",
        "date": now.isoformat(),
        "config": {
            "map": MAP,
            "max_cycles": MAX_CYCLES,
            "games_per_matchup": games_per_pair
        },
        "anchors": {
            cls: {"name": info["name"], "weight": info["weight"], "tier": info["tier"]}
            for cls, info in ANCHORS.items()
        },
        "results": team_results,
        "head_to_head": h2h_results
    }

    results_file = f"{RESULTS_DIR}/tournament_{timestamp}.json"
    with open(results_file, 'w') as f:
        json.dump(tournament_data, f, indent=2)

    print(f"Results saved to {results_file}")
    return tournament_data


if __name__ == "__main__":
    games = 1
    skip_h2h = False
    submissions_dir = "submissions"

    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == "--games" and i + 1 < len(args):
            games = int(args[i + 1])
            i += 2
        elif args[i] == "--skip-h2h":
            skip_h2h = True
            i += 1
        elif args[i] == "--submissions-dir" and i + 1 < len(args):
            submissions_dir = args[i + 1]
            i += 2
        else:
            print(f"Unknown argument: {args[i]}")
            sys.exit(1)

    run_tournament(games_per_pair=games, skip_h2h=skip_h2h, submissions_dir=submissions_dir)
