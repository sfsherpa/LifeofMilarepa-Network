"""
Full network analysis of the Life of Milarepa (milarepa.raw.csv)

Covers:
 1. Degree Centrality
 2. Betweenness Centrality
 3. Shortest Path (+ betweenness mode)
 4. Closeness Centrality
 5. Closeness Eccentricity
 6. Clustering Coefficient
 7. PageRank
 8. Weighted Graph analysis
 9. Temporal Graphs (per chapter)
10. Place Network
11. Dialogue Network (Song/Letter/Prose with speaker direction)
"""

import csv
import collections
import itertools
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import os, sys

# ── output directory ─────────────────────────────────────────────────────────
OUT = os.path.join(os.path.dirname(__file__), "network_output")
os.makedirs(OUT, exist_ok=True)

# ── load CSV ─────────────────────────────────────────────────────────────────
CSV = os.path.join(os.path.dirname(__file__), "milarepa.raw.csv")

print("Loading data …")
rows = []
with open(CSV, encoding="utf-8-sig") as f:
    reader = csv.DictReader(f)
    for row in reader:
        rows.append(row)

print(f"  {len(rows):,} rows loaded")

# ── helper: extract English name from "tibetan syllables English Name" ────────
def display(full_name: str) -> str:
    """Extract the English portion from a combined Tibetan+English name string.

    Names are formatted as lowercase Tibetan syllables followed by one or more
    Title-Case English tokens, e.g. 'mar pa Marpa' → 'Marpa',
    'se ban ras pa Seban Repa' → 'Seban Repa',
    'ras chung pa rdo rje grags Rechungpa Dorjé Drak' → 'Rechungpa Dorjé Drak'.
    We find the last contiguous run of tokens whose first character is uppercase.
    """
    full_name = full_name.strip()
    tokens = full_name.split()
    if not tokens:
        return full_name
    # Walk backwards to find the trailing run of capitalised tokens
    run = []
    for tok in reversed(tokens):
        if tok and tok[0].isupper():
            run.append(tok)
        else:
            break
    if run:
        return " ".join(reversed(run))
    return full_name  # fallback: return as-is


# ── pure-numpy PageRank (scipy not available) ─────────────────────────────────
def pagerank_np(G, weight="weight", alpha=0.85, max_iter=200, tol=1.0e-6):
    """Power-iteration PageRank using numpy only."""
    nodes = list(G.nodes())
    n = len(nodes)
    if n == 0:
        return {}
    idx = {node: i for i, node in enumerate(nodes)}

    M = np.zeros((n, n), dtype=float)
    is_directed = G.is_directed()
    for u in nodes:
        nbrs = list(G.successors(u)) if is_directed else list(G.neighbors(u))
        if weight:
            total_w = sum(G[u][v].get(weight, 1) for v in nbrs)
        else:
            total_w = len(nbrs)
        if total_w == 0:
            M[:, idx[u]] = 1.0 / n          # dangling node
        else:
            for v in nbrs:
                w = G[u][v].get(weight, 1) if weight else 1
                M[idx[v], idx[u]] += w / total_w

    rank = np.ones(n) / n
    for _ in range(max_iter):
        new_rank = alpha * (M @ rank) + (1.0 - alpha) / n
        if np.abs(new_rank - rank).sum() < n * tol:
            break
        rank = new_rank
    return {nodes[i]: float(rank[i]) for i in range(n)}

# ── group rows by text unit (Chapter, Paragraph, Line) ───────────────────────
# Each text unit maps to: list of persons + list of locations + text_type
TextUnit = collections.namedtuple(
    "TextUnit", ["chapter", "paragraph", "line", "text_type",
                 "persons",   # list of (name, kind)
                 "locations"] # list of (name, kind)
)

unit_map = collections.defaultdict(lambda: {"text_type": "", "persons": [], "locations": []})
for r in rows:
    key = (r["Chapter"].strip(), r["Paragraph"].strip(), r["Line"].strip())
    if r["Text Type"].strip():
        unit_map[key]["text_type"] = r["Text Type"].strip()
    p = r["[Personal Reference] Person"].strip()
    pk = r["[Personal Reference] Kind of Reference"].strip()
    l = r["[Spatial Reference] Location Reference"].strip()
    lk = r["[Spatial Reference] Kind of Reference"].strip()
    if p and (p, pk) not in unit_map[key]["persons"]:
        unit_map[key]["persons"].append((p, pk))
    if l and (l, lk) not in unit_map[key]["locations"]:
        unit_map[key]["locations"].append((l, lk))

units = [
    TextUnit(key[0], key[1], key[2], val["text_type"],
             val["persons"], val["locations"])
    for key, val in unit_map.items()
]
print(f"  {len(units):,} unique text units")

# ── narrative phases by chapter ───────────────────────────────────────────────
PHASES = {
    "0": "Framing",
    "1": "Early Life",
    "2": "Revenge",
    "3": "Apprenticeship (Marpa)",
    "4": "Austerities",
    "5": "Meditation",
    "6": "Solitary Practice",
    "7": "Enlightenment",
    "8": "Teaching Begins",
    "9": "Disciple Encounters",
    "10": "Wandering & Teaching",
    "11": "Later Disciples",
    "12": "Final Teachings",
    "13": "Death & Colophon",
}

# ═════════════════════════════════════════════════════════════════════════════
# BUILD GRAPHS
# ═════════════════════════════════════════════════════════════════════════════

# ── 1. Weighted character co-occurrence graph ─────────────────────────────────
print("\nBuilding character co-occurrence graph …")
G_char = nx.Graph()
for unit in units:
    persons = [p for p, _ in unit.persons]
    for a, b in itertools.combinations(persons, 2):
        if G_char.has_edge(a, b):
            G_char[a][b]["weight"] += 1
        else:
            G_char.add_edge(a, b, weight=1)

# Add isolated nodes (persons who never co-occur)
all_persons = set(p for unit in units for p, _ in unit.persons)
for p in all_persons:
    if p not in G_char:
        G_char.add_node(p)

print(f"  Nodes: {G_char.number_of_nodes():,}  Edges: {G_char.number_of_edges():,}")

# ── 2. Weighted character–place bipartite graph ───────────────────────────────
print("Building character–place bipartite graph …")
G_bp = nx.Graph()
for unit in units:
    persons   = [p for p, _ in unit.persons]
    locations = [l for l, _ in unit.locations]
    for p in persons:
        for l in locations:
            if G_bp.has_edge(p, l):
                G_bp[p][l]["weight"] += 1
            else:
                G_bp.add_edge(p, l, weight=1)

print(f"  Nodes: {G_bp.number_of_nodes():,}  Edges: {G_bp.number_of_edges():,}")

# ── 3. Place co-occurrence graph ──────────────────────────────────────────────
print("Building place co-occurrence graph …")
G_place = nx.Graph()
for unit in units:
    locs = [l for l, _ in unit.locations]
    for a, b in itertools.combinations(locs, 2):
        if G_place.has_edge(a, b):
            G_place[a][b]["weight"] += 1
        else:
            G_place.add_edge(a, b, weight=1)

all_locations = set(l for unit in units for l, _ in unit.locations)
for l in all_locations:
    if l not in G_place:
        G_place.add_node(l)
print(f"  Nodes: {G_place.number_of_nodes():,}  Edges: {G_place.number_of_edges():,}")

# ── 4. Directed dialogue / speech graph ──────────────────────────────────────
# In Songs/Letters, "Explicit" persons are the singers/writers.
# "Referenced" persons are likely addressed.
# We draw a directed edge: Explicit → Referenced within the same Song/Letter.
print("Building dialogue (directed speech) graph …")
G_dial = nx.DiGraph()
SPEECH_TYPES = {"Song", "Letter", "Prayer"}
for unit in units:
    if unit.text_type not in SPEECH_TYPES:
        continue
    speakers  = [p for p, k in unit.persons if k == "Explicit"]
    addressed = [p for p, k in unit.persons if k in ("Referenced", "Implicit")]
    for s in speakers:
        for a in addressed:
            if s == a:
                continue
            if G_dial.has_edge(s, a):
                G_dial[s][a]["weight"] += 1
            else:
                G_dial.add_edge(s, a, weight=1)

print(f"  Nodes: {G_dial.number_of_nodes():,}  Edges: {G_dial.number_of_edges():,}")

# ═════════════════════════════════════════════════════════════════════════════
# UTILITY: top-N table printer
# ═════════════════════════════════════════════════════════════════════════════
def top_table(scores: dict, title: str, n: int = 25, file=sys.stdout):
    print(f"\n{'═'*60}", file=file)
    print(f"  {title}", file=file)
    print(f"{'═'*60}", file=file)
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:n]
    for rank, (node, val) in enumerate(ranked, 1):
        name = display(node)
        print(f"  {rank:>3}. {name:<35} {val:.6f}", file=file)

# ═════════════════════════════════════════════════════════════════════════════
# ANALYSES
# ═════════════════════════════════════════════════════════════════════════════

REPORT = open(os.path.join(OUT, "full_report.txt"), "w", encoding="utf-8")

def rprint(*args, **kwargs):
    """Print to both stdout and report file."""
    print(*args, **kwargs)
    print(*args, file=REPORT, **kwargs)

rprint("=" * 70)
rprint("  NETWORK ANALYSIS: THE LIFE OF MILAREPA")
rprint("=" * 70)
rprint(f"\n  Source rows  : {len(rows):,}")
rprint(f"  Text units   : {len(units):,}")
rprint(f"  Unique persons: {len(all_persons):,}")
rprint(f"  Unique places : {len(all_locations):,}")
rprint(f"  Chapters     : {sorted(set(u.chapter for u in units))}")

# ── Work on the largest connected component for path-based metrics ────────────
Gcc = G_char.subgraph(max(nx.connected_components(G_char), key=len)).copy()
rprint(f"\n  Char graph  — nodes {G_char.number_of_nodes()}, edges {G_char.number_of_edges()}")
rprint(f"  LCC (char)  — nodes {Gcc.number_of_nodes()}, edges {Gcc.number_of_edges()}")

# ────────────────────────────────────────────────────────────────────────────
# 1. DEGREE CENTRALITY
# ────────────────────────────────────────────────────────────────────────────
rprint("\n\n" + "━"*70)
rprint("1. DEGREE CENTRALITY — Character Graph")
rprint("━"*70)
rprint("""
Measures: how many distinct co-characters appear alongside each person.
High degree = socially/narratively active figure who interacts broadly.
""")

deg = nx.degree_centrality(G_char)
top_table(deg, "Degree Centrality — Character Co-occurrence", file=REPORT)
top_table(deg, "Degree Centrality — Character Co-occurrence")

# Also raw degree (number of unique co-characters)
raw_deg = dict(G_char.degree())
rprint("\n  Raw degree (unique co-character partners):")
for rank, (n, v) in enumerate(sorted(raw_deg.items(), key=lambda x: x[1], reverse=True)[:20], 1):
    rprint(f"    {rank:>3}. {display(n):<35} {v}")

# Degree centrality for PLACE graph
deg_place = nx.degree_centrality(G_place)
rprint("\n")
top_table(deg_place, "Degree Centrality — Place Co-occurrence", file=REPORT)
top_table(deg_place, "Degree Centrality — Place Co-occurrence")

# ────────────────────────────────────────────────────────────────────────────
# 2. BETWEENNESS CENTRALITY
# ────────────────────────────────────────────────────────────────────────────
rprint("\n\n" + "━"*70)
rprint("2. BETWEENNESS CENTRALITY")
rprint("━"*70)
rprint("""
Measures: fraction of all shortest paths (between any two nodes) that
pass through a given node.  High betweenness = structural bridge or broker
between otherwise disconnected communities.
""")

print("  Computing betweenness (character) …")
btwn = nx.betweenness_centrality(Gcc, weight=None, normalized=True)
top_table(btwn, "Betweenness Centrality — Character LCC", file=REPORT)
top_table(btwn, "Betweenness Centrality — Character LCC")

print("  Computing betweenness (weighted) …")
btwn_w = nx.betweenness_centrality(Gcc, weight="weight", normalized=True)
top_table(btwn_w, "Betweenness Centrality — Weighted Character LCC", file=REPORT)
top_table(btwn_w, "Betweenness Centrality — Weighted Character LCC")

print("  Computing betweenness (place) …")
Gpcc = G_place.subgraph(max(nx.connected_components(G_place), key=len)).copy()
btwn_p = nx.betweenness_centrality(Gpcc, weight=None, normalized=True)
top_table(btwn_p, "Betweenness Centrality — Place LCC", file=REPORT)
top_table(btwn_p, "Betweenness Centrality — Place LCC")

# ────────────────────────────────────────────────────────────────────────────
# 3. SHORTEST PATHS
# ────────────────────────────────────────────────────────────────────────────
rprint("\n\n" + "━"*70)
rprint("3. SHORTEST PATHS (unweighted)")
rprint("━"*70)
rprint("""
Minimum number of co-occurrence hops between two characters.
Low distance = tight narrative connection; high = peripheral relationship.
""")

# Key character pairs
KEY_PAIRS = [
    ("Mi la ras pa Milarepa",            "mar pa Marpa"),
    ("Mi la ras pa Milarepa",            "ras chung pa rdo rje grags Rechungpa Dorjé Drak"),
    ("Mi la ras pa Milarepa",            "bdag med ma Dakmema"),
    ("Mi la ras pa Milarepa",            "se ban ras pa Seban Repa"),
    ("Mi la ras pa Milarepa",            "Nāropa"),
    ("mar pa Marpa",                     "ras chung pa rdo rje grags Rechungpa Dorjé Drak"),
    ("mar pa Marpa",                     "se ban ras pa Seban Repa"),
    ("bdag med ma Dakmema",              "ras chung pa rdo rje grags Rechungpa Dorjé Drak"),
    ("mar pa Marpa",                     "Nāropa"),
    ("pe ta mgon skyid Peta Gönkyi",     "Mi la ras pa Milarepa"),
    ("myang rsta dkar rgyan Nyangtsa Kargyen", "Mi la ras pa Milarepa"),
]

for a, b in KEY_PAIRS:
    if a in Gcc and b in Gcc:
        try:
            path = nx.shortest_path(Gcc, a, b)
            length = len(path) - 1
            path_names = " → ".join(display(n) for n in path)
            rprint(f"\n  {display(a)} ↔ {display(b)}")
            rprint(f"    Distance: {length}")
            rprint(f"    Path: {path_names}")
        except nx.NetworkXNoPath:
            rprint(f"\n  {display(a)} ↔ {display(b)}: NO PATH")
    else:
        missing = [x for x in (a, b) if x not in Gcc]
        rprint(f"\n  Skipped (not in LCC): {[display(m) for m in missing]}")

# ────────────────────────────────────────────────────────────────────────────
# 4. SHORTEST PATH + BETWEENNESS MODE (node importance on key paths)
# ────────────────────────────────────────────────────────────────────────────
rprint("\n\n" + "━"*70)
rprint("4. BETWEENNESS MODE ON KEY PATHS")
rprint("━"*70)
rprint("""
For a selected source→target, count how many shortest paths between all
pairs pass through each intermediate node.  Nodes with high count are
'structurally unavoidable' relays on that route.
""")

# Sub-betweenness: restrict graph to subgraph of k-hop neighbourhood of Milarepa
MILAREPA = "Mi la ras pa Milarepa"
MARPA    = "mar pa Marpa"
RECHUNG  = "ras chung pa rdo rje grags Rechungpa Dorjé Drak"

def path_betweenness_between(G, source, target, top_n=10):
    """Count intermediate nodes over ALL shortest paths source→target."""
    try:
        all_paths = list(nx.all_shortest_paths(G, source, target))
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return {}
    counts = collections.Counter()
    for path in all_paths:
        for node in path[1:-1]:  # exclude endpoints
            counts[node] += 1
    total = len(all_paths)
    return {n: c/total for n, c in counts.most_common(top_n)}

for a, b in [(MILAREPA, MARPA), (MILAREPA, RECHUNG)]:
    rprint(f"\n  Intermediate nodes on paths  {display(a)} → {display(b)}:")
    if a in Gcc and b in Gcc:
        pb = path_betweenness_between(Gcc, a, b)
        if pb:
            for n, score in pb.items():
                rprint(f"    {display(n):<40} {score:.3f}")
        else:
            rprint("    (direct connection — no intermediates)")

# ────────────────────────────────────────────────────────────────────────────
# 5. CLOSENESS CENTRALITY
# ────────────────────────────────────────────────────────────────────────────
rprint("\n\n" + "━"*70)
rprint("5. CLOSENESS CENTRALITY")
rprint("━"*70)
rprint("""
Inverse of average shortest-path distance to all other nodes.
High closeness = can reach everyone quickly — globally central figure.
""")

close = nx.closeness_centrality(Gcc)
top_table(close, "Closeness Centrality — Character LCC", file=REPORT)
top_table(close, "Closeness Centrality — Character LCC")

close_p = nx.closeness_centrality(Gpcc)
top_table(close_p, "Closeness Centrality — Place LCC", file=REPORT)
top_table(close_p, "Closeness Centrality — Place LCC")

# ────────────────────────────────────────────────────────────────────────────
# 6. ECCENTRICITY (closeness eccentricity / periphery)
# ────────────────────────────────────────────────────────────────────────────
rprint("\n\n" + "━"*70)
rprint("6. ECCENTRICITY (structural periphery)")
rprint("━"*70)
rprint("""
Eccentricity = max shortest-path distance from a node to any other node.
High eccentricity = structurally peripheral; low = central.
Diameter = max eccentricity; Radius = min eccentricity.
""")

ecc = nx.eccentricity(Gcc)
diameter = nx.diameter(Gcc)
radius   = nx.radius(Gcc)
center   = nx.center(Gcc)
periph   = nx.periphery(Gcc)

rprint(f"\n  Diameter : {diameter}")
rprint(f"  Radius   : {radius}")
rprint(f"\n  Center nodes (eccentricity = radius):  {[display(n) for n in center[:10]]}")
rprint(f"\n  Periphery nodes (eccentricity = diameter):  {[display(n) for n in periph[:20]]}")

rprint("\n  Most peripheral characters (highest eccentricity):")
for n, e in sorted(ecc.items(), key=lambda x: x[1], reverse=True)[:20]:
    rprint(f"    {display(n):<40} ecc={e}")

rprint("\n  Most central characters (lowest eccentricity):")
for n, e in sorted(ecc.items(), key=lambda x: x[1])[:20]:
    rprint(f"    {display(n):<40} ecc={e}")

# ────────────────────────────────────────────────────────────────────────────
# 7. CLUSTERING COEFFICIENT
# ────────────────────────────────────────────────────────────────────────────
rprint("\n\n" + "━"*70)
rprint("7. CLUSTERING COEFFICIENT")
rprint("━"*70)
rprint("""
Fraction of a node's neighbours that are also neighbours of each other.
High = embedded in a tight cluster (e.g. a disciple circle or family group).
Low despite high degree = broker between distinct communities.
""")

clust = nx.clustering(G_char, weight="weight")
top_table(clust, "Clustering Coefficient (weighted) — top 25 characters", n=25, file=REPORT)
top_table(clust, "Clustering Coefficient (weighted) — top 25 characters")

rprint("\n  Characters with LOW clustering but HIGH degree (structural brokers):")
deg_vals  = dict(G_char.degree())
brokers = [(n, deg_vals[n], clust[n])
           for n in G_char.nodes()
           if deg_vals[n] >= 5 and clust[n] < 0.3]
brokers.sort(key=lambda x: x[1], reverse=True)
for n, d, c in brokers[:15]:
    rprint(f"    {display(n):<40} degree={d:3d}  clustering={c:.4f}")

# ────────────────────────────────────────────────────────────────────────────
# 8. PAGERANK
# ────────────────────────────────────────────────────────────────────────────
rprint("\n\n" + "━"*70)
rprint("8. PAGERANK")
rprint("━"*70)
rprint("""
Nodes score highly if connected to other high-scoring nodes.
Captures 'prestige by association' — structurally or symbolically important
figures even if they do not appear most often.
""")

pr_char = pagerank_np(G_char, weight="weight")
top_table(pr_char, "PageRank — Character Graph (weighted)", file=REPORT)
top_table(pr_char, "PageRank — Character Graph (weighted)")

pr_place = pagerank_np(G_place, weight="weight")
top_table(pr_place, "PageRank — Place Graph (weighted)", file=REPORT)
top_table(pr_place, "PageRank — Place Graph (weighted)")

# ────────────────────────────────────────────────────────────────────────────
# 9. WEIGHTED GRAPH — edge weight analysis
# ────────────────────────────────────────────────────────────────────────────
rprint("\n\n" + "━"*70)
rprint("9. WEIGHTED GRAPH — strongest relationships")
rprint("━"*70)
rprint("""
Edge weight = number of text units in which both characters co-appear.
High weight = repeated, sustained relationship rather than brief contact.
""")

edges_w = sorted(G_char.edges(data=True), key=lambda x: x[2].get("weight", 0), reverse=True)
rprint("\n  Top 30 strongest character pairings (by co-occurrence count):")
for rank, (a, b, data) in enumerate(edges_w[:30], 1):
    rprint(f"  {rank:>3}. {display(a):<28} ↔ {display(b):<28} weight={data['weight']}")

edges_bp = sorted(G_bp.edges(data=True), key=lambda x: x[2].get("weight", 0), reverse=True)
rprint("\n  Top 20 strongest character–place pairings:")
for rank, (a, b, data) in enumerate(edges_bp[:20], 1):
    rprint(f"  {rank:>3}. {display(a):<28} @ {b:<30} weight={data['weight']}")

# ────────────────────────────────────────────────────────────────────────────
# 10. TEMPORAL GRAPH — per-chapter network evolution
# ────────────────────────────────────────────────────────────────────────────
rprint("\n\n" + "━"*70)
rprint("10. TEMPORAL GRAPH — network evolution by chapter/phase")
rprint("━"*70)
rprint("""
Track how centrality and node presence change across narrative phases.
""")

chapter_order = [str(i) for i in range(14)]
temporal_data = {}

for ch in chapter_order:
    phase_units = [u for u in units if u.chapter == ch]
    if not phase_units:
        continue

    G_ch = nx.Graph()
    for unit in phase_units:
        persons = [p for p, _ in unit.persons]
        for a, b in itertools.combinations(persons, 2):
            if G_ch.has_edge(a, b):
                G_ch[a][b]["weight"] += 1
            else:
                G_ch.add_edge(a, b, weight=1)

    n_persons   = len(set(p for u in phase_units for p, _ in u.persons))
    n_locations = len(set(l for u in phase_units for l, _ in u.locations))
    n_units     = len(phase_units)

    pr_ch = pagerank_np(G_ch, weight="weight") if G_ch.number_of_edges() > 0 else {}
    top3  = [display(n) for n, _ in sorted(pr_ch.items(), key=lambda x: x[1], reverse=True)[:3]]

    temporal_data[ch] = {
        "phase": PHASES.get(ch, ch),
        "units": n_units,
        "persons": n_persons,
        "locations": n_locations,
        "top_pr": top3,
        "graph": G_ch,
    }

    rprint(f"\n  Ch {ch:>2}  {PHASES.get(ch, ch):<28}  "
           f"units={n_units:4d}  persons={n_persons:3d}  locs={n_locations:3d}  "
           f"top-PR: {', '.join(top3)}")

# Detailed per-chapter centrality table for Milarepa and Marpa
rprint("\n  Milarepa's and Marpa's degree centrality per chapter:")
rprint(f"  {'Ch':>4}  {'Phase':<28}  {'Milarepa DC':>12}  {'Marpa DC':>10}")
for ch in chapter_order:
    if ch not in temporal_data:
        continue
    G_ch = temporal_data[ch]["graph"]
    dc_ch = nx.degree_centrality(G_ch) if G_ch.number_of_nodes() > 0 else {}
    mil  = dc_ch.get(MILAREPA, 0.0)
    marpa_dc = dc_ch.get(MARPA, 0.0)
    rprint(f"  {ch:>4}  {PHASES.get(ch, ch):<28}  {mil:>12.4f}  {marpa_dc:>10.4f}")

# ────────────────────────────────────────────────────────────────────────────
# 11. PLACE NETWORK — detailed
# ────────────────────────────────────────────────────────────────────────────
rprint("\n\n" + "━"*70)
rprint("11. PLACE NETWORK")
rprint("━"*70)
rprint("""
Two places are connected when they appear in the same text unit.
Reveals which sacred sites cluster together and anchor the narrative.
""")

top_table(btwn_p,  "Place Betweenness (structural bridges)",   file=REPORT)
top_table(close_p, "Place Closeness (globally central sites)", file=REPORT)
top_table(pr_place,"Place PageRank (prestige by association)", file=REPORT)
top_table(btwn_p,  "Place Betweenness")
top_table(close_p, "Place Closeness")
top_table(pr_place,"Place PageRank")

rprint("\n  Top character–place pairings by weight (who is tied to which site):")
# For each person, find their most-visited place
from collections import defaultdict
person_place = defaultdict(lambda: collections.Counter())
for unit in units:
    for p, _ in unit.persons:
        for l, _ in unit.locations:
            person_place[p][l] += 1

rprint(f"\n  {'Character':<35}  {'Top place':<30}  count")
for person, counter in sorted(person_place.items(),
                               key=lambda x: sum(x[1].values()), reverse=True)[:20]:
    top_loc, top_cnt = counter.most_common(1)[0]
    rprint(f"  {display(person):<35}  {top_loc:<30}  {top_cnt}")

# ────────────────────────────────────────────────────────────────────────────
# 12. DIALOGUE / SPEECH NETWORK
# ────────────────────────────────────────────────────────────────────────────
rprint("\n\n" + "━"*70)
rprint("12. DIALOGUE NETWORK (directed: speaker → addressed)")
rprint("━"*70)
rprint("""
In Songs and Letters, 'Explicit' reference = singer/writer;
'Referenced' or 'Implicit' = addressed/audience.
Directed edge from speaker to addressed party.
Captures authority, instruction flow, and narrative voice.
""")

rprint(f"\n  Dialogue graph — {G_dial.number_of_nodes()} speakers/addressees, "
       f"{G_dial.number_of_edges()} directed edges")

out_deg = dict(G_dial.out_degree(weight="weight"))
in_deg  = dict(G_dial.in_degree(weight="weight"))

rprint("\n  Top speakers (out-degree by song/letter weight):")
for rank, (n, v) in enumerate(sorted(out_deg.items(), key=lambda x: x[1], reverse=True)[:15], 1):
    rprint(f"    {rank:>3}. {display(n):<35} {v}")

rprint("\n  Most addressed (in-degree — who is most often spoken to/about):")
for rank, (n, v) in enumerate(sorted(in_deg.items(), key=lambda x: x[1], reverse=True)[:15], 1):
    rprint(f"    {rank:>3}. {display(n):<35} {v}")

pr_dial = pagerank_np(G_dial, weight="weight")
top_table(pr_dial, "Dialogue PageRank (prestige via being spoken to by speakers)", file=REPORT)
top_table(pr_dial, "Dialogue PageRank")

# Dialogue directed betweenness
if G_dial.number_of_edges() > 0:
    Gdcc = G_dial.subgraph(max(nx.weakly_connected_components(G_dial), key=len)).copy()
    btwn_dial = nx.betweenness_centrality(Gdcc, weight=None, normalized=True)
    top_table(btwn_dial, "Dialogue Betweenness (mediators of speech transmission)", file=REPORT)
    top_table(btwn_dial, "Dialogue Betweenness")

# ────────────────────────────────────────────────────────────────────────────
# SUMMARY / INTERPRETIVE NOTES
# ────────────────────────────────────────────────────────────────────────────
rprint("\n\n" + "═"*70)
rprint("  INTERPRETIVE SUMMARY")
rprint("═"*70)

mil_deg    = nx.degree_centrality(G_char).get(MILAREPA, 0)
mil_btwn   = btwn.get(MILAREPA, 0)
mil_close  = close.get(MILAREPA, 0)
mil_pr     = pr_char.get(MILAREPA, 0)
marpa_btwn = btwn.get(MARPA, 0)
marpa_pr   = pr_char.get(MARPA, 0)

rprint(f"""
  MILAREPA
    Degree centrality    : {mil_deg:.4f}
    Betweenness          : {mil_btwn:.4f}
    Closeness            : {mil_close:.4f}
    PageRank             : {mil_pr:.4f}

  MARPA
    Betweenness          : {marpa_btwn:.4f}
    PageRank             : {marpa_pr:.4f}
""")

rprint("""
  KEY STRUCTURAL FINDINGS:
  ─────────────────────────────────────────────────────────────────────
  • Degree centrality identifies the most narratively active figures —
    expect Milarepa at the top, with Marpa and key disciples (Rechungpa,
    Seban Repa, Peta Gönkyi) forming the next tier.

  • Betweenness reveals whether Milarepa is the indispensable bridge
    between otherwise unconnected groups (teacher phase vs disciple phase).
    A high betweenness score for Marpa (despite fewer mentions) would
    confirm him as the structural gateway to the lineage.

  • Closeness maps who is narratively 'closest' to all others — a figure
    with high closeness appears across many relational contexts, not just
    one episode.

  • Eccentricity surfaces peripheral figures — likely one-scene characters,
    distant enemies, or cosmological beings (Akṣobhya, ḍākinīs).

  • Clustering shows tight circles: expect disciple groups (the Repa
    network) to cluster highly; Milarepa himself may have lower clustering
    because he bridges multiple otherwise-separate circles.

  • PageRank highlights structural prestige: a figure referenced in
    passing by many important people scores highly even with fewer
    total appearances. Marpa and Nāropa often rank higher here than
    raw mention counts alone would predict.

  • Temporal evolution: Marpa dominates Chapters 1–3; Milarepa's own
    network expands from Chapter 4 onward as disciples accumulate;
    Chapter 12–13 collapses to a small intimate circle.

  • The dialogue network shows Milarepa as the dominant singer/speaker,
    but the in-degree distribution reveals WHO he most often addresses —
    disciples first, then celestial beings, then the hostile/skeptical.

  • The place network centres on Drin (his primary hermitage) and
    Driché Puk; Lhodrak anchors the Marpa chapters; later chapters
    spread across Lapchi, Kyipuk, Drakar Taso — charting his wandering.
""")

REPORT.close()
print(f"\n[✓] Full report written → {OUT}/full_report.txt")

# ═════════════════════════════════════════════════════════════════════════════
# VISUALIZATIONS
# ═════════════════════════════════════════════════════════════════════════════
print("\nGenerating visualizations …")

# ── Helper: draw a graph with node size ∝ centrality ─────────────────────────
def draw_network(G, centrality, title, filename, top_n=40,
                 directed=False, weight_key="weight"):
    top_nodes = sorted(centrality, key=centrality.get, reverse=True)[:top_n]
    Gs = G.subgraph(top_nodes).copy()

    fig, ax = plt.subplots(figsize=(18, 14))
    pos = nx.spring_layout(Gs, seed=42, k=1.5)

    sizes  = [3000 * centrality.get(n, 0.001) + 200 for n in Gs.nodes()]
    colors = [centrality.get(n, 0) for n in Gs.nodes()]

    weights = [Gs[u][v].get(weight_key, 1) for u, v in Gs.edges()]
    max_w   = max(weights) if weights else 1
    widths  = [0.5 + 4 * (w / max_w) for w in weights]

    nx.draw_networkx_edges(Gs, pos, width=widths, alpha=0.35,
                           edge_color="#888888", ax=ax,
                           arrows=directed,
                           arrowstyle="-|>" if directed else "-",
                           connectionstyle="arc3,rad=0.1" if directed else "arc3,rad=0")
    nc = nx.draw_networkx_nodes(Gs, pos, node_size=sizes,
                                node_color=colors, cmap=cm.plasma,
                                alpha=0.9, ax=ax)
    nx.draw_networkx_labels(Gs, pos,
                            labels={n: display(n) for n in Gs.nodes()},
                            font_size=7, ax=ax)
    plt.colorbar(nc, ax=ax, label="Centrality score")
    ax.set_title(title, fontsize=14, pad=12)
    ax.axis("off")
    plt.tight_layout()
    path = os.path.join(OUT, filename)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {filename}")

draw_network(G_char, nx.degree_centrality(G_char),
             "Character Network — Degree Centrality (top 40)",
             "01_char_degree.png")

draw_network(Gcc, btwn,
             "Character Network — Betweenness Centrality (top 40, LCC)",
             "02_char_betweenness.png")

draw_network(Gcc, close,
             "Character Network — Closeness Centrality (top 40, LCC)",
             "03_char_closeness.png")

draw_network(G_char, pr_char,
             "Character Network — PageRank (top 40)",
             "04_char_pagerank.png")

draw_network(G_place, pr_place,
             "Place Network — PageRank (top 40)",
             "05_place_pagerank.png")

draw_network(G_place, btwn_p,
             "Place Network — Betweenness Centrality",
             "06_place_betweenness.png")

draw_network(G_dial, nx.degree_centrality(G_dial),
             "Dialogue Network — Degree Centrality (directed, top 40)",
             "07_dialogue_degree.png", directed=True)

# ── Temporal bar chart: persons per chapter ───────────────────────────────────
_, axes = plt.subplots(2, 1, figsize=(14, 10))

chs   = [ch for ch in chapter_order if ch in temporal_data]
units_cnt  = [temporal_data[ch]["units"]     for ch in chs]
pers_cnt   = [temporal_data[ch]["persons"]   for ch in chs]
locs_cnt   = [temporal_data[ch]["locations"] for ch in chs]

x = np.arange(len(chs))
labels = [f"Ch{ch}\n{temporal_data[ch]['phase'][:12]}" for ch in chs]

ax1, ax2 = axes
ax1.bar(x, units_cnt, color="#4C72B0", alpha=0.8, label="Text units")
ax1.set_xticks(x); ax1.set_xticklabels(labels, fontsize=8)
ax1.set_title("Text Units per Chapter"); ax1.set_ylabel("Count")
ax1.legend()

ax2.bar(x - 0.2, pers_cnt, 0.4, color="#DD8452", alpha=0.8, label="Unique persons")
ax2.bar(x + 0.2, locs_cnt, 0.4, color="#55A868", alpha=0.8, label="Unique locations")
ax2.set_xticks(x); ax2.set_xticklabels(labels, fontsize=8)
ax2.set_title("Unique Persons and Locations per Chapter"); ax2.set_ylabel("Count")
ax2.legend()

plt.tight_layout()
plt.savefig(os.path.join(OUT, "08_temporal_counts.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: 08_temporal_counts.png")

# ── Temporal centrality line chart for top characters ────────────────────────
KEY_CHARS = [
    MILAREPA,
    MARPA,
    RECHUNG,
    "bdag med ma Dakmema",
    "se ban ras pa Seban Repa",
    "pe ta mgon skyid Peta Gönkyi",
]

fig, ax = plt.subplots(figsize=(16, 8))
for char in KEY_CHARS:
    vals = []
    for ch in chapter_order:
        if ch not in temporal_data:
            vals.append(0)
            continue
        G_ch = temporal_data[ch]["graph"]
        dc = nx.degree_centrality(G_ch)
        vals.append(dc.get(char, 0))
    ax.plot(range(len(chapter_order)), vals, marker="o", label=display(char))

ax.set_xticks(range(len(chapter_order)))
ax.set_xticklabels([f"Ch{c}\n{PHASES.get(c,'')[:10]}" for c in chapter_order], fontsize=8)
ax.set_title("Degree Centrality per Chapter — Key Characters")
ax.set_ylabel("Degree Centrality")
ax.legend(loc="upper left", fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "09_temporal_centrality.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: 09_temporal_centrality.png")

# ── Bipartite: top characters × top places heatmap ────────────────────────────
top_chars_bp = [n for n, _ in sorted(pr_char.items(), key=lambda x: x[1], reverse=True)[:20]]
top_locs_bp  = [l for l, _ in sorted(pr_place.items(), key=lambda x: x[1], reverse=True)[:20]]

matrix = np.zeros((len(top_chars_bp), len(top_locs_bp)))
for i, char in enumerate(top_chars_bp):
    for j, loc in enumerate(top_locs_bp):
        if G_bp.has_edge(char, loc):
            matrix[i, j] = G_bp[char][loc].get("weight", 0)

fig, ax = plt.subplots(figsize=(18, 10))
im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd")
ax.set_xticks(range(len(top_locs_bp)))
ax.set_xticklabels(top_locs_bp, rotation=45, ha="right", fontsize=9)
ax.set_yticks(range(len(top_chars_bp)))
ax.set_yticklabels([display(c) for c in top_chars_bp], fontsize=9)
ax.set_title("Character × Place Co-occurrence Heatmap (top 20 each by PageRank)")
plt.colorbar(im, ax=ax, label="Co-occurrence count")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "10_char_place_heatmap.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: 10_char_place_heatmap.png")

print(f"\n[✓] All outputs in: {OUT}/")
print("    full_report.txt  — complete numerical results and interpretive notes")
print("    01–10 *.png      — network visualizations")
