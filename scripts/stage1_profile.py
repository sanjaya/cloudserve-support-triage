"""Stage 1 discovery profile of development_tickets.json.
Reproduces every figure in the Stage 1 workbook draft.
Usage: python stage1_profile.py path/to/development_tickets.json
"""
import collections as C
import datetime
import json
import statistics as S
import sys

d = json.load(open(sys.argv[1]))
n = len(d)


def outcomes(v):
    H = lambda f: [t["history"][f] for t in v]
    m = len(v)
    return (f"n={m:<4} FCR={100*sum(H('first_contact_resolution'))/m:5.1f}% "
            f"esc={100*sum(H('escalated'))/m:5.1f}% CSAT={S.mean(H('csat_rating')):.2f} "
            f"median_min={S.median(H('resolution_time_minutes')):.0f} "
            f"mean_min={S.mean(H('resolution_time_minutes')):.0f} "
            f"repeat={100*sum(H('repeat_contact'))/m:4.1f}% "
            f"answerable={100*sum(t['labels']['answerable_from_docs'] for t in v)/m:5.1f}%")


def by(name, key):
    print(f"\n== by {name}")
    g = C.defaultdict(list)
    for t in d:
        g[key(t)].append(t)
    for k, v in sorted(g.items(), key=lambda x: -len(x[1])):
        print(f"{str(k):30} {outcomes(v)}")


print("tickets", n, "customers", len(set(t["customer_id"] for t in d)),
      "range", min(t["received_at"] for t in d), "to", max(t["received_at"] for t in d))
print("overall", outcomes(d))
by("channel", lambda t: t["channel"])
by("tier", lambda t: t["customer_tier"])
by("region", lambda t: t["customer_region"])
by("fluency", lambda t: t["language_fluency"])
by("tier x fluency", lambda t: (t["customer_tier"], t["language_fluency"]))
by("urgency", lambda t: t["labels"]["urgency"])
by("intent", lambda t: t["labels"]["intent"])
by("answerable", lambda t: t["labels"]["answerable_from_docs"])
by("escalated", lambda t: t["history"]["escalated"])
by("weekend", lambda t: datetime.date.fromisoformat(t["received_at"][:10]).weekday() >= 5)

esc = [t for t in d if t["history"]["escalated"]]
print("\nescalated", len(esc),
      "answerable", sum(t["labels"]["answerable_from_docs"] for t in esc),
      "expected auto", sum(t["labels"]["expected_route"] == "auto_respond" for t in esc))
tot = sum(t["history"]["resolution_time_minutes"] for t in d)
print("\n== effort proxy: share of volume vs share of resolution minutes")
for i, v in C.Counter(t["labels"]["intent"] for t in d).most_common():
    m = sum(t["history"]["resolution_time_minutes"] for t in d if t["labels"]["intent"] == i)
    print(f"{i:26} vol={100*v/n:4.1f}% effort={100*m/tot:4.1f}%")
ae = [t for t in esc if t["labels"]["answerable_from_docs"]]
print("answerable & escalated: vol", f"{100*len(ae)/n:.1f}%", "effort",
      f"{100*sum(t['history']['resolution_time_minutes'] for t in ae)/tot:.1f}%")
print("\nmust_not_auto_respond", C.Counter(t["labels"]["intent"] for t in d if t["labels"]["must_not_auto_respond"]))
print("expected_route", C.Counter(t["labels"]["expected_route"] for t in d))
print("weekday", C.Counter(datetime.date.fromisoformat(t["received_at"][:10]).strftime("%a") for t in d))
bodies = C.Counter(t["body"].strip().lower() for t in d)
print("distinct bodies", len(bodies), "top", bodies.most_common(1))
print("csat distribution", sorted(C.Counter(t["history"]["csat_rating"] for t in d).items()))
