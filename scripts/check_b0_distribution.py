import json
from collections import Counter

class_names = ['MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC']

with open('C:/Users/Dishitha Ramesh/Desktop/ml project/csda_al/results/runs/test_b0/oracle_ledger.json') as f:
    ledger = json.load(f)

init_labels = ledger['round_0000']['acquired_labels']
init_counts = Counter(init_labels)

acquired_labels = []
for r in ['round_0001', 'round_0002', 'round_0003', 'round_0004', 'round_0005', 'round_0006']:
    acquired_labels.extend(ledger[r]['acquired_labels'])
acq_counts = Counter(acquired_labels)

total_labels = init_labels + acquired_labels
total_counts = Counter(total_labels)

print("Class Distribution Breakdown (B0):")
print(f"{'Class':<6} | {'Initial (N=80)':<14} | {'Acquired (N=240)':<17} | {'Total Labeled (N=320)':<21} | {'Total %':<8}")
print("-" * 75)
for i, name in enumerate(class_names):
    c_init = init_counts.get(i, 0)
    c_acq = acq_counts.get(i, 0)
    c_tot = total_counts.get(i, 0)
    p_init = (c_init / len(init_labels)) * 100
    p_acq = (c_acq / len(acquired_labels)) * 100
    p_tot = (c_tot / len(total_labels)) * 100
    print(f"{name:<6} | {c_init:>4} ({p_init:>5.1f}%) | {c_acq:>5} ({p_acq:>5.1f}%) | {c_tot:>6} ({p_tot:>5.1f}%) | {p_tot:>5.1f}%")

print("\nPer-round acquisition breakdown:")
for r in ['round_0000', 'round_0001', 'round_0002', 'round_0003', 'round_0004', 'round_0005', 'round_0006']:
    lbls = ledger[r]['acquired_labels']
    cnts = Counter(lbls)
    row_str = " | ".join([f"{class_names[i]}: {cnts.get(i, 0):>2}" for i in range(8)])
    print(f"{r} (N={len(lbls)}): {row_str}")
