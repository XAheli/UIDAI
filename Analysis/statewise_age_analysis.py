#!/usr/bin/env python3
"""
State-wise Age Distribution Analysis for UIDAI Aadhaar Data
- New Enrollment: age_0_5, age_5_17, age_18_greater
- Biometric Authentication: bio_age_5_17, bio_age_17_
Generates publication-quality charts for presentation
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

# ─── Configuration ───────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 18,
    'axes.labelsize': 15,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 13,
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures', 'age_analysis')
os.makedirs(OUTPUT_DIR, exist_ok=True)

BASE = os.path.join(os.path.dirname(__file__), '..', 'Dataset')

# ─── Load Data ───────────────────────────────────────────────────────────────
print("Loading datasets...")
bio_df = pd.read_csv(os.path.join(BASE, 'cleaned', 'biometric', 'biometric', 'final_cleaned_biometric.csv'))
enroll_df = pd.read_csv(os.path.join(BASE, 'cleaned', 'enrollment', 'enrollment', 'final_cleaned_enrollment.csv'))

print(f"Biometric: {len(bio_df):,} records | Enrollment: {len(enroll_df):,} records")
print(f"Biometric columns: {list(bio_df.columns)}")
print(f"Enrollment columns: {list(enroll_df.columns)}")

# ─── 1. STATE-WISE BIOMETRIC AGE DISTRIBUTION ────────────────────────────────
print("\n=== Biometric Age Distribution by State ===")
bio_state = bio_df.groupby('state')[['bio_age_5_17', 'bio_age_17_']].sum().reset_index()
bio_state.columns = ['state', 'Age 5-17', 'Age 17+']
bio_state['total'] = bio_state['Age 5-17'] + bio_state['Age 17+']
bio_state = bio_state.sort_values('total', ascending=False)
bio_state['pct_5_17'] = (bio_state['Age 5-17'] / bio_state['total'] * 100).round(1)
bio_state['pct_17_plus'] = (bio_state['Age 17+'] / bio_state['total'] * 100).round(1)

print(f"\nTop 15 states by biometric volume:")
print(bio_state.head(15)[['state', 'Age 5-17', 'Age 17+', 'total', 'pct_5_17', 'pct_17_plus']].to_string(index=False))

# ─── 2. STATE-WISE ENROLLMENT AGE DISTRIBUTION ───────────────────────────────
print("\n=== Enrollment Age Distribution by State ===")
enroll_state = enroll_df.groupby('state')[['age_0_5', 'age_5_17', 'age_18_greater']].sum().reset_index()
enroll_state.columns = ['state', 'Age 0-5', 'Age 5-17', 'Age 18+']
enroll_state['total'] = enroll_state['Age 0-5'] + enroll_state['Age 5-17'] + enroll_state['Age 18+']
enroll_state = enroll_state.sort_values('total', ascending=False)
enroll_state['pct_0_5'] = (enroll_state['Age 0-5'] / enroll_state['total'] * 100).round(1)
enroll_state['pct_5_17'] = (enroll_state['Age 5-17'] / enroll_state['total'] * 100).round(1)
enroll_state['pct_18_plus'] = (enroll_state['Age 18+'] / enroll_state['total'] * 100).round(1)

print(f"\nTop 15 states by enrollment volume:")
print(enroll_state.head(15)[['state', 'Age 0-5', 'Age 5-17', 'Age 18+', 'total', 'pct_0_5', 'pct_5_17', 'pct_18_plus']].to_string(index=False))

# ═══════════════════════════════════════════════════════════════════════════════
# CHART 1: Top 20 States — Biometric Age Distribution (Stacked Horizontal Bar)
# ═══════════════════════════════════════════════════════════════════════════════
print("\nGenerating Chart 1: Biometric stacked bar...")
top_bio = bio_state.head(20).iloc[::-1]  # reverse for horizontal

fig, ax = plt.subplots(figsize=(14, 10))
y = np.arange(len(top_bio))
bars1 = ax.barh(y, top_bio['Age 5-17'], height=0.65, label='Age 5–17', color='#4CAF50', edgecolor='white', linewidth=0.5)
bars2 = ax.barh(y, top_bio['Age 17+'], left=top_bio['Age 5-17'], height=0.65, label='Age 17+', color='#FF8F00', edgecolor='white', linewidth=0.5)

ax.set_yticks(y)
ax.set_yticklabels(top_bio['state'], fontsize=13)
ax.set_xlabel('Total Biometric Authentications', fontsize=15)
ax.set_title('State-wise Biometric Authentication by Age Group\n(Top 20 States)', fontsize=18, fontweight='bold', pad=16)
ax.legend(loc='lower right', fontsize=14, frameon=True, fancybox=True)
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f'{x/1e6:.1f}M' if x >= 1e6 else f'{x/1e3:.0f}K'))
ax.grid(axis='x', alpha=0.3, linestyle='--')

# Add percentage labels
for i, (idx, row) in enumerate(top_bio.iterrows()):
    if row['total'] > 50000:
        ax.text(row['total'] + row['total']*0.01, i, f"{row['pct_5_17']:.0f}% | {row['pct_17_plus']:.0f}%",
                va='center', fontsize=10, color='#555')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'chart1_biometric_statewise_age.png'))
plt.close()

# ═══════════════════════════════════════════════════════════════════════════════
# CHART 2: Top 20 States — Enrollment Age Distribution (Stacked Horizontal Bar)
# ═══════════════════════════════════════════════════════════════════════════════
print("Generating Chart 2: Enrollment stacked bar...")
top_enroll = enroll_state.head(20).iloc[::-1]

fig, ax = plt.subplots(figsize=(14, 10))
y = np.arange(len(top_enroll))
bars1 = ax.barh(y, top_enroll['Age 0-5'], height=0.65, label='Age 0–5', color='#1565C0', edgecolor='white', linewidth=0.5)
bars2 = ax.barh(y, top_enroll['Age 5-17'], left=top_enroll['Age 0-5'], height=0.65, label='Age 5–17', color='#4CAF50', edgecolor='white', linewidth=0.5)
bars3 = ax.barh(y, top_enroll['Age 18+'], left=top_enroll['Age 0-5'] + top_enroll['Age 5-17'], height=0.65, label='Age 18+', color='#FF8F00', edgecolor='white', linewidth=0.5)

ax.set_yticks(y)
ax.set_yticklabels(top_enroll['state'], fontsize=13)
ax.set_xlabel('Total New Enrollments', fontsize=15)
ax.set_title('State-wise New Enrollment by Age Group\n(Top 20 States)', fontsize=18, fontweight='bold', pad=16)
ax.legend(loc='lower right', fontsize=14, frameon=True, fancybox=True)
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f'{x/1e6:.1f}M' if x >= 1e6 else f'{x/1e3:.0f}K'))
ax.grid(axis='x', alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'chart2_enrollment_statewise_age.png'))
plt.close()

# ═══════════════════════════════════════════════════════════════════════════════
# CHART 3: Age Proportion Comparison — Top 15 States (Grouped Bar)
# ═══════════════════════════════════════════════════════════════════════════════
print("Generating Chart 3: Age proportion comparison...")
top15_bio = bio_state.head(15).copy()
top15_enroll = enroll_state.head(15).copy()

# Merge on state for comparison
merged = pd.merge(
    top15_bio[['state', 'pct_5_17', 'pct_17_plus']].rename(columns={'pct_5_17': 'Bio 5-17%', 'pct_17_plus': 'Bio 17+%'}),
    top15_enroll[['state', 'pct_0_5', 'pct_5_17', 'pct_18_plus']].rename(columns={'pct_0_5': 'Enroll 0-5%', 'pct_5_17': 'Enroll 5-17%', 'pct_18_plus': 'Enroll 18+%'}),
    on='state', how='outer'
)
merged = merged.dropna(subset=['Bio 5-17%']).sort_values('Bio 5-17%', ascending=False).head(15)

fig, axes = plt.subplots(1, 2, figsize=(18, 9))

# Left: Biometric age split
ax = axes[0]
x = np.arange(len(merged))
width = 0.35
ax.bar(x - width/2, merged['Bio 5-17%'], width, label='Age 5–17', color='#4CAF50', edgecolor='white')
ax.bar(x + width/2, merged['Bio 17+%'], width, label='Age 17+', color='#FF8F00', edgecolor='white')
ax.set_xticks(x)
ax.set_xticklabels(merged['state'], rotation=55, ha='right', fontsize=11)
ax.set_ylabel('Percentage (%)', fontsize=14)
ax.set_title('Biometric Age Split by State', fontsize=16, fontweight='bold')
ax.legend(fontsize=12)
ax.set_ylim(0, 100)
ax.axhline(50, color='gray', linestyle='--', alpha=0.5, linewidth=1)
ax.grid(axis='y', alpha=0.3)

# Right: Enrollment age split
ax = axes[1]
if 'Enroll 0-5%' in merged.columns:
    valid = merged.dropna(subset=['Enroll 0-5%'])
    x2 = np.arange(len(valid))
    width = 0.25
    ax.bar(x2 - width, valid['Enroll 0-5%'], width, label='Age 0–5', color='#1565C0', edgecolor='white')
    ax.bar(x2, valid['Enroll 5-17%'], width, label='Age 5–17', color='#4CAF50', edgecolor='white')
    ax.bar(x2 + width, valid['Enroll 18+%'], width, label='Age 18+', color='#FF8F00', edgecolor='white')
    ax.set_xticks(x2)
    ax.set_xticklabels(valid['state'], rotation=55, ha='right', fontsize=11)
ax.set_ylabel('Percentage (%)', fontsize=14)
ax.set_title('Enrollment Age Split by State', fontsize=16, fontweight='bold')
ax.legend(fontsize=12)
ax.set_ylim(0, 100)
ax.grid(axis='y', alpha=0.3)

plt.suptitle('State-wise Age Distribution Comparison: Biometric vs Enrollment', fontsize=19, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'chart3_age_proportion_comparison.png'))
plt.close()

# ═══════════════════════════════════════════════════════════════════════════════
# CHART 4: Heatmap — Child Ratio (5-17) per State for Biometric
# ═══════════════════════════════════════════════════════════════════════════════
print("Generating Chart 4: Child ratio heatmap...")
bio_all = bio_state.copy().sort_values('pct_5_17', ascending=False)

fig, ax = plt.subplots(figsize=(16, 10))
colors = plt.cm.RdYlGn_r(np.linspace(0.15, 0.85, len(bio_all)))
bars = ax.barh(range(len(bio_all)), bio_all['pct_5_17'], color=colors, edgecolor='white', linewidth=0.5, height=0.75)

ax.set_yticks(range(len(bio_all)))
ax.set_yticklabels(bio_all['state'], fontsize=11)
ax.set_xlabel('Child (Age 5–17) Percentage (%)', fontsize=15)
ax.set_title('Biometric Authentication: Child (5–17) Share by State\n(Sorted by child percentage)', fontsize=18, fontweight='bold', pad=16)
ax.axvline(50, color='red', linestyle='--', alpha=0.7, linewidth=1.5, label='50% line')

# Add value labels
for i, (idx, row) in enumerate(bio_all.iterrows()):
    ax.text(row['pct_5_17'] + 0.5, i, f"{row['pct_5_17']:.1f}%", va='center', fontsize=10, fontweight='600')

ax.legend(fontsize=13)
ax.set_xlim(0, max(bio_all['pct_5_17']) + 8)
ax.grid(axis='x', alpha=0.3, linestyle='--')
ax.invert_yaxis()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'chart4_biometric_child_ratio_all_states.png'))
plt.close()

# ═══════════════════════════════════════════════════════════════════════════════
# CHART 5: Enrollment — Infant (0-5) Share by State
# ═══════════════════════════════════════════════════════════════════════════════
print("Generating Chart 5: Enrollment infant share...")
enroll_all = enroll_state.copy().sort_values('pct_0_5', ascending=False)

fig, ax = plt.subplots(figsize=(16, 10))
colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(enroll_all)))
bars = ax.barh(range(len(enroll_all)), enroll_all['pct_0_5'], color=colors, edgecolor='white', linewidth=0.5, height=0.75)

ax.set_yticks(range(len(enroll_all)))
ax.set_yticklabels(enroll_all['state'], fontsize=11)
ax.set_xlabel('Infant (Age 0–5) Percentage (%)', fontsize=15)
ax.set_title('New Enrollment: Infant (0–5) Share by State\n(Sorted by infant percentage)', fontsize=18, fontweight='bold', pad=16)

for i, (idx, row) in enumerate(enroll_all.iterrows()):
    ax.text(row['pct_0_5'] + 0.3, i, f"{row['pct_0_5']:.1f}%", va='center', fontsize=10, fontweight='600')

ax.set_xlim(0, max(enroll_all['pct_0_5']) + 5)
ax.grid(axis='x', alpha=0.3, linestyle='--')
ax.invert_yaxis()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'chart5_enrollment_infant_share_states.png'))
plt.close()

# ═══════════════════════════════════════════════════════════════════════════════
# CHART 6: Treemap-style — Volume + Age composition (Top 20 Biometric)
# ═══════════════════════════════════════════════════════════════════════════════
print("Generating Chart 6: Biometric volume + age bubble chart...")
top20 = bio_state.head(20).copy()

fig, ax = plt.subplots(figsize=(14, 9))
scatter_sizes = (top20['total'] / top20['total'].max() * 2000) + 100
scatter = ax.scatter(top20['pct_5_17'], top20['pct_17_plus'],
                     s=scatter_sizes, c=top20['total'], cmap='YlOrRd',
                     alpha=0.75, edgecolors='#333', linewidth=1.5)

for _, row in top20.iterrows():
    ax.annotate(row['state'], (row['pct_5_17'], row['pct_17_plus']),
                fontsize=10, ha='center', va='bottom',
                fontweight='bold', color='#333')

cbar = plt.colorbar(scatter, ax=ax, shrink=0.8, pad=0.02)
cbar.set_label('Total Biometric Records', fontsize=13)
cbar.ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f'{x/1e6:.1f}M' if x >= 1e6 else f'{x/1e3:.0f}K'))

ax.plot([0, 100], [100, 0], 'k--', alpha=0.3, linewidth=1)  # diagonal reference
ax.set_xlabel('Child (5–17) Percentage (%)', fontsize=15)
ax.set_ylabel('Adult (17+) Percentage (%)', fontsize=15)
ax.set_title('Biometric Age Composition vs Volume\n(Bubble size = total records)', fontsize=18, fontweight='bold', pad=16)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'chart6_biometric_age_bubble.png'))
plt.close()

# ═══════════════════════════════════════════════════════════════════════════════
# CHART 7: Enrollment — 3-way age composition scatter
# ═══════════════════════════════════════════════════════════════════════════════
print("Generating Chart 7: Enrollment 3-way age scatter...")
top20e = enroll_state.head(20).copy()

fig, ax = plt.subplots(figsize=(14, 9))
scatter_sizes = (top20e['total'] / top20e['total'].max() * 2000) + 100
scatter = ax.scatter(top20e['pct_0_5'], top20e['pct_18_plus'],
                     s=scatter_sizes, c=top20e['pct_5_17'], cmap='viridis',
                     alpha=0.75, edgecolors='#333', linewidth=1.5)

for _, row in top20e.iterrows():
    ax.annotate(row['state'], (row['pct_0_5'], row['pct_18_plus']),
                fontsize=10, ha='center', va='bottom',
                fontweight='bold', color='#333')

cbar = plt.colorbar(scatter, ax=ax, shrink=0.8, pad=0.02)
cbar.set_label('Age 5–17 Percentage (%)', fontsize=13)

ax.set_xlabel('Infant (0–5) Percentage (%)', fontsize=15)
ax.set_ylabel('Adult (18+) Percentage (%)', fontsize=15)
ax.set_title('Enrollment Age Composition by State\n(Bubble size = total enrollments, color = 5–17 share)', fontsize=18, fontweight='bold', pad=16)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'chart7_enrollment_age_scatter.png'))
plt.close()

# ═══════════════════════════════════════════════════════════════════════════════
# CHART 8: Side-by-side — Per-state youth dependency ratio
# ═══════════════════════════════════════════════════════════════════════════════
print("Generating Chart 8: Youth dependency ratio comparison...")

# Biometric: child/adult ratio
bio_ratio = bio_state.copy()
bio_ratio['youth_ratio'] = bio_ratio['Age 5-17'] / bio_ratio['Age 17+']
bio_ratio = bio_ratio.sort_values('youth_ratio', ascending=False).head(20)

# Enrollment: child/adult ratio
enroll_ratio = enroll_state.copy()
enroll_ratio['youth_ratio'] = (enroll_ratio['Age 0-5'] + enroll_ratio['Age 5-17']) / enroll_ratio['Age 18+'].replace(0, 1)
enroll_ratio = enroll_ratio.sort_values('youth_ratio', ascending=False).head(20)

fig, axes = plt.subplots(1, 2, figsize=(18, 9))

ax = axes[0]
colors = ['#4CAF50' if r > 1 else '#FF8F00' for r in bio_ratio['youth_ratio']]
ax.barh(range(len(bio_ratio)), bio_ratio['youth_ratio'], color=colors, edgecolor='white', height=0.7)
ax.set_yticks(range(len(bio_ratio)))
ax.set_yticklabels(bio_ratio['state'], fontsize=11)
ax.set_xlabel('Child-to-Adult Ratio', fontsize=14)
ax.set_title('Biometric: Youth Dependency Ratio\n(Child 5–17 / Adult 17+)', fontsize=16, fontweight='bold')
ax.axvline(1.0, color='red', linestyle='--', alpha=0.7, label='Parity (1.0)')
for i, (_, row) in enumerate(bio_ratio.iterrows()):
    ax.text(row['youth_ratio'] + 0.02, i, f"{row['youth_ratio']:.2f}", va='center', fontsize=10)
ax.legend(fontsize=12)
ax.invert_yaxis()
ax.grid(axis='x', alpha=0.3)

ax = axes[1]
colors = ['#1565C0' if r > 1 else '#FF8F00' for r in enroll_ratio['youth_ratio']]
ax.barh(range(len(enroll_ratio)), enroll_ratio['youth_ratio'], color=colors, edgecolor='white', height=0.7)
ax.set_yticks(range(len(enroll_ratio)))
ax.set_yticklabels(enroll_ratio['state'], fontsize=11)
ax.set_xlabel('Child-to-Adult Ratio', fontsize=14)
ax.set_title('Enrollment: Youth Dependency Ratio\n(Child 0–17 / Adult 18+)', fontsize=16, fontweight='bold')
ax.axvline(1.0, color='red', linestyle='--', alpha=0.7, label='Parity (1.0)')
for i, (_, row) in enumerate(enroll_ratio.iterrows()):
    ax.text(row['youth_ratio'] + 0.02, i, f"{row['youth_ratio']:.2f}", va='center', fontsize=10)
ax.legend(fontsize=12)
ax.invert_yaxis()
ax.grid(axis='x', alpha=0.3)

plt.suptitle('Youth Dependency Ratio: Which States Have More Children?', fontsize=19, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'chart8_youth_dependency_ratio.png'))
plt.close()

# ═══════════════════════════════════════════════════════════════════════════════
# DETAILED STATISTICAL SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("DETAILED STATISTICAL ANALYSIS")
print("="*80)

print("\n─── BIOMETRIC AUTHENTICATION ───")
total_bio = bio_state['total'].sum()
total_5_17 = bio_state['Age 5-17'].sum()
total_17_plus = bio_state['Age 17+'].sum()
print(f"Total biometric records: {total_bio:,}")
print(f"  Age 5-17: {total_5_17:,} ({total_5_17/total_bio*100:.1f}%)")
print(f"  Age 17+:  {total_17_plus:,} ({total_17_plus/total_bio*100:.1f}%)")
print(f"  National child-to-adult ratio: {total_5_17/total_17_plus:.3f}")
print(f"\nState with highest child share: {bio_state.iloc[bio_state['pct_5_17'].values.argmax()]['state']} ({bio_state['pct_5_17'].max():.1f}%)")
print(f"State with lowest child share:  {bio_state.iloc[bio_state['pct_5_17'].values.argmin()]['state']} ({bio_state['pct_5_17'].min():.1f}%)")
print(f"Std dev of child share across states: {bio_state['pct_5_17'].std():.2f}%")

print("\n─── NEW ENROLLMENT ───")
total_enroll = enroll_state['total'].sum()
total_0_5 = enroll_state['Age 0-5'].sum()
total_5_17e = enroll_state['Age 5-17'].sum()
total_18_plus = enroll_state['Age 18+'].sum()
print(f"Total enrollment records: {total_enroll:,}")
print(f"  Age 0-5:  {total_0_5:,} ({total_0_5/total_enroll*100:.1f}%)")
print(f"  Age 5-17: {total_5_17e:,} ({total_5_17e/total_enroll*100:.1f}%)")
print(f"  Age 18+:  {total_18_plus:,} ({total_18_plus/total_enroll*100:.1f}%)")
print(f"  National youth ratio (0-17/18+): {(total_0_5+total_5_17e)/max(total_18_plus,1):.3f}")
print(f"\nState with highest infant (0-5) share: {enroll_state.iloc[enroll_state['pct_0_5'].values.argmax()]['state']} ({enroll_state['pct_0_5'].max():.1f}%)")
print(f"State with highest adult (18+) share: {enroll_state.iloc[enroll_state['pct_18_plus'].values.argmax()]['state']} ({enroll_state['pct_18_plus'].max():.1f}%)")

# ─── Cross-state comparison ─────────────────────────────────────────────────
print("\n─── CROSS-STATE INSIGHTS ───")
# States where children dominate biometric
child_dominant_bio = bio_state[bio_state['pct_5_17'] > 50]
print(f"States where children (5-17) > 50% of biometric: {len(child_dominant_bio)}")
for _, row in child_dominant_bio.iterrows():
    print(f"  - {row['state']}: {row['pct_5_17']:.1f}% children")

# States where infants are high in enrollment
high_infant = enroll_state[enroll_state['pct_0_5'] > 10]
print(f"\nStates where infants (0-5) > 10% of enrollment: {len(high_infant)}")
for _, row in high_infant.iterrows():
    print(f"  - {row['state']}: {row['pct_0_5']:.1f}% infants")

# States where adults dominate enrollment
adult_dom = enroll_state[enroll_state['pct_18_plus'] > 80]
print(f"\nStates where adults (18+) > 80% of enrollment: {len(adult_dom)}")
for _, row in adult_dom.iterrows():
    print(f"  - {row['state']}: {row['pct_18_plus']:.1f}% adults")

print(f"\n8 charts saved to: {os.path.abspath(OUTPUT_DIR)}")
print("Done!")
