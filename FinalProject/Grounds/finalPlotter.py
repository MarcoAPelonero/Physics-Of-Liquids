import pandas as pd
import glob
import matplotlib.pyplot as plt
import seaborn as sns

all_data = []

for filepath in glob.glob('results/*'):
    with open(filepath) as f:
        lines = f.readlines()
    cutoff = float(lines[0].split(':')[1].strip())
    df = pd.read_csv(
        filepath, 
        sep=r'\s+', 
        skiprows=1,
        engine='python'
    )
    df['cutoff_radius'] = cutoff
    all_data.append(df)

data = pd.concat(all_data, ignore_index=True)

sns.set(style="whitegrid")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

unique_ls = sorted(data['l'].unique())
palette = sns.color_palette("Purples", len(unique_ls))

sns.lineplot(
    data=data, x='cutoff_radius', y='Q_l', hue='l', 
    marker='o', palette=palette, ax=axes[0]
)
axes[0].set_title('Q_l vs Cutoff Radius', fontsize=14)
axes[0].set_xlabel('Cutoff Radius')
axes[0].set_ylabel('Q_l')
sns.despine(ax=axes[0], right=True, top=True)

sns.lineplot(
    data=data, x='cutoff_radius', y='Q_l_global', hue='l', 
    marker='o', palette=palette, ax=axes[1]
)
axes[1].set_title('Q_l_global vs Cutoff Radius', fontsize=14)
axes[1].set_xlabel('Cutoff Radius')
axes[1].set_ylabel('Q_l_global')

sns.despine(ax=axes[1], right=True, top=True)
plt.tight_layout()
plt.legend(title='l', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()
