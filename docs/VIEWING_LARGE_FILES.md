# Viewing Large CSV Files

The augmented CSV files are too large (265-603 MB) for VS Code's CSV editor to handle. Here are several ways to view and explore them:

## 📁 Sample Files (Open in VS Code)

Small sample files that VS Code can easily open:
```bash
Dataset/augmented/samples/
├── biometric_sample_10000.csv      # First 10K records
├── demographic_sample_10000.csv    # First 10K records
└── enrollment_sample_10000.csv     # First 10K records
```

**Open these in VS Code** - they're small enough to view comfortably.

## 🔍 Using csvlens (Recommended for Full Files)

You have csvlens installed - it's perfect for large CSV files:

```bash
# Open in interactive viewer
csvlens Dataset/augmented/demographic_augmented.csv
csvlens Dataset/augmented/biometric_augmented.csv
csvlens Dataset/augmented/enrollment_augmented.csv
```

**Keyboard shortcuts in csvlens:**
- `↑↓←→` - Navigate
- `/` - Search
- `Tab` - Switch between views
- `q` - Quit

## 🐍 Python/Pandas (For Analysis)

```python
import pandas as pd

# Load first N rows
df = pd.read_csv('Dataset/augmented/demographic_augmented.csv', nrows=100)
print(df.head())

# Or use chunks for full analysis
for chunk in pd.read_csv('Dataset/augmented/demographic_augmented.csv', chunksize=10000):
    # Process each 10K chunk
    print(chunk.describe())
```

## 📊 Quick Command-Line Views

```bash
# View first 10 rows (formatted)
head -10 Dataset/augmented/demographic_augmented.csv | column -t -s,

# View column headers
head -1 Dataset/augmented/demographic_augmented.csv | tr ',' '\n' | nl

# Count total records
wc -l Dataset/augmented/demographic_augmented.csv

# Search for specific data
grep -i "telangana" Dataset/augmented/demographic_augmented.csv | head

# Get file info
ls -lh Dataset/augmented/*.csv
```

## 🔢 Statistics Scripts

```bash
# Quick statistics
python -c "
import pandas as pd
df = pd.read_csv('Dataset/augmented/demographic_augmented.csv')
print('Records:', len(df))
print('Columns:', list(df.columns))
print('\nState distribution:')
print(df['state'].value_counts().head(10))
print('\nSummary statistics:')
print(df.describe())
"
```

## 💡 Tips

1. **Sample files** are in `Dataset/augmented/samples/` - open these in VS Code
2. **csvlens** is the best tool for browsing full files interactively
3. **Python/pandas** for analysis and transformations
4. **Command-line tools** for quick checks

## 🚀 Create Custom Samples

```bash
# Create a new sample (edit create_samples.py to customize)
python create_samples.py

# Filter specific states
python -c "
import pandas as pd
df = pd.read_csv('Dataset/augmented/demographic_augmented.csv')
telangana = df[df['state'] == 'Telangana']
telangana.to_csv('Dataset/augmented/samples/telangana_only.csv', index=False)
print(f'Saved {len(telangana)} Telangana records')
"
```
