"""
Create sample files from large augmented datasets for easy viewing in VS Code.
"""
import pandas as pd
from pathlib import Path

# Configuration
SAMPLE_SIZE = 10000  # Records per sample
OUTPUT_DIR = Path("Dataset/augmented/samples")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

datasets = {
    'biometric': 'Dataset/augmented/biometric_augmented.csv',
    'demographic': 'Dataset/augmented/demographic_augmented.csv',
    'enrollment': 'Dataset/augmented/enrollment_augmented.csv'
}

for name, filepath in datasets.items():
    print(f'\n=== Processing {name} ===')
    
    # Read sample
    df = pd.read_csv(filepath, nrows=SAMPLE_SIZE)
    
    # Save sample
    sample_file = OUTPUT_DIR / f"{name}_sample_{SAMPLE_SIZE}.csv"
    df.to_csv(sample_file, index=False)
    print(f'Created: {sample_file}')
    print(f'Records: {len(df):,}')
    print(f'Columns: {len(df.columns)}')
    print(f'Size: {sample_file.stat().st_size / 1024 / 1024:.2f} MB')

print('\n✓ Sample files created successfully!')
print(f'\nYou can now open files in: {OUTPUT_DIR}/')
print('\nTo view full files, use:')
print('  csvlens Dataset/augmented/demographic_augmented.csv')
print('  head -100 Dataset/augmented/demographic_augmented.csv | column -t -s,')
print('  python -c "import pandas as pd; print(pd.read_csv(\'Dataset/augmented/demographic_augmented.csv\', nrows=50))"')
