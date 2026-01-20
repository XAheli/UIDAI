import pandas as pd

# Check biometric for duplicate patterns
print('=== CHECKING BIOMETRIC DATASET ===')
# Read one of the original files
df_orig = pd.read_csv('Dataset/corrected_dataset/biometric/final_main_corrected_biometric.csv', dtype=str)
print(f'Original records: {len(df_orig):,}')

# Find duplicates
duplicates = df_orig[df_orig.duplicated(keep=False)]
print(f'Total duplicate rows (all copies): {len(duplicates):,}')

if len(duplicates) > 0:
    # Get just the first few duplicated records
    dup_subset = duplicates.head(20).sort_values(['date', 'state', 'district', 'pincode'])
    print('\nSample of duplicate rows:')
    print(dup_subset[['date', 'state', 'district', 'pincode', 'bio_age_5_17', 'bio_age_17_']].to_string())
    
    # Check if duplicates are EXACT matches
    print('\n=== EXACT DUPLICATE ANALYSIS ===')
    first_dup_group = df_orig[df_orig.duplicated(keep=False)].head(10)
    print(first_dup_group.to_string())
    
    # Count unique vs duplicates
    print(f'\n=== STATISTICS ===')
    print(f'Unique records (first occurrence): {len(df_orig) - df_orig.duplicated().sum():,}')
    print(f'Duplicate copies to remove: {df_orig.duplicated().sum():,}')

print('\n=== CHECKING DEMOGRAPHIC DATASET ===')
df_demo = pd.read_csv('Dataset/corrected_dataset/demographic/corrected_api_data_aadhar_demographic_0_500000.csv', dtype=str, nrows=500000)
print(f'Sample records: {len(df_demo):,}')
print(f'Duplicate copies in sample: {df_demo.duplicated().sum():,}')

if df_demo.duplicated().sum() > 0:
    dup_demo = df_demo[df_demo.duplicated(keep=False)].head(10).sort_values(['date', 'state', 'district'])
    print('\nSample demographic duplicates:')
    print(dup_demo[['date', 'state', 'district', 'pincode']].to_string())
