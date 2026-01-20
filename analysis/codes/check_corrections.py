import pandas as pd
import os
import warnings
warnings.filterwarnings('ignore')

# Check all corrected datasets
corrected_dir = "Dataset/corrected_dataset"

print("=" * 80)
print("CHECKING STATE NAMES STANDARDIZATION")
print("=" * 80)

datasets = {
    'biometric': 'biometric/final_main_corrected_biometric.csv',
    'demographic': 'demographic/corrected_api_data_aadhar_demographic_0_500000.csv',
    'enrollment': 'enrollement/corrected_api_data_aadhar_enrolment_0_500000.csv'
}

# Check unique states in biometric (consolidated dataset)
print("\n" + "=" * 80)
print("UNIQUE STATES IN BIOMETRIC DATASET:")
print("=" * 80)
bio_df = pd.read_csv(os.path.join(corrected_dir, 'biometric/final_main_corrected_biometric.csv'))
states = sorted(bio_df['state'].unique())
print(f"\nTotal unique states: {len(states)}")
for state in states:
    count = len(bio_df[bio_df['state'] == state])
    print(f"  {state}: {count} records")

# Check for any inconsistencies in formatting
print("\n" + "=" * 80)
print("CHECKING FOR FORMATTING ISSUES:")
print("=" * 80)
issues = []

# Check for leading/trailing spaces
for state in states:
    if state != state.strip():
        issues.append(f"  ⚠️  State '{state}' has leading/trailing spaces")

# Check for mixed case issues
for state in states:
    # Count how many different ways this state appears (case-insensitive)
    similar = [s for s in states if s.lower() == state.lower()]
    if len(similar) > 1:
        issues.append(f"  ⚠️  Multiple case variations for: {similar}")

if not issues:
    print("✓ No formatting issues detected!")
else:
    for issue in set(issues):  # Remove duplicates
        print(issue)

# Compare with original data if available
print("\n" + "=" * 80)
print("SAMPLE COMPARISON: ORIGINAL vs CORRECTED")
print("=" * 80)

original_files = [
    "Dataset/api_data_aadhar_biometric/api_data_aadhar_biometric_0_500000.csv",
    "Dataset/api_data_aadhar_biometric/api_data_aadhar_biometric_500000_1000000.csv"
]

original_states = set()
for file in original_files:
    if os.path.exists(file):
        df = pd.read_csv(file)
        original_states.update(df['state'].unique())

print(f"\nStates in original data: {len(original_states)}")
print(f"States in corrected data: {len(states)}")

# Check if all original states have been mapped
original_states_sorted = sorted(original_states)
print("\nOriginal states (first 10):")
for state in original_states_sorted[:10]:
    print(f"  '{state}'")

print("\nCorrected states (first 10):")
for state in states[:10]:
    print(f"  '{state}'")

# Check districts
print("\n" + "=" * 80)
print("DISTRICT CONSISTENCY CHECK:")
print("=" * 80)
districts = bio_df['district'].unique()
print(f"Total unique districts: {len(districts)}")
print(f"Sample districts (first 10):")
for district in sorted(districts)[:10]:
    print(f"  '{district}'")

# Check for potential issues with districts
district_issues = []
for district in districts:
    if district != district.strip():
        district_issues.append(f"  ⚠️  District '{district}' has leading/trailing spaces")
    if pd.isna(district):
        district_issues.append(f"  ⚠️  Found NaN in districts")

if district_issues:
    print("\nDistrict Issues:")
    for issue in set(district_issues):
        print(issue)
else:
    print("\n✓ No district formatting issues detected!")

# Check pincodes
print("\n" + "=" * 80)
print("PINCODE CONSISTENCY CHECK:")
print("=" * 80)
pincodes = bio_df['pincode'].unique()
print(f"Total unique pincodes: {len(pincodes)}")
print(f"Sample pincodes: {sorted(pincodes)[:10]}")

# Check if all pincodes are valid
invalid_pincodes = []
for pincode in pincodes:
    if pd.isna(pincode):
        invalid_pincodes.append("NaN")
    elif str(pincode).strip() == '':
        invalid_pincodes.append("Empty string")
    elif len(str(int(pincode))) != 6:
        invalid_pincodes.append(f"Invalid length: {pincode}")

if invalid_pincodes:
    print(f"\nPincode Issues Found: {len(set(invalid_pincodes))}")
    for issue in set(invalid_pincodes):
        print(f"  ⚠️  {issue}")
else:
    print("\n✓ All pincodes appear valid!")

print("\n" + "=" * 80)
