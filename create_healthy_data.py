import pandas as pd

# Read the metadata and microbiome abundance data
metadata = pd.read_csv('course_data/metadata.csv')
microbiome = pd.read_csv('course_data/microbiome_relative_abundance.csv')

print(f"Total samples in metadata: {len(metadata)}")
print(f"Total samples in microbiome data: {len(microbiome)}")

# Check the data types and unique values in PATGROUPFINAL_C
print(f"PATGROUPFINAL_C data type: {metadata['PATGROUPFINAL_C'].dtype}")
print(f"Unique values in PATGROUPFINAL_C: {metadata['PATGROUPFINAL_C'].value_counts()}")

# Check if there are any string values that might be '8'
print(f"Unique values as strings: {metadata['PATGROUPFINAL_C'].astype(str).value_counts()}")

# Filter metadata for healthy samples (PATGROUPFINAL_C == 8 or == '8')
healthy_metadata = metadata[metadata['PATGROUPFINAL_C'] == 8]
if len(healthy_metadata) == 0:
    # Try with string comparison
    healthy_metadata = metadata[metadata['PATGROUPFINAL_C'].astype(str) == '8']
    
print(f"Healthy samples in metadata: {len(healthy_metadata)}")

# Get the SampleIDs of healthy samples
healthy_sample_ids = healthy_metadata['SampleID'].tolist()
print(f"First 5 healthy sample IDs: {healthy_sample_ids[:5]}")

# Filter microbiome data for healthy samples only
healthy_microbiome = microbiome[microbiome['SampleID'].isin(healthy_sample_ids)]
print(f"Healthy samples found in microbiome data: {len(healthy_microbiome)}")

# Merge the data - include metadata columns and microbiome abundance
# First merge on SampleID
healthy_combined = pd.merge(healthy_metadata, healthy_microbiome, on='SampleID', how='inner')
print(f"Final combined healthy dataset: {len(healthy_combined)}")

# Save the healthy-only dataset
output_filename = 'course_data/healthy_samples_only.csv'
healthy_combined.to_csv(output_filename, index=False)

print(f"\nHealthy-only dataset saved to: {output_filename}")
print(f"Dataset contains {len(healthy_combined)} samples with {len(healthy_combined.columns)} columns")
print(f"Columns include: {list(healthy_combined.columns[:10])}...")  # Show first 10 columns
