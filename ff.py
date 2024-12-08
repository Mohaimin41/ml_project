import pandas as pd
import gc

df = pd.read_csv('dataset/NF_CSECICID2018/data/NF-CSE-CIC-IDS2018-v2.csv')

print("READ OG CSV")

# Assume df is your existing DataFrame
# df = pd.read_csv('path/to/your/file.csv')  # Uncomment if loading from a CSV file

# Step 1: Select all rows where 'Label' is 1
df_label_1 = df[df['Label'] == 1]

print("SPLIT UP ATTACK SAMPLES")

# Step 2: Randomly sample 2,300,000 rows without replacement where 'Label' is 0
df_label_0 = df[df['Label'] == 0].sample(n=2300000, random_state=42)  # Set random_state for reproducibility

print("SPLIT UP BENIGN SAMPLES")

df = None

gc.collect()
# Step 3: Concatenate the two subsets
downsampled_df = pd.concat([df_label_1, df_label_0])

print("JOINED ")

df_label_0 = None
df_label_1 = None

gc.collect()

# Step 4: Write the downsampled DataFrame to a CSV file
downsampled_df.to_csv('downsampled_df.csv', index=False)
