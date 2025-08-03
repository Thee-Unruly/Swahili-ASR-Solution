import pandas as pd

# Load validated.tsv
validated_df = pd.read_csv(r'C:\Users\ibrahim.fadhili\OneDrive - Agile Business Solutions\Desktop\ASR\kiswahili_asr\data\common_voice_swahili\tsv_cleaned\validated_cleaned.tsv', sep='\t')
print(validated_df.head())  # View first few rows
print(f"Total entries: {len(validated_df)}")  # Check number of clips
print(validated_df['sentence'].isna().sum())  # Check for missing transcriptions
print(validated_df['up_votes'].describe())  # Check vote distribution