import pandas as pd

validated_df = pd.read_csv(r'C:\Users\ibrahim.fadhili\OneDrive - Agile Business Solutions\Desktop\ASR\kiswahili_asr\data\common_voice_swahili\tsv_cleaned\validated_cleaned.tsv', sep='\t')
print(validated_df.head())  # View first few rows

validated_df['sentence'] = validated_df['sentence'].str.lower().str.strip()
validated_df = validated_df[validated_df['sentence'].notna()]  # Remove rows with missing transcriptions