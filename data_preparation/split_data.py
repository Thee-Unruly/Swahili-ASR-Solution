from sklearn.model_selection import train_test_split
import pandas as pd

validated_df = pd.read_csv(r'C:\Users\ibrahim.fadhili\OneDrive - Agile Business Solutions\Desktop\ASR\kiswahili_asr\data\common_voice_swahili\tsv_cleaned\validated_cleaned.tsv', sep='\t')
print(validated_df.head())  # View first few rows

train_df, temp_df = train_test_split(validated_df, test_size=0.2, random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
train_df.to_csv('train.tsv', sep='\t', index=False)
val_df.to_csv('val.tsv', sep='\t', index=False)
test_df.to_csv('test.tsv', sep='\t', index=False)