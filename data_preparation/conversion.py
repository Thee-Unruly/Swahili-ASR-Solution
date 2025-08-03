import os
import pandas as pd
from pydub import AudioSegment


validated_df = pd.read_csv(r'C:\Users\ibrahim.fadhili\OneDrive - Agile Business Solutions\Desktop\ASR\kiswahili_asr\data\common_voice_swahili\tsv_cleaned\validated_cleaned.tsv', sep='\t')
print(validated_df.head())  # View first few rows

input_dir = r'C:\Users\ibrahim.fadhili\OneDrive - Agile Business Solutions\Desktop\ASR\kiswahili_asr\data\common_voice_swahili\clips'
output_dir = r'C:\Users\ibrahim.fadhili\OneDrive - Agile Business Solutions\Desktop\ASR\kiswahili_asr\data\common_voice_swahili\wav'
os.makedirs(output_dir, exist_ok=True)

for mp3_file in validated_df['path']:
    mp3_path = os.path.join(input_dir, mp3_file)
    wav_file = mp3_file.replace('.mp3', '.wav')
    wav_path = os.path.join(output_dir, wav_file)
    try:
        audio = AudioSegment.from_mp3(mp3_path)
        audio = audio.set_channels(1).set_frame_rate(16000)  # Mono, 16 kHz
        audio.export(wav_path, format='wav')
    except Exception as e:
        print(f"Error converting {mp3_file}: {e}")