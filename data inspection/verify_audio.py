from pydub import AudioSegment
audio = AudioSegment.from_mp3(r'C:\Users\ibrahim.fadhili\OneDrive - Agile Business Solutions\Desktop\ASR\kiswahili_asr\data\common_voice_swahili\clips\common_voice_sw_34995130.mp3')
audio.export('test.wav', format='wav')  # Convert one file to test