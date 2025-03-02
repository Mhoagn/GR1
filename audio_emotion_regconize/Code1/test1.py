import pydub
from pydub import AudioSegment 
from pydub.playback import play

audio = AudioSegment.from_wav("d:/GR1_github/audio_emotion_regconize/dataTrain/OAF_angry/OAF_back_angry.wav")  # Tải file audio
play(audio)