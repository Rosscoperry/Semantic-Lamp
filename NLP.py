import speech_recognition as sr

r = sr.Recognizer()

mic = sr.Microphone()

with mic as source:
    audio = r.listen(source)

msg = r.recognize_google(audio)

print(msg)


# harvard = sr.AudioFile(r'D:\Python projects\LampProject\harvard.wav')
# with harvard as source:
#     audio = r.record(source)

# r.recognize_google(audio)