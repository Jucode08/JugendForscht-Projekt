import importlib.metadata

deps_whisper = importlib.metadata.requires("openai-whisper")
print("-------------deps_whisper--------------")
print(deps_whisper)
print("---------------------------------------")

deps_deep_translator = importlib.metadata.requires("deep-translator")
print("-------------deps_deep_translator--------------")
print(deps_deep_translator)
print("---------------------------------------")

deps_gTTS = importlib.metadata.requires("gTTS")
print("-------------deps_gTTS--------------")
print(deps_gTTS)
print("---------------------------------------")

deps_numpy = importlib.metadata.requires("numpy")
print("-------------deps_numpy--------------")
print(deps_numpy)
print("---------------------------------------")

deps_langdetect = importlib.metadata.requires("langdetect")
print("-------------deps_langdetect--------------")
print(deps_langdetect)
print("---------------------------------------")

deps_sounddevice = importlib.metadata.requires("sounddevice")
print("-------------deps_sounddevice--------------")
print(deps_sounddevice)
print("---------------------------------------")


# 'more-itertools', 
# 'numba', 
# 'numpy', 
# 'tiktoken', 
# 'torch', 
# 'tqdm', 

# 'beautifulsoup4
# 'requests (>=2.23.0,<3.0.0)'

# click<8.2,>=7.1',

# six

# cffi