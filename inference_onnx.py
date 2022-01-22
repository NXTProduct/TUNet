import librosa
import numpy as np
import onnxruntime
import soundfile as sf
import torch
from config import CONFIG
from utils.utils import frame, overlap_add

window = CONFIG.DATA.window_size
stride = CONFIG.DATA.stride
input_audio, _ = librosa.load('test_samples/input.wav', sr=16000)
d = max(len(input_audio) // stride + 1, 2) * stride
input_audio = np.hstack((input_audio, np.zeros(d - len(input_audio))))

session = onnxruntime.InferenceSession('lightning_logs/best_model.onnx')
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

chunks = frame(input_audio, window, stride)
buffer = []
for chunk in chunks:
    chunk = chunk[np.newaxis, np.newaxis, :].astype(np.float32)
    pred = session.run(None, input_feed={input_name: chunk})
    buffer.append(torch.tensor(pred[0]))

buffer = torch.cat(buffer)
output_audio = overlap_add(buffer, window, stride, (1, 1, len(input_audio)))
output_audio = torch.squeeze(output_audio).numpy()
sf.write('out.wav', output_audio, samplerate=16000, subtype='PCM_16')
