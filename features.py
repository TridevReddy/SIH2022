from collections import Counter
import librosa
import numpy as np
import math
import statistics
import glob

def rms(list):
  ret = []
  for i in list:
      ret.append(i ** 2)
  m = round(sum(ret) / len(ret), 2)
  return math.sqrt(m)

# df_new = pd.DataFrame(columns=["Mean", "Median", "Mode", "Standard Deviation", "Zero Crossing Rate", "RootMean Square", "Amplitude envelope"])
# df_new = pd.DataFrame(columns=["Mode", "H/R"])
for file in glob.glob("/content/drive/MyDrive/Human Audio/Child volume/*.wav"):
  data, sr = librosa.load(file)
  final = []
  for i in np.arange(3, 15.5, 1.5):
    win_num = 1
    time_stamp = i
    split_frame = sr*time_stamp
    if i==3.0:
      new_data = data[:int(split_frame-1)]
      final.append(new_data)
    elif i==15.0:
      new_data = data[int(sr*(time_stamp-3)) : int(split_frame-1)]
      final.append(new_data)
      new_data2 = data[int(split_frame):]
      final.append(new_data2)
    else:
      new_data = data[int(sr*(time_stamp-3)) : int(split_frame-1)]
      final.append(new_data)

  c = Counter(final[0])
  mode_ = c.most_common(1)[0][0]
  # mean = round(sum(final[0]) / len(final[0]), 2)
  # variance = sum([((x - mean) ** 2) for x in final[0]]) / len(final[0])
  # std = round(variance ** 0.5, 2)
  # med = statistics.median(final[0])
  # zero_crosses = np.nonzero(np.diff(final[0][:] > 0))[0]
  # rms_ = rms(final[0])
  # ampl_env = max(final[0])
  # df_new.loc[len(df_new)] = [mean, med, mode_, std, zero_crosses.size, rms_, ampl_env]
  df_new.loc[len(df_new)] = [mode_, "0"]
  final.clear()
