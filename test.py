import glob
import os

result = []

for x in os.walk("data/lpd_5"):
    for y in glob.glob(os.path.join(x[0], '*.npz')):
        result.append(y)

print(len(result))