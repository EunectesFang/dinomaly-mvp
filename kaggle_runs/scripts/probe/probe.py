import os

ROOT = "/kaggle/input/datasets"
print("root exists", os.path.isdir(ROOT))
for dirpath, dirnames, filenames in os.walk(ROOT):
    depth = dirpath[len(ROOT):].count(os.sep)
    if depth > 3:
        dirnames[:] = []
        continue
    print(dirpath, "dirs=", dirnames[:10], "files=", filenames[:5])
    if depth == 2:
        dirnames[:] = dirnames[:3]
print("PROBE2_OK")
