"""Download a small MVTec subset through one Kaggle session.

All test images and masks for grid, transistor and screw, plus 40 evenly
spaced train/good images per class. The 40-image cap is the stage-8 memory
size and keeps this run inside the time limit.
"""
import os
import sys
import time

from kaggle.api.kaggle_api_extended import (
    KaggleApi,
    _extract_and_remove_zip,
    _is_auto_compressed,
    _resolve_download_path,
)
from kagglesdk.datasets.types.dataset_api_service import ApiDownloadDatasetRequest

DEST = r"e:\dinomaly-mvp\data\mvtec"
DATASET = "ipythonx/mvtec-ad"
PREFIXES = tuple(
    f"{name.strip().strip('/')}/" for name in sys.argv[1:]
) or ("grid/", "transistor/", "screw/")
TRAIN_KEEP = 40


def list_names(api):
    token = None
    chosen = []
    while True:
        response = api.dataset_list_files(DATASET, page_token=token, page_size=500)
        for item in response.files or []:
            name = item.name.replace("\\", "/")
            if name.startswith(PREFIXES):
                chosen.append(name)
        token = response.next_page_token
        if not token:
            break
    return chosen


def select(names):
    train = {prefix[:-1]: [] for prefix in PREFIXES}
    keep = []
    for name in names:
        parts = name.split("/")
        if len(parts) >= 4 and parts[1] == "train" and parts[2] == "good":
            train[parts[0]].append(name)
        elif "/test/" in name or "/ground_truth/" in name:
            keep.append(name)
    for prefix, paths in train.items():
        paths = sorted(paths)
        if len(paths) <= TRAIN_KEEP:
            picked = paths
        else:
            indexes = sorted({
                round(i * (len(paths) - 1) / (TRAIN_KEEP - 1))
                for i in range(TRAIN_KEEP)
            })
            picked = [paths[i] for i in indexes]
        keep.extend(picked)
        print(f"train {prefix}: available={len(paths)} kept={len(picked)}", flush=True)
    order = {"grid": 0, "transistor": 1, "screw": 2}
    keep.sort(key=lambda name: (order.get(name.split("/")[0], 9), name))
    return keep


def download_one(api, kaggle, name):
    local = os.path.join(DEST, name.replace("/", os.sep))
    if os.path.isfile(local) and os.path.getsize(local) > 0:
        return False
    request = ApiDownloadDatasetRequest()
    request.owner_slug = "ipythonx"
    request.dataset_slug = "mvtec-ad"
    request.file_name = name
    response = kaggle.datasets.dataset_api_client.download_dataset(request)
    outfile = _resolve_download_path(DEST, name, response.request.url)
    api.download_file(response, outfile, kaggle.http_client(), quiet=True)
    if _is_auto_compressed(outfile, name):
        _extract_and_remove_zip(outfile, os.path.dirname(outfile), unwrap_to=outfile[:-4])
    return True


def main():
    os.makedirs(DEST, exist_ok=True)
    api = KaggleApi()
    api.authenticate()
    names = select(list_names(api))
    print(f"selected={len(names)}", flush=True)
    done = 0
    with api.build_kaggle_client() as kaggle:
        for name in names:
            for attempt in range(4):
                try:
                    fetched = download_one(api, kaggle, name)
                    break
                except Exception as exc:
                    if attempt == 3:
                        raise
                    wait = 25 if "429" in str(exc) else 2
                    print(f"retry {name} {type(exc).__name__} sleep={wait}", flush=True)
                    time.sleep(wait)
            done += 1
            if done % 25 == 0 or done == len(names):
                print(f"downloaded {done}/{len(names)}", flush=True)
            if fetched:
                time.sleep(0.45)
    print("download complete", flush=True)


if __name__ == "__main__":
    main()
