import base64
import pathlib
import zlib

root = pathlib.Path(__file__).resolve().parents[1]
src = (root / "stage12" / "masked_recon_v12.py").read_bytes()
blob = base64.b64encode(zlib.compress(src, 9)).decode("ascii")
target = pathlib.Path(__file__).with_name("hires_knn_v14.py")
text = target.read_text(encoding="utf-8")
needle = '_STAGE12_ZLIB = ""  # VENDOR'
if needle not in text:
    raise SystemExit("needle missing")
text = text.replace(needle, '_STAGE12_ZLIB = """' + blob + '"""', 1)
target.write_text(text, encoding="utf-8")
print("src", len(src), "blob", len(blob), "file", target.stat().st_size)
