import os
import zipfile

ZIP_PATH = "/tmp/trainer.zip"
DEST = "/app"

with zipfile.ZipFile(ZIP_PATH) as archive:
    if archive.testzip() is not None:
        raise RuntimeError("trainer.zip failed ZIP integrity check")

    for info in archive.infolist():
        name = info.filename.replace("\\", "/")
        if not name:
            continue
        target = os.path.normpath(os.path.join(DEST, name))
        if target != DEST and not target.startswith(DEST + os.sep):
            raise RuntimeError(f"Unsafe ZIP path: {info.filename!r}")
        if info.is_dir() or name.endswith("/"):
            os.makedirs(target, exist_ok=True)
            continue
        os.makedirs(os.path.dirname(target), exist_ok=True)
        with archive.open(info) as src, open(target, "wb") as dst:
            dst.write(src.read())

required = "/app/trainer/requirements-web.txt"
if not os.path.isfile(required):
    raise RuntimeError(f"Missing required file after extraction: {required}")
print("Trainer source extracted successfully; requirements-web.txt found.")
