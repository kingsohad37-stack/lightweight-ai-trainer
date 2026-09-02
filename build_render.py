"""Prepare the bundled trainer source for Render's non-Docker Python runtime."""
from pathlib import Path
import shutil
import zipfile

ROOT = Path(__file__).resolve().parent
DEST = ROOT / "trainer"
ZIP = ROOT / "trainer.zip"
OVERRIDES = ROOT / "web-overrides"

if not ZIP.is_file():
    raise RuntimeError("trainer.zip is missing")

if DEST.exists():
    shutil.rmtree(DEST)

with zipfile.ZipFile(ZIP) as archive:
    if archive.testzip() is not None:
        raise RuntimeError("trainer.zip failed ZIP integrity check")
    for info in archive.infolist():
        name = info.filename.replace("\\", "/")
        if not name:
            continue
        target = (ROOT / name).resolve()
        if target != ROOT and ROOT not in target.parents:
            raise RuntimeError(f"Unsafe ZIP path: {info.filename!r}")
        if info.is_dir() or name.endswith("/"):
            target.mkdir(parents=True, exist_ok=True)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        with archive.open(info) as src, target.open("wb") as dst:
            shutil.copyfileobj(src, dst)

web = DEST / "web"
for filename in ("index.html", "app.js", "styles.css"):
    source = OVERRIDES / filename
    if source.is_file():
        web.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, web / filename)

required = DEST / "requirements-web.txt"
if not required.is_file():
    raise RuntimeError("Trainer extraction failed: trainer/requirements-web.txt not found")
print("Render build preparation complete.")
