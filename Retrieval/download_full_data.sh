#!/usr/bin/env bash
set -euo pipefail
DEST="${1:-data/retrieval-full}"
BASE="https://github.com/yxy-919/LegalEvolver/releases/download/retrieval-data-v1"
ARCHIVE="LegalEvolver-retrieval-full-v1.tar.gz"
mkdir -p "$DEST"
curl -fL --retry 4 -o "$DEST/$ARCHIVE" "$BASE/$ARCHIVE"
curl -fL --retry 4 -o "$DEST/SHA256SUMS" "$BASE/SHA256SUMS"
python3 - "$DEST" "$ARCHIVE" <<'PY'
from pathlib import Path
import hashlib, sys
root, name = Path(sys.argv[1]), sys.argv[2]
expected = (root / 'SHA256SUMS').read_text().split()[0]
h = hashlib.sha256()
with (root / name).open('rb') as f:
    for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
        h.update(block)
if h.hexdigest() != expected:
    raise SystemExit('Checksum mismatch; download again.')
print('Archive SHA-256 verified.')
PY
tar -xzf "$DEST/$ARCHIVE" -C "$DEST"
printf 'Retrieval resources extracted to %s\n' "$DEST"
