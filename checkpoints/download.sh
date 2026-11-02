#!/usr/bin/env bash
set -euo pipefail
DEST="${1:-checkpoints/downloads}"
mkdir -p "$DEST"
if ! command -v gh >/dev/null 2>&1; then
  echo 'Install GitHub CLI (gh) first; authenticate with gh auth login for a private repository.' >&2
  exit 1
fi
gh release download checkpoint-best-v1 --repo yxy-919/LegalEvolver \
  --pattern 'LegalEvolver-Qwen3-8B.tar.part-*' --pattern SHA256SUMS \
  --pattern model.SHA256SUMS --dir "$DEST" --skip-existing
python3 - "$DEST" <<'PY'
import hashlib
from pathlib import Path
import sys
root = Path(sys.argv[1])
for line in (root / 'SHA256SUMS').read_text().splitlines():
    expected, name = line.split(maxsplit=1)
    path = root / name.lstrip('*')
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    if digest.hexdigest() != expected:
        raise SystemExit(f'Checksum mismatch: {path}. Remove this part and rerun the download.')
print('All archive parts verified.')
PY
cat "$DEST"/LegalEvolver-Qwen3-8B.tar.part-* | tar -xf - -C "$DEST"
python3 - "$DEST" <<'PY'
import hashlib
from pathlib import Path
import sys
root = Path(sys.argv[1])
for line in (root / 'model.SHA256SUMS').read_text().splitlines():
    expected, name = line.split(maxsplit=1)
    path = root / 'LegalEvolver-Qwen3-8B' / name.lstrip('*')
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    if digest.hexdigest() != expected:
        raise SystemExit(f'Model checksum mismatch: {path}')
print(f'Model verified: {root / "LegalEvolver-Qwen3-8B"}')
PY
