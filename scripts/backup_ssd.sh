#!/usr/bin/env bash
# Mirror the small, irreplaceable SSD directories to the internal disk.
# Everything else on the SSD (raw/, interim/, processed/ lakes) is rebuildable:
# raw/ re-downloads from Fannie Mae; the lakes re-derive from raw/ via dev/pipeline.
#
# Usage: dev/tools/backup_ssd.sh        (run after every milestone gate)
# Destination: <repo-root>/ssd_mirror/  (gitignored; rides iCloud/Time Machine with the Mac)
set -euo pipefail

SRC="/Volumes/SSD Felipe/dissertation"
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
DEST="$REPO_ROOT/ssd_mirror"

[ -d "$SRC" ] || { echo "ERROR: SSD not mounted at $SRC"; exit 1; }
mkdir -p "$DEST"

# Note: no --delete on purpose — an accidental deletion on the SSD must not
# propagate to the backup. The mirror only ever accumulates.
for d in models outputs logs processed/macro processed/training; do
    if [ -d "$SRC/$d" ]; then
        if [ "$d" = "processed/training" ]; then
            # training shards are rebuildable — back up only the manifests
            mkdir -p "$DEST/$d"
            rsync -a --include='*/' --include='manifest.json' --exclude='*' "$SRC/$d/" "$DEST/$d/"
        else
            mkdir -p "$DEST/$d"
            rsync -a "$SRC/$d/" "$DEST/$d/"
        fi
        echo "backed up: $d"
    fi
done

echo "---"
du -sh "$DEST"
echo "Backup complete: $DEST  ($(date '+%Y-%m-%d %H:%M'))"
