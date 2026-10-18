"""Inspects internal tar member ordering from local .tar.gz chunks without extraction."""

import io
import tarfile
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parent.parent
CAPTURE_ID = "m--20210701--1058--0000000--pilot--relightablehandsy--participant0--two-hands"
MUGSY_DIR = ROOT / "datasets" / "reinterhand" / "captures" / CAPTURE_ID / "Mugsy_cameras"

class ChainedStream(io.RawIOBase):
    """Presents multiple file chunks as a continuous read-only binary stream."""
    def __init__(self, file_paths):
        self.file_paths = [p for p in file_paths if p.exists()]
        self.current_idx = 0
        self.current_file = open(self.file_paths[0], "rb") if self.file_paths else None

    def readable(self):
        return True

    def readinto(self, b):
        if not self.current_file:
            return 0
        n = self.current_file.readinto(b)
        if n == 0:
            self.current_file.close()
            self.current_idx += 1
            if self.current_idx < len(self.file_paths):
                self.current_file = open(self.file_paths[self.current_idx], "rb")
                return self.readinto(b)
            else:
                self.current_file = None
                return 0
        return n

    def close(self):
        if self.current_file:
            self.current_file.close()
        super().close()

def inspect_ordering():
    print("=" * 70)
    print(" TAR INTERNAL ORDERING & LAYOUT PROBE")
    print("=" * 70)

    chunks = [
        MUGSY_DIR / "envmap_per_segment.tar.gzaa",
        MUGSY_DIR / "envmap_per_segment.tar.gzab"
    ]
    
    stream = ChainedStream(chunks)
    members_inspected = 0
    cameras_seen = defaultdict(int)
    directories_seen = set()
    first_entries = []
    
    try:
        # Stream read: does not seek, unpacks on the fly
        with tarfile.open(fileobj=stream, mode="r|gz") as tar:
            for member in tar:
                members_inspected += 1
                name = member.name
                if members_inspected <= 25:
                    first_entries.append((name, member.size))
                
                parts = name.split("/")
                if len(parts) > 1:
                    directories_seen.add(parts[0])
                if len(parts) >= 3 and parts[0] in ("images", "masks"):
                    cameras_seen[f"{parts[0]}/{parts[1]}"] += 1
                
                # Sample up to 1,000 headers then summarize
                if members_inspected >= 1000:
                    break

    except (tarfile.TarError, EOFError, OSError) as e:
        print(f"\n[Stream Ended] Reached chunk boundary or stream end: {e}")

    print(f"\nInspected {members_inspected:,} archive entries from local chunks.")
    
    print("\n--- FIRST 20 ARCHIVE ENTRIES ---")
    for path, size in first_entries[:20]:
        print(f"  {path:<60} ({size:,} bytes)")

    print("\n--- TOP-LEVEL DIRECTORIES IN TAR ---")
    for d in sorted(directories_seen):
        print(f"  - {d}")

    print("\n--- CAMERA / MODALITY GROUPINGS DETECTED ---")
    for cam_mod, count in sorted(cameras_seen.items()):
        print(f"  {cam_mod:<30}: {count} files in sample window")

    print("\n" + "=" * 70)

if __name__ == "__main__":
    inspect_ordering()
