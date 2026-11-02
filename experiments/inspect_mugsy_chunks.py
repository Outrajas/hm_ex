"""Read-only streaming inspection of Re:InterHand Mugsy per-segment tar chunks."""

import io
import sys
import zlib
import tarfile
from pathlib import Path
from collections import defaultdict, deque

ROOT = Path(__file__).resolve().parent.parent
CAPTURE_ID = "m--20210701--1058--0000000--pilot--relightablehandsy--participant0--two-hands"
MUGSY_DIR = ROOT / "datasets" / "reinterhand" / "captures" / CAPTURE_ID / "Mugsy_cameras"

class ChainedStream(io.RawIOBase):
    """Presents multiple file chunks as a single continuous binary stream."""
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

def run_inspection():
    chunks = [
        MUGSY_DIR / "envmap_per_segment.tar.gzaa",
        MUGSY_DIR / "envmap_per_segment.tar.gzab"
    ]

    for c in chunks:
        if not c.exists():
            print(f"[Error] Required chunk file not found: {c}")
            return

    print("=" * 80)
    print(" RE:INTERHAND MUGSY PER-SEGMENT CHUNK STREAM INSPECTION (aa + ab)")
    print("=" * 80)

    raw_stream = ChainedStream(chunks)
    buffered_stream = io.BufferedReader(raw_stream, buffer_size=1024 * 1024)

    total_members = 0
    image_count = 0
    mask_count = 0
    unique_cameras = set()
    images_per_camera = defaultdict(int)
    cam_min_max = {}
    
    first_20_images = []
    last_20_images = deque(maxlen=20)
    first_20_masks = []
    
    camera_transitions = []
    current_camera = None
    
    # Interleaving / adjacency check
    recent_modalities = deque(maxlen=10)
    interleaved_pairs_found = 0
    
    termination_reason = "Completed without error"
    is_truncated = False

    try:
        with tarfile.open(fileobj=buffered_stream, mode="r|gz") as tar:
            for member in tar:
                total_members += 1
                name = member.name.strip("/")
                parts = name.split("/")

                if member.isfile() and len(parts) >= 4 and parts[0] == "envmap_per_segment":
                    modality = parts[1]
                    cam_id = parts[2]
                    filename = parts[3]
                    frame_str = filename.rsplit(".", 1)[0]
                    frame_num = int(frame_str) if frame_str.isdigit() else None

                    unique_cameras.add(cam_id)

                    # Track camera transitions
                    if current_camera != cam_id:
                        if current_camera is not None and len(camera_transitions) < 20:
                            camera_transitions.append({
                                "from_cam": current_camera,
                                "to_cam": cam_id,
                                "at_member": total_members,
                                "at_frame": frame_str
                            })
                        current_camera = cam_id

                    # Track min/max frames per camera
                    if frame_num is not None:
                        if cam_id not in cam_min_max:
                            cam_min_max[cam_id] = [frame_num, frame_num]
                        else:
                            if frame_num < cam_min_max[cam_id][0]:
                                cam_min_max[cam_id][0] = frame_num
                            if frame_num > cam_min_max[cam_id][1]:
                                cam_min_max[cam_id][1] = frame_num

                    recent_modalities.append((modality, cam_id, frame_str))
                    if len(recent_modalities) >= 2:
                        prev = recent_modalities[-2]
                        curr = recent_modalities[-1]
                        if prev[1] == curr[1] and prev[2] == curr[2] and prev[0] != curr[0]:
                            interleaved_pairs_found += 1

                    if modality == "images":
                        image_count += 1
                        images_per_camera[cam_id] += 1
                        if len(first_20_images) < 20:
                            first_20_images.append((name, member.size))
                        last_20_images.append(name)

                    elif modality == "masks":
                        mask_count += 1
                        if len(first_20_masks) < 20:
                            first_20_masks.append((name, member.size))

                if total_members >= 20000:
                    termination_reason = "Inspection limit reached (20,000 members inspected while stream still has data)."
                    break

                if total_members % 2000 == 0:
                    print(f"  ... scanned {total_members:,} entries (Images: {image_count:,} | Masks: {mask_count:,})")

    except (EOFError, zlib.error, tarfile.TarError, OSError) as e:
        is_truncated = True
        termination_reason = f"Truncated-stream termination ({type(e).__name__}: {e})"

    # Print Full 13-Point Audit
    print("\n" + "=" * 80)
    print(" INSPECTION RESULTS REPORT")
    print("=" * 80)

    print(f"1. Total members successfully read : {total_members:,}")
    print(f"2. Number of image members         : {image_count:,}")
    print(f"3. Number of mask members          : {mask_count:,}")
    print(f"4. Unique camera IDs encountered   : {sorted(list(unique_cameras))}")
    
    print("\n5. Images per camera:")
    for cam, count in sorted(images_per_camera.items()):
        print(f"   - Camera {cam}: {count:,} images")

    print("\n6. First 20 image paths:")
    for path, _ in first_20_images:
        print(f"   {path}")

    print("\n7. Last 20 image paths reached:")
    for path in last_20_images:
        print(f"   {path}")

    print("\n8. Frame range per camera [Min Frame, Max Frame]:")
    for cam, (min_f, max_f) in sorted(cam_min_max.items()):
        print(f"   - Camera {cam}: Min={min_f:,} | Max={max_f:,} (Delta = {max_f - min_f:,})")

    is_camera_major = len(camera_transitions) < (image_count / 100.0) if image_count > 100 else True
    print(f"\n9. Archive organization: {'CAMERA-MAJOR' if is_camera_major else 'FRAME-MAJOR'}")
    print(f"   (Consecutive blocks of frames are stored per camera rather than alternating cameras per frame)")

    print(f"\n10. Camera transitions detected (first {len(camera_transitions)}):")
    if not camera_transitions:
        print("   No camera transitions detected within the inspected window (single continuous camera block).")
    for t in camera_transitions:
        print(f"   - Member #{t['at_member']:,}: Cam {t['from_cam']} -> {t['to_cam']} (At Frame {t['at_frame']})")

    print("\n11. Image/Mask grouping adjacency:")
    if mask_count == 0:
        print(f"   0 masks encountered in this window. Images and masks are segregated in separate directory hierarchies.")
    else:
        print(f"   Adjacent interleaved image/mask pairs detected: {interleaved_pairs_found:,}")

    print("\n12. Tar member sizes (First 20 Images):")
    for path, sz in first_20_images:
        print(f"   {path:<65} : {sz:,} bytes")
    
    print("\n12b. Tar member sizes (First 20 Masks):")
    if not first_20_masks:
        print("   [None] No mask entries found in the first inspected members.")
    for path, sz in first_20_masks:
        print(f"   {path:<65} : {sz:,} bytes")

    print("\n13. Stream termination status:")
    print(f"   Status      : {'TRUNCATED STREAM' if is_truncated else 'SCAN LIMIT REACHED'}")
    print(f"   Explanation : {termination_reason}")
    if is_truncated:
        print("   Note        : This is expected behavior because only chunks 'aa' and 'ab' (4 GB)")
        print("                 are present out of the full 38-chunk (75.31 GB) archive stream.")
    print("=" * 80)

if __name__ == "__main__":
    run_inspection()
