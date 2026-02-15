"""Generate test fixture images."""
import os
import struct
import numpy as np
from PIL import Image

FIXTURE_DIR = os.path.join(os.path.dirname(__file__), "fixtures")
os.makedirs(FIXTURE_DIR, exist_ok=True)


def make_gradient(w, h, channels=3):
    """Create a gradient image for deterministic testing."""
    arr = np.zeros((h, w, channels), dtype=np.uint8)
    for c in range(channels):
        arr[:, :, c] = np.linspace(0, 255, w, dtype=np.uint8)[np.newaxis, :]
    # Add vertical gradient
    arr[:, :, 0] = (arr[:, :, 0].astype(np.uint16) + np.linspace(0, 128, h, dtype=np.uint8)[:, np.newaxis]).clip(0, 255).astype(np.uint8)
    return arr


# 1920x1080 JPEG (landscape)
img = Image.fromarray(make_gradient(1920, 1080))
img.save(os.path.join(FIXTURE_DIR, "sample.jpg"), quality=95)

# 1920x1080 PNG
img.save(os.path.join(FIXTURE_DIR, "sample.png"))

# 4000x2000 landscape JPEG
img = Image.fromarray(make_gradient(4000, 2000))
img.save(os.path.join(FIXTURE_DIR, "landscape.jpg"), quality=95)

# 2000x4000 portrait JPEG
img = Image.fromarray(make_gradient(2000, 4000))
img.save(os.path.join(FIXTURE_DIR, "portrait.jpg"), quality=95)

# RGBA PNG (with alpha channel)
rgba = np.zeros((500, 500, 4), dtype=np.uint8)
rgba[:, :, :3] = make_gradient(500, 500)
rgba[:, :, 3] = np.linspace(0, 255, 500, dtype=np.uint8)[np.newaxis, :]
img = Image.fromarray(rgba, mode="RGBA")
img.save(os.path.join(FIXTURE_DIR, "rgba.png"))

# Grayscale JPEG
gray = np.linspace(0, 255, 800 * 600, dtype=np.uint8).reshape(600, 800)
img = Image.fromarray(gray, mode="L")
img.save(os.path.join(FIXTURE_DIR, "gray.jpg"), quality=95)

# Corrupt file (random bytes)
with open(os.path.join(FIXTURE_DIR, "corrupt.bin"), "wb") as f:
    f.write(os.urandom(1024))

# --- Phase 8 fixtures ---

# WebP image (convert from sample JPEG)
webp_img = Image.fromarray(make_gradient(640, 480))
webp_img.save(os.path.join(FIXTURE_DIR, "sample.webp"), quality=90)

# WebP landscape
webp_landscape = Image.fromarray(make_gradient(1920, 1080))
webp_landscape.save(os.path.join(FIXTURE_DIR, "landscape.webp"), quality=90)

# JPEG with EXIF orientation=6 (rotated 90 CW)
# Create a 400x200 image, save as JPEG, then inject EXIF orientation tag
exif_img = Image.fromarray(make_gradient(400, 200))
exif_img.save(os.path.join(FIXTURE_DIR, "exif_orient6.jpg"), quality=95)

# Use piexif-free approach: save with PIL exif data
import io

def make_exif_jpeg(width, height, orientation, filename):
    """Create a JPEG with a specific EXIF orientation tag."""
    arr = make_gradient(width, height)
    img = Image.fromarray(arr)
    # Build minimal EXIF data with orientation tag
    # EXIF structure: APP1 marker (FFE1) + length + "Exif\0\0" + TIFF header + IFD
    # Use little-endian (II) TIFF header
    ifd_entries = 1  # just orientation
    # TIFF header: byte order (II) + magic (42) + offset to first IFD (8)
    tiff_header = b'II' + struct.pack('<H', 42) + struct.pack('<I', 8)
    # IFD: count + entries + next IFD offset (0)
    ifd = struct.pack('<H', ifd_entries)
    # Orientation tag: tag=0x0112, type=SHORT(3), count=1, value
    ifd += struct.pack('<HHI', 0x0112, 3, 1)
    ifd += struct.pack('<HH', orientation, 0)  # value + padding
    ifd += struct.pack('<I', 0)  # next IFD offset

    exif_payload = b'Exif\x00\x00' + tiff_header + ifd
    exif_segment = b'\xff\xe1' + struct.pack('>H', len(exif_payload) + 2) + exif_payload

    # Save JPEG without EXIF first
    buf = io.BytesIO()
    img.save(buf, format='JPEG', quality=95)
    jpeg_bytes = buf.getvalue()

    # Inject EXIF after SOI marker (first 2 bytes = FF D8)
    out_bytes = jpeg_bytes[:2] + exif_segment + jpeg_bytes[2:]

    with open(os.path.join(FIXTURE_DIR, filename), 'wb') as f:
        f.write(out_bytes)

# Orientation 6: rotate 90 CW — a 400x200 image should appear as 200x400 after correction
make_exif_jpeg(400, 200, 6, "exif_orient6.jpg")

# Orientation 3: rotate 180 — dimensions stay the same but pixels are flipped
make_exif_jpeg(400, 200, 3, "exif_orient3.jpg")

# Orientation 1: normal (no transform) — for regression testing
make_exif_jpeg(400, 200, 1, "exif_orient1.jpg")

print("Fixtures generated successfully in", FIXTURE_DIR)
