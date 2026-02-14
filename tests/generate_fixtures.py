"""Generate test fixture images."""
import os
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

print("Fixtures generated successfully in", FIXTURE_DIR)
