"""Generate alison_icon.ico -- the A.L.I.S.O.N. Windows application icon.

Renders a cyan/violet "A" mark on the dark canvas and emits a multi-resolution
.ico (16/32/48/64/128/256) suitable for Nuitka/PyInstaller --windows-icon.
"""

import os

try:
    from PIL import Image, ImageDraw
except ImportError:  # pragma: no cover
    raise SystemExit("Pillow is required: pip install pillow")


CYAN = (0, 229, 255, 255)
VIOLET = (124, 77, 255, 255)
BG = (10, 12, 16, 255)

SIZES = [16, 32, 48, 64, 128, 256]


def render(size):
    img = Image.new("RGBA", (size, size), BG)
    d = ImageDraw.Draw(img)
    s = size
    # Stylised 'A'
    apex = (s * 0.50, s * 0.16)
    base_l = (s * 0.16, s * 0.86)
    base_r = (s * 0.84, s * 0.86)
    inner_l = (s * 0.40, s * 0.50)
    inner_r = (s * 0.60, s * 0.50)
    d.polygon([apex, base_r, inner_r, inner_l, base_l], fill=CYAN)
    # Crossbar in violet
    bar_w = max(1, s // 14)
    d.line([(s * 0.34, s * 0.66), (s * 0.66, s * 0.66)],
           fill=VIOLET, width=bar_w)
    return img


def main():
    # PIL's ICO writer resizes a single source to the requested `sizes`,
    # so render once at the largest size and let it derive the rest.
    big = render(max(SIZES))
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "alison_icon.ico")
    big.save(out, sizes=[(s, s) for s in SIZES])
    print(f"[icon] wrote {out}")


if __name__ == "__main__":
    main()
