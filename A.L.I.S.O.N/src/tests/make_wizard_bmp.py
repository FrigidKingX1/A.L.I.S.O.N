"""Generate assets/wizard_logo.bmp -- the A.L.I.S.O.N. Inno Setup wizard bitmap.

Renders a cyan/violet "A" mark on the dark brand canvas at the standard Inno
wizard bitmap size (164x314). Saved as 24-bit RGB BMP (Inno requires RGB, not
RGBA). Also usable as the small wizard image.
"""

import os

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:  # pragma: no cover
    raise SystemExit("Pillow is required: pip install pillow")


CYAN = (0, 229, 255)
VIOLET = (124, 77, 255)
BG = (10, 12, 16)
W, H = 164, 314


def main():
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)

    # Soft violet glow band behind the mark
    d.rectangle([(0, 60), (W, 200)], fill=(16, 14, 28))

    s = 120
    cx = W / 2
    top = 70
    apex = (cx, top)
    base_l = (cx - s * 0.34, top + s * 0.70)
    base_r = (cx + s * 0.34, top + s * 0.70)
    inner_l = (cx - s * 0.14, top + s * 0.34)
    inner_r = (cx + s * 0.14, top + s * 0.34)
    d.polygon([apex, base_r, inner_r, inner_l, base_l], fill=CYAN)
    d.line([(cx - s * 0.20, top + s * 0.50), (cx + s * 0.20, top + s * 0.50)],
           fill=VIOLET, width=max(2, s // 16))

    # Wordmark
    try:
        font = ImageFont.truetype("arial.ttf", 22)
    except OSError:
        font = ImageFont.load_default()
    text = "A.L.I.S.O.N."
    tw = d.textlength(text, font=font)
    d.text((cx - tw / 2, 210), text, fill=CYAN, font=font)
    try:
        sub = ImageFont.truetype("arial.ttf", 11)
    except OSError:
        sub = ImageFont.load_default()
    tag = "Adaptive Learning Interface"
    sw = d.textlength(tag, font=sub)
    d.text((cx - sw / 2, 240), tag, fill=(150, 160, 170), font=sub)

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "wizard_logo.bmp")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    img.save(out, "BMP")
    print(f"[wizard] wrote {out}")


if __name__ == "__main__":
    main()
