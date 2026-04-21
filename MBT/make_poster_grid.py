"""
Rearrange mbt_poster_vis/gradcam_grid.png (12 rows × 4 cols) into poster figures.

- Horizontal: 4 panel-rows × N sample-columns → mbt_poster_vis/gradcam_grid_poster.png
- Vertical:   N sample-rows × 4 panel-columns → mbt_poster_vis/gradcam_grid_poster_vertical.png

Strips baked-in matplotlib titles from each cell. Labels: bold black.
"""

from PIL import Image, ImageDraw, ImageFont
import os

INPUT = "mbt_poster_vis/gradcam_grid.png"
OUTPUT_HORIZONTAL = "mbt_poster_vis/gradcam_grid_poster.png"
OUTPUT_VERTICAL = "mbt_poster_vis/gradcam_grid_poster_vertical.png"

SELECTED_ROWS_1IDX = [3, 4, 6, 7, 8, 9, 10]

TOTAL_ROWS = 12
TOTAL_COLS = 4
OUTPUT_DPI = 300

ROW_LABELS = ["Tactile", "Tac. CAM", "RGB", "RGB CAM"]

TITLE_CROP_FRAC = 0.13
CELL_GAP = 6
LABEL_FONT_PX = 35


def load_font(size, bold=False):
    if bold:
        candidates = [
            "/usr/share/fonts/liberation/LiberationSans-Bold.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
        ]
    else:
        candidates = [
            "/usr/share/fonts/liberation/LiberationSans-Regular.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
        ]
    for p in candidates:
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def _geometry(img):
    W, H = img.size
    cell_w = W // TOTAL_COLS
    cell_h = H // TOTAL_ROWS
    crop_top = int(cell_h * TITLE_CROP_FRAC)
    content_h = cell_h - crop_top
    selected = [r - 1 for r in SELECTED_ROWS_1IDX]
    return cell_w, cell_h, crop_top, content_h, selected


def save_horizontal(img, path):
    cell_w, cell_h, crop_top, content_h, selected = _geometry(img)
    n_samples = len(selected)

    font = load_font(LABEL_FONT_PX, bold=True)
    probe = ImageDraw.Draw(Image.new("RGB", (400, 100)))
    max_label_w = 0
    for lbl in ROW_LABELS:
        bb = probe.textbbox((0, 0), lbl, font=font)
        max_label_w = max(max_label_w, bb[2] - bb[0])

    left_pad = max_label_w + 24
    top_pad, bottom_pad, right_pad = 12, 16, 16

    canvas_w = left_pad + n_samples * cell_w + (n_samples - 1) * CELL_GAP + right_pad
    canvas_h = top_pad + TOTAL_COLS * content_h + (TOTAL_COLS - 1) * CELL_GAP + bottom_pad

    canvas = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    grid_left = left_pad

    for panel_row, lbl in enumerate(ROW_LABELS):
        dest_y = top_pad + panel_row * (content_h + CELL_GAP)
        cy = dest_y + content_h // 2
        bb = draw.textbbox((0, 0), lbl, font=font)
        th, tw = bb[3] - bb[1], bb[2] - bb[0]
        draw.text((grid_left - tw - 16, cy - th // 2), lbl, fill=(0, 0, 0), font=font)

    for panel_row in range(TOTAL_COLS):
        dest_y = top_pad + panel_row * (content_h + CELL_GAP)
        orig_col = panel_row
        for sample_col, orig_row in enumerate(selected):
            left = orig_col * cell_w
            top = orig_row * cell_h + crop_top
            cell = img.crop((left, top, left + cell_w, top + content_h))
            dest_x = grid_left + sample_col * (cell_w + CELL_GAP)
            canvas.paste(cell, (dest_x, dest_y))

    canvas.save(path, dpi=(OUTPUT_DPI, OUTPUT_DPI))
    print(f"Saved (horizontal): {path}  {canvas.width}×{canvas.height} px")


def save_vertical(img, path):
    cell_w, cell_h, crop_top, content_h, selected = _geometry(img)
    n_samples = len(selected)

    font = load_font(LABEL_FONT_PX, bold=True)
    probe = ImageDraw.Draw(Image.new("RGB", (800, 100)))
    max_th = 0
    for lbl in ROW_LABELS:
        bb = probe.textbbox((0, 0), lbl, font=font)
        max_th = max(max_th, bb[3] - bb[1])

    col_hdr_pad = max(40, max_th + 16)
    top_pad = col_hdr_pad
    left_pad, bottom_pad, right_pad = 12, 16, 16

    canvas_w = left_pad + TOTAL_COLS * cell_w + (TOTAL_COLS - 1) * CELL_GAP + right_pad
    canvas_h = (
        top_pad
        + n_samples * content_h
        + (n_samples - 1) * CELL_GAP
        + bottom_pad
    )

    canvas = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    grid_left = left_pad
    y_hdr = max(0, (col_hdr_pad - max_th) // 2)

    for c, lbl in enumerate(ROW_LABELS):
        cx = grid_left + c * (cell_w + CELL_GAP) + cell_w // 2
        bb = draw.textbbox((0, 0), lbl, font=font)
        tw, th = bb[2] - bb[0], bb[3] - bb[1]
        draw.text((cx - tw // 2, y_hdr), lbl, fill=(0, 0, 0), font=font)

    for sample_row, orig_row in enumerate(selected):
        dest_y = top_pad + sample_row * (content_h + CELL_GAP)
        for panel_col in range(TOTAL_COLS):
            left = panel_col * cell_w
            top = orig_row * cell_h + crop_top
            cell = img.crop((left, top, left + cell_w, top + content_h))
            dest_x = grid_left + panel_col * (cell_w + CELL_GAP)
            canvas.paste(cell, (dest_x, dest_y))

    canvas.save(path, dpi=(OUTPUT_DPI, OUTPUT_DPI))
    print(f"Saved (vertical):   {path}  {canvas.width}×{canvas.height} px")


def main():
    img = Image.open(INPUT)
    save_horizontal(img, OUTPUT_HORIZONTAL)
    save_vertical(img, OUTPUT_VERTICAL)
    print(f"DPI: {OUTPUT_DPI}")


if __name__ == "__main__":
    main()
