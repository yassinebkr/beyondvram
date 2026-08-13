"""Create the measured hierarchy PNG with Pillow (no plotting dependency)."""

from __future__ import annotations

import csv
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from common import SUMMARY_CSV


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "plots" / "system_memory_hierarchy.png"


def main() -> None:
    rows = []
    if SUMMARY_CSV.exists():
        with SUMMARY_CSV.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    selected = []
    wanted = {"B01", "B03", "B04", "B05"}
    for row in rows:
        if row["benchmark_id"] in wanted and row["unit"] == "GB/s_payload" and row["status"] == "ok":
            selected.append((row["benchmark_id"], row["test"], float(row["value"])))
    latest = {}
    for item in selected:
        latest[item[0]] = item
    ordered = [latest[key] for key in ["B01", "B03", "B04", "B05"] if key in latest]

    width, height = 1200, 700
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=18)
    title_font = ImageFont.load_default(size=28)
    draw.text((60, 35), "Measured system memory hierarchy", fill="black", font=title_font)
    draw.text((60, 78), "Median payload bandwidth; missing tests are not estimates", fill="#444444", font=font)
    chart_left, chart_top, chart_right, chart_bottom = 340, 135, 1130, 610
    draw.line((chart_left, chart_top, chart_left, chart_bottom), fill="black", width=2)
    draw.line((chart_left, chart_bottom, chart_right, chart_bottom), fill="black", width=2)
    max_value = max((item[2] for item in ordered), default=1.0)
    bar_height = 70
    gap = 35
    colors = {"B01": "#4C78A8", "B03": "#72B7B2", "B04": "#F58518", "B05": "#E45756"}
    for index, (benchmark_id, label, value) in enumerate(ordered):
        y = chart_top + 35 + index * (bar_height + gap)
        bar_width = int((chart_right - chart_left - 80) * value / max_value)
        draw.rectangle((chart_left, y, chart_left + bar_width, y + bar_height), fill=colors[benchmark_id])
        draw.text((35, y + 20), f"{benchmark_id} {label}", fill="black", font=font)
        draw.text((chart_left + bar_width + 12, y + 20), f"{value:.3f} GB/s", fill="black", font=font)
    missing = [key for key in ["B01", "B03", "B04", "B05"] if key not in latest]
    if missing:
        draw.text((60, 650), f"Not measured: {', '.join(missing)}", fill="#A33A2B", font=font)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    image.save(OUTPUT)


if __name__ == "__main__":
    main()

