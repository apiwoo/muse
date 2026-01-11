# Project MUSE - Icon Generator
# Creates a professional application icon for MUSE.exe
# (C) 2025 MUSE Corp. All rights reserved.

"""
Icon Design Concept:
- Modern, minimalist design reflecting beauty/camera theme
- Purple-to-pink gradient (matching Discord-style UI)
- Circular shape reminiscent of camera lens or beauty mirror
- White M lettermark
- Concentric circle details for lens effect
"""

import os
import math
from PIL import Image, ImageDraw, ImageFont


def create_gradient_circle(size, color1, color2):
    """
    Create a circular image with radial gradient from center to edge.

    Args:
        size: Image size (width, height tuple)
        color1: Center color (RGBA tuple)
        color2: Edge color (RGBA tuple)

    Returns:
        PIL Image with gradient circle
    """
    img = Image.new('RGBA', size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    center_x, center_y = size[0] // 2, size[1] // 2
    max_radius = min(center_x, center_y)

    # Draw gradient from outside to inside (so inner circles cover outer)
    for r in range(max_radius, 0, -1):
        # Calculate interpolation factor (0 at center, 1 at edge)
        t = r / max_radius

        # Interpolate colors
        color = tuple(int(color1[i] + (color2[i] - color1[i]) * t) for i in range(4))

        # Draw circle
        draw.ellipse(
            [center_x - r, center_y - r, center_x + r, center_y + r],
            fill=color
        )

    return img


def create_muse_icon(size=256):
    """
    Create the MUSE application icon.

    Args:
        size: Icon size in pixels

    Returns:
        PIL Image with the MUSE icon
    """
    # Colors (RGBA)
    purple = (138, 43, 226, 255)    # Blue Violet
    pink = (255, 105, 180, 255)     # Hot Pink
    white = (255, 255, 255, 255)
    semi_white = (255, 255, 255, 80)

    # Create base gradient circle
    img = create_gradient_circle((size, size), purple, pink)
    draw = ImageDraw.Draw(img)

    center = size // 2

    # Add subtle concentric circles for lens effect
    for radius_factor in [0.85, 0.75]:
        r = int(center * radius_factor)
        draw.ellipse(
            [center - r, center - r, center + r, center + r],
            outline=semi_white,
            width=max(1, size // 128)
        )

    # Add inner glow circle
    inner_r = int(center * 0.65)
    glow_color = (255, 255, 255, 30)
    draw.ellipse(
        [center - inner_r, center - inner_r, center + inner_r, center + inner_r],
        fill=glow_color
    )

    # Draw the M lettermark
    # Calculate font size (approximately 50% of icon size)
    font_size = int(size * 0.45)

    # Try to use a bold system font, fall back to default if not available
    font = None
    font_paths = [
        "C:/Windows/Fonts/arialbd.ttf",      # Arial Bold
        "C:/Windows/Fonts/segoeui.ttf",       # Segoe UI
        "C:/Windows/Fonts/calibrib.ttf",      # Calibri Bold
        "C:/Windows/Fonts/arial.ttf",         # Arial Regular
    ]

    for font_path in font_paths:
        try:
            font = ImageFont.truetype(font_path, font_size)
            break
        except (IOError, OSError):
            continue

    if font is None:
        # Use default font (smaller, but works)
        font = ImageFont.load_default()

    # Draw M with slight shadow for depth
    letter = "M"

    # Get text bounding box
    bbox = draw.textbbox((0, 0), letter, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    # Center the text
    text_x = (size - text_width) // 2
    text_y = (size - text_height) // 2 - bbox[1]  # Adjust for font baseline

    # Draw subtle shadow
    shadow_offset = max(1, size // 64)
    shadow_color = (0, 0, 0, 50)
    draw.text((text_x + shadow_offset, text_y + shadow_offset), letter, font=font, fill=shadow_color)

    # Draw main letter
    draw.text((text_x, text_y), letter, font=font, fill=white)

    # Add subtle highlight arc at top
    highlight_start = -60
    highlight_end = 60
    highlight_r = int(center * 0.9)
    arc_width = max(2, size // 64)

    # Draw highlight arc
    draw.arc(
        [center - highlight_r, center - highlight_r, center + highlight_r, center + highlight_r],
        start=highlight_start - 90,
        end=highlight_end - 90,
        fill=(255, 255, 255, 100),
        width=arc_width
    )

    return img


def create_ico_file(output_path, base_size=256):
    """
    Create a multi-resolution ICO file using BMP format for compatibility.

    Args:
        output_path: Path to save the ICO file
        base_size: Base size for the highest resolution
    """
    # Standard Windows icon sizes
    sizes = [256, 128, 64, 48, 32, 16]

    # Create the base high-resolution icon
    base_icon = create_muse_icon(base_size)

    # Create resized versions and convert to proper format
    icons = []
    for size in sizes:
        if size == base_size:
            resized = base_icon.copy()
        else:
            # Use high-quality downsampling
            resized = base_icon.resize((size, size), Image.Resampling.LANCZOS)

        # Convert to RGBA to ensure compatibility
        if resized.mode != 'RGBA':
            resized = resized.convert('RGBA')

        icons.append(resized)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save as ICO - Pillow will handle the format
    # Use bitmap=True for better compatibility (BMP format inside ICO)
    base_icon.save(
        output_path,
        format='ICO',
        sizes=[(s, s) for s in sizes],
        bitmap_format='bmp'  # Force BMP format for compatibility
    )

    print(f"Icon saved to: {output_path}")
    print(f"Included sizes: {', '.join(f'{s}x{s}' for s in sizes)}")

    return output_path


def main():
    """Main entry point."""
    # Determine output path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    output_path = os.path.join(project_root, "assets", "icon.ico")

    print("=" * 60)
    print("   PROJECT MUSE - Icon Generator")
    print("=" * 60)
    print()

    # Create the icon
    create_ico_file(output_path)

    print()
    print("Icon generation complete!")
    print()

    # Also save a PNG preview
    preview_path = os.path.join(project_root, "assets", "icon_preview.png")
    preview = create_muse_icon(256)
    preview.save(preview_path, format='PNG')
    print(f"Preview saved to: {preview_path}")


if __name__ == "__main__":
    main()
