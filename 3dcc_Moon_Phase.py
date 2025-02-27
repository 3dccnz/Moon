"""
Moon Phase Arc Animation Generator
Created by: 3dccnz
Last Modified: 2025-02-27 UTC
"""

import os
import math
from PIL import Image, ImageDraw, ImageChops, ImageOps

def create_moon_animation(
    cycles=1,                      # Number of complete moon cycles
    frames_per_arc=28,             # Number of frames per arc
    phases_per_cycle=8,            # Number of different moon phases per cycle
    width=800,                     # Output frame width
    height=400,                    # Output frame height
    moon_size=201,                 # Moon diameter in pixels (odd number recommended)
    arc_height=150,                # Maximum height of the arc (positive for inverted arc)
    arc_start_x=100,               # Starting X position of the arc (for southern hemisphere)
    arc_end_x=700,                 # Ending X position of the arc (for southern hemisphere)
    phases_to_include=[1, 2],      # Phases to include: 1=waxing/waning, 2=full/new
    output_dir="moon_arc_animation",
    moon_image="",                 # Path to optional moon texture (leave empty for generated moon)
    black_background=False,        # Whether to use a black background instead of transparency
    padding=10,                    # Extra padding to prevent clipping (in pixels)
    northern_hemisphere=False      # Set to True for northern hemisphere view
):
    os.makedirs(output_dir, exist_ok=True)
    
    # Add padding to moon size to prevent clipping
    padded_moon_size = moon_size + 2 * padding
    rotation_axis = (padded_moon_size - 1) // 2  # Center axis for rotation
    total_frames = frames_per_arc * phases_per_cycle * cycles * len(phases_to_include)
    
    # Swap arc start and end for northern hemisphere
    if northern_hemisphere:
        arc_start_x, arc_end_x = arc_end_x, arc_start_x
    
    def load_or_create_moon():
        moon = Image.new("L", (padded_moon_size, padded_moon_size), 0)
        draw = ImageDraw.Draw(moon)
        draw.ellipse((padding, padding, padding + moon_size, padding + moon_size), fill=255)
        return moon

    def create_left_half_circle(size):
        img = Image.new("L", (size, size), 0)
        draw = ImageDraw.Draw(img)
        draw.ellipse((padding, padding, padding + moon_size, padding + moon_size), fill=255)
        draw.rectangle((rotation_axis, 0, size, size), fill=0)
        return img

    def horizontal_scale_center(src, scale):
        w, h = src.size
        new_w = max(2, int(w * scale))
        scaled = src.resize((new_w, h), Image.BICUBIC)
        out_img = Image.new("L", (w, h), 0)
        x_offset = rotation_axis - (new_w // 2)
        out_img.paste(scaled, (x_offset, 0))
        return out_img

    def rotate_black_disc(angle_degs, size):
        base_left = create_left_half_circle(size)
        angle_rads = math.radians(angle_degs)
        scale = abs(math.cos(angle_rads))
        if angle_degs > 90:
            base_left = base_left.transpose(Image.FLIP_LEFT_RIGHT)
        return horizontal_scale_center(base_left, scale)

    def generate_moon_phase(angle, phase):
        moon = load_or_create_moon()
        black_mask = rotate_black_disc(angle, padded_moon_size)
        partial = ImageChops.subtract(moon, black_mask)
        
        moon_mask = Image.new("L", (padded_moon_size, padded_moon_size), 0)
        draw_mask = ImageDraw.Draw(moon_mask)
        draw_mask.ellipse((padding, padding, padding + moon_size, padding + moon_size), fill=255)
        
        if angle > 90:
            right_half_mask = Image.new("L", (padded_moon_size, padded_moon_size), 0)
            draw_right_mask = ImageDraw.Draw(right_half_mask)
            draw_right_mask.rectangle((rotation_axis - 1, 0, padded_moon_size, padded_moon_size), fill=255)
            inverted_moon = ImageChops.invert(partial)
            right_inverted = Image.composite(inverted_moon, partial, right_half_mask)
            partial = Image.composite(right_inverted, partial, moon_mask)
        elif angle <= 90:
            left_half_mask = Image.new("L", (padded_moon_size, padded_moon_size), 0)
            draw_left_mask = ImageDraw.Draw(left_half_mask)
            draw_left_mask.rectangle((rotation_axis - 1, 0, padded_moon_size, padded_moon_size), fill=255)
            partial = Image.composite(black_mask, partial, left_half_mask)
        
        if phase == 2:
            partial = ImageOps.invert(partial)
        
        moon_img = Image.new("RGBA", (padded_moon_size, padded_moon_size), (0, 0, 0, 255))
        partial_rgba = Image.new("RGBA", (padded_moon_size, padded_moon_size), (0, 0, 0, 255))
        partial_rgba.paste(partial.convert("RGB"), (0, 0), mask=moon_mask)
        moon_img.paste(partial_rgba, (0, 0), mask=moon_mask)

        return moon_img

    def split_and_merge_transparent_frame(frame):
        w, h = frame.size
        left_half = frame.crop((0, 0, w // 2 - 1, h))
        right_half = frame.crop((w // 2 + 1, 0, w, h))
        merged_frame = Image.new("RGBA", (w - 2, h), (0, 0, 0, 255))
        merged_frame.paste(left_half, (0, 0))
        merged_frame.paste(right_half, (w // 2 - 1, 0))
        return merged_frame

    def rotate_image(image, angle):
        return image.rotate(angle, resample=Image.BICUBIC, center=(rotation_axis, rotation_axis), expand=False, fillcolor=(0, 0, 0, 255))

    def calculate_arc_position(progress):
        x = arc_start_x + (arc_end_x - arc_start_x) * progress
        arc_width = arc_end_x - arc_start_x
        center_x = arc_start_x + arc_width / 2
        a = 4 * arc_height / (arc_width ** 2)
        y = a * ((x - center_x) ** 2) + (height / 2 - arc_height)
        return int(x), int(y)

    def apply_texture(frame, moon_image, northern_hemisphere):
        """
        Applies texture to a frame while preserving transparency.
        For northern hemisphere, use texture as-is.
        For southern hemisphere, flip texture upside down.
        """
        if moon_image and os.path.exists(moon_image):
            # Extract alpha channel to use as mask
            r, g, b, a = frame.split()
            
            # Load texture and resize to match moon size
            moon_texture = Image.open(moon_image).convert("RGB")
            # Resize texture to match the actual moon size, not the padded size
            moon_texture = moon_texture.resize((moon_size, moon_size), Image.LANCZOS)
            
            # For southern hemisphere, rotate 180 degrees (appear upside down)
            if not northern_hemisphere:
                moon_texture = moon_texture.rotate(180)
                
            
            # Create a new texture image with padding to match frame size
            padded_texture = Image.new("RGB", (padded_moon_size, padded_moon_size), (0, 0, 0))
            # Paste texture in center of padded area
            texture_x = (padded_moon_size - moon_size) // 2
            texture_y = (padded_moon_size - moon_size) // 2
            padded_texture.paste(moon_texture, (texture_x, texture_y))
            
            # Create RGB version of frame (no alpha)
            frame_rgb = Image.new("RGB", frame.size, (0, 0, 0))
            frame_rgb.paste(frame.convert("RGB"), (0, 0), mask=a)
            
            # Multiply texture with RGB frame
            textured = ImageChops.multiply(frame_rgb, padded_texture)
            
            # Create new RGBA image and paste textured image using original alpha
            result = Image.new("RGBA", frame.size, (0, 0, 0, 0))
            result.paste(textured, (0, 0), mask=a)
            
            return result
        return frame

    frame_index = 0
    for cycle in range(cycles):
        for phase_type_index, phase_type in enumerate(phases_to_include):
            for phase_num in range(phases_per_cycle):
                angle = 180.0 * phase_num / (phases_per_cycle - 1)
                
                moon_img = generate_moon_phase(angle, phase_type)
                moon_img = split_and_merge_transparent_frame(moon_img)
                
                # Set rotation angle based on phase_type and hemisphere
                if phase_type == 1:
                    global_rotation_angle = 30
                else:  # phase_type == 2
                    global_rotation_angle = -30
                    
                moon_img = rotate_image(moon_img, global_rotation_angle)
                
                # Mirror the moon image for northern hemisphere
                if northern_hemisphere:
                    moon_img = moon_img.transpose(Image.FLIP_LEFT_RIGHT)

                # Apply texture after all transformations
                if moon_image and os.path.exists(moon_image):
                    moon_img = apply_texture(moon_img, moon_image, northern_hemisphere)
                
                for frame in range(frames_per_arc):
                    progress = frame / (frames_per_arc - 1)
                    x, y = calculate_arc_position(progress)
                    
                    background_color = (0, 0, 0, 255) if black_background else (0, 0, 0, 0)
                    full_frame = Image.new("RGBA", (width, height), background_color)
                    
                    moon_x = x - (padded_moon_size // 2)
                    moon_y = y - (padded_moon_size // 2)
                    
                    full_frame.paste(moon_img, (moon_x, moon_y), moon_img)
                    
                    out_name = f"frame_{frame_index:04d}.png"
                    out_path = os.path.join(output_dir, out_name)
                    full_frame.save(out_path, format="PNG")
                    print(f"Saved {out_path} at {angle} {phases_to_include}")
                    
                    frame_index += 1

    print(f"Animation complete: {frame_index} frames generated in {output_dir}")

if __name__ == "__main__":
    create_moon_animation(
        cycles=1,
        frames_per_arc=10,
        phases_per_cycle=8,
        width=800,
        height=400,
        moon_size=81,
        arc_height=120,
        arc_start_x=800,          # For southern hemisphere (will be auto-swapped for northern)
        arc_end_x=0,           # For southern hemisphere (will be auto-swapped for northern)
        phases_to_include=[1, 2], # Phases to include: 1=waxing/waning, 2=full to new
        black_background=True,
        padding=10,
        northern_hemisphere=False,  # Set to False for southern hemisphere, True for northern
        moon_image="moon_texture.jpg"  # Path to moon texture
        #moon_image=""  # No Texture
    )