from PIL import Image, ImageDraw


def create_diagram():
    # Create an image with white background
    img = Image.new('RGB', (259, 194), 'white')
    draw = ImageDraw.Draw(img)

    # Set the color
    color = (0, 153, 204)  # Light blue color

    # Draw rectangles
    draw.rectangle([50, 20, 210, 30], outline=color, fill=color)
    draw.rectangle([120, 60, 140, 70], outline=color, fill=color)

    # Draw connectors between rectangles
    draw.line([120, 30, 120, 60], fill=color)
    draw.line([140, 30, 140, 60], fill=color)

    # Draw triangles
    def draw_triangle(centre_x, top_y):
        draw.polygon([
            (centre_x - 30, top_y + 40),
            (centre_x, top_y),
            (centre_x + 30, top_y + 40)
        ], outline=color, fill=color)

    # Coordinates for triangles based on given diagram
    triangles_coordinates = [
        (60, 70), (180, 70),  # Top row
        (40, 110), (100, 110), (140, 110), (200, 110)  # Bottom row
    ]

    for coord in triangles_coordinates:
        draw_triangle(*coord)

    # Draw horizontal connectors on the top
    draw.line([70, 25, 190, 25], fill=color)

    # Draw horizontal connectors on the bottom
    draw.line([50, 105, 210, 105], fill=color)

    return img


# Generate the image
image = create_diagram()
# Save the image
image.save("diagram_image.png")
image.show()
