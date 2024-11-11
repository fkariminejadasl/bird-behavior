import math
from io import BytesIO

import requests
from PIL import Image

TILE_SIZE = 256  # Size of a tile in pixels


def deg2pixel(lat_deg, lon_deg, zoom):
    """
    Convert latitude and longitude to pixel coordinates in the global map at a specific zoom level.
    https://wiki.openstreetmap.org/wiki/Slippy_map_tilenames
    """
    lat_rad = math.radians(lat_deg)
    n = 2**zoom
    x = (lon_deg + 180.0) / 360.0 * n * TILE_SIZE
    y = (1 - math.asinh(math.tan(lat_rad)) / math.pi) / 2 * n * TILE_SIZE
    return x, y


def pixel2tile(px, py):
    """
    Convert pixel coordinates to tile numbers.
    """
    return int(px // TILE_SIZE), int(py // TILE_SIZE)


def get_osm_tile(zoom, xtile, ytile):
    """
    Fetch a single OSM tile image.
    """
    base_url = "https://tile.openstreetmap.org/{}/{}/{}.png"
    url = base_url.format(zoom, xtile, ytile)
    headers = {"User-Agent": "YourAppName/1.0 (your@email.com)"}
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return Image.open(BytesIO(response.content))
    else:
        raise Exception(
            f"Failed to fetch tile {zoom}/{xtile}/{ytile}: {response.status_code}"
        )


def get_centered_map_image(lat, lon, zoom=15, width=640, height=480):
    """
    Fetch a map image centered at the given latitude and longitude.
    """
    # Convert lat/lon to global pixel coordinates
    center_px, center_py = deg2pixel(lat, lon, zoom)

    # Calculate the pixel coordinates for the top-left corner of the desired image
    top_left_px = center_px - (width / 2)
    top_left_py = center_py - (height / 2)
    top_left_tile_x, top_left_tile_y = pixel2tile(top_left_px, top_left_py)

    # Calculate the pixel coordinates for the bottom-right corner of the desired image
    bottom_right_px = center_px + (width / 2)
    bottom_right_py = center_py + (height / 2)
    bottom_right_tile_x, bottom_right_tile_y = pixel2tile(
        bottom_right_px, bottom_right_py
    )

    # Number of tiles to fetch horizontally and vertically
    num_tiles_x = bottom_right_tile_x - top_left_tile_x + 1
    num_tiles_y = bottom_right_tile_y - top_left_tile_y + 1

    # Create a new blank image to stitch tiles
    stitched_image = Image.new(
        "RGB", (num_tiles_x * TILE_SIZE, num_tiles_y * TILE_SIZE)
    )

    # Fetch and paste each tile into the stitched image
    for x in range(top_left_tile_x, bottom_right_tile_x + 1):
        for y in range(top_left_tile_y, bottom_right_tile_y + 1):
            try:
                tile_image = get_osm_tile(zoom, x, y)
                stitched_image.paste(
                    tile_image,
                    (
                        (x - top_left_tile_x) * TILE_SIZE,
                        (y - top_left_tile_y) * TILE_SIZE,
                    ),
                )
            except Exception as e:
                print(e)
                # Optionally, you can paste a blank tile or a placeholder in case of failure
                placeholder = Image.new("RGB", (TILE_SIZE, TILE_SIZE), (255, 255, 255))
                stitched_image.paste(
                    placeholder,
                    (
                        (x - top_left_tile_x) * TILE_SIZE,
                        (y - top_left_tile_y) * TILE_SIZE,
                    ),
                )

    # Calculate the pixel offset within the stitched image to crop
    offset_x = int(top_left_px - top_left_tile_x * TILE_SIZE)
    offset_y = int(top_left_py - top_left_tile_y * TILE_SIZE)

    # Crop the stitched image to the desired size
    cropped_image = stitched_image.crop(
        (offset_x, offset_y, offset_x + width, offset_y + height)
    )

    return cropped_image


# lat, lon = 52.36977, 4.9954
# lat, lon = 52.00947, 4.34438
# map_image = get_centered_map_image(lat, lon, zoom=15)
# map_image.show()  # map_image.convert('RGB')

# print("done")
