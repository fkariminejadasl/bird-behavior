import json
from datetime import datetime, timedelta
from pathlib import Path


def get_file_info(image_folder):
    """
    Build a dictionary of filenames, indexed by their index. Filename example: 2044,6210,2016-05-09 11:09:55.png

    Parameters:
        imu_folder (str): The path to the folder containing IMU images.

    Returns:
        dict: A dictionary where the key is the index and the value is a tuple (device_id, starting_date_time).
        e.g. item (6210, datetime.datetime(2016, 5, 9, 11, 9, 39))
    """
    file_infos = {}
    for file_path in Path(image_folder).glob(
        "*.png"
    ):  # Adjust for file extensions if needed
        filename = file_path.name
        try:
            parts = filename.split(",")
            index = int(parts[0])  # Assuming index is always an integer
            device_id = int(parts[1])
            date_time_str = parts[2].split(".")[0]  # Remove extension
            starting_date_time = datetime.strptime(date_time_str, "%Y-%m-%d %H:%M:%S")
            file_infos[index] = (device_id, starting_date_time)
        except Exception as e:
            print(f"Error parsing file {filename}: {e}")
    return file_infos


def generate_sql_or_json_query(index, file_infos, output_file, IS_JSON=True):
    """
    Generate an SQL or JSON query file based from device id and start time for https://birdvis.e-ecology.nl.
    """

    # Get device id and datetime
    file_info = file_infos[index]
    device_id, date_time = file_info
    s_datetime = date_time - timedelta(hours=1)
    e_datetime = date_time + timedelta(hours=1)
    s_date_str = s_datetime.strftime("%Y-%m-%d")
    e_date_str = e_datetime.strftime("%Y-%m-%d")

    # Create JSON structure
    json_data = {
        "tracker": {"serial_number": str(device_id), "data_source": "GPS"},
        "time": {
            "start": f"{s_date_str}T{s_datetime.hour:02d}:{s_datetime.minute:02d}",
            "end": f"{e_date_str}T{e_datetime.hour:02d}:{e_datetime.minute:02d}",
        },
        "area": {"north": None, "south": None, "west": None, "east": None},
        "project": {"name": ""},
        "bird": {
            "ring_number": "",
            "bird_name_English": "",
            "bird_name_Latin": "",
            "bird_sex": "",
        },
    }

    # Generate SQL query
    sql_query = f"""
    SELECT 
        trackpoint.*,
        
        tracksession.project_id,
        tracksession.track_session_id,
        tracksession.individual_id,
        tracksession.tracker_id
    FROM gps.ee_track_session_limited AS tracksession
    JOIN gps.ee_individual_limited AS bird ON tracksession.individual_id = bird.individual_id
    JOIN gps.ee_species_limited AS bird_s ON bird.species_latin_name=bird_s.latin_name
    JOIN gps.ee_tracking_speed_limited AS trackpoint ON tracksession.device_info_serial = trackpoint.device_info_serial AND
        trackpoint.date_time BETWEEN tracksession.start_date AND tracksession.end_date

    WHERE 1=1
        AND trackpoint.device_info_serial = {device_id} AND trackpoint.date_time >= '{s_date_str} 00:00' AND trackpoint.date_time < '{e_date_str} 23:59' ORDER BY trackpoint.date_time
    """

    if IS_JSON:
        # Save to a JSON file
        try:
            with open(output_file, "w") as file:
                json.dump(json_data, file, indent=4)
            print(f"JSON file saved to {output_file}")
        except Exception as e:
            print(f"Error saving JSON file: {e}")

    else:
        # Save to a text file
        try:
            with open(output_file, "w") as file:
                file.write(sql_query.strip())
            print(f"SQL query saved to {output_file}")
        except Exception as e:
            print(f"Error saving SQL query: {e}")


def print_dev_datetime(file_info):
    return f"{file_info[0]},{file_info[1]}"


# Example usage
# Folder containing the IMU images
imu_folder = Path("/home/fatemeh/Downloads/bird/results/gt2/all")
output_file = Path("/home/fatemeh/Desktop/tmp2.json")

file_infos = get_file_info(imu_folder)

# Generate query for index
index = 2043
generate_sql_or_json_query(index, file_infos, output_file)
print_dev_datetime(file_infos[index])
print("done")
