import json

# Define the tidal offset to apply (in meters)

DEPTH_OFFSET = -0.975  #this is the average tideal offset for the area for the time period of the DAS data, above sea map zero, according to kartverket.no

# Load JSON
with open(r"C:\Users\helen\Documents\PythonProjects\my-project\libs\resources\upsampled_cable-layout.json", "r") as f:
    data = json.load(f)

# Loop through all features
for feature in data["features"]:
    coords = feature["geometry"]["coordinates"]

    # For MultiLineString: coordinates is a list of lists of points
    for line in coords:
        for point in line:
            # point = [lon, lat] or [lon, lat, depth]
            if len(point) == 3 and point[2] != 0:
                point[2] = point[2] + DEPTH_OFFSET
            else:
                # If no depth is present,  skip or insert one
                # point.append(DEPTH_OFFSET)
                pass

# Save updated JSON
with open(r"C:\Users\helen\Documents\PythonProjects\my-project\libs\resources\suspected_cable-layout.json", "w") as f:
    json.dump(data, f, indent=2)
