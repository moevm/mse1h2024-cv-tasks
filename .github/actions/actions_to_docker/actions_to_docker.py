import json
import subprocess
import shutil
from pathlib import Path

JSON_PATH = "../right_requests.json"
SOURCE_DIR_PATH = "."
DEST_DIR_PATH = "../../../app_files"

with open(JSON_PATH) as f:
    data = json.load(f)
print(data)

for i in range (len(data)):
    command = "gh pr checkout "+str(data[i]["number"])
    subprocess.run(command, shell=True, executable="/bin/bash")
    for file in data[i]["files"]:
        source_path = file["path"]
        destination_path = "app_files/"+source_path
        destination_path = "/".join(destination_path.split("/")[:-1])
        Path(destination_path).mkdir(parents=True, exist_ok=True)
        shutil.copy(source_path, destination_path)
