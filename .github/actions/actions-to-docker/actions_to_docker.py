import os
import json
import subprocess
import shutil
from pathlib import Path

SOURCE_DIR_PATH = f"{os.environ['GITHUB_WORKSPACE']}"
DEST_DIR_PATH = f"{os.environ['GITHUB_WORKSPACE']}/pull-request-data"

data = json.loads(os.environ['INPUT_CORRECTPULLREQUESTS'])
print(data)

for i in range(len(data)):
    command = "gh pr checkout "+str(data[i]["number"])
    subprocess.run(command, shell=True, executable="/bin/bash")
    for file in data[i]["files"]:
        source_path = SOURCE_DIR_PATH + file["path"]
        destination_path = DEST_DIR_PATH + file["path"] 
        destination_path = "/".join(destination_path.split("/")[:-1])
        Path(destination_path).mkdir(parents=True, exist_ok=True)
        shutil.copy(source_path, destination_path)
