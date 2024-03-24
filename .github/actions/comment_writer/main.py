import os
import json
import subprocess

data = json.loads(os.environ['INPUT_CORRECTPULLREQUESTS'])
print(data)

for i in range(len(data)):
    command = "gh pr comment " + str(data[i]["number"]) + " --body " + "\"" + str(data[i]["comment"]) + "\""
    subprocess.run(command, shell=True, executable="/bin/bash")
