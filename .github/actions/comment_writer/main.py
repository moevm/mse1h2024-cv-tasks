import os
import json
import subprocess

data = json.loads(os.environ['INPUT_CORRECTPULLREQUESTS'])
print(data)

for i in range(len(data)):
    #command = "gh pr comment "+str(data[i]["number"])+" --body " +str(data[i]["comment"])
    command = "gh pr comment "+str(data[i]["number"])+" --body " + "\"Test comment for pr "+str(data[i]["number"])+"\""
    subprocess.run(command, shell=True, executable="/bin/bash")
