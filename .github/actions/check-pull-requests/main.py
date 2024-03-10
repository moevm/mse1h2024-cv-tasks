import os
import json
import re
from reqular_expressions import *

def write_result(result):
    with open(os.path.abspath(os.environ["GITHUB_OUTPUT"]), "a") as output_file:
        output_file.write(f"correctPullRequests={result}")

def main():
    with open("opened_pull_requests.json", "r") as input_file:
        opened_pull_requests_json = input_file.read()

    opened_pull_requests = json.loads(opened_pull_requests_json)
    correct_pull_requests = []

    for pull_request in opened_pull_requests:

        result = re.match(pull_request_title, pull_request["title"])

        if result is None:
            continue

        pull_request_info = get_pull_request_info(pull_request["title"])

        general_folder_path = get_general_folder_path(pull_request_info)
        report_path = get_report_path(pull_request_info)
        sources_paths = get_sources_paths(pull_request_info)

        checks = {
            "all_files_in_general_folder": True,
            "contains_report": False,
            "contains_sources": [False] * len(sources_paths)
        }

        for file in pull_request["files"]:
            if file["path"] == report_path:
                checks["contains_report"] = True
                continue

            is_source_file = False
            for i in range(len(sources_paths)):
                if file["path"] == sources_paths[i]:
                    checks["contains_sources"][i] = True
                    is_source_file = True
                    break

            if is_source_file:
                continue

            if file["path"].startswith(general_folder_path):
                continue

            checks["all_files_in_general_folder"] = False
            break

        if not(checks["contains_report"] and \
            checks["all_files_in_general_folder"] and \
            all(is_contained for is_contained in checks["contains_sources"])):
            continue

        correct_pull_requests.append(pull_request)

    write_result(json.dumps(correct_pull_requests))

if __name__ == "__main__":
    main()
