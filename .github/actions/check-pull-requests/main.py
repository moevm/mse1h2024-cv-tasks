import os
import json
import re
from regular_expressions import *

class PullRequestChecker():

    def __init__(self, pull_request):
        self._title = pull_request["title"]
        self._files = pull_request["files"]
        self._number = pull_request["number"]
        self._comment = ""
        self._checks = {
            "title": True,
            "all_files_in_general_folder": True,
            "contains_all_files": True
        }

        self._check_title()

        if self._checks["title"]:
            self._check_files()

    def _check_title(self):
        pull_request_title_re = self._get_pull_request_title_re()

        if re.match(pull_request_title_re, self._title) is None:
            self._checks["title"] = False
            self._comment += "Неверное название пулл-реквеста.\n"

    def _check_files(self):
        group_number, last_name, first_name, lab_number = self._title.split("_")
        lab_number = lab_number[3:]

        general_path = self._replace_dummies(GENERAL_PATH, group_number, last_name, first_name, lab_number) 

        lab_files = FILES[int(lab_number) - 1]
        lab_file_paths = self._get_lab_file_paths(group_number, last_name, first_name, lab_number)
        lab_file_checks = [False for _ in range(len(lab_file_paths))]
        
        for file in self._files:
            file_path = file["path"]

            if not file_path.startswith(general_path):
                self._checks["all_files_in_general_folder"] = False
                self._comment += f"Файл {file_path} расположен не в общей папке пулл-реквеста {general_path}\n"
                continue

            for i in range(len(lab_file_paths)):
                if re.match(lab_file_paths[i], file_path) is not None:
                    lab_file_checks[i] = True

        for i in range(len(lab_file_paths)):
            if not lab_file_checks[i]:
                self._checks["contains_all_files"] = False
                self._comment += f"Не найден файл, содержащий {lab_files[i]['context']}\n"

        if all(self._checks[item] for item in self._checks):
            self._comment += "Пулл-реквест корректен.\n"

    def _get_pull_request_title_re(self):
        group_number_re = f"({'|'.join(GROUPS)})"

        last_names, first_names = zip(*STUDENTS)
        last_name_re = f"({'|'.join(last_names)})"
        first_name_re = f"({'|'.join(first_names)})"

        labs_list = [str(item) for item in range(1, len(FILES) + 1)]
        lab_numbers_re = f"({'|'.join(labs_list)})"

        return f"^{group_number_re}_{last_name_re}_{first_name_re}_lab{lab_numbers_re}$"

    def _replace_dummies(self, input_string, group_number, last_name, first_name, lab_number):
        return input_string \
                .replace(r"\group_number", group_number) \
                .replace(r"\last_name", last_name) \
                .replace(r"\first_name", first_name) \
                .replace(r"\lab_number", lab_number)

    def _get_lab_file_paths(self, group_number, last_name, first_name, lab_number):
        general_path = self._replace_dummies(GENERAL_PATH, group_number, last_name, first_name, lab_number) 

        result = []
        lab_files = FILES[int(lab_number) - 1]
        for file in lab_files:
            path = self._replace_dummies(file["path"], group_number, last_name, first_name, lab_number)
            file_name = self._replace_dummies(file["file_name"], group_number, last_name, first_name, lab_number) 
            full_path = f"^{general_path}/{path}/{file_name}$"
            result.append(full_path)

        return result

    def get_json(self):
        return {
            "title": self._title,
            "number": self._number,
            "files": self._files,
            "comment": self._comment,
            "correct": all(self._checks[item] for item in self._checks)
        }

def write_result(result):
    with open(os.path.abspath(os.environ["GITHUB_OUTPUT"]), "a") as output_file:
        output_file.write(f"correctPullRequests={result}")

def main():
    with open("opened_pull_requests.json", "r") as input_file:
        opened_pull_requests_json = input_file.read()

    opened_pull_requests = json.loads(opened_pull_requests_json)
    processed_pull_requests = []

    for pull_request in opened_pull_requests:
        processed_pull_requests.append(PullRequestChecker(pull_request).get_json())

    write_result(json.dumps(processed_pull_requests))

if __name__ == "__main__":
    main()
