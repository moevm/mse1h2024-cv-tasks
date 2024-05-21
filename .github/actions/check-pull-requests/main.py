import os
import json
import re
import subprocess
from regular_expressions import *

class PullRequestChecker():

    def __init__(self, pull_request):
        self._title = pull_request["title"]
        self._files = pull_request["files"]
        self._number = pull_request["number"]
        self._labels = pull_request["labels"]
        self._work_tag = None
        self._comment = ""
        self._checks = {
            "title": True,
            "all_files_in_general_folder": True,
            "contains_all_files": True
        }

        if len(pull_request["labels"]) != 0 and \
            all(label["name"] != "error" for label in self._labels) and \
            any(label["name"] == "ok" for label in self._labels):
            _, _, _, self._work_tag = self._title.split("_")
            return

        self._check_title()

        if self._checks["title"]:
            self._check_files()

        self._update_labels()

    def _update_labels(self):
        for label in self._labels:
            label_name = label["name"]
            command = f"gh pr edit {self._number} --remove-label '{label_name}'"
            subprocess.run(command, shell=True, executable="/bin/bash")

        if self._work_tag is not None:
            command = f"gh pr edit {self._number} --add-label '{self._work_tag}'"
            subprocess.run(command, shell=True, executable="/bin/bash")

        ok_or_error = "ok" if all(self._checks[item] for item in self._checks) else "error"
        command = f"gh pr edit {self._number} --add-label '{ok_or_error}'"
        subprocess.run(command, shell=True, executable="/bin/bash")

    def _check_title(self):
        pull_request_title_re = self._get_pull_request_title_re()

        if re.match(pull_request_title_re, self._title) is None:
            self._checks["title"] = False
            self._comment += "Неверное название пулл-реквеста.\n"

    def _check_files(self):
        group_number, last_name, first_name, work_tag = self._title.split("_")
        self._work_tag = work_tag

        general_path = self._replace_dummies(GENERAL_PATH, group_number, last_name, first_name, work_tag) 

        lab_files = FILES[work_tag]
        lab_file_paths = self._get_lab_file_paths(group_number, last_name, first_name, work_tag)
        lab_file_checks = [False for _ in range(len(lab_file_paths))]
        
        for file in self._files:
            file_path = file["path"]

            if not file_path.startswith(general_path):
                self._checks["all_files_in_general_folder"] = False
                self._comment += f"Файл {file_path} расположен не в общей папке пулл-реквеста {general_path}.\n"
                continue

            for i in range(len(lab_file_paths)):
                if re.match(lab_file_paths[i], file_path) is not None:
                    lab_file_checks[i] = True

        if any(lab_file_checks[i] == False for i in range(len(lab_file_paths))):
            self._checks["contains_all_files"] = False
            self._comment += f"Не найдены файлы:\n"

        for i in range(len(lab_file_paths)):
            if not lab_file_checks[i]:
                self._comment += f"- {lab_files[i]['context']}\n"

        if all(self._checks[item] for item in self._checks):
            self._comment += "Пулл-реквест корректен.\n"

    def _get_pull_request_title_re(self):
        group_number_re = f"({'|'.join(GROUPS)})"

        last_names, first_names = zip(*STUDENTS)
        last_name_re = f"({'|'.join(last_names)})"
        first_name_re = f"({'|'.join(first_names)})"

        work_tags_re = f"({'|'.join(WORK_TAGS)})"

        return f"^{group_number_re}_{last_name_re}_{first_name_re}_{work_tags_re}$"

    def _replace_dummies(self, input_string, group_number, last_name, first_name, work_tag):
        return input_string \
                .replace(r"\group_number", group_number) \
                .replace(r"\last_name", last_name) \
                .replace(r"\first_name", first_name) \
                .replace(r"\work_tag", work_tag)

    def _get_lab_file_paths(self, group_number, last_name, first_name, work_tag):
        general_path = self._replace_dummies(GENERAL_PATH, group_number, last_name, first_name, work_tag) 

        result = []
        lab_files = FILES[work_tag]
        for file in lab_files:
            path = self._replace_dummies(file["path"], group_number, last_name, first_name, work_tag)
            file_name = self._replace_dummies(file["file_name"], group_number, last_name, first_name, work_tag) 
            full_path = f"^{general_path}/{path}/{file_name}$"
            result.append(full_path)

        return result

    def get_json(self):
        return {
            "title": self._title,
            "number": self._number,
            "files": self._files,
            "work_tag": self._work_tag,
            "comment": self._comment,
            "correct": all(self._checks[item] for item in self._checks)
        }

def write_result(result):
    with open(os.path.abspath(os.environ["GITHUB_OUTPUT"]), "a") as output_file:
        output_file.write(f"correctPullRequests={result}\n")

def write_lab_tag(pull_requests):
    all_lab_tags = []
    for pr in pull_requests:
        all_lab_tags.append(pr["lab_tag"])
    lab_tag = max(set(all_lab_tags), key=all_lab_tags.count)
    print(lab_tag)
    with open(os.path.abspath(os.environ["GITHUB_OUTPUT"]), "a") as output_file:
        output_file.write(f"lab-tag={lab_tag}")
    pass

def main():
    with open("opened_pull_requests.json", "r") as input_file:
        opened_pull_requests_json = input_file.read()

    opened_pull_requests = json.loads(opened_pull_requests_json)
    processed_pull_requests = []

    for pull_request in opened_pull_requests:
        processed_pull_requests.append(PullRequestChecker(pull_request).get_json())
    
    write_result(json.dumps(processed_pull_requests))
    write_lab_tag(processed_pull_requests)
if __name__ == "__main__":
    main()
