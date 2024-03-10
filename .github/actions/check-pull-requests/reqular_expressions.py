"""
    general pull-request title that should specify all necessary information about pull-request
"""
pull_request_title = r"^13(03|04|81|84)_[A-Z][a-z]*_[A-Z][a-z]*_lab(1|2|3|4|5)$"

"""
    return all necessary information about pull request, derived from its title
"""
def get_pull_request_info(pull_request_title):
    pull_request_data = pull_request_title.split("_")

    pull_request_info = {
        "group_number": pull_request_data[0],
        "last_name": pull_request_data[1],
        "first_name": pull_request_data[2],
        "lab_number": pull_request_data[3]
    }

    return pull_request_info

"""
    reutrns the path where all files in pull request should be located
"""
def get_general_folder_path(pull_request_info):
    student_folder_title = f"{pull_request_info['group_number']}_{pull_request_info['last_name']}_{pull_request_info['first_name']}"

    subfolder_title = pull_request_info["lab_number"]

    return f"{student_folder_title}/{subfolder_title}"

"""
    return path in which report should be located, including report file title
"""
def get_report_path(pull_request_info):
    general_folder_path = get_general_folder_path(pull_request_info)

    report_subfolder = "report"

    report_title = f"{pull_request_info['group_number']}_{pull_request_info['last_name']}_{pull_request_info['first_name']}_{pull_request_info['lab_number']}.pdf"

    return f"{general_folder_path}/{report_subfolder}/{report_title}"

"""
    return list of paths in which should be located pull-request source files, including their titles
"""
def get_sources_paths(pull_request_info):
    general_folder_path = get_general_folder_path(pull_request_info)

    sources_subfolder = "src"

    source_title = "main.py"

    return [f"{general_folder_path}/{sources_subfolder}/{source_title}"]
