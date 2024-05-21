GROUPS = [
    "1303",
    "1304",
    "1381",
    "1384"
]

STUDENTS = [
    (r"[A-Za-z]*", r"[A-Za-z]*")
]

GENERAL_PATH = r"\group_number_\last_name_\first_name/\work_tag"

WORK_TAGS = ["lab1", "lab2", "prac1", "prac2", "prac3", "prac4", "cw"]

FILES = {
    work_tag: [
        {
            "path": r"report",
            "file_name": r"\group_number_\last_name_\first_name_\work_tag.pdf", 
            "context": "отчет"
        },
        {
            "path": r"src",
            "file_name": r"model.py",
            "context": "модель"
        },
        {
            "path": r"src",
            "file_name": r"weights_link.txt",
            "context": "ссылки на веса"
        }
    ]
    for work_tag in WORK_TAGS    
}
