GROUPS = [
    "1303",
    "1304",
    "1381",
    "1384"
]

STUDENTS = [
    (r"[A-Za-z]*", r"[A-Za-z]*")
]

GENERAL_PATH = r"\group_number_\last_name_\first_name/lab\lab_number"

FILES = [
    [
        {
            "path": r"report",
            "file_name": r"\group_number_\last_name_\first_name_lab\lab_number.pdf", 
            "context": "отчет"
        },
        {
            "path": r"src",
            "file_name": r".*\.py",
            "context": "PY"
        }
    ]
    for _ in range(5)    
]
