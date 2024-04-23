import argparse
from main import run_checks as run_main_checks

def main():
    parser = argparse.ArgumentParser(description="Interface for running checks on lab tasks")
    parser.add_argument("lab_number", type=int, help="The number of the lab task to run checks for")

    args = parser.parse_args()

    # Проверка номера лабораторной работы
    lab_number = args.lab_number
    if lab_number not in [0, 1, 2, 3]:
        print("Выберите номер лабораторной работы из доступных: 1, 2, 3")
        return

    # Запуск проверок для выбранной лабораторной работы
    if lab_number == 0:
        run_main_checks()
    elif lab_number == 1:
        run_lab_1_checks()
    elif lab_number == 2:
        run_lab_2_checks()
    elif lab_number == 3:
        run_lab_3_checks()

# Реализуйте здесь проверки для разных лабораторных работ
def run_lab_1_checks():
    print("Запуск проверок для первой лабораторной работы...")

def run_lab_2_checks():
    print("Запуск проверок для второй лабораторной работы...")

def run_lab_3_checks():
    print("Запуск проверок для третьей лабораторной работы...")

if __name__ == "__main__":
    main()
