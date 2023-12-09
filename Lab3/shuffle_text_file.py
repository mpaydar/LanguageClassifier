import random

def shuffle_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    random.shuffle(lines)

    with open(file_path, 'w', encoding='utf-8') as file:
        file.writelines(lines)


file_path = './Data.txt'
shuffle_text_file(file_path)