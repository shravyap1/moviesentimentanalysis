import csv


def load_data(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    return lines


def parse_data(filepath):
    texts = []
    labels = []

    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)  # skip header: review,sentiment

        for row in reader:
            if len(row) == 2:
                texts.append(row[0])
                labels.append(row[1])

    return texts, labels