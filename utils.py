import csv


def read_csv(path):
    """[summary]

    Args:
        path ([type]): [description]

    Returns:
        [type]: [description]
    """
    csv_file = open(path, 'r')
    csv_reader = csv.reader(csv_file)
    samples = [sample for sample in csv_reader]
    csv_file.close()

    return samples


def write_csv(path, content):
    """[summary]

    Args:
        path ([type]): [description]
        content ([type]): [description]
    """
    csv_file = open(path, 'w')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerows(content)
    csv_file.close()
