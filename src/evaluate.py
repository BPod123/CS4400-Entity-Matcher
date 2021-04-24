import pandas as pd
from os.path import join, abspath


def calculate_scores(validation_set: pd.DataFrame, output_file=join(abspath('../out'), "output.csv")):
    """
    Calculates the precision, recall, and F1 scores
    :param validation_set:
    :return: precision, recall, F1
    """
    output = pd.read_csv(output_file)
    validations = set()
    predictions = set()
    for i in range(len(output)):
        predictions.add((output["ltable_id"][i], output["rtable_id"][i]))
    for i in range(len(validation_set)):
        validations.add((validation_set["ltable_id"][i], validation_set["rtable_id"][i]))
    if len(predictions) == 0:
        return 0, 0, 0
    predicted_correct_matches = len(predictions.intersection(validations))
    precision = float(predicted_correct_matches) / float(len(predictions))
    recall = float(predicted_correct_matches) / float(len(validations))
    if recall == 0 or precision == 0:
        F1 = 0
    else:
        F1 = 2.0 / (1.0 / precision + 1.0 / recall)
    return precision, recall, F1


def generate_output(training_set, ltable, rtable, class_name="label", output_file=join(abspath('../out'), "output.csv")):
    from src.solution import run_solution
    run_solution(training_set, ltable, rtable, class_name, output_file=output_file)


def evaluate(training_set, validation_set, ltable, rtable, class_name="label",
             output_file=join(abspath('../out'), "output.csv")):
    generate_output(training_set, ltable, rtable, class_name, output_file=output_file)
    return calculate_scores(validation_set, output_file)


