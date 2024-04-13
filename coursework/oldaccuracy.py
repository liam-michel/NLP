import numpy as np
label_encoding = {"B-O": 0, "B-AC": 1, "B-LF": 2, "I-LF": 3}
inverse_label_map = {v: k for k, v in label_encoding.items()}

def calculate_results(trainer, dataset):

    predictions, labels, _ = trainer.predict(dataset)
    predictions = np.argmax(predictions, axis=2)

    textual_true_predictions = [
        [inverse_label_map[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    textual_true_labels = [
        [inverse_label_map[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions=textual_true_predictions, references=textual_true_labels)

    return results

