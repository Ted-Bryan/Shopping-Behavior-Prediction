import csv
import sys

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")

    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data(sys.argv[1])
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    # Train model and make predictions
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(y_test, predictions)

    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate (Sensitivity): {100 * sensitivity:.2f}%")
    print(f"True Negative Rate (Specificity): {100 * specificity:.2f}%")


def load_data(filename):
    """
    Load shopping data from a CSV file `filename` and convert into a list of
    evidence lists and a list of labels. Return a tuple (evidence, labels).

    evidence should be a list of lists, where each list contains the
    following values, in order:
        - Administrative, an integer
        - Administrative_Duration, a floating point number
        - Informational, an integer
        - Informational_Duration, a floating point number
        - ProductRelated, an integer
        - ProductRelated_Duration, a floating point number
        - BounceRates, a floating point number
        - ExitRates, a floating point number
        - PageValues, a floating point number
        - SpecialDay, a floating point number
        - Month, an index from 0 (January) to 11 (December)
        - OperatingSystems, an integer
        - Browser, an integer
        - Region, an integer
        - TrafficType, an integer
        - VisitorType, an integer 0 (not returning) or 1 (returning)
        - Weekend, an integer 0 (if false) or 1 (if true)

    labels should be the corresponding list of labels, where each label
    is 1 if Revenue is true, and 0 otherwise.
    """
    # Map month abbreviations to a 0-based index
    months = {
        "Jan": 0, "Feb": 1, "Mar": 2, "Apr": 3,
        "May": 4, "June": 5, "Jul": 6, "Aug": 7,
        "Sep": 8, "Oct": 9, "Nov": 10, "Dec": 11
    }

    evidence = []
    labels = []

    with open(filename, "r", newline='', encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Parse evidence
            data = []
            data.append(int(row["Administrative"]))
            data.append(float(row["Administrative_Duration"]))
            data.append(int(row["Informational"]))
            data.append(float(row["Informational_Duration"]))
            data.append(int(row["ProductRelated"]))
            data.append(float(row["ProductRelated_Duration"]))
            data.append(float(row["BounceRates"]))
            data.append(float(row["ExitRates"]))
            data.append(float(row["PageValues"]))
            data.append(float(row["SpecialDay"]))
            data.append(months[row["Month"]])  # Convert month string to index
            data.append(int(row["OperatingSystems"]))
            data.append(int(row["Browser"]))
            data.append(int(row["Region"]))
            data.append(int(row["TrafficType"]))

            # VisitorType: 1 if "Returning_Visitor", else 0
            if row["VisitorType"] == "Returning_Visitor":
                data.append(1)
            else:
                data.append(0)

            # Weekend: 1 if TRUE, else 0
            if row["Weekend"] == "TRUE":
                data.append(1)
            else:
                data.append(0)

            evidence.append(data)

            # Labels: 1 if Revenue == "TRUE", else 0
            label = 1 if row["Revenue"] == "TRUE" else 0
            labels.append(label)

    return (evidence, labels)


def train_model(evidence, labels):
    """
    Given a list of evidence lists and a list of labels, return a
    fitted k-nearest neighbor model (k=1) trained on the data.
    """
    model = KNeighborsClassifier(n_neighbors=1)
    model.fit(evidence, labels)
    return model


def evaluate(labels, predictions):
    """
    Given a list of actual labels and a list of predicted labels,
    return a tuple (sensitivity, specificity).

    Assume each label is either a 1 (positive) or 0 (negative).

    `sensitivity` is the proportion of actual positive labels that
    were accurately identified.
    `specificity` is the proportion of actual negative labels that
    were accurately identified.
    """
    true_positives = 0
    false_negatives = 0
    true_negatives = 0
    false_positives = 0

    for actual, predicted in zip(labels, predictions):
        if actual == 1 and predicted == 1:
            true_positives += 1
        elif actual == 1 and predicted == 0:
            false_negatives += 1
        elif actual == 0 and predicted == 0:
            true_negatives += 1
        elif actual == 0 and predicted == 1:
            false_positives += 1

    # Avoid division by zero in edge cases
    sensitivity = true_positives / \
        (true_positives + false_negatives) if (true_positives + false_negatives) else 0
    specificity = true_negatives / \
        (true_negatives + false_positives) if (true_negatives + false_positives) else 0

    return (sensitivity, specificity)


if __name__ == "__main__":
    main()
