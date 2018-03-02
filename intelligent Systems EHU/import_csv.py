import pandas as pd

def load_data_from_csv(input_csv):
    df = pd.read_csv(input_csv, header=0)
    csv_headings = list(df.columns.values)
    feature_names = csv_headings[:len(csv_headings) - 1]
    df = df._get_numeric_data()
    numpy_array = df.as_matrix()
    _, number_of_columns = numpy_array.shape
    instances = numpy_array[:, 0:number_of_columns - 1]

    labels = []
    for label in numpy_array[:, number_of_columns - 1:number_of_columns].tolist():
        labels.append(label[0])
    return feature_names, instances, labels