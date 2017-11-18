from urllib import request
from numpy import *
import plotly
from plotly.graph_objs import Scatter, Layout, Figure


def get_sample(url):
    page = request.urlopen(url).read().decode('utf-8')
    sample_input = []
    sample_output = []
    for line in page.splitlines():
        if line:
            tokens = line.split(',')
            sample_input.append(
                [float(tokens[0]), float(tokens[1]), float(tokens[2]), float(tokens[3])])
            sample_output.append(get_output_class(tokens[4]))
    return sample_input, sample_output


def get_output_class(string):
    if string == 'Iris-setosa':
        return 0
    if string == 'Iris-versicolor':
        return 1
    else:
        return 2


def get_folds_sets(input_values, output_values, folds):
    size = len(input_values)
    greater_rows = size % folds
    block = size // folds
    offset = (block + 1) * greater_rows
    folds_sets = []
    for i in range(folds):
        if i < greater_rows:
            indices = arange((block + 1) * i, (block + 1) * (i + 1))
            test_input = input_values[(block + 1) * i:(block + 1) * (i + 1), :]
            test_output = output_values[(block + 1) * i:(block + 1) * (i + 1)]
            train_input = delete(input_values, indices, axis=0)
            train_output = delete(output_values, indices, axis=0)
        else:
            start = offset + i * block
            end = offset + (i + 1) * block
            indices = arange(start, end)
            test_input = input_values[start:end, :]
            test_output = output_values[start:end]
            train_input = delete(input_values, indices, axis=0)
            train_output = delete(output_values, indices, axis=0)
        folds_sets.append([train_input, train_output, test_input, test_output])
    return folds_sets


def get_distance(first_item, second_item):
    sum = 0.0
    size = 4
    for i in range(size):
        sum += (first_item[i] - second_item[i]) ** 2
    return pow(sum, 0.5)


def kNN_classifier(sample_input, sample_output, nearest_neighbours_amount, item_to_classify):
    distances = []
    for sample_item in sample_input:
        current_distance = get_distance(sample_item, item_to_classify)
        distances.append(current_distance)
    sorted_indexes = argsort(array(distances))
    sorted_output = sample_output[sorted_indexes]
    classes = {0: 0, 1: 0, 2: 0}
    for index in range(nearest_neighbours_amount):
        class_value = sorted_output[index]
        classes[class_value] += 1
    max_class = max(classes, key=(lambda key: classes[key]))
    return max_class


def shuffle_sample(input_data, output_data):
    indexes = arange(input_data.shape[0])
    random.shuffle(indexes)
    end_index = int(0.9 * input_data.shape[0])
    sample_indexes = indexes[:end_index]
    sample_input = input_data[sample_indexes]
    sample_output = output_data[sample_indexes]
    test_indexes = indexes[end_index:]
    test_input = input_data[test_indexes]
    test_output = output_data[test_indexes]
    return sample_input, sample_output, test_input, test_output


def show_plot(data, title, filename='plot.html'):
    layout_comp = Layout(
        title=title
    )
    fig_comp = Figure(data=data, layout=layout_comp)
    plotly.offline.plot(fig_comp, filename=filename)


def do_cross_validation(sample_input, sample_output, fold_amount, minimum, maximum):
    folds = get_folds_sets(sample_input, sample_output, fold_amount)
    average_errors = zeros(maximum - minimum + 1)
    k_values = list(range(minimum, maximum + 1))
    data = []
    for i in range(fold_amount):
        print('Fold № ' + str(i) + ' started...')
        input_values, output_values, test_input, test_output = folds[i]
        fold_errors = []
        for k in k_values:
            error = 0
            fold_size = test_input.shape[0]
            for index in range(fold_size):
                output_class = kNN_classifier(input_values, output_values, k, test_input[index])
                if output_class != test_output[index]:
                    error += 1
            error /= fold_size
            fold_errors.append(error)
            average_errors[k - minimum] += error / fold_amount
    #    trace = Scatter(
    #        x=k_values,
    #        y=fold_errors,
    #        name='Fold ' + str(i + 1)
    #    )
    #    data.append(trace)
    #show_plot(data, 'Error', 'folds.html')
    #avg_trace = Scatter(
    #    x=k_values,
    #    y=average_errors,
    #    name='Average fold error'
    #)
    #show_plot([avg_trace], 'Average error', 'avg_errors.html')
    errors_indexes = argsort(average_errors)
    best_neighbour_amount = minimum + errors_indexes[0]
    return best_neighbour_amount


input_data, output_data = get_sample('http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data')
sample_input, sample_output, test_input, test_output = shuffle_sample(array(input_data), array(output_data))
best_neighbour_amount = do_cross_validation(sample_input, sample_output, 5, 1, 100)
error = 0
size = test_input.shape[0]
for index in range(size):
    output_class = kNN_classifier(sample_input, sample_output, best_neighbour_amount, test_input[index])
    if output_class != test_output[index]:
        error += 1
print('Wrong classes: '  + str(error) + ' of ' + str(size))
print('Error: '   + str(error / size))