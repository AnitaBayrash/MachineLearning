import urllib.request
import os.path
import subprocess
import numpy
from svmutil import *
from math import sqrt
import plotly
from plotly.graph_objs import Scatter, Layout, Figure


def get_sample(url):
    print("Reading sample data...")
    page = urllib.request.urlopen(url).read().decode('utf-8')
    sample = []
    sample_strings = []
    for line in page.splitlines():
        if line:
            sample_item = []
            tokens = line.split(',')
            sample_item.append(int(tokens[57]))
            sample_string = str(int(tokens[57])) + " "
            for i in range(57):
                sample_string += str(i + 1) + ":" + tokens[i]
                if i != 56:
                    sample_string += " "
                else:
                    sample_string += '\n'
                sample_item.append(float(tokens[i]))
            sample.append(sample_item)
            sample_strings.append(sample_string)
    return sample, sample_strings


def write_files(sample_lines, train_filename, test_filename):
    print("Writing to files...")
    with open(os.path.join(train_filename), 'w') as train_file:
        train_file.writelines(sample_lines[0:3450])
    with open(os.path.join(test_filename), 'w') as test_file:
        test_file.writelines(sample_lines[3450:4601])


def scale_sample(svm_scale, parameters_file, train_filename, train_scaled):
    cmd = '{0} -l 0 -u 1 -s "{1}" "{2}" > "{3}"'.format(svm_scale, parameters_file, train_filename, train_scaled)
    print('Scaling training sample...')
    out, err = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE).communicate()
    print(out)


def scale_test_sample(svm_scale, parameters_file, train_filename, train_scaled):
    cmd = '{0} -l 0 -u 1 -r "{1}" "{2}" > "{3}"'.format(svm_scale, parameters_file, train_filename, train_scaled)
    print('Scaling test sample...')
    out, err = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE).communicate()
    print(out)


def get_sample_matrix(sample_data_file, weights_size):
    matrix = []
    results = []
    with open(sample_data_file, 'r') as sample_file:
        lines = sample_file.readlines()
    for line in lines:
        weights = [0] * weights_size
        tokens = line.split('\r\n')[0].split(' ')
        length = len(tokens) - 1
        results.append(int(tokens[0]))
        for i in range(1, length):
            weights_items = tokens[i].split(':')
            index = int(weights_items[0]) - 1
            weight = float(weights_items[1])
            weights[index] = weight
        matrix.append(weights)
    return numpy.array(matrix), numpy.array(results)


def get_folds_sets(input_data, output_values, folds):
    size = len(input_data)
    greater_rows = size % folds
    block = size // folds
    offset = (block + 1) * greater_rows
    folds_sets = []
    input_values = numpy.matrix(input_data)
    for i in range(folds):
        if i < greater_rows:
            indices = numpy.arange((block + 1) * i, (block + 1) * (i + 1))
            test_input = input_values[(block + 1) * i:(block + 1) * (i + 1), :]
            test_output = output_values[(block + 1) * i:(block + 1) * (i + 1)]
            train_input = numpy.delete(input_values, indices, axis=0)
            train_output = numpy.delete(output_values, indices, axis=0)
        else:
            start = offset + i * block
            end = offset + (i + 1) * block
            indices = numpy.arange(start, end)
            test_input = input_values[start:end, :]
            test_output = output_values[start:end]
            train_input = numpy.delete(input_values, indices, axis=0)
            train_output = numpy.delete(output_values, indices, axis=0)
        folds_sets.append([numpy.array(train_input), train_output, test_input, test_output])
    return folds_sets


def get_errors_info(errors):
    length = len(errors)
    average_error = sum(errors) / length
    result_sum = 0.0
    for i in range(length):
        result_sum += (errors[i] - average_error) ** 2
    result_sum /= length
    result = sqrt(result_sum)
    return result, average_error


def train_svm(fold_sets, c_parameter, d_parameter):
    print('  Selected parameters: C=' + str(c_parameter) + "; d=" + str(d_parameter))
    errors = []
    for k in range(10):
        print("Fold № " + str(k))
        train_input, train_output, test_input, test_output = fold_sets[k]
        problem = svm_problem(train_output.tolist(), train_input.tolist())
        params = "-t 1 -q -d " + str(d_parameter) + " -c " + str(c_parameter)
        fold_model = svm_train(problem, params)
        #svm_save_model('fold_' + str(k) + '_scale.model', fold_model)
        test_output_predicted, p_acc, p_vals = svm_predict(test_output.tolist(), test_input.tolist(), fold_model)
        ACC, MSE, SCC = evaluations(test_output.tolist(), test_output_predicted)
        errors.append(MSE)
    print(errors)
    deviation, average = get_errors_info(errors)
    train_input, train_output, test_input, test_output = fold_sets[0]
    sample_input = test_input.tolist()
    sample_input.extend(train_input.tolist())
    sample_output = test_output.tolist()
    sample_output.extend(train_output.tolist())
    sample_problem = svm_problem(sample_output, sample_input)
    model = svm_train(sample_problem, "-t 1 -q -d " + str(d_parameter) + " -c " + str(c_parameter))
    svm_save_model('model.model', model)
    output_predicted, p_acc, p_vals = svm_predict(sample_output, sample_input, model)
    ACC_sample, MSE, SCC = evaluations(sample_output, output_predicted)
    return deviation, average, 100 - ACC_sample


def plot_errors(fold_sets, k, degree, filename):
    C_values = []
    errors_plus_derivation = []
    errors_minus_derivation = []
    errors = []
    min_error = 1000
    best_degree = 0
    best_C = 0
    for k in range(-k, k + 1):
        C = pow(2, k)
        C_values.append(C)
        derivation, average, emp_risk = train_svm(fold_sets, C, degree)
        errors.append(average)
        errors_minus_derivation.append(average - derivation)
        errors_plus_derivation.append(average + derivation)
        print("C=2**" + str(k) + ": empirical risk = " + str(emp_risk / 100))
        if average < min_error:
            min_error = average
            best_degree = degree
            best_C = C
    trace = Scatter(
        x=C_values,
        y=errors,
        name='Average error'
    )
    trace_minus = Scatter(
        x=C_values,
        y=errors_minus_derivation,
        name='Average error - derivation'
    )
    trace_plus = Scatter(
        x=C_values,
        y=errors_plus_derivation,
        name='Average error + derivation'
    )
    data = [trace, trace_plus, trace_minus]
    layout_comp = Layout(
        title='Degree d={0}'.format(str(degree))
    )
    fig_comp = Figure(data=data, layout=layout_comp)
    plotly.offline.plot(fig_comp, filename=filename)
    return min_error, best_C, best_degree


def get_min_error_pair(error_results):
    error_data = numpy.array(error_results)
    ind = numpy.argsort(error_data[:, 0])
    error_data = error_data[ind]
    best_C = numpy.asscalar(error_data[0, 1])
    best_degree = numpy.asscalar(error_data[0, 2])
    return best_error, best_C, best_degree


def plot_best_c_errors(fold_sets, test_input, test_output, c_parameter):
    degrees = []
    k_fold_errors = []
    test_errors = []
    k_fold_sv = []
    for d_parameter in range(1, 5):
        degrees.append(d_parameter)
        deviation, average, risk = train_svm(fold_sets, c_parameter, d_parameter)
        k_fold_errors.append(average)
        with open("model.model", 'r') as model_file:
            lines = model_file.readlines()
            for line in lines:
                if line.startswith('total_sv'):
                    k_fold_sv.append(int(line.split(' ')[1]))
                    break
        model = svm_load_model("model.model")
        output_predicted, p_acc, p_vals = svm_predict(test_output.tolist(), test_input.tolist(), model)
        ACC_sample, MSE, SCC = evaluations(test_output.tolist(), output_predicted)
        test_errors.append(MSE)
    trace_cv = Scatter(
        x=degrees,
        y=k_fold_errors,
        name='Cross validation error'
    )
    trace_test = Scatter(
        x=degrees,
        y=test_errors,
        name='Test error'
    )
    data = [trace_cv, trace_test]
    layout_comp = Layout(
        title='Degree errors'
    )
    fig_comp = Figure(data=data, layout=layout_comp)
    plotly.offline.plot(fig_comp, filename="errors.html")
    trace_sv = Scatter(
        x=degrees,
        y=k_fold_sv,
        name='SV amount'
    )
    data = [trace_sv]
    layout_sv = Layout(
        title='SV amount'
    )
    fig_sv = Figure(data=data, layout=layout_sv)
    plotly.offline.plot(fig_sv, filename="sv_amount.html")


url = 'http://www.cs.nyu.edu/~mohri/yml/spambase.data.shuffled'
# sample parameters
train_filename = "train.data"
train_scaled_filename = "train_scaled.data"
test_filename = "test.data"
parameters_file = "parameters.data"
test_scaled_filename = "test_scaled.data"
# cross-validation parameters
parts = 10
k = 20
sample, sample_strings = get_sample(url)
write_files(sample_strings, train_filename, test_filename)
scale_sample(os.path.join("svm-scale.exe"), parameters_file, train_filename, train_scaled_filename)

sample_input, sample_output = get_sample_matrix(os.path.join(train_scaled_filename), 57)
fold_sets = get_folds_sets(sample_input, sample_output, parts)
error_results = []
best_error, best_C, best_degree = plot_errors(fold_sets, k, 1, "degree_1.html")
error_results.append([best_error, best_C, best_degree])
print('Degree=1: Best C: ' + str(best_C) + "; d=" + str(best_degree))
best_error, best_C, best_degree = plot_errors(fold_sets, k, 2, "degree_2.html")
error_results.append([best_error, best_C, best_degree])
print('Degree=2: Best C: ' + str(best_C) + "; d=" + str(best_degree))
best_error, best_C, best_degree = plot_errors(fold_sets, k, 3, "degree_3.html")
error_results.append([best_error, best_C, best_degree])
print('Degree=3: Best C: ' + str(best_C) + "; d=" + str(best_degree))
best_error, best_C, best_degree = plot_errors(fold_sets, k, 4, "degree_4.html")
error_results.append([best_error, best_C, best_degree])
print('Degree=4: Best C: ' + str(best_C) + "; d=" + str(best_degree))
best_error, best_C, best_degree = get_min_error_pair(error_results)
print('Total: Best C: ' + str(best_C) + "; d=" + str(best_degree))

scale_test_sample(os.path.join("svm-scale.exe"), parameters_file, test_filename, test_scaled_filename)
test_input, test_output = get_sample_matrix(os.path.join(test_scaled_filename), 57)
plot_best_c_errors(fold_sets, test_input, test_output, best_C)
