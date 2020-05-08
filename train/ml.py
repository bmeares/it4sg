#! /usr/bin/python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8

"""
"""
import random
from copy import deepcopy
from matplotlib import pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import axes3d, Axes3D
from itertools import combinations
all_data = []
all_data_original = []
all_data_mins = []
all_data_maxes = []
all_data_means = []
test_data = []
train_data = []
dor_data = []
bod_data = []
typ_data = []
W = []
feature_count = 0
feature_labels = []
temp_W = W
init_J = 0
alpha = 0
init_alpha = alpha
alpha_decay = 0
z = []
degree = 2
iterations = 0
iter_data = []

def main():
    global test_data, train_data, W, iterations, iter_data, all_data, all_data_original

    alphas = [1.0]
    decays = [0.925]
    degrees = [5]
    j_value = 99999999
    best_J = j_value
    best_alpha = -1
    best_decay = -1
    display = False
    answer = input('Plot data? (def. y/N): ')
    if 'y' in answer.lower():
        display = True
    for a in alphas:
        for d in decays:
            for deg in degrees:
                init(al = a, al_decay = d, deg = degree, ask = True)
                print('alpha:', a, 'decay:', d, 'degree:', degree)
                print('Training, please wait',end="",flush=True)
                improve_weights()
                print_results()
                j_value = J(W, test_data)
                if (j_value < best_J):
                    best_J = j_value
                    best_alpha = a
                    best_decay = d
                j_title = str(a) + "_" + str(d) + "_" + str(deg)  + "_plot.png"
    if display:
        save_J_plot(j_title, iter_data, begin = 1)
        save_scatter3d(all_data_original)
        print('best_J:',best_J, 'best_alpha:',best_alpha, 'best_decay', best_decay)

    ### prompt user to enter new data
    while True:
        p = prompt_user()
        if p == 0:
            break
        loop(W,p)

"""
name: improve_weights
desc: the primary training driver function
"""
def improve_weights():
    global W, alpha, train_data, test_data, temp_W,iterations, iter_data, alpha_decay
    init_alpha = alpha
    J_change = -1
    last_J = J(W, test_data) + 1
    current_J = J(W, test_data)
    iter_data.append(current_J) ### include for zero iterations
    while J_change < 0:
        iterations += 1
        for i in range(len(W)):
            temp_W[i] = float(apply_alpha(i))
        print('.', end="", flush=True)
        W = deepcopy(temp_W)
        alpha = float(alpha_decay * alpha) ### scale alpha by alpha_decay to slowly approach best weights
        last_J = current_J
        current_J = J(W, test_data)
        J_change = current_J - last_J
        iter_data.append(current_J) ### store current J value for plotting later


def build_z(d):
    global z, feature_count
    def makeCounter_rec(base):
        def incDigit(num, pos):
            new = num[:]
            if(num[pos] == base - 1):
                new[pos] = 0
                if(pos < len(num) - 1):
                    return incDigit(new, pos + 1)
            else:
                new[pos] = new[pos] + 1
            return new

        def inc(num):
            return incDigit(num, 0)
        return inc

    z = []
    terms = []
    base = int(feature_count + 1)
    inc = makeCounter_rec(base)
    combo = [0] * int(d)
    z.append(combo)
    terms.append(combo)
    for i in range(base ** len(combo)):
        combo = inc(combo)
        z_ = sorted(deepcopy(combo))
        if z_ not in terms:
            terms.append(z_)
            z.append(z_)

"""
name   : zx_swap
desc   : Used for treating a degree 2 polynomial like a linear model.
         Groups together x's for each weight. For example, z0 corresponds to w0, and z[0] contains
         a list containing 0, which is the index for x0. All x's whose indicies are listed in z
         are multiplied together.
         z is defined above main()
returns: z, otherwise all corresponding x's to a weight with a given index
"""
def zx_swap(index, X):
    global z
    total = float(1)
    for i in z[index]:
        total *= X[i]

    return total

"""
name   : apply_alpha
desc   : Essentially the partial derivative of J. Applies alpha to a weight and the derivative to
         descend J.
returns: New weight value for a weight w
"""
def apply_alpha(index):
    global alpha, train_data, W
    derivative = float(0)
    w = W[index]
    for p in train_data:
        x_list = [1]
        for f in p:
            x_list.append(f)
        X = tuple(x_list)
        typ = p[-1]
        diff = h(W,X) - typ
        derivative += (diff * zx_swap(index, X))
    return w - (alpha * (derivative / float(len(train_data))))

"""
name: sq_err
desc: returns the squared error between a true type and predicted type.
"""
def sq_err(X,typ):
    global W
    pred_typ = h(W,X)
    diff = pred_typ - typ
    return diff ** 2

"""
name   : J
desc   : applies a set of weights (W) against a data set.
returns: cost value of weights W
"""
def J(W, data_set):
    total_sq_error = float(0)
    for p in data_set:
        X = [1]
        for f in p[:-1]:
            X.append(f)
        total_sq_error += float(sq_err(X, p[-1]))
    return float(total_sq_error / (2 * len(test_data)))

"""
name   : h
desc   : the hypothesis function. The relation between inputs (X) and weights (W).
returns: predicted typ
"""
def h_old(W, X):
    y = float(W[0])
    y += W[1] * X[1]
    y += W[2] * X[2]
    y += (W[3] * X[1]) * X[2]
    y += W[4] * (X[1] ** 2)
    y += W[5] * (X[2] ** 2)
    return y

def h(W, X):
    global degree
    y = 0.0
    for i in range(degree + 1):
        y += W[i] * zx_swap(i,X)
    return y


"""
name: init
desc: initializes weights, alpha, and calculates first J(W)
"""
def init(al = 0.5, al_decay = 0.995, deg = 2, ask = True):
    global W, temp_W, init_J, test_data, alpha, all_data, degree, z, feature_count, alpha_decay, init_alpha, iterations
    global degree

    def_filename = "normalized.csv"
    degree = ""
    if ask:
        degree = input("What degree polynomial would you like to model? (default: " + str(deg) + "): ")
    if degree == "":
        degree = deg
    else:
        degree = int(degree)
    filename = ""
    if ask:
        filename = input("Enter filename (default: " + def_filename + "): ")
    if filename == "":
        filename = def_filename
    read_data(filename)
    build_z(degree)
    W = [10.0] * len(z)
    temp_W = W

    divide_data()
    alpha = al
    alpha_decay = al_decay
    init_alpha = alpha
    init_J = J(W,test_data)
    iterations = 0

"""
name: read_data
desc: parses a tab-seperated file into several lists
"""
def read_data(filename):
    global all_data, total, feature_count, feature_labels, all_data_original
    all_data = []
    feature_labels = []
    all_data_original = [] 
    with open(filename,"r") as file:
        for line in file:
            d = line.strip().split(',')
            if len(d) == 1:
                total = int(d[0])
                continue
            try:
                d = [float(i) for i in d]
            except:
                feature_labels = d
                feature_count = len(d) - 1
                continue
            all_data.append(tuple(d))
    all_data_original = deepcopy(all_data)

"""
name: divide_data
desc: shuffles and splits data into training and test sets
"""
def divide_data():
    global train_data, test_data, all_data
    all_data_stats()
    all_data = random.sample(all_data, len(all_data))
    all_data = scale_data_set(all_data)
    tr_index = int(0.8 * len(all_data))
    train_data = all_data[:tr_index]
    test_data = all_data[tr_index:]


"""
name    : all_data_stats
desc    : calulates minimums, maximums, and means of features in all_data
"""
def all_data_stats():
    global all_data, all_data_mins, all_data_maxes, all_data_means, feature_count
    x_i_total,x_i_maxes,x_i_mins = [0.0] * feature_count,[0.0] * feature_count,[0.0] * feature_count
    x_i_means = [0.0] * feature_count
    for p in all_data:
        for i in range(len(p) - 1):
            x_i_total[i] += p[i]
            if p[i] > x_i_maxes[i]:
                x_i_maxes[i] = p[i]
            if p[i] < x_i_mins[i]:
                x_i_mins[i] = p[i]

    for f in range(feature_count):
        x_i_means[i] = float(x_i_total[i] / len(all_data))
    
    all_data_maxes = tuple(x_i_maxes)
    all_data_mins = tuple(x_i_mins)
    all_data_means = tuple(x_i_means)


"""
name    : scale_data
desc    : scales every point in a data set
returns : list of tuples (new data set)
"""
def scale_data_set(data):
    global all_data_mins, all_data_maxes, all_data_means
    scaled_data = []

    for p in data:
        scaled_data.append(scale_point(p,all_data_maxes,all_data_mins,all_data_means))
    return scaled_data


"""
name    : scale_point
desc    : scales features in a point according to mean normalization
returns : tuple (bod,dors,typ)
"""
def scale_point(p,maxes,mins,means):
    y = p[-1]
    new_point_list = []

    for i in range(len(maxes)):
        x_i = p[i]
        x_i_max = maxes[i]
        x_i_min = mins[i]
        x_i_mean = means[i]
        x_i = (x_i - x_i_mean) / (x_i_max - x_i_min)
        new_point_list.append(x_i)
    new_point_list.append(y)

    return tuple(new_point_list)


"""
name    : un_scale_data
desc    : un_scales every point in a data set
returns : list of tuples (new data set)
"""
def un_scale_data(data):
    global all_data_mins, all_data_maxes, all_data_means
    un_scaled_data = []
    for p in data:
        un_scaled_data.append(un_scale_point(p,all_data_maxes,all_data_mins,all_data_means))
    return un_scaled_data
    

"""
name    : un_scale_point
desc    : converts a scaled point to its original magnitude
returns : tuple (bod,dors,typ)
"""
def un_scale_point(s_p, maxes, mins, means):
    y = s_p[-1]
    new_point_list = []

    for i in range(len(maxes)):
        x_i = s_p[i]
        x_i_max = maxes[i]
        x_i_min = mins[i]
        x_i_mean = means[i]
        x_i = (x_i * (x_i_max - x_i_min)) + x_i_mean
        new_point_list.append(x_i)
    new_point_list.append(y)

    return tuple(new_point_list)


"""
name   : prompt_user
desc   : asks the user for values
returns: tuple of dorsal and body length
"""
def prompt_user():
    global all_data_maxes,all_data_mins,all_data_means,feature_count
    error = "Please enter the correct value."

    feats = []
    for i in range(feature_count):
        while True:
            try:
                l = "{:22s}".format(feature_labels[i])
                prompt = str(l) + ": "
                f = float(input(prompt))
                
            except:
                print(error)
                break
            else:
                break
        feats.append(f)

    if(feats == [0] * int(feature_count)):
        return 0
    feats.append(-1)
    p = tuple(feats)
    return scale_point(p, all_data_maxes,all_data_mins,all_data_means)

"""
name: print_results
desc: prints relevant information after training, such as number of iterations, alpha values,
      J(W) values, and the model definition
"""
def print_results():
    global iterations, alpha, init_alpha, W, test_data, init_J
   
    print("\nRESULTS:\n----------")
    print("Training iterations   :", iterations)
    print("Initial alpha value   :", init_alpha)
    print("End alpha value       : {:.2e}".format(alpha))
    print("Initial J(W)(test)    : {:.2e}".format(init_J))
    print("Final J(W)(test)      : {:.2e}".format(J(W,test_data)))
    print("Hypothesis model      :", equation_string(W))

"""
name   : equation_string
desc   : formats the weights into a human-readable string
returns: the mathematical definition of the hypothesis function (h(X)) as a string
"""
def equation_string(W):
    global z
    out = "h(X) = "
    w_i = 0
    for x_i_list in z:
        out += "({:.2e})".format(W[w_i])
        for x_i in x_i_list:
            if x_i > 0:
                out += "(x" + str(x_i) + ")"
        out += " + "
        w_i += 1

    return out[:-3]


"""
name: loop
desc: driver script for calculating and printing predicted types
"""
def loop(W,p):
    x_list = [1]
    for f in p:
        x_list.append(f)
    X = tuple(x_list)
    pred_typ = (h(W,X))
    out = "\n{:22s}: ".format('Predicted Usage')
    out += str(pred_typ) + '\n'
    for i in range(23):
        out += "#"
    print(out)

"""
name: save_scatter3d
desc: displays and writes 3D scatterplot
"""
def save_scatter3d(data):
        global feature_labels
        all_data_T = np.transpose(data)
        seen_plots = []
        for i in range(len(all_data_T)-1):
            for j in range(len(all_data_T)-1):
                plot_combo = sorted([i,j])
                if(i == j) or plot_combo in seen_plots:
                    continue
                else:
                    seen_plots.append(plot_combo)
                    fig = plt.figure(figsize = (20,10))
                    ax=fig.add_subplot(111,projection="3d")
                    plt.title('Standardized Data')
                    ax.scatter(all_data_T[i], all_data_T[j], all_data_T[-1], c='r',marker='D')#, cmap='hsv')
                    ax.set_xlabel(feature_labels[i])
                    ax.set_ylabel(feature_labels[j])
                    ax.set_zlabel(feature_labels[-1])
                    plt.show()

"""
name: save_J_plot
desc: displays and saves a plot of J-values with any given begin and end indicies
"""
def save_J_plot(filename, data, begin = 0, end = None, display = True):
    if end is None:
        end = len(data)
    sub_data = data[begin:end]
    #  plt.xticks(np.arange(begin,end,int(0.1 * (end - begin))))
    t = np.arange(begin,end,1)
    plt.xlabel('Iterations (' + str(begin) + '—' + str(end) + ')')
    plt.ylabel('J(W)')
    plt.title('Model Accuracy After ' + str((end)) + ' Iterations')
    figure_text = 'J(W) values of note:\nMin: ' + "{:.2e}".format(min(sub_data)) + '\n  (' + str(data.index(min(sub_data))) + ' iterations)'
    figure_text += '\nMax: ' + "{:.2e}".format(max(sub_data)) + '\n  (' + str(data.index(max(sub_data))) + ' iterations)'
    plt.figtext(0.5,0.65,figure_text) 
    plt.plot(t, data[begin:end], marker="o")
    plt.grid(True)
    plt.savefig(filename)
    if display:
        plt.show()
    plt.clf()

import datetime
def get_date():
    error = "\nEnter the date in this exact format: yyyy-mm-dd hr:min\n"
    while True:
        try:
            first,second = "",""
            result = input('\nEnter date (yyyy-mm-dd hr:min): ').split(' ')
            first,second = result
            yr = int(first.split('-')[0])
            mo = int(first.split('-')[1])
            dy = int(first.split('-')[2])
            hr = int(second.split(':')[0])
            mn = int(second.split(':')[1])

        except:
            if result == ['0']:
                return 0
            print(error)
        else:
            break
    return date_to_int(yr,mo,dy,hr,mn)

def date_to_int(yr,mo,dy,hr = 0, mn = 0):
    
    u = int(datetime.datetime(yr,mo,dy,hr,mn).timestamp())
    return u


if __name__ == "__main__":
    main()
