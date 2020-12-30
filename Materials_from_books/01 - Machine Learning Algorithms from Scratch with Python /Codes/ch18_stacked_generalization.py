from math import sqrt, exp

# Prepare the KNN model
def knn_model(train):
    return train

# Calculate the Euclidean distance between two vectors
def euclidean_distance(row1, row2):
    distance = 0.0
    for i in range(len(row1)-1):
        distance += (row1[i]-row2[i])**2
    return sqrt(distance)

# Locate neighbors for a new row
def get_neighbors(train, test_row, num_neighbors):
    distances = list()
    for train_row in train:
        dist = euclidean_distance(test_row, train_row)
        distances.append((train_row, dist))
    distances.sort(key=lambda tup: tup[1])
    neighbors = list()
    for i in range(num_neighbors):
        neighbors.append(distances[i][0])
    return neighbors

# Make a prediction with KNN
def knn_predict(model, test_row, num_neighbors=2):
    neighbors = get_neighbors(model, test_row, num_neighbors)
    output_values = [row[-1] for row in neighbors]
    prediction = max(set(output_values), key=output_values.count)
    return prediction

# Make a prediction with weights
def perceptron_predict(model, row):
    activation = model[0]
    for i in range(len(row)-1):
        activation += model[i + 1] * row[i]
    return 1.0 if activation >= 0.0 else 0.0

# Estimate perceptron weights using stochastic gradient descent
def perceptron_model(train, l_rate=0.01, n_epoch=5000):
    weights = [0.0 for i in range(len(train[0]))]
    for _ in range(n_epoch):
        for row in train:
            prediction = perceptron_predict(weights, row)
            error = row[-1] - prediction
            weights[0] = weights[0] +l_rate * error
            for i in range(len(row)-1):
                weights[i + 1] = weights[i + 1] + l_rate * error * row[i] 
    return weights

# Make a prediction with coefficients
def logistic_regression_predict(model, row):
    yhat = model[0]
    for i in range(len(row)-1):
        yhat += model[i+1] * row[i]
    return 1.0 / (1.0 + exp(-yhat))

# Estimate logistic regression coefficients using stochastic gradient descent
def logistic_regression_model(train, l_rate=0.01, n_epoch=5000):
    coef = [0.0 for i in range(len(train[0]))]
    for _ in range(n_epoch):
        for row in train:
            yhat = logistic_regression_predict(coef, row)
            error = row[-1] - yhat
            coef[0] = coef[0] + l_rate * error * yhat * (1.0 - yhat)
            for i in range(len(row)-1):
                coef[i + 1] = coef[i + 1] + l_rate * error * yhat * (1.0 - yhat) * row[i]
    return coef

# Make predictions with sub-models and construct a new stacked row
def to_stacked_row(models, predict_list, row):
    stacked_row = list()
    for i in range(len(models)):
        prediction = predict_list[i](models[i], row)
        stacked_row.append(prediction)
    stacked_row.append(row[-1])
    return row[0:len(row)-1] + stacked_row

# Stacked Generalization Algorithm
def stacking(train, test):
    model_list = [knn_model, perceptron_model]
    predict_list = [knn_predict, perceptron_predict]
    models = list()
    for i in range(len(model_list)):
        model = model_list[i](train)
        models.append(model)
    stacked_dataset = list()
    for row in train:
        stacked_row = to_stacked_row(models, predict_list, row)
        stacked_dataset.append(stacked_row)
    stacked_model = logistic_regression_model(stacked_dataset)
    predictions = list()
    for row in test:
        stacked_row = to_stacked_row(models, predict_list, row)
        stacked_dataset.append(stacked_row)
        prediction = logistic_regression_predict(stacked_model, stacked_row)
        prediction = round(prediction)
        predictions.append(prediction) 
    return predictions