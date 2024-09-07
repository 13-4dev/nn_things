import numpy as np
import functions
import cv2
import os

rows = 64
cols = 64
channels = 3

def read_image(file_path):
  #check itegrity file
  statfile = os.stat(file_path)
  filesize = statfile.st_size
  if filesize == 0:
    os.remove(file_path)
  else:
    img = cv2.imread(file_path, cv2.IMREAD_COLOR)
    return cv2.resize(img, (rows, cols), interpolation=cv2.INTER_CUBIC)

def prepare_data(images):
    m = len(images)
    image = np.zeros((m, rows, cols, channels), dtype=np.uint8)
    true_label = np.zeros((1, m))
    for i, image_file in enumerate(images):
        image[i,:] = read_image(image_file)
        if 'dog' in image_file.lower():
            true_label[0, i] = 1
        elif 'cat' in image_file.lower():
            true_label[0, i] = 0
    return image, true_label

def initialize_with_zeros_weigths(dim):
    weights = np.zeros((dim, 1))
    return weights

def initialize_with_zeros_bias():
    bias = 0
    return bias

def sigmoid(x):
  # Sigmoid activation function: f(x) = 1 / (1 + e^(-x))
  return 1 / (1 + np.exp(-x))

train_dir = 'data/train/'
test_dir = 'data/test/'

train_images = [train_dir+i for i in os.listdir(train_dir)]
test_images =  [test_dir+i for i in os.listdir(test_dir)]

train_set_x, train_set_y = prepare_data(train_images)
test_set_x, test_set_y = prepare_data(test_images)

train_set_x_flatten = train_set_x.reshape(train_set_x.shape[0], rows*cols*channels).T
test_set_x_flatten = test_set_x.reshape(test_set_x.shape[0], -1).T

train_set_x = train_set_x_flatten/255
test_set_x = test_set_x_flatten/255




class NeuralNetwork:

  def __init__(self):
    # Weights
    self.weight = initialize_with_zeros_weigths(train_set_x.shape[0])
    # Biases
    self.bias = initialize_with_zeros_bias()

  def optimize(self, w, b, X, Y, num_iterations, learning_rate, print_cost = True):
    costs = []    
    for i in range(num_iterations):
        # Cost and gradient calculation
        gradients, cost = self.propagate(w, b, X, Y)
        
        # Retrieve derivatives from grads
        deriv_weight = gradients["gradient_loss_weight"]
        deriv_bias = gradients["gradient_loss_bias"]

        # update w and b
        w = w - learning_rate*deriv_weight
        b = b - learning_rate*deriv_bias

        # Record the costs
        if i % 10 == 0:
            costs.append(cost)
            
        # Print the cost every 100 training iterations
        if print_cost and i % 10 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
            # update w and b to dictionary

    params = {"w": w,
                "b": b}
    
        # update derivatives to dictionary
    grads = {"dw": deriv_weight,
              "db": deriv_bias}
    
    return params, grads, costs

  def propagate(self, w, b, X, Y):
    m = X.shape[1]
    
     # FORWARD PROPAGATION (FROM X TO COST)
    z = np.dot(w.T, X)+b # tag 1
    A = sigmoid(z) # tag 2                                    
    cost = (-np.sum(Y*np.log(A)+(1-Y)*np.log(1-A)))/m # logistic cost
    
    # BACKWARD PROPAGATION (TO FIND GRAD)
    dw = (np.dot(X,(A-Y).T))/m # tag 6
    db = np.average(A-Y) # tag 7

    cost = np.squeeze(cost)
    grads = {"gradient_loss_weight": dw,
             "gradient_loss_bias": db}
    
    return grads, cost

  def model(self,X_train, 
            Y_train, 
            X_test, 
            Y_test, 
            num_iterations = 2000, 
            learning_rate = 0.5, 
            print_cost = True):
    # Gradient descent
    parameters, grads, costs = self.optimize(self.weight, self.bias, X_train, Y_train, num_iterations, learning_rate, print_cost)
    
    # Retrieve parameters w and b from dictionary "parameters"
    w = parameters["w"]
    b = parameters["b"]
    
    # Predict test/train set examples
    Y_prediction_test = self.predict(w,b,X_test)
    Y_prediction_train = self.predict(w,b,X_train)

    # Print train/test Errors
    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

    
    dict = {"costs": costs,
            "Y_prediction_test": Y_prediction_test,
            "Y_prediction_train": Y_prediction_train,
            "w": w,
            "b": b,
            "learning_rate": learning_rate,
            "num_iterations:": num_iterations}
    
    return dict

  def predict(self,w, b, X):
    m = X.shape[1]
    Y_prediction = np.zeros((1, m))
    w = w.reshape(X.shape[0], 1)
    
    z = np.dot(w.T, X) + b
    A = sigmoid(z)
    
    for i in range(A.shape[1]):
        # Convert probabilities A[0,i] to actual predictions p[0,i]
        if A[0,i] > 0.5:
            Y_prediction[[0],[i]] = 1
        else: 
            Y_prediction[[0],[i]] = 0
    
    return Y_prediction

neural_network = NeuralNetwork()
model_nt = neural_network.model(train_set_x, 
                                train_set_y, 
                                test_set_x, 
                                test_set_y, 
                                5000, 
                                0.003)

test_image = "cat.jpg"
my_image = read_image(test_image).reshape(1, rows*cols*channels).T
my_predicted_image = neural_network.predict(model_nt["w"], model_nt["b"], my_image)

print("cat.jpg is: ",np.squeeze(my_predicted_image))

test_image = "dog.jpg"
my_image = read_image(test_image).reshape(1, rows*cols*channels).T
my_predicted_image = neural_network.predict(model_nt["w"], model_nt["b"], my_image)


print("dog.jpg is ",np.squeeze(my_predicted_image))