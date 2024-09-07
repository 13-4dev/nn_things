import numpy as np
import array
import matplotlib.pyplot as plt
np.random.seed(0)

plt.style.use('_mpl-gallery')
fig, ax = plt.subplots()

errors = []
epochsArr = []
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

amountData = 0



def binary_cross_entropy(y_pred, y_true):
    return np.squeeze(-np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))/amountData)

def binary_cross_entropy_derivative(y_pred, y_true):
    #print("inputX: ", inputX)
    #print("y_pred: ", y_pred)
    #print("y_true: ", y_true)

    return y_pred * (y_pred - y_true).mean() 

def deriv_sigmoid(x):
    fx = sigmoid(x)
    return fx * (1 - fx)

def softmax(xs):
    return np.exp(xs) / np.sum(np.exp(xs))

# Define dataset
x_data = np.array([
  [180, 120],  # Fat
  [180, 60],   # Skin
  [175, 110],   # Fat
  [170, 50], # Skin
  [170, 120],  # Fatnn.py
  [170, 60],   # Skin
  [175, 130],   # Fat
  [165, 50], # Skin
  [180, 100],  # Fat
  [180, 63],   # Skin
  [175, 130],   # Fat
  [170, 55], # Skin
  [173, 120],  # Fat
  [170, 58],   # Skin
  [177, 130],   # Fat
  [165, 54], # Skin

  [170, 105],  # Fat
  [165, 56],   # Skin
  [175, 113],   # Fat
  [163, 54], # Skin
  [167, 126],  # Fat
  [166, 56],   # Skin
  [190, 153],   # Fat
  [164, 50], # Skin
  [182, 106],  # Fat
  [185, 63],   # Skin
  [179, 133],   # Fat
  [170, 55], # Skin
  [173, 124],  # Fat
  [165, 58],   # Skin
  [173, 136],   # Fat
  [168, 55], # Skin
# fix it  
  [185, 117],  # Fat
  [174, 67],   # Skin
  [179, 119],   # Fat
  [177, 57], # Skin
  [171, 121],  # Fat
  [172, 63],   # Skin
  [174, 137],   # Fat
  [162, 54], # Skin
  [188, 108],  # Fat
  [182, 63],   # Skin
  [173, 134],   # Fat
  [179, 55], # Skin
  [192, 125],  # Fat
  [173, 58],   # Skin
  [173, 135],   # Fat
  [165, 54], # Skin

  [177, 135],  # Fat
  [164, 56],   # Skin
  [176, 117],   # Fat
  [166, 56], # Skin
  [177, 124],  # Fat
  [183, 55],   # Skin
  [194, 156],   # Fat
  [163, 53], # Skin
  [185, 126],  # Fat
  [183, 65],   # Skin
  [175, 136],   # Fat
  [175, 53], # Skin
  [177, 126],  # Fat
  [168, 54],   # Skinnn.py
  [175, 133],   # Fat
  [169, 59], # Skin
  
  [171, 68],  # Fat
  [172, 70],   # Fat
  [170, 69],   # Fat
  [166, 73], # Fat
  [174, 75],  # Fat
  [176, 67],   # Fat
  [178, 66],   # Fat
  [179, 80], # Fat
  [174, 80],  # Fat
  [165, 84],   # Fat
  [166, 66],   # Fat
  [168, 67], # Fat
  [178, 90],  # Fat
  [169, 70],   # Fat
  [180, 90],   # Fat
  [170, 70], # Fat

  [170, 67],  # Fat
  [178, 66],   # Fat
  [174, 83],   # Fat
  [164, 86], # Fat
  [169, 84],  # Fat
  [160, 83],   # Fat
  [180, 80],   # Fat
  [167, 78], # Fat
  [170, 76],  # Fat
  [162, 75],   # Fat
  [163, 73],   # Fat
  [167, 71], # Fat
  [173, 70],  # Fat
  [178, 82],   # Fat
  [179, 81],   # Fat
  [173, 80], # Fat
])




amountData = len(x_data) 

y_data = np.array([
  [1], 
  [0], 
  [1], 
  [0],
  [1], 
  [0], 
  [1], 
  [0],
  [1], 
  [0], 
  [1], 
  [0],
  [1], 
  [0], 
  [1], 
  [0],

  [1], 
  [0], 
  [1], 
  [0],
  [1], 
  [0], 
  [1], 
  [0],
  [1], 
  [0], 
  [1], 
  [0],
  [1], 
  [0], 
  [1], 
  [0],

  [1], 
  [0], 
  [1], 
  [0],
  [1], 
  [0], 
  [1], 
  [0],
  [1], 
  [0], 
  [1], 
  [0],
  [1], 
  [0], 
  [1], 
  [0],

  [1], 
  [0], 
  [1], 
  [0],
  [1], 
  [0], 
  [1], 
  [0],
  [1], 
  [0], 
  [1], 
  [0],
  [1], 
  [0], 
  [1], 
  [0],
    #FAT
  [1], 
  [1], 
  [1], 
  [1],
  [1], 
  [1], 
  [1], 
  [1],
  [1], 
  [1], 
  [1], 
  [1],
  [1], 
  [1], 
  [1], 
  [1], 
    #FAT
  [1], 
  [1], 
  [1], 
  [1],
  [1], 
  [1], 
  [1], 
  [1],
  [1], 
  [1], 
  [1], 
  [1],
  [1], 
  [1], 
  [1], 
  [1], 
])

debug_info_feedforward = False
debug_info_backpropogate = False

def FindMaxNumpyArray(arr, axis):
    largest = arr[0][axis]
    for i in range(len(arr)):
        if arr[i][axis]>largest:
            largest = arr[i][axis]

    return largest

def NormalizeArray(arr, MaxElem, axis):
    for i in range(len(arr)):
        arr[i][axis] = arr[i][axis] / MaxElem  
    
    return arr

class NeuralNetwork:
    def __init__(self, dim):
        # Hyperparametrs
        #self.learn_rate = 0.35
        self.learn_rate = 0.01
        # Weights
        self.w1 = np.random.randn(2, 1)
        print("w1", self.w1)
        #self.w2 = np.random.randn(2, 1) 
        #self.w3 = np.random.randn(2, 1)

        # Biases
        self.b1 = np.random.normal()
        #self.b2 = np.random.normal()
        #self.b3 = np.random.normal()

    def feedforward(self, x):
        #for xElem in range(len(x)):
            #print("---------------------")
            #print("height: ", x[xElem][0])
            #print("weight: ", x[xElem][1])
            #print("w1: ", self.w1)
            #print("w2: ", self.w2)
            #self.hidden_sum = self.w1[xElem][0] * x[xElem][0] + self.w1[xElem][1] * x[xElem][1] + self.b1
            #print("self.w1", self.w1)
            #print("w1 (1): ", self.w1[0])
            #print("w1 (2): ", self.w1[1])

            #print("x", x[xElem])
        #maxElemHeights = FindMaxNumpyArray(x, 0)
        #maxElemWeights = FindMaxNumpyArray(x, 1)
        #x = NormalizeArray(x, maxElemHeights, 0) 
        #x = NormalizeArray(x, maxElemWeights, 1)        
        if debug_info_feedforward:
            #print("---------------------------------------------------")
            #print("-                 ITERATION:",xElem,"                  -")
            print("---------------------------------------------------")
            print("-                 INPUT LAYER                     -")
            print("---------------------------------------------------")
            print("x: ", x)
            print("self.w1", self.w1)
            #print("self.w1.T", self.w1.T)
            #print("Max Heights: ", maxElemHeights)
            #print("Max Weights: ", maxElemWeights)

            #print("x[xElem]", x[xElem])
        


        self.input_sum = np.sum(np.dot(self.w1.T, x.T))  + self.b1
            
        if debug_info_feedforward:                
            print("input_sum: ", self.input_sum)
            
        self.input_sigmoid = sigmoid(self.input_sum)

        #if debug_info_feedforward:            
        #    print("input: ", self.input_sigmoid)
        #    print("---------------------------------------------------")
        #    print("-                 HIDDEN LAYER                    -")
        #    print("---------------------------------------------------")
            #print("x: ", x[xElem])

        #self.hidden_sum = np.sum(np.dot(self.w2, self.input_sigmoid))  + self.b2
            
        #if debug_info_feedforward:             
        #    print("hidden_sum: ", self.hidden_sum)
            
        #self.hidden_sigmoid = sigmoid(self.hidden_sum)
            
        #if debug_info_feedforward: 
        #    print("hidden: ", self.hidden_sigmoid)
        #    print("---------------------------------------------------")
        #    print("-                 OUTPUT LAYER                    -")
        #    print("---------------------------------------------------")
            #print("x: ", x[xElem])
            
        #self.output_sum = np.sum(np.dot(self.w3, self.hidden_sigmoid))  + self.b3
        #if debug_info_feedforward:             
        #    print("output_sum: ", self.output_sum)
            
        #self.output_sigmoid = sigmoid(self.output_sum)
        #if debug_info_feedforward: 
        #    print("output: ", self.output_sigmoid)
        #self.hidden_sum = np.dot() + self.b2 
        #self.hidden = sigmoid(self.hidden_sum)
            #print("hidden sum: ", self.hidden_sum)
            #print("hidden: ", self.hidden)
            #print("self.w2: ", self.w2)
        #self.output_sum = np.dot(self.w2.T, self.hidden) + self.b3
        #self.output = sigmoid(self.output_sum)
            #print("output sum: ", self.output_sum)
            #print("output : ", self.output)
        #print("---------------------")
        return self.input_sigmoid
    
    def predict(self, x):
        #print("height: ", x[0])
        #print("weight: ", x[1])
        #self.hidden_sum = self.w1 * x[0] + self.w1[1] * x[1] + self.b1
        #self.hidden_sum = np.dot(self.w1.T, x) 
        #self.hidden = sigmoid(self.hidden_sum)
        #print("hidden sum: ", self.hidden_sum)
        #print("hidden: ", self.hidden)
        #self.output_sum = np.dot(self.w2, self.hidden)
        #self.output = sigmoid(self.output_sum)
        #print("w1 predict: ", self.w1)
        #print("w2 predict: ", self.w2)
        #print("w1[0] weight: ", self.w1[0])
        #print("w1[1] weight: ", self.w1[1])
        #print("bias: ", self.b1)
        #print("manual calc: ", self.w1[0] * x[0] + self.w1[1] * x[1] + self.b1)
        self.input_sum = np.sum(np.dot(self.w1.T, x) + self.b1)
        print("input_sum: ", self.input_sum)
        self.input_sigmoid = sigmoid(self.input_sum)
        print("input_sigmoid: ", self.input_sigmoid)
        self.hidden_sum = np.sum(np.dot(self.w2, self.input_sigmoid) + self.b2)
        print("hidden_sum: ", self.hidden_sum)
        print("hidden_softmax: ", softmax(self.input_sum))
        self.hidden_sigmoid = sigmoid(self.hidden_sum)
        print("hidden_sigmoid: ", self.hidden_sigmoid)
        self.output_sum = np.dot(self.w3.T, self.hidden_sigmoid + self.b3)
        print("output_sum: ", self.output_sum)
        return self.output_sum

    def predict_sigmoid(self, x):
        self.input_sum = np.sum(np.dot(self.w1.T, x) + self.b1)
        self.input_sigmoid = sigmoid(self.input_sum)
        return self.input_sigmoid

    def backpropogation(self, x, y, error, index):
         

        #derivative_loss_binarycrossentropy = -(y[index]/self.output) + ((1-y[index])/(1-self.output))
        #print("derivative_loss_binarycrossentropy: ", derivative_loss_binarycrossentropy)
        #output = derivative_loss_binarycrossentropy 
        #print("index: ", index)
        #print("error backpropogation: ", error)
        #print("x", x)
        #print("x.T", x.T)
        #print("deriv_sigmoid(self.output_sum)", deriv_sigmoid(self.output_sum))
        #print("self.output_sum", self.output_sum)
        #hiddenLayer1 = np.dot(x.T, error * deriv_sigmoid(self.output_sum))
        #hiddenLayer2 = np.dot(x.T, error * deriv_sigmoid(self.output_sum))
        #hiddenLayer3 = np.dot(x.T, error * deriv_sigmoid(self.output_sum))
        #print("hiddenLayer1: ", hiddenLayer1)
        #print("hiddenLayer2: ", hiddenLayer2)
        #print("hiddenLayer3: ", hiddenLayer3)
        #print("len(x)", len(x))
        #hiddenLayer = np.dot(x,(self.output-y).T)
        #print("hiddenLayer", hiddenLayer)        
        #db2 = np.sum(self.output-y)
        #print("db2", db2)
        #print("hiddenLayer: ", hiddenLayer)
        #print("x: ", x)
        #print("self.hidden_sum: ", error * deriv_sigmoid(self.hidden_sum))
        #inputLayer = np.dot(x.T, error * deriv_sigmoid(self.hidden_sum))
        #print("self.hidden: ", self.hidden)
        #print("y: ", y)
        #print("x: ", x)
        #inputLayer = np.dot(x,(self.hidden-y[index]).T) 
        #db1 = np.sum(self.hidden-y)
        #print("inputLayer: ", inputLayer)
        #self.b1
        #self.w2 -= self.learn_rate * hiddenLayer
        #self.b2 -= self.learn_rate * db2
        #print("inputLayer: ", inputLayer)
        #print("self.learn_rate: ", self.learn_rate)
        #print("self.w1: ", self.w1)
        #self.w1 -= self.learn_rate * inputLayer
        #self.b1 -= self.learn_rate * db1 
        
        #self.output_sum = np.dot(self.w1, error * deriv_sigmoid(self.output_sum)) 
        #print("self.output_sum", self.output_sum)
        
        #print("self.w1: ", self.w1)
        #print("self.learn_rate: ", self.learn_rate)
        #print("derivative_loss: ", derivative_loss)
        #print("y[index]: ", y[index])
        #print("self.output_sigmoid: ", self.output_sigmoid)
        derivative_loss = mse_derivative(y, self.input_sigmoid)
        #print("derivative_loss: ", derivative_loss)
        #derivative_output = self.output_sigmoid * deriv_sigmoid(self.output_sum) 
        #derivative_output_weight = x * deriv_sigmoid(self.output_sum)
        #derivative_bias_output = deriv_sigmoid(self.output_sum)

        #derivative_hidden = self.hidden_sigmoid * deriv_sigmoid(self.hidden_sum)
        #derivative_hidden_weight = x * deriv_sigmoid(self.hidden_sum)
        #derivative_bias_hidden = deriv_sigmoid(self.hidden_sum)

        #derivative_input = x * deriv_sigmoid(self.input_sum)
        #derivative_input_weight = x * deriv_sigmoid(self.input_sum)
        #derivative_bias_input = deriv_sigmoid(self.input_sum)
        """ 
        self.w1 -= self.learn_rate * derivative_input * derivative_input_weight * derivative_loss
        self.b1 -= self.learn_rate * derivative_loss * derivative_bias_output

        self.w2 -= self.learn_rate * derivative_hidden * derivative_hidden_weight * derivative_loss
        self.b2 -= self.learn_rate * derivative_loss * derivative_bias_hidden
        
        self.w3 -= self.learn_rate * derivative_output * derivative_output_weight * derivative_loss
        self.b3 -= self.learn_rate * derivative_loss * derivative_bias_output 
        """

        #
        if y[index][0] == 0:         
            self.w1[0] -= self.learn_rate * derivative_loss[index]
            #print("self.w1[0] ", self.w1[0])

        #print("y ", y[index][0])
        #print("self.w1[0] ", self.w1[0])
        #print("index: ", index)
        #print("self.w1 error", derivative_loss[index]) 
        
        if y[index][0] == 1:  
            self.w1[1] -= self.learn_rate * derivative_loss[index]
            #print("self.w1[1] ", self.w1[1])
        
        #print("self.w1[1]  error", self.w1[0])
        self.b1 -= self.learn_rate * derivative_loss

        #self.w2 -= self.learn_rate * derivative_loss
        #self.b2 -= self.learn_rate * derivative_loss 

        #self.w3 -= self.learn_rate * derivative_loss
        #self.b3 -= self.learn_rate * derivative_loss

        #print("w1 backprop: ", self.w1)
        #print("w2 backprop: ", self.w2)
        #print("w3 backprop: ", self.w3)
        #print("b1 backprop: ", self.b1)
        #print("b2 backprop: ", self.b2)
        #pass
    def update_weights():
        pass
    def train(self, x, y):
        total_error = 0
        epochs = 700

        for epoch in range(epochs):
            for oneElemData in range(len(x)): 
                #print("x: ", x)
                #y_pred = self.feedforward(x)
                y_pred = self.feedforward(x) 
                #all_pred = np.apply_along_axis(self.feedforward, 1, x)
                #print("all pred: ", y_pred)
                #print("pred: ", y_pred)
                #print("x: ", x)
                #print("y[oneElemData]: ", y[oneElemData])
                #print("y: ", y)           
                # Compute error
                #error = np.sum((y - y_pred) * deriv_sigmoid(y_pred))
                #print("y_pred = ", y_pred, " y = ", y[oneElemData])
                
                error = mse_loss(y_pred, y)
                errors.append(error)
                epochsArr.append(epoch)
                
                #error = mse_loss(x, y) 
                #total_error += error
                #print("error = ", error)
                #derivative_loss = mse_derivative(y[oneElemData], y_pred) 
                #error = y_pred - y
                #error = binary_cross_entropy(y_pred, y)
                #print("error: ", error)
                #error_binary_cross_entropy = binary_cross_entropy(y_pred, y)
                #print("----------------------")
                #print("Epoch: ", epoch)
                #print("Loss: ", error)
                #print("Loss binary cross entropy: ", error_binary_cross_entropy)

                #print("----------------------")
                self.backpropogation(y_pred, y, error, oneElemData)
        

def CheckIsScuf(x):
    if x > 0.5:
        print("^ Femboy")
    else:
        print("^ Skuf")

network = NeuralNetwork(len(x_data))
#network.feedforward(x_data)

#network.backpropogation(x_data, y_data)

# 195 and 145 - max number used for normalization
network.train(x_data, y_data)

myData = np.array([195 / 160, 145 / 65])
myDataFat = np.array([195 / 175, 145 / 140])
myDataFatUnreal = np.array([195 / 195, 145 / 145])
#myDataSkin = np.array([195 / 195, 145 / 145])
myDataSkin2 = np.array([195 / 160, 145 / 60])
myDataUsual = np.array([195 / 170, 145 / 70])
myDataUsual2 = np.array([195 / 175, 145 / 75])
myDataSmallFat = np.array([195 / 175, 145 / 80])

print("160 cm, 65 kg: ", network.predict_sigmoid(myData))
CheckIsScuf(network.predict_sigmoid(myData))
print("175 cm, 140 kg: ", network.predict_sigmoid(myDataFat))
CheckIsScuf(network.predict_sigmoid(myDataFat))
print("195 cm, 145 kg: ", network.predict_sigmoid(myDataFatUnreal))
CheckIsScuf(network.predict_sigmoid(myDataFatUnreal))
print("160 cm, 60 kg: ", network.predict_sigmoid(myDataSkin2))
CheckIsScuf(network.predict_sigmoid(myDataSkin2))
print("170 cm, 70 kg: ", network.predict_sigmoid(myDataUsual))
CheckIsScuf(network.predict_sigmoid(myDataUsual))
print("175 cm, 75 kg: ", network.predict_sigmoid(myDataUsual2))
CheckIsScuf(network.predict_sigmoid(myDataUsual2))
print("175 cm, 80 kg: ", network.predict_sigmoid(myDataSmallFat))
CheckIsScuf(network.predict_sigmoid(myDataSmallFat))
#print("100 cm, 50 kg: ", network.predict_sigmoid(myDataSkin2))
#print("array errors: ", errors)
ax.set_xlabel('Epoch')
ax.set_ylabel('Error')
ax.plot(epochsArr, errors)
#ax.set(xlim=(0, 8), xticks=np.arange(1, 8),
#       ylim=(0, 8), yticks=np.arange(1, 8))
plt.show()
