import numpy as np

def Sigmoid(x):
    return 1/(1+np.exp(-x))

def der_fun(x):
    return x*(1-x)    
 

def predict(X,wt0,wt1):
    input_layer = X
    Hidden_layer = Sigmoid(np.dot(input_layer,wt0))
    output_layer = Sigmoid(np.dot(Hidden_layer,wt1))

    print output_layer


def main():    
    X = np.array([[0,0,1],
                [0,1,1],
                [1,0,1],
                [1,1,1]])
                    
    y = np.array([[0],
                [1],
                [1],
                [0]])

    np.random.seed(1)

    # randomly initialize our weights
    wt0 = 2*np.random.random((3,4)) - 1
    wt1 = 2*np.random.random((4,1)) - 1

    print wt0
    print wt1

    for j in xrange(60000):

        
        input_layer = X
        Hidden_layer = Sigmoid(np.dot(input_layer,wt0))
        output_layer = Sigmoid(np.dot(Hidden_layer,wt1))

        
        output_error = y - output_layer                     #output layer error

        if (j% 10000) == 0:
            print "Error:" + str(np.mean(np.abs(output_error)))
            
        
        output_delta = output_error*der_fun(output_layer)   #delta of output layer

        
        hidden_error = output_delta.dot(wt1.T)              #Error in hidden layer
        
        
        hidden_delta = hidden_error * der_fun(Hidden_layer) #Hidden layer delta 

        #updating weights
        wt1 += Hidden_layer.T.dot(output_delta)            
        wt0 += input_layer.T.dot(hidden_delta)

    predict(np.array([[1,0,1]]),wt0,wt1)        



if __name__ == '__main__':
    main()        