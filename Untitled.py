import numpy
import theano
import theano.tensor as T
import csv
def runTheano(N, feats, training_steps, loginID):
    rng = numpy.random

                                       # training sample size
                                  # number of input variables
    
# generate a dataset: D = (input_values, target_class)
    D = (rng.randn(N, feats), rng.randint(size=N, low=0, high=2))
    

# Declare Theano symbolic variables
    x = T.dmatrix("x")
    y = T.dvector("y")

# initialize the weight vector w randomly
#
# this and the following bias variable b
# are shared so they keep their values
# between training iterations (updates)
    w = theano.shared(rng.randn(feats), name="w")

# initialize the bias term
    b = theano.shared(0., name="b")

    print("Initial model:")
    print(w.get_value())
    print(b.get_value())

# Construct Theano expression graph
    p_1 = 1 / (1 + T.exp(-T.dot(x, w) - b))   # Probability that target = 1
    prediction = p_1 > 0.5                    # The prediction thresholded
    xent = -y * T.log(p_1) - (1-y) * T.log(1-p_1) # Cross-entropy loss function
    cost = xent.mean() + 0.01 * (w ** 2).sum()# The cost to minimize
    gw, gb = T.grad(cost, [w, b])             # Compute the gradient of the cost
                                          # w.r.t weight vector w and
                                          # bias term b
                                          # (we shall return to this in a
                                          # following section of this tutorial)

# Compile
    train = theano.function(
              inputs=[x,y],
              outputs=[prediction, xent],
              updates=((w, w - 0.1 * gw), (b, b - 0.1 * gb)))
    predict = theano.function(inputs=[x], outputs=prediction)

# Train
    for i in range(training_steps):
        pred, err = train(D[0], D[1])

    print("Final model:")
    print(w.get_value())
    print(b.get_value())
    print("target values for D:")
    print(D[1])
    predd = predict(D[0])
    print("pred values for D")
    print(predd)
    print(predd == D[1])
    with open("log.txt", "a") as logFile:
        logFileWriter = csv.writer(logFile)
        logFileWriter.writerow([loginID, N, feats, training_steps, predd==D[1]])
        logFile.close()



print("Recur Theano Trainer")
loginID = str(input("Identification?: "))
n_ng = int(input("Training Sample Size?: "))
feats_ng = int(input("sigInput Vars?: "))
training_steps_ng = int(input("Training Steps?: "))
training_steps_step = int(input("StepDownINT?: "))
print("Welcome, "+loginID+ ". You will run training with the above variables.")
print("Computing...")
runTheano(n_ng, feats_ng, training_steps_ng, loginID)
    
