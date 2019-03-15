import Foundation

// Note: This is only a simple example. For better results, you ought to
// normalize the training data, split the data into a train and test set,
// shuffle the data, and so on...

let X = Matrix(irisExamples)
let y = irisLabels

print("Training one-vs-all logistic regression...")
let W = trainOneVsAll(data: X, targets: y, numLabels: irisClassNames.count, lambda: 0.1, iterations: 50)

let accuracy = predictOneVsAll(data: X, targets: y, weights: W)
print("Train accuracy: \(accuracy)")
