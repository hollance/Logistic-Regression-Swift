import Foundation

/* Computes loss and gradient for logistic regression. */
func logisticRegression(data X: Matrix, targets y: Matrix, weights W: Matrix)
                        -> (loss: Double, gradient: Matrix) {
  // Number of training examples, used to take the mean over the loss.
  let m = Double(X.rows)

  // The logistic regression formula.
  let g = (X <*> W).sigmoid()

  // The loss for a particular choice of the weights.
  let loss = (g.log()′ <*> y + (1 - g).log()′ <*> (1 - y)) / -m

  // The partial derivatives of the loss w.r.t. each weight.
  let gradient = X′ <*> (g - y) / m

  return (loss.scalar, gradient)
}

/* Compute loss and gradient for logistic regression with regularization. */
func logisticRegressionRegularized(data X: Matrix,
                                   targets y: Matrix,
                                   weights W: Matrix,
                                   lambda: Double)
                                   -> (loss: Double, gradient: Matrix) {
  // First run the regular logistic regression.
  var (loss, gradient) = logisticRegression(data: X, targets: y, weights: W)

  // Don't regularize W[0], which is the weight for the bias term.
  var temp = W
  temp[0] = 0

  // Add the L2-norm of the weights to the loss.
  let m = Double(X.rows)
  loss += (lambda / (2*m)) * temp.pow(2).sum()

  // Also add the derivative of the L2-norm to the gradient.
  gradient += (lambda / m) * temp

  return (loss, gradient)
}

/*
  Trains multiple logistic regression classifiers. Returns a new matrix where
  the i-th column holds the weights of the classifier for label i.
*/
func trainOneVsAll(data X: Matrix,
                   targets y: [Int],
                   numLabels: Int,
                   lambda: Double,
                   iterations: Int) -> Matrix {
  var allWeights = Matrix(rows: X.columns, columns: numLabels, value: 0)

  // Train a classifier for each class versus all the other classes.
  for label in 0..<numLabels {
    print("   class \(label)", terminator: "")

    // Filter out all the results from y that do not belong to this class.
    var thisClass = Matrix(rows: y.count, columns: 1, value: 0)
    for r in 0..<y.count {
      thisClass[r] = (y[r] == label) ? 1 : 0
    }

    // As a sanity check, compute the loss for untrained weights.
    // This should be equal to log(2) or 0.6931.
    let initialWeights = Matrix(rows: X.columns, columns: 1, value: 0)
    let (initialLoss, _) = logisticRegressionRegularized(data: X,
                                                         targets: thisClass,
                                                         weights: initialWeights,
                                                         lambda: lambda)
    print(String(format: "   initial loss: %.4f, ", initialLoss), terminator: "")

    // Find an optimal set of weights using the fmincg optimizer.
    let (W, losses, iterationsPerformed) = fmincg(data: initialWeights, length: iterations) {
      W in logisticRegressionRegularized(data: X, targets: thisClass, weights: W, lambda: lambda)
    }
    print(String(format: "iterations %i, loss: %.4f", iterationsPerformed, losses.last ?? .infinity))

    allWeights[column: label] = W
  }
  return allWeights
}

/*
  Predict the label for a trained one-vs-all classifier.
*/
func predictOneVsAll(data X: Matrix, targets y: [Int], weights W: Matrix) -> Double {
  let prediction = (X <*> W).sigmoid()

  // The prediction matrix has a row for each example. Each column corresponds
  // to one of the possible classes. We return the index of the column with the
  // maximum value as the prediction for the example.
  var correct = 0
  for r in 0..<prediction.rows {
    let (_, label) = prediction.max(row: r)
    if label == y[r] {
      correct += 1
    }
  }

  return Double(correct * 100) / Double(X.rows)
}
