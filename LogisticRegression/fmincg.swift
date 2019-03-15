import Foundation

/**
  Minimizes a continuous differentiable multivariate function. 

  Usage:

      let (X, fX, i) = fmincg(f, X, length)

  The starting point is given by `X` (a row vector).

  The function `f` must return a function value and a vector of partial 
  derivatives. It has the following signature:

      Matrix -> (value: Double, gradients: Matrix)

  The `length` gives the length of the run: if it is positive, it gives the
  maximum number of line searches ("iterations"), if negative its absolute
  gives the maximum allowed number of function evaluations ("epochs").
  
  Note: In the original version of fmincg, you could (optionally) give `length`
  a second component, which will indicate the reduction in function value to be 
  expected in the first line-search (defaults to 1.0). In this version, the
  reduction is always 1.0.

  The function returns the found solution `X`, a vector of function values `fX`
  indicating the progress made, and `i` the number of iterations (line searches
  or function evaluations, depending on the sign of `length`) used.

  The Polack-Ribiere flavour of conjugate gradients is used to compute search
  directions, and a line search using quadratic and cubic polynomial approximations 
  and the Wolfe-Powell stopping criteria is used together with the slope ratio 
  method for guessing initial step sizes. Additionally a bunch of checks are made
  to make sure that exploration is taking place and that extrapolation will not
  be unboundedly large.

  The function returns when either its length is up, or if no further progress
  can be made (i.e., we are at a minimum, or so close that due to numerical 
  problems, we cannot get any closer).
  
  If the function terminates within a few iterations, it could be an indication 
  that the function value and derivatives are not consistent (i.e., there may be
  a bug in the implementation of your `f` function).
  
  See also: Non-linear Conjugate Gradient

  ---

  License from the original Octave/MATLAB implementation:
  
  Copyright (C) 2001 and 2002 by Carl Edward Rasmussen. Date 2002-02-13

  (C) Copyright 1999, 2000 & 2001, Carl Edward Rasmussen

  Permission is granted for anyone to copy, use, or modify these
  programs and accompanying documents for purposes of research or
  education, provided this copyright notice is retained, and note is
  made of any changes that have been made.

  These programs and documents are distributed without any warranty,
  express or implied.  As the programs were written for research
  purposes only, they have not been tested to the degree that would be
  advisable in any important application.  All use of these programs is
  entirely at the user's own risk.

  [ml-class] Changes Made:
  1) Function name and argument specifications
  2) Output display
  
  MIH 31-03-2016: Ported to Swift and cleaned up the code a little.
*/
public func fmincg(data X: Matrix,
                   length: Int = 100,
                   f: (Matrix) -> (value: Double, gradients: Matrix))
                   -> (X: Matrix, fX: [Double], iterations: Int) {
  var X = X                            // make local copy so we can modify the variable

  let RHO = 0.01                              // a bunch of constants for line searches
  let SIG = 0.5         // RHO and SIG are the constants in the Wolfe-Powell conditions
  let INT = 0.1      // don't reevaluate within 0.1 of the limit of the current bracket
  let EXT = 3.0                      // extrapolate maximum 3 times the current bracket
  let RATIO = 100.0                                      // maximum allowed slope ratio
  let MAX = 20                           // max 20 function evaluations per line search

  // The reduction in function value to be expected in the first line-search.
  // The original version did the following, but we simply fix it to 1.0:
  // if max(size(length)) == 2 { red=length(2); length=length(1) } else { red=1 }
  let red = 1.0

  /*
    Key to the variable names being used:
    
      i     counts iterations or function evaluations ("epochs")
      s     search_direction (a vector)

      f1    cost (a scalar), also f0, f2, f3
      df1   gradient (a vector), also df0, df2, df3

      d1    slope (a scalar), also d2, d3
      z1    point (a scalar), also z2, z3
      
      M     counter for maximum function evaluations per line search
  */

  let countEpochs = (length < 0) ? 1 : 0                        // count function calls?
  let countIterations = (length > 0) ? 1 : 0                     // or count iterations?

  var i = 0                                               // zero the run length counter
  var lineSearchFailed = false                     // no previous line search has failed
  var fX = [Double]()

  var (f1, df1) = f(X)                                // get function value and gradient
  i += countEpochs

  var s = -df1                                           // search direction is steepest
  var d1 = (-s′ <*> s).scalar                                       // this is the slope
  var z1 = red/(1 - d1)                                   // initial step is red/(|s|+1)

  while i < abs(length) {                                          // while not finished
    i += countIterations

    let X0 = X                                          // make a copy of current values
    let f0 = f1
    let df0 = df1

    X = X + z1 * s                                                  // begin line search
    var (f2, df2) = f(X)
    i += countEpochs

    var d2 = (df2′ <*> s).scalar

    var f3 = f1                                   // initialize point 3 equal to point 1
    var d3 = d1
    var z3 = -z1

    var M = (length > 0) ? MAX : min(MAX, -length - i)

    var success = false                                         // initialize quantities
    var limit = -1.0

    while true {
      while ((f2 > f1+z1*RHO*d1) || (d2 > -SIG*d1)) && (M > 0) {
        limit = z1                                                // tighten the bracket

        var z2 = 0.0
        if f2 > f1 {
          z2 = z3 - (0.5*d3*z3*z3)/(d3*z3+f2-f3)                        // quadratic fit
        } else {
          let A = 6*(f2-f3)/z3+3*(d2+d3)                                    // cubic fit
          let B = 3*(f3-f2)-z3*(d3+2*d2)
          z2 = (sqrt(B*B-A*d2*z3*z3)-B)/A              // numerical error possible - ok!
        }

        if z2.isNaN || z2.isInfinite {
          z2 = z3/2                         // if we had a numerical problem then bisect
        }

        z2 = max(min(z2, INT*z3),(1-INT)*z3)         // don't accept too close to limits
        z1 = z1 + z2                                                  // update the step
        X = X + z2*s
        (f2, df2) = f(X)
        i += countEpochs
        M -= 1
        d2 = (df2′ <*> s).scalar
        z3 = z3-z2                           // z3 is now relative to the location of z2
      }

      if f2 > f1+z1*RHO*d1 || d2 > -SIG*d1 {
        break                                                       // this is a failure
      } else if d2 > SIG*d1 {
        success = true                                                        // success
        break
      } else if M == 0 {
        break                                                                 // failure
      }

      let A = 6*(f2-f3)/z3+3*(d2+d3)                         // make cubic extrapolation
      let B = 3*(f3-f2)-z3*(d3+2*d2)
      var z2 = -d2*z3*z3/(B+sqrt(B*B-A*d2*z3*z3))           // num. error possible - ok!

      if z2.isNaN || z2.isInfinite || z2 < 0 {                // num prob or wrong sign?
        if limit < -0.5 {                                   // if we have no upper limit
          z2 = z1 * (EXT-1)                       // then extrapolate the maximum amount
        } else {
          z2 = (limit-z1)/2                                          // otherwise bisect
        }
      } else if (limit > -0.5) && (z2+z1 > limit) {         // extrapolation beyond max?
        z2 = (limit-z1)/2                                                      // bisect
      } else if (limit < -0.5) && (z2+z1 > z1*EXT) {       // extrapolation beyond limit
        z2 = z1*(EXT-1.0)                                  // set to extrapolation limit
      } else if z2 < -z3*INT {
        z2 = -z3*INT
      } else if (limit > -0.5) && (z2 < (limit-z1)*(1.0-INT)) {   // too close to limit?
        z2 = (limit-z1)*(1.0-INT)
      }

      f3 = f2                                            // set point 3 equal to point 2
      d3 = d2
      z3 = -z2

      z1 = z1 + z2                                          //  update current estimates
      X = X + z2*s
      (f2, df2) = f(X)
      i += countEpochs
      M -= 1
      d2 = (df2′ <*> s).scalar
    }

    if success {                                             // if line search succeeded
      f1 = f2
      fX.append(f1)

      //print(String(format: "iteration %4i, loss: %4.6e", i, f1))      // show progress

                                                             // Polack-Ribiere direction
      s = (df2′ <*> df2 - df1′ <*> df2).scalar / (df1′ <*> df1).scalar * s - df2
      swap(&df1, &df2)                                               // swap derivatives

      d2 = (df1′ <*> s).scalar
      if d2 > 0 {                                          // new slope must be negative
        s = -df1                                     // otherwise use steepest direction
        d2 = (-s′ <*> s).scalar
      }

      z1 = z1 * min(RATIO, d1/(d2 - .leastNormalMagnitude)) // slope ratio but max RATIO
      d1 = d2
      lineSearchFailed = false                          // this line search did not fail
    } else {
      X = X0                             // restore point from before failed line search
      f1 = f0
      df1 = df0
      if lineSearchFailed || i > abs(length) {      // line search failed twice in a row
        break                                    // or we ran out of time, so we give up
      }
      swap(&df1, &df2)                                               // swap derivatives

      s = -df1                                                           // try steepest
      d1 = (-s′ <*> s).scalar
      z1 = 1/(1-d1)
      lineSearchFailed = true                                 // this line search failed
    }
  }

  return (X, fX, i)
}
