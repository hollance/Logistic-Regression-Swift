import Foundation

public struct Matrix {
  public let rows: Int
  public let columns: Int
  var grid: [Double]
}

extension Matrix {
  public init(rows: Int, columns: Int, value: Double) {
    self.rows = rows
    self.columns = columns
    self.grid = .init(repeating: value, count: rows * columns)
  }

  /* Creates a matrix from an array: [[a, b], [c, d], [e, f]]. */
  public init(_ data: [[Double]]) {
    let m = data.count
    let n = data[0].count
    self.init(rows: m, columns: n, value: 0)

    for (i, row) in data.enumerated() {
      for j in 0..<row.count {
        self[i, j] = row[j]
      }
    }
  }
}

extension Matrix {
  /* Subscript a single element. */
  public subscript(row: Int, column: Int) -> Double {
    get { return grid[(row * columns) + column] }
    set { grid[(row * columns) + column] = newValue }
  }

  /* Subscript for when the matrix is a row or column vector. */
  public subscript(i: Int) -> Double {
    get {
      precondition(rows == 1 || columns == 1, "Not a row or column vector")
      return grid[i]
    }
    set {
      precondition(rows == 1 || columns == 1, "Not a row or column vector")
      grid[i] = newValue
    }
  }

  /* Get or set an entire row. */
  public subscript(row r: Int) -> Matrix {
    get {
      var v = Matrix(rows: 1, columns: columns, value: 0)
      for c in 0..<columns { v[c] = self[r, c] }
      return v
    }
    set(v) {
      precondition(v.rows == 1 && v.columns == columns, "Not a compatible row vector")
      for c in 0..<columns { self[r, c] = v[c] }
    }
  }

  /* Get or set an entire column. */
  public subscript(column c: Int) -> Matrix {
    get {
      var v = Matrix(rows: rows, columns: 1, value: 0)
      for r in 0..<rows { v[r] = self[r, c] }
      return v
    }
    set(v) {
      precondition(v.rows == rows && v.columns == 1, "Not a compatible column vector")
      for r in 0..<rows { self[r, c] = v[r] }
    }
  }

  /* Useful for when the matrix is 1x1. */
  public var scalar: Double {
    return grid[0]
  }
}

extension Matrix {
  public func transpose() -> Matrix {
    var m = Matrix(rows: columns, columns: rows, value: 0)
    for r in 0..<rows {
      for c in 0..<columns {
        m[c, r] = self[r, c]
      }
    }
    return m
  }
}

postfix operator ′
public postfix func ′ (m: Matrix) -> Matrix {
  return m.transpose()
}

/* Element-by-element addition. */
public func + (lhs: Matrix, rhs: Matrix) -> Matrix {
  if lhs.columns == rhs.columns {
    if rhs.rows == 1 {  // rhs is row vector
      var m = lhs
      for r in 0..<m.rows {
        for c in 0..<m.columns { m[r, c] += rhs[0, c] }
      }
      return m
    } else if lhs.rows == rhs.rows {  // lhs and rhs are same size
      var m = lhs
      for r in 0..<m.rows {
        for c in 0..<m.columns { m[r, c] += rhs[r, c] }
      }
      return m
    }
  } else if lhs.rows == rhs.rows && rhs.columns == 1 {  // rhs is column vector
    var m = lhs
    for r in 0..<m.rows {
      for c in 0..<m.columns { m[r, c] += rhs[r, 0] }
    }
    return m
  }
  fatalError("Cannot add \(lhs.rows)×\(lhs.columns) matrix and \(rhs.rows)×\(rhs.columns) matrix")
}

public func += (lhs: inout Matrix, rhs: Matrix) {
  lhs = lhs + rhs
}

/* Adds a scalar to each element of the matrix. */
public func + (lhs: Matrix, rhs: Double) -> Matrix {
  var m = lhs
  for r in 0..<m.rows {
    for c in 0..<m.columns {
      m[r, c] += rhs
    }
  }
  return m
}

public func += (lhs: inout Matrix, rhs: Double) {
  lhs = lhs + rhs
}

/* Adds a scalar to each element of the matrix. */
public func + (lhs: Double, rhs: Matrix) -> Matrix {
  return rhs + lhs
}

/* Element-by-element subtraction. */
public func - (lhs: Matrix, rhs: Matrix) -> Matrix {
  if lhs.columns == rhs.columns {
    if rhs.rows == 1 {  // rhs is row vector
      var m = lhs
      for r in 0..<m.rows {
        for c in 0..<m.columns { m[r, c] -= rhs[0, c] }
      }
      return m
    } else if lhs.rows == rhs.rows {  // lhs and rhs are same size
      var m = lhs
      for r in 0..<m.rows {
        for c in 0..<m.columns { m[r, c] -= rhs[r, c] }
      }
      return m
    }
  } else if lhs.rows == rhs.rows && rhs.columns == 1 {  // rhs is column vector
    var m = lhs
    for r in 0..<m.rows {
      for c in 0..<m.columns { m[r, c] -= rhs[r, 0] }
    }
    return m
  }
  fatalError("Cannot subtract \(rhs.rows)×\(rhs.columns) matrix from \(lhs.rows)×\(lhs.columns) matrix")
}

public func -= (lhs: inout Matrix, rhs: Matrix) {
  lhs = lhs - rhs
}

/* Subtracts a scalar from each element of the matrix. */
public func - (lhs: Matrix, rhs: Double) -> Matrix {
  return lhs + (-rhs)
}

public func -= (lhs: inout Matrix, rhs: Double) {
  lhs = lhs - rhs
}

/* Subtracts each element of the matrix from a scalar. */
public func - (lhs: Double, rhs: Matrix) -> Matrix {
  var m = Matrix(rows: rhs.rows, columns: rhs.columns, value: 0)
  for r in 0..<m.rows {
    for c in 0..<m.columns {
      m[r, c] = lhs - rhs[r, c]
    }
  }
  return m
}

/* Negates each element of the matrix. */
prefix public func -(x: Matrix) -> Matrix {
  var m = Matrix(rows: x.rows, columns: x.columns, value: 0)
  for r in 0..<m.rows {
    for c in 0..<m.columns {
      m[r, c] = -x[r, c]
    }
  }
  return m
}

infix operator <*> : MultiplicationPrecedence

/* Multiplies two matrices, or a matrix with a vector. */
public func <*> (lhs: Matrix, rhs: Matrix) -> Matrix {
  precondition(lhs.columns == rhs.rows, "Cannot multiply \(lhs.rows)×\(lhs.columns) matrix and \(rhs.rows)×\(rhs.columns) matrix")
  var m = Matrix(rows: lhs.rows, columns: rhs.columns, value: 0)
  for r in 0..<lhs.rows {
    for k in 0..<rhs.columns {
      for c in 0..<lhs.columns {
        m[r, k] += lhs[r, c] * rhs[c, k]
      }
    }
  }
  return m
}

/* Multiplies each element of the matrix with a scalar. */
public func * (lhs: Matrix, rhs: Double) -> Matrix {
  var m = lhs
  for r in 0..<m.rows {
    for c in 0..<m.columns {
      m[r, c] *= rhs
    }
  }
  return m
}

/* Multiplies each element of the matrix with a scalar. */
public func * (lhs: Double, rhs: Matrix) -> Matrix {
  return rhs * lhs
}

/* Divides each element of the matrix by a scalar. */
public func / (lhs: Matrix, rhs: Double) -> Matrix {
  var m = lhs
  for r in 0..<m.rows {
    for c in 0..<m.columns {
      m[r, c] /= rhs
    }
  }
  return m
}

/* Divides a scalar by each element of the matrix. */
public func / (lhs: Double, rhs: Matrix) -> Matrix {
  var m = Matrix(rows: rhs.rows, columns: rhs.columns, value: 0)
  for r in 0..<m.rows {
    for c in 0..<m.columns {
      m[r, c] = lhs / rhs[r, c]
    }
  }
  return m
}

extension Matrix {
  /* Exponentiates each element of the matrix. */
  public func exp() -> Matrix {
    var m = Matrix(rows: rows, columns: columns, value: 0)
    for r in 0..<rows {
      for c in 0..<columns {
        m[r, c] = Foundation.exp(self[r, c])
      }
    }
    return m
  }

  /* Takes the natural logarithm of each element of the matrix. */
  public func log() -> Matrix {
    var m = Matrix(rows: rows, columns: columns, value: 0)
    for r in 0..<rows {
      for c in 0..<columns {
        m[r, c] = Foundation.log(self[r, c])
      }
    }
    return m
  }

  /* Raised each element of the matrix to power alpha. */
  public func pow(_ alpha: Double) -> Matrix {
    var m = Matrix(rows: rows, columns: columns, value: 0)
    for r in 0..<rows {
      for c in 0..<columns {
        m[r, c] = Foundation.pow(self[r, c], alpha)
      }
    }
    return m
  }
}

extension Matrix {
  /* Adds up all the elements in the matrix. */
  public func sum() -> Double {
    var result = 0.0
    for r in 0..<rows {
      for c in 0..<columns {
        result += self[r, c]
      }
    }
    return result
  }

  /* Returns the maximum value in a column, as well as the column index. */
  public func max(row r: Int) -> (Double, Int) {
    var result = self[r, 0]
    var index = 0
    for c in 1..<columns {
      if self[r, c] > result {
        result = self[r, c]
        index = c
      }
    }
    return (result, index)
  }
}

extension Matrix {
  /* Computes the logistic sigmoid function on every element of the matrix. */
  public func sigmoid() -> Matrix {
    return 1 / (1 + (-self).exp())
  }
}
