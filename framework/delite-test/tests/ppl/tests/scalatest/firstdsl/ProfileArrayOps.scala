package ppl.tests.scalatest.firstdsl

import ppl.delite.framework.DeliteOps

// a simple way of enumerating choices in our syntax
class Reporter
object average extends Reporter
object median extends Reporter

class ProfileArray(val _numMeasurements: Int) {
  val _data = new Array[Double](_numMeasurements)

  // add report and length methods to Rep[ProfileArray]
  def report(y: Reporter): Double = {
    if (y == median) {
      val size = length
      val d = new Array[Double](size)
      System.arraycopy(_data, 0, d, 0, size)
      scala.util.Sorting.quickSort(d)
      d(Math.ceil(size/2).asInstanceOf[Int])
    } else { //if (y == average) {
      val sum = DeliteOps.reduce(length)(0.0)(_+_)(_data(_))
      sum / length
    }
  }
  def length: Int = _numMeasurements
}