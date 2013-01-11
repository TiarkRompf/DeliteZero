package ppl.delite.framework

import scala.reflect.ClassTag

object DeliteOps {

  def reduce[A](size: Int)(zero: A)(red: (A,A) => A)(f: Int => A): A = {
    var x = zero
    for (i <- 0 until size) {
      x = red(x, f(i))
    }
    x
  }

  def collect[A:ClassTag](size: Int)(f: Int => A): Array[A] = {
    val a = new Array[A](size)
    for (i <- 0 until size) {
      a(i) = f(i)
    }
    a
  }

  def collectIf[A:ClassTag](size: Int)(c: Int => Boolean)(f: Int => A): Array[A] = {
    val a = new Array[A](size)
    var j = 0
    for (i <- 0 until size) {
      if (c(i)) {
        a(j) = f(i)
        j += 1
      }
    }
    a.take(j)
  }

}
