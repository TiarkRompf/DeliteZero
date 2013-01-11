package ppl.delite.framework.datastructures

import scala.reflect.ClassTag


trait DeliteCollection[T] {

}


trait DeliteArray[T] extends DeliteCollection[T] {

  implicit val ct: ClassTag[T]

  val darray: Array[T]

  def length: Int = darray.length
  def apply(i: Int): T = darray.apply(i)
  def update(i: Int, x: T): Unit = darray.update(i,x)
  def map[B:Manifest](f: T => B) = darray.map(f)
  def zip[B:Manifest,R:Manifest](y: DeliteArray[B])(f: (T,B) => R): DeliteArray[R] = DeliteArray((darray.zip(y.darray)) map (p=>f(p._1,p._2)))
  def reduce(f: (T,T) => T, zero: T): T = darray.foldLeft(zero)(f)
  def mkString(del: String): String = darray.mkString(del)
  def union(rhs: DeliteArray[T]) = DeliteArray(darray.union(rhs.darray).toArray)
  def intersect(rhs: DeliteArray[T]) = DeliteArray(darray.intersect(rhs.darray))
  def take(n: Int) = DeliteArray(darray.take(n))
  def sort(implicit o: Ordering[T]) = DeliteArray(darray.sorted)
  def toSeq = darray.toSeq
}


object DeliteArray {
  def apply[T:ClassTag](da: Array[T]) = new DeliteArray[T] { val darray = da; val ct = implicitly[ClassTag[T]] }
  def apply[T:ClassTag](length: Int) = new DeliteArray[T] { val darray = new Array[T](length); val ct = implicitly[ClassTag[T]] }
  //def apply[T:Manifest](length: Rep[Int])(f: Rep[Int] => Rep[T]) = darray_fromFunction(length, f)
}


