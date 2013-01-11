package ppl.dsl.optila

import ppl.delite.framework._
import ppl.delite.framework.datastructures._


trait OptiLAScalaOpsPkg
trait OptiLAUtilities
trait OptiLAScalaOpsPkgExp

trait GenericDefs

object matrix
object vector
object extern
object capabilities
object io


trait OptiLA extends DeliteApplication { this: OptiLAApplication =>

  val Vector: VectorCompanion = ???
  abstract class VectorCompanion {
    def apply[A](size: Int, isRow: Boolean): Vector[A] = ???
    def apply[A](x: A*): Vector[A] = ???
    def zeros(size: Int): DenseVector[Double] = ???
    def ones(size: Int): DenseVector[Double] = ???
    def range(x: Int, y: Int, z: Int = 0): RangeVector = ???
    def rand(size: Int): Vector[Double] = ???

    def flatten[A](x:Vector[Vector[A]]): Vector[A] = ???


/*
    def apply[A](len: Int, isRow: Boolean) = densevector_obj_new(unit(len), unit(isRow)) // needed to resolve ambiguities
    def apply[A](len: Int, isRow: Boolean)(implicit o: Overloaded1) = densevector_obj_new(len, isRow)
    def apply[A](xs: A*) = DenseVector[A](xs.map(e=>unit(e)): _*)
    def apply[A](xs: A*)(implicit o: Overloaded1) = DenseVector[A](xs: _*)
*/
    def dense[A](len: Int, isRow: Boolean): DenseVector[A] = ???
    def sparse[A](len: Int, isRow: Boolean): SparseVector[A] = ???
/*        
    def ones(len: Int) = DenseVector.ones(len)
    def onesf(len: Int) = DenseVector.onesf(len)
    def zeros(len: Int) = DenseVector.zeros(len)
    def zerosf(len: Int) = DenseVector.zerosf(len)
    def rand([Alen: Int) = DenseVector.rand(len)
    def randf(len: Int) = DenseVector.randf(len)
    def range(start: Int, end: Int, stride: Int = unit(1), isRow: Boolean = unit(true)) =
      vector_obj_range(start, end, stride, isRow)
    def uniform(start: Double, step_size: Double, end: Double, isRow: Boolean = unit(true)) =
      DenseVector.uniform(start, step_size, end, isRow)
*/  }
  abstract class Vector[A] {
/*    def length: Int
    def isRow: Boolean
    def apply(x: Int): A
    def apply(x: Vector[Int]): Vector[A]
    def update(x: Int, y: A): Unit
    def t: Vector[A]
    def +(x: Vector[A]): Vector[A]
    def +(x: A): Vector[A]
    def -(x: Vector[A]): Vector[A]
    def -(x: A): Vector[A]
    def *(x: Matrix[A]): Vector[A]
    def *(x: Vector[A]): Vector[A]
    def *(x: A): Vector[A]
    def *:*(x: Vector[A]): A
    def **(x: Vector[A]): Matrix[A]
    def sum: A
    def median: A
    def mean: A
    def contains(x: A): Boolean
    def sort: Vector[A]
    def foreach(f: A => Unit): Unit
    def map[B](f: A => B): Vector[B]
    def filter(f: A => Boolean): Vector[A]
    def count(f: A => Boolean): Int
    def zip[B,C](x: Vector[B])(f: (A,B) => C): Vector[C]
    def mutable: Vector[A]
    def Clone: Vector[A]
    def ToString: String
    def pprint: Unit
*/

    // *** sparse ops
    def finish: SparseVector[A] = ???
    def nnz: Int = ???
    def mapNZ[B](f: A => B): SparseVector[B] = ???

    // *** stream row ops
    def index: Int = ???


    // DeliteCollection
    def dcSize = length
    def dcApply(n: Int): A = apply(n)
    def dcUpdate(n: Int, y: A) = update(n,y)
    def unsafeImmutable = this

    // conversions
    def toBoolean(implicit conv: A => Boolean) =  map(e => conv(e))
    def toDouble(implicit conv: A => Double) =  map(e => conv(e))
    def toFloat(implicit conv: A => Float) = map(e => conv(e))
    def toInt(implicit conv: A => Int) = map(e => conv(e))
    //def toLong(implicit conv: A => Rep[Long]) = map(e => conv(e))
    
    // accessors
    def length: Int  = ???
    def isRow: Boolean  = ???
    def apply(n: Int): A  = ???
    def apply(n: IndexVector): Vector[A]  = ???
    def apply(n: Int,x:Int,y:Int): Vector[A]  = ??? // ???
    def isEmpty = length == unit(0)
    def first = apply(unit(0))
    def last = apply(length - unit(1))
    def indices = (unit(0)::length)
    def drop(count: Int) = slice(count, length)
    def take(count: Int) = slice(unit(0), count)
    def slice(start: Int, end: Int): Vector[A]  = ???//= vector_slice[A,VA](x, start, end)
    def contains(y: A): Boolean  = ???//= vector_contains(x,y)
    def distinct: Vector[A]  = ???//= vector_distinct[A,VA](x)  
    
    // general
    def t: Vector[A]  = ???// TODO: move to type-system
    def mt(): Vector[A] = ???
    def Clone(): Vector[A]  = ???//= vector_clone[A,VA](x) 
    def mutable(): Vector[A]  = ???//= vector_mutable_clone[A,VA](x)
    def pprint(): Unit  = ???//= vector_pprint(x)
    //def replicate(i: Int, j: Int): Rep[MA] = vector_repmat[A,MA](x,i,j)
    def replicate(i: Int, j: Int): DenseMatrix[A]  = ???//= vector_repmat[A](x,i,j)
    def mkString(sep: String = unit("")): String  = ???//= vector_mkstring(x, sep)
    
    // data operations
    // TODO: these should probably be moved to another interface (MutableVector), analogously to MatrixBuildable. 
    def update(n: Int, y: A): this.type = ???
    def update(i: IndexVector, y: A): this.type = ???
    def update(i: IndexVector, y: Vector[A]): this.type = ???
    def :+(y: A): Vector[A] = {
      val out = mutable()
      out += y
      out.unsafeImmutable
      // val out = builder[A].alloc(length+1, isRow)
      // for (i <- 0 until out.length) {
      //   out(i) = apply(i)
      // }
      // out(length) = y
      // out
    }

    def +=(y: A): Unit = insert(length,y)
    def ++(y: Vector[A]): Vector[A] = ???
    def ++=(y: Vector[A]) = insertAll(length,y)
    def copyFrom(pos: Int, y: Vector[A]): Unit = ???
    def insert(pos: Int, y: A): Unit = ???
    def insertAll(pos: Int, y: Vector[A]): Unit = ???
    def remove(pos: Int) = removeAll(pos,unit(1))
    def removeAll(pos: Int, len: Int): Unit = ???
    def trim(): Unit = ???
    def clear(): Unit = ???
    
    // arithmetic operations
    
    //val plusInfo: OpInfo[A,VPLUSR,Vector[A]]    
    //def +(y: Vector[A])(implicit a: Arith[A]) = vector_plus[A,VPLUSR](x,y)(manifest[A], implicitly[Arith[A]], plusInfo.mR, plusInfo.b)
    def +(y: Vector[A])(implicit a: Arith[A]): Vector[A]  = ???//= vector_plus[A,VA](x,y)
    //def +[B](y: Vector[B])(implicit a: Arith[A], c: B => A) = vector_plus_withconvert[B,A,VA](y,x)
    //def +(y: Vector[A])(implicit a: Arith[A]): Vector[A]  = ???//= vector_plus[A,VA](x,y) // needed for Arith        
    def +(y: A)(implicit a: Arith[A], o: Overloaded1): Vector[A]  = ???//= vector_plus_scalar[A,VA](x,y) 
    def +=(y: Vector[A])(implicit a: Arith[A]): Unit  = ???//= { vector_plusequals[A](x,y); elem }
    def +=(y: Vector[A])(implicit a: Arith[A], o: Overloaded1): Unit  = ???//= { vector_plusequals[A](x,y); elem }
    def :+=(y: A)(implicit a: Arith[A], o: Overloaded1)  = ???//= vector_plusequals_scalar[A](x,y) 
    
    def -(y: Vector[A])(implicit a: Arith[A]): Vector[A]  = ???//= vector_minus[A,VA](x,y)
    //def -[B](y: Interface[Vector[B]])(implicit a: Arith[A], c: B => A) = vector_minus_withconvert[B,A,VA](y,x)
    //def -(y: Vector[A])(implicit a: Arith[A])  = ???//= vector_minus[A,VA](x,y)
    def -(y: A)(implicit a: Arith[A], o: Overloaded1): Vector[A]  = ???//= vector_minus_scalar[A,VA](x,y)
    def -=(y: Vector[A])(implicit a: Arith[A]): Unit  = ???//= { vector_minusequals[A](x,y); x }
    //def -=(y: Vector[A])(implicit a: Arith[A])  = ???//= { vector_minusequals[A](x,y); x }    
    def -=(y: A)(implicit a: Arith[A], o: Overloaded1): Unit  = ???//= vector_minusequals_scalar[A](x,y)
    
    // TODO: need to extend Arith to support this using CanXX dispatch
    // Rep[DenseVector[Double]] * Rep[RangeVector] (Rep[DenseVector[Double]] * Interface[Vector[Int]])    
    def *(y: Vector[A])(implicit a: Arith[A]): Vector[A]  = ???//= vector_times[A,VA](x,y)        
    def *(y: Matrix[A])(implicit a: Arith[A]): Vector[A]  = ???//= vector_times[A,VA](x,y)        
    //def *[B](y: Interface[Vector[B]])(implicit a: Arith[A], c: B => A)  = ???//= vector_times_withconvert[B,A,VA](y,x)
    //def *(y: Vector[A])(implicit a: Arith[A])  = ???//= vector_times[A,VA](x,y)
    def *(y: A)(implicit a: Arith[A], o: Overloaded1): Vector[A]  = ???//= vector_times_scalar[A,VA](x,y)
    def *=(y: Vector[A])(implicit a: Arith[A]): Unit  = ???//= vector_timesequals[A](x,y)    
    //def *=(y: Vector[A])(implicit a: Arith[A])  = ???//= vector_timesequals[A](x,y)
    def *=(y: A)(implicit a: Arith[A], o: Overloaded1): Unit  = ???//= vector_timesequals_scalar[A](x,y)    
    //def *(y: Rep[DenseMatrix[A]])(implicit a: Arith[A],o: Overloaded2) = vector_times_matrix[A,VTIMESR](x,y)
    def **(y: Vector[A])(implicit a: Arith[A]): Matrix[A]  = ???//= vector_outer[A](x,y)
    //def **(y: Vector[A])(implicit a: Arith[A]) = vector_outer[A,MA](x,y)
    def *:*(y: Vector[A])(implicit a: Arith[A]): A  = ???//= vector_dot_product(x,y)
    def dot(y: Vector[A])(implicit a: Arith[A]): A  = ???//= vector_dot_product(x,y)

    def /(y: Vector[A])(implicit a: Arith[A]): Vector[A]  = ???//= vector_divide[A,VA](x,y)    
    //def /[B](y: Interface[Vector[B]])(implicit a: Arith[A], c: B => A) = vector_divide_withconvert[B,A,VA](y,x)
    //def /(y: Vector[A])(implicit a: Arith[A])  = ???//= vector_divide[A,VA](x,y)
    def /(y: A)(implicit a: Arith[A], o: Overloaded1): Vector[A]  = ???//= vector_divide_scalar[A,VA](x,y)    
    def /=(y: Vector[A])(implicit a: Arith[A]): Unit  = ???//= vector_divideequals[A](x,y)    
    //def /=(y: Vector[A])(implicit a: Arith[A])  = ???//= vector_divideequals[A](x,y)
    def /=(y: A)(implicit a: Arith[A], o: Overloaded1): Unit  = ???//= vector_divideequals_scalar[A](x,y)
    
    def sum(implicit a: Arith[A]): A  = ???//= vector_sum(x)
    def abs(implicit a: Arith[A]): A  = ???//= vector_abs[A,VA](x)
    def exp(implicit a: Arith[A]): A  = ???//= vector_exp[A,VA](x)
    
    def sort(implicit o: Ordering[A]): Vector[A]
    def min(implicit o: Ordering[A], mx: HasMinMax[A]): A  = ???//= vector_min(x)
    def minIndex(implicit o: Ordering[A], mx: HasMinMax[A]): Int  = ???//= vector_minindex(x)
    def max(implicit o: Ordering[A], mx: HasMinMax[A]): A  = ???//= vector_max(x)
    def maxIndex(implicit o: Ordering[A], mx: HasMinMax[A]): Int  = ???//= vector_maxindex(x)
    def median(implicit o: Ordering[A]): A  = ???//= vector_median(x)
    def :>(y: Vector[A])(implicit o: Ordering[A]): Vector[Boolean]  = ???//= zip(y) { (a,b) => a > b }
    def :<(y: Vector[A])(implicit o: Ordering[A]): Vector[Boolean]  = ???//= zip(y) { (a,b) => a < b }    
    
    // bulk operations
    def map[B](f: A => B): Vector[B]  = ???//= vector_map[A,B,V[B]](x,f)
    def mmap(f: A => A): this.type  = ???//= { vector_mmap(x,f); elem }
    def foreach(block: A => Rep[Unit]): Rep[Unit]  = ???//= vector_foreach(x, block)
    def zip[B,R](y: Vector[B])(f: (A,B) => R): Vector[R]  = ???//= vector_zipwith[A,B,R,V[R]](x,y,f)
    def mzip[B](y: Vector[B])(f: (A,B) => A): Vector[B]  = ???//= { vector_mzipwith(x,y,f); elem }
    def reduce(f: (A,A) => A)(implicit a: Arith[A]): A  = ???//= vector_reduce(x,f)
    def filter(pred: A => Boolean): Vector[A]  = ???//= vector_filter[A,VA](x,pred)
    
    def find(pred: A => Boolean): Vector[Int]  = ???//= vector_find[A,V[Int]](x,pred)    
    def count(pred: A => Boolean): Int  = ???//= vector_count(x, pred)
    def flatMap[B](f: A => Vector[B]): Vector[B]  = ???//= vector_flatmap[A,B,V[B]](x,f)
    def partition(pred: A => Boolean): (Vector[A], Vector[A])   = ???//= vector_partition[A,VA](x,pred)
    def groupBy[K](pred: A => K): DenseVector[Vector[A]]  = ???//= vector_groupby[A,K,VA](x,pred)    
  }
  val DenseVector: DenseVectorCompanion = ???
  abstract class DenseVectorCompanion extends VectorCompanion
  abstract class DenseVector[A] extends Vector[A]

  val SparseVector: SparseVectorCompanion = ???
  abstract class SparseVectorCompanion extends VectorCompanion
  abstract class SparseVector[A] extends Vector[A]


  val IndexVector: IndexVectorCompanion = ???
  abstract class IndexVectorCompanion {
    def apply(x: Int*): IndexVector = ???
  }

  abstract class IndexVector extends Vector[Int]

  abstract class RangeVector extends IndexVector {
    def apply[A](f: Int => A): Vector[A] = ???
  }




  implicit def intVector2FloatVector(x: Vector[Int]): Vector[Float] = ???
  implicit def intVector2DoubleVector(x: Vector[Int]): Vector[Double] = ???
  implicit def floatVector2DoubleVector(x: Vector[Float]): Vector[Double] = ???

  implicit def intMatrix2FloatMatrix(x: Matrix[Int]): Matrix[Float] = ???
  implicit def intMatrix2DoubleMatrix(x: Matrix[Int]): Matrix[Double] = ???
  implicit def floatMatrix2DoubleMatrix(x: Matrix[Float]): Matrix[Double] = ???


  implicit class int2VectorAdd(x: Int) {
    def +(y: Vector[Int])(implicit o: Overloaded1) = y + x
    def +(y: Vector[Float])(implicit o: Overloaded2)  = y + x
    def +(y: Vector[Double])(implicit o: Overloaded3)  = y + x
    def +(y: Matrix[Int])(implicit o: Overloaded1)  = y + x
    def +(y: Matrix[Float])(implicit o: Overloaded2)  = y + x
    def +(y: Matrix[Double])(implicit o: Overloaded3)  = y + x
  }

  implicit class float2VectorAdd(x: Float) {
    def +(y: Vector[Int])(implicit o: Overloaded1)  = y + x
    def +(y: Vector[Float])(implicit o: Overloaded2)  = y + x
    def +(y: Vector[Double])(implicit o: Overloaded3)  = y + x
    def +(y: Matrix[Int])(implicit o: Overloaded1)  = y + x
    def +(y: Matrix[Float])(implicit o: Overloaded2)  = y + x
    def +(y: Matrix[Double])(implicit o: Overloaded3)  = y + x
  }

  implicit class double2VectorAdd(x: Double) {
    def +(y: Vector[Int])(implicit o: Overloaded1)  = y + x
    def +(y: Vector[Float])(implicit o: Overloaded2)  = y + x
    def +(y: Vector[Double])(implicit o: Overloaded3)  = y + x
    def +(y: Matrix[Int])(implicit o: Overloaded1)  = y + x
    def +(y: Matrix[Float])(implicit o: Overloaded2)  = y + x
    def +(y: Matrix[Double])(implicit o: Overloaded3)  = y + x
  }


  implicit class int2IndexVectorOps(x: Int) {
    def ::(y:Int): RangeVector = Vector.range(x,y)
  }

  implicit class tupleRangeVectorOps1(x: (RangeVector,RangeVector)) {
    def apply[A](f: (Int,Int) => A): Matrix[A] = ???
  }

  implicit class tupleRangeVectorOps2(x: (RangeVector,*.type)) {
    def apply[A](f: Int => Vector[A]): Matrix[A] = ???
  }

  abstract class Wildcard
  object * extends Wildcard

  trait Arith[A]

  implicit def intArith: Arith[Int] = ???
  implicit def floatArith: Arith[Float] = ???
  implicit def doubleArith: Arith[Double] = ???

  trait HasMinMax[A]

  implicit def intMinMax: HasMinMax[Int] = ???
  implicit def floatMinMax: HasMinMax[Float] = ???
  implicit def doubleMinMax: HasMinMax[Double] = ???



  val Matrix: MatrixCompanion = ???
  abstract class MatrixCompanion {
    def apply[A](x: Int, y: Int): Matrix[A] = ???
    def apply[A](x: Vector[A]*): Matrix[A] = ???
    def zeros(x: Int, y: Int): Matrix[Double] = ???
    def ones(x: Int, y: Int): Matrix[Double] = ???
    //def range(x: Int, y: Int): Vector[Int] = ???
    def rand(x: Int, y: Int): Matrix[Double] = ???

    def dense[A](numRows: Int, numCols: Int): DenseMatrix[A] = ???
    def sparse[A](numRows: Int, numCols: Int): SparseMatrix[A] = ???


/*
    def apply[A](numRows: Int, numCols: Int) = densematrix_obj_new(numRows, numCols)
    def apply[A](xs: Rep[DenseVector[DenseVector[A]]]): Rep[DenseMatrix[A]] = densematrix_obj_fromvec(xs)
    def apply[A](xs: Rep[DenseVector[DenseVectorView[A]]])(implicit mA: Manifest[A], o: Overloaded1): Matrix[A] = densematrix_obj_fromvec(xs.asInstanceOf[Rep[DenseVector[DenseVector[A]]]])
    def apply[A](xs: Rep[DenseVector[A]]*): Rep[DenseMatrix[A]] = DenseMatrix(DenseVector(xs: _*))

    def dense[A](numRows: Int, numCols: Int) = densematrix_obj_new(numRows, numCols)
    def sparse[A](numRows: Int, numCols: Int) = sparsematrix_obj_new(numRows, numCols)   
    
    def diag[A](w: Int, vals: Vector[A]) = DenseMatrix.diag[A](w,vals)
    def identity(w: Int) = DenseMatrix.identity(w)
    def zeros(numRows: Int, numCols: Int) = DenseMatrix.zeros(numRows,numCols)
    def zerosf(numRows: Int, numCols: Int) = DenseMatrix.zerosf(numRows,numCols)
    def mzerosf(numRows: Int, numCols: Int) = DenseMatrix.mzerosf(numRows,numCols)
    def ones(numRows: Int, numCols: Int) = DenseMatrix.ones(numRows,numCols)
    def onesf(numRows: Int, numCols: Int) = DenseMatrix.onesf(numRows,numCols)
    def rand(numRows: Int, numCols: Int) = DenseMatrix.rand(numRows,numCols)
    def randf(numRows: Int, numCols: Int) = DenseMatrix.randf(numRows,numCols)
    def randn(numRows: Int, numCols: Int) = DenseMatrix.randn(numRows,numCols)
    def randnf(numRows: Int, numCols: Int) = DenseMatrix.randnf(numRows,numCols)
    def mrandnf(numRows: Int, numCols: Int) = DenseMatrix.mrandnf(numRows,numCols)
*/
  }
  abstract class MatrixView[A] extends Matrix[A]
  abstract class Matrix[A] {
    def dcSize: Int = ???
    def dcApply(n: Int): A = ???
    def dcUpdate(n: Int, y: A): Unit = ???
    def numRows: Int = ???
    def numCols: Int = ???
    def inv(implicit conv: A => Double): Matrix[Double] = ???
    def vview(start: Int, stride: Int, length: Int, isRow: Boolean): MatrixView[A] = ???
    
    // conversions
    def toBoolean(implicit conv: A => Boolean)  = ???
    def toDouble(implicit conv: A => Double)  = ???
    def toFloat(implicit conv: A => Float)  = ???
    def toInt(implicit conv: A => Int)  = ???
    //def toLong(implicit conv: A => Rep[Long])  = ???

    // accessors
    def apply(i: Int, j: Int): A  = ???
    def apply(i: Int): Vector[A] = ???
    def apply(i: IndexVector): Matrix[A] = ???
    def apply(i: Wildcard, j: IndexVector): Matrix[A] = ???
    def apply(i: IndexVector, j: Wildcard): Matrix[A] = ???
    def apply(i: IndexVector, j: IndexVector): Matrix[A] = ???
    def size: Int  = ???
    def getRow(row: Int): Vector[A] = ???
    def getCol(col: Int): Vector[A] = ???
    def slice(startRow: Int, endRow: Int, startCol: Int, endCol: Int): Matrix[A] = ???
    def sliceRows(start: Int, end: Int): Matrix[A] = ???

    // general
    def t: Matrix[A]  = ???
    // TODO: implicit won't trigger
    //override def clone 
    def Clone(): Matrix[A]  = ???
    def mutable(): Matrix[A]  = ???
    def pprint(): Rep[Unit]  = ???
    def replicate(i: Int, j: Int): Matrix[A]  = ???

    // data operations
    def :+(y: Vector[A]): Matrix[A]  = ???

    // arithmetic operations
    def +(y: Matrix[A])(implicit a: Arith[A]): Matrix[A]  = ???
    def +(y: A)(implicit a: Arith[A], o: Overloaded1): Matrix[A]  = ???
    //def +[B](y: Matrix[B])(implicit a: Arith[A], c: B => A)  = ???
    def +=(y: Matrix[A])(implicit a: Arith[A]): this.type = ???
    def -(y: Matrix[A])(implicit a: Arith[A]): Matrix[A]  = ???
    def -(y: A)(implicit a: Arith[A], o: Overloaded1): Matrix[A]  = ???
    //def -[B](y: Matrix[B])(implicit a: Arith[A], c: B => A)  = ???
    def *:*(y: Matrix[A])(implicit a: Arith[A]): Matrix[A]  = ???
    def *(y: Matrix[A])(implicit a: Arith[A]): Matrix[A]  = ???
    def *(y: Vector[A])(implicit a: Arith[A], o: Overloaded1): Vector[A]  = ???
    def *(y: A)(implicit a: Arith[A], o: Overloaded2): Matrix[A]  = ???
    //def *:*[B](y: Matrix[B])(implicit a: Arith[A], c: B => A)  = ???
    def /(y: Matrix[A])(implicit a: Arith[A]): Matrix[A]  = ???
    def /(y: A)(implicit a: Arith[A], o: Overloaded1): Matrix[A]  = ???
    //def /[B](y: Matrix[B])(implicit a: Arith[A], c: B => A)  = ???
    //def unary_-(implicit a: Arith[A])  = ???
    def abs(implicit a: Arith[A]): Matrix[A]  = ???
    def exp(implicit a: Arith[A]): Matrix[A]  = ???
    def sum(implicit a: Arith[A]): A = ???
    def sumRow(implicit a: Arith[A]): Vector[A]  = ???
    def sumCol(implicit a: Arith[A]): Vector[A]  = ???
    def sigmoid(implicit conv: A => Double): Matrix[Double] = ???
    def sigmoidf(implicit conv: A => Float): Matrix[Float] = ???

    // ordering operations
    def min(implicit o: Ordering[A], mx: HasMinMax[A])  = ???
    def minRow(implicit o: Ordering[A], mx: HasMinMax[A]): Vector[A]  = ???
    def max(implicit o: Ordering[A], mx: HasMinMax[A])  = ???
    def maxRow(implicit o: Ordering[A], mx: HasMinMax[A]): Vector[A]  = ???
    def :>(y: Matrix[A])(implicit o: Ordering[A])  = ???
    def :<(y: Matrix[A])(implicit o: Ordering[A])  = ???

    // bulk operations
    def map[B](f: A => B): Matrix[B] = ???
    /// TODO: rename to transform?
    def mmap(f: A => A): this.type = ???
    def mapRowsToVector[B](f: Vector[A] => B, isRow: Boolean): Vector[B] = ???
    def foreach(block: A => Unit): Unit = ???
    def foreachRow(block: Vector[A] => Unit): Unit = ???
    def zip[B,R](y: Matrix[B])(f: (A,B) => R): Matrix[R]  = ???
    def filterRows(pred: Vector[A] => Boolean): Matrix[A]  = ???
    def groupRowsBy[K](pred: Vector[A] => K): Vector[Matrix[K]] = ???
    def count(pred: A => Boolean): Int = ???
    def mapRows[B](f: Vector[A] => Vector[B]): Matrix[A]  = ???
    def reduceRows(f: (Vector[A],Vector[A]) => Vector[A]): Vector[A] = ???


    // update ops
    def update(i: Int, j: Int, y: A): this.type = ???
    def update(i: Int, y: Vector[A]): this.type = ???
    def insertRow(pos: Int, y: Vector[A]): this.type = ???
    def insertAllRows(pos: Int, y: Matrix[A]): this.type = ???
    def insertCol(pos: Int, y: Vector[A]): this.type = ???
    def insertAllCols(pos: Int, y: Matrix[A]): this.type = ???
    def removeRows(pos: Int, len: Int): this.type = ???
    def removeCols(pos: Int, len: Int): this.type = ???



    // sparse ops
    def +=(x: Vector[A]): this.type = ???
    def finish: SparseMatrix[A] = ???
    def nnz: Int = ???
  }
  val DenseMatrix: DenseMatrixCompanion = ???
  abstract class DenseMatrixCompanion extends MatrixCompanion
  abstract class DenseMatrix[A] extends Matrix[A]
  val SparseMatrix: SparseMatrixCompanion = ???
  abstract class SparseMatrixCompanion extends MatrixCompanion
  abstract class SparseMatrix[A] extends Matrix[A]



  val Stream: StreamCompanion = ???
  abstract class StreamCompanion extends MatrixCompanion {
    def apply[A](x: Int, y: Int)(f: (Int,Int) => A): Stream[A] = ???
  }
  abstract class Stream[A] extends Matrix[A] {
    def isPure: Boolean = ???
  }

  abstract class StreamRow[A] extends Vector[A] {
  }




  def random[A]: A = ???

  def sum[A](x: Int, y: Int)(f: Int => A): A = ???
  def sumIf[A,IGNORE](x: Int, y: Int)(c: Int => Boolean)(f: Int => A): A = ???

  def aggregate[A](x: Int, y: Int)(f: Int => A): A = ???
  def aggregateIf[A,IGNORE](x: Int, y: Int)(c: Int => Boolean)(f: Int => A): Vector[A] = ???

  def aggregate[A](x: RangeVector, y: RangeVector)(f: (Int,Int) => A): Vector[A] = ???
  def aggregateIf[A](x: RangeVector, y: RangeVector)(c: (Int,Int) => Boolean)(f: (Int,Int) => A): Vector[A] = ???


  def pow(x: Double, y: Int): Double = ???
  def exp(x: Double): Double = ???
  def abs(x: Double): Double = ???


  def mean[A](x: Matrix[A]): A = ???
  def max[A](x: Matrix[A]): A = ???
  def min[A](x: Matrix[A]): A = ???

  def median[A](x: Vector[A]): A = ???
  def mean[A](x: Vector[A]): A = ???
  def max[A](x: Vector[A]): A = ???
  def min[A](x: Vector[A]): A = ???

  def mean[A](x: A*): A = ???

  def dist[A](i: A, j: A): Double = ???

  def nearestNeighborIndex[A](x: Int, m: Matrix[A], any: Boolean = false): Int = ???

  def sample[A,IGNORE](x: Vector[A], y: Int): Vector[A] = ???

  def tic(): Unit = ???
  def toc(): Unit = ???

  def infix_ToString(x:Any): String = ???

  def readMatrix(x: String): Matrix[Double] = ???
  def writeMatrix(x: Matrix[Double], y: String): Unit = ???
}

trait OptiLACompiler extends OptiLA { this: OptiLAApplication with OptiLAExp =>
}


/**
 * These are the corresponding IR nodes for OptiLA.
 */
trait OptiLAExp extends OptiLACompiler { this: OptiLAApplication with OptiLAExp =>
}


// ex. object GDARunner extends OptiLAApplicationRunner with GDA
trait OptiLAApplicationRunner extends OptiLAApplication with DeliteApplication with OptiLAExp

// ex. trait GDA extends OptiLAApplication
trait OptiLAApplication extends OptiLA {
  var args: Array[String]
  def main(): Unit
}

trait OptiLAInteractive extends OptiLAApplication with DeliteInteractive

trait OptiLAInteractiveRunner extends OptiLAApplicationRunner with DeliteInteractiveRunner

object OptiLA {
  def apply[R](b: => R) = ???
}

