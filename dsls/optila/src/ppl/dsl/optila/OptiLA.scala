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

  val Vector: VectorCompanion = new VectorCompanion
  class VectorCompanion {
    def apply[A](size: Int, isRow: Boolean): Vector[A] = ???
    def apply[A](x: A*): Vector[A] = ???

    def flatten[A](x:Vector[Vector[A]]): Vector[A] = ???


/*
    def apply[A](len: Int, isRow: Boolean) = densevector_obj_new(unit(len), unit(isRow)) // needed to resolve ambiguities
    def apply[A](len: Int, isRow: Boolean)(implicit o: Overloaded1) = densevector_obj_new(len, isRow)
    def apply[A](xs: A*) = DenseVector[A](xs.map(e=>unit(e)): _*)
    def apply[A](xs: A*)(implicit o: Overloaded1) = DenseVector[A](xs: _*)
*/
    def dense[A](len: Int, isRow: Boolean): DenseVector[A] = ???
    def sparse[A](len: Int, isRow: Boolean): SparseVector[A] = ???

    def ones(len: Int): Vector[Double] = ???
    def onesf(len: Int): Vector[Float] = ???
    def zeros(len: Int): Vector[Double] = ???
    def zerosf(len: Int): Vector[Float] = ???
    def rand(Alen: Int): Vector[Double] = ???
    def randf(len: Int): Vector[Float] = ???
    def range(start: Int, end: Int, stride: Int = unit(1), isRow: Boolean = unit(true)): RangeVector = ???
    def uniform(start: Double, step_size: Double, end: Double, isRow: Boolean = unit(true)): Vector[Double] = ???
  }
  abstract class Vector[A] {

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
    def abs(implicit a: Arith[A]): Vector[A]  = ???//= vector_abs[A,VA](x)
    def exp(implicit a: Arith[A]): Vector[A]  = ???//= vector_exp[A,VA](x)
    
    def sort(implicit o: Ordering[A]): Vector[A] = ???
    def sortWithIndex(implicit o: Ordering[A]): (Vector[A], Vector[Int]) = ???
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
    
    def find(pred: A => Boolean): IndexVector  = ???//= vector_find[A,V[Int]](x,pred)    
    def count(pred: A => Boolean): Int  = ???//= vector_count(x, pred)
    def flatMap[B](f: A => Vector[B]): Vector[B]  = ???//= vector_flatmap[A,B,V[B]](x,f)
    def partition(pred: A => Boolean): (Vector[A], Vector[A])   = ???//= vector_partition[A,VA](x,pred)
    def groupBy[K](pred: A => K): DenseVector[Vector[A]]  = ???//= vector_groupby[A,K,VA](x,pred)    
  }
  val DenseVector: DenseVectorCompanion = new DenseVectorCompanion
  class DenseVectorCompanion extends VectorCompanion
  type DenseVector[A] = Vector[A]

  val SparseVector: SparseVectorCompanion = new SparseVectorCompanion  
  class SparseVectorCompanion extends VectorCompanion
  type SparseVector[A] = Vector[A]


  val IndexVector: IndexVectorCompanion = new IndexVectorCompanion
  class IndexVectorCompanion {
    def apply(x: Int*): IndexVector = ???
    def apply(x: Int,b: Boolean): IndexVector = ???
  }

  type IndexVector = Vector[Int]
  type IndexVectorDense = IndexVector

  class RangeVector extends IndexVector {
    def apply[A](f: Int => A): Vector[A] = ???
  }

  def EmptyVector[A]: EmptyVector[A] = ???
  type EmptyVector[A] = Vector[A]
  type DenseVectorView[A] = DenseVector[A]





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

  implicit def vectorArith[A:Arith]: Arith[Vector[A]] = ???


  trait HasMinMax[A]

  implicit def intMinMax: HasMinMax[Int] = ???
  implicit def floatMinMax: HasMinMax[Float] = ???
  implicit def doubleMinMax: HasMinMax[Double] = ???



  val Matrix: MatrixCompanion = new MatrixCompanion
  class MatrixCompanion {
    def apply[A](x: Int, y: Int): Matrix[A] = ???
    def apply[A](x: Vector[Vector[A]]): Matrix[A] = ???
    def apply[A](x: Vector[A]*): Matrix[A] = ???

    def dense[A](numRows: Int, numCols: Int): DenseMatrix[A] = ???
    def sparse[A](numRows: Int, numCols: Int): SparseMatrix[A] = ???


/*
    def apply[A](numRows: Int, numCols: Int) = densematrix_obj_new(numRows, numCols)
    def apply[A](xs: Rep[DenseVector[DenseVector[A]]]): Rep[DenseMatrix[A]] = densematrix_obj_fromvec(xs)
    def apply[A](xs: Rep[DenseVector[DenseVectorView[A]]])(implicit mA: Manifest[A], o: Overloaded1): Matrix[A] = densematrix_obj_fromvec(xs.asInstanceOf[Rep[DenseVector[DenseVector[A]]]])
    def apply[A](xs: Rep[DenseVector[A]]*): Rep[DenseMatrix[A]] = DenseMatrix(DenseVector(xs: _*))
*/
    
    def diag[A](w: Int, vals: Vector[A]): Matrix[A] = ???
    def identity(w: Int): Matrix[Double] = ???
    def zeros(numRows: Int, numCols: Int): Matrix[Double] = ???
    def zerosf(numRows: Int, numCols: Int): Matrix[Float] = ???
    def mzerosf(numRows: Int, numCols: Int): Matrix[Float] = ???
    def ones(numRows: Int, numCols: Int): Matrix[Double] = ???
    def onesf(numRows: Int, numCols: Int): Matrix[Float] = ???
    def rand(numRows: Int, numCols: Int): Matrix[Double] = ???
    def randf(numRows: Int, numCols: Int): Matrix[Float] = ???
    def randn(numRows: Int, numCols: Int): Matrix[Double] = ???
    def randnf(numRows: Int, numCols: Int): Matrix[Float] = ???
    def mrandnf(numRows: Int, numCols: Int): Matrix[Float] = ???
  }
  abstract class MatrixView[A] extends Matrix[A]
  abstract class Matrix[A] {
    def unsafeImmutable = this
    def dcSize: Int = ???
    def dcApply(n: Int): A = ???
    def dcUpdate(n: Int, y: A): Unit = ???
    def numRows: Int = ???
    def numCols: Int = ???
    def inv(implicit conv: A => Double): Matrix[Double] = ???
    def vview(start: Int, stride: Int, length: Int, isRow: Boolean): MatrixView[A] = ???
    
    // conversions
    def toBoolean(implicit conv: A => Boolean): Matrix[Boolean]  = ???
    def toDouble(implicit conv: A => Double): Matrix[Double]  = ???
    def toFloat(implicit conv: A => Float): Matrix[Float]  = ???
    def toInt(implicit conv: A => Int): Matrix[Int]  = ???
    //def toLong(implicit conv: A => Rep[Long])  = ???

    // accessors
    def apply(i: Int, j: Int): A  = ???
    def apply(i: Int): Vector[A] = ???
    def apply(i: IndexVector): Matrix[A] = ???
    def apply(i: IndexVector, j: Int): Vector[A] = ???
    def apply(i: Wildcard, j: IndexVector): Matrix[A] = ???
    def apply(i: IndexVector, j: Wildcard): Matrix[A] = ???
    def apply(i: IndexVector, j: IndexVector): Matrix[A] = ???
    def size: Int  = ???
    def getRow(row: Int): Vector[A] = ???
    def getCol(col: Int): Vector[A] = ???
    def slice(startRow: Int, endRow: Int, startCol: Int, endCol: Int): Matrix[A] = ???
    def sliceRows(start: Int, end: Int): Matrix[A] = ???
    def rows: Vector[Vector[A]] = ???
    //def cols: Vector[Vector[A]] = ???

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
    def :>(y: Matrix[A])(implicit o: Ordering[A]): Matrix[Boolean]  = ???
    def :<(y: Matrix[A])(implicit o: Ordering[A]): Matrix[Boolean]  = ???

    // bulk operations
    def map[B](f: A => B): Matrix[B] = ???
    /// TODO: rename to transform?
    def mmap(f: A => A): this.type = ???
    def mapRowsToVector[B](f: Vector[A] => B, isRow: Boolean = true): Vector[B] = ???
    def foreach(block: A => Unit): Unit = ???
    def foreachRow(block: Vector[A] => Unit): Unit = ???
    def zip[B,R](y: Matrix[B])(f: (A,B) => R): Matrix[R]  = ???
    def filterRows(pred: Vector[A] => Boolean): Matrix[A]  = ???
    def groupRowsBy[K](pred: Vector[A] => K): Vector[Matrix[K]] = ???
    def count(pred: A => Boolean): Int = ???
    def mapRows[B](f: Vector[A] => Vector[B]): Matrix[A]  = ???
    def reduceRows(f: (Vector[A],Vector[A]) => Vector[A]): Vector[A] = ???
    def sumRowsWhere(c: Int => Boolean)(implicit a: Arith[A]): Vector[A] = ???


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
    def nzRowIndices: { 
      def apply[A](f: Int => A): Vector[A] 
      def apply(i: Int): Int
    } = ???
  }
  val DenseMatrix: DenseMatrixCompanion = new DenseMatrixCompanion
  class DenseMatrixCompanion extends MatrixCompanion
  type DenseMatrix[A] = Matrix[A]
  val SparseMatrix: SparseMatrixCompanion = new SparseMatrixCompanion
  class SparseMatrixCompanion extends MatrixCompanion
  type SparseMatrix[A] = Matrix[A]


  implicit def boolMtoFloat(x: Matrix[Boolean]): Matrix[Float] = ???// FIXME: for rbm


  val Stream: StreamCompanion = new StreamCompanion
  class StreamCompanion extends MatrixCompanion {
    def apply[A](x: Int, y: Int)(f: (Int,Int) => A): Stream[A] = ???
  }
  class Stream[A] extends Matrix[A] {
    def isPure: Boolean = ???
  }

  class StreamRow[A] extends Vector[A] {
  }


  class Record




  def random[A]: A = ???
  def random(k:Int): Int = ???
  def random(k:IndexVector): Int = ??? // not sure about signature

  def reseed: Unit = ???

  def randomGaussian: Double = ???

  def sum[A](x: Int, y: Int)(f: Int => A): A = ???
  def sumIf[A,IGNORE](x: Int, y: Int)(c: Int => Boolean)(f: Int => A): A = ???

  def aggregate[A](x: Int, y: Int)(f: Int => A): Vector[A] = ???
  def aggregateIf[A,IGNORE](x: Int, y: Int)(c: Int => Boolean)(f: Int => A): Vector[A] = ???

  def aggregate[A](x: RangeVector, y: RangeVector)(f: (Int,Int) => A): Vector[A] = ???
  def aggregateIf[A](x: RangeVector, y: RangeVector)(c: (Int,Int) => Boolean)(f: (Int,Int) => A): Vector[A] = ???


  trait Triangle // IndexVector2
  def utriangle(x: Int, b: Boolean): Triangle = ???

  def aggregate[A](x: Triangle)(f: (Int,Int) => A): Vector[A] = ???


  def pow(x: Double, y: Double): Double = ???
  def exp(x: Double): Double = ???
  def log(x: Double): Double = ???
  def abs(x: Double): Double = ???
  def floor(x: Double): Double = ???
  def ceil(x: Double): Double = ???
  def square(x: Double): Double = ???
  def sqrt(x: Double): Double = ???
  def cos(x: Double): Double = ???
  def sin(x: Double): Double = ???
  def acos(x: Double): Double = ???
  def acin(x: Double): Double = ???


  def mean[A](x: Matrix[A]): A = ???
  def max[A](x: Matrix[A]): A = ???
  def min[A](x: Matrix[A]): A = ???
  def sum[A](x: Matrix[A]): A = ???
  def square[A](x: Matrix[A]): Matrix[A] = ???

  def det[A](x: Matrix[A]): A = ???


  def median[A](x: Vector[A]): A = ???
  def mean[A](x: Vector[A]): A = ???
  def max[A](x: Vector[A]): A = ???
  def min[A](x: Vector[A]): A = ???
  def sum[A](x: Vector[A]): A = ???
  def square[A](x: Vector[A]): Vector[A] = ???

  def mean[A](x: A*): A = ???
  def max[A](x: A*): A = ???
  def min[A](x: A*): A = ???

  def INF = Double.PositiveInfinity  
  def nINF = Double.NegativeInfinity   

  class DistanceMetric
  object ABS extends DistanceMetric
  object EUC extends DistanceMetric
  object SQUARE extends DistanceMetric

  def dist[A](i: A, j: A, spec: DistanceMetric = ABS): Double = ???

  def nearestNeighborIndex[A](x: Int, m: Matrix[A], any: Boolean = false): Int = ???

  def sample[A,IGNORE](x: Vector[A], y: Int): Vector[A] = ???

  def tic(d:Any*): Unit = ???
  def toc(d:Any*): Unit = ???

  def fatal(s: String) = sys.error(s)

  implicit class any2Overload(x:Any) {
    def ToString: String = x.toString
    def AsInstanceOf[T] = x.asInstanceOf[T]
  }

  def t2[A,B](x: (A, B)) = x
  def t3[A,B,C](x: (A, B, C)) = x
  def t4[A,B,C,D](x: (A, B, C,D)) = x
  def make_tuple2[A,B](x: A, y: B) = (x,y)
  implicit def tuple3ArithWitness[A,B,C]: Arith[(A,B,C)] = ???
  implicit class tuple3Arith[A,B,C](x: (A,B,C)) {
    def -(y: (A,B,C)): (A,B,C) = ???
    def /(y: (A,B,C)): (A,B,C) = ???
    def /(y: Int): (A,B,C) = ???
  }


  def readVector(x: String, d: String = "\t"): Vector[Double] = ???
  def readMatrix(x: String, d: String = "\t"): Matrix[Double] = ???

  def readARFF[A](x: String, y: Vector[String] => A): Vector[A] = ??? //sig?

  def writeVector(x: Vector[Double], y: String): Unit = ???
  def writeMatrix(x: Matrix[Double], y: String): Unit = ???

  def readVector[A](x: String, f: String => A, d: String = "\t"): Vector[A] = ???
  def readMatrix[A](x: String, f: String => A, d: String = "\t"): Matrix[A] = ???
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

