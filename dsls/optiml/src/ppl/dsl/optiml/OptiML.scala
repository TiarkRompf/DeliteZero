package ppl.dsl.optiml

import ppl.delite.framework._
import ppl.delite.framework.datastructures._

import ppl.dsl.optila._

import library.cluster._
import library.regression._

object application {
  trait ApplicationOps
}

object datastruct {
  object scala
}



trait OptiML extends DeliteApplication with OptiMLKmeans with OptiMLLinReg { this: OptiMLApplication =>

  object Rect {
    def apply(x:Int,y:Int,w:Int,h:Int): Rect = ???
  }
  abstract class Rect


  object Image {
    def apply[A](x:Int,y:Int): Matrix[A] = ???
    def apply[A](x:Matrix[A]): Matrix[A] = x
  }
  type Image[A] = Matrix[A]

  object Graph {
    def apply[V,E](): Graph[V,E] = ???
  }
  abstract class Graph[V,E] {
    def addVertex(v: Vertex[V,E])
    def addEdge(ab: Edge[V,E], a: Vertex[V,E], b: Vertex[V,E])
    def freeze()
    def vertices: Iterable[Vertex[V,E]]
    def edges: Iterable[Edge[V,E]]
  }


  case class Vertex[V,E](g: Graph[V,E], val data: V) {
    def edges: Iterable[Edge[V,E]] = ???
    def addTask(x: E): Unit = ???
    def addTask(v: Vertex[V,E]): Unit = ??? //??
  }
  case class Edge[V,E](g: Graph[V,E], inData: E, outData: E, inV: Vertex[V,E], outV: Vertex[V,E]) {
    def in(v: Vertex[V,E]): E = ???
    def out(v: Vertex[V,E]): E = ???
    def target(v: Vertex[V,E]): E = ???
  }

  case class vertexData(name: String, x: Int)
  case class edgeData(name: String)



  def DenoiseVertexData(_id : Int, _belief : DenseVector[Double], _potential : DenseVector[Double]): DenoiseVertexData = ???
  def DenoiseEdgeData(_msg : DenseVector[Double], _oldMsg : DenseVector[Double]): DenoiseEdgeData = ???

  class DenoiseVertexData (
    val potential: Vector[Double],
    var belief: Vector[Double],
    val id: Int
  ) {
    def setBelief(b: Vector[Double]) = belief = b
  }

  class DenoiseEdgeData (
    var message: Vector[Double],
    var oldMessage: Vector[Double]
  ) {
    def setMessage(m: Vector[Double]) = message = m
    def setOldMessage(oM: Vector[Double]) = oldMessage = oM
    def Clone : DenoiseEdgeData  = new DenoiseEdgeData(message = this.message, oldMessage = this.oldMessage)
  }




  case class SupervisedTrainingSet[A,B](data: Matrix[A], labels: Vector[B]) extends Matrix[A] {
    def numSamples: Int = ???
    def numFeatures: Int = ???    
  }

  def UnsupervisedTrainingSet[A](x: Matrix[A]) = x
  type UnsupervisedTrainingSet[A] = Matrix[A]

  implicit class TrainingSetOps[A](x: UnsupervisedTrainingSet[A]) {
    def numSamples: Int = ???
    def numFeatures: Int = ???
  }

  object MLOutputWriter {
    def writeImgPgm(img: Matrix[Double], s: String): Unit = ???
  }

  def readTokenMatrix(file: String): (Matrix[Double], Vector[Double]) = ???

  def untilconverged[A](init: Vector[A], y: Int, clone_prev_val: Boolean)(g: Vector[A] => Vector[A])(d:(A,A)=>A): Vector[A] = ???
  def untilconverged[A](init: Matrix[A], y: Double)(g: Matrix[A] => Matrix[A]): Matrix[A] = ???
  def untilconverged[A](init: A, f: A => A, y: Int, b: Boolean)(g: A => A): A = ???
  def untilconverged[V,E](g: Graph[V,E])(f: Vertex[V,E] => Unit): Graph[V,E] = ???

}

trait OptiMLExp extends OptiML with OptiLAExp { this: OptiMLApplication with OptiMLExp =>
}


// ex. object GDARunner extends OptiLAApplicationRunner with GDA
trait OptiMLApplicationRunner extends OptiLAApplicationRunner with OptiMLApplication with DeliteApplication with OptiMLExp

// ex. trait GDA extends OptiLAApplication
trait OptiMLApplication extends OptiLAApplication with OptiML with OptiMLLift {
  var args: Array[String]
  def main(): Unit
}

trait OptiMLInteractive extends OptiMLApplication with DeliteInteractive

trait OptiMLInteractiveRunner extends OptiMLApplicationRunner with DeliteInteractiveRunner

trait OptiMLApplicationRunnerBase 

trait OptiMLNoCSE

trait OptiMLLift