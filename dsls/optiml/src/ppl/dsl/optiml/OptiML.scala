package ppl.dsl.optiml

import ppl.delite.framework._
import ppl.delite.framework.datastructures._

import ppl.dsl.optila._

import library.cluster.{OptiMLKmeans}

object application {
  trait ApplicationOps
}

object datastruct {
  object scala
}



trait OptiML extends DeliteApplication with OptiMLKmeans { this: OptiMLApplication =>

  object Rect {
    def apply(x:Int,y:Int,w:Int,h:Int): Rect = ???
  }
  abstract class Rect


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


  case class Vertex[V,E](g: Graph[V,E], val data: vertexData) {
    def edges: Iterable[Edge[V,E]] = ???
  }
  case class Edge[V,E](g: Graph[V,E], inData: edgeData, outData: edgeData, in: Vertex[V,E], out: Vertex[V,E])

  case class vertexData(name: String, x: Int)
  case class edgeData(name: String)


  def UnsupervisedTrainingSet[A](x: Matrix[A]) = x
  type UnsupervisedTrainingSet[A] = Matrix[A]

  implicit class TrainingSetOps[A](x: UnsupervisedTrainingSet[A]) {
    def numSamples: Int = ???
    def numFeatures: Int = ???
  }


  def untilconverged[A](init: Matrix[A], y: Double)(g: Matrix[A] => Matrix[A]): Matrix[A] = ???
  def untilconverged[A](init: A, f: A => A, y: Int, b: Boolean)(g: A => A): A = ???


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