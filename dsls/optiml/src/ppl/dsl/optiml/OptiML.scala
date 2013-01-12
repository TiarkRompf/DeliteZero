package ppl.dsl.optiml

import ppl.delite.framework._
import ppl.delite.framework.datastructures._

import ppl.dsl.optila._

object application {
  trait ApplicationOps
}

object datastruct {
  object scala
}



trait OptiML extends DeliteApplication { this: OptiMLApplication =>

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

}

trait OptiMLExp extends OptiML with OptiLAExp { this: OptiMLApplication with OptiMLExp =>
}


// ex. object GDARunner extends OptiLAApplicationRunner with GDA
trait OptiMLApplicationRunner extends OptiLAApplicationRunner with OptiMLApplication with DeliteApplication with OptiMLExp

// ex. trait GDA extends OptiLAApplication
trait OptiMLApplication extends OptiLAApplication with OptiML {
  var args: Array[String]
  def main(): Unit
}

trait OptiMLInteractive extends OptiMLApplication with DeliteInteractive

trait OptiMLInteractiveRunner extends OptiMLApplicationRunner with DeliteInteractiveRunner
