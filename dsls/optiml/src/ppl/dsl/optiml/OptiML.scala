package ppl.dsl.optiml

import ppl.delite.framework._
import ppl.delite.framework.datastructures._

import ppl.dsl.optila._

import library.cluster._
import library.regression._

object application {
  trait ApplicationOps
}



trait OptiML extends DeliteApplication with OptiMLKmeans with OptiMLLinReg { this: OptiMLApplication =>

  object Rect {
    def apply(x:Int,y:Int,w:Int,h:Int): Rect = ???
  }
  abstract class Rect {
    val x: Int
    val y: Int
    val width: Int
    val height: Int
  }


  object Image {
    def apply[A](x:Int,y:Int): Matrix[A] = ???
    def apply[A](x:Matrix[A]): Matrix[A] = x
  }
  type Image[A] = Matrix[A]
  object GrayscaleImage {
    def apply(x:Matrix[Double]): Matrix[Int] = ???
  }
  type GrayscaleImage = Image[Int]

  def varToGrayscaleImageOps(x: GrayscaleImage) = new {
    def bitwiseOrDownsample(): GrayscaleImage = ???
  }
  def repGrayscaleImageToGrayscaleImageOps(x: GrayscaleImage) = new {
    def gradients(b: Boolean): (DenseMatrix[Float], DenseMatrix[Float]) = ???
    def windowedFilter(w:Int, h: Int)(f: Matrix[Int] => Int) = ???
  }


  object BiGGDetection {
    def apply(name: Rep[String], score: Rep[Float], roi: Rep[Rect], mask: Rep[GrayscaleImage], index: Rep[Int], x: Rep[Int], y: Rep[Int], tpl: Rep[BinarizedGradientTemplate], crt_tpl: Rep[BinarizedGradientTemplate]): BiGGDetection = ???
  }

  class BiGGDetection (
    val name: String,
    val score: Float,
    val roi: Rect,
    val mask: Image[Int],
    val index: Int,
    val x: Int,
    val y: Int,
    val tpl: BinarizedGradientTemplate,
    val crt_tpl: BinarizedGradientTemplate
  )

  object BinarizedGradientPyramid {
    def apply(pyramid: Rep[DenseVector[GrayscaleImage]], start_level: Rep[Int], levels: Rep[Int], fixedLevelIndex: Rep[Int]): BinarizedGradientPyramid = ???
  }

  //implicit def repBinarizedGradientPyramidToBinarizedGradientPyramidOps(x: Rep[BinarizedGradientPyramid]) = new binarizedgradientpyramidOpsCls(x)

  class BinarizedGradientPyramid (
    val pyramid: DenseVector[Image[Int]],
    val start_level: Int,
    val levels: Int,
    val fixedLevelIndex: Int
  )
 


  object BinarizedGradientTemplate {
    def apply(radius: Rep[Int], rect: Rep[Rect], mask_list: Rep[DenseVector[Int]], level: Rep[Int], binary_gradients: Rep[DenseVector[Int]], match_list: Rep[IndexVectorDense], occlusions: Rep[DenseVector[DenseVector[Int]]], templates: Rep[DenseVector[BinarizedGradientTemplate]], hist: Rep[DenseVector[Float]]): BinarizedGradientTemplate = ???
  }

  //implicit def repBinarizedGradientTemplateToBinarizedGradientTemplateOps(x: Rep[BinarizedGradientTemplate]) = new binarizedgradienttemplateOpsCls(x)

  // object BinarizedGradientTemplate {
  //   def apply(val radius: Int, ...) = newStruct(("radius","Int", radius),
  // }

  class BinarizedGradientTemplate (
    // In the reduced image. The side of the template square is then 2*r+1.
    val radius: Int,

    // Holds a tighter bounding box of the object in the original image scale
    val rect: Rect,
    val mask_list: DenseVector[Int],

    // Pyramid level of the template (reduction_factor = 2^level)
    val level: Int,

    // The list of gradients in the template
    val binary_gradients: DenseVector[Int],

    // indices to use for matching (skips zeros inside binary_gradients)
    val match_list: IndexVectorDense,

    // This is a match list of list of sub-parts. Currently unused.
    val occlusions: DenseVector[DenseVector[Int]],

    val templates: DenseVector[BinarizedGradientTemplate],

    val hist: DenseVector[Float]
  )
 
  def readGrayscaleImage(s: String): GrayscaleImage = ???
  def readTemplateModels(s: String) = ???




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