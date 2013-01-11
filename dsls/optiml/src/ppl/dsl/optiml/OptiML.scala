package ppl.dsl.optiml

import ppl.delite.framework._
import ppl.delite.framework.datastructures._

import ppl.dsl.optila._

object application {
  trait ApplicationOps
}

trait OptiML extends DeliteApplication { this: OptiMLApplication =>

  val Rect: RectCompanion = ???
  abstract class RectCompanion {
    def apply(x:Int,y:Int,w:Int,h:Int): Rect
  }
  abstract class Rect

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
