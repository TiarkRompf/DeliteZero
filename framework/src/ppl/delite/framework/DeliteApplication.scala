package ppl.delite.framework


object Config
object Util

trait Interfaces extends Base
trait FunctionBlocksExp extends Base
trait Variables extends Base
trait Base extends OverloadHack {
  type Rep[T] = T
  type Var[T] = T
  type Interface[T] = T
  class Def[T]
  def readVar[T](x:T):T = x
  def unit[T](x:T):T = x
}
trait BaseExp extends Base

trait DeliteCollectionOps
trait DeliteArrayOps

trait OverloadHack {

  class Overloaded0
  class Overloaded1
  class Overloaded2
  class Overloaded3
  class Overloaded4
  class Overloaded5
  class Overloaded6
  class Overloaded7
  class Overloaded8
  class Overloaded9

  class Overloaded10
  class Overloaded11
  class Overloaded12
  class Overloaded13
  class Overloaded14
  class Overloaded15
  class Overloaded16
  class Overloaded17
  class Overloaded18
  class Overloaded19

  class Overloaded20
  class Overloaded21
  class Overloaded22
  class Overloaded23
  class Overloaded24
  class Overloaded25
  class Overloaded26
  class Overloaded27
  class Overloaded28
  class Overloaded29


  implicit object OV0 extends Overloaded0
  implicit object OV1 extends Overloaded1
  implicit object OV2 extends Overloaded2 
  implicit object OV3 extends Overloaded3
  implicit object OV4 extends Overloaded4
  implicit object OV5 extends Overloaded5
  implicit object OV6 extends Overloaded6
  implicit object OV7 extends Overloaded7
  implicit object OV8 extends Overloaded8
  implicit object OV9 extends Overloaded9

  implicit object OV10 extends Overloaded10
  implicit object OV11 extends Overloaded11
  implicit object OV12 extends Overloaded12 
  implicit object OV13 extends Overloaded13
  implicit object OV14 extends Overloaded14
  implicit object OV15 extends Overloaded15
  implicit object OV16 extends Overloaded16
  implicit object OV17 extends Overloaded17
  implicit object OV18 extends Overloaded18
  implicit object OV19 extends Overloaded19

  implicit object OV20 extends Overloaded20
  implicit object OV21 extends Overloaded21
  implicit object OV22 extends Overloaded22 
  implicit object OV23 extends Overloaded23
  implicit object OV24 extends Overloaded24
  implicit object OV25 extends Overloaded25
  implicit object OV26 extends Overloaded26
  implicit object OV27 extends Overloaded27
  implicit object OV28 extends Overloaded28
  implicit object OV29 extends Overloaded29

}




trait DeliteApplication extends Base {

  var args: Array[String] = _

  def main(): Unit
  
  final def main(args: Array[String]) {
    this.args = args
    main()
  }

}
