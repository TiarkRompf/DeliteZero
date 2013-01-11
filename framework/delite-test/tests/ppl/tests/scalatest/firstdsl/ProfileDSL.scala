package ppl.tests.scalatest.firstdsl

import ppl.delite.framework._

/* Application packages */
trait ProfileApplicationRunner extends ProfileApplication 
  with DeliteApplication
trait ProfileApplication {
  var args: Array[String]
  def main(): Unit
}