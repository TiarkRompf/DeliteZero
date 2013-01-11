package ppl.tests.scalatest

import org.scalatest._
import ppl.delite.framework.DeliteApplication


trait DeliteSuite extends Suite {
  def compileAndTest(x: DeliteTestRunner) {
    x.args = Array()
    x.main
  }
}

trait DeliteTestModule {
  def collect(s: Boolean) { 
    println(s); assert(s) 
  }
  def mkReport {}
}

trait DeliteTestRunner extends DeliteApplication {
}
