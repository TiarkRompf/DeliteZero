package ppl.tests.scalatest.firstdsl


// this is the abstract interface of our profiling methods

object profile {
  def apply(n: Int) = new {
    def times(body: => Any) = {
      val out = new ProfileArray(n)
      var i = 0
      while (i < n) {
        val start = System.currentTimeMillis()
        body
        val end = System.currentTimeMillis()
        val duration = (end - start)/1000f
        out._data(i) = duration
        i += 1
      }
      out
    }
  }
}
