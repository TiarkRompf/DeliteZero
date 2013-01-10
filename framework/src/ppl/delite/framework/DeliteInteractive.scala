package ppl.delite.framework

trait DeliteInteractive {
}

trait DeliteInteractiveRunner extends DeliteApplication with DeliteInteractive {
}

object DeliteSnippet {
  def apply[A,B](b: => Unit) = b
}
