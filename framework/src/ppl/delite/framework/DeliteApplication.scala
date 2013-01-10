package ppl.delite.framework





trait DeliteApplication {

  var args: Array[String] = _

  def main(): Unit
  
  final def main(args: Array[String]) {
    this.args = args
    main()
  }

}
