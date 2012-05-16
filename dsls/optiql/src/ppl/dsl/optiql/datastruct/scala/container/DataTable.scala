package ppl.dsl.optiql.datastruct.scala.container

object DataTable {

  def printAsTable(m: Map[String,Any], max_rows: Int = 0) { // FIXME: max_rows not used!
    // Check if Table is empty
    if(m.size == 0) {
      println("=====================================================")
      println("|                  EMPTY TABLE                      |")
      println("=====================================================")
      return
    }

    implicit val tableStr = new StringBuilder
    val numRows = m("size").asInstanceOf[Int]
    val fields = m("data").asInstanceOf[Map[String,Any]]

    val columnSizes = getTableColSizes(fields)

    def repeat(s: String, n:Int) {
      //assert(n < 0 || n > 300, "Incorrect value supplied for n in repeat")
      //todo for now, just ignore bad value of n
      if(n < 0 || n > 300)
        return
      var idx = 0
      while(idx != n) {
        tableStr.append(s);
        idx += 1
      }
    }

    def horizontalRule = {
      for(i <- 0 until columnSizes.size )
        repeat("=",columnSizes(i)+1)
      tableStr append("=\n")
    }

    def newLine = {
      tableStr append("\n")
    }

    horizontalRule
    tableStr append("|")
    val fieldStrs = fields.keys.toArray
    for(i <- 0 until columnSizes.size) {
      tableStr append( " " + fieldStrs(i))
      repeat(" " , columnSizes(i) - fieldStrs(i).size - 1  )
      tableStr append("|")
    }
    newLine
    horizontalRule
    print(tableStr.toString)
    tableStr.clear

    for(r <- 0 until numRows) {
      emitRecordAsRow(r, columnSizes)
    }

    print(tableStr.toString)
    tableStr.clear

    horizontalRule
    println(tableStr.toString)

    def emitRecordAsRow(r: Int, columnSizes: Array[Int]) {
      tableStr append("| ")
      var str = ""

      var idx = 0
      for(f <- fields.keys) {
        str = readArray(fields(f), r)
        tableStr append(str); repeat(" ", columnSizes(idx) - str.size - 1); tableStr append("| ")
        idx += 1
      }
      tableStr append("\n")
    }
  }

  private def max(a: Int, b: Int) = if(a > b) a else b

  private def readArray(a: Any, idx: Int) = a match {
    case i: Array[Int] => i(idx).toString
    case l: Array[Long] => l(idx).toString
    case d: Array[Double] => d(idx).toString
    case f: Array[Float] => f(idx).toString
    case c: Array[Char] => c(idx).toString
    case s: Array[Short] => s(idx).toString
    case b: Array[Byte] => b(idx).toString
    case z: Array[Boolean] => z(idx).toString
    case r: Array[AnyRef] => r(idx).toString
    case ar: AnyRef => throw new IllegalArgumentException(ar.getClass.getSimpleName + " cannot be printed as a table")
  }

  private def arrayToString(a: Any) = a match {
    case i: Array[Int] => i map { e => e.toString }
    case l: Array[Long] => l map { e => e.toString }
    case d: Array[Double] => d map { e => e.toString }
    case f: Array[Float] => f map { e => e.toString }
    case c: Array[Char] => c map { e => e.toString }
    case s: Array[Short] => s map { e => e.toString }
    case b: Array[Byte] => b map { e => e.toString }
    case z: Array[Boolean] => z map { e => e.toString }
    case r: Array[AnyRef] => r map { e => e.toString }
    case ar: AnyRef => throw new IllegalArgumentException(ar.getClass.getSimpleName + " cannot be printed as a table")
  }

  private def getTableColSizes(fields: Map[String,Any]) = {
    val colSizes = new Array[Int](fields.size)

    //Columns should be at least the size of the headers
    var idx = 0
    for(f <- fields.keys) {
      colSizes(idx) = max(colSizes(idx), f.length + 2)
      idx += 1
    }
    //columns should be at least the size of maximal element
    idx = 0
    for (f <- fields.keys) {
      for (d <- arrayToString(fields(f))) {
        colSizes(idx) = max(colSizes(idx), d.length + 2)
      }
      idx += 1
    }

    colSizes
  }
}
