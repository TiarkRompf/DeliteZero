package ppl.delite.framework.codegen.delite.generators

import ppl.delite.framework.codegen.delite.DeliteCodegen
import ppl.delite.framework.ops.{VariantsOpsExp, DeliteOpsExp}
import ppl.delite.framework.{Util, Config}
import collection.mutable.{ArrayBuffer, ListBuffer, HashMap}
import java.io.{StringWriter, FileWriter, File, PrintWriter}
import scala.virtualization.lms.common.LoopFusionOpt
import scala.virtualization.lms.internal.{GenerationFailedException}

trait DeliteGenTaskGraph extends DeliteCodegen with LoopFusionOpt {
  val IR: DeliteOpsExp
  import IR._

  private def vals(sym: Sym[Any]) : List[Sym[Any]] = sym match {
    case Def(Reify(s, u, effects)) => if (s.isInstanceOf[Sym[Any]]) List(s.asInstanceOf[Sym[Any]]) else Nil
    case Def(Reflect(NewVar(v), u, effects)) => Nil
    case _ => List(sym)
  }

  private def vars(sym: Sym[Any]) : List[Sym[Any]] = sym match {
    case Def(Reflect(NewVar(v), u, effects)) => List(sym)
    case _ => Nil
  }

  private def mutating(kernelContext: State, sym: Sym[Any]) : List[Sym[Any]] =
    kernelContext flatMap {
      //case Def(Mutation(x,effects)) => if (syms(x) contains sym) List(sym) else Nil
      case Def(Reflect(x,Write(as),effects)) => if (as contains sym) List(sym) else Nil
      case _ => Nil
    }

/*
  override def getFreeVarNode(rhs: Def[Any]): List[Sym[Any]] = rhs match { // getFreeVarBlock(syms(rhs), boundSyms(rhs))
    case Reflect(s, u, effects) => getFreeVarNode(s)
    case _ => super.getFreeVarNode(rhs)
  }
  // this is all quite unfortunate
  private def appendScope() = {
    (emittedNodes flatMap { e => if (findDefinition(e).isDefined) List(findDefinition(e).get.asInstanceOf[TP[Any]]) else Nil }) ::: scope
  }

  def unwrapVar(v: Var[Any]) = v match {
    case Variable(x) => x
  }
*/

  override def emitFatNode(sym: List[Sym[Any]], rhs: FatDef)(implicit stream: PrintWriter): Unit = {
    assert(generators.length >= 1)

    println("DeliteGenTaskGraph.emitNode "+sym+"="+rhs)

    val kernelName = sym.map(quote).mkString("")

    var resultIsVar = false
    var skipEmission = false
    var nestedNode: TP[Any] = null

    // we will try to generate any node that is not purely an effect node
    rhs match {
      case ThinDef(Reflect(s, u, effects)) =>
        //controlDeps = effects; // <---  now handling control deps here...!! <--- would like to, but need to hand *precise* schedule to runtime
        super.emitFatNode(sym, rhs); return
      case ThinDef(Reify(s, u, effects)) =>
        //controlDeps = effects
        super.emitFatNode(sym, rhs); return
      case ThinDef(NewVar(x)) => resultIsVar = true // if sym is a NewVar, we must mangle the result type
      case _ => // continue and attempt to generate kernel
    }

    // validate that generators agree on inputs (similar to schedule validation in DeliteCodegen)
    //val dataDeps = ifGenAgree(g => (g.syms(rhs) ++ g.getFreeVarNode(rhs)).distinct, true)
    
    val dataDeps = { // don't use getFreeVarNode...
      val bound = boundSyms(rhs)
      val used = syms(rhs)
      focusFatBlock(used) { freeInScope(bound, used) } // distinct
      //syms(rhs).flatMap(s => focusBlock(s) { freeInScope(boundSyms(rhs), s) } ).distinct
    }

    val inVals = dataDeps flatMap { vals(_) }
    val inVars = dataDeps flatMap { vars(_) }

    implicit val supportedTargets = new ListBuffer[String]
    implicit val returnTypes = new ListBuffer[Pair[String, String]]
    implicit val outputSlotTypes = new HashMap[String, ListBuffer[(String, String)]]
    implicit val metadata = new ArrayBuffer[Pair[String, String]]

    // parameters for delite overrides
    deliteInputs = (inVals ++ inVars)
    deliteResult = Some(sym) //findDefinition(rhs) map { _.sym }

    if (!skipEmission) for (gen <- generators) {
      val buildPath = Config.buildDir + java.io.File.separator + gen + java.io.File.separator
      val outDir = new File(buildPath); outDir.mkdirs()
      val outFile = new File(buildPath + kernelName + "." + gen.kernelFileExt)
      val kstream = new PrintWriter(outFile)
      val bodyString = new StringWriter()
      val bodyStream = new PrintWriter(bodyString)

      try{
        // DISCUSS: use a predicate instead of inheriting from DeliteOp?
        rhs match {
//          case op:DeliteFatOp => deliteKernel = true
          case op:AbstractFatLoop => deliteKernel = true
          case ThinDef(op:DeliteOp[_]) => deliteKernel = true
          case _ => deliteKernel = false
        }

        //initialize
        gen.kernelInit(sym, inVals, inVars, resultIsVar)

        // emit kernel to bodyStream //TODO: must kernel body be emitted before kernel header?
        gen.emitFatNode(sym, rhs)(bodyStream)
        bodyStream.flush

        var hasOutputSlotTypes = false
        
        val resultType: String = (gen.toString, rhs) match {
          case ("scala", op: AbstractFatLoop) =>
            hasOutputSlotTypes = true
            "generated.scala.DeliteOpMultiLoop[" + "activation_"+kernelName + "]"
          case ("scala", ThinDef(z)) => z match {
            case op: AbstractLoop[_] => system.error("should not encounter thin loops here but only fat ones")
            case map: DeliteOpMap[_,_,_] => "generated.scala.DeliteOpMap[" + gen.remap(map.v.Type) + "," + gen.remap(map.func.Type) + "," + gen.remap(map.alloc.Type) + "]"
            case zip: DeliteOpZipWith[_,_,_,_] => "generated.scala.DeliteOpZipWith[" + gen.remap(zip.v._1.Type) + "," + gen.remap(zip.v._2.Type) + "," + gen.remap(zip.func.Type) + "," + gen.remap(zip.alloc.Type) +"]"
            case red: DeliteOpReduce[_] => "generated.scala.DeliteOpReduce[" + gen.remap(red.func.Type) + "]"
            case mapR: DeliteOpMapReduce[_,_,_] => "generated.scala.DeliteOpMapReduce[" + gen.remap(mapR.mV.Type) + "," + gen.remap(mapR.reduce.Type) + "]"
            case foreach: DeliteOpForeach[_,_] => "generated.scala.DeliteOpForeach[" + gen.remap(foreach.v.Type) + "]"
            case _ => gen.remap(sym.head.Type)
          }
          case _ => 
            assert(sym.length == 1) // if not set hasOutputSlotTypes and use activation record
            gen.remap(sym.head.Type)
        }

        assert(hasOutputSlotTypes || sym.length == 1)

        // emit kernel
        gen.emitKernelHeader(sym, inVals, inVars, resultType, resultIsVar)(kstream)
        kstream.println(bodyString.toString)
        gen.emitKernelFooter(sym, inVals, inVars, resultType, resultIsVar)(kstream)

        // record that this kernel was successfully generated
        supportedTargets += gen.toString
        if (!hasOutputSlotTypes) { // return type is sym type
          if (resultIsVar) {
            returnTypes += new Pair[String,String](gen.toString,"generated.scala.Ref[" + gen.remap(sym.head.Type) + "]") {
              override def toString = "\"" + _1 + "\" : \"" + _2 + "\""
            }
          } else {
            returnTypes += new Pair[String,String](gen.toString,gen.remap(sym.head.Type)) {
              override def toString = "\"" + _1 + "\" : \"" + _2 + "\""
            }
          }
        } else { // return type is activation record
          returnTypes += new Pair[String,String](gen.toString,"activation_" + kernelName) {
            override def toString = "\"" + _1 + "\" : \"" + _2 + "\""
          }
          for (s <- sym) {
            outputSlotTypes.getOrElseUpdate(quote(s), new ListBuffer) += new Pair[String,String](gen.toString,gen.remap(s.Type)) {
              override def toString = "\"" + _1 + "\" : \"" + _2 + "\""
            }
          }
        }

        //add MetaData
        if (gen.hasMetaData) {
          metadata += new Pair[String,String](gen.toString, gen.getMetaData) {
            override def toString = "\"" + _1 + "\" : " + _2
          }
        }

        kstream.close()
        
      } catch {
        case e:GenerationFailedException => // no generator found
          gen.exceptionHandler(e, outFile, kstream)
		      //println(gen.toString + ":" + quote(sym))
          //e.printStackTrace
          
          //if(gen.nested > 1) {
          //  nestedNode = gen.lastNodeAttempted
          //}
        case e:Exception => throw(e)
      }
    }

    if (skipEmission == false && supportedTargets.isEmpty) {
      var msg = "Node " + quote(sym) + "[" + rhs + "] could not be generated by any code generator"
      //if(nested > 1) msg = "Failure is in nested node " + quote(nestedNode.sym) + "[" + nestedNode.rhs + "]. " + msg
      system.error(msg)
    }

    val outputs = sym
    
    
    val inputs = deliteInputs
    //val kernelContext = getEffectsKernel(sym, rhs)
    val kernelContext = getEffectsBlock(sym) //ifGenAgree( _.getEffectsBlock(sym), true )
    val inMutating = (inputs flatMap { mutating(kernelContext, _) }).distinct

    // additional data deps: for each of my inputs, look at the kernels already generated and see if any of them
    // mutate it, and if so, add that kernel as a data-dep
    val extraDataDeps = (kernelMutatingDeps filter { case (s, mutates) => (!(inputs intersect mutates).isEmpty) }).keys
    val inControlDeps = (controlDeps ++ extraDataDeps).distinct

    // anti deps: for each of my mutating inputs, look at the kernels already generated and see if any of them
    // read it, add that kernel as an anti-dep
    val antiDeps = (kernelInputDeps filter { case (s, in) => (!(inMutating intersect in).isEmpty) }).keys.toList

    // add this kernel to global generated state
    sym foreach { s => kernelInputDeps += { s -> inputs } }
    sym foreach { s => kernelMutatingDeps += { s -> inMutating } }

    // debug
    /*
    stream.println("inputs: " + inputs)
    stream.println("mutating inputs: " + inMutating)
    stream.println("extra data deps: " + extraDataDeps)
    stream.println("control deps: " + inControlDeps)
    stream.println("anti deps:" + antiDeps)
    */
    println(outputSlotTypes)

    // emit task graph node
    rhs match {
      case op: AbstractFatLoop => 
        emitMultiLoop(kernelName, outputs, inputs, inMutating, inControlDeps, antiDeps, op.body.exists(_.isInstanceOf[DeliteReduceElem[Any]]))
      case ThinDef(z) => z match {
        case c:DeliteOpCondition[_] => emitIfThenElse(c.cond, c.thenp, c.elsep, kernelName, outputs, inputs, inMutating, inControlDeps, antiDeps)
        case w:DeliteOpWhileLoop => emitWhileLoop(w.cond, w.body, kernelName, outputs, inputs, inMutating, inControlDeps, antiDeps)
        case s:DeliteOpSingleTask[_] => emitSingleTask(kernelName, outputs, inputs, inMutating, inControlDeps, antiDeps)
        case m:DeliteOpMap[_,_,_] => emitMap(z, kernelName, outputs, inputs, inMutating, inControlDeps, antiDeps)
        case r:DeliteOpReduce[_] => emitReduce(kernelName, outputs, inputs, inMutating, inControlDeps, antiDeps)
        case a:DeliteOpMapReduce[_,_,_] => emitMapReduce(z, kernelName, outputs, inputs, inMutating, inControlDeps, antiDeps)
        case z:DeliteOpZipWith[_,_,_,_] => emitZipWith(kernelName, outputs, inputs, inMutating, inControlDeps, antiDeps)
        case f:DeliteOpForeach[_,_] => emitForeach(z, kernelName, outputs, inputs, inMutating, inControlDeps, antiDeps)
        case _ => emitSingleTask(kernelName, outputs, inputs, inMutating, inControlDeps, antiDeps) // things that are not specified as DeliteOPs, emit as SingleTask nodes
      }
    }

    // whole program gen (for testing)
    //emitValDef(sym, "embedding.scala.gen.kernel_" + quote(sym) + "(" + inputs.map(quote(_)).mkString(",") + ")")
  }

  /**
   * @param sym         the symbol representing the kernel
   * @param inputs      a list of real kernel dependencies (formal kernel parameters)
   * @param controlDeps a list of control dependencies (must execute before this kernel)
   * @param antiDeps    a list of WAR dependencies (need to be committed in program order)
   */

  def emitMultiLoop(id: String, outputs: List[Exp[Any]], inputs: List[Exp[Any]], mutableInputs: List[Exp[Any]], controlDeps: List[Exp[Any]], antiDeps: List[Exp[Any]], needsCombine: Boolean)
       (implicit stream: PrintWriter, supportedTgt: ListBuffer[String], returnTypes: ListBuffer[Pair[String, String]], outputSlotTypes: HashMap[String, ListBuffer[(String, String)]], metadata: ArrayBuffer[Pair[String,String]]) = {
   stream.print("{\"type\":\"MultiLoop\", \"needsCombine\":" + needsCombine)
   emitExecutionOpCommon(id, outputs, inputs, mutableInputs, controlDeps, antiDeps)
   stream.println("},")
  }

  def emitSingleTask(id: String, outputs: List[Exp[Any]], inputs: List[Exp[Any]], mutableInputs: List[Exp[Any]], controlDeps: List[Exp[Any]], antiDeps: List[Exp[Any]])
        (implicit stream: PrintWriter, supportedTgt: ListBuffer[String], returnTypes: ListBuffer[Pair[String, String]], outputSlotTypes: HashMap[String, ListBuffer[(String, String)]], metadata: ArrayBuffer[Pair[String,String]]) = {
    stream.print("{\"type\":\"SingleTask\"")
    emitExecutionOpCommon(id, outputs, inputs, mutableInputs, controlDeps, antiDeps)
    stream.println("},")
  }

  def emitMap(rhs: Def[Any], id: String, outputs: List[Exp[Any]], inputs: List[Exp[Any]], mutableInputs: List[Exp[Any]], controlDeps: List[Exp[Any]], antiDeps: List[Exp[Any]])
        (implicit stream: PrintWriter, supportedTgt: ListBuffer[String], returnTypes: ListBuffer[Pair[String, String]], outputSlotTypes: HashMap[String, ListBuffer[(String, String)]], metadata: ArrayBuffer[Pair[String,String]]) = {
    stream.print("{\"type\":\"Map\"")
    emitExecutionOpCommon(id, outputs, inputs, mutableInputs, controlDeps, antiDeps)
    emitVariant(rhs, id, outputs, inputs, mutableInputs, controlDeps, antiDeps)
    stream.println("},")
  }

  def emitReduce(id: String, outputs: List[Exp[Any]], inputs: List[Exp[Any]], mutableInputs: List[Exp[Any]], controlDeps: List[Exp[Any]], antiDeps: List[Exp[Any]])
        (implicit stream: PrintWriter, supportedTgt: ListBuffer[String], returnTypes: ListBuffer[Pair[String, String]], outputSlotTypes: HashMap[String, ListBuffer[(String, String)]], metadata: ArrayBuffer[Pair[String,String]]) = {
    stream.print("{\"type\":\"Reduce\"")
    emitExecutionOpCommon(id, outputs, inputs, mutableInputs, controlDeps, antiDeps)
    stream.println("},")
  }

  def emitMapReduce(rhs: Def[Any], id: String, outputs: List[Exp[Any]], inputs: List[Exp[Any]], mutableInputs: List[Exp[Any]], controlDeps: List[Exp[Any]], antiDeps: List[Exp[Any]])
        (implicit stream: PrintWriter, supportedTgt: ListBuffer[String], returnTypes: ListBuffer[Pair[String, String]], outputSlotTypes: HashMap[String, ListBuffer[(String, String)]], metadata: ArrayBuffer[Pair[String,String]]) = {
    stream.print("{\"type\":\"MapReduce\"")
    emitExecutionOpCommon(id, outputs, inputs, mutableInputs, controlDeps, antiDeps)
    emitVariant(rhs, id, outputs, inputs, mutableInputs, controlDeps, antiDeps)
    stream.println("},")
  }

  def emitZipWith(id: String, outputs: List[Exp[Any]], inputs: List[Exp[Any]], mutableInputs: List[Exp[Any]], controlDeps: List[Exp[Any]], antiDeps: List[Exp[Any]])
        (implicit stream: PrintWriter, supportedTgt: ListBuffer[String], returnTypes: ListBuffer[Pair[String, String]], outputSlotTypes: HashMap[String, ListBuffer[(String, String)]], metadata: ArrayBuffer[Pair[String,String]]) = {
    stream.print("{\"type\":\"ZipWith\"")
    emitExecutionOpCommon(id, outputs, inputs, mutableInputs, controlDeps, antiDeps)
    stream.println("},")
  }

  def emitForeach(rhs: Def[Any], id: String, outputs: List[Exp[Any]], inputs: List[Exp[Any]], mutableInputs: List[Exp[Any]], controlDeps: List[Exp[Any]], antiDeps: List[Exp[Any]])
        (implicit stream: PrintWriter, supportedTgt: ListBuffer[String], returnTypes: ListBuffer[Pair[String, String]], outputSlotTypes: HashMap[String, ListBuffer[(String, String)]], metadata: ArrayBuffer[Pair[String,String]]) = {
    stream.print("{\"type\":\"Foreach\"")
    emitExecutionOpCommon(id, outputs, inputs, mutableInputs, controlDeps, antiDeps)
    emitVariant(rhs, id, outputs, inputs, mutableInputs, controlDeps, antiDeps)
    stream.println("},")
  }

  def emitIfThenElse(cond: Exp[Boolean], thenp: Exp[Any], elsep: Exp[Any], id: String, outputs: List[Exp[Any]], inputs: List[Exp[Any]], mutableInputs: List[Exp[Any]], controlDeps: List[Exp[Any]], antiDeps: List[Exp[Any]])
        (implicit stream: PrintWriter, supportedTgt: ListBuffer[String], returnTypes: ListBuffer[Pair[String, String]], outputSlotTypes: HashMap[String, ListBuffer[(String, String)]], metadata: ArrayBuffer[Pair[String,String]]) = {
    stream.print("{\"type\":\"Conditional\",")
    stream.println("  \"outputId\" : \"" + id + "\",")
    emitSubGraph("cond", cond)
    emitSubGraph("then", thenp)
    emitSubGraph("else", elsep)
    stream.println("  \"condOutput\": \"" + quote(getBlockResult(cond)) + "\",")
    stream.println("  \"thenOutput\": \"" + quote(getBlockResult(thenp)) + "\",")
    stream.println("  \"elseOutput\": \"" + quote(getBlockResult(elsep)) + "\",")
    stream.println("  \"controlDeps\":[" + makeString(controlDeps) + "],")
    stream.println("  \"antiDeps\":[" + makeString(antiDeps) + "],")
    if (remap(thenp.Type) != remap(elsep.Type))
      throw new RuntimeException("Delite conditional with different then and else return types")
    val returnTypesStr = if(returnTypes.isEmpty) "" else returnTypes.mkString(",")
    stream.println("  \"return-types\":{" + returnTypesStr + "}")
    stream.println("},")
  }

  def emitWhileLoop(cond: Exp[Boolean], body: Exp[Unit], id: String, outputs: List[Exp[Any]], inputs: List[Exp[Any]], mutableInputs: List[Exp[Any]], controlDeps: List[Exp[Any]], antiDeps: List[Exp[Any]])
        (implicit stream: PrintWriter, supportedTgt: ListBuffer[String], returnTypes: ListBuffer[Pair[String, String]], outputSlotTypes: HashMap[String, ListBuffer[(String, String)]], metadata: ArrayBuffer[Pair[String,String]]) = {
    stream.println("{\"type\":\"WhileLoop\",")
    stream.println("  \"outputId\" : \"" + id + "\",")
    emitSubGraph("cond", cond)
    emitSubGraph("body", body)
    stream.println("  \"condOutput\": \"" + quote(getBlockResult(cond)) + "\",")
    //stream.println("  \"bodyOutput\": \"" + quote(getBlockResult(body)) + "\",")
    stream.println("  \"controlDeps\":[" + makeString(controlDeps) + "],")
    stream.println("  \"antiDeps\":[" + makeString(antiDeps) + "]")
    stream.println("},")
  }

  def emitControlFlowOpCommon(id: String, outputs: List[Exp[Any]], inputs: List[Exp[Any]], mutableInputs: List[Exp[Any]], controlDeps: List[Exp[Any]], antiDeps: List[Exp[Any]])
        (implicit stream: PrintWriter, supportedTgt: ListBuffer[String], returnTypes: ListBuffer[Pair[String, String]], outputSlotTypes: HashMap[String, ListBuffer[(String, String)]], metadata: ArrayBuffer[Pair[String,String]]) = {
  }

  def emitExecutionOpCommon(id: String, outputs: List[Exp[Any]], inputs: List[Exp[Any]], mutableInputs: List[Exp[Any]], controlDeps: List[Exp[Any]], antiDeps: List[Exp[Any]])
        (implicit stream: PrintWriter, supportedTgt: ListBuffer[String], returnTypes: ListBuffer[Pair[String, String]], outputSlotTypes: HashMap[String, ListBuffer[(String, String)]], metadata: ArrayBuffer[Pair[String,String]]) = {
    stream.print(" , \"kernelId\" : \"" + id + "\" ")
    stream.print(" , \"supportedTargets\": [" + supportedTgt.mkString("\"","\",\"","\"") + "],\n")
    stream.print("  \"outputs\":[" + outputs.map("\""+quote(_)+"\"").mkString(",") + "],\n")
    stream.print("  \"inputs\":[" + inputs.map("\""+quote(_)+"\"").mkString(",") + "],\n")
    stream.print("  \"mutableInputs\":[" + mutableInputs.map("\""+quote(_)+"\"").mkString(",") + "],\n")
    emitDepsCommon(controlDeps, antiDeps)
    val metadataStr = if (metadata.isEmpty) "" else metadata.mkString(",")
    stream.print("  \"metadata\":{" + metadataStr + "},\n")
    val returnTypesStr = if(returnTypes.isEmpty) "" else returnTypes.mkString(",")
    stream.print("  \"return-types\":{" + returnTypesStr + "}")
    if (!outputSlotTypes.isEmpty) {
      stream.print(",\n")
      val rts = for (s <- outputs) yield {
        val str = quote(s)
        "  \""+str+"\":{" + outputSlotTypes(str).mkString(",") + "}"
      }
      stream.print("\"output-types\":{" + rts.mkString(",") + "}\n")
    } else
      stream.print("\n")
  }


  def emitDepsCommon(controlDeps: List[Exp[Any]], antiDeps: List[Exp[Any]], last:Boolean = false)(implicit stream: PrintWriter) {
    stream.print("  \"controlDeps\":[" + makeString(controlDeps) + "],\n")
    stream.print("  \"antiDeps\":[" + makeString(antiDeps) + "]" + (if(last) "\n" else ",\n"))
  }

  var nested = 0

  def emitVariant(rhs: Def[Any], id: String, outputs: List[Exp[Any]], inputs: List[Exp[Any]], mutableInputs: List[Exp[Any]], controlDeps: List[Exp[Any]], antiDeps: List[Exp[Any]])
                 (implicit stream: PrintWriter, supportedTgt: ListBuffer[String], returnTypes: ListBuffer[Pair[String, String]], metadata: ArrayBuffer[Pair[String,String]]) {

    if (!rhs.isInstanceOf[Variant] || nested > Config.nestedVariantsLevel) return
    
    nested += 1

    // pre
    val saveMutatingDeps = kernelMutatingDeps
    val saveInputDeps = kernelInputDeps
    kernelMutatingDeps = Map()
    kernelInputDeps = Map()
    stream.print(",\"variant\": {")
    stream.print("\"ops\":[" )

    // variant
    rhs match {
      case mvar:DeliteOpMapLikeWhileLoopVariant => emitMapLikeWhileLoopVariant(mvar, id, outputs, inputs, mutableInputs, controlDeps, antiDeps)
      case rvar:DeliteOpReduceLikeWhileLoopVariant => emitReduceLikeWhileLoopVariant(rvar, id, outputs, inputs, mutableInputs, controlDeps, antiDeps)
      case _ =>
    }

    // post
    emitEOG()
    emitOutput(getBlockResult(rhs.asInstanceOf[Variant].variant))
    stream.println("}")
    kernelInputDeps = saveInputDeps
    kernelMutatingDeps = saveMutatingDeps
    
    nested -= 1    
  }

  def emitMapLikeWhileLoopVariant(vw: DeliteOpMapLikeWhileLoopVariant, id: String, outputs: List[Exp[Any]], inputs: List[Exp[Any]], mutableInputs: List[Exp[Any]], controlDeps: List[Exp[Any]], antiDeps: List[Exp[Any]])
    (implicit stream: PrintWriter, supportedTgt: ListBuffer[String], returnTypes: ListBuffer[Pair[String, String]], metadata: ArrayBuffer[Pair[String,String]]) {

    // manually lift alloc out of the variant loop. TODO: this should not be required, see comment in DeliteOps.scala
    // we should be able to remove this when we merge with opfusing
    //val save = scope
    emitBlock(vw.alloc)
    //scope = appendScope()
    emitBlock(vw.variant)
    //scope = save
  }

  def emitReduceLikeWhileLoopVariant(vw: DeliteOpReduceLikeWhileLoopVariant, id: String, outputs: List[Exp[Any]], inputs: List[Exp[Any]], mutableInputs: List[Exp[Any]], controlDeps: List[Exp[Any]], antiDeps: List[Exp[Any]])
    (implicit stream: PrintWriter, supportedTgt: ListBuffer[String], returnTypes: ListBuffer[Pair[String, String]], metadata: ArrayBuffer[Pair[String,String]]) {

    //val save = scope
    emitBlock(vw.Index)
    emitBlock(vw.Acc)
    //scope = appendScope()

    def emitSubGraphOp(block: Exp[Any], controlDeps: List[Exp[Any]], antiDeps: List[Exp[Any]]) {
      stream.print("{\"type\":\"SubGraph\", ")
      val resultId = if (quote(getBlockResult(block)) == "()") quote(block) else quote(getBlockResult(block)) //TODO: :(
      stream.print("\"outputId\":\"" + resultId + "\",\n")
      emitDepsCommon(controlDeps, antiDeps)
      emitSubGraph("", block)
      emitOutput(block)
      stream.println("},")
    }

    emitSubGraphOp(vw.init, Nil, Nil)
    emitSubGraphOp(vw.variant, List(vw.init), Nil)
    //scope = save
  }

  def emitOutput(x: Exp[Any])(implicit stream: PrintWriter) = {
    x match {
      case c:Const[Any] => stream.println("  \"outputType\": \"const\",")
                           stream.println("  \"outputValue\": \"" + quote(x) + "\"")
      case s:Sym[Any] =>   stream.println("  \"outputType\": \"symbol\",")
                           stream.println("  \"outputValue\": \"" + quote(getBlockResult(x)) + "\"")
    }
  }

  def emitEOG()(implicit stream: PrintWriter) = {
    stream.print("{\"type\":\"EOG\"}\n],\n")
  }

  def emitSubGraph(prefix: String, e: Exp[Any])(implicit stream: PrintWriter) = e match {
    case c:Const[Any] => stream.println("  \"" + prefix + "Type\": \"const\",")
                         stream.println("  \"" + prefix + "Value\": \"" + quote(e) + "\",")
    case s:Sym[Any] =>  stream.println("  \"" + prefix + "Type\": \"symbol\",")
                        stream.println("  \"" + prefix + "Ops\": [")
                        val saveMutatingDeps = kernelMutatingDeps
                        val saveInputDeps = kernelInputDeps
                        kernelMutatingDeps = Map()
                        kernelInputDeps = Map()
                        emitBlock(e)
                        emitEOG()
                        kernelInputDeps = saveInputDeps
                        kernelMutatingDeps = saveMutatingDeps
  }

  private def makeString(list: List[Exp[Any]]) = {
    if(list.isEmpty) "" else list.map(quote(_)).mkString("\"","\",\"","\"")
  }

/*
  // more quirks
  override def quote(x: Exp[Any]) = x match {
    case r:Reify[Any] => quote(r.x) //DISCUSS <- what's the purpose of this? it will never match because Reify is a Def, not Exp
    case _ => super.quote(x)
  }
*/

  def nop = throw new RuntimeException("Not Implemented Yet")

}
