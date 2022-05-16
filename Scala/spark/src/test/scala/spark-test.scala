
import smfsb.*
import breeze.linalg.*
import org.scalatest.*
import flatspec.*

class SmfsbSpec extends AnyFlatSpec:

  "Step.gillespie" should "create and step LV model" in {
    val model = SpnModels.lv[IntState]()
    val step = Step.gillespie(model)
    val output = step(DenseVector(50, 100), 0.0, 1.0)
    assert(output.length === 2)
  } 



