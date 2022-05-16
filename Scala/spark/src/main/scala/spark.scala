/*
Stub for scala-smfsb code
*/

import smfsb.*
import breeze.linalg.*
import breeze.numerics.*

@main def go() =
  val model = SpnModels.lv[IntState]()
  val step = Step.gillespie(model)
  val ts = Sim.ts(DenseVector(50, 100), 0.0, 20.0, 0.05, step)
  Sim.plotTs(ts, "Gillespie simulation of LV model")

