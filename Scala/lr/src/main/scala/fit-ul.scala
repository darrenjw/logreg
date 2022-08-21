/*
fit-ul.scala
Bayesian MCMC for logistic regression model in scala using an unadjusted Langevin algorithm (approximate)
*/

import smfsb.*
import scalaglm.*
import breeze.linalg.*
import breeze.numerics.*
import breeze.stats.distributions.Gaussian
import breeze.stats.distributions.Rand.FixedSeed.randBasis
import smile.data.pimpDataFrame

def ulKernel(glp: DVD => DVD, pre: DVD, dt: Double): DVD => DVD =
  val sdt = math.sqrt(dt)
  val spre = sqrt(pre)
  def advance(beta: DVD): DVD =
    beta + (0.5*dt)*(pre*:*glp(beta))
  beta => advance(beta) + sdt*spre.map(Gaussian(0,_).sample())

@main def ul() =
  println("First read and process the data")
  val df = smile.read.parquet("../../pima.parquet")
  print(df)
  val y = DenseVector(df.select("type").
    map(_(0).asInstanceOf[String]).
    map(s => if (s == "Yes") 1.0 else 0.0).toArray)
  println(y)
  val names = df.drop("type").names.toList
  println(names)
  val x = DenseMatrix(df.drop("type").toMatrix.toArray:_*)
  println(x)
  val ones = DenseVector.ones[Double](x.rows)
  val X = DenseMatrix.horzcat(ones.toDenseMatrix.t, x)
  println(X)
  val p = X.cols
  println("Now do a simple MLE for a sanity check")
  val lr = Glm(y, x, names, LogisticGlm)
  println(lr.summary)
  println(lr.coefficients)
  println("Now do unadjusted Langevin")
  val pvar = DenseVector(100.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0)
  def glp(beta: DVD): DVD =
    val glpr = -beta /:/ pvar
    val gll = (X.t)*(y - ones/:/(ones + exp(-X*beta)))
    glpr + gll
  println(glp(lr.coefficients))
  val pre = DenseVector(100.0,1.0,1.0,1.0,1.0,1.0,25.0,1.0)
  val kern = ulKernel(glp, pre, 1.0e-6)
  val s = LazyList.iterate(lr.coefficients)(kern)
  val out = s.drop(150).thin(2000).take(10000)
  println("Starting unadjusted Langevin run now. Be patient...")
  //out.zipWithIndex.foreach(println)
  val aa: Array[Array[Double]] = out.toArray.map(_.toArray)
  val odf = smile.data.DataFrame.of(aa)
  smile.write.csv(odf, "fit-ul.csv")
  Mcmc.summary(out,false)
  println("Done.")
