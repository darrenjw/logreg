/*
fit-mala.scala
Bayesian MCMC for logistic regression model in scala
*/

import smfsb.*
import scalaglm.*
import breeze.linalg.*
import breeze.numerics.*
import breeze.stats.distributions.Gaussian
import breeze.stats.distributions.Rand.FixedSeed.randBasis
import smile.data.pimpDataFrame

def malaKernel(lpi: DVD => Double, glpi: DVD => DVD, pre: DVD, dt: Double = 1e-4) =
  val sdt = math.sqrt(dt)
  val spre = sqrt(pre)
  val p = pre.length
  def advance(beta: DVD): DVD =
    beta + (0.5*dt)*(pre*:*glpi(beta))
  def rprop(beta: DVD): DVD =
    advance(beta) + sdt*spre.map(Gaussian(0,_).sample())
  def dprop(n: DVD, o: DVD): Double = 
    val ao = advance(o)
    (0 until p).map(i => Gaussian(ao(i), spre(i)*sdt).logPdf(n(i))).sum
  mhKernel(lpi, rprop, dprop)

@main def mala() =
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
  println("Now do MALA")
  def ll(beta: DVD): Double =
      sum(-log(ones + exp(-1.0*(2.0*y - ones)*:*(X * beta))))
  def lprior(beta: DVD): Double =
    Gaussian(0,10).logPdf(beta(0)) + 
      sum(beta(1 until p).map(Gaussian(0,1).logPdf(_)))
  def lpost(beta: DVD): Double =
    ll(beta) + lprior(beta)
  println(lpost(lr.coefficients))
  val pvar = DenseVector(100.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0)
  def glp(beta: DVD): DVD =
    val glpr = -beta /:/ pvar
    val gll = (X.t)*(y - ones/:/(ones + exp(-X*beta)))
    glpr + gll
  println(glp(lr.coefficients))
  val pre = DenseVector(100.0,1.0,1.0,1.0,1.0,1.0,25.0,1.0)
  val kern = malaKernel(lpost, glp, pre, dt = 1e-5)
  val s = LazyList.iterate((lr.coefficients, -Inf))(kern) map (_._1)
  val out = s.drop(150).thin(1000).take(10000)
  println("Starting MALA run now. Be patient...")
  val aa: Array[Array[Double]] = out.toArray.map(_.toArray)
  val odf = smile.data.DataFrame.of(aa)
  smile.write.csv(odf, "fit-mala.csv")
  Mcmc.summary(out, false)
  println("Done.")
