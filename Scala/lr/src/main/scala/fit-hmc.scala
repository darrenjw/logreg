/*
fit-hmc.scala
Bayesian MCMC for logistic regression model in scala using HMC
*/

import smfsb.*
import scalaglm.*
import breeze.linalg.*
import breeze.numerics.*
import breeze.stats.distributions.Gaussian
import breeze.stats.distributions.Rand.FixedSeed.randBasis
import smile.data.pimpDataFrame
import annotation.tailrec

def hmcKernel(lpi: DVD => Double, glpi: DVD => DVD, dmm: DVD,
  eps: Double = 1e-4, l: Int = 10) =
  val sdmm = sqrt(dmm)
  def leapf(q: DVD): DVD = 
    val p = sdmm map (sd => Gaussian(0,sd).draw())
    @tailrec def go(q0: DVD, p0: DVD, l: Int): DVD =
      val q = q0 + eps*(p0/:/dmm)
      val p = if (l > 0)
        p0 + eps*glpi(q)
      else
        p0 + 0.5*eps*glpi(q)
      if (l == 1)
        DenseVector.vertcat(q, -p)
      else
        go(q, p, l-1)
    go(q, p + 0.5*eps*glpi(q), l)
  def rprop(x: DVD): DVD =
    leapf(x(0 until (x.length/2)))
  def alpi(x: DVD): Double =
    val d = x.length/2
    lpi(x(0 until d)) - 0.5*sum(pow(x(d until 2*d),2) /:/ dmm)
  mhKernel(alpi, rprop)

@main def hmc() =
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
  println("Now do HMC")
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
  val kern = hmcKernel(lpost, glp, 1.0 / pre, eps=1e-3, l=50)
  val s = LazyList.iterate((DenseVector.vertcat(lr.coefficients,
    1e10*DenseVector.ones[Double](pre.length)), -Inf))(kern) map (_._1)
  val out = s.drop(150).thin(20).map(x => x(0 until pre.length)).take(10000)
  println("Starting HMC run now. Be patient...")
  Mcmc.summary(out,true)
  println("Done.")
