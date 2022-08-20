/*
fit-hmc.scala
Bayesian MCMC for logistic regression model in scala using HMC
*/

import smfsb.*
import scalaglm.*
import breeze.linalg.*
import breeze.numerics.*
import breeze.stats.distributions.{Gaussian, Uniform}
import breeze.stats.distributions.Rand.FixedSeed.randBasis
import smile.data.pimpDataFrame
import annotation.tailrec

def mhKern[S](
    logPost: S => Double, rprop: S => S,
    dprop: (S, S) => Double = (n: S, o: S) => 1.0
  ): (S) => S =
    val r = Uniform(0.0,1.0)
    x0 =>
      val x = rprop(x0)
      val ll0 = logPost(x0)
      val ll = logPost(x)
      val a = ll - ll0 + dprop(x0, x) - dprop(x, x0)
      if (math.log(r.draw()) < a) x else x0

def hmcKernel(lpi: DVD => Double, glpi: DVD => DVD, dmm: DVD,
  eps: Double = 1e-4, l: Int = 10) =
  val sdmm = sqrt(dmm)
  def leapf(q: DVD, p: DVD): (DVD, DVD) = 
    @tailrec def go(q0: DVD, p0: DVD, l: Int): (DVD, DVD) =
      val q = q0 + eps*(p0/:/dmm)
      val p = if (l > 1)
        p0 + eps*glpi(q)
      else
        p0 + 0.5*eps*glpi(q)
      if (l == 1)
        (q, -p)
      else
        go(q, p, l-1)
    go(q, p + 0.5*eps*glpi(q), l)
  def alpi(x: (DVD, DVD)): Double =
    val (q, p) = x
    lpi(q) - 0.5*sum(pow(p,2) /:/ dmm)
  def rprop(x: (DVD, DVD)): (DVD, DVD) =
    val (q, p) = x
    leapf(q, p)
  val mhk = mhKern(alpi, rprop)
  (q: DVD) =>
    val d = q.length
    val p = sdmm map (sd => Gaussian(0,sd).draw())
    mhk((q, p))._1

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
  val s = LazyList.iterate(lr.coefficients)(kern)
  val out = s.drop(150).thin(20).take(10000)
  println("Starting HMC run now. Be patient...")
  val aa: Array[Array[Double]] = out.toArray.map(_.toArray)
  val odf = smile.data.DataFrame.of(aa)
  smile.write.csv(odf, "fit-hmc.csv")
  Mcmc.summary(out,false)
  println("Done.")
