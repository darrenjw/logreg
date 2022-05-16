/*
fit-bayes.scala
Bayesian MCMC for logistic regression model in scala
*/

import smfsb.*
import scalaglm.*
import breeze.linalg.*
import breeze.numerics.*
import breeze.stats.distributions.Gaussian
import breeze.stats.distributions.Rand.FixedSeed.randBasis
import smile.data.pimpDataFrame

@main def go() =
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
  println("Now do RW MH")
  def ll(beta: DenseVector[Double]): Double =
      sum(-log(ones + exp(-1.0*(2.0*y - ones)*:*(X * beta))))
  def lprior(beta: DenseVector[Double]): Double =
    Gaussian(0,10).logPdf(beta(0)) + 
      sum(beta(1 until p).map(Gaussian(0,1).logPdf(_)))
  def lpost(beta: DenseVector[Double]): Double =
    ll(beta) + lprior(beta)
  println(lpost(lr.coefficients))
  val pre = DenseVector(10.0,1.0,1.0,1.0,1.0,1.0,5.0,1.0)
  def rprop(beta: DoubleState): DoubleState = beta + pre *:* (DenseVector(Gaussian(0.0,0.02).sample(p).toArray))
  def dprop(n: DoubleState, o: DoubleState): Double = 1.0
  val s = Mcmc.mhStream(lr.coefficients, lpost, rprop, dprop,
    (p: DoubleState) => 1.0, verb = false)
  val out = s.drop(150).thin(1000).take(10000)
  println("Starting RW MH run now. Be patient...")
  //out.zipWithIndex.foreach(println)
  Mcmc.summary(out,true)
  println("Done.")
