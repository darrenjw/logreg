/*
fit-bayes.scala
Bayesian MCMC for logistic regression model in scala
*/

import smfsb.*
import scalaglm.*
import breeze.linalg.*
import breeze.numerics.*
import breeze.stats.distributions.{Gaussian, Uniform}
import breeze.stats.distributions.Rand.FixedSeed.randBasis
import smile.data.pimpDataFrame

type DVD = DenseVector[Double]

def mhKernel[S](
    logPost: S => Double, rprop: S => S,
    dprop: (S, S) => Double = (n: S, o: S) => 1.0
  ): ((S, Double)) => (S, Double) =
    val r = Uniform(0.0,1.0)
    state =>
      val (x0, ll0) = state
      val x = rprop(x0)
      val ll = logPost(x)
      val a = ll - ll0 + dprop(x0, x) - dprop(x, x0)
      if (math.log(r.draw()) < a)
        (x, ll)
      else
        (x0, ll0)

@main def rwmh() =
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
  def ll(beta: DVD): Double =
      sum(-log(ones + exp(-1.0*(2.0*y - ones)*:*(X * beta))))
  def lprior(beta: DVD): Double =
    Gaussian(0,10).logPdf(beta(0)) + 
      sum(beta(1 until p).map(Gaussian(0,1).logPdf(_)))
  def lpost(beta: DVD): Double =
    ll(beta) + lprior(beta)
  println(lpost(lr.coefficients))
  val pre = DenseVector(10.0,1.0,1.0,1.0,1.0,1.0,5.0,1.0)
  def rprop(beta: DVD): DVD = beta + pre *:* (DenseVector(Gaussian(0.0,0.02).sample(p).toArray))
  val kern = mhKernel(lpost, rprop)
  val s = LazyList.iterate((lr.coefficients, -Inf))(kern) map (_._1)
  val out = s.drop(150).thin(1000).take(10000)
  println("Starting RW MH run now. Be patient...")
  //out.zipWithIndex.foreach(println)
  val aa: Array[Array[Double]] = out.toArray.map(_.toArray)
  val odf = smile.data.DataFrame.of(aa)
  print(odf)
  //smile.write.arrow(odf, "fit-bayes.arrow")
  smile.write.csv(odf, "fit-bayes.csv")
  Mcmc.summary(out,true)
  println("Done.")
