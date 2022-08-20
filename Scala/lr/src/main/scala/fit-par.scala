/*
fit-par.scala
Bayesian MCMC for logistic regression model in scala
*/

import smfsb.*
import scalaglm.*
import breeze.linalg.*
import breeze.numerics.*
import breeze.stats.distributions.Gaussian
import breeze.stats.distributions.Rand.FixedSeed.randBasis
import smile.data.pimpDataFrame
import scala.collection.parallel.CollectionConverters.*

@main def par() =
  println("First read and process the data")
  val df = smile.read.parquet("../../pima.parquet")
  print(df)
  val y = df.select("type").
    map(_(0).asInstanceOf[String]).
    map(s => if (s == "Yes") 1.0 else 0.0).toArray.toVector.par
  println(y)
  val names = df.drop("type").names.toList
  println(names)
  val x = df.drop("type").toMatrix.toArray.toVector.par.map(_.toList)
  println(x)
  val X = x.map(r => DenseVector((1.0 :: r).toArray))
  println(X)
  val p = X.head.length
  val Xy = X zip y
  println(Xy)
  def ll(beta: DVD): Double =
    Xy.map{case (x, y) => -math.log(1.0 + math.exp(-1.0*(2.0*y-1.0)*(x.dot(beta))))}.sum
  def lprior(beta: DVD): Double =
    Gaussian(0,10).logPdf(beta(0)) + 
      sum(beta(1 until p).map(Gaussian(0,1).logPdf(_)))
  def lpost(beta: DVD): Double =
    ll(beta) + lprior(beta)
  val pre = DenseVector(10.0,1.0,1.0,1.0,1.0,1.0,5.0,1.0)
  def rprop(beta: DoubleState): DoubleState = beta + pre *:* (DenseVector(Gaussian(0.0,0.02).sample(p).toArray))
  val init = DenseVector(-9.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0)
  val kern = mhKernel(lpost, rprop)
  val s = LazyList.iterate((init, -Inf))(kern) map (_._1)
  val out = s.drop(150).thin(1000).take(10000)
  println("Starting RW MH run now. Be patient...")
  val aa: Array[Array[Double]] = out.toArray.map(_.toArray)
  val odf = smile.data.DataFrame.of(aa)
  smile.write.csv(odf, "fit-par.csv")
  Mcmc.summary(out,false)
  println("Done.")
