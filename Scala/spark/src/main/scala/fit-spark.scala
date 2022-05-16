// fit-spark.scala

import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature.RFormula
import org.apache.spark.ml.linalg.{DenseVector, Vectors}
import org.apache.spark.sql.functions.udf

import smfsb._
import breeze.stats.distributions.Gaussian
import breeze.linalg.{DenseVector => BDV, _}
import breeze.numerics._
//import breeze.stats.distributions.Rand.FixedSeed.randBasis

object FitSpark {

  def toBDV(v: DenseVector): BDV[Double] = v match {
    case DenseVector(vals) => BDV[Double](vals)
  }

  def toDV(v: BDV[Double]): DenseVector = v match {
    case v: BDV[Double] => new DenseVector(v.toArray)
  }

  def main(args: Array[String]): Unit = {

    val spark = SparkSession.builder.appName("Fit Spark").getOrCreate()
    import spark.implicits._

    val df = spark.read.parquet("../../pima.parquet")
    def s2d(s: String): Double = if (s=="Yes") 1.0 else 0.0
    def s2d_udf = udf(s2d _)
    val df2 = df.withColumn("y", s2d_udf(df.col("type"))).drop("type")
    val df3 = new RFormula().setFormula("Type ~ .").fit(df2).
      transform(df2).select("y", "features")
    println(df3 show 5)
    val p = 8
    def ll(beta: BDV[Double]): Double = 
      df3.map{row =>
        val y = row.getAs[Double](0)
        val x = row.getAs[DenseVector](1)
        -math.log(1.0 + math.exp(-1.0*(2.0*y-1.0)*(x.dot(beta))))}.reduce(_+_)
    def lprior(beta: BDV[Double]): Double =
      Gaussian(0,10).logPdf(beta(0)) +
        sum(beta(1 to p).map(Gaussian(0,1).logPdf(_)))
    def lpost(beta: BDV[Double]): Double =
      ll(beta) + lprior(beta)
    val pre = BDV[Double](10.0,1.0,1.0,1.0,1.0,1.0,5.0,1.0)
    def rprop(beta: DoubleState): DoubleState = beta + pre *:* (BDV(Gaussian(0.0,0.02).sample(p).toArray))
    def dprop(n: DoubleState, o: DoubleState): Double = 1.0
    val init = BDV[Double](-9.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0)

    val s = Mcmc.mhStream(init, lpost, rprop, dprop,
      (p: DoubleState) => 1.0, verb = false)
    val out = s.drop(150).thin(10).take(10000)
    println("Starting RW MH run now. Be patient...")
    //out.zipWithIndex.foreach(println)
    Mcmc.summary(out,true)
    println("Done.")

  }

}
