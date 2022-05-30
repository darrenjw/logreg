// fit-spark.scala

import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature.RFormula
import org.apache.spark.ml.linalg.{DenseVector, Vectors}
import org.apache.spark.sql.functions.udf

import smfsb._
import breeze.stats.distributions.{Gaussian, Uniform}
import breeze.linalg.{DenseVector => BDV, _}
import breeze.numerics._

object FitSpark {

  type DVD = BDV[Double]

  def mhKernel(logPost: DVD => Double, rprop: DVD => DVD,
    dprop: (DVD, DVD) => Double = (n, o) => 1.0): ((DVD, Double)) => (DVD, Double) = {
      val r = Uniform(0.0,1.0)
      state => {
        val (x0, ll0) = state
        val x = rprop(x0)
        val ll = logPost(x)
        val a = ll - ll0 + dprop(x0, x) - dprop(x, x0)
        if (math.log(r.draw()) < a)
          (x, ll)
        else
          (x0, ll0)
      }
  }

  def toBDV(v: DenseVector): DVD = v match {
    case DenseVector(vals) => BDV[Double](vals)
  }

  def toDV(v: DVD): DenseVector = v match {
    case v: DVD => new DenseVector(v.toArray)
  }

  def main(args: Array[String]): Unit = {

    val spark = SparkSession.builder.appName("Fit Spark").getOrCreate()
    import spark.implicits._

    val df = spark.read.parquet("../../pima.parquet")
    def s2d(s: String): Double = if (s=="Yes") 1.0 else 0.0
    def s2d_udf = udf(s2d _)
    val df2 = df.withColumn("y", s2d_udf(df.col("type"))).drop("type")
    val df3 = new RFormula().setFormula("y ~ .").fit(df2).
      transform(df2).select("y", "features")
    println(df3 show 5)
    println(df3.head.getAs[DenseVector](1))
    val p = 8
    def ll(beta: DVD): Double = 
      df3.map{row =>
        val y = row.getAs[Double](0)
        val x = BDV.vertcat(BDV(1.0),toBDV(row.getAs[DenseVector](1)))
        -math.log(1.0 + math.exp(-1.0*(2.0*y-1.0)*(x.dot(beta))))}.reduce(_+_)
    def lprior(beta: DVD): Double =
      Gaussian(0,10).logPdf(beta(0)) +
        sum(beta(1 until p).map(Gaussian(0,1).logPdf(_)))
    def lpost(beta: DVD): Double =
      ll(beta) + lprior(beta)
    val pre = BDV(10.0,1.0,1.0,1.0,1.0,1.0,5.0,1.0)
    def rprop(beta: DoubleState): DoubleState = beta + pre *:* (BDV(Gaussian(0.0,0.02).sample(p).toArray))
    def dprop(n: DoubleState, o: DoubleState): Double = 1.0
    val init = BDV(-9.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0)
    val kern = mhKernel(lpost, rprop)
    val s = Stream.iterate((init, -Inf))(kern) map (_._1)
    val out = s.drop(150).thin(50).take(10000)
    println("Starting RW MH run now. Be patient...")
    out.zipWithIndex.foreach(println)
    Mcmc.summary(out,true)
    println("Done.")

  }

}
