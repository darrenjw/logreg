# Scala code for MCMC for logistic regression

The Scala examples require a recent JVM and [sbt](https://www.scala-sbt.org/) to run. Typing `sbt run` from the `lr` subdirectory (not this directory) should download missing dependencies and run the examples.

See the `sbt` web page for installation instructions. But also note that [coursier](https://get-coursier.io/) can be used to install a complete Scala development environment (including JVMs, `sbt`, the Scala compiler, etc.), so that is also worth considering.

To run a particular example, do, eg.
```bash
sbt "runMain rwmh"
```
Note that `sbt` is also designed to be used interactively. eg. do `sbt` to get an `sbt` prompt, and then type `run` at the `sbt` prompt.

## Spark example

The [Apache Spark](https://spark.apache.org/) example requires a Spark installation in addition to `sbt`. Running `sbt assembly` from the `spark` subdirectory will produce a `jar` that can be submitted to a Spark cluster using `spark-submit`. See the Spark docs for more information on installing and using Spark clusters.

Note that Spark is intended for working with very large datasets. On small datasets it will be *much* slower than non-Spark Scala code.

## Learning more about Scala

If you want to learn more about [Scala](https://www.scala-lang.org/), the (free) on-line video series, [Scala at light speed](https://www.youtube.com/playlist?list=PLmtsMNDRU0BxryRX4wiwrTZ661xcp6VPM) from [Rock the JVM](https://rockthejvm.com/) is quite a good place to start. For more on scientific and statistical computing, follow up with my on-line course, [Scala for statistical computing and data science](https://github.com/darrenjw/scala-course/blob/master/StartHere.md).

