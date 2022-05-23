name := "logreg"

version := "0.1-SNAPSHOT"

scalacOptions ++= Seq(
  "-unchecked", "-deprecation", "-feature"
)

libraryDependencies  ++= Seq(
  "org.scalatest" %% "scalatest" % "3.2.9" % "test",
  //"org.scalanlp" %% "breeze" % "2.0",
  "org.scalanlp" %% "breeze-viz" % "2.0",
  //"org.scalanlp" %% "breeze-natives" % "2.0",
  "org.scala-lang.modules" %% "scala-parallel-collections" % "1.0.4",
  "com.github.darrenjw" %% "scala-smfsb" % "0.9",
  "com.github.darrenjw" %% "scala-glm" % "0.7",
  ("com.stripe" %% "rainier-core" % "0.3.5").cross(CrossVersion.for3Use2_13),
  ("com.github.haifengl" %% "smile-scala" % "2.6.0").cross(CrossVersion.for3Use2_13),
  "org.apache.parquet" % "parquet-hadoop" % "1.10.1",  // for parquet
  "org.apache.hadoop" % "hadoop-common" % "3.1.4"      // for parquet
)

resolvers ++= Seq(
  "Sonatype Snapshots" at
    "https://oss.sonatype.org/content/repositories/snapshots/",
  "Sonatype Releases" at
    "https://oss.sonatype.org/content/repositories/releases/"
)

scalaVersion := "3.1.1"

