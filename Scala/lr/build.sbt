name := "logreg"

version := "0.1-SNAPSHOT"

scalacOptions ++= Seq(
  "-unchecked", "-deprecation", "-feature"
)

libraryDependencies  ++= Seq(
  "org.scalatest" %% "scalatest" % "3.2.9" % "test",
  "org.scalanlp" %% "breeze-viz" % "2.1.0",
  "org.scala-lang.modules" %% "scala-parallel-collections" % "1.0.4",
  "com.github.darrenjw" %% "scala-smfsb" % "1.0",
  "com.github.darrenjw" %% "scala-glm" % "0.8",
  ("com.github.haifengl" %% "smile-scala" % "3.0.2").cross(CrossVersion.for3Use2_13),
  "org.apache.parquet" % "parquet-hadoop" % "1.13.1",  // for parquet
  "org.apache.hadoop" % "hadoop-common" % "3.3.6"      // for parquet
)


resolvers ++= Seq(
  "Sonatype Snapshots" at
    "https://oss.sonatype.org/content/repositories/snapshots/",
  "Sonatype Releases" at
    "https://oss.sonatype.org/content/repositories/releases/"
)

conflictWarning := ConflictWarning.disable

scalaVersion := "3.3.0"

