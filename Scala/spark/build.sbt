name := "spark"

version := "0.1-SNAPSHOT"

scalacOptions ++= Seq(
  "-unchecked", "-deprecation", "-feature"
)

libraryDependencies  ++= Seq(
  "org.scalatest" %% "scalatest" % "3.2.9" % "test",
  "org.apache.spark" %% "spark-core" % "3.2.1" % Provided,
  "org.apache.spark" %% "spark-sql" % "3.2.1" % Provided,
  "org.apache.spark" %% "spark-mllib" % "3.2.1" % Provided,
  //"org.scalanlp" %% "breeze" % "2.0",
  "org.scalanlp" %% "breeze-viz" % "1.0",
  //"org.scalanlp" %% "breeze-natives" % "2.0",
  "com.github.darrenjw" %% "scala-smfsb" % "0.7"
)

resolvers ++= Seq(
  "Sonatype Snapshots" at
    "https://oss.sonatype.org/content/repositories/snapshots/",
  "Sonatype Releases" at
    "https://oss.sonatype.org/content/repositories/releases/"
)

scalaVersion := "2.12.15"

