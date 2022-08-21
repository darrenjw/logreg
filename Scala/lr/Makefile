# Makefile

LIBPATH=BUILTIN_LIBRARIES:.

FORCE:
	make fit-bayes.out
	make fit-nopar.out
	make fit-par.out
	make fit-ul.out
	make fit-mala.out
	make fit-hmc.out

fit-bayes.csv: src/main/scala/fit-bayes.scala
	sbt "runMain rwmh"

fit-nopar.csv: src/main/scala/fit-nopar.scala
	sbt "runMain nopar"

fit-par.csv: src/main/scala/fit-par.scala
	sbt "runMain par"

fit-ul.csv: src/main/scala/fit-ul.scala
	sbt -mem 6000 "runMain ul"

fit-mala.csv: src/main/scala/fit-mala.scala
	sbt "runMain mala"

fit-hmc.csv: src/main/scala/fit-hmc.scala
	sbt "runMain hmc"

%.out: %.csv
	./analyse.R $< > $@
	mv Rplots.pdf $@.pdf

edit:
	emacs Makefile *.md *.R src/main/scala/*.scala &

clean:
	rm -f *.out *.csv *~ *.pdf


# eof

