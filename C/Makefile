# Makefile

CFLAGS=-g -O2 -Wall -ansi -pedantic
LDFLAGS=-lgsl -lgslcblas -lm

FORCE: fit-bayes.ex
	time ./fit-bayes.ex > fit-bayes.out
	Rscript analyse.R

%.ex: %.o
	gcc $(CFLAGS) -o $@ $< $(LDFLAGS)

clean:
	rm -f *.o *.ex *~ *.bak *.out *.pdf core

edit:
	emacs Makefile *.md *.c &


# eof
