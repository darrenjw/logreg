# Makefile

CFLAGS=-g -O2 -Wall -ansi -pedantic
LDFLAGS=-lgsl -lgslcblas -lm

FORCE: fit-bayes.ex
	time ./fit-bayes.ex > fit-bayes.out

%.ex: %.o
	gcc $(CFLAGS) -o $@ $< $(LDFLAGS)

clean:
	rm -f *.o *.ex *~ *.bak core

edit:
	emacs Makefile *.md *.c &


# eof
