# Makefile

# Edit this for the path to your Dex library (containing prelude.dx)
DEXLIB=~/src/git/dex-lang/lib


LIBPATH=$(DEXLIB):.

FORCE:
	make fit-bayes.out
	make fit-mala.out
	make fit-mala-ad.out
	make fit-hmc.out
	make fit-hmc-ad.out

%.tsv: %.dx
	time dex --lib-path $(LIBPATH) script $<

%.out: %.tsv
	./analyse.R $< > $@
	mv Rplots.pdf $@.pdf

edit:
	emacs Makefile *.md *.R *.py *.dx &

clean:
	rm -f *.out *.tsv *~ *.pdf

repl:
	dex --lib-path $(LIBPATH) repl

sandpit:
	dex --lib-path $(LIBPATH) script sandpit.dx


# eof

