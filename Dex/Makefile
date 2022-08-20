# Makefile

LIBPATH=BUILTIN_LIBRARIES:.

FORCE:
	make fit-bayes.out
	make fit-ul.out
	make fit-ul-ad.out
	make fit-mala.out
	make fit-mala-ad.out
	make fit-hmc.out
	make fit-hmc-ad.out

%.tsv: %.dx djwutils.dx
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

