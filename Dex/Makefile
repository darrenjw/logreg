# Makefile

LIBPATH=BUILTIN_LIBRARIES:.

FORCE:
	make djwutils.html
	make fit-bayes.out
	make fit-ul.out
	make fit-ul-ad.out
	make fit-mala.out
	make fit-mala-ad.out
	make fit-hmc.out
	make fit-hmc-ad.out

%.tsv: %.dx djwutils.dx
	time dex --lib-path $(LIBPATH) script $<

%.html: %.dx djwutils.dx
	dex --lib-path $(LIBPATH) -O script $< --outfmt html > $@

%.out: %.tsv
	./analyse.R $< > $@
	mv Rplots.pdf $@.pdf

edit:
	emacs Makefile *.md *.R *.py *.dx &

clean:
	rm -f *.out *.tsv *~ *.html *.pdf

repl:
	dex --lib-path $(LIBPATH) repl

sandpit:
	dex --lib-path $(LIBPATH) script sandpit.dx


.PRECIOUS: %.html
%: %.html
	make $<
	xdg-open $<



# eof

