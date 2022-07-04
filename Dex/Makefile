# Makefile

FORCE:
	make fit-bayes.out
	make fit-mala.out
	make fit-mala-ad.out


%.tsv: %.dx
	time dex script $<

%.out: %.tsv
	./analyse.R $< > $@
	mv Rplots.pdf $@.pdf

edit:
	emacs Makefile *.md *.R *.py *.dx &

clean:
	rm -f *.out *.tsv *~ *.pdf

# eof

