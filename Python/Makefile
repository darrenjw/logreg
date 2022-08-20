# Makefile

FILES=$(wildcard *.py)
OUT=$(patsubst %.py,%.out,$(FILES))

FORCE:
	make $(OUT)

%.parquet: %.py
	time ./$<

%.out: %.parquet
	./analyse.R $< > $@
	mv Rplots.pdf $@.pdf

edit:
	emacs Makefile *.md *.R *.py &

clean:
	rm -f *.out *~ *.pdf *.png


# eof

