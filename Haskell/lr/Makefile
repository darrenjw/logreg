# Makefile


FORCE:
	make build
	make rwmh.out
	make rwmhP.out

build:
	stack build

rwmh.mat:
	make build
	time stack exec lr-exe rwmh

rwmhP.mat:
	make build
	time stack exec lr-exe rwmhP

%.out: %.mat
	./analyse.R $< > $@
	cp Rplots.pdf $@.pdf


edit:
	emacs *.yaml *.md *.hs *.R Makefile src/*.hs app/*.hs &

repl:
	stack ghci

clean:
	rm -f *.mat *.out

# eof
