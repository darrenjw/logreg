# Makefile


FORCE:
	make build
	make rwmh.out
	make rwmhP.out
	make mala.out
	make hmc.out

build:
	stack build

rwmh.mat: app/Rwmh.hs
	make build
	time stack exec lr-exe rwmh

rwmhP.mat: app/RwmhP.hs
	make build
	time stack exec lr-exe rwmhP

mala.mat: app/Mala.hs
	make build
	time stack exec lr-exe mala

hmc.mat: app/Hmc.hs
	make build
	time stack exec lr-exe hmc

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
