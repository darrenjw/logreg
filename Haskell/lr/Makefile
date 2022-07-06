# Makefile


FORCE:
	make build
	make rwmh.out
	make rwmhP.out
	make rwmhPS.out
	make mala.out
	make malaAd.out
	make hmc.out

build:
	stack build

rwmh.mat: app/Rwmh.hs
	make build
	time stack exec lr-exe rwmh

rwmhP.mat: app/RwmhP.hs
	make build
	time stack exec lr-exe rwmhP

rwmhPS.mat: app/RwmhPS.hs
	make build
	time stack exec lr-exe rwmhPS

mala.mat: app/Mala.hs
	make build
	time stack exec lr-exe mala

malaAd.mat: app/MalaAd.hs
	make build
	time stack exec lr-exe malaAd

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
	rm -f *.mat *.out *.pdf

# eof
