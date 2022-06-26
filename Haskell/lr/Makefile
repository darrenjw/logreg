# Makefile


FORCE:
	make rwmh.mat
	make rwmh.out

rwmh.mat: app/Main.hs
	stack build && time stack exec lr-exe

rwmh.out: rwmh.mat
	./analyse.R > rwmh.out


edit:
	emacs *.yaml *.md *.hs *.R Makefile src/*.hs app/*.hs &

repl:
	stack ghci


# eof
