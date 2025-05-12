
name = presentation
lang = en_US

.PHONY: all
all: $(name).pdf

$(name).toc: $(name).tex
	pdflatex -halt-on-error $<

$(name).pdf: $(name).tex $(name).toc
	pdflatex -halt-on-error $<

.PHONY: spellcheck
spellcheck: $(name).tex
	aspell check --mode=tex --lang=$(lang) $<

.PHONY: clean
clean:
	rm --force -- ./*.{aux,bbl,bcf,blg,lof,log,lol,lot,out,tdo,toc,pdf,toc,xml,nav,snm,vrb}

