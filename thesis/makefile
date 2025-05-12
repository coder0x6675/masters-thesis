
name = thesis
lang = en_US

.PHONY: all
all: $(name).pdf

$(name).pdf: $(name).tex $(name).bib parts/*.tex
	pdflatex -halt-on-error $<
	biber $(name).bcf
	pdflatex -halt-on-error $<

.PHONY: spellcheck
spellcheck: $(name).tex parts/*.tex
	for file in $^ ; do aspell check --mode=tex --lang=$(lang) "$$file" ; done

.PHONY: clean
clean:
	rm --force -- ./*.{aux,bbl,bcf,blg,lof,log,lol,lot,out,tdo,toc,pdf,toc,xml,bak}
	rm --force -- ./parts/*.{aux,bak}

