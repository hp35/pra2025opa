#
# Makefile for compilation of reports written in plain TeX.
#
# Copyright (C) 2025, Fredrik Jonsson, under GPL 3.0. See enclosed LICENSE.
#
FIGURES = $(wildcard metapost/*.pdf)
PROJECT = bwopatheory

all: $(PROJECT).pdf

$(PROJECT).pdf: $(PROJECT).ps
	ps2pdf $(PROJECT).ps $(PROJECT).pdf

$(PROJECT).ps: $(PROJECT).dvi
	dvips -D1200 -ta4 $(PROJECT).dvi -o $(PROJECT).ps

$(PROJECT).dvi: $(PROJECT).tex
	tex $(PROJECT).tex
	tex $(PROJECT).tex

archive:
	make -ik clean
	tar --directory=../ -cf ../$(PROJECT).tar $(PROJECT)

codepdf:
	for file in "python/graphs" "python/rectfourier" ; do \
		echo $$file.py ;\
		enscript -E -q -Z -p - -f Courier10 --highlight=python \
			--color --line-numbers $$file.py \
			| ps2pdf - $$file.pdf ;\
	done

clean:
	rm -Rf *~ *.aux *.toc *.log *.dvi *.ps *.tar.gz
