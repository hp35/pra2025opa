#
# Makefile designed for use with gcc
#
# Copyright (C) 1996-2025 under GPLv3, Fredrik Jonsson
#

#
# TeX-specific / document preparing declarations:
#
TEX       = tex
DVIPS     = dvips
DVIPSOPTS = -D1200 -E
METAPOST  = mpost
PS2PDF    = ps2pdf
PDFCROP   = pdfcrop
PDF2SVG   = pdf2svg

#
# The POINCARE program for mapping paths of Stokes parameters onto the
# Poincare sphere. For the latest version of this special program, see
# https://github.com/hp35/poincare
#
POINCAREPATH = ../../../2025-pulsedopa/poincare/
POINCARE = poincare

all: graph-01-poincare graph-01-teaser

graph-01-poincare:
	make -C $(POINCAREPATH)
	@for delta in "0.50" "1.00" "1.50" ; do \
	   srcname=data-01-delta-$$delta-beta ;\
	   fname=graph-01-delta-$$delta-poincare ;\
	   cat ../python/data/$$srcname*.dat > $$fname.dat ;\
	   $(POINCAREPATH)/$(POINCARE) --verbose --normalize \
		--draw_hidden_dashed \
		--inputfile $$fname.dat \
		--outputfile $$fname.mp \
		--axislengths 0.3 1.55 0.3 2.45 0.3 1.35 \
		--axislabels  "S_1(z)/S_0(z)" bot \
				"S_2(z)/S_0(z)" bot \
				"S_3(z)/S_0(z)" rt \
		--rotatephi 15.0 --rotatepsi -60.0 --shading 0.75 0.99 \
		--rhodivisor 50  --phidivisor 80  --scalefactor 26.0 \
		--draw_paths_as_arrows \
		--paththickness 0.8 --arrowthickness 0.4 \
		--auxsource stokepathlabels.mp ;\
	   $(METAPOST) $$fname.mp ;\
	   $(TEX) -jobname=$$fname "\input epsf.tex\nopagenumbers\
		\centerline{\epsfxsize=155mm\epsfbox{"$$fname".1}}\bye" ;\
	   $(DVIPS) $(DVIPSOPTS) $$fname -o $$fname.eps ;\
	   $(PS2PDF) $$fname.eps $$fname.pdf ;\
	   $(PDFCROP) $$fname.pdf $$fname-crop.pdf ;\
	   $(PDF2SVG) $$fname-crop.pdf $$fname.svg ;\
	done

clean:
	-rm -f *~ *.aux *.dvi *.log *.1 *.mpx *.dat graph*
