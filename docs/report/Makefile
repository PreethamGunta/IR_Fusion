# Compile the report with a normal TeX installation.
# The shipped PDF was built with Tectonic 0.17.0 (`tectonic -X compile main.tex`).
all: main.pdf

main.pdf: main.tex $(wildcard figures/*.png)
	pdflatex -interaction=nonstopmode main.tex
	pdflatex -interaction=nonstopmode main.tex

clean:
	rm -f main.aux main.log main.out main.toc main.pdf

.PHONY: all clean
