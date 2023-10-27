MAIN=paper
LATEX=latexmk -pdf -shell-escape -halt-on-error

.PHONY:all clean cleaner pvc ${MAIN}.pdf

all: ${MAIN}.pdf

${MAIN}.pdf: ${MAIN}.tex
	${LATEX} ${MAIN}.tex

pvc:
	${LATEX} -pvc ${MAIN}.tex

clean:
	${LATEX} -c
	rm -f *.nav *.snm

cleaner: clean
	${LATEX} -C

