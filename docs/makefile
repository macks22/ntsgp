CONVERT=pandoc -s -S

week1: week1-data-munging-and-basic-svd.md
	$(CONVERT) \
		week1-data-munging-and-basic-svd.md \
		-o week1-data-munging-and-basic-svd.pdf

week2: week2-temporal-aspects.md
	$(CONVERT) \
		week2-temporal-aspects.md \
		-o week2-temporal-aspects.pdf

week3: ieee.csl week3.bib week3-the-prediction-task-and-cold-start.md
	$(CONVERT) \
		week3-the-prediction-task-and-cold-start.md \
		--filter pandoc-citeproc \
		-o week3-the-prediction-task-and-cold-start.pdf
