# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = source
BUILDDIR      = build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

clean:
	@rm -rf build source/generated/*
	@rm -rf .jupyter_cache
	@find source/ -type d -name "_plots" -print0 | while IFS= read -r -d '' dir; do \
		echo "Cleaning $$dir"; \
		rm -rf "$$dir"/* "$$dir"/.[^.]* "$$dir"/..?* 2>/dev/null || true; \
	done

apidoc:
	@sphinx-apidoc -f -o source ../src/archimedes ../src/archimedes/experimental/*