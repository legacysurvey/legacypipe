#
# Template Makefile for use with desiInstall.  You can assume that
# desiInstall will set these environment variables:
#
# WORKING_DIR   : The directory containing the svn export
# INSTALL_DIR   : The directory the installed product will live in.
# (PRODUCT)     : Where (PRODUCT) is replaced with the name of the
#                 product in upper case, e.g. DESITEMPLATE.  This should
#                 be the same as WORKING_DIR for typical installs.
#
# Use this shell to interpret shell commands, & pass its value to sub-make
#
export SHELL = /bin/sh
#
# This is like doing 'make -w' on the command line.  This tells make to
# print the directory it is in.
#
MAKEFLAGS = w
#
# This is a list of subdirectories that make should descend into.  Makefiles
# in these subdirectories should also understand 'make all' & 'make clean'.
# This list can be empty, but should still be defined.
#
SUBDIRS = src
#
# This is a list of directories that make should copy to $INSTALL_DIR.
# If a Makefile is present in these directories, 'make install' will be
# called on them.  Otherwise it will just be a plain copy.
#
INSTALLDIRS = pro src
#
# This is a message to make that these targets are 'actions' not files.
#
.PHONY : doc all install clean
#
# This will compile Doxygen docs.
#
doc :
	@ if test -f doc/Doxygen.Makefile; then $(MAKE) -C doc -f Doxygen.Makefile all; fi
#
# This should compile all code prior to it being installed.
#
all : doc
	@ for f in $(SUBDIRS); do if test -f $$f/Makefile; then $(MAKE) -C $$f all; fi; done
#
# This should handle the installation of files in $INSTALL_DIR.  Note that
# 'all' is a dependency of 'install'.
#
install : all
	@ for f in $(INSTALLDIRS); do \
		if test -f $$f/Makefile; then $(MAKE) -C $$f install; else \
			if test -d $(WORKING_DIR)/$$f -a ! -d $(INSTALL_DIR)/$$f; then \
				/bin/cp -Rvf $(WORKING_DIR)/$$f $(INSTALL_DIR); fi; fi; done
	@ if test -f doc/html; then \
		if ! test -f $(INSTALL_DIR)/doc/html; then \
			/bin/mkdir -p $(INSTALL_DIR)/doc/html; fi; \
		/bin/cp -Rvf $(WORKING_DIR)/doc/html $(INSTALL_DIR)/doc/html/doxygen; fi
#
# GNU make pre-defines $(RM).  The - in front of $(RM) causes make to
# ignore any errors produced by $(RM).
#
clean :
	- $(RM) *~ core
	@ for f in $(SUBDIRS); do $(MAKE) -C $$f clean ; done
