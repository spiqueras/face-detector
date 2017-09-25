PUGIXMLDIR = ../../lib/pugixml-1.2/src/

INCLUDE_DIRS = -I../../headers -I$(PUGIXMLDIR)

CFLAGS= -std=c++0x -O3 -L/usr/X11R6/lib -lm -lpthread -lX11 -Wall $(INCLUDE_DIRS)
DCFLAGS= -std=c++0x -g -L/usr/X11R6/lib -lm -lpthread -lX11 -Wall $(INCLUDE_DIRS)

CDIR= ../
CFILES= main.cpp $(CDIR)Classifier.cpp $(CDIR)Image.cpp $(CDIR)StrongClassifier.cpp $(CDIR)ImageReader.cpp $(CDIR)CascadeClassifier.cpp $(PUGIXMLDIR)pugixml.cpp

BINDIR= ../../bin/
GPP=g++-4.7

all: $(CFILES)
	$(GPP) -o $(BINDIR)detector $(CFILES) $(CFLAGS)

debug: $(CFILES)
	$(GPP) -o $(BINDIR)detector_debug $(CFILES) $(DCFLAGS)

parallel: $(CPFILES)
	$(GPP) -o $(BINDIR)detectorp $(CFILES) $(CFLAGS) -fopenmp
