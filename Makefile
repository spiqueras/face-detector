PUGIXMLDIR = ./lib/pugixml-1.2/src/

INCLUDE_DIRS = -I./headers -I$(PUGIXMLDIR)

CFLAGS= -std=c++0x -O3 -L/usr/X11R6/lib -lm -lpthread -lX11 -Wall $(INCLUDE_DIRS)
DCFLAGS= -std=c++0x -g -L/usr/X11R6/lib -lm -lpthread -lX11 -Wall $(INCLUDE_DIRS)

CDIR= ./src/
CFILES= $(CDIR)main.cpp $(CDIR)Classifier.cpp $(CDIR)Image.cpp $(CDIR)StrongClassifier.cpp $(CDIR)CascadeClassifier.cpp $(CDIR)ImageReader.cpp $(PUGIXMLDIR)pugixml.cpp

BINDIR= ./bin/
GPP=g++-4.7

all: $(CFILES)
	$(GPP) -o $(BINDIR)train $(CFILES) $(CFLAGS) 

debug: $(CFILES)
	$(GPP) -o $(BINDIR)traindebug $(CFILES) $(DCFLAGS)

parallel: $(CPFILES)
	$(GPP) -o $(BINDIR)trainp $(CFILES) $(CFLAGS) -fopenmp 

