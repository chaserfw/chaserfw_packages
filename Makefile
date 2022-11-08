PIP_CMD = pip

all: utilities.o preprocessing.o math.o

utilities.o:
	$(PIP_CMD) install -e ./utilities/

preprocessing.o:
	$(PIP_CMD) install -e ./preprocessing/

math.o:
	$(PIP_CMD) install -e ./math/