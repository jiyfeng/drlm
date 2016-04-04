CC=g++
LIBS=-Lcnn/build/cnn -lcnn -lstdc++ -lm -lboost_serialization -lboost_filesystem -lboost_system -lboost_random -lboost_program_options
CFLAGS=-Icnn -Icnn/eigen -I./cnn/external/easyloggingpp/src -std=gnu++11 -O3 -Wunused -Wreturn-type # -Wall
OBJ=util.o training.o main.o test.o

all: lvrnn

%.o: %.cc
	$(CC) $(CFLAGS) -c -o $@ $< 

lvrnn: main.o training.o test.o util.o output.hpp hidden.hpp baseline.hpp
	$(CC) $(CFLAGS) -o $@ $^ $(LIBS)

clean:
	rm -rf *.o *.*~ lvrnn

