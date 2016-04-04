#ifndef TRAINING_HPP
#define TRAINING_HPP

#include "output.hpp"
#include "hidden.hpp"
#include "baseline.hpp"
#include "util.hpp"

int train(char* ftrn,
	  char* fdev,
	  unsigned nlayers = 2, 
	  unsigned inputdim = 16,
	  unsigned hiddendim = 48, 
	  string flag = "output",
	  string expfolder="tmp",
	  float lr0 = 0.1,
	  float reg = 0.00001,
	  bool use_adagrad = false,
	  unsigned nparticle = 10,
	  int docthresh = 5,
	  bool use_observed = true,
	  unsigned nlatvar = 5,
	  int reportfreq = 20,
	  bool use_dropout = true,
	  string fmodel = "");

#endif
