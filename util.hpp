#ifndef UTIL_HPP
#define UTIL_HPP

#include "cnn/nodes.h"
#include "cnn/cnn.h"
#include "cnn/training.h"
#include "cnn/timing.h"
#include "cnn/rnn.h"
#include "cnn/gru.h"
#include "cnn/lstm.h"
#include "cnn/dict.h"
#include "cnn/expr.h"
#include "cnn/tensor.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <thread>
#include <cmath>

#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/filesystem.hpp>
#include <boost/random/discrete_distribution.hpp>
#include <boost/random/mersenne_twister.hpp>

using namespace std;
using namespace cnn;

// ********************************************************
// Predefined information, used for the entire project
// ********************************************************
// Redefined types
typedef vector<int> Sent;
typedef vector<Sent> Doc;
typedef vector<Doc> Corpus;

// hidden states and particles for sampling
typedef vector<int> LatentSeq;
typedef vector<LatentSeq> Particles;
typedef vector<float> Prob;

// Predefined variables
// char DELIM = '='; // Document boundary

// *******************************************************
// load model from a archive file
// *******************************************************
int load_model(string fname, Model& model);

// *******************************************************
// save model from a archive file
// *******************************************************
int save_model(string fname, Model& model);

// *******************************************************
// save dict from a archive file
// *******************************************************
int save_dict(string fname, cnn::Dict d);

// *******************************************************
// load dict from a archive file
// *******************************************************
int load_dict(string fname, cnn::Dict& d);

// *******************************************************
// read sentences and convect tokens to indices
// *******************************************************
Sent MyReadSentence(const std::string& line, Dict* sd, 
		    bool update);

// *****************************************************
// 
// *****************************************************
Doc makeDoc();

// *****************************************************
// read training and dev data
// *****************************************************
Corpus readData(char* filename, cnn::Dict* dptr, 
		bool b_update = true);


// ******************************************************
// Convert 1-D tensor to vector<float>
// so we can create an expression for it
// ******************************************************
vector<float> convertT2V(const Tensor& t);

// ******************************************************
// Check the directory, if doesn't exist, create one
// ******************************************************
int check_dir(string path);

// ******************************************************
// Get a random number from the given prob
// ******************************************************
vector<int> get_randnums(Prob p, int count);

// ******************************************************
// Segment a long document into several short ones
// ******************************************************
Corpus segment_doc(Corpus doc, int thresh);

// ******************************************************
// Split relation indices from one Doc instance
// ******************************************************
int split_relaidx(Doc& doc, LatentSeq& obsseq);

// *******************************************************
// Infer latent variable distribution from particles
// *******************************************************
vector<Prob> get_emdist(Particles particles, unsigned nlatval);

// *******************************************************
// Normalize vector
// *******************************************************
int normalize_vector(vector<float>& in_vec);

// *******************************************************
// Get argmax
// *******************************************************
int argmax(const vector<float>& vec);

// *******************************************************
// Print out particles
// *******************************************************
int printparticles(const Particles& particles, const LatentSeq& latseq);

// *******************************************************
// L2 norm
// *******************************************************
float l2_norm(const vector<float>& vec);

// ********************************************************
// Normalize vector in the log scale
// ********************************************************
int log_normalize(vector<float>& vec);

// ********************************************************
// Compute prediction accuracy
// ********************************************************
int count_prediction(const LatentSeq& obsseq,
		     const LatentSeq& decodedseq,
		     float& total, float& correct);

#endif
