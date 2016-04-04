#include "util.hpp"

boost::mt19937 gen;

// *******************************************************
// load model from a archive file
// *******************************************************
int load_model(string fname, Model& model){
  ifstream in(fname + ".model");
  boost::archive::text_iarchive ia(in);
  ia >> model; in.close();
  return 0;
}

// *******************************************************
// save model from a archive file
// *******************************************************
int save_model(string fname, Model& model){
  ofstream out(fname + ".model");
  boost::archive::text_oarchive oa(out);
  oa << model; out.close();
  return 0;
}

// *******************************************************
// save dict from a archive file
// *******************************************************
int save_dict(string fname, cnn::Dict d){
  fname += ".dict";
  ofstream out(fname);
  boost::archive::text_oarchive odict(out);
  odict << d; out.close();
  return 0;
}

// *******************************************************
// load dict from a archive file
// *******************************************************
int load_dict(string fname, cnn::Dict& d){
  fname += ".dict";
  ifstream in(fname);
  boost::archive::text_iarchive ia(in);
  ia >> d; in.close();
  return 0;
}

// *******************************************************
// read sentences and convect tokens to indices
// *******************************************************
Sent MyReadSentence(const std::string& line, 
		    Dict* sd, 
		    bool update) {
  vector<string> strs, items;
  int ridx;
  string text;
  boost::split(items, line, boost::is_any_of("\t"));
  if (items.size() == 2){
    text = items[0];
    ridx = std::stoi(items[1]); // string to int
  } else if (items.size() == 1){
    text = items[0];
    ridx = -1;
  } else {
    cerr << "unrecognized data format\n\t "
	 << line << endl;
    abort();
  }
  boost::split(strs, text, boost::is_any_of(" "));
  // istringstream in(line);
  // string word;
  Sent res;
  res.push_back(sd->Convert("<s>"));
  for (auto& word : strs){
    if (word.empty()) break;
    // cerr << "word = " << word << endl;
    if (update){
      res.push_back(sd->Convert(word));
    } else {
      if (sd->Contains(word)){
	res.push_back(sd->Convert(word));
      }else{
	res.push_back(sd->Convert("UNK"));
      }
    }
  }
  res.push_back(sd->Convert("</s>"));
  // push back relation index
  res.push_back(ridx);
  return res;
}

// *****************************************************
// 
// *****************************************************
Doc makeDoc(){
  vector<vector<int>> doc;
  return doc;
}

// *****************************************************
// read training and dev data
// *****************************************************
Corpus readData(char* filename, 
		cnn::Dict* dptr,
		bool b_update){
  cerr << "Reading data from "<< filename << endl;
  Corpus corpus;
  Doc doc;
  Sent sent;
  string line;
  int tlc = 0;
  int toks = 0;
  ifstream in(filename);
  while(getline(in, line)){
    ++tlc;
    if (line[0] != '='){
      sent = MyReadSentence(line, dptr, b_update);
      if (sent.size() > 0){
	doc.push_back(sent);
	toks += doc.back().size();
      } else {
	cerr << "Empty sentence: " << line << endl;
      }
    } else {
      if (doc.size() > 0){
	corpus.push_back(doc);
	doc = makeDoc();
      } else {
	cerr << "Empty document " << endl;
      }
    }
  }
  if (doc.size() > 0){
    corpus.push_back(doc);
  }
  cerr << corpus.size() << " docs, " << tlc << " lines, " 
       << toks << " tokens, " << dptr->size() 
       << " types." << endl;
  return(corpus);
}

// ******************************************************
// Convert 1-D tensor to vector<float>
// so we can create an expression for it
// ******************************************************
vector<float> convertT2V(const Tensor& t){
  vector<float> vf;
  int dim = t.d.d[0];
  for (int idx = 0; idx < dim; idx++){
    vf.push_back(t.v[idx]);
  }
  return vf;
}

// ******************************************************
// Check the directory, if doesn't exist, create one
// ******************************************************
int check_dir(string path){
  boost::filesystem::path dir(path);
  if(!(boost::filesystem::exists(dir))){
    if (boost::filesystem::create_directory(dir)){
      std::cout << "....Successfully Created !" << "\n";
    }
  }
  return 0;
}

// ******************************************************
// Generate sample for the given prob dist
// ******************************************************
vector<int> get_randnums(Prob p, int count){
  boost::random::discrete_distribution<> dist(p);
  vector<int> randnums;
  for (int i = 0; i < count; i++)
    randnums.push_back(dist(gen));
  return randnums;
}

// ******************************************************
// Segment a long document into several short ones
// ******************************************************
Corpus segment_doc(Corpus corpus, int thresh){
  Corpus newcorpus;
  for (auto& doc : corpus){
    if (doc.size() <= thresh){
      newcorpus.push_back(doc);
      continue;
    }
    Doc tmpdoc;
    int counter = 0;
    for (auto& sent : doc){
      if (counter < thresh){
	tmpdoc.push_back(sent);
	counter ++;
      } else {
	newcorpus.push_back(tmpdoc);
	tmpdoc.clear();
	tmpdoc.push_back(sent);
	counter = 1;
      }
    }
    if (tmpdoc.size() > 0){
      newcorpus.push_back(tmpdoc);
      tmpdoc.clear();
    }
  }
  return newcorpus;
}

// ******************************************************
// Split relation indices from one Doc instance
// ******************************************************
int split_relaidx(Doc& doc, LatentSeq& obsseq){
  // clean obsseq
  obsseq.clear();
  // split
  for (unsigned sidx = 0; sidx < doc.size(); sidx ++){
    // get the last element
    int ridx = doc[sidx].back();
    // store ridx
    obsseq.push_back(ridx);
    // remove the last element from current sentence
    doc[sidx].pop_back();
  }
  return 0;
}


// *******************************************************
// Infer latent variable distribution from particles
// *******************************************************
vector<Prob> get_emdist(Particles particles, unsigned nlatval){
  vector<Prob> vec_prob;
  // initialize
  for (int i = 0; i < particles[0].size(); i++){
    Prob prob = Prob(nlatval, 0.0);
    vec_prob.push_back(prob);
  }
  // summarize
  for (auto& particle : particles){
    for (int i = 0; i < particle.size(); i++){
      int p = particle[i];
      vec_prob[i][p] += 1.0;
    }
  }
  // normalize
  for (auto& prob : vec_prob){
    normalize_vector(prob);
  }
  // return
  return vec_prob;
}

// *******************************************************
// Normalize vector
// *******************************************************
int normalize_vector(vector<float>& in_vec){
  // sum over
  float sum = 0.0;
  for (auto& elem : in_vec){
    sum += elem;
  }
  // normalize
  for (int i = 0; i < in_vec.size(); i++){
    in_vec[i] = (in_vec[i] / sum);
  }
  // return
  return 0;
}

// *******************************************************
// Get argmax
// *******************************************************
int argmax(const vector<float>& vec){
  int max_idx = -1;
  float max_val = -1e+30;
  for (int i = 0; i < vec.size(); i++){
    if (vec[i] > max_val){
      max_idx = i; max_val = vec[i];
    }
  }
  return max_idx;
}

// *******************************************************
// Print out particles
// *******************************************************
int printparticles(const Particles& particles, const LatentSeq& latseq){
  cout << "===========" << endl;
  cout << "0 : ";
  for (auto& v : latseq) cout << v << " "; cout << endl;
  cout << "-----------" << endl;
  int idx = 1;
  for (auto& seq : particles){
    cout << idx << " : ";
    for (auto& v : seq){
      cout << v << " ";
    }
    cout << endl;
    idx ++;
  }
  return 0;
}

// *******************************************************
// L2 norm
// *******************************************************
float l2_norm(const vector<float>& vec){
  float sqsum = 0.0;
  for (auto& v : vec) sqsum += (v * v);
  return sqrt(sqsum);
}

// ********************************************************
// Normalize vector in the log scale
// ********************************************************
int log_normalize(vector<float>& vec){
  float maxval = 0.0, sum = 0.0, logsum=0.0;
  // get maximum val
  maxval = vec[0];
  for (int i = 1; i < vec.size(); i ++)
    if (vec[i] > maxval)
      maxval = vec[i];
  // get log sum
  for (int i = 0; i < vec.size(); i ++)
    sum += exp(vec[i] - maxval);
  logsum = log(sum) + maxval;
  // cout << "logsum = " << logsum << endl;
  // normalize in log scale
  for (int i = 0; i < vec.size(); i++)
    vec[i] = vec[i] - logsum;
  return 0;
}

// ********************************************************
// Compute prediction accuracy
// ********************************************************
int count_prediction(const LatentSeq& obsseq,
		       const LatentSeq& decodedseq,
		       float& total, float& correct){
  assert(obsseq.size() == decodedseq.size());
  for (int idx = 0; idx < obsseq.size(); idx ++){
    if (obsseq[idx] >= 0) total += 1.0;
    if (obsseq[idx] == decodedseq[idx]) correct += 1.0;
  }
  return 0;
}
