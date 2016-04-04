#include "training.hpp"
#include <boost/format.hpp>

// For logging
#define ELPP_NO_DEFAULT_LOG_FILE
#include "easylogging++.h"
INITIALIZE_EASYLOGGINGPP

cnn::Dict d;
int kSOS, kEOS;
string MODELPATH("models/");
string LOGPATH("logs/");

// ********************************************************
// train
// ********************************************************
int train(char* ftrn, char* fdev, unsigned nlayers, 
	  unsigned inputdim, unsigned hiddendim, 
	  string flag, string expfolder, float lr0,
	  float reg, bool use_adagrad, unsigned nparticle,
	  int docthresh, bool use_observed,
	  unsigned nlatvar, int reportfreq,
	  bool use_dropout, string fmodel){
  // initialize logging
  int argc = 1; 
  char** argv = new char* [1];
  START_EASYLOGGINGPP(argc, argv);
  delete[] argv;

  // ---------------------------------------------
  // predefined files
  ostringstream os;
  os << flag << '_' << nlayers
     << '_' << inputdim
     << '_' << hiddendim
     << '_' << lr0
     << '_' << reg
     << '_' << nlatvar
     << '_' << docthresh
     << '_' << use_observed
     << '_' << use_adagrad
     << '_' << use_dropout
     << '_' << nparticle
     << "-pid" << getpid();
  const string fprefix = os.str();
  string fname = MODELPATH + fprefix;
  string flog = LOGPATH + expfolder + "/" + fprefix + ".log";
  // check model path
  check_dir(MODELPATH);

  // ----------------------------------------------
  // Pre-defined constants
  double best = exp(9e+99);
  double bestacc = 0.0;
  unsigned report_every_i = 50;
  unsigned dev_every_i_reports = reportfreq;
  unsigned si = 0; // training index
  unsigned nskip = 0;

  // --------------------------------------------
  // Logging
  el::Configurations defaultConf;
  // defaultConf.setToDefault();
  defaultConf.set(el::Level::Info, 
  		  el::ConfigurationType::Format, 
  		  "%datetime{%b-%d-%h:%m:%s} %level %msg");
  defaultConf.set(el::Level::Info, 
  		  el::ConfigurationType::Filename, flog.c_str());
  el::Loggers::reconfigureLogger("default", defaultConf);
  LOG(INFO) << "[lvrnn] Training data: " << ftrn;
  LOG(INFO) << "[lvrnn] Dev data: " << fdev;
  LOG(INFO) << "[lvrnn] Latent values: " << nlatvar;
  LOG(INFO) << "[lvrnn] Input dim: " << inputdim;
  LOG(INFO) << "[lvrnn] Hidden dim: " << hiddendim;
  LOG(INFO) << "[lvrnn] Layers of RNN: " << nlayers;
  LOG(INFO) << "[lvrnn] Model flag: " << flag;
  LOG(INFO) << "[lvrnn] Initial learning rate: " << lr0;
  LOG(INFO) << "[lvrnn] Regularization parameter: " << reg;
  LOG(INFO) << "[lvrnn] Document length threshold: " << docthresh;
  LOG(INFO) << "[lvrnn] Trainer: " << (use_adagrad ? "AdaGrad" : "SimpleSGD");
  LOG(INFO) << "[lvrnn] With Dropout: " << (use_dropout ? "True" : "False");
  LOG(INFO) << "[lvrnn] With observation: " << (use_observed ? "True" : "False");
  LOG(INFO) << "[lvrnn] Pretrained model: " << (fmodel.size() ? fmodel : "None");
  LOG(INFO) << "[lvrnn] Parameters files: " << fname;

  // ---------------------------------------------
  if (((flag == "outputrela")||(flag == "baselinerela")) && (!use_observed)){
    cout << "Training OUTPUT-RELA with observations" << endl;
    return 0;
  }

  // ---------------------------------------------
  // Either create a dict or load one from the model file
  Corpus training, dev;
  if (fmodel.size() == 0){
    kSOS = d.Convert("<s>"); //Convert is a method of dict
    kEOS = d.Convert("</s>");
    LOG(INFO) << "Create dict from training data ...";
    // read training data
    training = readData(ftrn, &d);
    // no new word types allowed
    d.Freeze(); 
    // reading dev data
    dev = readData(fdev, &d);
  } else {
    LOG(INFO) << "Load dict from pre-trained model ...";
    load_dict(fmodel, d);
    training = readData(ftrn, &d);
    dev = readData(fdev, &d);
  }
  // get dict size
  unsigned vocabsize = d.size();
  LOG(INFO) << "Vocab size = " << vocabsize;
  // save dict
  save_dict(fname, d);
  LOG(INFO) << "Save dict into: " << fname;

  // --------------------------------------------
  // document segmentation
  training = segment_doc(training, docthresh);
  LOG(INFO) << "New training size = " << training.size();
  dev = segment_doc(dev, docthresh);
  LOG(INFO) << "New dev size = " << dev.size();

  // --------------------------------------------
  // strip latent variables from doc
  vector<LatentSeq> trn_obsseq, dev_obsseq;
  for (int didx = 0; didx < training.size(); didx ++){
    LatentSeq obsseq;
    split_relaidx(training[didx], obsseq);
    if (!use_observed){
      obsseq = LatentSeq(obsseq.size(), -1);
    }
    trn_obsseq.push_back(obsseq);
  }
  for (int didx = 0; didx < dev.size(); didx ++){
    LatentSeq obsseq;
    split_relaidx(dev[didx], obsseq);
    if (!use_observed){
      obsseq = LatentSeq(obsseq.size(), -1);
    }
    dev_obsseq.push_back(obsseq);
  }

  // --------------------------------------------
  // define model
  Model bmodel, omodel;
  // only one of them is used in the following
  LVRNNBaseline<LSTMBuilder> blm(bmodel, nlayers, inputdim, 
				 hiddendim, vocabsize, nlatvar);
  LVRNNOutput<LSTMBuilder> olm(omodel, nlayers, inputdim, 
			       hiddendim, vocabsize, nlatvar);
  
  // --------------------------------------------
  // load model
  if (fmodel.size() == 0){
    LOG(INFO) << "Randomly initializing model parameters ...";
  } else {
    if ((flag == "output")||(flag == "outputrela")){
      LOG(INFO) << "Load pretrained OUTPUT model parameters ...";
      load_model(fmodel, omodel);
    } else if ((flag == "baseline")||(flag == "baselinerela")){
      LOG(INFO) << "Load pretrained BASELINE model parameters ...";
      load_model(fmodel, bmodel);
    }
  }

  // --------------------------------------------
  // define learner
  Trainer* sgd = nullptr;
  if ((flag == "output") || (flag == "outputrela")){
    if (use_adagrad){
      sgd = new AdagradTrainer(&omodel, reg, lr0);
    } else { 
      sgd = new SimpleSGDTrainer(&omodel, reg, lr0);
      // cout << "Adam ..." << endl;
      // sgd = new AdamTrainer(&omodel, reg);
    }
  } else if ((flag == "baseline") || (flag == "baselinerela")){
    if (use_adagrad){
      sgd = new AdagradTrainer(&bmodel, reg, lr0);
    } else {
      sgd = new SimpleSGDTrainer(&bmodel, reg, lr0);
      // cout << "Adam ..." << endl;
      // sgd = new AdamTrainer(&omodel, reg);
    }
  } else {
    LOG(INFO) << "unrecognized flag: " << flag;
    return -1;
  }

  // ---------------------------------------------
  // define the indices so we can shuffle the docs
  vector<unsigned> order(training.size());
  for (unsigned i = 0; i < order.size(); i++) order[i] = i;
  int report = 0;
  
  // ---------------------------------------------
  // start training
  Prob prob;
  unsigned epoch = 0;
  for (int i = 0; i < nlatvar; i++){ prob.push_back(1.0); }
  LOG(INFO) << "Start training ...";
  while(true) {
    // loss 
    double loss = 0, ploss = 0;
    // prediction error
    float trn_total = 0.0, trn_correct = 0.0;
    // word counts
    unsigned words = 0;
    //iterating over documents
    for (unsigned i = 0; i < report_every_i; ++i) { 
      //check if it's the number of documents
      if (si == training.size()) {
	LOG(INFO) << "Skip " << nskip << " training documents";
        si = 0; nskip = 0; epoch ++;
	sgd->update_epoch();
	LOG(INFO) << "=== SHUFFLE ===";
        shuffle(order.begin(), order.end(), *rndeng);
      }
      // get one document
      auto& doc = training[order[si]];
      auto& observedseq = trn_obsseq[order[si]];
      si ++;
      // get the right model
      if (flag == "output"){
	// output model with joint obj
	LatentSeq decodedseq = olm.DecodeGraph(doc);
	count_prediction(observedseq, decodedseq, trn_total,
			 trn_correct);
	ComputationGraph* cgptr = new ComputationGraph;
	// with or without dropout
	olm.BuildObjGraph(doc, *cgptr, decodedseq,
			  observedseq, use_dropout);
	ploss = as_scalar(cgptr->forward());
	cgptr->backward(); sgd->update();
	loss += ploss; cgptr->clear(); delete cgptr;
      } else if (flag == "outputrela"){
	// optimize rela obj in output model
	LatentSeq decodedseq = olm.DecodeGraph(doc);
	count_prediction(observedseq, decodedseq, trn_total,
			 trn_correct);
	ComputationGraph* cgptr = new ComputationGraph;
	// with or without dropout
	olm.BuildRelaObjGraph(doc, *cgptr, decodedseq,
			      observedseq, use_dropout);
	ploss = as_scalar(cgptr->forward());
	if (ploss > 0){
	  cgptr->backward(); sgd->update(); loss += ploss;
	}
	cgptr->clear(); delete cgptr;
      } else if (flag == "baseline"){
	// sum over all possible values
	ComputationGraph* cgptr = new ComputationGraph;
	blm.BuildObjGraph(doc, *cgptr, observedseq,
			  use_dropout);
	ploss = as_scalar(cgptr->forward());
	cgptr->backward(); sgd->update(); loss += ploss;
	cgptr->clear(); delete cgptr;
	LatentSeq decodedseq = blm.DecodeGraph(doc);
	count_prediction(observedseq, decodedseq, trn_total,
			 trn_correct);
      } else if (flag == "baselinerela"){
	LatentSeq decodedseq = blm.DecodeGraph(doc);
	count_prediction(observedseq, decodedseq, trn_total,
			 trn_correct);
	ComputationGraph* cgptr = new ComputationGraph;
	blm.BuildRelaObjGraph(doc, *cgptr, observedseq,
			      use_dropout);
	ploss = as_scalar(cgptr->forward());
	if (ploss > 0){
	  cgptr->backward(); sgd->update(); loss += ploss;
	}
	cgptr->clear(); delete cgptr;
      } else {
	LOG(INFO) << "Unrecognized flag: " << flag << endl;
	return -1;
      }
      // get word counts for computing PPL
      for (auto& sent : doc) words += sent.size();
    }
    sgd->status();
    if ((flag == "outputrela")||(flag == "baselinerela")){
      LOG(INFO) << "Err = " 
		<< boost::format("%1.4f") % (loss / trn_total);
    } else {
      LOG(INFO) << "Err = "
    		<< boost::format("%1.4f") % exp(loss / words);
    }
    
    // ----------------------------------------
    report++;
    if (report % dev_every_i_reports == 0) {
      LOG(INFO) << "Start evaluating on dev set ...";
      double dloss = 0, dploss = 0;
      int dwords = 0;
      float total = 0.0, correct = 0.0;
      for (int ddidx = 0; ddidx < dev.size(); ddidx ++){
	// strip observed latent seq from doc
	auto& doc = dev[ddidx];
	auto& observedseq = dev_obsseq[ddidx];
    	// get the right model
    	if (flag == "output"){
	  // compute obj value
	  ComputationGraph* cgptr = new ComputationGraph;
	  olm.BuildInferGraph(doc, *cgptr);
	  dploss = as_scalar(cgptr->forward());
	  dloss += dploss; cgptr->clear(); delete cgptr;
	  // compute prediction accuracy
	  if (use_observed){
	    LatentSeq decodedseq = olm.DecodeGraph(doc);
	    count_prediction(observedseq, decodedseq,
			     total, correct);
	  }
    	} else if (flag == "outputrela"){
	  // need decoding first
	  LatentSeq decodedseq = olm.DecodeGraph(doc);
	  // compute obj value
	  ComputationGraph* cgptr = new ComputationGraph;
	  olm.BuildRelaObjGraph(doc, *cgptr, decodedseq,
				observedseq, false);
	  dploss = as_scalar(cgptr->forward());
	  dloss += dploss; cgptr->clear(); delete cgptr;
	  // compute prediction accuracy
	  if (use_observed)
	    count_prediction(observedseq, decodedseq,
			     total, correct);
    	} else if (flag == "baseline"){
	  // compute obj value
	  ComputationGraph* cgptr = new ComputationGraph;
	  blm.BuildInferGraph(doc, *cgptr, observedseq);
	  dploss = as_scalar(cgptr->forward());
	  dloss += dploss; cgptr->clear(); delete cgptr;
	  // compute prediction accuracy
	  if (use_observed){
	    LatentSeq decodedseq = blm.DecodeGraph(doc);
	    count_prediction(observedseq, decodedseq,
			     total, correct);
	  }
	} else if (flag == "baselinerela"){
	  // compute rela obj value
	  ComputationGraph* cgptr = new ComputationGraph;
	  blm.BuildRelaObjGraph(doc, *cgptr, observedseq,
				false);
	  dploss = as_scalar(cgptr->forward());
	  dloss += dploss; cgptr->clear(); delete cgptr;
	  if (use_observed){
	    LatentSeq decodedseq = blm.DecodeGraph(doc);
	    count_prediction(observedseq, decodedseq,
			     total, correct);
	  }
	}
	// count words
    	for (auto& sent : doc) dwords += sent.size() - 1;
      }
      // print dev information
      double acc = (correct / total);
      if ((flag == "outputrela")||(flag == "baselinerela")){
	LOG(INFO) << "DEV [epoch = " 
		  << boost::format("%1.2f") % (((epoch * training.size()) + si) / (float)training.size())
		  << "] E = "
		  << boost::format("%5.4f") % (dloss / total) 
		  << " ( "
		  << boost::format("%5.4f") % (best / total)
		  << " )";
	if (bestacc < acc){
	  LOG(INFO) << "Save model into: "<< fname;
	  if (flag == "outputrela"){
	    save_model(fname, omodel);
	  } else if (flag == "baselinerela"){
	    save_model(fname, bmodel);
	  }
	}
      } else {
	LOG(INFO) << "DEV [epoch = " 
		  << boost::format("%1.2f") % (((epoch * training.size()) + si) / (float)training.size())
		  << "] E = "
		  << boost::format("%5.4f") % exp(dloss / dwords) 
		  << " ( "
		  << boost::format("%5.4f") % exp(best / dwords)
		  << " )";
	if (best > dloss){
	  LOG(INFO) << "Save model into: "<< fname;
	  if (flag == "output"){
	    save_model(fname, omodel);
	  } else if (flag == "baseline"){
	    save_model(fname, bmodel);
	  }
	}
      }
      // print dev accuracy if necessary
      if (use_observed){
	LOG(INFO) << "DEV accuracy = "
		  << boost::format("%1.4f") % acc
		  << " ("
		  << boost::format("%1.4f") % bestacc
		  << ")";
	if (bestacc < acc) bestacc = acc;
      }
      if (best > dloss) best = dloss;
    }
  }
  delete sgd;
}
