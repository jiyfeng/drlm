#include "util.hpp"
#include "training.hpp"
#include "test.hpp"
#include <stdlib.h>
#include <boost/program_options.hpp>

namespace po = boost::program_options;

// main function
int main(int argc, char** argv) {
  // ----------------------------------------------------
  // init cnn
  cout << "argc = " << argc << endl;
  cnn::Initialize(argc, argv);
  cout << "argc = " << argc << endl;

  // ------------------------------------------------
  // parameter options
  po::options_description desc("Allowed options");
  desc.add_options()
    ("help", "produce this help message")
    ("action", po::value<string>(), "action for training or test")
    ("trnfile", po::value<string>(), "training file name")
    ("devfile", po::value<string>(), "development file name")
    ("tstfile", po::value<string>(), "test file name")
    ("nlatval", po::value<unsigned>(), "number of latent values")
    ("modelfile", po::value<string>()->default_value(string("")), "model file name")
    ("flag", po::value<string>()->default_value(string("output")), "model flag")
    ("nlayers", po::value<int>()->default_value((int)2), "number of LSTM layers")
    ("inputdim", po::value<int>()->default_value((int)16), "input dimension")
    ("hiddendim", po::value<int>()->default_value((int)32), "hidden dimension")
    ("lr", po::value<float>()->default_value((float)1e-1), "initial learning rate")
    ("reg", po::value<float>()->default_value((float)1e-6), "regularization weight")
    ("docthresh", po::value<int>()->default_value((int)5), "document length threshold")
    ("nparticle", po::value<unsigned>()->default_value((unsigned)5), "number of particles")
    ("logfolder", po::value<string>()->default_value(string("tmp")), "folder for log files")
    ("with-adagrad", po::value<bool>()->default_value((bool)1), "whether use AdaGrad")
    ("with-dropout", po::value<bool>()->default_value((bool)1), "whether use Dropout")
    ("with-observation", po::value<bool>()->default_value((bool)1), "whether use observations of latent variables for training")
    ("dev-evalfreq", po::value<int>()->default_value((int)10), "dev evaluation freq");
  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);
  // 
  po::positional_options_description p;
  p.add("trnfile",1);
  p.add("devfile",2);
  // 
  po::store(po::command_line_parser(argc,argv).
	    options(desc).positional(p).run(), vm);
	    po::notify(vm);
  if (vm.count("help")) { cout << desc << "\n"; return 1;}
  if (!vm.count("nlatval")) {
    cout << "Please specifiy the number of latent values" <<endl;
    return 1;
  }
  if (!vm.count("action")){
    cerr << "Please provide one specific action value" << endl;
    return 1;
  }
  // ----------------------------------------------------
  // get parameter values
  string cmd = vm["action"].as<string>();
  
  // ----------------------------------------------------
  if (cmd == "train"){
    cout << "Action: " << cmd <<endl;
    if ((!vm.count("trnfile"))||(!vm.count("devfile"))){
      cerr << "Must specify a training and a dev file" << endl;
      return -1;
    }
    string ftrn = vm["trnfile"].as<string>();
    string fdev = vm["devfile"].as<string>();
    int nlayers = vm["nlayers"].as<int>(); // number of layers
    int inputdim = vm["inputdim"].as<int>(); // input dim
    int hiddendim = vm["hiddendim"].as<int>(); // hidden dim
    string flag = vm["flag"].as<string>(); // flag
    string logfolder = vm["logfolder"].as<string>(); // 
    float lr = vm["lr"].as<float>(); // initial learning rate
    float reg = vm["reg"].as<float>(); 
    bool withadagrad = vm["with-adagrad"].as<bool>();
    bool withdropout = vm["with-dropout"].as<bool>();
    unsigned nparticle = vm["nparticle"].as<unsigned>();
    int docthresh = vm["docthresh"].as<int>();
    bool withobs = vm["with-observation"].as<bool>();
    unsigned nlatval = vm["nlatval"].as<unsigned>();
    int reportfreq = vm["dev-evalfreq"].as<int>();
    string fmodel = vm["modelfile"].as<string>();
    train((char*)ftrn.c_str(), (char*)fdev.c_str(),
    	  nlayers, inputdim, hiddendim, flag, logfolder,
    	  lr, reg, withadagrad, nparticle, docthresh,
	  withobs, nlatval, reportfreq,
	  withdropout, fmodel);
  } else if(cmd == "test") {
    cout << "Action: "<< cmd<<endl;
    string fprefix = vm["modelfile"].as<string>();
    if ((!vm.count("tstfile"))||(fprefix.size() == 0)){
      cerr << "Must specify a test and a model file" << endl;
      return -1;
    }
    string ftst = vm["tstfile"].as<string>();
    string flag = vm["flag"].as<string>();
    unsigned nlatval = vm["nlatval"].as<unsigned>();
    int nlayers = vm["nlayers"].as<int>();
    test((char*)ftst.c_str(), (char*)fprefix.c_str(),
  	 flag, nlatval, nlayers);
  }
  else{
    cerr << "Unrecognized command " << argv[1]<<endl;
  }
}
