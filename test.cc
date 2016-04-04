#include "test.hpp"

#include <boost/format.hpp>

// ********************************************************
// test
// ********************************************************
int test(char* ftst, char* prefix, string flag,
         unsigned nlatval, unsigned nlayers)
{
  // ---------------------------------------------
  //
  cnn::Dict d;
  cerr << "ftst = " << ftst << endl;
  cerr << "prefix = " << prefix << endl;
  cerr << "flag = " << flag << endl;
  cerr << "nlatval = " << nlatval << endl;
  
  // ---------------------------------------------
  // predefined variable (will be overwritten after
  //    loading model)
  unsigned inputdim = 16, hiddendim = 48;
  // model and dict file name prefix
  string fprefix = string(prefix);
  string fout = string(ftst);
  ofstream pplfile, labelfile;
  pplfile.open(fout + "." + flag + ".ppl");
  labelfile.open(fout + "." + flag + ".label");
  
  // ---------------------------------------------
  // check model name
  if (fprefix.size() == 0) {
    cerr << "Unspecified model name" << endl;
    return -1;
  }
  // load dict and freeze it
  load_dict(fprefix, d);
  unsigned vocabsize = d.size();
  cerr << "Vocab size = " << vocabsize << endl;
  d.Freeze();
  Corpus tst = readData(ftst, &d, false);
  cerr << "Test size = " << tst.size() << endl;
  // tst = segment_doc(tst, 5);
  // cerr << "New test size = " << tst.size() << endl;
  // strip labels from tst data
  vector<LatentSeq> tst_obsseq;
  for (int didx = 0; didx < tst.size(); didx ++) {
    LatentSeq obsseq;
    split_relaidx(tst[didx], obsseq);
    tst_obsseq.push_back(obsseq);
  }
  
  // ----------------------------------------------
  // define model
  Model omodel;
  // only one of them is used in the following
  LVRNNOutput<LSTMBuilder> olm(omodel, nlayers, inputdim,
			       hiddendim, vocabsize, nlatval);
  
  // Load model
  cerr << "Load model from: " << fprefix << ".model" << endl;
  if ((flag == "output") || (flag == "outputrela")) {
    load_model(fprefix, omodel);
  } else {
    cerr << "Unrecognized flag" << endl;
    return -1;
  }
  
  // ---------------------------------------------
  // start testing
  int nskip = 0;
  double loss = 0, dloss = 0;
  unsigned words = 0, dwords = 0;
  float total = 0.0, correct = 0.0;
  // proposal distribution
  Prob prob = Prob(nlatval, 1.0);
  // iterating over documents
  int ncount = 0;
  for (int idx = 0; idx < tst.size(); idx ++) {
    ostringstream os;
    auto& doc = tst[idx];
    auto& observedseq = tst_obsseq[idx];
    // choose the evaluation model
    if (flag == "output"){
      // compute obj value
      ComputationGraph* cgptr = new ComputationGraph;
      olm.BuildInferGraph(doc, *cgptr);
      dloss = as_scalar(cgptr->forward());
      loss += dloss; cgptr->clear(); delete cgptr;
      // compute prediction accuracy
      LatentSeq decodedseq = olm.DecodeGraph(doc);
      count_prediction(observedseq, decodedseq,
		       total, correct);
      for (unsigned t = 0; t < observedseq.size(); t ++){
	os << observedseq[t] << "\t" << decodedseq[t] << "\n";
      }
    } else if (flag == "outputrela"){
      // need decoding first
      LatentSeq decodedseq = olm.DecodeGraph(doc);
      // compute obj value
      ComputationGraph* cgptr = new ComputationGraph;
      olm.BuildRelaObjGraph(doc, *cgptr, decodedseq,
				observedseq, false);
      dloss = as_scalar(cgptr->forward());
      loss += dloss; cgptr->clear(); delete cgptr;
      // compute prediction accuracy
      count_prediction(observedseq, decodedseq,
		       total, correct);
      for (unsigned t = 0; t < observedseq.size(); t ++){
	os << observedseq[t] << "\t" << decodedseq[t] << "\n";
      }
    }
    dwords = 0;
    for (auto& sent : doc) dwords += (sent.size() - 1);
    words += dwords; ncount ++;
    cerr << ncount << " : "
	 << boost::format("%5.4f") % exp(dloss / dwords)
	 << endl;
    pplfile << boost::format("%5.4f") % dloss
	    << " : "
	    << boost::format("%4.1f") % dwords
	    << endl;
    labelfile << os.str() << "===" << endl;
  }
  cerr << " Skip " << nskip << " docs " << endl;
  cerr << " E = "
       << boost::format("%1.4f") % (loss / words)
       << " PPL = "
       << boost::format("%5.4f") % exp(loss / words)
       << endl;
  pplfile.close(); labelfile.close();
  return 0;
}
