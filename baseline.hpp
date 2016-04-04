#ifndef LVRNN_BASELINE_HPP
#define LVRNN_BASELINE_HPP

#include "util.hpp"

template <class Builder>
class LVRNNBaseline{
private:
  LookupParameters* p_W; // word embeddings VxK1
  Parameters* p_R; // output weight
  Parameters* p_C; // forward context vector: VxK2
  Parameters* p_T;
  Parameters* p_TC;
  Parameters* p_bias; // bias V x K
  Parameters* p_context; // default context vector
  Parameters* p_L; // for latent variable prediction
  Parameters* p_lbias; // for latent variable prediction
  Parameters* p_T_d; // for dummy relation
  Parameters* p_TC_d; // for dummy relation
  Parameters* p_bias_d; // for dummy relation 
  Builder builder;
  unsigned nlatvar;
  // for inference
  vector<float> final_h;

public:
  LVRNNBaseline(Model& model, unsigned nlayers, unsigned inputdim, 
		unsigned hiddendim, unsigned vocabsize, 
		unsigned nlatent):builder(nlayers, inputdim, 
					 hiddendim, &model){
    // number of latent variables
    nlatvar = nlatent;
    // word representation
    p_W = model.add_lookup_parameters(vocabsize, {inputdim});
    // output weight
    p_R = model.add_parameters({vocabsize, hiddendim});
    // context weight
    p_C = model.add_parameters({vocabsize, hiddendim});
    // prediction bias
    p_bias = model.add_parameters({vocabsize, nlatent});
    // context transform matrix
    p_T = model.add_parameters({hiddendim, hiddendim, nlatent});
    p_TC = model.add_parameters({hiddendim, hiddendim, nlatent});
    p_lbias = model.add_parameters({nlatent}, 1e-9);
    // default context vector
    p_context = model.add_parameters({hiddendim});
    // latent variable distribution
    p_L = model.add_parameters({nlatent, hiddendim});
    // for dummy relation types
    p_T_d = model.add_parameters({hiddendim, hiddendim});
    p_TC_d = model.add_parameters({hiddendim, hiddendim});
    p_bias_d = model.add_parameters({vocabsize});
  }

  /************************************************
   * Build CG of a given doc with a latent sequence
   *
   * doc: 
   * cg: computation graph
   * latseq: latent sequence from decoding
   * obsseq: latent sequence from observation
   * flag: what we expected to get from this function
   ************************************************/
  Expression BuildGraph(const Doc& doc, ComputationGraph& cg,
			LatentSeq obsseq, const string& flag,
			bool with_dropout){
    builder.new_graph(cg);
    // define expression
    Expression i_R = parameter(cg, p_R);
    Expression i_C = parameter(cg, p_C);
    Expression i_T = parameter(cg, p_T);
    Expression i_TC = parameter(cg, p_TC);
    Expression i_bias = parameter(cg, p_bias);
    Expression i_context = parameter(cg, p_context);
    Expression i_L = parameter(cg, p_L);
    Expression i_lbias = parameter(cg, p_lbias);
    vector<Expression> negloglik, neglogprob;
    // -----------------------------------------
    // iterate over latent sequences
    // get LV-related transformation matrix
    for (unsigned k = 0; k < doc.size(); k++){
      // using latent size as constraint
      builder.start_new_sequence();
      // for each sentence in this doc
      Expression cvec;
      auto& sent = doc[k];
      // start a new sequence for each sentence
      if (k == 0)
	cvec = i_context;
      else
	cvec = input(cg, {(unsigned)final_h.size()}, final_h);
      // if dropout
      if (with_dropout) cvec = dropout(cvec, 0.5);
      // latent variable distribution
      Expression r_k = (i_L * cvec) + i_lbias;
      Expression lvprob, Tk, TCk, biask;
      // get transform matrix
      if ((obsseq[k] >= 0) && (flag == "OBJ")){
	// only for training
	// if discourse information is observed
	int latval = obsseq[k];
	// delta distribution
	vector<float> vec_prob = vector<float>(nlatvar, 0.0);
	vec_prob[latval] = 1.0;
	lvprob = input(cg, {(unsigned)nlatvar}, vec_prob);
	// get lv prediction error
	Expression k_neglogprob = pickneglogsoftmax(r_k, latval);
      	neglogprob.push_back(k_neglogprob);
	// get Tk, TCk, biask
	Tk = contract3d_1d(i_T, lvprob);
	TCk = contract3d_1d(i_TC, lvprob);
	biask = i_bias * lvprob;
      } else if (flag == "INFER"){
	// only for language modeling inference
	// no matter whether discourse information is observed
	lvprob = softmax(r_k);
	// get Tk, TCk, biask
	Tk = contract3d_1d(i_T, lvprob);
	TCk = contract3d_1d(i_TC, lvprob);
	biask = i_bias * lvprob;
      } else if ((obsseq[k] < 0) && (flag == "OBJ")){
	Tk = parameter(cg, p_T_d);
	TCk = parameter(cg, p_TC_d);
	biask = parameter(cg, p_bias_d);
      } else {
	cout << "Unrecognized situation in joint model" << endl;
	abort();
      }
      
      // build RNN for the current sentence
      Expression ccpb = (i_C * (TCk * cvec)) + biask;
      unsigned slen = sent.size() - 1;
      Expression i_x_t, i_h_t, i_y_t, i_negloglik, new_h;
      for (unsigned t = 0; t < slen; t++){
	// get word representation
	i_x_t = lookup(cg, p_W, sent[t]);
	if (with_dropout) i_x_t = dropout(i_x_t, 0.5);
	// compute hidden state
	i_h_t = builder.add_input(i_x_t);
	// if dropout
	if (with_dropout)
	  new_h = dropout(i_h_t, 0.5);
	else
	  new_h = i_h_t;
	// compute prediction
	i_y_t = (i_R * (Tk * new_h)) + ccpb;
	// get word prediction error
	i_negloglik = pickneglogsoftmax(i_y_t, sent[t+1]);
	// add back
	negloglik.push_back(i_negloglik);
      }
      // update latent representation
      final_h.clear();
      final_h = as_vector(i_h_t.value());
    }
    // get result
    Expression res;
    if ((flag != "INFER") && (flag != "OBJ")){
      cerr << "Unrecognized flag: " << flag << endl;
      abort();
    } else if ((neglogprob.size() > 0) && (flag == "OBJ")){
      res = sum(negloglik) + sum(neglogprob);
    } else {
      res = sum(negloglik);
    }
    return res;
  }

   /************************************************
   * Build CG of a given doc with a latent sequence
   *
   * doc: 
   * cg: computation graph
   * latseq: latent sequence from decoding
   * obsseq: latent sequence from observation
   * with_dropout: whether use dropout for training
   ************************************************/
  Expression BuildRelaGraph(const Doc& doc, ComputationGraph& cg,
			    LatentSeq obsseq, bool with_dropout){
    builder.new_graph(cg);
    // define expression
    Expression i_C = parameter(cg, p_C);
    Expression i_R = parameter(cg, p_R);
    Expression i_T = parameter(cg, p_T);
    Expression i_TC = parameter(cg, p_TC);
    Expression i_bias = parameter(cg, p_bias);
    Expression i_context = parameter(cg, p_context);
    Expression i_L = parameter(cg, p_L);
    Expression i_lbias = parameter(cg, p_lbias);
    // vector<Expression> negloglik, neglogprob;
    // -----------------------------------------
    // iterate over latent sequences
    // get LV-related transformation matrix
    Expression i_h_t;
    vector<Expression> obj;
    for (unsigned k = 0; k < doc.size(); k++){
      auto& sent = doc[k];
      // start a new sequence for each sentence
      Expression cvec;
      if (k == 0)
	cvec = i_context;
      else
	cvec = input(cg, {(unsigned)final_h.size()}, final_h);
      // if dropout
      if (with_dropout) cvec = dropout(cvec, 0.5);
      // two parts of the objective function
      Expression sent_objpart1;
      vector<Expression> sent_objpart2;
      for (int latval = 0; latval < nlatvar; latval ++){
	builder.start_new_sequence();
	// latent variable distribution
	Expression l_neglogprob = pickneglogsoftmax((i_L * cvec) + i_lbias, latval);
	// get rnn
	Expression Tk, TCk, biask, lvprob;
	// for each particular relation
	vector<float> vec_prob = vector<float>(nlatvar, 0.0);
	vec_prob[latval] = 1.0;
	lvprob = input(cg, {(unsigned)nlatvar}, vec_prob);
	// get relation specific transformation
	Tk = contract3d_1d(i_T, lvprob);
	TCk = contract3d_1d(i_TC, lvprob);
	biask = i_bias * lvprob;
	// define expressions
	Expression ccpb, i_x_t, i_h_t, i_y_t, i_negloglik, new_h;
	vector<Expression> l_negloglik;
	// context vector part
	ccpb = (i_C * (TCk * cvec)) + biask;
	unsigned slen = sent.size() - 1;
	for (unsigned t = 0; t < slen; t++){
	  // get word representation
	  i_x_t = lookup(cg, p_W, sent[t]);
	  if (with_dropout) i_x_t = dropout(i_x_t, 0.5);
	  // compute hidden state
	  i_h_t = builder.add_input(i_x_t);
	  // dropout to get a new_h, as the old i_h_t will
	  //   be used as context vector
	  if (with_dropout)
	    new_h = dropout(i_h_t, 0.5);
	  else
	    new_h = i_h_t;
	  // compute prediction
	  i_y_t = (i_R * (Tk * new_h)) +  ccpb;
	  // get prediction error
	  i_negloglik = pickneglogsoftmax(i_y_t, sent[t+1]);
	  // add back
	  l_negloglik.push_back(i_negloglik);
	}
	// - log P(y, z) given Y and a specific Z value
	Expression pxz = sum(l_negloglik) + l_neglogprob;
	// log P(y, z)
	sent_objpart2.push_back(pxz * (-1.0));
	if (obsseq[k] == latval){
	  // pick the right part as objective
	  sent_objpart1 = pxz * (-1.0);
	}
	// update context vector
	if (latval == (nlatvar - 1)){
	  final_h.clear();
	  final_h = as_vector(i_h_t.value());
	}
      }
      // if the latent variable is observed
      if (obsseq[k] >= 0){
	// log of softmax
	Expression sent_obj = logsumexp(sent_objpart2)
	  - sent_objpart1;
	obj.push_back(sent_obj);
      }
    }
    // get the objectve for entire doc
    if (obj.size() > 0){
      // if at least one observed latent value
      return sum(obj);
    } else {
      // otherwise
      Expression zero = input(cg, 0.0);
      return zero;
    }
  }

  /*********************************************
   * Build computation graph for one sentence
   * 
   * sent: Sent instance
   *********************************************/
  Expression BuildSentGraph(const Sent& sent, const unsigned sidx,
			    ComputationGraph& cg,
			    const int latval){
    builder.new_graph(cg);
    builder.start_new_sequence();
    // define expression
    Expression i_R = parameter(cg, p_R);
    Expression i_C = parameter(cg, p_C);
    Expression i_T = parameter(cg, p_T);
    Expression i_TC = parameter(cg, p_TC);
    Expression i_bias = parameter(cg, p_bias);
    Expression i_context = parameter(cg, p_context);
    Expression i_L = parameter(cg, p_L);
    Expression i_lbias = parameter(cg, p_lbias);
    // Initialize cvec
    Expression cvec;
    if (sidx == 0)
      cvec = i_context;
    else
      cvec = input(cg, {(unsigned)final_h.size()}, final_h);
    // -------------------------------------------
    // compute the prob for the given latval
    Expression r_k, lv_neglogprob;
    r_k = (i_L * cvec) + i_lbias;
    lv_neglogprob = pickneglogsoftmax(r_k, latval);
    // for each relation
    vector<float> vec_prob = vector<float>(nlatvar, 0.0);
    vec_prob[latval] = 1.0;
    Expression Tk, TCk, biask, lvprob;
    lvprob = input(cg, {(unsigned)nlatvar}, vec_prob);
    // get relation specific transformation
    Tk = contract3d_1d(i_T, lvprob);
    TCk = contract3d_1d(i_TC, lvprob);
    biask = i_bias * lvprob;
    // -------------------------------------------
    // compute likelihood
    vector<Expression> negloglik;
    Expression i_negloglik, i_x_t, i_h_t, i_y_t, ccpb;
    ccpb = (i_C * (TCk * cvec)) + biask;
    unsigned slen = sent.size() - 1;
    for (unsigned t = 0; t < slen; t++){
      // get word representation
      i_x_t = const_lookup(cg, p_W, sent[t]);
      // compute hidden state
      i_h_t = builder.add_input(i_x_t);
      // compute prediction
      i_y_t = (i_R * (Tk * i_h_t)) + ccpb;
      // get prediction error
      i_negloglik = pickneglogsoftmax(i_y_t, sent[t+1]);
      // push back
      negloglik.push_back(i_negloglik);
    }
    // update final_h, if latval = nlatvar - 1
    if (latval == (nlatvar - 1)){
      final_h = as_vector(i_h_t.value());
    }
    // result (posterior)
    Expression res = (sum(negloglik) + lv_neglogprob) * (-1.0);
    return res;
  }
	
  /*********************************************
   * Sample particles for a given document
   * 
   * doc: 
   *********************************************/
  LatentSeq DecodeGraph(const Doc doc){
    // ----------------------------------------
    // init
    int nsent = doc.size();
    LatentSeq latseq;
    // ----------------------------------------
    // for each sentence in doc, each latval, compute
    // the posterior prob p(R|cvec, sent)
    vector<float> U;
    for (unsigned sidx = 0; sidx < nsent; sidx ++){
      for (int val = 0; val < nlatvar; val ++){
	ComputationGraph cg;
	BuildSentGraph(doc[sidx], sidx, cg, val);
	float prob = as_scalar(cg.forward());
	U.push_back(prob);
	cg.clear();
      }
      // normalize and get the argmax
      log_normalize(U);
      // local decoding
      int max_idx = argmax(U);
      U.clear();
      latseq.push_back(max_idx);
    }
    return latseq;
  }

  /**********************************************
   * Build Obj graph for learning
   *
   **********************************************/
  Expression BuildObjGraph(const Doc& doc,
			   ComputationGraph& cg,
			   LatentSeq obsseq,
			   bool with_dropout){
    Expression obj = BuildGraph(doc, cg, obsseq, "OBJ",
				with_dropout);
    return obj;
  }

  /**********************************************
   * Build graph for inference
   *
   **********************************************/
  Expression BuildInferGraph(const Doc& doc,
			     ComputationGraph& cg,
			     LatentSeq obsseq){
    Expression obj = BuildGraph(doc, cg, obsseq, "INFER",
				false);
    return obj;
  }

  /**********************************************
   * Build Rela Obj graph for learning
   *
   **********************************************/
  Expression BuildRelaObjGraph(const Doc& doc,
			       ComputationGraph& cg,
			       LatentSeq obsseq,
			       bool with_dropout){
    Expression obj = BuildRelaGraph(doc, cg, obsseq,
				    with_dropout);
    return obj;
  }
};

#endif
