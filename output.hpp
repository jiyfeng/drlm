#ifndef LVRNN_OUTPUT_HPP
#define LVRNN_OUTPUT_HPP

#include "util.hpp"

template <class Builder>
class LVRNNOutput{
private:
  LookupParameters* p_W; // word embeddings VxK1
  LookupParameters* p_T;
  LookupParameters* p_TC;
  LookupParameters* p_bias; // bias Vx1
  Parameters* p_R; // output weight
  Parameters* p_C; // forward context vector: VxK2
  Parameters* p_context; // default context vector
  Parameters* p_L; // for latent variable prediction
  Parameters* p_lbias; // for latent variable prediction
  Parameters* p_T_d; // for dummy relation
  Parameters* p_TC_d; // for dummy relation
  Parameters* p_bias_d; // for dummy relation 
  Builder builder;
  unsigned nlatvar;
  vector<float> final_h;

public:
  LVRNNOutput(Model& model, unsigned nlayers, unsigned inputdim, 
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
    p_bias = model.add_lookup_parameters(nlatent, {vocabsize});
    // default context vector
    p_context = model.add_parameters({hiddendim});
    // latent variable distribution
    p_L = model.add_parameters({nlatent, hiddendim});
    p_lbias = model.add_parameters({nlatent}, 1e-9);
    // transform matrix
    p_T = model.add_lookup_parameters(nlatent,
				      {hiddendim, hiddendim});
    p_TC = model.add_lookup_parameters(nlatent,
				      {hiddendim, hiddendim});
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
			LatentSeq latseq, LatentSeq obsseq,
			const string& flag, bool with_dropout){
    // check flag
    if ((flag != "INFER") && (flag != "OBJ")){
      cerr << "Unrecognized flag: " << flag << endl;
      abort();
    }
    // renew the graph
    builder.new_graph(cg);
    // define expression
    Expression i_C = parameter(cg, p_C);
    Expression i_R = parameter(cg, p_R);
    // Expression i_bias = parameter(cg, p_bias);
    Expression i_context = parameter(cg, p_context);
    Expression i_L = parameter(cg, p_L);
    Expression i_lbias = parameter(cg, p_lbias);
    vector<Expression> negloglik, neglogprob;
    // -----------------------------------------
    // check hidden variable list
    assert(latseq.size() <= doc.size());
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
      if (k == 0){
	cvec = i_context;
      } else {
	cvec = input(cg, {(unsigned)final_h.size()}, final_h);
      }
      // if dropout
      if (with_dropout) cvec = dropout(cvec, 0.5);
      // latent variable distribution
      int latval = 0;
      // For relation prediction
      Expression ylv = (i_L * cvec) + i_lbias;
      // Relation specific parameters
      Expression i_bias_k, i_Tk, i_TCk;
      if ((obsseq[k] >=0) && (flag == "OBJ")){
	// only for joint training,
	// if discourse information is observed
      	latval = obsseq[k];
	Expression k_neglogprob = pickneglogsoftmax(ylv, latval);
	neglogprob.push_back(k_neglogprob);
	// relation specific term
	i_bias_k = lookup(cg, p_bias, latval);
	i_Tk = lookup(cg, p_T, latval);
	i_TCk = lookup(cg, p_TC, latval);
      } else if (flag == "INFER") {
	// for language modeling inference
	// don't matter whether discourse information is observed
	vector<float> prob = as_vector(softmax(ylv).value());
	latval = argmax(prob);
	// relation specific term
	i_bias_k = lookup(cg, p_bias, latval);
	i_Tk = lookup(cg, p_T, latval);
	i_TCk = lookup(cg, p_TC, latval);
      } else if((obsseq[k] < 0) && (flag == "OBJ")) {
	// only for joint training
	// if discourse information is not observed
	i_bias_k = parameter(cg, p_bias_d);
	i_Tk = parameter(cg, p_T_d);
	i_TCk = parameter(cg, p_TC_d);
      } else {
	cout << "Unrecognized situation in joint model" << endl;
	abort();
      }
      
      // build RNN for the current sentence
      Expression ccpb, i_x_t, i_h_t, i_y_t, i_negloglik, new_h;
      ccpb = (i_C * (i_TCk * cvec)) + i_bias_k;
      // ccpb = (i_C * (i_TCk * cvec));
      unsigned slen = sent.size() - 1;
      for (unsigned t = 0; t < slen; t++){
	// get word representation
	i_x_t = lookup(cg, p_W, sent[t]);
	if (with_dropout) i_x_t = dropout(i_x_t, 0.5);
	// compute hidden state
	i_h_t = builder.add_input(i_x_t);
	if (with_dropout)
	  new_h = dropout(i_h_t, 0.5);
	else
	  new_h = i_h_t;
	// compute prediction
	i_y_t = (i_R * (i_Tk * new_h)) + ccpb;
	// get prediction error
	i_negloglik = pickneglogsoftmax(i_y_t, sent[t+1]);
	// add back
	negloglik.push_back(i_negloglik);
      }
      final_h.clear();
      final_h = as_vector(i_h_t.value());
    }
    // get result
    Expression res;
    if ((neglogprob.size() > 0) && (flag == "OBJ")){
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
			    LatentSeq latseq, LatentSeq obsseq,
			    bool with_dropout){
    builder.new_graph(cg);
    // define expression
    Expression i_C = parameter(cg, p_C);
    Expression i_R = parameter(cg, p_R);
    // Expression i_bias = parameter(cg, p_bias);
    Expression i_context = parameter(cg, p_context);
    Expression i_L = parameter(cg, p_L);
    Expression i_lbias = parameter(cg, p_lbias);
    vector<Expression> negloglik, neglogprob;
    // -----------------------------------------
    // check hidden variable list
    assert(latseq.size() <= doc.size());
    // -----------------------------------------
    // iterate over latent sequences
    // get LV-related transformation matrix
    vector<Expression> obj;
    for (unsigned k = 0; k < doc.size(); k++){
      auto& sent = doc[k];
      // start a new sequence for each sentence
      Expression cvec;
      if (k == 0){
	cvec = i_context;
      } else {
	cvec = input(cg, {(unsigned)final_h.size()}, final_h);
      }
      // if dropout
      if (with_dropout) cvec = dropout(cvec, 0.5);
      // two parts of the objective function
      Expression sent_objpart1;
      vector<Expression> sent_objpart2;
      for (int latval = 0; latval < nlatvar; latval ++){
	builder.start_new_sequence();
	// latent variable distribution
	vector<Expression> l_negloglik;
	Expression l_neglogprob = pickneglogsoftmax((i_L * cvec) + i_lbias, latval);
	// build RNN for the current sentence
	Expression ccpb, i_x_t, i_h_t, i_y_t, i_negloglik, new_h;
	Expression i_Tk = lookup(cg, p_T, latval);
	Expression i_TCk = lookup(cg, p_TC, latval);
	Expression i_bias_k = lookup(cg, p_bias, latval);
	// context + bias
	ccpb = (i_C * (i_TCk * cvec)) + i_bias_k;
	// for each word
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
	  i_y_t = (i_R * (i_Tk * new_h)) +  ccpb;
	  // get prediction error
	  i_negloglik = pickneglogsoftmax(i_y_t, sent[t+1]);
	  // add back
	  l_negloglik.push_back(i_negloglik);
	}
	// update context vector
	if (latval == (nlatvar - 1)){
	  final_h.clear();
	  final_h = as_vector(i_h_t.value());
	}
	// - log P(y, z) given Y and a specific Z value
	Expression pxz = sum(l_negloglik) + l_neglogprob;
	// log P(y, z)
	sent_objpart2.push_back(pxz * (-1.0));
	if (obsseq[k] == latval){
	  // pick the right part as objective
	  sent_objpart1 = pxz * (-1.0);
	}
      }
      // if the latent variable is observed
      if (obsseq[k] >= 0){
	// log of softmax
	Expression sent_obj = logsumexp(sent_objpart2) - sent_objpart1;
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
    // define expressions
    Expression i_C = parameter(cg, p_C);
    Expression i_R = parameter(cg, p_R);
    // Expression i_bias = parameter(cg, p_bias);
    Expression i_context = parameter(cg, p_context);
    Expression i_L = parameter(cg, p_L);
    Expression i_lbias = parameter(cg, p_lbias);
    // initialize cvec
    Expression cvec;
    if (sidx == 0)
      cvec = i_context;
    else
      cvec = input(cg, {(unsigned)final_h.size()}, final_h);
    // compute the prob for the given latval
    Expression i_Tk = lookup(cg, p_T, latval);
    Expression i_TCk = lookup(cg, p_TC, latval);
    Expression i_bias_k = lookup(cg, p_bias, latval);
    Expression lv_neglogprob = pickneglogsoftmax(((i_L * cvec) + i_lbias), latval);
    vector<Expression> negloglik;
    Expression i_negloglik, i_x_t, i_h_t, i_y_t, ccpb;
    // context + bias
    ccpb = (i_C * (i_TCk * cvec)) + i_bias_k;
    // for each word
    unsigned slen = sent.size() - 1;
    for (unsigned t = 0; t < slen; t++){
      // get word representation
      i_x_t = lookup(cg, p_W, sent[t]);
      // compute hidden state
      i_h_t = builder.add_input(i_x_t);
      // compute prediction
      i_y_t = (i_R * (i_Tk * i_h_t)) + ccpb;
      // get prediction error
      i_negloglik = pickneglogsoftmax(i_y_t, sent[t+1]);
      // push back
      negloglik.push_back(i_negloglik);
    }
    // update final_h, if latval = nlatvar - 1
    if (latval == (nlatvar - 1)){
      final_h.clear();
      final_h = as_vector(i_h_t.value());
    }
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
      // cerr << "max_latval = " << max_idx << endl;
      latseq.push_back(max_idx);
    }
    // cerr << "====" << endl;
    return latseq;
  }

  /**********************************************
   * Build Obj graph for learning
   *
   **********************************************/
  Expression BuildObjGraph(const Doc& doc,
			   ComputationGraph& cg,
			   LatentSeq latseq,
			   LatentSeq obsseq,
			   bool with_dropout = false){
    Expression obj = BuildGraph(doc, cg, latseq, obsseq, "OBJ",
				with_dropout);
    return obj;
  }

  /**********************************************
   * Build Rela Obj graph for learning
   *
   **********************************************/
  Expression BuildRelaObjGraph(const Doc& doc,
			       ComputationGraph& cg,
			       LatentSeq latseq,
			       LatentSeq obsseq,
			       bool with_dropout = false){
    Expression obj = BuildRelaGraph(doc, cg, latseq, obsseq,
				    with_dropout);
    return obj;
  }

  /**********************************************
   * Build graph for inference
   *
   **********************************************/
  Expression BuildInferGraph(const Doc& doc,
			     ComputationGraph& cg){
    // renew the graph
    builder.new_graph(cg);
    // define expression
    Expression i_C = parameter(cg, p_C);
    Expression i_R = parameter(cg, p_R);
    // Expression i_bias = parameter(cg, p_bias);
    Expression i_context = parameter(cg, p_context);
    Expression i_L = parameter(cg, p_L);
    Expression i_lbias = parameter(cg, p_lbias);
    vector<Expression> negloglik;
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
      // For relation prediction
      Expression ylv = (i_L * cvec) + i_lbias;
      // for language modeling inference
      vector<float> prob = as_vector(softmax(ylv).value());
      vector<Expression> ccpbvec, Tkvec;
      for (unsigned latval = 0; latval < nlatvar; latval++){
	// relation specific term
	Expression i_Tk = lookup(cg, p_T, latval);
	Tkvec.push_back(i_Tk);
	// 
	Expression i_bias_k = lookup(cg, p_bias, latval);
	Expression i_TCk = lookup(cg, p_TC, latval);
	// bias term
	Expression ccpb = (i_C * (i_TCk * cvec)) + i_bias_k;
	ccpbvec.push_back(ccpb);
      }

      // --------------------------------------
      // build RNN for the current sentence
      Expression i_h_t;
      // ccpb = (i_C * (i_TCk * cvec));
      unsigned slen = sent.size() - 1;
      for (unsigned t = 0; t < slen; t++){
	// get word representation
	Expression i_x_t = lookup(cg, p_W, sent[t]);
	// compute hidden state
	i_h_t = builder.add_input(i_x_t);
	// sum over latent variable
	Expression i_y_t, i_y_tn;
	for (unsigned n = 0; n < nlatvar; n++){
	  if (n == 0){
	    i_y_tn = (i_R * (Tkvec[n] * i_h_t)) + ccpbvec[n];
	    i_y_t = i_y_tn * prob[n];
	  } else {
	    i_y_tn = (i_R * (Tkvec[n] * i_h_t)) + ccpbvec[n];
	    i_y_t = i_y_t + (i_y_tn * prob[n]);
	  }
	}
	// get prediction error
	Expression i_negloglik = pickneglogsoftmax(i_y_t,
						   sent[t+1]);
	// add back
	negloglik.push_back(i_negloglik);
      }
      // --------------------------------------
      // Keep record the last hidden state
      final_h.clear();
      final_h = as_vector(i_h_t.value());
    }
    // get result
    Expression res = sum(negloglik);
    return res;
  }
};

#endif
