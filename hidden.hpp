#ifndef LVRNN_HIDDEN_HPP
#define LVRNN_HIDDEN_HPP

#include "util.hpp"

template <class Builder>
class LVRNNHidden{
private:
  LookupParameters* p_W; // word embeddings VxK1
  LookupParameters* p_T;
  Parameters* p_R; // output weight
  Parameters* p_C; // forward context vector: VxK2
  Parameters* p_bias; // bias Vx1
  Parameters* p_context; // default context vector
  Parameters* p_L; // for latent variable prediction
  Parameters* p_lbias; // for latent variable prediction
  Builder builder;
  unsigned nlatvar;
  vector<float> final_h;
  vector<vector<float>> final_hlist;

public:
  LVRNNHidden(Model& model, unsigned nlayers, unsigned inputdim, 
	     unsigned hiddendim, unsigned vocabsize, 
	     unsigned nlatent):builder(nlayers, inputdim+hiddendim, 
					 hiddendim, &model){
    // number of latent variables
    nlatvar = nlatent;
    // word representation
    p_W = model.add_lookup_parameters(vocabsize, {inputdim});
    // output weight
    p_R = model.add_parameters({vocabsize, hiddendim});
    // prediction bias
    p_bias = model.add_parameters({vocabsize});
    // default context vector
    p_context = model.add_parameters({hiddendim});
    // latent variable distribution
    p_L = model.add_parameters({nlatent, hiddendim});
    p_lbias = model.add_parameters({nlatent});
    // transform matrix
    p_T = model.add_lookup_parameters(nlatent,
				      {inputdim+hiddendim,
				       inputdim+hiddendim});
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
			const string& flag){
    builder.new_graph(cg);
    // define expression
    Expression i_R = parameter(cg, p_R);
    Expression i_bias = parameter(cg, p_bias);
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
    Expression i_h_t;
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
      // latent variable distribution
      int latval = 0;
      if (obsseq[k] >=0){
      	latval = obsseq[k];
	Expression k_neglogprob = pickneglogsoftmax((i_L * cvec) + i_lbias, latval);
	neglogprob.push_back(k_neglogprob);
      } else {
      	latval = latseq[k];
      }
      // build RNN for the current sentence
      Expression i_x_t, i_h_t, i_y_t, i_negloglik;
      Expression i_Tk = lookup(cg, p_T, latval);
      unsigned slen = sent.size() - 1;
      for (unsigned t = 0; t < slen; t++){
	// get word representation
	i_x_t = lookup(cg, p_W, sent[t]);
	vector<Expression> vecexp;
	vecexp.push_back(i_x_t);
	vecexp.push_back(cvec);
	i_x_t = concatenate(vecexp);
	// compute hidden state
	i_h_t = builder.add_input(i_Tk * i_x_t);
	// compute prediction
	i_y_t = (i_R * i_h_t) + i_bias;
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
   * flag: what we expected to get from this function
   *       "PROB": compute the probability of the last sentence 
   *               given the latent value
   *       "ERROR": compute the prediction error of entire doc
   *       "INFER": compute prediction error on words with 
   *                inferred latent variables
   ************************************************/
  Expression BuildRelaGraph(const Doc& doc, ComputationGraph& cg,
			    LatentSeq latseq, LatentSeq obsseq){
    builder.new_graph(cg);
    // define expression
    Expression i_R = parameter(cg, p_R);
    Expression i_bias = parameter(cg, p_bias);
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
    Expression i_h_t;
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
      // two parts of the objective function
      Expression sent_objpart1;
      vector<Expression> sent_objpart2;
      for (int latval = 0; latval < nlatvar; latval ++){
	builder.start_new_sequence();
	// latent variable distribution
	vector<Expression> l_negloglik;
	Expression l_neglogprob = pickneglogsoftmax((i_L * cvec) + i_lbias, latval); 
	// build RNN for the current sentence
	Expression i_x_t, i_h_t, i_y_t, i_negloglik;
	Expression i_Tk = lookup(cg, p_T, latval);
	// for each word
	unsigned slen = sent.size() - 1;
	for (unsigned t = 0; t < slen; t++){
	  // get word representation
	  i_x_t = const_lookup(cg, p_W, sent[t]);
	  vector<Expression> vecexp;
	  vecexp.push_back(i_x_t);
	  vecexp.push_back(cvec);
	  i_x_t = concatenate(vecexp);
	  // compute hidden state
	  i_h_t = builder.add_input(i_Tk * i_x_t);
	  // compute prediction
	  i_y_t = (i_R * i_h_t) + i_bias;
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
	// - log P(Y, Z) given Y and a specific Z value
	Expression pxz = sum(l_negloglik) + l_neglogprob;
	sent_objpart2.push_back(pxz * (-1.0));
	if (obsseq[k] == latval){
	  sent_objpart1 = pxz * (-1.0);
	}
      }
      // if the latent variable is observed
      if (obsseq[k] >= 0){
	Expression sent_obj = logsumexp(sent_objpart2) - sent_objpart1;
	obj.push_back(sent_obj);
	// cout << as_scalar(sent_obj.value()) << endl;
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
    Expression i_R = input(cg, p_R->dim, as_vector(p_R->values));
    Expression i_bias = input(cg, p_bias->dim, as_vector(p_bias->values));
    Expression i_context = input(cg, p_context->dim, as_vector(p_context->values));
    Expression i_L = input(cg, p_L->dim, as_vector(p_L->values));
    Expression i_lbias = input(cg, p_lbias->dim, as_vector(p_lbias->values));
    // Initialize cvec
    Expression cvec;
    if (sidx == 0)
      cvec = i_context;
    else
      cvec = input(cg, {(unsigned)final_h.size()}, final_h);
    // compute the prob for the given latval
    Expression i_Tk = const_lookup(cg, p_T, latval);
    Expression lv_neglogprob = pickneglogsoftmax(((i_L * cvec) + i_lbias), latval);
    vector<Expression> negloglik;
    Expression i_negloglik, i_x_t, i_h_t, i_y_t;
    unsigned slen = sent.size() - 1;
    for (unsigned t = 0; t < slen; t++){
      // get word representation
      i_x_t = const_lookup(cg, p_W, sent[t]);
      vector<Expression> vecexp;
      vecexp.push_back(i_x_t);
      vecexp.push_back(cvec);
      i_x_t = concatenate(vecexp);
      // compute hidden state
      i_h_t = builder.add_input(i_Tk * i_x_t);
      // compute prediction
      i_y_t = (i_R * i_h_t) + i_bias;
      // get prediction error
      i_negloglik = pickneglogsoftmax(i_y_t, sent[t+1]);
      // push back
      negloglik.push_back(i_negloglik);
    }
    // update final_h, if latval = nlatvar - 1
    vector<float> temp_h = as_vector(i_h_t.value());
    final_hlist.push_back(temp_h);
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
      final_hlist.clear();
      for (int val = 0; val < nlatvar; val ++){
	ComputationGraph cg;
	BuildSentGraph(doc[sidx], sidx, cg, val);
	float prob = as_scalar(cg.forward());
	U.push_back(prob);
	cg.clear();
      }
      // normalize and get the argmax
      log_normalize(U);
      // greedy decoding
      int max_idx = argmax(U);
      // get the corresponding context vector
      final_h = final_hlist[max_idx];
      // 
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
			   LatentSeq obsseq){
    Expression obj = BuildGraph(doc, cg, latseq,
				obsseq, "OBJ");
    return obj;
  }

  /**********************************************
   * Build Rela Obj graph for learning
   *
   **********************************************/
  Expression BuildRelaObjGraph(const Doc& doc,
			       ComputationGraph& cg,
			       LatentSeq latseq,
			       LatentSeq obsseq){
    Expression obj = BuildRelaGraph(doc, cg, latseq, obsseq);
    return obj;
  }

  /**********************************************
   * Build graph for inference
   *
   **********************************************/
  Expression BuildInferGraph(const Doc& doc, ComputationGraph& cg,
			   LatentSeq latseq, LatentSeq obsseq){
    Expression obj = BuildGraph(doc, cg, latseq, obsseq, "INFER");
    return obj;
  }
};

#endif
