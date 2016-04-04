#ifndef LVRNN_OUTPUT_MARKOV_HPP
#define LVRNN_OUTPUT_MARKOV_HPP

#include "util.hpp"

template <class Builder>
class DCLMOutput{
private:
  LookupParameters* p_W; // word embeddings VxK1
  Parameters* p_R; // output weight
  Parameters* p_C; // forward context vector: VxK2
  LookupParameters* p_T;
  Parameters* p_bias; // bias Vx1
  Parameters* p_context; // default context vector
  Parameters* p_L; // for latent variable prediction
  Parameters* p_lbias; // for latent variable prediction
  Builder builder;
  LookupParameters* p_M; // markov transition matrix
  unsigned nlatvar; // number of latent variable
  vector<float> final_h; // context vector
  unsigned last_latval; // the last latent value

public:
  DCLMOutputMarkov(Model& model, unsigned nlayers, unsigned inputdim, 
		   unsigned hiddendim, unsigned vocabsize, 
		   unsigned nlatent):builder(nlayers, inputdim, 
					     hiddendim, &model){
    // number of latent variables
    last_latval = nlatent; // get the default value
    nlatvar = nlatent;
    // word representation
    p_W = model.add_lookup_parameters(vocabsize, {inputdim});
    // output weight
    p_R = model.add_parameters({vocabsize, hiddendim});
    // context weight
    p_C = model.add_parameters({vocabsize, hiddendim});
    // prediction bias
    p_bias = model.add_parameters({vocabsize});
    // context transform matrix
    p_T = model.add_lookup_parameters(nlatent,
				      {hiddendim,
					  hiddendim});
    p_lbias = model.add_parameters({nlatent});
    // default context vector
    p_context = model.add_parameters({hiddendim});
    // latent variable distribution
    p_L = model.add_parameters({nlatent, hiddendim});
    // markov
    p_M = model.add_lookup_parameters(nlatent+1,
				      {nlatent});
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
  Expression BuildGraph(const Doc& doc, ComputationGraph& cg,
			LatentSeq latseq, LatentSeq obsseq,
			const string& flag){
    builder.new_graph(cg);
    // define expression
    Expression i_R = parameter(cg, p_R);
    Expression i_C = parameter(cg, p_C);
    Expression i_bias = parameter(cg, p_bias);
    Expression i_context = parameter(cg, p_context);
    Expression i_L = parameter(cg, p_L);
    Expression i_lbias = parameter(cg, p_lbias);
    Expression cvec, i_x_t, i_h_t, i_y_t, ccpb, i_negloglik, k_neglogprob;
    vector<Expression> negloglik, neglogprob;
    // -----------------------------------------
    // check hidden variable list
    assert(latseq.size() <= doc.size());
    // -----------------------------------------
    // iterate over latent sequences
    // get LV-related transformation matrix
    Expression i_Tk, l_dist;
    for (unsigned k = 0; k < doc.size(); k++){
      // for each sentence in this doc
      auto& sent = doc[k];
      // using latent size as constraint
      builder.start_new_sequence();
      // start a new sequence for each sentence
      if (k == 0) cvec = i_context;
      // latent variable distribution
      int latval;
      if (obsseq[k] >=0){
	latval = obsseq[k];
      } else {
	latval = latseq[k];
      }
      i_Tk = lookup(cg, p_T, latval);
      k_neglogprob = pickneglogsoftmax((i_L * cvec) + i_lbias,
				       latval);
      // only optimize observed labels
      if (obsseq[k] >= 0){
      	neglogprob.push_back(k_neglogprob);
      }
      // neglogprob.push_back(k_neglogprob);
      // build RNN for the current sentence
      ccpb = (i_C * cvec) + i_bias;
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
	// add back
	negloglik.push_back(i_negloglik);
      }
      // update context vector
      cvec = i_h_t;
    }
    //
    Expression res;
    if ((flag != "INFER") && (flag != "OBJ")){
      cerr << "Unrecognized flag: " << flag << endl;
      abort();
    } else if ((neglogprob.size() > 0) && (flag == "OBJ")){
      // res = sum(negloglik);
      // cerr << "res.value = " << as_scalar(res.value()) << endl;
      // res = res + (sum(neglogprob) * 100.0);
      // cerr << "res.value = " << as_scalar(res.value()) << endl;
      res = sum(neglogprob)+sum(negloglik);
      // cerr << "==========" << endl;
    } else {
      res = sum(negloglik);
    }
    return res;
  }

  /*********************************************
   * Build computation graph for one sentence
   * 
   * sent: Sent instance
   *********************************************/
  Expression BuildSentGraph(const Sent& sent, const unsigned sidx,
			    ComputationGraph& cg, const int latval){
    builder.new_graph(cg);
    builder.start_new_sequence();
    // define expression
    Expression i_R = parameter(cg, p_R);
    Expression i_C = parameter(cg, p_C);
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
    // compute the prob for the given latval
    Expression i_Tk = lookup(cg, p_T, latval);
    Expression lv_neglogprob = pickneglogsoftmax(((i_L * cvec) + i_lbias),
						 latval);
    vector<Expression> negloglik;
    Expression i_negloglik, i_x_t, i_h_t, i_y_t, ccpb;
    // context + bias term
    ccpb = (i_C * cvec) + i_bias;
    // 
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
      // cerr << "update final_h when latval = " << latval << endl;
      final_h = as_vector(i_h_t.value());
    }
    //
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
      // cerr << "U = ";
      // for (auto& u : U) cerr << u << " ";
      // cerr << endl;
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
  Expression BuildObjGraph(const Doc& doc, ComputationGraph& cg,
			   LatentSeq latseq, LatentSeq obsseq){
    Expression obj = BuildGraph(doc, cg, latseq, obsseq, "OBJ");
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
