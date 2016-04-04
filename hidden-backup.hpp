#ifndef LVRNN_HIDDEN_HPP
#define LVRNN_HIDDEN_HPP

#include "util.hpp"

template <class Builder>
class LVRNNHidden{
private:
  LookupParameters* p_W; // word embeddings VxK1
  Parameters* p_R; // hidden weights
  LookupParameters* p_T; // forward context vector: VxK2
  Parameters* p_bias; // bias Vx1
  Parameters* p_context; // default context vector
  Parameters* p_L; // for latent variable prediction
  Parameters* p_lbias; // for latent variable prediction
  Builder builder;
  // for sampling
  Particles particles;
  unsigned nlatent;
  // for keeping the last state of each hidden layer
  //   used to initialize start_new_sequence
  vector<vector<float>> h_vec;
  vector<float> final_h, U;

public:
  LVRNNHidden(Model& model, unsigned nlayers, unsigned inputdim, 
	     unsigned hiddendim, unsigned vocabsize, 
	     unsigned nlatent=4):builder(nlayers,
					 inputdim+hiddendim, 
					 hiddendim, &model){
    // number of latent variables
    nlatent = nlatent;
    // word representation
    p_W = model.add_lookup_parameters(vocabsize, {inputdim});
    // output weight
    p_R = model.add_parameters({vocabsize, hiddendim}, 1e-9);
    // context transform matrix
    p_T = model.add_lookup_parameters(nlatent,
				      {inputdim+hiddendim,
					  inputdim+hiddendim});
    // prediction bias
    p_bias = model.add_parameters({vocabsize}, 1e-9);
    // default context vector
    p_context = model.add_parameters({hiddendim}, 1e-9);
    // latent variable distribution
    p_L = model.add_parameters({nlatent,hiddendim}, 1e-9);
    p_lbias = model.add_parameters({nlatent}, 1e-9);
  }

  /************************************************
   * Build CG of a given doc with a latent sequence
   *
   * doc: 
   * cg: computation graph
   * latseq: latent sequence
   * obsseq: partially observed sequence
   * flag: what we expected to get from this function
   *       "PROB": compute the probability of the last sentence 
   *               given the latent value
   *       "ERROR": compute the prediction error of entire doc
   *       "INFER": compute prediction error on words with 
   *                inferred latent variables
   ************************************************/
  Expression BuildGraph(const Doc doc, ComputationGraph& cg,
			LatentSeq latseq, LatentSeq obsseq,
			string flag = "ERROR"){
    // define expression
    Expression i_R = parameter(cg, p_R);
    Expression i_bias = parameter(cg, p_bias);
    Expression i_context = parameter(cg, p_context);
    Expression i_L = parameter(cg, p_L);
    Expression i_lbias = parameter(cg, p_lbias);
    // vector<float> tmp_bias = as_vector(i_lbias.value());
    // for (int i = 0; i < tmp_bias.size(); i++) cerr << "b[i] = " << tmp_bias[i] << " ";
    // cerr<<endl;
    Expression cvec, i_x_t, i_h_t, i_y_t, i_err, l_err, ccpb;
    vector<Expression> i_errs, l_errs, neglogprob;
    // -----------------------------------------
    // check hidden variable list
    assert(latseq.size() <= doc.size());
    // -----------------------------------------
    // build CG for the doc
    int nlatseq = latseq.size();
    // iterate over latent sequences
    // get LV-related transformation matrix
    Expression i_Tk, l_dist;
    // default context vector
    cvec = i_context;
    for (unsigned k = 0; k < nlatseq; k++){
      // for each sentence in this doc
      auto sent = doc[k];
      // using latent size as constraint
      builder.start_new_sequence();
      // compute latent-variable related transformation
      if ((flag == "ERROR") || (flag == "PROB") ||
	  (flag == "INFER")){
	// pick one p_T
	i_Tk = lookup(cg, p_T, latseq[k]);
	l_err = pickneglogsoftmax((i_L*cvec + i_lbias), latseq[k]);
	// l_err = pickneglogsoftmax((i_L*cvec), latseq[k]);
	if (obsseq[k] >= 0){
	  l_errs.push_back(l_err);
	}
	// l_errs.push_back(l_err);
      } else {
	cout << "Unrecognized flag: " << flag << endl;
	abort();
      }
      // build RNN for the current sentence
      // transformed with lv-related matrix
      // cvec = i_Tk * cvec;
      vector<Expression> vec_exp;
      unsigned slen = sent.size() - 1;
      for (unsigned t = 0; t < slen; t++){
	// get word representation
	i_x_t = lookup(cg, p_W, sent[t]);
	vec_exp.clear();
	vec_exp.push_back(i_x_t);
	vec_exp.push_back(cvec);
	i_x_t = concatenate(vec_exp);
	// compute hidden state
	i_h_t = builder.add_input(i_Tk * i_x_t);
	// compute prediction
	i_y_t = (i_R * i_h_t) + i_bias;
	// get prediction error
	i_err = pickneglogsoftmax(i_y_t, sent[t+1]);
	// push back
	i_errs.push_back(i_err);
	if (k == nlatseq - 1){
	  // only if this the last sentence
	  // the last latent variable in the seq
	  neglogprob.push_back(i_err);
	}
      }
      // update context vector
      cvec = i_h_t;
    }
    Expression res;
    if (flag == "ERROR"){
      // sum over all the prediction neglogprob
      // do we need a weight lambda here to balance
      // the loss
      // if there is any partially observed latent values
      // optimize the corresponding node too
      if (l_errs.size() > 0){
	res = sum(i_errs) + sum(l_errs);
      } else {
	res = sum(i_errs);
      }
    } else if (flag == "PROB"){
      // compute the likelihood given the value of 
      //   the last hidden variable
      res = (sum(neglogprob) + l_err) * (-1.0);
    } else if (flag == "INFER"){
      // only return i_errs
      res = sum(i_errs);
    }
    return res;
  }

  /*********************************************
   *
   *********************************************/
  Expression BuildSentGraph(const Sent& sent, const unsigned sidx,
			    ComputationGraph& cg, const int latval,
			    const unsigned partidx){
    // define expression
    Expression i_R = parameter(cg, p_R);
    Expression i_bias = parameter(cg, p_bias);
    Expression i_context = parameter(cg, p_context);
    Expression i_L = parameter(cg, p_L);
    Expression i_lbias = parameter(cg, p_lbias);
    Expression cvec, lv_neglogprob;
    vector<Expression> init_states;
    // initialize cvec and assign values to states
    if (sidx == 0){
      cvec = i_context;
      // builder.start_new_sequence();
    } else {
      // get value of cvec
      cvec = input(cg, {(unsigned)h_vec[partidx].size()},
		   h_vec[partidx]);
    }
    builder.start_new_sequence();
    // get lv-related transformation matrix
    Expression i_Tk = lookup(cg, p_T, latval);
    // get prob of latval given cvec
    lv_neglogprob = pickneglogsoftmax((i_L*cvec + i_lbias), latval);
    // lv_neglogprob = pickneglogsoftmax((i_L*cvec), latval);
    // tranform cvec with the transformation matrix
    // cvec = i_Tk * cvec;
    // start building sentence graph
    vector<Expression> vec_exp, neglogprob;
    Expression i_neglogprob, i_x_t, i_h_t, i_y_t;
    unsigned slen = sent.size() - 1;
    for (unsigned t = 0; t < slen; t++){
      i_x_t = lookup(cg, p_W, sent[t]);
      // concatenate x_t and h_{t-1}
      vec_exp.clear();
      vec_exp.push_back(i_x_t);
      vec_exp.push_back(cvec);
      i_x_t = concatenate(vec_exp);
      // compute hidden state
      i_h_t = builder.add_input(i_Tk * i_x_t);
      // compute prediction
      i_y_t = (i_R * i_h_t) + i_bias;
      // get prediction error
      i_neglogprob = pickneglogsoftmax(i_y_t, sent[t+1]);
      // push back
      neglogprob.push_back(i_neglogprob);
    }
    // update context vector (in vector form)
    h_vec[partidx] = as_vector(i_h_t.value());
    // get prob
    Expression res;
    // cout << "neglogprob = " << exp(-as_scalar(lv_neglogprob.value())) << endl;
    res = (sum(neglogprob) + lv_neglogprob) * (-1.0);
    // cout << "res = " << as_scalar(res.value()) << endl;
    return res;
  }
  
  /*********************************************
   * Sample particles for a given document
   * 
   * doc: 
   * priorprob: vector of float
   * npart: number of particles
   *********************************************/
  Particles SampleGraph(const Doc doc, Prob proprob, 
			unsigned npart,
			const LatentSeq& obsseq){
    // not necessary to new_graph() here !!!
    // ----------------------------------------
    // predefined variable
    int nsent = doc.size();
    // check length
    assert(nsent == obsseq.size());
    // clean particles if it has any
    particles.clear();
    U.clear();
    h_vec.clear();
    // initialize particles
    for (int np = 0; np < npart; np++){
      LatentSeq latseq;
      // init particles 
      particles.push_back(latseq);
      // init particle weights
      U.push_back(0);
      // init h_vec
      vector<float> h = vector<float>(nlatent, 0.0);
      h_vec.push_back(h);
    }
    // ----------------------------------------
    // start sampling
    Particles newparticles;
    vector<float> W = vector<float>(npart, 0.0);
    for (unsigned si = 0; si < nsent; si++){
      // sample z from proposal dist for the first sentences
      // sample from prior prob
      vector<int> Z = get_randnums(proprob, npart);
      // distribute Z to each particle
      for (int np = 0; np < npart; np++){
	if (obsseq[si] >= 0){
	  // if the hidden value is observed
	  particles[np].push_back(obsseq[si]);
	} else {
	  // if not
	  particles[np].push_back(Z[np]);
	}
      }
      // compute weights U
      // cout << "compute weights ..." << endl;
      float u; int latval;
      for (unsigned pidx = 0; pidx < npart; pidx ++){
	// LatentSeq latseq = particles[np];
	latval = particles[pidx][si];
	ComputationGraph cg;
	// Renew this cg before using it 
	//   (actually, not necessary here)
	builder.new_graph(cg);
	// build computation graph for each particle
	//   then get prob for resampling
	BuildSentGraph(doc[si], si, cg, latval, pidx);
	// push back prob into U
	u = as_scalar(cg.forward());
	U[pidx] = U[pidx] + u;
	cg.clear();
      }
      // cout << "h_vec.size = " << h_vec.size() << endl;
      // Well, to avoid overflow or underfolow
      log_normalize(U);
      // get sampleing weights W
      cout << "W = ";
      for (unsigned np = 0; np < npart; np++){
      	W[np] = exp(U[np]);
	cout << W[np] << " ";
      }
      cout << endl;
      // sample particle indices with the weights
      //   without normalizing
      cout << "before resampling ... " << endl;
      printparticles(particles, obsseq);
      if (l2_norm(W) < 0.9){
	vector<int> resampled_pidx = get_randnums(W, npart);
	// resample particles with the indices
	newparticles.clear();
	for (auto& idx : resampled_pidx){
	  cout << idx << " ";
	  newparticles.push_back(particles[idx]);
	}
	cout << endl;
	// update particles
	particles = newparticles;
      }
      else {
      	cout << "Skip the resampling step ..." <<endl;
      }
      cout << "after resampling ... " << endl;
      printparticles(particles, obsseq);
    }
    // ----------------------------------------------
    // Now, we have particles for the entire sequence
    // for (auto& w : W) cout << w << " "; cout << endl;
    // cout << "||W||_2 = " << l2_norm(W) << endl;
    return particles;
  }

  /***************************************************
   * Build objective function with the given particles 
   *   for the full sequence
   * 
   * doc: a training document
   * cg: computation graph
   * particles: particle samples from SampleGraph()
   ***************************************************/
  Expression BuildObjGraph(const Doc doc, ComputationGraph& cg, 
			   LatentSeq latseq){
    // cout << "Start building obj graph" << endl;
    builder.new_graph(cg);
    // Expression err;
    // vector<Expression> errs;
    // for (auto& latseq : particles){
    //   err = BuildGraph(doc, cg, latseq, "ERROR");
    //   errs.push_back(err);
    // }
    // Expression obj = (sum(errs) / (float)particles.size());
    Expression obj = BuildGraph(doc, cg, latseq, "ERROR");
    return obj;
  }

  Expression BuildObjGraph(const Doc doc, ComputationGraph& cg, 
			   Particles particles, LatentSeq obsseq){
    // cout << "Start building obj graph" << endl;
    builder.new_graph(cg);
    Expression err;
    vector<Expression> errs;
    for (auto& latseq : particles){
      err = BuildGraph(doc, cg, latseq, obsseq, "ERROR");
      errs.push_back(err);
    }
    Expression obj = (sum(errs) / (float)particles.size());
    return obj;
  }

  /**************************************************
   * Build cg for inference on dev or test
   *
   * doc: a dev or test document
   * cg: computation graph
   * comment: now i use BuildObjGraph with particles to
   *          compute the dev performance, as i noticed 
   *          that use expectation is not as good as i 
   *          expected. Still working on this part now
   *          (11-02-2015)
   **************************************************/
  Expression BuildInferGraph(const Doc doc, ComputationGraph& cg,
			     Particles particles, LatentSeq obsseq){
    builder.new_graph(cg);
    Expression err;
    vector<Expression> errs;
    for (auto& latseq: particles){
      err = BuildGraph(doc, cg, latseq, obsseq, "INFER");
      errs.push_back(err);
    } 
    Expression inferrs = (sum(errs) / (float)particles.size());
    return inferrs;
  }

};

#endif
