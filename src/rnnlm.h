/**
 * \file rnnlm.h
 * \defgroup lmbuilders lmbuilders
 * \brief Language models builders
 *
 * Adapted from example implementation of a simple neural language model
 * based on RNNs
 * Adapted by Cong Duy Vu Hoang (vhoang2@student.unimelb.edu.au)
 *
 */
#pragma once

#include "dynet/globals.h"
#include "dynet/nodes.h"
#include "dynet/dynet.h"
#include "dynet/training.h"
#include "dynet/timing.h"
#include "dynet/rnn.h"
#include "dynet/gru.h"
#include "dynet/lstm.h"
#include "dynet/dglstm.h"
#include "dynet/dict.h"
#include "dynet/expr.h"

#include "relopt-def.h"

#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/range/irange.hpp>

#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>

using namespace std;
using namespace dynet;

/**
 * \ingroup lmbuilders
 *
 * \struct RNNLanguageModel
 * \brief This structure wraps any RNN to train a language model with minibatching
 * \details Recurrent neural network based language modelling maximizes the likelihood
 * of a sentence \f$\textbf s=(w_1,\dots,w_n)\f$ by modelling it as :
 *
 * \f$L(\textbf s)=p(w_1,\dots,w_n)=\prod_{i=1}^n p(w_i\vert w_1,\dots,w_{i-1})\f$
 *
 * Where \f$p(w_i\vert w_1,\dots,w_{i-1})\f$ is given by the output of the RNN at step \f$i\f$
 *
 * In the case of training with minibatching, the sentences must be of the same length in
 * each minibatch. This requires some preprocessing (see `train_rnnlm-batch.cc` for example).
 * 
 * Reference : [Mikolov et al., 2010](http://www.fit.vutbr.cz/research/groups/speech/publi/2010/mikolov_interspeech2010_IS100722.pdf)
 *
 * \tparam Builder This can be any RNNBuilder
 */

int kSOS;// sentinel markers
int kEOS;
int kUNK;

namespace dynet {

template <class Builder>
struct RNNLanguageModel {

typedef std::shared_ptr<Builder> BuilderPtr;

protected:
	// Hyper-parameters
	unsigned layers_ = 2;
	unsigned input_dim_ = 8;  // 256
	unsigned hidden_dim_ = 24;  // 1024
	unsigned vocab_size_ = 0;
	bool reverse_ = false;// left-to-right (false) or right-to-left (true) direction
	float dropout_p = 0.f;

	// Parameters
	LookupParameter p_c_;
	Parameter p_R_;
	Parameter p_bias_;
	  
	// Intermediate expressions
	Expression i_R;
	Expression i_bias;
	  
	Expression i_We_;// storing word embedding matrix of all words in the vocabulary

	// RNN builder (e.g., SimpleRNN, LSTM, GRU, DGLSTM)
	//Builder rnn;
	BuilderPtr rnn_;

	// ParameterCollection pointer
	ParameterCollection* pmodel_ = nullptr;

public:
	/**
	* \brief Constructor for the batched RNN language model
	*
	* \param model Model to hold all parameters for training
	* \param layers_ Number of layers of the RNN
	* \param input_dim_ Embedding dimension for the words
	* \param hidden_dim_ Dimension of the hidden states
	* \param vocab_size_ Size of the input vocabulary
	*/
	explicit RNNLanguageModel(){}

	explicit RNNLanguageModel(ParameterCollection* model,
		unsigned layers,
		unsigned input_dim,
		unsigned hidden_dim,
		unsigned vocab_size, 
		bool reverse) : layers_(layers), input_dim_(input_dim),
		hidden_dim_(hidden_dim), vocab_size_(vocab_size), reverse_(reverse) 
	{
		// Add embedding parameters to the model
		p_c_ = model->add_lookup_parameters(vocab_size_, {input_dim_});
		p_R_ = model->add_parameters({vocab_size_, hidden_dim_});
		p_bias_ = model->add_parameters({vocab_size_});
		rnn_ = BuilderPtr(new Builder(layers_, input_dim_, hidden_dim_, *model)); 
		pmodel_ = model;
	}

	void CreateModel(ParameterCollection* model,
		unsigned layers,
		unsigned input_dim,
		unsigned hidden_dim,
		unsigned vocab_size, 
		bool reverse)
	{
		layers_ = layers;
		input_dim_ = input_dim;
		hidden_dim_ = hidden_dim;
		vocab_size_ = vocab_size;
		reverse_ = reverse;
	
		// Add embedding parameters to the model
		p_c_ = model->add_lookup_parameters(vocab_size_, {input_dim_});
		p_R_ = model->add_parameters({vocab_size_, hidden_dim_});
		p_bias_ = model->add_parameters({vocab_size_});
		rnn_ = BuilderPtr(new Builder(layers_, input_dim_, hidden_dim_, *model)); 

		pmodel_ = model;
	}

	void LoadModel(ParameterCollection* model, const std::string& model_file){
		cerr << "Loading model from: " << model_file << endl;

		ifstream in(model_file + ".cfg");
		boost::archive::text_iarchive ia(in);
		ia >> layers_ >> input_dim_ >> hidden_dim_ >> vocab_size_ >> dropout_p >> reverse_;

		// Add embedding parameters to the model
		p_c_ = model->add_lookup_parameters(vocab_size_, {input_dim_});
		p_R_ = model->add_parameters({vocab_size_, hidden_dim_});
		p_bias_ = model->add_parameters({vocab_size_});
		rnn_ = BuilderPtr(new Builder(layers_, input_dim_, hidden_dim_, *model)); 
		pmodel_ = model;
		
		dynet::load_dynet_model(model_file, pmodel_);// FIXME: use binary streaming instead for saving disk spaces
	}

	// an alternative of << operator
	void SaveModel(const std::string& model_file){
		ofstream out(model_file + ".cfg");
		boost::archive::text_oarchive oa(out);
		oa << layers_ << input_dim_ << hidden_dim_ << vocab_size_ << dropout_p << reverse_;
		
		// dynet v2
		dynet::save_dynet_model(model_file, pmodel_);// FIXME: use binary streaming instead for saving disk spaces
	}

	void DisableDropout(){
		rnn_->disable_dropout();
	}

	void EnableDropout(float p){
		rnn_->set_dropout(p);
	}

	/**
	* \brief Computes the negative log probability on a batch
	*
	* \param sents Full training set
	* \param id Start index of the batch
	* \param bsize Batch size (`id` + `bsize` should be smaller than the size of the dataset)
	* \param tokens Number of tokens processed by the model (used for loos per token computation)
	* \param cg Computation graph
	* \return Expression for $\f$\sum_{s\in\mathrm{batch}}\log(p(s))\f$
	*/
	Expression BuildLMGraph(const vector<vector<int> >& sents,
		unsigned sid/*sentence ID in a mini-batch*/,
		unsigned bsize/*mini-batch size*/,
		unsigned & tokens/*token count*/,
		unsigned & unk_tokens/*<unk> token count*/,
		ComputationGraph& cg) 
	{
		const unsigned slen = sents[sid].size();
	
		// Initialize the RNN for a new computation graph
		rnn_->new_graph(cg);
	
		// Prepare for new sequence (essentially set hidden states to 0)
		rnn_->start_new_sequence();
	
		// Instantiate embedding parameters in the computation graph
		// output -> word rep parameters (matrix + bias)
		i_R = parameter(cg, p_R_);
		i_bias = parameter(cg, p_bias_);
	
		// Initialize variables for batch errors
		vector<Expression> errs;
	
		// Set all inputs to the SOS symbol
		vector<unsigned> last_arr(bsize, sents[0][0]/*SOS*/), next_arr(bsize);
	
		// Run rnn on batch
		for (unsigned t = 1; t < slen; ++t) {
			auto ct = t;
			if (reverse_ == true) ct = slen - t - 1;// right-to-left direction

			// Fill next_arr (tokens to be predicted)
			for (unsigned i = 0; i < bsize; ++i) {
				if (reverse_ == true && ct == 0/*EOS*/)
					next_arr[i] = sents[sid + i][t];
				else
					next_arr[i] = sents[sid + i][ct];
		  
			  	// count non-EOS tokens
				if (next_arr[i] != (unsigned)*sents[sid].rbegin()) tokens++;
				if (next_arr[i] == (unsigned)kUNK) unk_tokens++;
			}
		  
			// Embed the current tokens
			Expression i_x_t = lookup(cg, p_c_, last_arr);
			 
			// Run one step of the rnn : y_t = RNN(x_t)
			Expression i_y_t = rnn_->add_input(i_x_t);
	
			// Project to the token space using an affine transform
			Expression i_r_t = i_bias + i_R * i_y_t;
			 
			// Compute error for each member of the batch
			Expression i_err = pickneglogsoftmax(i_r_t, next_arr);
			errs.push_back(i_err);
			 
			// Change input
			last_arr = next_arr;
		}
	
		// Add all errors
		Expression i_nerr = sum_batches(sum(errs));
		return i_nerr;
	}

	Expression BuildLMGraph(const vector<vector<int>>& sents,
		unsigned & tokens/*token count*/,
		unsigned & unk_tokens/*<unk> token count*/,
		ComputationGraph& cg) 
	{
		// Initialize the RNN for a new computation graph
		rnn_->new_graph(cg);
	
		// Prepare for new sequence (essentially set hidden states to 0)
		rnn_->start_new_sequence();
	
		// Instantiate embedding parameters in the computation graph
		// output -> word rep parameters (matrix + bias)
		i_R = parameter(cg, p_R_);
		i_bias = parameter(cg, p_bias_);

		std::vector<Expression> errs;

		const unsigned len = sents[0].size() - 1; 
		std::vector<unsigned> next_words(sents.size()), words(sents.size());

		for (unsigned t = 0; t < len; ++t) {
			auto ct = t;
			if (reverse_ == true) ct = len - t - 1;// right-to-left direction

			for(size_t bs = 0; bs < sents.size(); bs++){
				words[bs] = (sents[bs].size() > t) ? ((reverse_==true && t != 0/*BOS*/) ? (unsigned)sents[bs][ct+1] : (unsigned)sents[bs][t]) : kEOS;
				next_words[bs] = (sents[bs].size() > (t+1)) ? ((reverse_==true && ct != 0/*EOS*/) ? (unsigned)sents[bs][ct] : (unsigned)sents[bs][t+1]) : kEOS;
				if (sents[bs].size() > t) {
					tokens++;
					if (sents[bs][t] == kUNK) unk_tokens++;
				}
			}
			
			// Embed the current tokens
			Expression i_x_t = lookup(cg, p_c_, words);
		 
			// Run one step of the rnn : {y_t} = RNN({x_t})
			Expression i_y_t = rnn_->add_input(i_x_t);
	
			// Project to the token space using an affine transform
			Expression i_r_t = i_bias + i_R * i_y_t;
		 
			// Compute error for each member of the batch
			Expression i_err = pickneglogsoftmax(i_r_t, next_words);
	
			errs.push_back(i_err);
		}

		// Add all errors
		Expression i_nerr = sum_batches(sum(errs));
		return i_nerr;
	}

	Expression BuildLMGraph(const vector<int>& sent,
		unsigned & tokens/*token count*/,
		unsigned & unk_tokens/*<unk> token count*/,
		ComputationGraph& cg) 
	{
		const unsigned slen = sent.size();
		std::vector<int> rsent = sent;
		if (reverse_ == true) std::reverse(rsent.begin() + 1/*BOS*/, rsent.end() - 1/*EOS*/);;
	
		// Initialize the RNN for a new computation graph
		rnn_->new_graph(cg);
	
		// Prepare for new sequence (essentially set hidden states to 0)
		rnn_->start_new_sequence();
	
		// Instantiate embedding parameters in the computation graph
		// output -> word rep parameters (matrix + bias)
		i_R = parameter(cg, p_R_);
		i_bias = parameter(cg, p_bias_);
	
		vector<Expression> errs;
		
		// Run rnn on batch
		for (unsigned t = 0; t < slen - 1; ++t) {
			// Count non-EOS words
			tokens++;
			if (rsent[t] == kUNK) unk_tokens++;
	
			// Embed the current tokens
			Expression i_x_t = lookup(cg, p_c_, rsent[t]);
		 
			// Run one step of the rnn : y_t = RNN(x_t)
			Expression i_y_t = rnn_->add_input(i_x_t);
	
			// Project to the token space using an affine transform
			Expression i_r_t = i_bias + i_R * i_y_t;
		 
			// Compute error
			Expression i_err = pickneglogsoftmax(i_r_t, rsent[t + 1]);
		  
			errs.push_back(i_err);
		}
	
		// Add all errors
		Expression i_nerr = sum(errs);
		return i_nerr;
	}

	Expression ComputeNLL(const vector<int>& sent,
		ComputationGraph& cg) 
	{
		const unsigned slen = sent.size();
		std::vector<int> rsent = sent;
		if (reverse_ == true) std::reverse(rsent.begin() + 1/*BOS*/, rsent.end() - 1/*EOS*/);;
	
		// Initialize the RNN for a new computation graph
		rnn_->new_graph(cg);
	
		// Prepare for new sequence (essentially set hidden states to 0)
		rnn_->start_new_sequence();
	
		// Instantiate embedding parameters in the computation graph
		// output -> word rep parameters (matrix + bias)
		i_R = parameter(cg, p_R_);
		i_bias = parameter(cg, p_bias_);
	
		vector<Expression> errs;
		
		// Run rnn on batch
		for (unsigned t = 0; t < slen - 1; ++t) {	
			// Embed the current tokens
			Expression i_x_t = lookup(cg, p_c_, rsent[t]);
		 
			// Run one step of the rnn : y_t = RNN(x_t)
			Expression i_y_t = rnn_->add_input(i_x_t);
	
			// Project to the token space using an affine transform
			Expression i_r_t = i_bias + i_R * i_y_t;
		 
			// Compute error 
			Expression i_err = pickneglogsoftmax(i_r_t, rsent[t + 1]);
		  
			errs.push_back(i_err);
		}
	
		// Add all errors
		Expression i_nerr = sum(errs);
		return i_nerr;
	}

	//---------------------------------------------------------------------------------------------  
	// Build the relaxation optimization graph for the given sentence including returned loss
	void ComputeWordEmbeddingMatrix(ComputationGraph& cg)
	{
		std::vector<Expression> vEs(vocab_size_);
		for (unsigned i = 0; i < vocab_size_; i++)
			vEs[i] = lookup(cg, p_c_, i);//hidden_dim x 1
		i_We_ = concatenate_cols(vEs);/*hidden_dim x vocab_size_*/
	}
  
	Expression GetWordEmbeddingVector(const Expression& i_y)
	{
		// expected embedding
		return (i_We_/*hidden_dim x vocab_size_*/ * i_y/*vocab_size_ x 1*/);//hidden_dim x 1
	}

	Expression BuildRelOptGraph( size_t algo
		, std::vector<dynet::Parameter>& v_params
		, ComputationGraph& cg
		, dynet::Dict &d) 
	{
		// Initialize the RNN for a new computation graph
		//cerr << "BuildRelOptGraph::1" << endl;
		rnn_->new_graph(cg);

		// Prepare for new sequence (essentially set hidden states to 0)
		//cerr << "BuildRelOptGraph::2" << endl;
		rnn_->start_new_sequence();

		// Instantiate embedding parameters in the computation graph
		//cerr << "BuildRelOptGraph::3" << endl;
		int slen = v_params.size();// desired target length (excluding BOS and EOS tokens)
		int ind_bos = d.convert("<s>"), ind_eos = d.convert("</s>");

		std::vector<Expression> v_i_params(slen);
		std::vector<Expression> i_wes(slen + 1);
		i_wes[0] = lookup(cg, p_c_, ind_bos);// known BOS embedding
		for(auto t : boost::irange(0, slen)){
			auto ct = t;
			if (reverse_ == true) ct = slen - t - 1;

			if (algo == RELOPT_ALGO::SOFTMAX){// SOFTMAX approach
				Expression i_p = parameter(cg, v_params[ct]);
				v_i_params[t] = softmax(i_p);
			}
			else if (algo == RELOPT_ALGO::SPARSEMAX){// SPARSEMAX approach
				Expression i_p = parameter(cg, v_params[ct]);
				v_i_params[t] = sparsemax(i_p);
			}
			else if (algo == RELOPT_ALGO::EG || algo == RELOPT_ALGO::AEG){// EG or AEG approach
				Expression i_p = parameter(cg, v_params[ct]);
				v_i_params[t] = i_p;
			}
			else
				assert("Unknown relopt algo! Failed!");

			i_wes[t + 1] = GetWordEmbeddingVector(v_i_params[t]);
		}
	
		// output -> word rep parameters (matrix + bias)
		//cerr << "BuildRelOptGraph::4" << endl;
		i_R = parameter(cg, p_R_);
		i_bias = parameter(cg, p_bias_);

		vector<Expression> costs;

		// Run RNN
		//cerr << "BuildRelOptGraph::5" << endl;
		for (auto t = 0; t < slen + 1; ++t) {
			// Embed the current tokens
			//cerr << "BuildRelOptGraph::5(a)" << endl;
			Expression i_x_t = i_wes[t];

			// Run one step of the rnn : y_t = RNN(x_t)
			//cerr << "BuildRelOptGraph::5(b)" << endl;
			Expression i_y_t = rnn_->add_input(i_x_t);

			// Project to the token space using an affine transform
			//cerr << "BuildRelOptGraph::5(c)" << endl;
			Expression i_r_t = affine_transform({i_bias, i_R, i_y_t});//i_bias + i_R * i_y_t;

			// Run the softmax and calculate the cost
			//cerr << "BuildRelOptGraph::5(d)" << endl;
			Expression i_cost;
			if (t >= slen){// for predicting EOS
				i_cost = pickneglogsoftmax(i_r_t, ind_eos);
			}
			else{// for predicting others
				//i_cost = -log(transpose(i_y) * i_softmax);
				i_cost = -transpose(v_i_params[t]) * log_softmax(i_r_t);
			}

			costs.push_back(i_cost);
		}

		//cerr << "BuildRelOptGraph::6" << endl;
		return sum(costs);
	}
  	//---------------------------------------------------------------------------------------------  

	/**
	* \brief Samples a string of words/characters from the model
	* \details This can be used to debug and/or have fun. Try it on
	* new datasets!
	*
	* \param d Dictionary to use (should be same as the one used for training)
	* \param max_len maximu number of tokens to generate
	* \param temp Temperature for sampling (the softmax computed is
	* \f$\frac{e^{\frac{r_t^{(i)}}{T}}}{\sum_{j=1}^{\vert V\vert}e^{\frac{r_t^{(j)}}{T}}}\f$).
	*  Intuitively lower temperature -> less deviation from the distribution (= more "standard" samples)
	*/
	void RandomSample(dynet::Dict& d, int max_len = 150, float temp = 1.0) {
		int kSOS = d.convert("<s>");
		int kEOS = d.convert("</s>");
		  
		// Make some space
		cerr << endl;
		// Initialize computation graph
		ComputationGraph cg;
		// Initialize the RNN for the new computation graph
		rnn_->new_graph(cg);
		// Initialize for new sequence (set hidden states, etc..)
		rnn_->start_new_sequence();
		// Instantiate embedding parameters in the computation graph
		// output -> word rep parameters (matrix + bias)
		Expression i_R = parameter(cg, p_R_);
		Expression i_bias = parameter(cg, p_bias_);

		// Start generating
		int len = 0;
		int cur = kSOS;
		while (len < max_len) {
			++len;
			// Embed current token
			Expression i_x_t = lookup(cg, p_c_, cur);
			// Run one step of the rnn
			// y_t = RNN(x_t)
			Expression i_y_t = rnn_->add_input(i_x_t);
			// Project into token space
			Expression i_r_t = i_bias + i_R * i_y_t;
			// Get distribution over tokens (with temperature)
			Expression ydist = softmax(i_r_t / temp);

			// Sample token
			unsigned w = 0;
			while (w == 0 || (int)w == kSOS) {
				auto dist = as_vector(cg.incremental_forward(ydist));
				double p = rand01();
				for (; w < dist.size(); ++w) {
					p -= dist[w];
					if (p < 0.0) { break; }
				}
				if (w == dist.size()) w = kEOS;
			}

			if (w == kEOS) {
				// If the sampled token is an EOS, reinitialize network and start generating a new sample
				rnn_->start_new_sequence();
				cerr << endl;
				cur = kSOS;
			} 
			else {
				// Otherwise print token and continue
				cerr << (cur == kSOS ? "" : " ") << d.convert(w);
				cur = w;
			}
		}

		cerr << endl;
	}
};

}; // namespace dynet

