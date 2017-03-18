#pragma once

#include "dynet/nodes.h"
#include "dynet/dynet.h"
#include "dynet/training.h"
#include "dynet/timing.h"
#include "dynet/rnn.h"
#include "dynet/gru.h"
#include "dynet/lstm.h"
//#include "dynet/dglstm.h" // FIXME: add this to dynet?
#include "dynet/dict.h"
#include "dynet/expr.h"
#include "expr-xtra.h"

#include "relopt-def.h"

#include "dict-utils.h"

#include <algorithm>
#include <iostream>
#include <fstream>
#include <sstream>

#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/range/irange.hpp>

#define RNN_H0_IS_ZERO

unsigned SLAYERS = 1; // 2
unsigned TLAYERS = 1; // 2
unsigned HIDDEN_DIM = 64; // 1024
unsigned ALIGN_DIM = 32; // 128

unsigned SRC_VOCAB_SIZE = 0;
unsigned TGT_VOCAB_SIZE = 0;

int kSRC_SOS;
int kSRC_EOS;
int kSRC_UNK;
int kTGT_SOS;
int kTGT_EOS;
int kTGT_UNK;

using namespace std;

namespace dynet {

struct ModelStats {
	double loss = 0.0f;
	unsigned words_src = 0;
	unsigned words_tgt = 0;
	unsigned words_src_unk = 0;
	unsigned words_tgt_unk = 0;

	ModelStats(){}
};

template <class Builder>
struct AttentionalModel {
	explicit AttentionalModel(dynet::Model* model,
		unsigned _src_vocab_size, unsigned _tgt_vocab_size, unsigned slayers, unsigned tlayers, unsigned hidden_dim, 
		unsigned align_dim, bool _rnn_src_embeddings, bool _giza_positional, 
		bool _giza_markov, bool _giza_fertility, bool _doc_context,
		bool _global_fertility,
		LookupParameter* _p_cs=nullptr, LookupParameter* _p_ct=nullptr);

	~AttentionalModel();

	Expression BuildGraph(const std::vector<int>& source, const std::vector<int>& target, 
		ComputationGraph& cg, ModelStats& tstats, Expression* alignment=0, const std::vector<int>* ctx=0,
		Expression *coverage=0, Expression *fertility=0);
	void BuildGraph(const std::vector<int>& source, const std::vector<int>& target, 
		ComputationGraph& cg, std::vector<std::vector<float>>& v_preds, bool with_softmax=true);
	Expression BuildGraph_Batch(const std::vector<std::vector<int>>& sources, const std::vector<std::vector<int>>& targets, 
		ComputationGraph& cg, ModelStats& tstats
		, Expression *coverage=0, Expression *fertility=0);// for supporting mini-batch training
	Expression Forward(const std::vector<int> & sent, int t
		, bool log_prob
		, RNNPointer& prev_state, RNNPointer& state
		, dynet::ComputationGraph & cg
		, std::vector<Expression> & align_out);

	//---------------------------------------------------------------------------------------------
	// Build the relaxation optimization graph for the given sentence including returned loss
	// (Hoang et al., 2017; https://arxiv.org/abs/1701.02854)
	void StartNewInstance(size_t algo
		, std::vector<dynet::Parameter>& v_params
		, Dict &sd /*source vocabulary*/
		, ComputationGraph &cg);
	Expression AddInput(const Expression& i_ewe_t
		, int t
		, ComputationGraph &cg
		, RNNPointer *prev_state=0);
	void ComputeTrgWordEmbeddingMatrix(dynet::ComputationGraph& cg);
	void ComputeSrcWordEmbeddingMatrix(dynet::ComputationGraph& cg);
	Expression GetWordEmbeddingVector(
		const Expression& i_y);
	Expression BuildRelOptGraph(
		size_t algo
		, std::vector<dynet::Parameter>& v_params /*target*/
		, dynet::ComputationGraph & cg
		, Dict &d
		, bool reverse = false
		, Expression *entropy=0 /*entropy regularization*/
		, Expression *alignment=0 /*soft alignment*/
		, Expression *coverage=0, float coverage_C=1.f /*coverage penalty*/
		, Expression *fertility=0/*global fertility model*/);
	Expression BuildRevRelOptGraph(
		size_t algo
		, std::vector<dynet::Parameter>& v_params /*source*/
		, const std::vector<int>& target
		, dynet::ComputationGraph & cg
		, Dict &sd
		, Expression *alignment=0);
	std::string GetRelOptOutput(dynet::ComputationGraph& cg
			 , const std::vector<dynet::Parameter>& v_relopt_params, size_t algo, Dict &d, bool verbose=false);
	Expression i_We;// word embedding matrix
	//---------------------------------------------------------------------------------------------

	// enable/disable dropout for source and target RNNs following Gal et al., 2016
	void Set_Dropout(float do_enc, float do_dec);
	void Enable_Dropout();
	void Disable_Dropout();
	
	void Display_ASCII(const std::vector<int> &source, const std::vector<int>& target, 
		ComputationGraph& cg, const Expression& alignment, Dict &sd, Dict &td);

	void Display_TIKZ(const std::vector<int> &source, const std::vector<int>& target, 
		ComputationGraph& cg, const Expression& alignment, Dict &sd, Dict &td);

	void Display_Fertility(const std::vector<int> &source, Dict &sd);

	void Display_Empirical_Fertility(const std::vector<int> &source, const std::vector<int> &target, Dict &sd);

	std::vector<int> Greedy_Decode(const std::vector<int> &source, ComputationGraph& cg, 
		Dict &tdict, const std::vector<int>* ctx=0);

	std::vector<int> Beam_Decode(const std::vector<int> &source, ComputationGraph& cg, 
		unsigned beam_width, Dict &tdict, const std::vector<int>* ctx=0);

	std::vector<int> Sample(const std::vector<int> &source, ComputationGraph& cg, 
		Dict &tdict, const std::vector<int>* ctx=0);

	void Add_Global_Fertility_Params(dynet::Model* model, unsigned hidden_dim, bool _rnn_src_embeddings);

	LookupParameter p_cs;// source vocabulary lookup
	LookupParameter p_ct;// target vocabulary lookup
	Parameter p_R;
	Parameter p_Q;
	Parameter p_P;
	Parameter p_S;
	Parameter p_bias;
	Parameter p_Wa;
	std::vector<Parameter> p_Wh0;
	Parameter p_Ua;
	Parameter p_va;
	Parameter p_Ta;
	Parameter p_Wfhid;
	Parameter p_Wfmu;
	Parameter p_Wfvar;
	Parameter p_bfhid;
	Parameter p_bfmu;
	Parameter p_bfvar;
	
	Builder builder;
	Builder builder_src_fwd;
	Builder builder_src_bwd;
	
	bool rnn_src_embeddings;
	
	bool giza_positional;
	bool giza_markov;
	bool giza_fertility;
	bool doc_context;
	bool global_fertility;
	
	unsigned src_vocab_size;
	unsigned tgt_vocab_size;

	float dropout_dec;
	float dropout_enc;

	// statefull functions for incrementally creating computation graph, one
	// target word at a time
	void StartNewInstance(const std::vector<int> &src, ComputationGraph &cg, ModelStats& tstats, const std::vector<int> *ctx=0);
	void StartNewInstance(const std::vector<int> &src, ComputationGraph &cg, const std::vector<int> *ctx=0);
	void StartNewInstance_Batch(const std::vector<std::vector<int>> &srcs, ComputationGraph &cg, ModelStats& tstats);// for supporting mini-batch training
	Expression AddInput(unsigned tgt_tok, unsigned t, ComputationGraph &cg, RNNPointer *prev_state=0);
	Expression AddInput_Batch(const std::vector<unsigned>& tgt_tok, unsigned t, ComputationGraph &cg);// for supporting mini-batch training
	
	std::vector<float> *auxiliary_vector(); // memory management

	// state variables used in the above two methods
	Expression src;
	Expression i_R;
	Expression i_Q;
	Expression i_P;
	Expression i_S;
	Expression i_bias;
	Expression i_Wa;
	Expression i_Ua;
	Expression i_va;
	Expression i_uax;
	Expression i_Ta;
	Expression i_src_idx;
	Expression i_src_len;
	Expression i_tt_ctx;

	std::vector<Expression> aligns; // soft word alignments
	unsigned slen; // source sentence length
	bool has_document_context;

	std::vector<std::vector<float>*> aux_vecs; // special storage for constant vectors
	unsigned num_aux_vecs;
};

#define WTF(expression) \
	std::cout << #expression << " has dimensions " << cg.nodes[expression.i]->dim << std::endl;
#define KTHXBYE(expression) \
	std::cout << *cg.get_value(expression.i) << std::endl;

#define LOLCAT(expression) \
	WTF(expression) \
	KTHXBYE(expression) 

template <class Builder>
AttentionalModel<Builder>::AttentionalModel(dynet::Model* model,
	unsigned _src_vocab_size, unsigned _tgt_vocab_size
	, unsigned slayers, unsigned tlayers
	, unsigned hidden_dim, unsigned align_dim
	, bool _rnn_src_embeddings
	, bool _giza_positional, bool _giza_markov, bool _giza_fertility
	, bool _doc_context
	, bool _global_fertility
	, LookupParameter* _p_cs, LookupParameter* _p_ct)
		: builder(tlayers, (_rnn_src_embeddings) ? 3*hidden_dim : 2*hidden_dim, hidden_dim, *model)
		, builder_src_fwd(slayers, hidden_dim, hidden_dim, *model)
		, builder_src_bwd(slayers, hidden_dim, hidden_dim, *model)
		, rnn_src_embeddings(_rnn_src_embeddings) 
		, giza_positional(_giza_positional), giza_markov(_giza_markov), giza_fertility(_giza_fertility)
		, doc_context(_doc_context)
		, global_fertility(_global_fertility)
		, src_vocab_size(_src_vocab_size)
		, tgt_vocab_size(_tgt_vocab_size)
		, num_aux_vecs(0)
{
	//std::cerr << "Attentionalmodel(" << src_vocab_size  << " " <<  _tgt_vocab_size  << " " <<  slayers << " " << tlayers << " " <<  hidden_dim << " " <<  align_dim  << " " <<  _rnn_src_embeddings  << " " <<  _giza_extentions  << " " <<  _doc_context << ")\n";
	
	p_cs = (_p_cs==nullptr)?model->add_lookup_parameters(src_vocab_size, {hidden_dim}):*_p_cs;
	p_ct = (_p_ct==nullptr)?model->add_lookup_parameters(tgt_vocab_size, {hidden_dim}):*_p_ct; 
	p_R = model->add_parameters({tgt_vocab_size, hidden_dim});
	p_P = model->add_parameters({hidden_dim, hidden_dim});
	p_bias = model->add_parameters({tgt_vocab_size});
	p_Wa = model->add_parameters({align_dim, tlayers*hidden_dim});

	if (rnn_src_embeddings) {
		p_Ua = model->add_parameters({align_dim, 2*hidden_dim});
		p_Q = model->add_parameters({hidden_dim, 2*hidden_dim});
	} 
	else {
		p_Ua = model->add_parameters({align_dim, hidden_dim});
		p_Q = model->add_parameters({hidden_dim, hidden_dim});
	}

	if (giza_positional || giza_markov || giza_fertility) {
		int num_giza = 0;
		if (giza_positional) num_giza += 3;
		if (giza_markov) num_giza += 3;
		if (giza_fertility) num_giza += 3;

		p_Ta = model->add_parameters({align_dim, (unsigned int)num_giza});
	}

	p_va = model->add_parameters({align_dim});

	if (doc_context) {
		if (rnn_src_embeddings) {
			p_S = model->add_parameters({hidden_dim, 2*hidden_dim});
		} 
		else {
			p_S = model->add_parameters({hidden_dim, hidden_dim});
		}
	}

	int hidden_layers = builder.num_h0_components();
	for (int l = 0; l < hidden_layers; ++l) {
	if (rnn_src_embeddings)
		p_Wh0.push_back(model->add_parameters({hidden_dim, 2*hidden_dim}));
	else
		p_Wh0.push_back(model->add_parameters({hidden_dim, hidden_dim}));
	}

	dropout_dec = 0.f;
	dropout_enc = 0.f;
}

// enable/disable dropout for source and target RNNs
template <class Builder>
void AttentionalModel<Builder>::Set_Dropout(float do_enc, float do_dec)
{
	dropout_dec = do_dec;
	dropout_enc = do_enc;
}

template <class Builder>
void AttentionalModel<Builder>::Enable_Dropout()
{
	builder.set_dropout(dropout_dec);
	builder_src_fwd.set_dropout(dropout_enc);
	builder_src_bwd.set_dropout(dropout_enc);
}

template <class Builder>
void AttentionalModel<Builder>::Disable_Dropout()
{
	builder.disable_dropout();
	builder_src_fwd.disable_dropout();
	builder_src_bwd.disable_dropout();
}

template <class Builder>
void AttentionalModel<Builder>::Add_Global_Fertility_Params(dynet::Model* model, unsigned hidden_dim, bool _rnn_src_embeddings)
{
	if (global_fertility){
		if (_rnn_src_embeddings) {
			p_Wfhid = model->add_parameters({hidden_dim, 2*hidden_dim});
		} 
		else {
			p_Wfhid = model->add_parameters({hidden_dim, hidden_dim});
		}
	 
		p_bfhid = model->add_parameters({hidden_dim});
		p_Wfmu = model->add_parameters({hidden_dim});
		p_bfmu = model->add_parameters({1});
		p_Wfvar = model->add_parameters({hidden_dim});
		p_bfvar = model->add_parameters({1});
	 }
}

template <class Builder>
AttentionalModel<Builder>::~AttentionalModel()
{
	for (auto v: aux_vecs) delete v;
}

template <class Builder>
std::vector<float>* AttentionalModel<Builder>::auxiliary_vector()
{
	while (num_aux_vecs >= aux_vecs.size())
		aux_vecs.push_back(new std::vector<float>());
	// NB, we return the last auxiliary vector, AND increment counter
	return aux_vecs[num_aux_vecs++]; 
}

template <class Builder>
void AttentionalModel<Builder>::StartNewInstance(const std::vector<int> &source, ComputationGraph &cg, const std::vector<int> *ctx)
{
	slen = source.size(); 
	std::vector<Expression> source_embeddings;
	if (!rnn_src_embeddings) {
		for (unsigned s = 0; s < slen; ++s)
			source_embeddings.push_back(lookup(cg, p_cs, source[s]));
	} 
	else {
		// run a RNN backward and forward over the source sentence
		// and stack the top-level hidden states from each model as 
		// the representation at each position
		std::vector<Expression> src_fwd(slen);
		builder_src_fwd.new_graph(cg);
		builder_src_fwd.start_new_sequence();
		for (unsigned i = 0; i < slen; ++i)
			src_fwd[i] = builder_src_fwd.add_input(lookup(cg, p_cs, source[i]));

		std::vector<Expression> src_bwd(slen);
		builder_src_bwd.new_graph(cg);
		builder_src_bwd.start_new_sequence();
		for (int i = slen-1; i >= 0; --i) {
			// offset by one position to the right, to catch </s> and generally
			// not duplicate the w_t already captured in src_fwd[t]
			src_bwd[i] = builder_src_bwd.add_input(lookup(cg, p_cs, source[i]));
		}

		for (unsigned i = 0; i < slen; ++i) 
			source_embeddings.push_back(concatenate(std::vector<Expression>({src_fwd[i], src_bwd[i]})));
	}
	src = concatenate_cols(source_embeddings); 

	// now for the target sentence
	i_R = parameter(cg, p_R); // hidden -> word rep parameter
	i_Q = parameter(cg, p_Q);
	i_P = parameter(cg, p_P);
	i_bias = parameter(cg, p_bias);  // word bias
	i_Wa = parameter(cg, p_Wa); 
	i_Ua = parameter(cg, p_Ua);
	i_va = parameter(cg, p_va);
	i_uax = i_Ua * src; 

	// reset aux_vecs counter, allowing the memory to be reused
	num_aux_vecs = 0;

	if (giza_fertility || giza_markov || giza_positional) {
	i_Ta = parameter(cg, p_Ta);   
		if (giza_positional) {
			i_src_idx = arange(cg, 0, slen, true, auxiliary_vector());
			i_src_len = repeat(cg, slen, std::log(1.0 + slen), auxiliary_vector());
		}
	}

	aligns.clear();
	aligns.push_back(repeat(cg, slen, 0.0f, auxiliary_vector()));

	// initialilse h from global information of the source sentence
#ifndef RNN_H0_IS_ZERO
	std::vector<Expression> h0;
	Expression i_src = average(source_embeddings); // try max instead?

	int hidden_layers = builder.num_h0_components();
	for (int l = 0; l < hidden_layers; ++l) {
		Expression i_Wh0 = parameter(cg, p_Wh0[l]);
		h0.push_back(tanh(i_Wh0 * i_src));
	}

	builder.new_graph(cg); 
	builder.start_new_sequence(h0);
#else
	builder.new_graph(cg); 
	builder.start_new_sequence();
#endif

	// document context; n.b. use "0" context for the first sentence
	if (doc_context && ctx != 0) { 
		const std::vector<int> &context = *ctx;

		std::vector<Expression> ctx_embed;
		if (!rnn_src_embeddings) {
			for (unsigned s = 1; s+1 < context.size(); ++s) 
				ctx_embed.push_back(lookup(cg, p_cs, context[s]));
		} 
		else {
			ctx_embed.resize(context.size()-1);
			builder_src_fwd.start_new_sequence();
			for (unsigned i = 0; i+1 < context.size(); ++i) 
				ctx_embed[i] = builder_src_fwd.add_input(lookup(cg, p_cs, context[i]));
		}

		Expression avg_context = average(source_embeddings); 
		i_S = parameter(cg, p_S);
		i_tt_ctx = i_S * avg_context;
		has_document_context = true;
	}
	else {
		has_document_context = false;
	}
}

template <class Builder>
void AttentionalModel<Builder>::StartNewInstance(const std::vector<int> &source, ComputationGraph &cg, ModelStats& tstats, const std::vector<int> *ctx)
{
	tstats.words_src += source.size() - 1;

	slen = source.size(); 
	std::vector<Expression> source_embeddings;
	if (!rnn_src_embeddings) {
		for (unsigned s = 0; s < slen; ++s){
			if (source[s] == kSRC_UNK) tstats.words_src_unk++;
			source_embeddings.push_back(lookup(cg, p_cs, source[s]));
		}
	} 
	else {
		// run a RNN backward and forward over the source sentence
		// and stack the top-level hidden states from each model as 
		// the representation at each position
		std::vector<Expression> src_fwd(slen);
		builder_src_fwd.new_graph(cg);
		builder_src_fwd.start_new_sequence();
		for (unsigned i = 0; i < slen; ++i){ 
			if (source[i] == kSRC_UNK) tstats.words_src_unk++;		
			src_fwd[i] = builder_src_fwd.add_input(lookup(cg, p_cs, source[i]));
		}

		std::vector<Expression> src_bwd(slen);
		builder_src_bwd.new_graph(cg);
		builder_src_bwd.start_new_sequence();
		for (int i = slen-1; i >= 0; --i) {
			// offset by one position to the right, to catch </s> and generally
			// not duplicate the w_t already captured in src_fwd[t]
			src_bwd[i] = builder_src_bwd.add_input(lookup(cg, p_cs, source[i]));
		}

		for (unsigned i = 0; i < slen; ++i) 
			source_embeddings.push_back(concatenate(std::vector<Expression>({src_fwd[i], src_bwd[i]})));
	}
	src = concatenate_cols(source_embeddings); 

	// now for the target sentence
	i_R = parameter(cg, p_R); // hidden -> word rep parameter
	i_Q = parameter(cg, p_Q);
	i_P = parameter(cg, p_P);
	i_bias = parameter(cg, p_bias);  // word bias
	i_Wa = parameter(cg, p_Wa); 
	i_Ua = parameter(cg, p_Ua);
	i_va = parameter(cg, p_va);
	i_uax = i_Ua * src; 

	// reset aux_vecs counter, allowing the memory to be reused
	num_aux_vecs = 0;

	if (giza_fertility || giza_markov || giza_positional) {
		i_Ta = parameter(cg, p_Ta);   
		if (giza_positional) {
			i_src_idx = arange(cg, 0, slen, true, auxiliary_vector());
			i_src_len = repeat(cg, slen, std::log(1.0 + slen), auxiliary_vector());
		}
	}

	aligns.clear();
	aligns.push_back(repeat(cg, slen, 0.0f, auxiliary_vector()));

	// initialilse h from global information of the source sentence
#ifndef RNN_H0_IS_ZERO
	std::vector<Expression> h0;
	Expression i_src = average(source_embeddings); // try max instead?

	int hidden_layers = builder.num_h0_components();
	for (int l = 0; l < hidden_layers; ++l) {
		Expression i_Wh0 = parameter(cg, p_Wh0[l]);
		h0.push_back(tanh(i_Wh0 * i_src));
	}

	builder.new_graph(cg); 
	builder.start_new_sequence(h0);
#else
	builder.new_graph(cg); 
	builder.start_new_sequence();
#endif

	// document context; n.b. use "0" context for the first sentence
	if (doc_context && ctx != 0) { 
		const std::vector<int> &context = *ctx;

		std::vector<Expression> ctx_embed;
		if (!rnn_src_embeddings) {
			for (unsigned s = 1; s+1 < context.size(); ++s) 
				ctx_embed.push_back(lookup(cg, p_cs, context[s]));
		}
		else {
			ctx_embed.resize(context.size()-1);
			builder_src_fwd.start_new_sequence();
			for (unsigned i = 0; i+1 < context.size(); ++i) 
				ctx_embed[i] = builder_src_fwd.add_input(lookup(cg, p_cs, context[i]));
		}
	
		Expression avg_context = average(source_embeddings); 
		i_S = parameter(cg, p_S);
		i_tt_ctx = i_S * avg_context;
		has_document_context = true;
	} 
	else {
		has_document_context = false;
	}
}

template <class Builder>
void AttentionalModel<Builder>::StartNewInstance_Batch(const std::vector<std::vector<int>> &sources
		, ComputationGraph &cg
		, ModelStats& tstats)
{
	// Get the max size
	size_t max_len = sources[0].size();
	for(size_t i = 1; i < sources.size(); i++) max_len = std::max(max_len, sources[i].size());

	slen = max_len;
	
	std::vector<unsigned> words(sources.size());

	std::vector<Expression> source_embeddings;   
	//cerr << "(1a) embeddings" << endl;
	if (!rnn_src_embeddings) {
		for (unsigned l = 0; l < max_len; l++){
			for (unsigned bs = 0; bs < sources.size(); ++bs){
				words[bs] = (l < sources[bs].size()) ? (unsigned)sources[bs][l] : kSRC_EOS;
				if (l < sources[bs].size()){ 
					tstats.words_src++; 
					if (sources[bs][l] == kSRC_UNK) tstats.words_src_unk++;
				}
			}
			source_embeddings.push_back(lookup(cg, p_cs, words));
		}
	}
	else {
		// run a RNN backward and forward over the source sentence
		// and stack the top-level hidden states from each model as 
		// the representation at each position
		std::vector<Expression> src_fwd(max_len);
		builder_src_fwd.new_graph(cg);
		builder_src_fwd.start_new_sequence();
		for (unsigned l = 0; l < max_len; l++){
			for (unsigned bs = 0; bs < sources.size(); ++bs){
				words[bs] = (l < sources[bs].size()) ? (unsigned)sources[bs][l] : kSRC_EOS;
				if (l < sources[bs].size()){ 
					tstats.words_src++; 
					if (sources[bs][l] == kSRC_UNK) tstats.words_src_unk++;
				}
			}
			src_fwd[l] = builder_src_fwd.add_input(lookup(cg, p_cs, words));
		}

		std::vector<Expression> src_bwd(max_len);
		builder_src_bwd.new_graph(cg);
		builder_src_bwd.start_new_sequence();
		for (int l = max_len - 1; l >= 0; --l) { // int instead of unsigned for negative value of l
			// offset by one position to the right, to catch </s> and generally
			// not duplicate the w_t already captured in src_fwd[t]
			for (unsigned bs = 0; bs < sources.size(); ++bs) 
				words[bs] = ((unsigned)l < sources[bs].size()) ? (unsigned)sources[bs][l] : kSRC_EOS;
			src_bwd[l] = builder_src_bwd.add_input(lookup(cg, p_cs, words));
		}

		for (unsigned l = 0; l < max_len; ++l) 
			source_embeddings.push_back(concatenate(std::vector<Expression>({src_fwd[l], src_bwd[l]})));
	}
	src = concatenate_cols(source_embeddings); 

	// now for the target sentence
	i_R = parameter(cg, p_R); // hidden -> word rep parameter
	i_Q = parameter(cg, p_Q);
	i_P = parameter(cg, p_P);
	i_bias = parameter(cg, p_bias);  // word bias
	i_Wa = parameter(cg, p_Wa); 
	i_Ua = parameter(cg, p_Ua);
	i_va = parameter(cg, p_va);
	i_uax = i_Ua * src; 

	// reset aux_vecs counter, allowing the memory to be reused
	num_aux_vecs = 0;

	//cerr << "(1c) structural biases" << endl;
	if (giza_fertility || giza_markov || giza_positional) {
		i_Ta = parameter(cg, p_Ta);   
		if (giza_positional) {
			i_src_idx = arange(cg, 0, max_len, true, auxiliary_vector());
			i_src_len = repeat(cg, max_len, std::log(1.0 + max_len), auxiliary_vector());
		}
	}

	//cerr << "(1d) init alignments" << endl;
	aligns.clear();
	aligns.push_back(repeat(cg, max_len, 0.0f, auxiliary_vector()));

	// initialilse h from global information of the source sentence
	//cerr << "(1e) init builder" << endl;
#ifndef RNN_H0_IS_ZERO
	std::vector<Expression> h0;
	Expression i_src = average(source_embeddings); // try max instead?
	int hidden_layers = builder.num_h0_components();

	for (int l = 0; l < hidden_layers; ++l) {
		Expression i_Wh0 = parameter(cg, p_Wh0[l]);
		h0.push_back(tanh(i_Wh0 * i_src));
	}

	builder.new_graph(cg); 
	builder.start_new_sequence(h0);
#else
	builder.new_graph(cg); 
	builder.start_new_sequence();
#endif
}

template <class Builder>
Expression AttentionalModel<Builder>::AddInput(unsigned trg_tok, unsigned t, ComputationGraph &cg, RNNPointer *prev_state)
{
	// alignment input 
	Expression i_wah_rep;
	if (t > 0) {
		Expression i_h_tm1;
		if (prev_state)
			i_h_tm1 = concatenate(builder.get_h(*prev_state));// This is required for beam search decoding implementation.
		else
			i_h_tm1 = concatenate(builder.final_h());

		Expression i_wah = i_Wa * i_h_tm1;
	
		// want numpy style broadcasting, but have to do this manually
		i_wah_rep = concatenate_cols(std::vector<Expression>(slen, i_wah));
	}

	Expression i_e_t;
	if (giza_markov || giza_fertility || giza_positional) {
		std::vector<Expression> alignment_context;
		if (giza_markov || giza_fertility) {
			if (t > 0) {
				if (giza_fertility) {
					auto i_aprev = concatenate_cols(aligns);
					auto i_asum = sum_cols(i_aprev);
					auto i_asum_pm = dither(cg, i_asum, 0.0f, auxiliary_vector());
					alignment_context.push_back(i_asum_pm);
				}
				if (giza_markov) {
					auto i_alast_pm = dither(cg, aligns.back(), 0.0f, auxiliary_vector());
					alignment_context.push_back(i_alast_pm);
				}
			}
			else {
				// just 6 repeats of the 0 vector
				auto zeros = repeat(cg, slen, 0, auxiliary_vector());
				if (giza_fertility) {
					alignment_context.push_back(zeros); 
					alignment_context.push_back(zeros);
					alignment_context.push_back(zeros);
				}
				if (giza_markov) {
					alignment_context.push_back(zeros);
					alignment_context.push_back(zeros);
					alignment_context.push_back(zeros);
				}
			}
		}
	
		if (giza_positional) {
			alignment_context.push_back(i_src_idx);
			alignment_context.push_back(i_src_len);
			auto i_tgt_idx = repeat(cg, slen, std::log(1.0 + t), auxiliary_vector());
			alignment_context.push_back(i_tgt_idx);
		}

		auto i_context = concatenate_cols(alignment_context);

		auto i_e_t_input = i_uax + i_Ta * transpose(i_context); 

		if (t > 0) i_e_t_input = i_e_t_input + i_wah_rep;
		i_e_t = transpose(tanh(i_e_t_input)) * i_va;
	} 
	else 
	{
		if (t > 0) 
			i_e_t = transpose(tanh(i_wah_rep + i_uax)) * i_va;
		else
			i_e_t = transpose(tanh(i_uax)) * i_va;
	}

	Expression i_alpha_t = softmax(i_e_t); // FIXME: consider summing to less than one?
	aligns.push_back(i_alpha_t);
	Expression i_c_t = src * i_alpha_t; // FIXME: effectively summing here, consider maxing?

	// word input
	Expression i_x_t = lookup(cg, p_ct, trg_tok);
	Expression input = concatenate(std::vector<Expression>({i_x_t, i_c_t})); 

	// y_t = RNN([x_t, a_t])
	Expression i_y_t;
	if (prev_state)
	   i_y_t = builder.add_input(*prev_state, input);
	else
	   i_y_t = builder.add_input(input);

	// document context if available
	if (doc_context && has_document_context)
		i_y_t = i_y_t + i_tt_ctx;
	
#ifndef VANILLA_TARGET_LSTM
	// Bahdanau does a max-out thing here; I do a tanh. Tomaatos tomateos.
	Expression i_tildet_t = tanh(affine_transform({i_y_t, i_Q, i_c_t, i_P, i_x_t}));
	Expression i_r_t = affine_transform({i_bias, i_R, i_tildet_t}); 
#else
	Expression i_r_t = affine_transform({i_bias, i_R, i_y_t}); 
#endif

	return i_r_t;
}

template <class Builder>
Expression AttentionalModel<Builder>::AddInput_Batch(const std::vector<unsigned>& trg_words, unsigned t, ComputationGraph &cg)
{
	// alignment input 
	Expression i_wah_rep;
	if (t > 0) {
		Expression i_h_tm1 = concatenate(builder.final_h());
		Expression i_wah = i_Wa * i_h_tm1;
	
		// want numpy style broadcasting, but have to do this manually
		i_wah_rep = concatenate_cols(std::vector<Expression>(slen, i_wah));
	}

	Expression i_e_t;
	if (giza_markov || giza_fertility || giza_positional) {
		std::vector<Expression> alignment_context;
		if (giza_markov || giza_fertility) {
			if (t > 0) {
				if (giza_fertility) {
					auto i_aprev = concatenate_cols(aligns);
					auto i_asum = sum_cols(i_aprev);
					auto i_asum_pm = dither(cg, i_asum, 0.0f, auxiliary_vector());
					alignment_context.push_back(i_asum_pm);
				}
				if (giza_markov) {
					auto i_alast_pm = dither(cg, aligns.back(), 0.0f, auxiliary_vector());
					alignment_context.push_back(i_alast_pm);
				}
			} else {
				// just 6 repeats of the 0 vector
				auto zeros = repeat(cg, slen, 0, auxiliary_vector());
				if (giza_fertility) {
					alignment_context.push_back(zeros); 
					alignment_context.push_back(zeros);
					alignment_context.push_back(zeros);
				}
				if (giza_markov) {
					alignment_context.push_back(zeros);
					alignment_context.push_back(zeros);
					alignment_context.push_back(zeros);
				}
			}
		}

		if (giza_positional) {
			alignment_context.push_back(i_src_idx);
			alignment_context.push_back(i_src_len);
			auto i_tgt_idx = repeat(cg, slen, std::log(1.0 + t), auxiliary_vector());
			alignment_context.push_back(i_tgt_idx);
		}
	
		auto i_context = concatenate_cols(alignment_context);

		auto i_e_t_input = i_uax + i_Ta * transpose(i_context); 

		if (t > 0) i_e_t_input = i_e_t_input + i_wah_rep;

		i_e_t = transpose(tanh(i_e_t_input)) * i_va;
	} 
	else {
		if (t > 0) 
			i_e_t = transpose(tanh(i_wah_rep + i_uax)) * i_va;
		else
			i_e_t = transpose(tanh(i_uax)) * i_va;
	}

	Expression i_alpha_t = softmax(i_e_t); // FIXME: consider summing to less than one?
	aligns.push_back(i_alpha_t);
	Expression i_c_t = src * i_alpha_t; // FIXME: effectively summing here, consider maxing?
	
	// target word inputs
	Expression i_x_t = lookup(cg, p_ct, trg_words);
	Expression input = concatenate(std::vector<Expression>({i_x_t, i_c_t})); 
	
	// y_t = RNN([x_t, a_t])
	Expression i_y_t = builder.add_input(input);

#ifndef VANILLA_TARGET_LSTM
	// Bahdanau does a max-out thing here; I do a tanh. Tomaatos tomateos.
	Expression i_tildet_t = tanh(affine_transform({i_y_t, i_Q, i_c_t, i_P, i_x_t}));
	Expression i_r_t = affine_transform({i_bias, i_R, i_tildet_t}); 
#else
	Expression i_r_t = affine_transform({i_bias, i_R, i_y_t}); 
#endif

	return i_r_t;
}

template <class Builder>
Expression AttentionalModel<Builder>::BuildGraph(const std::vector<int> &source,
	const std::vector<int>& target, ComputationGraph& cg, ModelStats& tstats
	, Expression *alignment, const std::vector<int>* ctx, Expression *coverage, Expression *fertility) 
{
	//std::cout << "source sentence length: " << source.size() << " target: " << target.size() << std::endl;
	StartNewInstance(source, cg, tstats, ctx);

	std::vector<Expression> errs;
	const unsigned tlen = target.size() - 1; 
	for (unsigned t = 0; t < tlen; ++t) {
		tstats.words_tgt++;
		if (target[t] == kTGT_UNK) tstats.words_tgt_unk++;

		Expression i_r_t = AddInput(target[t], t, cg);
		Expression i_err = pickneglogsoftmax(i_r_t, target[t+1]);
		errs.push_back(i_err);
	}

	// save the alignment for later
	if (alignment != 0) {
		// pop off the last alignment column
		*alignment = concatenate_cols(aligns);
	}

	if (coverage != nullptr || fertility != nullptr) {
		Expression i_aligns = (alignment != 0) ? *alignment : concatenate_cols(aligns);
		Expression i_totals = sum_cols(i_aligns);
		Expression i_total_trim = pickrange(i_totals, 1, slen-1);// only care about the non-null entries

		// AM for computer vision paper (Xu K. et al, 2015) has a penalty over alignment rows deviating from 1
		if (coverage != nullptr) {
			Expression i_ones = repeat(cg, slen-2, 1.0f, auxiliary_vector());
			Expression i_penalty = squared_distance(i_total_trim, i_ones);
			*coverage = i_penalty;
		} 

		// Contextual fertility model (Cohn et al., 2016)
		if (fertility != nullptr) {
			assert(global_fertility);

			Expression fbias = concatenate_cols(std::vector<Expression>(slen, parameter(cg, p_bfhid)));
			Expression mbias = concatenate(std::vector<Expression>(slen, parameter(cg, p_bfmu)));
			Expression vbias = concatenate(std::vector<Expression>(slen, parameter(cg, p_bfvar)));
			Expression fhid = tanh(transpose(fbias + parameter(cg, p_Wfhid) * src));  
			Expression mu = mbias + fhid * parameter(cg, p_Wfmu);
			Expression var = exp(vbias + fhid * parameter(cg, p_Wfvar));

			Expression mu_trim = pickrange(mu, 1, slen-1);
			Expression var_trim = pickrange(var, 1, slen-1);

#if 0
			/* log-Normal distribution */
			Expression log_fert = log(i_total_trim);
			Expression delta = log_fert - mu_trim;
			//Expression exponent = cdiv(-cwise_multiply(delta, delta), 2.0f * var_trim);
			Expression exponent = cdiv(-cmult(delta, delta), 2.0f * var_trim);// cmult is a new version of cwise_multiply
			Expression partition = -log_fert - 0.5 * log(2.0f * var_trim * 3.14159265359);
			*fertility = -sum_cols(transpose(partition + exponent));
#else
			/* Normal distribution */
			Expression delta = i_total_trim - mu_trim;
			//Expression exponent = cdiv(-cwise_multiply(delta, delta), 2.0f * var_trim);
			Expression exponent = cdiv(-cmult(delta, delta), 2.0f * var_trim);// cmult is a new version of cwise_multiply
			Expression partition = -0.5 * log(2.0f * var_trim * 3.14159265359);
			*fertility = -sum_cols(transpose(partition + exponent));
			// note that as this is the value of the normal density, the errors
			// are not strictly positive
#endif

			//LOLCAT(transpose(i_total_trim));
			//LOLCAT(transpose(mu_trim));
			//LOLCAT(transpose(var_trim));
			//LOLCAT(transpose(partition + exponent));
			//LOLCAT(exp(transpose(partition + exponent)));
		}
	}

	Expression i_nerr = sum(errs);
	return i_nerr;
}

template <class Builder>
Expression AttentionalModel<Builder>::BuildGraph_Batch(const std::vector<std::vector<int>> &sources, const std::vector<std::vector<int>>& targets
	, ComputationGraph& cg, ModelStats& tstats
	, Expression *coverage, Expression *fertility) 
{
	StartNewInstance_Batch(sources, cg, tstats);

	std::vector<Expression> errs;

	const unsigned tlen = targets[0].size() - 1; 
	std::vector<unsigned> next_words(targets.size()), words(targets.size());

	for (unsigned t = 0; t < tlen; ++t) {
		for(size_t bs = 0; bs < targets.size(); bs++){
			words[bs] = (targets[bs].size() > t) ? (unsigned)targets[bs][t] : kTGT_EOS;
			next_words[bs] = (targets[bs].size() > (t+1)) ? (unsigned)targets[bs][t+1] : kTGT_EOS;
			if (targets[bs].size() > t) {
				tstats.words_tgt++;
				if (targets[bs][t] == kTGT_UNK) tstats.words_tgt_unk++;
			}
		}
			
		Expression i_r_t = AddInput_Batch(words, t, cg);
		Expression i_err = pickneglogsoftmax(i_r_t, next_words);
	
		errs.push_back(i_err);
	}

	// FIXME: pop off the last alignment column?
	Expression i_alignment = concatenate_cols(aligns);

	if (coverage != nullptr || fertility != nullptr) {
		Expression i_totals = sum_cols(i_alignment);		
		Expression i_total_trim = pickrange(i_totals, 1, slen-1);// only care about the non-null entries

		// AM for computer vision paper (Xu K. et al, 2015) has a penalty over alignment rows deviating from 1
		if (coverage != nullptr) {
			Expression i_ones = repeat(cg, slen-2, 1.0f, auxiliary_vector());
			Expression i_penalty = squared_distance(i_total_trim, i_ones);
			*coverage = sum_batches(i_penalty);
		} 

		// Contextual fertility model (Cohn et al., 2016)
		if (fertility != nullptr) {
			assert(global_fertility);

			Expression fbias = concatenate_cols(std::vector<Expression>(slen, parameter(cg, p_bfhid)));
			Expression mbias = concatenate(std::vector<Expression>(slen, parameter(cg, p_bfmu)));
			Expression vbias = concatenate(std::vector<Expression>(slen, parameter(cg, p_bfvar)));
			Expression fhid = tanh(transpose(fbias + parameter(cg, p_Wfhid) * src));  
			Expression mu = mbias + fhid * parameter(cg, p_Wfmu);
			Expression var = exp(vbias + fhid * parameter(cg, p_Wfvar));

			Expression mu_trim = pickrange(mu, 1, slen-1);
			Expression var_trim = pickrange(var, 1, slen-1);

#if 0
			/* log-Normal distribution */
			Expression log_fert = log(i_total_trim);
			Expression delta = log_fert - mu_trim;
			//Expression exponent = cdiv(-cwise_multiply(delta, delta), 2.0f * var_trim);
			Expression exponent = cdiv(-cmult(delta, delta), 2.0f * var_trim);// cmult is a new version of cwise_multiply
			Expression partition = -log_fert - 0.5 * log(2.0f * var_trim * 3.14159265359);
			*fertility = -sum_cols(transpose(partition + exponent));
#else
			/* Normal distribution */
			Expression delta = i_total_trim - mu_trim;
			//Expression exponent = cdiv(-cwise_multiply(delta, delta), 2.0f * var_trim);
			Expression exponent = cdiv(-cmult(delta, delta), 2.0f * var_trim);// cmult is a new version of cwise_multiply
			Expression partition = -0.5 * log(2.0f * var_trim * 3.14159265359);
			*fertility = sum_batches(-sum_cols(transpose(partition + exponent)));
			// note that as this is the value of the normal density, the errors
			// are not strictly positive
#endif

			//LOLCAT(transpose(i_total_trim));
			//LOLCAT(transpose(mu_trim));
			//LOLCAT(transpose(var_trim));
			//LOLCAT(transpose(partition + exponent));
			//LOLCAT(exp(transpose(partition + exponent)));
		}
	}


	Expression i_nerr = sum_batches(sum(errs));
	return i_nerr;
}

template <class Builder>
void AttentionalModel<Builder>::BuildGraph(const std::vector<int> &source,
	const std::vector<int>& target, ComputationGraph& cg
	, std::vector<std::vector<float>>& v_preds, bool with_softmax) 
{
	//std::cout << "source sentence length: " << source.size() << " target: " << target.size() << std::endl;
	StartNewInstance(source, cg, 0);

	v_preds.clear();
	const unsigned tlen = target.size() - 1; 
	for (unsigned t = 0; t < tlen; ++t) {
		Expression i_r_t = AddInput(target[t], t, cg);
		if (with_softmax){// w/ softmax prediction
			Expression i_softmax = softmax(i_r_t);
			if (t != tlen - 1)// excluding EOS prediction
				v_preds.push_back(as_vector(cg.get_value(i_softmax.i)));
		}
		else{// w/o softmax prediction
			if (t != tlen - 1)// excluding EOS prediction
				v_preds.push_back(as_vector(cg.get_value(i_r_t.i)));
		}
	}
}

template <class Builder>
Expression AttentionalModel<Builder>::Forward(const std::vector<int> & sent, int t
	, bool log_prob
	, RNNPointer& prev_state, RNNPointer& state
	, dynet::ComputationGraph & cg
	, std::vector<Expression> & align_out)
{
	Expression i_r_t;
	if (state == RNNPointer(-1)){
		i_r_t = AddInput(sent[t], t, cg);
	}
	else{
		i_r_t = AddInput(sent[t], t, cg, &prev_state);
	}

	Expression i_softmax = (log_prob)?log_softmax(i_r_t):softmax(i_r_t);

	align_out.push_back(aligns.back()); 

	state = builder.state();

	return i_softmax;
}

//---------------------------------------------------------------------------------------------
// Build the relaxation optimization graph for the given sentence including returned loss
template <class Builder>
void AttentionalModel<Builder>::ComputeTrgWordEmbeddingMatrix(dynet::ComputationGraph& cg)
{
	std::vector<Expression> vEs(tgt_vocab_size);
	for (unsigned i = 0; i < tgt_vocab_size; i++)
		vEs[i] = lookup(cg, p_ct, i);//hidden_dim x 1
	i_We = concatenate_cols(vEs);/*hidden_dim x TGT_VOCAB_SIZE*/
}

template <class Builder>
void AttentionalModel<Builder>::ComputeSrcWordEmbeddingMatrix(dynet::ComputationGraph& cg)
{
	std::vector<Expression> vEs(src_vocab_size);
	for (unsigned i = 0; i < src_vocab_size; i++)
		vEs[i] = lookup(cg, p_cs, i);//hidden_dim x 1
	i_We = concatenate_cols(vEs);/*hidden_dim x SRC_VOCAB_SIZE*/
}

template <class Builder>
Expression AttentionalModel<Builder>::GetWordEmbeddingVector(const Expression& i_y)
{
	// expected embedding
	return (i_We/*hidden_dim x VOCAB_SIZE*/ * i_y/*VOCAB_SIZE x 1*/);//hidden_dim x 1
}

template <class Builder>
Expression AttentionalModel<Builder>::AddInput(const Expression& i_ewe_t, int t, ComputationGraph &cg, RNNPointer *prev_state)
{
	Expression i_wah_rep;
	if (t > 0) {
		auto i_h_tm1 = concatenate(builder.final_h());
		Expression i_wah = i_Wa * i_h_tm1;
		i_wah_rep = concatenate_cols(std::vector<Expression>(slen, i_wah));// want numpy style broadcasting, but have to do this manually
	}

	Expression i_e_t;
	if (giza_markov || giza_fertility || giza_positional) {
		std::vector<Expression> alignment_context;
		if (giza_markov || giza_fertility) {
			if (t > 0) {
				if (giza_fertility) {
					auto i_aprev = concatenate_cols(aligns);
					auto i_asum = sum_cols(i_aprev);
					auto i_asum_pm = dither(cg, i_asum, 0.0f, auxiliary_vector());
					alignment_context.push_back(i_asum_pm);
				}
				if (giza_markov) {
					auto i_alast_pm = dither(cg, aligns.back(), 0.0f, auxiliary_vector());
					alignment_context.push_back(i_alast_pm);
				}
			} else {
				// just 6 repeats of the 0 vector
				auto zeros = repeat(cg, slen, 0, auxiliary_vector());
				if (giza_fertility) {
					alignment_context.push_back(zeros); 
					alignment_context.push_back(zeros);
					alignment_context.push_back(zeros);
				}
				if (giza_markov) {
					alignment_context.push_back(zeros);
					alignment_context.push_back(zeros);
					alignment_context.push_back(zeros);
				}
			}
		}
		if (giza_positional) {
			alignment_context.push_back(i_src_idx);
			alignment_context.push_back(i_src_len);
			auto i_tgt_idx = repeat(cg, slen, std::log(1.0 + t), auxiliary_vector());
			alignment_context.push_back(i_tgt_idx);
		}
	
		auto i_context = concatenate_cols(alignment_context);

		auto i_e_t_input = i_uax + i_Ta * transpose(i_context); 

		if (t > 0) i_e_t_input = i_e_t_input + i_wah_rep;

		i_e_t = transpose(tanh(i_e_t_input)) * i_va;
	} 
	else {
		if (t > 0) 
			i_e_t = transpose(tanh(i_wah_rep + i_uax)) * i_va;
		else
			i_e_t = transpose(tanh(i_uax)) * i_va;
	}

	Expression i_alpha_t = softmax(i_e_t); // FIXME: consider summing to less than one?
	aligns.push_back(i_alpha_t);
	Expression i_c_t = src * i_alpha_t; // FIXME: effectively summing here, consider maxing?
	
	// word input
	Expression i_x_t = i_ewe_t;//lookup(cg, p_ct, trg_tok);
	Expression input = concatenate(std::vector<Expression>({i_x_t, i_c_t})); 

	// y_t = RNN([x_t, a_t])
	Expression i_y_t;
	if (prev_state)
	   i_y_t = builder.add_input(*prev_state, input);
	else
	   i_y_t = builder.add_input(input);

	// document context if available
	if (doc_context && has_document_context)
		i_y_t = i_y_t + i_tt_ctx;

#ifndef VANILLA_TARGET_LSTM
	// Bahdanau does a max-out thing here; I do a tanh. Tomaatos tomateos.
	Expression i_tildet_t = tanh(affine_transform({i_y_t, i_Q, i_c_t, i_P, i_x_t}));
	Expression i_r_t = affine_transform({i_bias, i_R, i_tildet_t}); 
#else
	Expression i_r_t = affine_transform({i_bias, i_R, i_y_t}); 
#endif

	return i_r_t;
}
   
template <class Builder>
Expression AttentionalModel<Builder>::BuildRelOptGraph(
	size_t algo
	, std::vector<dynet::Parameter>& v_params
	, dynet::ComputationGraph & cg
	, Dict &d
	, bool reverse
	, Expression *entropy
	, Expression *alignment
	, Expression *coverage, float coverage_C
	, Expression *fertility)
{
	int tlen = v_params.size();// desired target length (excluding BOS and EOS tokens)
	int ind_bos = d.convert("<s>"), ind_eos = d.convert("</s>");

	//std::cerr << "L*=" << tlen << std::endl;

	// collect expected word embeddings
	std::vector<Expression> i_wes(tlen+1);
	i_wes[0] = lookup(cg, p_ct, ind_bos);// known BOS embedding
	for(auto t : boost::irange(0, tlen)){
		auto ct = t;
		if (reverse == true) ct = tlen - t - 1;
		
		if (algo == RELOPT_ALGO::SOFTMAX){// SOFTMAX approach
			Expression i_p = parameter(cg, v_params[ct]);
			i_wes[t+1] = GetWordEmbeddingVector(softmax(i_p));
		}
		else if (algo == RELOPT_ALGO::SPARSEMAX){// SPARSEMAX approach
			Expression i_p = parameter(cg, v_params[ct]);
			i_wes[t+1] = GetWordEmbeddingVector(sparsemax(i_p));
		}
		else if (algo == RELOPT_ALGO::EG || algo == RELOPT_ALGO::AEG){// EG or AEG approach
			Expression i_p = parameter(cg, v_params[ct]);
			i_wes[t+1] = GetWordEmbeddingVector(i_p);
		}
		else
			assert("Unknown relopt algo! Failed!");		
	}

	// simulated generation step
	std::vector<Expression> v_costs, v_ents;
	for(auto t : boost::irange(0, tlen+1)) {
		//std::cerr << "t:" << t << std::endl;
		Expression i_r_t = AddInput(i_wes[t]/*expected word embedding*/, t, cg);

		// Run the softmax and calculate the cost
		Expression i_cost;
		if (t >= tlen){// for predicting EOS
			i_cost = pickneglogsoftmax(i_r_t, ind_eos);
		}
		else{// for predicting others
			Expression i_softmax = softmax(i_r_t);

			auto ct = t;
			if (reverse == true) ct = tlen - t - 1;
		
			Expression i_y;
			if (algo == RELOPT_ALGO::SOFTMAX){// SOFTMAX approach
				Expression i_p = parameter(cg, v_params[ct]);
				i_y = softmax(i_p);
			}
			else if (algo == RELOPT_ALGO::SPARSEMAX){// SPARSEMAX approach
				Expression i_p = parameter(cg, v_params[ct]);
				i_y = sparsemax(i_p);
			}
			else if (algo == RELOPT_ALGO::EG || algo == RELOPT_ALGO::AEG){// EG or AEG approach
				i_y = parameter(cg, v_params[ct]);
			}
			else
				assert("Unknown inference algo!");
			
			//i_cost = -log(transpose(i_y) * i_softmax);
			i_cost = -transpose(i_y) * log(i_softmax);//FIXME: use log_softmax(i_r_t) instead

			// FIXME: add entropy regularizer to SOFTMAX or SPARSEMAX cost function
			if (algo == RELOPT_ALGO::SOFTMAX || algo == RELOPT_ALGO::SPARSEMAX){
				Expression i_entropy = -transpose(i_y) * log(i_y);
				v_ents.push_back(i_entropy);
			}
		}
		
		v_costs.push_back(i_cost);
	}

	if (entropy != 0 && v_ents.size() > 0){
		*entropy = sum(v_ents);
	}

	// save the alignment for later
	if (alignment != 0) {
		// pop off the last alignment column
		*alignment = concatenate_cols(aligns);
	}

	if (coverage != nullptr || fertility != nullptr) {
		Expression i_aligns = (alignment != 0) ? *alignment : concatenate_cols(aligns);
		Expression i_totals = sum_cols(i_aligns);
		Expression i_total_trim = pickrange(i_totals, 1, slen-1);// only care about the non-null entries

		// AM for computer vision paper (Xu K. et al, 2015) has a penalty over alignment rows deviating from 1
		if (coverage != nullptr) {
			//Expression i_ones = repeat(cg, slen-2, 1.0f, auxiliary_vector());
			Expression i_ones = repeat(cg, slen-2, coverage_C, auxiliary_vector());
			Expression i_penalty = squared_distance(i_total_trim, i_ones);
			*coverage = i_penalty;
		} 

		// contextual fertility model (Cohn et al., 2016)
		if (fertility != nullptr) {
			assert(global_fertility);

			Expression fbias = concatenate_cols(std::vector<Expression>(slen, parameter(cg, p_bfhid)));
			Expression mbias = concatenate(std::vector<Expression>(slen, parameter(cg, p_bfmu)));
			Expression vbias = concatenate(std::vector<Expression>(slen, parameter(cg, p_bfvar)));
			Expression fhid = tanh(transpose(fbias + parameter(cg, p_Wfhid) * src));  
			Expression mu = mbias + fhid * parameter(cg, p_Wfmu);
			Expression var = exp(vbias + fhid * parameter(cg, p_Wfvar));

			Expression mu_trim = pickrange(mu, 1, slen-1);
			Expression var_trim = pickrange(var, 1, slen-1);

#if 0
			/* log-Normal distribution */
			Expression log_fert = log(i_total_trim);
			Expression delta = log_fert - mu_trim;
			//Expression exponent = cdiv(-cwise_multiply(delta, delta), 2.0f * var_trim);
			Expression exponent = cdiv(-cmult(delta, delta), 2.0f * var_trim);// cmult is a new version of cwise_multiply
			Expression partition = -log_fert - 0.5 * log(2.0f * var_trim * 3.14159265359);
			*fertility = -sum_cols(transpose(partition + exponent));
#else
			/* Normal distribution */
			Expression delta = i_total_trim - mu_trim;
			//Expression exponent = cdiv(-cwise_multiply(delta, delta), 2.0f * var_trim);
			Expression exponent = cdiv(-cmult(delta, delta), 2.0f * var_trim);// cmult is a new version of cwise_multiply
			Expression partition = -0.5 * log(2.0f * var_trim * 3.14159265359);
			*fertility = -sum_cols(transpose(partition + exponent));
			// note that as this is the value of the normal density, the errors
			// are not strictly positive
#endif

			//LOLCAT(transpose(i_total_trim));
			//LOLCAT(transpose(mu_trim));
			//LOLCAT(transpose(var_trim));
			//LOLCAT(transpose(partition + exponent));
			//LOLCAT(exp(transpose(partition + exponent)));
		}
	}

	Expression i_full_cost = sum(v_costs);
	return i_full_cost;
}

template <class Builder>
void AttentionalModel<Builder>::StartNewInstance(size_t algo
	, std::vector<dynet::Parameter>& v_params
	, Dict &sd
	, ComputationGraph &cg)
{
	std::vector<Expression> exp_wrd_embeddings;
	exp_wrd_embeddings.push_back(lookup(cg, p_cs, sd.convert("<s>")));
	for(auto t : boost::irange(0, (int)v_params.size())){		
	if (algo == RELOPT_ALGO::SOFTMAX){// SOFTMAX approach
		Expression i_p = parameter(cg, v_params[t]);
		exp_wrd_embeddings.push_back(GetWordEmbeddingVector(softmax(i_p)));
	}
	else if (algo == RELOPT_ALGO::SPARSEMAX){// SPARSEMAX approach
		Expression i_p = parameter(cg, v_params[t]);
		exp_wrd_embeddings.push_back(GetWordEmbeddingVector(sparsemax(i_p)));
	}
	else if (algo == RELOPT_ALGO::EG || algo == RELOPT_ALGO::AEG){// EG or AEG approach
		Expression i_p = parameter(cg, v_params[t]);
		exp_wrd_embeddings.push_back(GetWordEmbeddingVector(i_p));
	}
	else
		assert("Unknown relopt algo! Failed!");		
	}
	exp_wrd_embeddings.push_back(lookup(cg, p_cs, sd.convert("</s>")));

	slen = exp_wrd_embeddings.size();//v_params.size() + 2/*BOS and EOS*/; 
	std::vector<Expression> source_embeddings;
	if (!rnn_src_embeddings) {
	for (unsigned s = 0; s < slen; ++s) 
		source_embeddings.push_back(exp_wrd_embeddings[s]);
	}
	else {
		// run a RNN backward and forward over the source sentence
		// and stack the top-level hidden states from each model as 
		// the representation at each position
		std::vector<Expression> src_fwd(slen);
		builder_src_fwd.new_graph(cg);
		builder_src_fwd.start_new_sequence();
		for (unsigned i = 0; i < slen; ++i) 
			src_fwd[i] = builder_src_fwd.add_input(exp_wrd_embeddings[i]);

		std::vector<Expression> src_bwd(slen);
		builder_src_bwd.new_graph(cg);
		builder_src_bwd.start_new_sequence();
		for (int i = slen-1; i >= 0; --i) {
			// offset by one position to the right, to catch </s> and generally
			// not duplicate the w_t already captured in src_fwd[t]
			src_bwd[i] = builder_src_bwd.add_input(exp_wrd_embeddings[i]);
		}

		for (unsigned i = 0; i < slen; ++i) 
			source_embeddings.push_back(concatenate(std::vector<Expression>({src_fwd[i], src_bwd[i]})));
	}
	src = concatenate_cols(source_embeddings); 

	// now for the target sentence
	i_R = parameter(cg, p_R); // hidden -> word rep parameter
	i_Q = parameter(cg, p_Q);
	i_P = parameter(cg, p_P);
	i_bias = parameter(cg, p_bias);  // word bias
	i_Wa = parameter(cg, p_Wa); 
	i_Ua = parameter(cg, p_Ua);
	i_va = parameter(cg, p_va);
	i_uax = i_Ua * src; 

	// reset aux_vecs counter, allowing the memory to be reused
	num_aux_vecs = 0;

	if (giza_fertility || giza_markov || giza_positional) {
	i_Ta = parameter(cg, p_Ta);   
		if (giza_positional) {
			i_src_idx = arange(cg, 0, slen, true, auxiliary_vector());
			i_src_len = repeat(cg, slen, std::log(1.0 + slen), auxiliary_vector());
		}
	}

	aligns.clear();
	aligns.push_back(repeat(cg, slen, 0.0f, auxiliary_vector()));

	// initialilse h from global information of the source sentence
#ifndef RNN_H0_IS_ZERO
	std::vector<Expression> h0;
	Expression i_src = average(source_embeddings); // try max instead?

	int hidden_layers = builder.num_h0_components();
	for (int l = 0; l < hidden_layers; ++l) {
		Expression i_Wh0 = parameter(cg, p_Wh0[l]);
		h0.push_back(tanh(i_Wh0 * i_src));
	}

	builder.new_graph(cg); 
	builder.start_new_sequence(h0);
#else
	builder.new_graph(cg); 
	builder.start_new_sequence();
#endif
}

template <class Builder>
Expression AttentionalModel<Builder>::BuildRevRelOptGraph(
	size_t algo
	, std::vector<dynet::Parameter>& v_params /*source*/
	, const std::vector<int>& target
	, dynet::ComputationGraph & cg
	, Dict &sd /*source vocabulary*/
	, Expression *alignment)
{
	StartNewInstance(algo, v_params, sd, cg);

	std::vector<Expression> errs;
	const unsigned tlen = target.size() - 1; 
	for (unsigned t = 0; t < tlen; ++t) {
		Expression i_r_t = AddInput(target[t], t, cg);
		Expression i_err = pickneglogsoftmax(i_r_t, target[t+1]);
		errs.push_back(i_err);
	}

	// save the alignment for later
	if (alignment != 0) {
		// pop off the last alignment column
		*alignment = concatenate_cols(aligns);
	}

	Expression i_nerr = sum(errs);
	return i_nerr;
}

template <class Builder>
std::string AttentionalModel<Builder>::GetRelOptOutput(dynet::ComputationGraph& cg
	, const std::vector<dynet::Parameter>& v_relopt_params, size_t algo, Dict &d, bool verbose)
{
	int ind_eos = d.convert("</s>");
	
	std::stringstream ss;
	for (auto& p : v_relopt_params){
		Expression i_y;
		if (algo == RELOPT_ALGO::EG || algo == RELOPT_ALGO::AEG)
			i_y = parameter(cg, p);
		else if (algo == RELOPT_ALGO::SOFTMAX)
			i_y = softmax(parameter(cg, p));
		else if (algo == RELOPT_ALGO::SPARSEMAX)
			i_y = sparsemax(parameter(cg, p));

		//cg.incremental_forward();

		std::vector<float> v_y_dist = as_vector(i_y.value());

		// FIXME: Add the bos/eos/unk penalties if required
		//v_y_dist[] = -1.f;// penalty since <s> never appears in the middle of a target sentence.
		//v_y_dist[GetUnkId()] = -1.f;

		//cerr << "[y]=" << print_vec(v_y_dist) << endl;
		std::vector<float>::iterator it = std::max_element(v_y_dist.begin(), v_y_dist.end());
		int index = std::distance(v_y_dist.begin(), it);
		if (index == ind_eos)// FIXME: early ignorance
			break;
		ss << d.convert(index) << " ";
		if (verbose) std::cerr << d.convert(index) << "(" << *it << ")" << " ";// console output
	}

	if (verbose)  std::cerr << std::endl;

	return ss.str();// optimised output
}
//---------------------------------------------------------------------------------------------

template <class Builder>
void AttentionalModel<Builder>::Display_ASCII(const std::vector<int> &source, const std::vector<int>& target, 
	ComputationGraph &cg, const Expression &alignment, Dict &sd, Dict &td)
{
	using namespace std;

	// display the alignment
	//float I = target.size() - 1;
	//float J = source.size() - 1;
	unsigned I = target.size();
	unsigned J = source.size();
	//vector<string> symbols{"\u2588","\u2589","\u258A","\u258B","\u258C","\u258D","\u258E","\u258F"};
	vector<string> symbols{".","o","*","O","@"};
	int num_symbols = symbols.size();
	vector<float> thresholds;
	thresholds.push_back(0.8/I);
	float lgap = (0 - std::log(thresholds.back())) / (num_symbols - 1);
	for (auto rit = symbols.begin(); rit != symbols.end(); ++rit) {
		float thr = std::exp(std::log(thresholds.back()) + lgap);
		thresholds.push_back(thr);
	}
	// FIXME: thresholds > 1, what's going on?
	//cout << thresholds.back() << endl;

	const Tensor &a = cg.get_value(alignment.i);
	//cout << "I = " << I << " J = " << J << endl;

	cout.setf(ios_base::adjustfield, ios_base::left);
	cout << setw(12) << "source" << "  ";
	cout.setf(ios_base::adjustfield, ios_base::right);
	for (unsigned j = 0; j < J; ++j) 
		cout << setw(2) << j << ' ';
	cout << endl;

	for (unsigned i = 0; i < I; ++i) {
		cout.setf(ios_base::adjustfield, ios_base::left);
		//cout << setw(12) << td.convert(target[i+1]) << "  ";
		cout << setw(12) << td.convert(target[i]) << "  ";
		cout.setf(ios_base::adjustfield, ios_base::right);
		float max_v = 0;
		int max_j = -1;
		for (unsigned j = 0; j < J; ++j) {
			float v = TensorTools::AccessElement(a, Dim({(unsigned int)j, (unsigned int)i}));
			string symbol;
			for (int s = 0; s <= num_symbols; ++s) {
				if (s == 0) 
					symbol = ' ';
				else
					symbol = symbols[s-1];
				if (s != num_symbols && v < thresholds[s])
					break;
			}
			cout << setw(2) << symbol << ' ';
			if (v >= max_v) {
				max_v = v;
				max_j = j;
			}
		}
		cout << setw(20) << "max Pr=" << setprecision(3) << setw(5) << max_v << " @ " << max_j << endl;
	}
	cout << resetiosflags(ios_base::adjustfield);
	for (unsigned j = 0; j < J; ++j) 
		cout << j << ":" << sd.convert(source[j]) << ' ';
	cout << endl;
}

template <class Builder>
void AttentionalModel<Builder>::Display_TIKZ(const std::vector<int> &source, const std::vector<int>& target, 
	 ComputationGraph &cg, const Expression &alignment, Dict &sd, Dict &td)
{
	using namespace std;

	// display the alignment
	unsigned I = target.size();
	unsigned J = source.size();

	const Tensor &a = cg.get_value(alignment.i);
	cout << a.d[0] << " x " << a.d[1] << endl;

	cout << "\\begin{tikzpicture}[scale=0.5]\n";
	for (unsigned j = 0; j < J; ++j) 
		cout << "\\node[anchor=west,rotate=90] at (" << j+0.5 << ", " << I+0.2 << ") { " << sd.convert(source[j]) << " };\n";
	for (unsigned i = 0; i < I; ++i) 
		cout << "\\node[anchor=west] at (" << J+0.2 << ", " << I-i-0.5 << ") { " << td.convert(target[i]) << " };\n";

	float eps = 0.01;
	for (unsigned i = 0; i < I; ++i) {
		for (unsigned j = 0; j < J; ++j) {
			float v = TensorTools::AccessElement(a, Dim({(unsigned int)j, (unsigned int)i}));
			//int val = int(pow(v, 0.5) * 100);
			int val = int(v * 100);
			cout << "\\fill[blue!" << val << "!black] (" << j+eps << ", " << I-i-1+eps << ") rectangle (" << j+1-eps << "," << I-i-eps << ");\n";
		}
	}
	cout << "\\draw[step=1cm,color=gray] (0,0) grid (" << J << ", " << I << ");\n";
	cout << "\\end{tikzpicture}\n";
}


template <class Builder>
std::vector<int>
AttentionalModel<Builder>::Greedy_Decode(const std::vector<int> &source, ComputationGraph& cg, 
	dynet::Dict &tdict, const std::vector<int>* ctx)
{
	const int sos_sym = tdict.convert("<s>");
	const int eos_sym = tdict.convert("</s>");

	std::vector<int> target;
	target.push_back(sos_sym); 

	//std::cerr << tdict.convert(target.back());
	unsigned t = 0;
	StartNewInstance(source, cg, ctx);
	while (target.back() != eos_sym) 
	{
		Expression i_scores = AddInput(target.back(), t, cg);
		Expression ydist = softmax(i_scores); // compiler warning, but see below

		// find the argmax next word (greedy)
		unsigned w = 0;
		auto dist = as_vector(cg.incremental_forward(ydist));
		auto pr_w = dist[w];
		for (unsigned x = 1; x < dist.size(); ++x) {
			if (dist[x] > pr_w) {
				w = x;
				pr_w = dist[x];
			}
		}

		// break potential infinite loop
		if (t > 2*source.size()) {
			w = eos_sym;
			pr_w = dist[w];
		}

		//std::cerr << " " << tdict.convert(w) << " [p=" << pr_w << "]";
		t += 1;
		target.push_back(w);
	}
	//std::cerr << std::endl;

	return target;
}

struct Hypothesis {
	Hypothesis() {};
	Hypothesis(RNNPointer state, int tgt, float cst, std::vector<Expression> &al)
		: builder_state(state), target({tgt}), cost(cst), aligns(al) {}
	Hypothesis(RNNPointer state, int tgt, float cst, Hypothesis &last, std::vector<Expression> &al)
		: builder_state(state), target(last.target), cost(cst), aligns(al) {
		target.push_back(tgt);
	}
	RNNPointer builder_state;
	std::vector<int> target;
	float cost;
	std::vector<Expression> aligns;
};

template <class Builder>
std::vector<int>
AttentionalModel<Builder>::Beam_Decode(const std::vector<int> &source, ComputationGraph& cg, 
	unsigned beam_width, dynet::Dict &tdict, const std::vector<int>* ctx)
{
	const unsigned sos_sym = tdict.convert("<s>");
	const unsigned eos_sym = tdict.convert("</s>");

	StartNewInstance(source, cg, ctx);

	std::vector<Hypothesis> chart;
	chart.push_back(Hypothesis(builder.state(), sos_sym, 0.0f, aligns));

	std::vector<unsigned> vocab(boost::copy_range<std::vector<unsigned>>(boost::irange(0u, tgt_vocab_size)));
	std::vector<Hypothesis> completed;

	for (unsigned steps = 0; completed.size() < beam_width && steps < 2*source.size(); ++steps) {
		std::vector<Hypothesis> new_chart;

		for (auto &hprev: chart) {
			//std::cerr << "hypo t[-1]=" << tdict.convert(hprev.target.back()) << " cost " << hprev.cost << std::endl;
			if (giza_markov || giza_fertility) 
				aligns = hprev.aligns;
			Expression i_scores = AddInput(hprev.target.back(), hprev.target.size()-1, cg, &hprev.builder_state);
			Expression ydist = softmax(i_scores); 

			// find the top k best next words
			auto dist = as_vector(cg.incremental_forward(ydist));
			std::partial_sort(vocab.begin(), vocab.begin()+beam_width, vocab.end(), 
					[&dist](unsigned v1, unsigned v2) { return dist[v1] > dist[v2]; });

			// add to chart
			for (auto vi = vocab.begin(); vi < vocab.begin() + beam_width; ++vi) {
				//std::cerr << "\t++word " << tdict.convert(*vi) << " prob " << dist[*vi] << std::endl;
				//if (new_chart.size() < beam_width) {
					Hypothesis hnew(builder.state(), *vi, hprev.cost - std::log(dist[*vi]), hprev, aligns);
					if (*vi == eos_sym)
						completed.push_back(hnew);
					else
						new_chart.push_back(hnew);
				//} 
			}
		}

		if (new_chart.size() > beam_width) {
			// sort new_chart by score, to get kbest candidates
			std::partial_sort(new_chart.begin(), new_chart.begin()+beam_width, new_chart.end(),
					[](Hypothesis &h1, Hypothesis &h2) { return h1.cost < h2.cost; });
			new_chart.resize(beam_width);
		}
		chart.swap(new_chart);
	}

	// sort completed by score, adjusting for length -- not very effective, too short!
	auto best = std::min_element(completed.begin(), completed.end(),
			[](Hypothesis &h1, Hypothesis &h2) { return h1.cost/h1.target.size() < h2.cost/h2.target.size(); });
	assert(best != completed.end());

	return best->target;
}

template <class Builder>
std::vector<int> AttentionalModel<Builder>::Sample(const std::vector<int> &source, ComputationGraph& cg, dynet::Dict &tdict,
	const std::vector<int> *ctx)
{
	const int sos_sym = tdict.convert("<s>");
	const int eos_sym = tdict.convert("</s>");

	std::vector<int> target;
	target.push_back(sos_sym); 

	std::cerr << tdict.convert(target.back());
	int t = 0;
	StartNewInstance(source, cg, ctx);
	while (target.back() != eos_sym) 
	{
		Expression i_scores = AddInput(target.back(), t, cg);
		Expression ydist = softmax(i_scores);

	// in rnnlm.cc there's a loop around this block -- why? can incremental_forward fail?
		auto dist = as_vector(cg.incremental_forward(ydist));
	double p = rand01();
		unsigned w = 0;
		for (; w < dist.size(); ++w) {
		p -= dist[w];
		if (p < 0) break;
		}
	// this shouldn't happen
	if (w == dist.size()) w = eos_sym;

		std::cerr << " " << tdict.convert(w) << " [p=" << dist[w] << "]";
		t += 1;
		target.push_back(w);
	}
	std::cerr << std::endl;

	return target;
}

template <class Builder>
void AttentionalModel<Builder>::Display_Fertility(const std::vector<int> &source, Dict &sd)
{
	ComputationGraph cg;
	StartNewInstance(source, cg);
	assert(global_fertility);

	Expression fbias = concatenate_cols(std::vector<Expression>(slen, parameter(cg, p_bfhid)));
	Expression mbias = concatenate(std::vector<Expression>(slen, parameter(cg, p_bfmu)));
	Expression vbias = concatenate(std::vector<Expression>(slen, parameter(cg, p_bfvar)));
	Expression fhid = tanh(transpose(fbias + parameter(cg, p_Wfhid) * src));  
	Expression mu = mbias + fhid * parameter(cg, p_Wfmu);
	auto mu_vec = as_vector(cg.incremental_forward(mu)); 
	Expression var = exp(vbias + fhid * parameter(cg, p_Wfvar));
	auto var_vec = as_vector(cg.incremental_forward(var)); 

	for (unsigned j = 1; j < slen-1; ++j) 
		std::cout << sd.convert(source[j]) << '\t' << mu_vec[j] << '\t' << var_vec[j] << '\n';
}

template <class Builder>
void AttentionalModel<Builder>::Display_Empirical_Fertility(const std::vector<int> &source, const std::vector<int> &target, Dict &sd)
{
	ComputationGraph cg;
	Expression alignment;
	ModelStats stats; 
	BuildGraph(source, target, cg, stats, &alignment);

	Expression totals = sum_cols(alignment);
	auto totals_vec = as_vector(cg.incremental_forward(totals));

	for (unsigned j = 0; j < slen; ++j) 
		std::cout << sd.convert(source[j]) << '\t' << totals_vec[j] << '\n';
}

#undef WTF
#undef KTHXBYE
#undef LOLCAT

}; // namespace dynet
