#include "attentional.h"

#include <iostream>
#include <fstream>
#include <sstream>

#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/program_options/parsers.hpp>
#include <boost/program_options/variables_map.hpp>

using namespace std;
using namespace dynet;
using namespace boost::program_options;

#define WTF(expression) \
	cout << #expression << " has dimensions " << cg.nodes[expression.i]->dim << endl;

std::vector<int> Read_Numbered_Sentence(const std::string& line, Dict* sd, std::vector<int> &ids);

namespace dynet {

template <class Builder>
struct BiAttentionalModel {
	explicit BiAttentionalModel(Model *model, bool _rnn_src_embeddings, bool _giza_positional, 
		bool _giza_markov, bool _giza_fertility, bool _doc_context,
		bool _global_fertility, double trace_weight, bool _shared_embeddings=false) 
		: s2t_model(model, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, SLAYERS, TLAYERS,
		HIDDEN_DIM, ALIGN_DIM, _rnn_src_embeddings, _giza_positional, _giza_markov, _giza_fertility, _doc_context, _global_fertility),
		t2s_model(model, TGT_VOCAB_SIZE, SRC_VOCAB_SIZE, SLAYERS, TLAYERS,
		HIDDEN_DIM, ALIGN_DIM, _rnn_src_embeddings, _giza_positional, _giza_markov, _giza_fertility, _doc_context, _global_fertility, (_shared_embeddings==true)?&s2t_model.p_ct:nullptr, (_shared_embeddings==true)?&s2t_model.p_cs:nullptr)/*FIXME: testing required?*/
		, rnn_src_embeddings(_rnn_src_embeddings) 
		, giza_positional(_giza_positional), giza_markov(_giza_markov), giza_fertility(_giza_fertility)
		, doc_context(_doc_context)
		, global_fertility(_global_fertility)
	  	, shared_embeddings(_shared_embeddings)
		, m_trace_weight(trace_weight)
	{}

	AttentionalModel<Builder> s2t_model;
	AttentionalModel<Builder> t2s_model;

	bool rnn_src_embeddings;
	bool giza_positional;
	bool giza_markov;
	bool giza_fertility;
	bool doc_context;
	bool global_fertility;

	bool shared_embeddings = false;// both models can share their word embeddings.

	double m_trace_weight = 0.f;

	Expression s2t_align, t2s_align;
	Expression s2t_xent, t2s_xent, trace_bonus;

	void Add_Global_Fertility_Params(dynet::Model* model)
	{
		s2t_model.Add_Global_Fertility_Params(model, HIDDEN_DIM, rnn_src_embeddings);
		t2s_model.Add_Global_Fertility_Params(model, HIDDEN_DIM, rnn_src_embeddings);
	}

	// return Expression of total loss
	Expression BuildGraph(const vector<int> &source, const vector<int>& target, ComputationGraph& cg)
	{
		// FIXME: slightly wasteful, the embeddings of the source and target are done twice each
		// @vhoang2: fixed, see shared_embeddings!
		ModelStats stats_s2t, stats_t2s;// unused
		s2t_xent = s2t_model.BuildGraph(source, target, cg, stats_s2t, &s2t_align);
		t2s_xent = t2s_model.BuildGraph(target, source, cg, stats_t2s, &t2s_align);

		trace_bonus = trace_of_product(t2s_align, transpose(s2t_align));// FIXME: trace_of_product is not supporting CUDA yet!
		//cout << "xent: src=" << *cg.get_value(src_xent.i) << " tgt=" << *cg.get_value(tgt_xent.i) << endl;
		//cout << "trace bonus: " << *cg.get_value(trace_bonus.i) << endl;

		return s2t_xent + t2s_xent - m_trace_weight * trace_bonus;
		//return src_xent + tgt_xent;
	}

	void Load(const std::string &model_file, Model &model){
		cerr << "Initialising bi-model parameters from file: " << model_file << endl;
		dynet::load_dynet_model(model_file, &model);// FIXME: use binary streaming instead for saving disk spaces
	}

	void Initialise(const std::string &src_file, const std::string &tgt_file, Model &model)
	{
		if (shared_embeddings) assert("embedding_shared parameter is not correct! Initialisation only supports loading two separate pre-trained models.");	

		Model sm, tm;
		AttentionalModel<Builder> smb(&sm, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, 
			SLAYERS, TLAYERS, HIDDEN_DIM, ALIGN_DIM, rnn_src_embeddings, giza_positional, giza_markov, giza_fertility, doc_context, global_fertility);
		AttentionalModel<Builder> tmb(&tm, TGT_VOCAB_SIZE, SRC_VOCAB_SIZE,
			SLAYERS, TLAYERS, HIDDEN_DIM, ALIGN_DIM, rnn_src_embeddings, giza_positional, giza_markov, giza_fertility, doc_context, global_fertility);

		//for (const auto &p : sm.lookup_parameters_list())  
			//std::cerr << "\tlookup size: " << p->values[0].d << " number: " << p->values.size() << std::endl;
		//for (const auto &p : sm.parameters_list())  
			//std::cerr << "\tparam size: " << p->values.d << std::endl;

		std::cerr << "... loading " << src_file << " ..." << std::endl;
		{
			ifstream in(src_file);
			boost::archive::text_iarchive ia(in);
			ia >> sm;
		}

		std::cerr << "... loading " << tgt_file << " ... ";
		{
			ifstream in(tgt_file);
			boost::archive::text_iarchive ia(in);
			ia >> tm;
		}
		
		std::cerr << " done!" << endl;
		std::cerr << "... merging parameters ..." << std::endl;

		unsigned lid = 0;
		auto &lparams = model.lookup_parameters_list();
		//cerr << "lparams.size()=" << lparams.size() << "; " << "2*lookup_parameters_list().size()=" << sm.lookup_parameters_list().size() + tm.lookup_parameters_list().size() << endl;
		assert(lparams.size() == 2*sm.lookup_parameters_list().size());
		for (const auto &p : sm.lookup_parameters_list())  {
			for (unsigned i = 0; i < p->values.size(); ++i) 
				dynet::TensorTools::CopyElements(lparams[lid]->values[i], p->values[i]);
			//#if HAVE_CUDA
			//	CUDA_CHECK(cudaMemcpy(lparams[lid]->values[i].v, &p->values[i].v[0], sizeof(dynet::real) * p->values[i].d.size(), cudaMemcpyDeviceToHost));
			//#else
			//	memcpy(lparams[lid]->values[i].v, &p->values[i].v[0], sizeof(dynet::real) * p->values[i].d.size());
			//#endif
			lid++;
		}

		for (const auto &p : tm.lookup_parameters_list()) {
			for (unsigned i = 0; i < p->values.size(); ++i) 
				dynet::TensorTools::CopyElements(lparams[lid]->values[i], p->values[i]);
			//#if HAVE_CUDA
			//	CUDA_CHECK(cudaMemcpy(lparams[lid]->values[i].v, &p->values[i].v[0], sizeof(dynet::real) * p->values[i].d.size(), cudaMemcpyDeviceToHost));
			//#else
			//	memcpy(lparams[lid]->values[i].v, &p->values[i].v[0], sizeof(dynet::real) * p->values[i].d.size());
			//#endif
			lid++;		
		}
		assert(lid == lparams.size());

		unsigned did = 0;
		auto &dparams = model.parameters_list();
		for (const auto &p : sm.parameters_list()) {
			dynet::TensorTools::CopyElements(dparams[did++]->values, p->values);
			//#if HAVE_CUDA
			//	CUDA_CHECK(cudaMemcpy(dparams[did++]->values.v, &p->values.v[0], sizeof(dynet::real) * p->values.d.size(), cudaMemcpyDeviceToHost));
			//#else
			//	memcpy(dparams[did++]->values.v, &p->values.v[0], sizeof(dynet::real) * p->values.d.size());
			//#endif
		}
		for (const auto &p : tm.parameters_list()) {
			dynet::TensorTools::CopyElements(dparams[did++]->values, p->values);
			//#if HAVE_CUDA
			//	CUDA_CHECK(cudaMemcpy(dparams[did++]->values.v, &p->values.v[0], sizeof(dynet::real) * p->values.d.size(), cudaMemcpyDeviceToHost));
			//#else
			//	memcpy(dparams[did++]->values.v, &p->values.v[0], sizeof(dynet::real) * p->values.d.size());
			//#endif
		}
		assert(did == dparams.size());
	}
};

}
