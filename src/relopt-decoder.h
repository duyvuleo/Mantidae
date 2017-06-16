/*
 * relopt-decoder.h
 *
 *  Created on: 4 Oct 2016
 *	  Author: vhoang2
 */

#ifndef RELOPT_DECODER_H_
#define RELOPT_DECODER_H_

#include "relopt-def.h"

#include "attentional.h" // AM
#include "biattentional.h" // BAM
#include "rnnlm.h" // RNNLM

#include "dict-utils.h"

#include <dynet/dict.h>
#include <dynet/training.h>
#include <dynet/model.h>
#include <dynet/nodes.h>
#include <dynet/rnn.h>

#include <vector>
#include <time.h>
#include <ctime>
#include <limits> // std::numeric_limits
#include <sys/stat.h>
#include <unistd.h>
#include <algorithm>
#include <sstream>
#include <random>
#include <unistd.h>
#include <algorithm>

#include <boost/iterator/counting_iterator.hpp>
#include <boost/range/irange.hpp>

std::vector<std::string> SplitWords(const std::string & str);

using namespace std;
using namespace dynet;

namespace dynet {

struct RelOptOutput{
	std::string decoded_sent;
	float cost;
	size_t iteration;
	RelOptOutput(){
		decoded_sent = "";
		cost = 0.f;
		iteration = 0;
	}
	RelOptOutput(const std::string& ds, float c, size_t it){
		decoded_sent = ds;
		cost = c;
		iteration = it;
	}
};

struct DecTimeInfo{
	float elapsed_fwd;
	float elapsed_bwd;
	float elapsed_upd;
	float elapsed_oth;
	unsigned iterations;
	DecTimeInfo(){
		elapsed_fwd = 0.f;
		elapsed_bwd = 0.f;
		elapsed_upd = 0.f;
		elapsed_oth = 0.f;
		iterations = 0;
	}
	void Update(float f, float b, float u, float o, unsigned it){
		elapsed_fwd += f;
		elapsed_bwd += b;
		elapsed_upd += u;
		elapsed_oth += o;
		iterations += it;
	}
};

struct RelOptConfig{
	size_t algorithm;// 1:EG, 2:AEG, 3:SOFTMAX, 5:SPARSEMAX
	unsigned aeg_opt;// 0: Adam; 1: RMSProp
	size_t intialization_type;// 0:RANDOM, 1:REFERENCE_PROBABILITY, 2:REFERENCE_ONE_HOT

	// learning rate stuffs
	float eta;
	float eta_decay;
	float eta_power;

	// momemtum for gradient-based optimisation
	float momentum;

	// maximum no. of iterations
	size_t max_iters;

	// weight of left-to-right/source-to-target model
	float m_weight;

	// for coverage constraint
	float coverage_weight;
	float coverage_C;

	// for global fertility constraint
	float glofer_weight;

	// for joint decoding in bidirectional models
	float jdec_bidir_alpha;
	float jdec_biling_trace_alpha;

	// joint decoding with a monolingual language model
	float jdec_mlm_alpha;

	// joint decoding in bilingual AM models
	float jdec_biling_alpha;

	// add extra words
	unsigned add_extra_words;

	// gamma for entropy regularizer of SOFTMAX and SPARSEMAX
	float ent_gamma;

	// cyclical learning rate
	float clr_lb_eta;
	float clr_ub_eta;
	float clr_stepsize;
	float clr_gamma;

	// Adam optimization
	float aeg_beta_1;	
	float aeg_beta_2;
	float aeg_eps;	

	// for noise injection
	float noise_eta;
	float noise_gamma;
	unsigned sgld;

	RelOptConfig()
	{
		algorithm = RELOPT_ALGO::EG;
		aeg_opt = 0;
		intialization_type = RELOPT_INIT::UNIFORM;

		eta = 1.f;
		eta_decay = 2.f;
		eta_power = 0.f;

		momentum = 0.f;

		max_iters = 40;

		// weight of left-to-right/source-to-target model
		m_weight = 1.f;

		// coverage constraint
		coverage_weight = 0.f;
		coverage_C = 1.0f;

		// global fertility constraint weight
		glofer_weight = 0.f;

		// joint decoding in bidirectional AM models
		jdec_bidir_alpha = 0.f;

		// joint decoding with a monolingual language model
		jdec_mlm_alpha = 0.f;

		// joint decoding in bilingual AM models
		jdec_biling_alpha = 0.f;
		jdec_biling_trace_alpha = 0.f;

		add_extra_words = 0;

		// gamma for entropy regularizer of SOFTMAX and SPARSEMAX
		ent_gamma = 0.1f;

		// cyclical learning rate
		clr_lb_eta = 0.f;
		clr_ub_eta = 0.f;
		clr_stepsize = 0.f;// 1-8 x iterations
		clr_gamma = 0.f;// 0.99994f

		// Adam	
		aeg_beta_1 = 0.9f;
		aeg_beta_2 = 0.9f;
		aeg_eps = 1e-8;

		// for noise injection
		noise_eta = 0.f;
		noise_gamma = 0.55f;
		sgld = 0;
	}
};

template <class AM_t, class BAM_t, class RNNLM_t>
class RelOptDecoder {
public:
	void LoadModel(AM_t *am
		, AM_t *am_r2l
		, AM_t *am_t2s
		, BAM_t *bam
		, RNNLM_t *rnnlm
		, dynet::Dict* vocab_src
		, dynet::Dict* vocab_trg);

	RelOptOutput GDDecode(const string& src_sent
		, const std::string& trg_ref
		, const RelOptConfig& relopt_cf);
	RelOptOutput GDDecode(const string& src_sent
		, const std::string& trg_ref
		, const RelOptConfig& relopt_cf, DecTimeInfo& dti);
	RelOptOutput GDDecodeNBest_Batch(const string& src_sent
		, const std::vector<std::string>& trg_nbest
		, const RelOptConfig& relopt_cf);
	RelOptOutput GDDecodeNBest(const string& src_sent
		, const std::vector<std::string>& trg_nbest
		, const RelOptConfig& relopt_cf);
	RelOptOutput GDDecode_MinMax(const string& src_sent
		, const std::string& trg_ref
		, const RelOptConfig& relopt_cf);

	float GetNLLCost(const string& src_sent, const std::string& trg_sent);
	float GetNLLCost(const string& src_sent, const std::string& trg_sent
		, dynet::ComputationGraph& cg);

	void SetVerbose(bool verbose){verbose_ = verbose;}

protected:
	void InitializeParameters(const std::string& src, const std::string& trg_ref
		, dynet::Model& model
		, std::vector<dynet::Parameter>& v_params
		, const RelOptConfig& relopt_cf);
	void InitializeParameters(const std::string& src, const std::vector<std::string>& trg_nbest
		, dynet::Model& model
		, std::vector<std::vector<dynet::Parameter>>& v_nbest_params
		, const RelOptConfig& relopt_cf);

public:
	RelOptDecoder();
	virtual ~RelOptDecoder();

protected:

	Dict *vocab_src_, *vocab_trg_;// vocabularies

	AM_t* am_;// attentional encoder-decoder object pointer (including encoders, decoder, attention mechanism)
	AM_t* am_r2l_;// right-to-left attentional encoder-decoder object pointer
	AM_t* am_t2s_;// target-to-source attentional encoder-decoder object pointer
	BAM_t* bam_;// biattentional (source-to-target and target-to-source) encoder-decoder object pointer
	RNNLM_t* rnnlm_;// additional monolingual RNNLM
	// other object pointers (if possible)
	//...

	bool verbose_;// to be chatty
};

template <class AM_t, class BAM_t, class RNNLM_t>
RelOptDecoder<AM_t,BAM_t,RNNLM_t>::RelOptDecoder() : vocab_src_(nullptr), vocab_trg_(nullptr)
	, am_(nullptr)
	, am_r2l_(nullptr)
	, am_t2s_(nullptr)
	, bam_(nullptr)
	, rnnlm_(nullptr)
	, verbose_(0)
{
	// TODO Auto-generated constructor stub
}

template <class AM_t, class BAM_t, class RNNLM_t>
RelOptDecoder<AM_t,BAM_t,RNNLM_t>::~RelOptDecoder() {
	// TODO Auto-generated destructor stub
}

template <class AM_t, class BAM_t, class RNNLM_t>
void RelOptDecoder<AM_t,BAM_t,RNNLM_t>::LoadModel(AM_t *am
	, AM_t *am_r2l
	, AM_t *am_t2s
	, BAM_t *bam
	, RNNLM_t *rnnlm
	, dynet::Dict* vocab_src
	, dynet::Dict* vocab_trg)
{
	am_ = am;
	am_r2l_ = am_r2l;
	am_t2s_ = am_t2s;
	bam_ = bam;
	rnnlm_ = rnnlm;
	vocab_src_ = vocab_src;
	vocab_trg_ = vocab_trg;
}

// computation cost (or NLL) given a source and target sentence pair
template <class AM_t, class BAM_t, class RNNLM_t>
float RelOptDecoder<AM_t,BAM_t,RNNLM_t>::GetNLLCost(const string& src_sent, const std::string& trg_sent)
{
	Sentence i_src_sent = ParseWords(*vocab_src_, src_sent);
	Sentence i_trg_sent;
	if (trg_sent.find("<s>", 0, 3) == std::string::npos) // FIXME: not a good way?
		 i_trg_sent = ParseWords(*vocab_trg_, "<s> " + trg_sent + " </s>");
	else
		 i_trg_sent = ParseWords(*vocab_trg_, trg_sent);

	dynet::ComputationGraph cg;
	ModelStats stats;
	auto iloss = (1.f / (float)(i_trg_sent.size() - 1)) * am_->BuildGraph(i_src_sent, i_trg_sent, cg, stats);// normalized NLL

	return as_scalar(cg.forward(iloss));
}

template <class AM_t, class BAM_t, class RNNLM_t>
float RelOptDecoder<AM_t,BAM_t,RNNLM_t>::GetNLLCost(const string& src_sent, const std::string& trg_sent
		, dynet::ComputationGraph& cg)
{
	Sentence i_src_sent = ParseWords(*vocab_src_, src_sent);
	Sentence i_trg_sent;
	if (trg_sent.find("<s>", 0, 3) == std::string::npos) // FIXME: not a good way?
		 i_trg_sent = ParseWords(*vocab_trg_, "<s> " + trg_sent + " </s>");
	else
		 i_trg_sent = ParseWords(*vocab_trg_, trg_sent);

	ModelStats stats;
	auto iloss = (1.f / (float)(i_trg_sent.size() - 1)) * am_->BuildGraph(i_src_sent, i_trg_sent, cg, stats);// normalized NLL

	return as_scalar(cg.forward(iloss));
}

//*************************************************************************************************************************************************
// Sentence decoding with relaxed optimization algorithms (initialization: uniform/greedy/beam)
template <class AM_t, class BAM_t, class RNNLM_t>
void RelOptDecoder<AM_t,BAM_t,RNNLM_t>::InitializeParameters(const std::string& src, const std::string& trg_ref
		, dynet::Model& model
		, std::vector<dynet::Parameter>& v_params
		, const RelOptConfig& relopt_cf)
{
	v_params.clear();

	// get target vocab size
	size_t trg_vocab_size = vocab_trg_->size();

	// add relaxed inference parameters
	if (relopt_cf.intialization_type == RELOPT_INIT::UNIFORM){
		dynet::ParameterInitConst param_init_const(1.f / (float)trg_vocab_size);
		std::vector<std::string> ref_words = SplitWords(trg_ref);
		unsigned exl = 0;
		if (ref_words[0] == "<s>" && ref_words[ref_words.size() - 1] == "</s>") exl = 2;
		for (size_t i = 0; i < ref_words.size() - exl/*whether excluding BOS and EOS?*/ + relopt_cf.add_extra_words; i++){
			v_params.push_back(model.add_parameters({(unsigned int)trg_vocab_size}, param_init_const));// uniform initialization
			//v_params.push_back(model.add_parameters({(unsigned int)trg_vocab_size})); // random initialization
		}
	}
	else if (relopt_cf.intialization_type == RELOPT_INIT::REFERENCE_PROBABILITY){
		Sentence i_src_sent = ParseWords(*vocab_src_, src);
		Sentence i_trg_sent = ParseWords(*vocab_trg_, trg_ref);

		int ibos = vocab_trg_->convert("<s>");
		int ieos = vocab_trg_->convert("</s>");
		if (*(i_trg_sent.begin()) != ibos)
			i_trg_sent.insert(i_trg_sent.begin(), ibos);
		if (*(i_trg_sent.end()-1) != ieos)
			i_trg_sent.push_back(ieos);

		// reset the computation graph
		dynet::ComputationGraph cg;
		std::vector<std::vector<float>> v_preds;
		if (relopt_cf.algorithm == RELOPT_ALGO::SOFTMAX
			|| relopt_cf.algorithm == RELOPT_ALGO::SPARSEMAX)
			am_->BuildGraph(i_src_sent, i_trg_sent, cg, v_preds, false);// v_preds are unnormalized predictions from the model
		else if (relopt_cf.algorithm == RELOPT_ALGO::EG || relopt_cf.algorithm == RELOPT_ALGO::AEG)
			am_->BuildGraph(i_src_sent, i_trg_sent, cg, v_preds);// v_preds are softmax-normalized predictions from the model

		unsigned ind = 0;
		for (auto& pred : v_preds){
			if (ind != v_preds.size() - 1){// excluding </s> prediction
				dynet::ParameterInitFromVector param_init_vector(pred);
				v_params.push_back(model.add_parameters({(unsigned int)trg_vocab_size}, param_init_vector));
			}

			ind++;
		}

		// add additional words, potentially helping the model expand the reference translations
		if (relopt_cf.add_extra_words > 0){
			// which one is better?
			//dynet::ParameterInitFromVector param_init_const(v_preds[ind - 1]);// initialization from </s> prediction
			dynet::ParameterInitFromVector param_init_const(v_preds[ind - 2]);// initialization from last word prediction
			//dynet::ParameterInitConst param_init_const(1.f / (float)trg_vocab_size);// uniform initialization
			for (unsigned i = 0; i < relopt_cf.add_extra_words; i++){
				v_params.push_back(model.add_parameters({(unsigned int)trg_vocab_size}, param_init_const));
			}
		}
	}
	else if (relopt_cf.intialization_type == RELOPT_INIT::REFERENCE_ONE_HOT){
		std::vector<std::string> ref_words = SplitWords(trg_ref);
		unsigned exl = 0;
		if (ref_words[0] == "<s>" && ref_words[ref_words.size() - 1] == "</s>") exl = 2;
		for (size_t i = 0; i < ref_words.size() - exl/*excluding BOS and EOS*/; i++){
			std::vector<float> v_vals(trg_vocab_size, 0.f);
			v_vals[vocab_trg_->convert(ref_words[i])] = 1.f /*+ log(vocab_trg_->size())*/;
			ParameterInitFromVector param_init_vector(v_vals);
			v_params.push_back(model.add_parameters({(unsigned int)trg_vocab_size}, param_init_vector));
		}

		// add additional words, potentially helping the model expand the reference translations
		if (relopt_cf.add_extra_words > 0){
			std::vector<float> v_vals(trg_vocab_size, 0.f);
			v_vals[vocab_trg_->convert("</s>")] = 1.f /*+ log(vocab_trg_->size())*/;
			ParameterInitFromVector param_init_const(v_vals);
			//dynet::ParameterInitConst param_init_const(1.f / (float)trg_vocab_size);// uniform initialization
			for (unsigned i = 0; i < relopt_cf.add_extra_words; i++){
				v_params.push_back(model.add_parameters({(unsigned int)trg_vocab_size}, param_init_const));
			}
		}
	}
	else assert("Unknown intialization!");
}

template <class AM_t, class BAM_t, class RNNLM_t>
RelOptOutput RelOptDecoder<AM_t,BAM_t,RNNLM_t>::GDDecode(const string& src_sent
		, const std::string& trg_ref
		, const RelOptConfig& relopt_cf)
{
	dynet::Model relopt_model;// local relaxed inference model

	//std::cerr << "GDDecode:1" << std::endl;
	vector<dynet::Parameter> v_relopt_params;// inference parameters live here!
	InitializeParameters(src_sent, trg_ref
			, relopt_model
			, v_relopt_params
			, relopt_cf);
	size_t L = v_relopt_params.size();// length of the desired target output (exclusing BOS and EOS)
	//cerr << "L=" << L << endl;

	// create SGD trainer
	//std::cerr << "GDDecode:2" << std::endl;
	dynet::Trainer* trainer = nullptr;
	if (relopt_cf.algorithm == RELOPT_ALGO::SOFTMAX
		|| relopt_cf.algorithm == RELOPT_ALGO::SPARSEMAX){//SOFTMAX or SPARSEMAX		
		if (relopt_cf.sgld != 0){// use Stochastic Gradient Langevin Dynamics
			if (relopt_cf.sgld == 1)
				trainer = new dynet::SGLDTrainer(relopt_model, relopt_cf.eta);// SGLD
			else if (relopt_cf.sgld == 2)
				trainer = new dynet::pSGLDTrainer(relopt_model, relopt_cf.eta);// pSGLD
		}
		else{// use conventional SGD methods
			if (relopt_cf.momentum != 0.f)
				trainer = new dynet::MomentumSGDTrainer(relopt_model, relopt_cf.eta, relopt_cf.momentum);// Momemtum SGD
			else
				trainer = new dynet::SimpleSGDTrainer(relopt_model, relopt_cf.eta);// Vanilla SGD
		}
		
		//FIXME: to support others as well!
		//trainer = new dynet::AdamTrainer(relopt_model);// Adam SGD
		//trainer = new dynet::AdadeltaTrainer(relopt_model);// AdaDelta SGD
		//trainer = new dynet::AdagradTrainer(relopt_model);// Adagrad SGD
		//trainer = new dynet::RmsPropTrainer(relopt_model);// RmsProp SGD
	}
	else if (relopt_cf.algorithm == RELOPT_ALGO::EG){//EG
		trainer = new dynet::EGTrainer(relopt_model, relopt_cf.eta, relopt_cf.momentum, relopt_cf.noise_eta);// use our own implementation of EGTrainer
		if (relopt_cf.clr_lb_eta != 0 && relopt_cf.clr_ub_eta != 0)
			((dynet::EGTrainer*)trainer)->enableCyclicalLR(relopt_cf.clr_lb_eta, relopt_cf.clr_ub_eta, relopt_cf.clr_stepsize, relopt_cf.clr_gamma);// EG with cyclical learning rate
	}
	else if (relopt_cf.algorithm == RELOPT_ALGO::AEG){//Adaptive EG (AEG) with Adam or RMSProp
		if (relopt_cf.aeg_opt == 0)
			trainer = new dynet::AdamEGTrainer(relopt_model, relopt_cf.eta, relopt_cf.aeg_beta_1, relopt_cf.aeg_beta_2, relopt_cf.aeg_eps);// use our own implementation of EGTrainer
		else
			trainer = new dynet::RMSPropEGTrainer(relopt_model, relopt_cf.eta, relopt_cf.aeg_eps, relopt_cf.aeg_beta_1);// use our own implementation of EGTrainer
	}
	else
		assert("Unknown relaxed optimization algorithm!");
	trainer->eta_decay = relopt_cf.eta_decay;// learning rate decay

	//std::cerr << "GDDecode:3" << std::endl;

	// convert the source sentence
	Sentence i_src_sent = ParseWords(*vocab_src_, src_sent);

	// perform the relaxed inference algo
	float best_fcost = std::numeric_limits<float>::max(), prev_fcost = best_fcost;
	unsigned t = 0, T = relopt_cf.max_iters, best_t = 0;
	std::string best_sent = "";
	while (t < T)
	{
		//cerr << "t=" << t << endl;

		//--------------------------------------------------------------------------------------------
		// FIXME: this code block can be moved outside the loop for efficiency.
		// FIXME: this step is repeated for every iteration due to cg?
		// (1) reset the computation graph
		dynet::ComputationGraph cg;

		// (2) pre-compute the source embedding representation
		//std::cerr << "GDDecode:3:(1)" << std::endl;
		if (bam_ != nullptr){
			bam_->s2t_model.ComputeTrgWordEmbeddingMatrix(cg);// source-to-target
			bam_->s2t_model.StartNewInstance(i_src_sent, cg, 0);
			bam_->t2s_model.ComputeSrcWordEmbeddingMatrix(cg);// target-to-source
		}
		else{
			am_->ComputeTrgWordEmbeddingMatrix(cg);// source-to-target/left-to-right model
			am_->StartNewInstance(i_src_sent, cg, 0);
		}
		if (relopt_cf.jdec_bidir_alpha > 0.f
			&& relopt_cf.jdec_bidir_alpha < 1.f){
			am_r2l_->ComputeTrgWordEmbeddingMatrix(cg);// right-to-left model
			am_r2l_->StartNewInstance(i_src_sent, cg, 0);
		}
		if (relopt_cf.jdec_mlm_alpha > 0.f
			&& relopt_cf.jdec_mlm_alpha < 1.f){
			rnnlm_->ComputeWordEmbeddingMatrix(cg);// monolingual RNN language model
		}
		if (relopt_cf.jdec_biling_alpha > 0.f
			&& relopt_cf.jdec_biling_alpha < 1.f){
			am_t2s_->ComputeSrcWordEmbeddingMatrix(cg);// target-to-source model
		}

		//cg.checkpoint();// checkpoint here refers to computation for encoder(s)
		//--------------------------------------------------------------------------------------------

		//cg.revert();// revert to last checkpoint
		
		// (3) build relaxed optimization graph
		//std::cerr << "GDDecode:3:(2a)" << std::endl;
		// left-to-right/source-to-target AM model
		dynet::expr::Expression i_alignment, i_coverage, i_glofer_nll, i_entropy;
		dynet::expr::Expression i_cost;
		if (bam_ != nullptr)
			i_cost =  (1.f/(float)(L + 1)) * bam_->s2t_model.BuildRelOptGraph(
					relopt_cf.algorithm
					, v_relopt_params
					, cg
					, *vocab_trg_
					, false //left-to-right model (default)
					, &i_entropy
					, &i_alignment
					, (relopt_cf.coverage_weight > 0.f) ? &i_coverage : nullptr, relopt_cf.coverage_C
					, (relopt_cf.glofer_weight > 0.f) ? &i_glofer_nll : nullptr);// normalized sum of negative log likelihoods (NLL)
		else
			i_cost =  (1.f/(float)(L + 1)) * am_->BuildRelOptGraph(
					relopt_cf.algorithm
					, v_relopt_params
					, cg
					, *vocab_trg_
					, false //left-to-right model (default)
					, &i_entropy
					, &i_alignment
					, (relopt_cf.coverage_weight > 0.f) ? &i_coverage : nullptr, relopt_cf.coverage_C
					, (relopt_cf.glofer_weight > 0.f) ? &i_glofer_nll : nullptr);// normalized sum of negative log likelihoods (NLL)
		//std::cerr << "GDDecode:3:(2b)" << std::endl;
		// right-to-left AM model
		dynet::expr::Expression i_cost_r2l;
		if (relopt_cf.jdec_bidir_alpha > 0.f
			&& relopt_cf.jdec_bidir_alpha < 1.f){
			i_cost_r2l = (1.f/(float)(L + 1)) * am_r2l_->BuildRelOptGraph(
					relopt_cf.algorithm
					, v_relopt_params /*target*/
					, cg
					, *vocab_trg_
					, true //right-to-left model
					, nullptr
					, nullptr
					, nullptr, 0
					, nullptr);// normalized sum of negative log likelihoods (NLL)
		}
		//std::cerr << "GDDecode:3:(2c)" << std::endl;
		// target-to-source AM model
		dynet::expr::Expression i_cost_t2s;
		dynet::expr::Expression i_alignment_t2s;
		if (bam_ != nullptr){
			i_cost_t2s = (1.f/(float)(i_src_sent.size() - 1)) * bam_->t2s_model.BuildRevRelOptGraph(
					relopt_cf.algorithm
					, v_relopt_params /*source*/
					, i_src_sent /*target*/
					, cg
					, *vocab_src_
					, &i_alignment_t2s);// normalized sum of negative log likelihoods (NLL)
		}
		else{
			if (relopt_cf.jdec_biling_alpha > 0.f && relopt_cf.jdec_biling_alpha < 1.f){
				i_cost_t2s = (1.f/(float)(i_src_sent.size() - 1)) * am_t2s_->BuildRevRelOptGraph(
					relopt_cf.algorithm
					, v_relopt_params /*source*/
					, i_src_sent /*target*/
					, cg
					, *vocab_src_
					, &i_alignment_t2s);// normalized sum of negative log likelihoods (NLL)
			}
		}
		//std::cerr << "GDDecode:3:(2d)" << std::endl;
		dynet::expr::Expression i_cost_mlm;
		if (relopt_cf.jdec_mlm_alpha > 0.f
			&& relopt_cf.jdec_mlm_alpha < 1.f){
			i_cost_mlm = (1.f/(float)(L + 1)) * rnnlm_->BuildRelOptGraph( relopt_cf.algorithm
					, v_relopt_params
					, cg
					, *vocab_trg_);// left-to-right or right-to-left direction will be implicitly recognized in model file.
		}

		// (4) compute the additional costs if required
		//std::cerr << "GDDecode:3:(3)" << std::endl;
		dynet::expr::Expression i_objective = relopt_cf.m_weight * i_cost;// NLL
		// coverage penalty
		if (relopt_cf.coverage_weight > 0.f){
			i_objective = i_objective + (1.f/(float)(L + 1)) * relopt_cf.coverage_weight * i_coverage;// FIXME: normalization for coverage is required!
		}
		// global fertility
		if (relopt_cf.glofer_weight > 0.f){
			i_objective = i_objective + (1.f/(float)(L + 1)) * relopt_cf.glofer_weight * i_glofer_nll;// FIXME: normalization for glofer is required!
		}
		// joint decoding in bidirectional models
		if (relopt_cf.jdec_bidir_alpha > 0.f
			&& relopt_cf.jdec_bidir_alpha < 1.f){
			i_objective = i_objective + relopt_cf.jdec_bidir_alpha * i_cost_r2l;
		}
		// joint decoding in bilingual models
		if ((relopt_cf.jdec_biling_alpha > 0.f
			&& relopt_cf.jdec_biling_alpha < 1.f) || bam_ != nullptr){
			i_objective = i_objective + relopt_cf.jdec_biling_alpha * i_cost_t2s;
			if (relopt_cf.jdec_biling_trace_alpha != 0.f){
				// if required, add a trace bonus of i_alignment and i_alignment_t2s following Cohn et al., 2016.
				dynet::expr::Expression i_trace_bonus = trace_of_product(i_alignment, transpose(i_alignment_t2s));// FIXME: this trace_of_product is not supporting CUDA yet!
				i_objective = i_objective - relopt_cf.jdec_biling_trace_alpha * i_trace_bonus;
			}
		}
		// joint decoding with monolingual RNN language model
		if (relopt_cf.jdec_mlm_alpha > 0.f
			&& relopt_cf.jdec_mlm_alpha < 1.f){
			i_objective = i_objective + relopt_cf.jdec_mlm_alpha * i_cost_mlm;
		}
		// entropy regularizer for SOFTMAX or SPARSEMAX
		//if (relopt_cf.algorithm == RELOPT_ALGO::SOFTMAX
		//	|| relopt_cf.algorithm == RELOPT_ALGO::SPARSEMAX)
		//	i_objective = i_objective - relopt_cf.ent_gamma *  (1.f/(float)L) * i_entropy;// FIXME: normalize the entropy sum according to sentence length?		

		// (5) do forward propagation step
		//std::cerr << "GDDecode:3:(4a)" << std::endl;
		cg.incremental_forward(i_objective);

		// grap the parts of the objective
		//std::cerr << "GDDecode:3:(4b)" << std::endl;
		float fcost = as_scalar(cg.get_value(i_cost.i));
		float fcoverage = 0.f;
		if (relopt_cf.coverage_weight > 0.f) {
			fcoverage = as_scalar(cg.get_value(i_coverage.i));
		}
		float fglofer = 0.f;
		if (relopt_cf.glofer_weight > 0.f){
			fglofer = as_scalar(cg.get_value(i_glofer_nll.i));
		}
		float fobj = as_scalar(cg.get_value(i_objective.i));
		float fcost_r2l = 0.f;
		if (relopt_cf.jdec_bidir_alpha > 0.f
			&& relopt_cf.jdec_bidir_alpha < 1.f)
			fcost_r2l = as_scalar(cg.get_value(i_cost_r2l.i));
		float fcost_t2s = 0.f;
		if (relopt_cf.jdec_biling_alpha > 0.f
			&& relopt_cf.jdec_biling_alpha < 1.f)
			fcost_t2s = as_scalar(cg.get_value(i_cost_t2s.i));
		float fcost_mlm = 0.f;
		if (relopt_cf.jdec_mlm_alpha > 0.f
			&& relopt_cf.jdec_mlm_alpha < 1.f)
			fcost_mlm = as_scalar(cg.get_value(i_cost_mlm.i));

		// (6) do backpropagation step (including computation of gradients which are stored in relopt_model)
		//std::cerr << "GDDecode:3:(5)" << std::endl;
		cg.backward(i_objective);// backpropagate for all nodes

		// (7) update inference parameters with requested optimization method (e.g., SGD, EG)
		//std::cerr << "GDDecode:3:(6)" << std::endl;
		float scale = 1.f;// FIXME: can be used if required!
		float clr = relopt_cf.eta / std::pow(1.f + t * relopt_cf.eta_decay, relopt_cf.eta_power);// simple learning rate annealing
		trainer->eta = clr;
		if (relopt_cf.algorithm == RELOPT_ALGO::EG){//EG
			((dynet::EGTrainer*)trainer)->update_noise_eta(relopt_cf.noise_eta / std::pow(1.f + t, relopt_cf.noise_gamma));// noise level annealing according to https://arxiv.org/pdf/1511.06807.pdf.
		}
		trainer->update(scale);// only update inference parameters

		// (8) verbose output
		//std::cerr << "GDDecode:3:(7)" << std::endl;
		if (verbose_) cerr << "All costs at step "
			<< t << " (eta=" << trainer->eta << ")" << ": "
			<< "l2r_nll=" << fcost
			<< "; r2l_nll=" << fcost_r2l
			<< "; t2s_nll=" << fcost_t2s
			<< "; mlm_nll=" << fcost_mlm
			<< "; C_coverage=" << fcoverage
			<< "; glofer_fertility=" << fglofer
			<< "; total_objective=" << fobj << endl;
		string decoded_sent = am_->GetRelOptOutput(cg, v_relopt_params, relopt_cf.algorithm, *vocab_trg_, verbose_);
		if (verbose_) cerr << "Result at step " << t << ": " << decoded_sent << " (discrete cost=" << GetNLLCost(src_sent, decoded_sent, cg) << ")" << endl;

		// (9) update the best result so far
		//std::cerr << "GDDecode:3:(8)" << std::endl;
		if (fcost < best_fcost){// FIXME: fobj or fcost?
			best_fcost = fcost;
			best_sent = decoded_sent;
			best_t = t;
		}

		// simple stopping criterion if change in costs is very small!
		if (t >= 1 && std::abs(prev_fcost - fcost) < 0.0001f)// FIXME: maybe smaller???
			break;

		prev_fcost = fcost;// update previous cost

		t++;// next iteration
	}

	if (verbose_) cerr << "***Best decoding result at step " << best_t << " (continuous cost=" << best_fcost << ")" << endl;

	delete trainer;

	return RelOptOutput(best_sent, best_fcost, best_t);
}

// (with time benchmarking)
template <class AM_t, class BAM_t, class RNNLM_t>
RelOptOutput RelOptDecoder<AM_t,BAM_t,RNNLM_t>::GDDecode(const string& src_sent
		, const std::string& trg_ref
		, const RelOptConfig& relopt_cf, DecTimeInfo& dti)
{
	dynet::Model relopt_model;// local relaxed inference model

	//std::cerr << "GDDecode:1" << std::endl;
	vector<dynet::Parameter> v_relopt_params;// inference parameters live here!
	InitializeParameters(src_sent, trg_ref
			, relopt_model
			, v_relopt_params
			, relopt_cf);
	size_t L = v_relopt_params.size();// length of the desired target output (exclusing BOS and EOS)
	//cerr << "L=" << L << endl;

	// create SGD trainer
	//std::cerr << "GDDecode:2" << std::endl;
	dynet::Trainer* trainer = nullptr;
	if (relopt_cf.algorithm == RELOPT_ALGO::SOFTMAX
		|| relopt_cf.algorithm == RELOPT_ALGO::SPARSEMAX){//SOFTMAX or SPARSEMAX
		if (relopt_cf.momentum != 0.f)
			trainer = new dynet::MomentumSGDTrainer(relopt_model, relopt_cf.eta, relopt_cf.momentum);// Momemtum SGD
		else
			trainer = new dynet::SimpleSGDTrainer(relopt_model, relopt_cf.eta);// Vanilla SGD
		//FIXME: to support others as well!
		//trainer = new dynet::AdamTrainer(relopt_model);// Adam SGD
		//trainer = new dynet::AdadeltaTrainer(relopt_model);// AdaDelta SGD
		//trainer = new dynet::AdagradTrainer(relopt_model);// Adagrad SGD
		//trainer = new dynet::RmsPropTrainer(relopt_model);// RmsProp SGD
	}
	else if (relopt_cf.algorithm == RELOPT_ALGO::EG){//EG
		trainer = new dynet::EGTrainer(relopt_model, relopt_cf.eta, relopt_cf.momentum, relopt_cf.noise_eta);// use our own implementation of EGTrainer
		if (relopt_cf.clr_lb_eta != 0 && relopt_cf.clr_ub_eta != 0)
			((dynet::EGTrainer*)trainer)->enableCyclicalLR(relopt_cf.clr_lb_eta, relopt_cf.clr_ub_eta, relopt_cf.clr_stepsize, relopt_cf.clr_gamma);// EG with cyclical learning rate
	}
	else if (relopt_cf.algorithm == RELOPT_ALGO::AEG){//Adaptive EG (AEG) with Adam or RMSProp
		if (relopt_cf.aeg_opt == 0)
			trainer = new dynet::AdamEGTrainer(relopt_model, relopt_cf.eta, relopt_cf.aeg_beta_1, relopt_cf.aeg_beta_2, relopt_cf.aeg_eps);// use our own implementation of EGTrainer
		else
			trainer = new dynet::RMSPropEGTrainer(relopt_model, relopt_cf.eta, relopt_cf.aeg_eps, relopt_cf.aeg_beta_1);// use our own implementation of EGTrainer
	}
	else
		assert("Unknown relaxed optimization algorithm!");
	trainer->eta_decay = relopt_cf.eta_decay;// learning rate decay

	//std::cerr << "GDDecode:3" << std::endl;

	// convert the source sentence
	Sentence i_src_sent = ParseWords(*vocab_src_, src_sent);

	//Timing
	Timer timer_steps("");
	float elapsed_fwd = 0.f, elapsed_bwd = 0.f, elapsed_upd = 0.f, elapsed_oth = 0.f;	

	// perform the relaxed inference algo
	float best_fcost = std::numeric_limits<float>::max(), prev_fcost = best_fcost;
	unsigned t = 0, T = relopt_cf.max_iters, best_t = 0;
	std::string best_sent = "";
	while (t < T)
	{
		//cerr << "t=" << t << endl;

		//--------------------------------------------------------------------------------------------
		// FIXME: this code block can be moved outside the loop for efficiency.
		// FIXME: this step is repeated for every iteration due to cg?
		// (1) reset the computation graph
		dynet::ComputationGraph cg;

		// (2) pre-compute the source embedding representation
		//std::cerr << "GDDecode:3:(1)" << std::endl;
		if (bam_ != nullptr){
			bam_->s2t_model.ComputeTrgWordEmbeddingMatrix(cg);// source-to-target
			bam_->s2t_model.StartNewInstance(i_src_sent, cg, 0);
			bam_->t2s_model.ComputeSrcWordEmbeddingMatrix(cg);// target-to-source
		}
		else{
			am_->ComputeTrgWordEmbeddingMatrix(cg);// source-to-target/left-to-right model
			am_->StartNewInstance(i_src_sent, cg, 0);
		}
		if (relopt_cf.jdec_bidir_alpha > 0.f
			&& relopt_cf.jdec_bidir_alpha < 1.f){
			am_r2l_->ComputeTrgWordEmbeddingMatrix(cg);// right-to-left model
			am_r2l_->StartNewInstance(i_src_sent, cg, 0);
		}
		if (relopt_cf.jdec_mlm_alpha > 0.f
			&& relopt_cf.jdec_mlm_alpha < 1.f){
			rnnlm_->ComputeWordEmbeddingMatrix(cg);// monolingual RNN language model
		}
		if (relopt_cf.jdec_biling_alpha > 0.f
			&& relopt_cf.jdec_biling_alpha < 1.f){
			am_t2s_->ComputeSrcWordEmbeddingMatrix(cg);// target-to-source model
		}

		//cg.checkpoint();// checkpoint here refers to computation for encoder(s)				
		
		//cg.revert();// revert to last checkpoint
		//--------------------------------------------------------------------------------------------

		// (3) build relaxed optimization graph
		//std::cerr << "GDDecode:3:(2a)" << std::endl;

		// left-to-right/source-to-target AM model
		dynet::expr::Expression i_alignment, i_coverage, i_glofer_nll, i_entropy;
		dynet::expr::Expression i_cost;
		if (bam_ != nullptr)
			i_cost =  (1.f/(float)(L + 1)) * bam_->s2t_model.BuildRelOptGraph(
					relopt_cf.algorithm
					, v_relopt_params
					, cg
					, *vocab_trg_
					, false //left-to-right model (default)
					, &i_entropy
					, &i_alignment
					, (relopt_cf.coverage_weight > 0.f) ? &i_coverage : nullptr, relopt_cf.coverage_C
					, (relopt_cf.glofer_weight > 0.f) ? &i_glofer_nll : nullptr);// normalized sum of negative log likelihoods (NLL)
		else
			i_cost =  (1.f/(float)(L + 1)) * am_->BuildRelOptGraph(
					relopt_cf.algorithm
					, v_relopt_params
					, cg
					, *vocab_trg_
					, false //left-to-right model (default)
					, &i_entropy
					, &i_alignment
					, (relopt_cf.coverage_weight > 0.f) ? &i_coverage : nullptr, relopt_cf.coverage_C
					, (relopt_cf.glofer_weight > 0.f) ? &i_glofer_nll : nullptr);// normalized sum of negative log likelihoods (NLL)
		//std::cerr << "GDDecode:3:(2b)" << std::endl;
		// right-to-left AM model
		dynet::expr::Expression i_cost_r2l;
		if (relopt_cf.jdec_bidir_alpha > 0.f
			&& relopt_cf.jdec_bidir_alpha < 1.f){
			i_cost_r2l = (1.f/(float)(L + 1)) * am_r2l_->BuildRelOptGraph(
					relopt_cf.algorithm
					, v_relopt_params /*target*/
					, cg
					, *vocab_trg_
					, true //right-to-left model
					, nullptr
					, nullptr
					, nullptr, 0
					, nullptr);// normalized sum of negative log likelihoods (NLL)
		}
		//std::cerr << "GDDecode:3:(2c)" << std::endl;
		// target-to-source AM model
		dynet::expr::Expression i_cost_t2s;
		dynet::expr::Expression i_alignment_t2s;
		if (bam_ != nullptr){
			i_cost_t2s = (1.f/(float)(i_src_sent.size() - 1)) * bam_->t2s_model.BuildRevRelOptGraph(
					relopt_cf.algorithm
					, v_relopt_params /*source*/
					, i_src_sent /*target*/
					, cg
					, *vocab_src_
					, &i_alignment_t2s);// normalized sum of negative log likelihoods (NLL)
		}
		else{
			if (relopt_cf.jdec_biling_alpha > 0.f && relopt_cf.jdec_biling_alpha < 1.f){
				i_cost_t2s = (1.f/(float)(i_src_sent.size() - 1)) * am_t2s_->BuildRevRelOptGraph(
					relopt_cf.algorithm
					, v_relopt_params /*source*/
					, i_src_sent /*target*/
					, cg
					, *vocab_src_
					, &i_alignment_t2s);// normalized sum of negative log likelihoods (NLL)
			}
		}
		//std::cerr << "GDDecode:3:(2d)" << std::endl;
		dynet::expr::Expression i_cost_mlm;
		if (relopt_cf.jdec_mlm_alpha > 0.f
			&& relopt_cf.jdec_mlm_alpha < 1.f){
			i_cost_mlm = (1.f/(float)(L + 1)) * rnnlm_->BuildRelOptGraph( relopt_cf.algorithm
					, v_relopt_params
					, cg
					, *vocab_trg_);// left-to-right or right-to-left direction will be implicitly recognized in model file.
		}

		// (4) compute the additional costs if required
		//std::cerr << "GDDecode:3:(3)" << std::endl;
		dynet::expr::Expression i_objective = relopt_cf.m_weight * i_cost;// NLL
		// coverage penalty
		if (relopt_cf.coverage_weight > 0.f){
			i_objective = i_objective + (1.f/(float)(L + 1)) * relopt_cf.coverage_weight * i_coverage;// FIXME: normalization for coverage is required!
		}
		// global fertility
		if (relopt_cf.glofer_weight > 0.f){
			i_objective = i_objective + (1.f/(float)(L + 1)) * relopt_cf.glofer_weight * i_glofer_nll;// FIXME: normalization for glofer is required!
		}
		// joint decoding in bidirectional models
		if (relopt_cf.jdec_bidir_alpha > 0.f
			&& relopt_cf.jdec_bidir_alpha < 1.f){
			i_objective = i_objective + relopt_cf.jdec_bidir_alpha * i_cost_r2l;
		}
		// joint decoding in bilingual models
		if ((relopt_cf.jdec_biling_alpha > 0.f
			&& relopt_cf.jdec_biling_alpha < 1.f) || bam_ != nullptr){
			i_objective = i_objective + relopt_cf.jdec_biling_alpha * i_cost_t2s;
			if (relopt_cf.jdec_biling_trace_alpha != 0.f){
				// if required, add a trace bonus of i_alignment and i_alignment_t2s following Cohn et al., 2016.
				dynet::expr::Expression i_trace_bonus = trace_of_product(i_alignment, transpose(i_alignment_t2s));// FIXME: this trace_of_product is not supporting CUDA yet!
				i_objective = i_objective - relopt_cf.jdec_biling_trace_alpha * i_trace_bonus;
			}
		}
		// joint decoding with monolingual RNN language model
		if (relopt_cf.jdec_mlm_alpha > 0.f
			&& relopt_cf.jdec_mlm_alpha < 1.f){
			i_objective = i_objective + relopt_cf.jdec_mlm_alpha * i_cost_mlm;
		}
		// entropy regularizer for SOFTMAX or SPARSEMAX
		//if (relopt_cf.algorithm == RELOPT_ALGO::SOFTMAX
		//	|| relopt_cf.algorithm == RELOPT_ALGO::SPARSEMAX)
		//	i_objective = i_objective + relopt_cf.ent_gamma * i_entropy;

		// (5) do forward propagation step
		//std::cerr << "GDDecode:3:(4a)" << std::endl;
		cg.forward(i_objective);

		elapsed_fwd += timer_steps.elapsed();
		timer_steps.reset();

		// grap the parts of the objective
		//std::cerr << "GDDecode:3:(4b)" << std::endl;
		float fcost = as_scalar(cg.get_value(i_cost.i));
		float fcoverage = 0.f;
		if (relopt_cf.coverage_weight > 0.f) {
			fcoverage = as_scalar(cg.get_value(i_coverage.i));
		}
		float fglofer = 0.f;
		if (relopt_cf.glofer_weight > 0.f){
			fglofer = as_scalar(cg.get_value(i_glofer_nll.i));
		}
		float fobj = as_scalar(cg.get_value(i_objective.i));
		float fcost_r2l = 0.f;
		if (relopt_cf.jdec_bidir_alpha > 0.f
			&& relopt_cf.jdec_bidir_alpha < 1.f)
			fcost_r2l = as_scalar(cg.get_value(i_cost_r2l.i));
		float fcost_t2s = 0.f;
		if (relopt_cf.jdec_biling_alpha > 0.f
			&& relopt_cf.jdec_biling_alpha < 1.f)
			fcost_t2s = as_scalar(cg.get_value(i_cost_t2s.i));
		float fcost_mlm = 0.f;
		if (relopt_cf.jdec_mlm_alpha > 0.f
			&& relopt_cf.jdec_mlm_alpha < 1.f)
			fcost_mlm = as_scalar(cg.get_value(i_cost_mlm.i));

		// (6) do backpropagation step (including computation of gradients which are stored in relopt_model)
		//std::cerr << "GDDecode:3:(5)" << std::endl;
		cg.backward(i_objective);// backpropagate for all nodes

		elapsed_bwd += timer_steps.elapsed();
		timer_steps.reset();

		// (7) update inference parameters with requested optimization method (e.g., SGD, EG)
		//std::cerr << "GDDecode:3:(6)" << std::endl;
		float scale = 1.f;// FIXME: can be used if required!
		float clr = relopt_cf.eta / std::pow(1.f + t * relopt_cf.eta_decay, relopt_cf.eta_power);// simple learning rate annealing
		trainer->eta = clr;
		if (relopt_cf.algorithm == RELOPT_ALGO::EG){//EG
			((dynet::EGTrainer*)trainer)->update_noise_eta(relopt_cf.noise_eta / std::pow(1.f + t, relopt_cf.noise_gamma));// noise level annealing according to https://arxiv.org/pdf/1511.06807.pdf.
		}
		trainer->update(scale);// only update inference parameters

		elapsed_upd += timer_steps.elapsed();
		timer_steps.reset();

		// verbose output
		//std::cerr << "GDDecode:3:(7)" << std::endl;
		if (verbose_) cerr << "All costs at step "
			<< t << " (eta=" << trainer->eta << ")" << ": "
			<< "l2r_nll=" << fcost
			<< "; r2l_nll=" << fcost_r2l
			<< "; t2s_nll=" << fcost_t2s
			<< "; mlm_nll=" << fcost_mlm
			<< "; C_coverage=" << fcoverage
			<< "; glofer_fertility=" << fglofer
			<< "; total_objective=" << fobj << endl;
		string decoded_sent = am_->GetRelOptOutput(cg, v_relopt_params, relopt_cf.algorithm, *vocab_trg_, verbose_);
		if (verbose_) cerr << "Result at step " << t << ": " << decoded_sent << " (discrete cost=" << GetNLLCost(src_sent, decoded_sent, cg) << ")" << endl;

		// (9) update the best result so far
		//std::cerr << "GDDecode:3:(8)" << std::endl;
		if (fcost < best_fcost){// FIXME: fobj or fcost?
			best_fcost = fcost;
			best_sent = decoded_sent;
			best_t = t;
		}

		// simple stopping criterion if change in costs is very small!
		if (t >= 1 && std::abs(prev_fcost - fcost) < 0.0001f)// FIXME: maybe smaller???
			break;

		prev_fcost = fcost;// update previous cost

		t++;// next iteration

		elapsed_oth += timer_steps.elapsed();
		timer_steps.reset();
	}

	if (verbose_) cerr << "***Best decoding result at step " << best_t << " (continuous cost=" << best_fcost << ")" << endl;

	dti.Update(elapsed_fwd, elapsed_bwd, elapsed_upd, elapsed_oth, best_t/*t*/);

	delete trainer;

	return RelOptOutput(best_sent, best_fcost, best_t);
}
//*************************************************************************************************************************************************

//*************************************************************************************************************************************************
// // Sentence decoding with relaxed optimization algorithms (initialization: n-best translation list) (work in progress)
template <class AM_t, class BAM_t, class RNNLM_t>
void RelOptDecoder<AM_t,BAM_t,RNNLM_t>::InitializeParameters(const std::string& src, const std::vector<std::string>& trg_nbest
		, dynet::Model& model
		, std::vector<std::vector<dynet::Parameter>>& v_nbest_params
		, const RelOptConfig& relopt_cf)
{
	v_nbest_params.clear();
	v_nbest_params.resize(trg_nbest.size());

	// get target vocab size
	size_t trg_vocab_size = vocab_trg_->size();

	// add relaxed inference parameters
	if (relopt_cf.intialization_type == RELOPT_INIT::UNIFORM){
		dynet::ParameterInitConst param_init_const(1.f / (float)trg_vocab_size);
		std::vector<std::string> ref_words = SplitWords(trg_nbest[0]);

		unsigned exl = 0;
		if (ref_words[0] == "<s>" && ref_words[ref_words.size() - 1] == "</s>") exl = 2;

		for (size_t n = 0; n < trg_nbest.size(); n++){
			for (size_t i = 0; i < ref_words.size() - exl/*whether excluding BOS and EOS?*/ + relopt_cf.add_extra_words; i++){
				v_nbest_params[n].push_back(model.add_parameters({(unsigned int)trg_vocab_size}, param_init_const));// uniform initialization
				//v_nbest_params[n].push_back(model.add_parameters({(unsigned int)trg_vocab_size})); // random initialization
			}
		}
	}
	else if (relopt_cf.intialization_type == RELOPT_INIT::REFERENCE_PROBABILITY){
		Sentence i_src_sent = ParseWords(*vocab_src_, src);

		int ibos = vocab_trg_->convert("<s>");
		int ieos = vocab_trg_->convert("</s>");

		//unsigned L_max = std::count(trg_nbest[0].begin(), trg_nbest[0].end(), ' ');// first has max length
		//if (trg_nbest[0].find("<s>") != std::string::npos) L_max--;
		//if (trg_nbest[0].find("</s>") != std::string::npos) L_max--;
		//cerr << L_max << endl;

		size_t n = 0;
		for (auto trg_ref : trg_nbest){
			Sentence i_trg_sent = ParseWords(*vocab_trg_, trg_ref);
			if (*(i_trg_sent.begin()) != ibos)
				i_trg_sent.insert(i_trg_sent.begin(), ibos);
			if (*(i_trg_sent.end()-1) != ieos)
				i_trg_sent.push_back(ieos);

			//cerr << trg_ref << endl;
			//cerr << "i_trg_sent.size()=" << i_trg_sent.size() << endl;

			// reset the computation graph
			dynet::ComputationGraph cg;// FIXME: the computation of encoder representation is repeated n times?
			std::vector<std::vector<float>> v_preds;
			if (relopt_cf.algorithm == RELOPT_ALGO::SOFTMAX
				|| relopt_cf.algorithm == RELOPT_ALGO::SPARSEMAX)
				am_->BuildGraph(i_src_sent, i_trg_sent, cg, v_preds, false);// v_preds are unnormalized predictions from the model
			else if (relopt_cf.algorithm == RELOPT_ALGO::EG || relopt_cf.algorithm == RELOPT_ALGO::AEG)
				am_->BuildGraph(i_src_sent, i_trg_sent, cg, v_preds);// v_preds are softmax-normalized predictions from the model

			unsigned ind = 0;
			for (auto& pred : v_preds){
				if (ind != v_preds.size() - 1){// excluding </s> prediction
					dynet::ParameterInitFromVector param_init_vector(pred);
					v_nbest_params[n].push_back(model.add_parameters({(unsigned int)trg_vocab_size}, param_init_vector));
				}

				ind++;
			}

			//cerr << "ind=" << ind << endl;

			// add </s> padding for batching
			//unsigned l = ind;
			//for (; ind < L_max + 1/*+1 for EOS from v_preds*/; ind++){
			//	dynet::ParameterInitFromVector param_init_vector(v_preds[l-1]);// </s> prediction vector
			//	v_nbest_params[n].push_back(model.add_parameters({(unsigned int)trg_vocab_size}, param_init_vector));
			//}

			//cerr << "v_nbest_params[n].size()=" << v_nbest_params[n].size() << endl;

			// add additional words, potentially helping the model expand the reference translations
			if (relopt_cf.add_extra_words > 0){
				// which one is better?
				//dynet::ParameterInitFromVector param_init_const(v_preds[ind - 1]);// initialization from </s> prediction
				dynet::ParameterInitFromVector param_init_const(v_preds[ind - 2]);// initialization from last word prediction
				//dynet::ParameterInitConst param_init_const(1.f / (float)trg_vocab_size);// uniform initialization
				for (unsigned i = 0; i < relopt_cf.add_extra_words; i++){
					v_nbest_params[n].push_back(model.add_parameters({(unsigned int)trg_vocab_size}, param_init_const));
				}
			}

			n++;
		}
	}
	else if (relopt_cf.intialization_type == RELOPT_INIT::REFERENCE_ONE_HOT){
		std::vector<std::string> ref_words = SplitWords(trg_nbest[0]);

		unsigned exl = 0;
		if (ref_words[0] == "<s>" && ref_words[ref_words.size() - 1] == "</s>") exl = 2;

		for (size_t n = 0; n < trg_nbest.size(); n++){
			for (size_t i = 0; i < ref_words.size() - exl/*excluding BOS and EOS*/; i++){
				std::vector<float> v_vals(trg_vocab_size, 0.f);
				v_vals[vocab_trg_->convert(ref_words[i])] = 1.f /*+ log(vocab_trg_->size())*/;
				ParameterInitFromVector param_init_vector(v_vals);
				v_nbest_params[n].push_back(model.add_parameters({(unsigned int)trg_vocab_size}, param_init_vector));
			}
		}

		// add additional words, potentially helping the model expand the reference translations
		if (relopt_cf.add_extra_words > 0){
			std::vector<float> v_vals(trg_vocab_size, 0.f);
			v_vals[vocab_trg_->convert("</s>")] = 1.f /*+ log(vocab_trg_->size())*/;
			ParameterInitFromVector param_init_const(v_vals);
			//dynet::ParameterInitConst param_init_const(1.f / (float)trg_vocab_size);// uniform initialization
			for (size_t n = 0; n < trg_nbest.size(); n++){
				for (unsigned i = 0; i < relopt_cf.add_extra_words; i++){
					v_nbest_params[n].push_back(model.add_parameters({(unsigned int)trg_vocab_size}, param_init_const));
				}
			}
		}
	}
	else assert("Unknown intialization!");
}

// (batched version)
template <class AM_t, class BAM_t, class RNNLM_t>
RelOptOutput RelOptDecoder<AM_t,BAM_t,RNNLM_t>::GDDecodeNBest_Batch(const string& src_sent
		, const std::vector<std::string>& trg_nbest
		, const RelOptConfig& relopt_cf)
{
	dynet::Model relopt_model;// local relaxed inference model

	// convert the source sentence
	Sentence i_src_sent = ParseWords(*vocab_src_, src_sent);

	//std::cerr << "GDDecodeNBest:1" << std::endl;
	std::vector<std::vector<dynet::Parameter>> v_relopt_params;// inference parameters live here!
	InitializeParameters(src_sent, trg_nbest
			, relopt_model
			, v_relopt_params
			, relopt_cf);
	size_t L = v_relopt_params[0].size();// length of the desired target output (exclusing BOS and EOS)
	//cerr << "L=" << L << endl;

	// create SGD trainer
	//std::cerr << "GDDecodeNBest:2" << std::endl;
	dynet::Trainer* trainer = nullptr;
	if (relopt_cf.algorithm == RELOPT_ALGO::SOFTMAX
		|| relopt_cf.algorithm == RELOPT_ALGO::SPARSEMAX){//SOFTMAX or SPARSEMAX
		if (relopt_cf.momentum != 0.f)
			trainer = new dynet::MomentumSGDTrainer(relopt_model, relopt_cf.eta, relopt_cf.momentum);// Momemtum SGD
		else
			trainer = new dynet::SimpleSGDTrainer(relopt_model, relopt_cf.eta);// Vanilla SGD
		//FIXME: to support others as well!
		//trainer = new dynet::AdamTrainer(relopt_model);// Adam SGD
		//trainer = new dynet::AdadeltaTrainer(relopt_model);// AdaDelta SGD
		//trainer = new dynet::AdagradTrainer(relopt_model);// Adagrad SGD
		//trainer = new dynet::RmsPropTrainer(relopt_model);// RmsProp SGD
	}
	else if (relopt_cf.algorithm == RELOPT_ALGO::EG){//EG
		trainer = new dynet::EGTrainer(relopt_model, relopt_cf.eta, relopt_cf.momentum, relopt_cf.noise_eta);// use our own implementation of EGTrainer
		if (relopt_cf.clr_lb_eta != 0 && relopt_cf.clr_ub_eta != 0)
			((dynet::EGTrainer*)trainer)->enableCyclicalLR(relopt_cf.clr_lb_eta, relopt_cf.clr_ub_eta, relopt_cf.clr_stepsize, relopt_cf.clr_gamma);// EG with cyclical learning rate
	}
	else if (relopt_cf.algorithm == RELOPT_ALGO::AEG){//Adaptive EG (AEG) with Adam or RMSProp
		if (relopt_cf.aeg_opt == 0)
			trainer = new dynet::AdamEGTrainer(relopt_model, relopt_cf.eta, relopt_cf.aeg_beta_1, relopt_cf.aeg_beta_2, relopt_cf.aeg_eps);// use our own implementation of EGTrainer
		else
			trainer = new dynet::RMSPropEGTrainer(relopt_model, relopt_cf.eta, relopt_cf.aeg_eps, relopt_cf.aeg_beta_1);// use our own implementation of EGTrainer
	}
	else
		assert("Unknown relaxed optimization algorithm!");
	trainer->eta_decay = relopt_cf.eta_decay;// learning rate decay

	//std::cerr << "GDDecodeNBest:3" << std::endl;	
	// perform the relaxed inference algo
	float best_fcost = std::numeric_limits<float>::max(), prev_fcost = best_fcost;
	unsigned t = 0, T = relopt_cf.max_iters, best_t = 0;
	std::string best_sent = "";
	while (t < T)
	{
		//cerr << "t=" << t << endl;

		//--------------------------------------------------------------------------------------------
		// FIXME: this code block can be moved outside the loop for efficiency.
		// FIXME: this step is repeated for every iteration due to cg?
		// (1) reset the computation graph
		dynet::ComputationGraph cg;

		// (2) pre-compute the source embedding representation
		//std::cerr << "GDDecodeNBest:3:(1)" << std::endl;
		if (bam_ != nullptr){
			bam_->s2t_model.ComputeTrgWordEmbeddingMatrix(cg);// source-to-target
			bam_->s2t_model.StartNewInstance(i_src_sent, cg, 0);
			bam_->t2s_model.ComputeSrcWordEmbeddingMatrix(cg);// target-to-source
		}
		else{
			am_->ComputeTrgWordEmbeddingMatrix(cg);// source-to-target/left-to-right model
			am_->StartNewInstance(i_src_sent, cg, 0);
		}
		if (relopt_cf.jdec_bidir_alpha > 0.f
			&& relopt_cf.jdec_bidir_alpha < 1.f){
			am_r2l_->ComputeTrgWordEmbeddingMatrix(cg);// right-to-left model
			am_r2l_->StartNewInstance(i_src_sent, cg, 0);
		}
		if (relopt_cf.jdec_mlm_alpha > 0.f
			&& relopt_cf.jdec_mlm_alpha < 1.f){
			rnnlm_->ComputeWordEmbeddingMatrix(cg);// monolingual RNN language model
		}
		if (relopt_cf.jdec_biling_alpha > 0.f
			&& relopt_cf.jdec_biling_alpha < 1.f){
			am_t2s_->ComputeSrcWordEmbeddingMatrix(cg);// target-to-source model
		}

		//cg.checkpoint();// checkpoint here refers to computation for encoder(s)
		//--------------------------------------------------------------------------------------------

		//cg.revert();// revert to last checkpoint

		// (3) build relaxed optimization graph
		//std::cerr << "GDDecodeNBest:3:(2a)" << std::endl;
		// left-to-right/source-to-target AM model
		dynet::expr::Expression i_alignment, i_coverage, i_glofer_nll, i_entropy;
		dynet::expr::Expression i_cost;
		if (bam_ != nullptr)
			i_cost = (1.f/(float)(L + 1)) * bam_->s2t_model.BuildRelOptGraph(
					relopt_cf.algorithm
					, v_relopt_params
					, cg
					, *vocab_trg_
					, false //left-to-right model (default)
					, &i_entropy
					, &i_alignment
					, (relopt_cf.coverage_weight > 0.f) ? &i_coverage : nullptr, relopt_cf.coverage_C
					, (relopt_cf.glofer_weight > 0.f) ? &i_glofer_nll : nullptr);// normalized sum of negative log likelihoods (NLL)
		else
			i_cost =  (1.f/(float)(L + 1)) * am_->BuildRelOptGraph(
					relopt_cf.algorithm
					, v_relopt_params
					, cg
					, *vocab_trg_
					, false //left-to-right model (default)
					, &i_entropy
					, &i_alignment
					, (relopt_cf.coverage_weight > 0.f) ? &i_coverage : nullptr, relopt_cf.coverage_C
					, (relopt_cf.glofer_weight > 0.f) ? &i_glofer_nll : nullptr);// normalized sum of negative log likelihoods (NLL)
		//std::cerr << "GDDecodeNBest:3:(2b)" << std::endl;
		// right-to-left AM model
		dynet::expr::Expression i_cost_r2l;
		if (relopt_cf.jdec_bidir_alpha > 0.f
			&& relopt_cf.jdec_bidir_alpha < 1.f){
			i_cost_r2l = (1.f/(float)(L + 1)) * am_r2l_->BuildRelOptGraph(
					relopt_cf.algorithm
					, v_relopt_params /*target*/
					, cg
					, *vocab_trg_
					, true //right-to-left model
					, nullptr
					, nullptr
					, nullptr, 0
					, nullptr);// normalized sum of negative log likelihoods (NLL)
		}
		//std::cerr << "GDDecodeNBest:3:(2c)" << std::endl;
		// target-to-source AM model
		dynet::expr::Expression i_cost_t2s;
		dynet::expr::Expression i_alignment_t2s;
		if (bam_ != nullptr){			
			i_cost_t2s = (1.f/(float)(i_src_sent.size() - 1)) * bam_->t2s_model.BuildRevRelOptGraph(
					relopt_cf.algorithm
					, v_relopt_params /*source*/
					, i_src_sent /*target*/
					, cg
					, *vocab_src_
					, &i_alignment_t2s);// normalized sum of negative log likelihoods (NLL)
		}
		else{			
			if (relopt_cf.jdec_biling_alpha > 0.f && relopt_cf.jdec_biling_alpha < 1.f){
				i_cost_t2s = (1.f/(float)(i_src_sent.size() - 1)) * am_t2s_->BuildRevRelOptGraph(
					relopt_cf.algorithm
					, v_relopt_params /*source*/
					, i_src_sent /*target*/
					, cg
					, *vocab_src_
					, &i_alignment_t2s);// normalized sum of negative log likelihoods (NLL)
			}
		}
		//std::cerr << "GDDecodeNBest:3:(2d)" << std::endl;
		dynet::expr::Expression i_cost_mlm;
		if (relopt_cf.jdec_mlm_alpha > 0.f
			&& relopt_cf.jdec_mlm_alpha < 1.f){
			// FIXME: to support batching?
			//i_cost_mlm = (1.f/(float)K) * (1.f/(float)(L + 1)) * rnnlm_->BuildRelOptGraph( relopt_cf.algorithm
			//		, v_relopt_params
			//		, cg
			//		, *vocab_trg_);// left-to-right or right-to-left direction will be implicitly recognized in model file.
		}

		// (4) compute the additional costs if required
		//std::cerr << "GDDecodeNBest:3:(3)" << std::endl;
		dynet::expr::Expression i_objective = relopt_cf.m_weight * i_cost;// NLL
		// coverage penalty
		if (relopt_cf.coverage_weight > 0.f){
			i_objective = i_objective + (1.f/(float)(L + 1)) * relopt_cf.coverage_weight * i_coverage;// FIXME: normalization for coverage is required!
		}
		// global fertility
		if (relopt_cf.glofer_weight > 0.f){
			i_objective = i_objective + (1.f/(float)(L + 1)) * relopt_cf.glofer_weight * i_glofer_nll;// FIXME: normalization for glofer is required!
		}
		// joint decoding in bidirectional models
		if (relopt_cf.jdec_bidir_alpha > 0.f
			&& relopt_cf.jdec_bidir_alpha < 1.f){
			i_objective = i_objective + relopt_cf.jdec_bidir_alpha * i_cost_r2l;
		}
		// joint decoding in bilingual models
		if ((relopt_cf.jdec_biling_alpha > 0.f
			&& relopt_cf.jdec_biling_alpha < 1.f) || bam_ != nullptr){
			i_objective = i_objective + relopt_cf.jdec_biling_alpha * i_cost_t2s;
			if (relopt_cf.jdec_biling_trace_alpha != 0.f){
				// if required, add a trace bonus of i_alignment and i_alignment_t2s following Cohn et al., 2016.
				dynet::expr::Expression i_trace_bonus = trace_of_product(i_alignment, transpose(i_alignment_t2s));// FIXME: this trace_of_product is not supporting CUDA yet!
				i_objective = i_objective - relopt_cf.jdec_biling_trace_alpha * i_trace_bonus;
			}
		}
		// joint decoding with monolingual RNN language model
		if (relopt_cf.jdec_mlm_alpha > 0.f
			&& relopt_cf.jdec_mlm_alpha < 1.f){
			i_objective = i_objective + relopt_cf.jdec_mlm_alpha * i_cost_mlm;
		}
		// entropy regularizer for SOFTMAX or SPARSEMAX
		//if (relopt_cf.algorithm == RELOPT_ALGO::SOFTMAX
		//	|| relopt_cf.algorithm == RELOPT_ALGO::SPARSEMAX)
		//	i_objective = i_objective - relopt_cf.ent_gamma * (1.f/(float)L) * i_entropy;// FIXME: normalize the entropy sum according to sentence length?	

		// (5) do forward propagation step
		//std::cerr << "GDDecodeNBest:3:(4a)" << std::endl;
		cg.incremental_forward(i_objective);

		// grap the parts of the objective
		//std::cerr << "GDDecodeNBest:3:(4b)" << std::endl;
		float fcost = as_scalar(cg.get_value(i_cost.i));
		float fcoverage = 0.f;
		if (relopt_cf.coverage_weight > 0.f) {
			fcoverage = as_scalar(cg.get_value(i_coverage.i));
		}
		float fglofer = 0.f;
		if (relopt_cf.glofer_weight > 0.f){
			fglofer = as_scalar(cg.get_value(i_glofer_nll.i));
		}
		float fobj = as_scalar(cg.get_value(i_objective.i));
		float fcost_r2l = 0.f;
		if (relopt_cf.jdec_bidir_alpha > 0.f
			&& relopt_cf.jdec_bidir_alpha < 1.f)
			fcost_r2l = as_scalar(cg.get_value(i_cost_r2l.i));
		float fcost_t2s = 0.f;
		if (relopt_cf.jdec_biling_alpha > 0.f
			&& relopt_cf.jdec_biling_alpha < 1.f)
			fcost_t2s = as_scalar(cg.get_value(i_cost_t2s.i));
		float fcost_mlm = 0.f;
		if (relopt_cf.jdec_mlm_alpha > 0.f
			&& relopt_cf.jdec_mlm_alpha < 1.f)
			fcost_mlm = as_scalar(cg.get_value(i_cost_mlm.i));

		// (6) do backpropagation step (including computation of gradients which are stored in relopt_model)
		//std::cerr << "GDDecodeNBest:3:(5)" << std::endl;
		cg.backward(i_objective);// backpropagate for all nodes

		// (7) update inference parameters with requested optimization method (e.g., SGD, EG)
		//std::cerr << "GDDecodeNBest:3:(6)" << std::endl;
		float scale = 1.f;// FIXME: can be used if required!
		float clr = relopt_cf.eta / std::pow(1.f + t * relopt_cf.eta_decay, relopt_cf.eta_power);// simple learning rate annealing
		trainer->eta = clr;
		if (relopt_cf.algorithm == RELOPT_ALGO::EG){//EG
			((dynet::EGTrainer*)trainer)->update_noise_eta(relopt_cf.noise_eta / std::pow(1.f + t, relopt_cf.noise_gamma));// noise level annealing according to https://arxiv.org/pdf/1511.06807.pdf.
		}
		trainer->update(scale);// only update inference parameters

		// (8) verbose output
		//std::cerr << "GDDecodeNBest:3:(7)" << std::endl;
		if (verbose_) cerr << "All costs at step "
			<< t << " (eta=" << trainer->eta << ")" << ": "
			<< "l2r_nll=" << fcost
			<< "; r2l_nll=" << fcost_r2l
			<< "; t2s_nll=" << fcost_t2s
			<< "; mlm_nll=" << fcost_mlm
			<< "; C_coverage=" << fcoverage
			<< "; glofer_fertility=" << fglofer
			<< "; total_objective=" << fobj << endl;
		//string decoded_sent = am_->GetRelOptOutput(cg, v_relopt_params, relopt_cf.algorithm, *vocab_trg_, verbose_);// FIXME: how to get the output from v_relopt_params?
		//if (verbose_) cerr << "Result at step " << t << ": " << decoded_sent << " (discrete cost=" << GetNLLCost(src_sent, decoded_sent, cg) << ")" << endl;

		// (9) update the best result so far
		//std::cerr << "GDDecodeNBest:3:(8)" << std::endl;
		if (fcost < best_fcost){// FIXME: fobj or fcost?
			best_fcost = fcost;
			best_sent = am_->GetRelOptOutput(0, cg, i_src_sent, v_relopt_params, relopt_cf.algorithm, *vocab_trg_, verbose_);	
			best_t = t;
		}

		// simple stopping criterion if change in two consecutive costs is very small!
		if (t >= 1 && std::abs(prev_fcost - fcost) < 0.0001f){// FIXME: maybe smaller???
			//get the output
			//best_sent = am_->GetRelOptOutput(0, cg, i_src_sent, v_relopt_params, relopt_cf.algorithm, *vocab_trg_, verbose_);	
			break;
		}

		prev_fcost = fcost;// update previous cost

		t++;// next iteration

		//if (t == T){
			//get the output
		//	best_sent = am_->GetRelOptOutput(0, cg, i_src_sent, v_relopt_params, relopt_cf.algorithm, *vocab_trg_, verbose_);
		//}
	}

	if (verbose_) cerr << "***Best decoding result at step " << best_t << " (continuous cost=" << best_fcost << ")" << endl;

	delete trainer;

	//cerr << "GDDecodeNBest:4" << endl;
	return RelOptOutput(best_sent, best_fcost, best_t);
}

// (non-batched version)
template <class AM_t, class BAM_t, class RNNLM_t>
RelOptOutput RelOptDecoder<AM_t,BAM_t,RNNLM_t>::GDDecodeNBest(const string& src_sent
		, const std::vector<std::string>& trg_nbest
		, const RelOptConfig& relopt_cf)
{
	dynet::Model relopt_model;// local relaxed inference model

	// convert the source sentence
	Sentence i_src_sent = ParseWords(*vocab_src_, src_sent);

	//std::cerr << "GDDecodeNBest:1" << std::endl;
	std::vector<std::vector<dynet::Parameter>> v_relopt_params;// inference parameters live here!
	InitializeParameters(src_sent, trg_nbest
			, relopt_model
			, v_relopt_params
			, relopt_cf);
	//size_t L = v_relopt_params[0].size();// length of the desired target output (exclusing BOS and EOS)
	//cerr << "L=" << L << endl;
	
	unsigned K = v_relopt_params.size();

	// create SGD trainer
	//std::cerr << "GDDecodeNBest:2" << std::endl;
	dynet::Trainer* trainer = nullptr;
	if (relopt_cf.algorithm == RELOPT_ALGO::SOFTMAX
		|| relopt_cf.algorithm == RELOPT_ALGO::SPARSEMAX){//SOFTMAX or SPARSEMAX
		if (relopt_cf.momentum != 0.f)
			trainer = new dynet::MomentumSGDTrainer(relopt_model, relopt_cf.eta, relopt_cf.momentum);// Momemtum SGD
		else
			trainer = new dynet::SimpleSGDTrainer(relopt_model, relopt_cf.eta);// Vanilla SGD
		//FIXME: to support others as well!
		//trainer = new dynet::AdamTrainer(relopt_model);// Adam SGD
		//trainer = new dynet::AdadeltaTrainer(relopt_model);// AdaDelta SGD
		//trainer = new dynet::AdagradTrainer(relopt_model);// Adagrad SGD
		//trainer = new dynet::RmsPropTrainer(relopt_model);// RmsProp SGD
	}
	else if (relopt_cf.algorithm == RELOPT_ALGO::EG){//EG
		trainer = new dynet::EGTrainer(relopt_model, relopt_cf.eta, relopt_cf.momentum, relopt_cf.noise_eta);// use our own implementation of EGTrainer
		if (relopt_cf.clr_lb_eta != 0 && relopt_cf.clr_ub_eta != 0)
			((dynet::EGTrainer*)trainer)->enableCyclicalLR(relopt_cf.clr_lb_eta, relopt_cf.clr_ub_eta, relopt_cf.clr_stepsize, relopt_cf.clr_gamma);// EG with cyclical learning rate
	}
	else if (relopt_cf.algorithm == RELOPT_ALGO::AEG){//Adaptive EG (AEG) with Adam or RMSProp
		if (relopt_cf.aeg_opt == 0)
			trainer = new dynet::AdamEGTrainer(relopt_model, relopt_cf.eta, relopt_cf.aeg_beta_1, relopt_cf.aeg_beta_2, relopt_cf.aeg_eps);// use our own implementation of EGTrainer
		else
			trainer = new dynet::RMSPropEGTrainer(relopt_model, relopt_cf.eta, relopt_cf.aeg_eps, relopt_cf.aeg_beta_1);// use our own implementation of EGTrainer
	}
	else
		assert("Unknown relaxed optimization algorithm!");
	trainer->eta_decay = relopt_cf.eta_decay;// learning rate decay

	//std::cerr << "GDDecodeNBest:3" << std::endl;	
	// perform the relaxed inference algo
	float best_fobj = std::numeric_limits<float>::max(), prev_fobj = best_fobj;
	unsigned t = 0, T = relopt_cf.max_iters, best_t = 0;
	std::string best_sent = "";
	std::vector<Expression> v_costs;
	v_costs.resize(K);
	while (t < T)
	{
		//cerr << "t=" << t << endl;

		//--------------------------------------------------------------------------------------------
		// FIXME: this code block can be moved outside the loop for efficiency.
		// FIXME: this step is repeated for every iteration due to cg?
		// (1) reset the computation graph
		dynet::ComputationGraph cg;

		// (2) pre-compute the source embedding representation
		//std::cerr << "GDDecodeNBest:3:(1)" << std::endl;
		if (bam_ != nullptr){
			bam_->s2t_model.ComputeTrgWordEmbeddingMatrix(cg);// source-to-target
			bam_->s2t_model.StartNewInstance(i_src_sent, cg, 0);
			bam_->t2s_model.ComputeSrcWordEmbeddingMatrix(cg);// target-to-source
		}
		else{
			am_->ComputeTrgWordEmbeddingMatrix(cg);// source-to-target/left-to-right model
			am_->StartNewInstance(i_src_sent, cg, 0);
		}
		if (relopt_cf.jdec_bidir_alpha > 0.f
			&& relopt_cf.jdec_bidir_alpha < 1.f){
			am_r2l_->ComputeTrgWordEmbeddingMatrix(cg);// right-to-left model
			am_r2l_->StartNewInstance(i_src_sent, cg, 0);
		}
		if (relopt_cf.jdec_mlm_alpha > 0.f
			&& relopt_cf.jdec_mlm_alpha < 1.f){
			rnnlm_->ComputeWordEmbeddingMatrix(cg);// monolingual RNN language model
		}
		if (relopt_cf.jdec_biling_alpha > 0.f
			&& relopt_cf.jdec_biling_alpha < 1.f){
			am_t2s_->ComputeSrcWordEmbeddingMatrix(cg);// target-to-source model
		}

		//cg.checkpoint();// checkpoint here refers to computation for encoder(s)
		//--------------------------------------------------------------------------------------------

		//cg.revert();// revert to last checkpoint
		
		for (unsigned ip = 0; ip < K; ip++){
			auto rlp = v_relopt_params[ip];

			// (3) build relaxed optimization graph
			//std::cerr << "GDDecodeNBest:3:(2a)" << std::endl;
			// left-to-right/source-to-target AM model
			dynet::expr::Expression i_alignment, i_coverage, i_glofer_nll, i_entropy;
		        v_costs[ip] =  /*(1.f/(float)(v_relopt_params[ip].size() + 1)) **/ am_->BuildRelOptGraph(
						relopt_cf.algorithm
						, rlp
						, cg
						, *vocab_trg_
						, false //left-to-right model (default)
						, &i_entropy
						, &i_alignment
						, (relopt_cf.coverage_weight > 0.f) ? &i_coverage : nullptr, relopt_cf.coverage_C
						, (relopt_cf.glofer_weight > 0.f) ? &i_glofer_nll : nullptr);// normalized sum of negative log likelihoods (NLL)
		}
	
		// (4) compute the additional costs if required
		//std::cerr << "GDDecodeNBest:3:(3)" << std::endl;
		dynet::expr::Expression i_objective = sum(v_costs);// NLL	

		// (5) do forward propagation step
		//std::cerr << "GDDecodeNBest:3:(4a)" << std::endl;
		cg.incremental_forward(i_objective);

		// grap the parts of the objective
		//std::cerr << "GDDecodeNBest:3:(4b)" << std::endl;
		float fobj = as_scalar(cg.get_value(i_objective.i));
		
		// (6) do backpropagation step (including computation of gradients which are stored in relopt_model)
		//std::cerr << "GDDecodeNBest:3:(5)" << std::endl;
		cg.backward(i_objective);// backpropagate for all nodes

		// (7) update inference parameters with requested optimization method (e.g., SGD, EG)
		//std::cerr << "GDDecodeNBest:3:(6)" << std::endl;
		float scale = 1.f;// FIXME: can be used if required!
		float clr = relopt_cf.eta / std::pow(1.f + t * relopt_cf.eta_decay, relopt_cf.eta_power);// simple learning rate annealing
		trainer->eta = clr;
		if (relopt_cf.algorithm == RELOPT_ALGO::EG){//EG
			((dynet::EGTrainer*)trainer)->update_noise_eta(relopt_cf.noise_eta / std::pow(1.f + t, relopt_cf.noise_gamma));// noise level annealing according to https://arxiv.org/pdf/1511.06807.pdf.
		}
		trainer->update(scale);// only update inference parameters

		unsigned k = 0;
		for (auto& i_cost : v_costs){
			float fcost = as_scalar(cg.get_value(i_cost.i));

			cerr << "cost_" << k << "=" << fcost << endl;
	
			k++;
		}

		// (8) verbose output
		//std::cerr << "GDDecodeNBest:3:(7)" << std::endl;
		if (verbose_) cerr << "All costs at step "
			<< t << " (eta=" << trainer->eta << ")" << ": "
			<< "; total_objective=" << fobj << endl;
		//string decoded_sent = am_->GetRelOptOutput(cg, v_relopt_params, relopt_cf.algorithm, *vocab_trg_, verbose_);// FIXME: how to get the output from v_relopt_params?
		//if (verbose_) cerr << "Result at step " << t << ": " << decoded_sent << " (discrete cost=" << GetNLLCost(src_sent, decoded_sent, cg) << ")" << endl;

		// (9) update the best result so far
		//std::cerr << "GDDecodeNBest:3:(8)" << std::endl;
		if (fobj < best_fobj){
			best_fobj = fobj;
			float best_fcost = std::numeric_limits<float>::max();
			unsigned k = 0, best_k = k;
			for (auto& i_cost : v_costs){
				float fcost = as_scalar(cg.get_value(i_cost.i));
				if (fcost < best_fcost){
					best_fcost = fcost;
					best_k = k;
				}
				
				k++;
			}
			
			best_sent = am_->GetRelOptOutput(cg, v_relopt_params[best_k], relopt_cf.algorithm, *vocab_trg_, verbose_);
			best_t = t;
		}

		// simple stopping criterion if change in two consecutive costs is very small!
		if (t >= 1 && std::abs(prev_fobj - fobj) < 0.0001f)// FIXME: maybe smaller???	
			break;

		prev_fobj = fobj;// update previous cost

		t++;// next iteration
	}

	if (verbose_) cerr << "***Best decoding result at step " << best_t << " (continuous cost=" << best_fobj << ")" << endl;

	delete trainer;

	//cerr << "GDDecodeNBest:4" << endl;
	return RelOptOutput(best_sent, best_fobj, best_t);
}

//*************************************************************************************************************************************************
// Sentence decoding with relaxed optimization algorithms (initialization: uniform/greedy/beam)
// Enhanced with Min-Max algorithm (for parsing task)
template <class AM_t, class BAM_t, class RNNLM_t>
RelOptOutput RelOptDecoder<AM_t,BAM_t,RNNLM_t>::GDDecode_MinMax(const string& src_sent
		, const std::string& trg_ref
		, const RelOptConfig& relopt_cf)
{
	// FIXME	
	return RelOptOutput("", 0.f, 0.f);
}


} /* namespace dynet */

#endif /* RELOPT_DECODER_H_ */
