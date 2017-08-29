/*
 * This is an implementation of the following work:
 * Dual Learning for Machine Translation
 * Yingce Xia, Di He, Tao Qin, Liwei Wang, Nenghai Yu, Tie-Yan Liu, Wei-Ying Ma
 * https://arxiv.org/abs/1611.00179 (accepted at NIPS 2016)
 * Developed by Cong Duy Vu Hoang (vhoang2@student.unimelb.edu.au)
 * Date: 21 May 2017
 *
*/

#include "attentional.h" // AM
#include "rnnlm.h" // RNNLM
#include "ensemble-decoder.h"

#include "dict-utils.h"

#include <iostream>
#include <fstream>
#include <sstream>

#include <boost/program_options/parsers.hpp>
#include <boost/program_options/variables_map.hpp>
#include <boost/algorithm/string.hpp>

using namespace std;
using namespace dynet;
using namespace boost::program_options;

dynet::Dict sdict; //global source vocab
dynet::Dict tdict; //global target vocab

unsigned MAX_EPOCH = 10;
unsigned DEV_ROUND = 25000;

bool VERBOSE;

typedef tuple<Sentence, Sentence> SentencePair; 
typedef vector<SentencePair> ParaCorpus;
typedef vector<Sentence> MonoCorpus;

int main_body(variables_map vm);

void InitialiseModel(ParameterCollection &model, const string &filename);
ParaCorpus Read_ParaCorpus(const string &filename);
MonoCorpus Read_MonoCorpus(const string &filename, Dict& d);

template<class AM_t, class LM_t>
void Dual_Learn(ParameterCollection& mod_am_s2t, AM_t& p_am_s2t
		, Model& mod_am_t2s, AM_t& p_am_t2s
		, Model& mod_mlm_s, LM_t& p_mlm_s
		, Model& mod_mlm_t, LM_t& p_mlm_t
		, const MonoCorpus& mono_cor_s, const MonoCorpus& mono_cor_t, const ParaCorpus& dev_cor
		, unsigned K, unsigned beam_size, float alpha, float gamma_1, float gamma_2
		, unsigned opt_type
		, const string& modfile_am_s2t, const string& modfile_am_t2s);

int main(int argc, char** argv) {
	dynet::initialize(argc, argv);

	// command line processing
	variables_map vm; 
	options_description opts("Allowed options");
	opts.add_options()
		("help", "print help message")
		("config,c", value<string>(), "config file specifying additional command line options")
		//-----------------------------------------		
		("train,t", value<string>(), "file containing training parallel sentences (for source/targer vocabulary building on-the-fly), with "
			"each line consisting of source ||| target.")
		("dev,d", value<string>(), "file containing development parallel sentences, with "
			"each line consisting of source ||| target.")
		("initialise_am_s2t", value<string>(), "load pre-trained model parameters (source-to-target AM model) from file")
		("initialise_am_t2s", value<string>(), "load pre-trained model parameters (target-to-source AM model) from file")
		("initialise_rnnlm_s", value<string>(), "load pre-trained model parameters (source RNNLM model) from file")
		("initialise_rnnlm_t", value<string>(), "load pre-trained model parameters (target RNNLM model) from file")
		//-----------------------------------------
		("save_am_s2t", value<string>(), "save learned model parameters (source-to-target AM model) to file")
		("saves_am_t2s", value<string>(), "save learned model parameters (target-to-source AM model) to file")
		//-----------------------------------------
		("sgd_trainer", value<unsigned>()->default_value(0), "use specific SGD trainer (0: vanilla SGD; 1: momentum SGD; 2: Adagrad; 3: AdaDelta; 4: Adam; 5: RMSProp)")
		//-----------------------------------------
		("model_conf", value<string>(), "config file specifying model configurations for all models (line by line)")
		("K", value<unsigned>()->default_value(10), "the K value for sampling K-best translations from AM models")
		("beam_size", value<unsigned>()->default_value(5), "the beam size of beam search decoding")
		("alpha", value<float>()->default_value(0.5f), "the alpha hyper-parameter for balancing the rewards")
		("gamma_1", value<float>()->default_value(0.1f), "the gamma 1 hyper-parameter for stochastic gradient update in tuning source-to-target AM model")
		("gamma_2", value<float>()->default_value(0.1f), "the gamma 2 hyper-parameter for stochastic gradient update in tuning target-to-source AM model")
		//-----------------------------------------
		("mono_s", value<string>()->default_value(""), "File to read the monolingual source from")
		("mono_t", value<string>()->default_value(""), "File to read the monolingual target from")
		//-----------------------------------------
		("epoch,e", value<unsigned>()->default_value(10), "number of training epochs, 10 by default")
		("dev_round", value<unsigned>()->default_value(25000), "number of rounds for evaluating over development data, 25000 by default")
		//-----------------------------------------
		("verbose,v", "be extremely chatty")
	;
	store(parse_command_line(argc, argv, opts), vm); 
	if (vm.count("config") > 0)
	{
		ifstream config(vm["config"].as<string>().c_str());
		store(parse_config_file(config, opts), vm); 
	}
	notify(vm);
	
	if (vm.count("help")) {// FIXME: check the missing ones?
		cout << opts << "\n";
		return EXIT_SUCCESS;
	}

	VERBOSE = vm.count("verbose");

	// hyper-parameters
	unsigned K = vm["K"].as<unsigned>();
	unsigned beam_size = vm["beam_size"].as<unsigned>();
	float alpha = vm["alpha"].as<unsigned>();
	float gamma_1 = vm["gamma_1"].as<unsigned>();
	float gamma_2 = vm["gamma_2"].as<unsigned>();
	MAX_EPOCH = vm["epoch"].as<unsigned>();
	DEV_ROUND = vm["dev_round"].as<unsigned>();
	
	//--- load data
	// parallel corpus for building vocabularies on-the-fly
	ParaCorpus train_cor, dev_cor;
	cerr << "Reading training parallel data from " << vm["train"].as<string>() << "...\n";
	kSRC_SOS = sdict.convert("<s>");
	kSRC_EOS = sdict.convert("</s>");
	kTGT_SOS = tdict.convert("<s>");
	kTGT_EOS = tdict.convert("</s>");
	train_cor = Read_ParaCorpus(vm["train"].as<string>());
	kSRC_UNK = sdict.convert("<unk>");// add <unk> if does not exist!
	kTGT_UNK = tdict.convert("<unk>");
	sdict.freeze(); // no new word types allowed
	tdict.freeze(); // no new word types allowed	
	SRC_VOCAB_SIZE = sdict.size();
	TGT_VOCAB_SIZE = tdict.size();

	dev_cor = Read_ParaCorpus(vm["dev"].as<string>());

	// monolingual corpora
	// Assume that these monolingual corpora use the same vocabularies with parallel corpus
	// FIXME: otherwise?
	MonoCorpus mono_cor_s, mono_cor_t;
	cerr << "Reading monolingual source data from " << vm["mono_s"].as<string>() << "...\n";
	mono_cor_s = Read_MonoCorpus(vm["mono_s"].as<string>(), sdict);
	cerr << "Reading monolingual target data from " << vm["mono_t"].as<string>() << "...\n";
	mono_cor_t = Read_MonoCorpus(vm["mono_t"].as<string>(), tdict);
	
	//--- load models
	Model model_am_s2t, model_am_t2s, model_mlm_s, model_mlm_t;
	std::shared_ptr<AttentionalModel<LSTMBuilder>> p_am_s2t, p_am_t2s;// FIXME: assume all models use the same LSTM RNN structure (for quick implementation)
	std::shared_ptr<RNNLanguageModel<LSTMBuilder>> p_mlm_s, p_mlm_t;// FIXME: how to support n-gram LM model (e.g., KenLM?) as well???

	string line;
	stringstream ss;
	ifstream inpf_conf(vm["model_conf"].as<string>());
	assert(inpf_conf);
	
	// s2t AM model
	getline(inpf_conf, line);
	ss.str(line);
	ss >> SLAYERS >> TLAYERS >> HIDDEN_DIM >> ALIGN_DIM;
	p_am_s2t.reset(new AttentionalModel<LSTMBuilder>(&model_am_s2t, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE,
		SLAYERS, TLAYERS, HIDDEN_DIM, ALIGN_DIM, true, false, false, false, false, false));// standard model for now
	InitialiseModel(model_am_s2t, vm["initialise_am_s2t"].as<string>());

	// t2s AM model
	getline(inpf_conf, line);
	ss.clear();
	ss.str(line);
	ss >> SLAYERS >> TLAYERS >> HIDDEN_DIM >> ALIGN_DIM;
	p_am_t2s.reset(new AttentionalModel<LSTMBuilder>(&model_am_t2s, TGT_VOCAB_SIZE, SRC_VOCAB_SIZE,
		SLAYERS, TLAYERS, HIDDEN_DIM, ALIGN_DIM, true, false, false, false, false, false));// standard model for now
	InitialiseModel(model_am_t2s, vm["initialise_am_t2s"].as<string>());

	// s RNNLM model
	getline(inpf_conf, line);
	ss.clear();
	ss.str(line);
	ss >> SLAYERS >> HIDDEN_DIM;
	p_mlm_s.reset(new RNNLanguageModel<LSTMBuilder>(&model_mlm_s, SLAYERS, HIDDEN_DIM, HIDDEN_DIM, SRC_VOCAB_SIZE, false));
	InitialiseModel(model_mlm_s, vm["initialise_rnnlm_s"].as<string>());
	
	// t RNNLM model
	getline(inpf_conf, line);
	ss.clear();
	ss.str(line);
	ss >> TLAYERS >> HIDDEN_DIM;
	p_mlm_t.reset(new RNNLanguageModel<LSTMBuilder>(&model_mlm_t, TLAYERS, HIDDEN_DIM, HIDDEN_DIM, TGT_VOCAB_SIZE, false));
	InitialiseModel(model_mlm_t, vm["initialise_rnnlm_t"].as<string>());

	//--- execute dual-learning
	Dual_Learn<AttentionalModel<LSTMBuilder>, RNNLanguageModel<LSTMBuilder>>(model_am_s2t, *p_am_s2t
			, model_am_t2s, *p_am_t2s
			, model_mlm_s, *p_mlm_s
			, model_mlm_t, *p_mlm_t
			, mono_cor_s, mono_cor_t, dev_cor
			, K, beam_size, alpha, gamma_1, gamma_2
			, vm["sgd_trainer"].as<unsigned>()
			, vm["save_am_s2t"].as<string>(), vm["save_am_t2s"].as<string>());
	
	// complete
	return EXIT_SUCCESS;
}

template<class AM_t, class LM_t>
void Dual_Learn(ParameterCollection& mod_am_s2t, AM_t& am_s2t
		, Model& mod_am_t2s, AM_t& am_t2s
		, Model& mod_mlm_s, LM_t& mlm_s
		, Model& mod_mlm_t, LM_t& mlm_t
		, const MonoCorpus& mono_cor_s, const MonoCorpus& mono_cor_t, const ParaCorpus& dev_cor
		, unsigned K, unsigned beam_size, float alpha, float gamma_1, float gamma_2
		, unsigned opt_type
		, const string& modfile_am_s2t, const string& modfile_am_t2s)
{
	// set up the decoder(s)
	EnsembleDecoder<AM_t> edec_s2t(std::vector<std::shared_ptr<AM_t>>({std::make_shared<AM_t>(am_s2t)}), &tdict);
	edec_s2t.SetBeamSize(beam_size);
	EnsembleDecoder<AM_t> edec_t2s(std::vector<std::shared_ptr<AM_t>>({std::make_shared<AM_t>(am_t2s)}), &sdict);
	edec_t2s.SetBeamSize(beam_size);

	// set up monolingual data
	vector<unsigned> orders_s(mono_cor_s.size());// IDs from mono_cor_s
	vector<unsigned> orders_t(mono_cor_t.size());// IDs from mono_cor_t
	shuffle(orders_s.begin(), orders_s.end(), *rndeng);// to make it random
	shuffle(orders_t.begin(), orders_t.end(), *rndeng);
	
	// set up optimizers
	std::shared_ptr<Trainer> p_sgd_s2t(nullptr), p_sgd_t2s(nullptr);
	if (opt_type == 1){
		p_sgd_s2t.reset(new MomentumSGDTrainer(mod_am_s2t, gamma_1));
		p_sgd_t2s.reset(new MomentumSGDTrainer(mod_am_t2s, gamma_2));
	}
	else if (opt_type == 2){
		p_sgd_s2t.reset(new AdagradTrainer(mod_am_s2t, gamma_1));
		p_sgd_t2s.reset(new AdagradTrainer(mod_am_t2s, gamma_2));
	}
	else if (opt_type == 3){
		p_sgd_s2t.reset(new AdadeltaTrainer(mod_am_s2t, gamma_1));
		p_sgd_t2s.reset(new AdadeltaTrainer(mod_am_t2s, gamma_2));
	}
	else if (opt_type == 4){
		p_sgd_s2t.reset(new AdamTrainer(mod_am_s2t, gamma_1));
		p_sgd_t2s.reset(new AdamTrainer(mod_am_t2s, gamma_2));
	}
	else if (opt_type == 5){
		p_sgd_s2t.reset(new RMSPropTrainer(mod_am_s2t, gamma_1));
		p_sgd_t2s.reset(new RMSPropTrainer(mod_am_t2s, gamma_2));
	}
	else if (opt_type == 0){//Vanilla SGD trainer
		p_sgd_s2t.reset(new SimpleSGDTrainer(mod_am_s2t, gamma_1));
		p_sgd_t2s.reset(new SimpleSGDTrainer(mod_am_t2s, gamma_2));
	}
	else
	   	assert("Unknown SGD trainer type! (0: vanilla SGD; 1: momentum SGD; 2: Adagrad; 3: AdaDelta; 4: Adam; 5: RMSProp)");

	// pointers for switching between the models
	EnsembleDecoder<AM_t> *p_edec = nullptr;
	AM_t *p_am_s2t = nullptr, *p_am_t2s = nullptr;
	LM_t* p_mlm = nullptr;
	
	// start the dual learning algorithm
	ModelStats stats;// normally unused
	double best_loss_s2t = 9e+99, best_loss_t2s = 9e+99;
	unsigned long id_s = 0, id_t = 0;
	unsigned r = 0/*round*/, epoch_s2t = 0, epoch_t2s = 0;
	bool flag = true;// role of source and target
	while (epoch_s2t < MAX_EPOCH 
		|| epoch_t2s < MAX_EPOCH)// FIXME: simple stopping criterion, another?
	{		
		ComputationGraph cg;

		if (id_s == orders_s.size()){
			shuffle(orders_s.begin(), orders_s.end(), *rndeng);// to make it random
			id_s = 0;// reset id
			epoch_s2t++;// FIXME: adjust the learning rate if required?
		}
		if (id_t == orders_t.size()){
			shuffle(orders_t.begin(), orders_t.end(), *rndeng);// to make it random
			id_t = 0;// reset id
			epoch_t2s++;// FIXME: adjust the learning rate if required?
		}

		// sample sentence sentA and sentB from mono_cor_s and mono_cor_s respectively
		Sentence sent;
		if (flag){// sample from A
			sent = mono_cor_s[orders_s[id_s++]];
			p_edec = &edec_s2t;
			p_am_s2t = &am_s2t;
			p_am_t2s = &am_t2s;
			p_mlm = &mlm_t;
		}
		else{// sample from B
			sent = mono_cor_t[orders_t[id_t++]];
			p_edec = &edec_t2s;
			p_am_s2t = &am_t2s;
			p_am_t2s = &am_s2t;
			p_mlm = &mlm_s;
		}

		// generate K translated sentences s_{mid,1},...,s_{mid,K} using beam search according to translation model P(.|sentA; mod_am_s2t).
		std::vector<EnsembleDecoderHypPtr> v_mid_hyps = p_edec->GenerateNbest(sent, K, cg);
		std::vector<Expression> v_r1, v_r2;
		for (auto& mid_hyp : v_mid_hyps){
			// get hypo info
			const Sentence& mid_trans = mid_hyp->GetSentence();
			//float mid_score = mid_hyp->GetScore();//unused for now
			//const Sentence& mid_aligns = mid_hyp->GetAlignment();//unused for now
			
			// set the language-model reward for current sampled sentence from p_mlm_t
			auto r1 = -p_mlm->ComputeNLL(mid_trans, cg)/*negative log likelihood*/ / mid_trans.size()/*length-normalized*/;// FIXME: negative entropy as reward, correct?

			// set the communication reward for current sampled sentence from p_am_t2s
			auto r2 = -p_am_t2s->BuildGraph(mid_trans, sent, cg, stats, nullptr, nullptr);// = log(P(sentA|trans; mod_am_t2s))

			auto r = alpha * r1 + (1.f - alpha) * r2;// reward interpolation
			v_r1.push_back(r * (-p_am_s2t->BuildGraph(sent, mid_trans, cg, stats, nullptr, nullptr))/*log(P(trans|sentA; mod_am_s2t))*/);
			v_r2.push_back((1.f - alpha) * r2);
		}

		// set total loss function
		Expression i_loss_s2t = sum(v_r1) / K;// use average(v_r1) instead?
		Expression i_loss_t2s = sum(v_r2) / K;// use average(v_r2) instead?
		Expression i_loss = i_loss_s2t + i_loss_t2s;

		// execute forward step
		cg.incremental_forward(i_loss);		
		
		// execute backward step (including computation of derivatives)
		cg.backward(i_loss);

		// update parameters
		p_sgd_s2t->update();// gradient ascent or descent?
		p_sgd_t2s->update();

		// switch source and target roles
		flag = !flag;

		if (id_s == id_t) r++;

		// testing over the development data to check the improvements (after a desired number of rounds)
		cg.clear();
		if (r == DEV_ROUND){
			// s2t model
			ModelStats dstats_s2t, dstats_t2s;
			for (unsigned i = 0; i < dev_cor.size(); ++i) {
				Sentence ssent, tsent;
				tie(ssent, tsent) = dev_cor[i];  

				auto i_xent_s2t = am_s2t.BuildGraph(ssent, tsent, cg, dstats_s2t, nullptr, nullptr, nullptr, nullptr);
				auto i_xent_t2s = am_t2s.BuildGraph(tsent, ssent, cg, dstats_t2s, nullptr, nullptr, nullptr, nullptr);
				dstats_s2t.loss += as_scalar(cg.forward(i_xent_s2t));
				dstats_t2s.loss += as_scalar(cg.forward(i_xent_t2s));
			}

			if (dstats_s2t.loss < best_loss_s2t) {
				best_loss_s2t = dstats_s2t.loss;
				dynet::save_dynet_model(modfile_am_s2t, &mod_am_s2t);
			}

			if (dstats_t2s.loss < best_loss_t2s) {
				best_loss_t2s = dstats_t2s.loss;
				dynet::save_dynet_model(modfile_am_t2s, &mod_am_t2s);
			}
	
			cerr << "--------------------------------------------------------------------------------------------------------" << endl;
			cerr << "***DEV (s2t) [epoch=" << epoch_s2t + (float)id_s/(float)orders_s.size() << " eta=" << p_sgd_s2t->learning_rate << "]" << " sents=" << dev_cor.size() << " src_unks=" << dstats_s2t.words_src_unk << " trg_unks=" << dstats_s2t.words_tgt_unk << " E=" << (dstats_s2t.loss / dstats_s2t.words_tgt) << " ppl=" << exp(dstats_s2t.loss / dstats_s2t.words_tgt) << endl;
			cerr << "***DEV (t2s) [epoch=" << epoch_t2s + (float)id_t/(float)orders_t.size() << " eta=" << p_sgd_t2s->learning_rate << "]" << " sents=" << dev_cor.size() << " src_unks=" << dstats_t2s.words_src_unk << " trg_unks=" << dstats_t2s.words_tgt_unk << " E=" << (dstats_t2s.loss / dstats_t2s.words_tgt) << " ppl=" << exp(dstats_t2s.loss / dstats_t2s.words_tgt) << endl;
			cerr << "--------------------------------------------------------------------------------------------------------" << endl;

			r = 0;
		}
	}
}

ParaCorpus Read_ParaCorpus(const string &filename)
{
	ifstream in(filename);
	assert(in);
	
	ParaCorpus corpus;
	string line;
	int lc = 0, stoks = 0, ttoks = 0;
	while (getline(in, line)) {
		++lc;
		Sentence source, target;
		read_sentence_pair(line, source, sdict, target, tdict);
		corpus.push_back(SentencePair(source, target));
		
		stoks += source.size();
		ttoks += target.size();

		if ((source.front() != kSRC_SOS && source.back() != kSRC_EOS) ||
				(target.front() != kTGT_SOS && target.back() != kTGT_EOS)) {
			cerr << "Sentence in " << filename << ":" << lc << " didn't start or end with <s>, </s>\n";
			abort();
		}
	}
	cerr << lc << " lines, " << stoks << " & " << ttoks << " tokens (s & t), " << sdict.size() << " & " << tdict.size() << " types\n";
	return corpus;
}

MonoCorpus Read_MonoCorpus(const string &filename, Dict& d)
{
	int sos = d.convert("<s>"), eos = d.convert("</s>");

	ifstream in(filename);
	assert(in);

	MonoCorpus corpus;
	string line;
	int lc = 0, toks = 0;
	while (getline(in, line)) {
		++lc;
		Sentence sent = read_sentence(line, d);
		corpus.push_back(sent);
		
		toks += sent.size();

		if ((sent.front() != sos && sent.back() != eos)) {
			cerr << "Sentence in " << filename << ":" << lc << " didn't start or end with <s>, </s>\n";
			abort();
		}
	}
	cerr << lc << " lines, " << toks << " tokens (s)" << endl;
	return corpus;
}

void InitialiseModel(ParameterCollection &model, const string &filename)
{
	cerr << "Initialising model parameters from file: " << filename << endl;
	dynet::load_dynet_model(filename, &model);
}



