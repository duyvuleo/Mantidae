/*
 * This is an implementation of the following work:
 * Dual Inference for Machine Learning
 * Yingce Xia, Jiang Bian, Tao Qin, Nenghai Yu, and Tie-Yan Liu
 * http://home.ustc.edu.cn/~xiayingc/pubs/ijcai_17.pdf (accepted at IJCAI 2017)
 * Developed by Cong Duy Vu Hoang (vhoang2@student.unimelb.edu.au)
 * Date: 21 August 2017
 *
*/

#include "attentional.h" // AM
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

int main_body(variables_map vm);

void InitialiseModel(ParameterCollection &model, const string &filename);
ParaCorpus Read_ParaCorpus(const string &filename);

template<class AM_t>
void Dual_Inference(AM_t& am_primal, AM_t& am_dual, bool dual_type
		, const string& test_file
		, unsigned nbest_size, unsigned beam_size, float alpha);

int main(int argc, char** argv) {
	dynet::initialize(argc, argv);

	// command line processing
	/* Example:
	1) >> ./build_gpu_v2/src/dual-inference --dynet_mem 100 -t /home/vhoang2/works/mantis-dev/experiments/data/multi30k/train.de-en.atok.cap -T /home/vhoang2/works/mantis-dev/experiments/data/multi30k/	test.de.atok.cap --initialise_am_s2t experiments/models/multi30k/de-en/params.de-en.AM.sl_2_tl_4_h_512_a_512_lstm_bidir_lr2 --initialise_am_t2s experiments/models/multi30k/en-de/params.en-de.AM.sl_2_tl_4_h_512_a_512_lstm_bidir_lr2 --model_conf experiments/models/multi30k/de-en/dual-inference.cfg --K 20 --beam_size 5 --alpha 0.5 | sed 's/<s> //g' | sed 's/ <\/s>//g' > experiments/models/multi30k/de-en/translation-beam5.test.de-en.AM.sl_2_tl_4_h_512_a_512_lstm_bidir_dualinfer20-5-05
	2) >> ./build_gpu_v2/src/dual-inference --dynet_mem 100 -t /home/vhoang2/works/mantis-dev/experiments/data/multi30k/train.de-en.atok.cap -T /home/vhoang2/works/mantis-dev/experiments/data/multi30k/test.de.atok.cap --initialise_am_s2t experiments/models/multi30k/de-en/params.de-en.AM.sl_2_tl_4_h_512_a_512_lstm_bidir_lr2 --initialise_am_r2l experiments/models/multi30k/de-en/params.de-en.AM.sl_2_tl_4_h_512_a_512_lstm_bidir_lr2_r2l --model_conf experiments/models/multi30k/de-en/dual-inference.cfg --K 20 --beam_size 5 --alpha 0.5 | sed 's/<s> //g' | sed 's/ <\/s>//g' > experiments/models/multi30k/de-en/translation-beam5.test.de-en.AM.sl_2_tl_4_h_512_a_512_lstm_bidir_dualinferr2l20-5-05
	*/
	variables_map vm; 
	options_description opts("Allowed options");
	opts.add_options()
		("help", "print help message")
		("config,c", value<string>(), "config file specifying additional command line options")
		//-----------------------------------------		
		("train,t", value<string>(), "file containing training parallel sentences (for source/targer vocabulary building on-the-fly), with "
			"each line consisting of source ||| target.")
		("test,T", value<string>(), "file containing testing sentences")
		//-----------------------------------------
		("initialise_am_s2t", value<string>(), "load pre-trained model parameters (source-to-target AM model) from file")
		("initialise_am_t2s", value<string>(), "load pre-trained model parameters (target-to-source AM model) from file")
		("initialise_am_r2l", value<string>(), "load pre-trained model parameters (right-to-left AM model) from file")
		//-----------------------------------------
		("model_conf", value<string>(), "config file specifying model configurations for all models (line by line)")
		("K", value<unsigned>()->default_value(10), "the K value for sampling K-best translations from AM models")
		("beam_size", value<unsigned>()->default_value(5), "the beam size of beam search decoding")
		("alpha", value<float>()->default_value(0.5f), "the alpha hyper-parameter for balancing the rewards")
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

	//--- load models
	ParameterCollection model_am_s2t, model_am_t2s, model_am_r2l;
	std::shared_ptr<AttentionalModel<LSTMBuilder>> p_am_s2t, p_am_t2s, p_am_r2l;// FIXME: assume all models use the same LSTM RNN structure (for quick implementation)

	string line;
	stringstream ss;
	/* Example configuration file
	2 4 512 512
	2 4 512 512
	*/
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
	if (vm.count("initialise_am_t2s")){
		p_am_t2s.reset(new AttentionalModel<LSTMBuilder>(&model_am_t2s, TGT_VOCAB_SIZE, SRC_VOCAB_SIZE,
			SLAYERS, TLAYERS, HIDDEN_DIM, ALIGN_DIM, true, false, false, false, false, false));
		InitialiseModel(model_am_t2s, vm["initialise_am_t2s"].as<string>());

		//--- execute dual-inference	
		Dual_Inference<AttentionalModel<LSTMBuilder>>(*p_am_s2t
			, *p_am_t2s, true
			, vm["test"].as<string>()
			, K, beam_size, vm["alpha"].as<float>());
	}
	else if (vm.count("initialise_am_r2l")){
		p_am_r2l.reset(new AttentionalModel<LSTMBuilder>(&model_am_r2l, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE,
			SLAYERS, TLAYERS, HIDDEN_DIM, ALIGN_DIM, true, false, false, false, false, false));
		InitialiseModel(model_am_r2l, vm["initialise_am_r2l"].as<string>());

		//--- execute dual-inference	
		Dual_Inference<AttentionalModel<LSTMBuilder>>(*p_am_s2t
			, *p_am_r2l, false
			, vm["test"].as<string>()
			, K, beam_size, vm["alpha"].as<float>());
	}
	else{
		cerr << "Dual inference requires additional dual model (either target-to-source or right-to-left)" << endl;
		return EXIT_FAILURE;
	}	
	
	// complete
	return EXIT_SUCCESS;
}

template<class AM_t>
void Dual_Inference(AM_t& am_primal, AM_t& am_dual, bool dual_type
		, const string& test_file
		, unsigned nbest_size, unsigned beam_size, float alpha)
{
	// set up the decoder(s)
	EnsembleDecoder<AM_t> edec_primal(std::vector<std::shared_ptr<AM_t>>({std::make_shared<AM_t>(am_primal)}), &tdict);
	edec_primal.SetBeamSize(beam_size);
	EnsembleDecoder<AM_t> edec_dual(std::vector<std::shared_ptr<AM_t>>({std::make_shared<AM_t>(am_dual)}), &sdict);
	edec_dual.SetBeamSize(beam_size);

	int lno = 0;
	cerr << "Reading test examples from " << test_file << endl;

	MyTimer timer_dec("completed in");

	ifstream in(test_file);
	assert(in);

	string line;
	Sentence last_source, source;
	ModelStats stats;// normally unused
	while (getline(in, line)) {
		source = read_sentence(line, sdict);

		if (source.front() != kSRC_SOS && source.back() != kSRC_EOS) {
			cerr << "Sentence in " << test_file << ":" << lno << " didn't start or end with <s>, </s>\n";
			abort();
		}
		
		// generate k-best translation list
		ComputationGraph cg;
		std::vector<EnsembleDecoderHypPtr> v_trg_hyps = edec_primal.GenerateNbest(source, nbest_size, cg);
		cg.clear();
		vector<float> all_scores;
		Sentence trg_trans_reversed;
		for (auto& trg_hyp : v_trg_hyps){
			const Sentence& trg_trans = trg_hyp->GetSentence();
			if (!dual_type){
				trg_trans_reversed = trg_trans;
				std::reverse(trg_trans_reversed.begin() + 1, trg_trans_reversed.end() - 1);
			}

			auto i_xent_primal = alpha * am_primal.BuildGraph(source, trg_trans, cg, stats, nullptr, nullptr, nullptr, nullptr);
			auto i_xent_dual = (1.f - alpha) * (dual_type == true)?am_dual.BuildGraph(trg_trans, source, cg, stats, nullptr, nullptr, nullptr, nullptr):am_dual.BuildGraph(source, trg_trans_reversed, cg, stats, nullptr, nullptr, nullptr, nullptr);

			auto i_xent = i_xent_primal + i_xent_dual;
			all_scores.push_back(as_scalar(cg.incremental_forward(i_xent)));

			cg.clear();
		}

		int i_best = std::distance(all_scores.begin(), std::min_element(all_scores.begin(), all_scores.end()));// find the argmin

		if (i_best == -1) i_best = 0;// something wrong, backoff to the best translation
		
		// convert to string
		stringstream ss;
		bool first = true;
		for (auto &w: v_trg_hyps[i_best]->GetSentence()) {
			if (!first) ss << " ";
			ss << tdict.convert(w);
			first = false;
		}

		// write to output console
		cout << ss.str() << endl;
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

void InitialiseModel(ParameterCollection &model, const string &filename)
{
	cerr << "Initialising model parameters from file: " << filename << endl;
	dynet::load_dynet_model(filename, &model);
}



