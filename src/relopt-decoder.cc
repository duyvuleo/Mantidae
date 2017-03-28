#include "relopt-decoder.h"

#include <iostream>
#include <fstream>
#include <sstream>

#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/program_options/parsers.hpp>
#include <boost/program_options/variables_map.hpp>
#include <boost/algorithm/string.hpp>

using namespace std;
using namespace dynet;
using namespace boost::program_options;

dynet::Dict sd;
dynet::Dict td;

bool verbose;

//typedef pair<Sentence, Sentence> SentencePair;
typedef tuple<Sentence, Sentence, int> SentencePair; // includes document id (optional)
typedef vector<SentencePair> Corpus;

template <class rnn_t>
int main_body(variables_map vm);

int main(int argc, char** argv) {
	dynet::initialize(argc, argv);

	// command line processing
	variables_map vm; 
	options_description opts("Allowed options");
	opts.add_options()
		("help", "print help message")
		("config,c", value<string>(), "config file specifying additional command line options")
		//-----------------------------------------		
		("train,t", value<string>(), "file containing training sentences, with "
			"each line consisting of source ||| target.")
		("initialise,i", value<string>(), "load pre-trained parameters (left-to-right AM model) from file")
		("initialise_r2l", value<string>(), "load pre-trained parameters (right-to-left AM model) from file")
		("initialise_t2s", value<string>(), "load pre-trained parameters (target-to-source AM model) from file")
		("initialise_bam", value<string>(), "load pre-trained parameters (integrated source-to-target and target-to-source BAM model) from file")
		("initialise_rnnlm", value<string>(), "load pre-trained parameters (RNNLM model) from file")
		//-----------------------------------------
		("slayers", value<unsigned>()->default_value(SLAYERS), "use <num> layers for source RNN components")
		("tlayers", value<unsigned>()->default_value(TLAYERS), "use <num> layers for target RNN components")
		("align,a", value<unsigned>()->default_value(ALIGN_DIM), "use <num> dimensions for alignment projection")
		("hidden,h", value<unsigned>()->default_value(HIDDEN_DIM), "use <num> dimensions for recurrent hidden states")
		//-----------------------------------------
		("gru", "use Gated Recurrent Unit (GRU) for recurrent structure; default RNN")
		("lstm", "use Long Short Term Memory (GRU) for recurrent structure; default RNN")
		("vlstm", "use Vanilla Long Short Term Memory (VLSTM) for recurrent structure; default RNN")
		("dglstm", "use Depth-Gated Long Short Term Memory (DGLSTM) (Kaisheng et al., 2015; https://arxiv.org/abs/1508.03790) for recurrent structure; default RNN") // FIXME: add this to dynet?
		//-----------------------------------------
		("bidirectional", "use bidirectional recurrent hidden states as source embeddings, rather than word embeddings")
		//-----------------------------------------
		("giza", "use GIZA++ style features in attentional components (corresponds to all of the 'gz' options)")
		("gz-position", "use GIZA++ positional index features")
		("gz-markov", "use GIZA++ markov context features")
		("gz-fertility", "use GIZA++ fertility type features")
		("fertility,f", "learn Normal model of word fertility values")
		//-----------------------------------------		
		("swap", "swap roles of source and target, i.e., learn p(source|target)")
		//-----------------------------------------		
		("relopt_algo", value<unsigned>()->default_value(RELOPT_ALGO::EG), "The algorithm (1:EG; 2:AdaptiveEG 3:SOFTMAX; 5:SPARSEMAX) for relaxed optimization")
		("relopt_init", value<unsigned>()->default_value(RELOPT_INIT::UNIFORM), "The initialization method (0:UNIFORM; 1:REFERENCE_PROBABILITY; 2:REFERENCE_ONE_HOT) for relaxed optimization")
		("relopt_eta", value<float>()->default_value(1.f), "The learning rate for relaxed optimization")
		("relopt_eta_decay", value<float>()->default_value(2.f), "The learning rate decay for relaxed optimization")
		("relopt_eta_power", value<float>()->default_value(0.f), "The learning rate power for relaxed optimization")
		("relopt_momentum", value<float>()->default_value(0.f), "The momentum weight for gradient descent training")
		("relopt_max_iters", value<unsigned>()->default_value(MAX_ALGO_STEP), "The maximum iteration for relaxed optimization")
		("relopt_m_weight", value<float>()->default_value(1.f), "The weight of main model (left-to-right/source-to-target)")
		("relopt_coverage_weight", value<float>()->default_value(0.f), "The coverage penalty weight")
		("relopt_coverage_C", value<float>()->default_value(1.f), "The coverage C (e.g., >=1 and <=3")
		("relopt_glofer_weight", value<float>()->default_value(0.f), "The global fertility weight")
		("jdec_bidir_alpha", value<float>()->default_value(0.f), "The interpolation weight for joint decoding in bidirectional models")
		("jdec_biling_alpha", value<float>()->default_value(0.f), "The interpolation weight for joint decoding in bilingual models")
		("jdec_biling_trace_alpha", value<float>()->default_value(0.f), "The trace bonus weight for joint decoding in bilingual models")
		("jdec_mlm_alpha", value<float>()->default_value(0.f), "The interpolation weight for joint decoding with additional language model(s)")
		("ent_gamma", value<float>()->default_value(1.f), "The hyper-parameter for weighting the entropy regularizer of SOFTMAX or SPARSEMAX")
		("relopt_add_extra_words", value<unsigned>()->default_value(0), "No. of extra words added")
		("src_in", value<string>()->default_value(""), "File to read the source from")
		("ref_in", value<string>()->default_value(""), "File to read the reference from")
		("trg_out", value<string>()->default_value(""), "File to write the output files")
		("cline", value<unsigned>()->default_value(0), "line number to be continued for processing (used in a crashed running)")
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
	
	if (vm.count("help") || vm.count("train") != 1 || vm.count("ref_in") != 1 || vm.count("src_in") != 1) {
		cout << opts << "\n";
		return 1;
	}

	if (vm.count("lstm"))// LSTM
		return main_body<LSTMBuilder>(vm);
	else if (vm.count("vlstm"))// VLSTM
		return main_body<VanillaLSTMBuilder>(vm);
	else if (vm.count("gru"))// GRU
		return main_body<GRUBuilder>(vm);
	//else if (vm.count("dglstm"))// DGLSTM
		//return main_body<DGLSTMBuilder>(vm);
	else// Vanilla RNN
		return main_body<SimpleRNNBuilder>(vm);
}

void Initialise(Model &model, const string &filename);

Corpus Read_Corpus(const string &filename, bool doco);
std::vector<int> Read_Numbered_Sentence(const std::string& line, Dict* sd, std::vector<int> &ids);
void Read_Numbered_Sentence_Pair(const std::string& line, std::vector<int>* s, Dict* sd, std::vector<int>* t, Dict* td, std::vector<int> &ids);

template <class rnn_t>
int main_body(variables_map vm)
{
	kSRC_SOS = sd.convert("<s>");
	kSRC_EOS = sd.convert("</s>");
	kTGT_SOS = td.convert("<s>");
	kTGT_EOS = td.convert("</s>");
	verbose = vm.count("verbose");

	SLAYERS = vm["slayers"].as<unsigned>();
	TLAYERS = vm["tlayers"].as<unsigned>();  
	ALIGN_DIM = vm["align"].as<unsigned>(); 
	HIDDEN_DIM = vm["hidden"].as<unsigned>(); 
	bool bidir = vm.count("bidirectional");
	bool giza = vm.count("giza");
	bool giza_pos = giza || vm.count("gz-position");
	bool giza_markov = giza || vm.count("gz-markov");
	bool giza_fert = giza || vm.count("gz-fertility");
	bool fert = vm.count("fertility");
	bool swap = vm.count("swap");
	bool doco = vm.count("document");
	string flavour = "RNN";
	if (vm.count("lstm"))	flavour = "LSTM";
	else if (vm.count("gru"))	flavour = "GRU";
	//else if (vm.count("dglstm")) flavour = "DGLSTM";

	Corpus training, devel, testing;
	string line;
	cerr << "Reading training data from " << vm["train"].as<string>() << "...\n";
	training = Read_Corpus(vm["train"].as<string>(), doco);
	sd.freeze(); // no new word types allowed
	td.freeze(); // no new word types allowed
	
	SRC_VOCAB_SIZE = sd.size();
	TGT_VOCAB_SIZE = td.size();

	if (vm.count("devel")) {
	cerr << "Reading dev data from " << vm["devel"].as<string>() << "...\n";
	devel = Read_Corpus(vm["devel"].as<string>(), doco);
	}

	if (vm.count("test")) {
		// otherwise "test" file is assumed just to contain source language strings
		cerr << "Reading test examples from " << vm["test"].as<string>() << endl;
		testing = Read_Corpus(vm["test"].as<string>(), doco);
	}

	if (swap) {
		cerr << "Swapping role of source and target\n";

		std::swap(sd, td);
		std::swap(kSRC_SOS, kTGT_SOS);
		std::swap(kSRC_EOS, kTGT_EOS);
		std::swap(kSRC_UNK, kTGT_UNK);
		std::swap(SRC_VOCAB_SIZE, TGT_VOCAB_SIZE);

		for (auto &sent: training)
			std::swap(get<0>(sent), get<1>(sent));
		for (auto &sent: devel)
			std::swap(get<0>(sent), get<1>(sent));
		for (auto &sent: testing)
			std::swap(get<0>(sent), get<1>(sent));
	}

	// pre-trained seq2seq model(s)
	Model model, model_l2r, model_t2s, model_mlm, model_bam;
	
	cerr << "%% Using " << flavour << " recurrent units" << endl;
	AttentionalModel<rnn_t> am(&model, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE,
		SLAYERS, TLAYERS, HIDDEN_DIM, ALIGN_DIM, bidir, giza_pos, giza_markov, giza_fert, doco, fert);
   
	if (fert)
		am.Add_Global_Fertility_Params(&model, HIDDEN_DIM, bidir);// testing/rescoring phase: add extra global fertility parameters (uninitialized model parameters)

	if (vm.count("initialise")){
		cerr << "*Loading left-to-right/source-to-target AM model..." << endl;
		Initialise(model, vm["initialise"].as<string>());
		cerr << "Count of model parameters: " << model.parameter_count() << endl;
	}
	else
		assert("No pre-trained model loaded! Failed!");

	// right-to-left and source-to-target model
	AttentionalModel<rnn_t>* p_am_r2l = nullptr;//FIXME: relax the use of pointer here (see RNNLanguageModel)?
	if (vm.count("initialise_r2l")){
		cerr << "*Loading right-to-left AM model..." << endl;
		p_am_r2l = new AttentionalModel<rnn_t>(&model_l2r, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE,
		SLAYERS, TLAYERS, HIDDEN_DIM, ALIGN_DIM, bidir, giza_pos, giza_markov, giza_fert, doco, fert);
   		Initialise(model_l2r, vm["initialise_r2l"].as<string>());
		cerr << "Count of model parameters: " << model_l2r.parameter_count() << endl;
	}

	// target-to-source model
	AttentionalModel<rnn_t>* p_am_t2s = nullptr;//FIXME: relax the use of pointer here (see RNNLanguageModel)?
	if (vm.count("initialise_t2s")){
		cerr << "*Loading target-to-source AM model..." << endl;
		p_am_t2s = new AttentionalModel<rnn_t>(&model_t2s, TGT_VOCAB_SIZE, SRC_VOCAB_SIZE,
		SLAYERS, TLAYERS, HIDDEN_DIM, ALIGN_DIM, bidir, giza_pos, giza_markov, giza_fert, doco, fert);
   		Initialise(model_t2s, vm["initialise_t2s"].as<string>());
		cerr << "Count of model parameters: " << model_t2s.parameter_count() << endl;
	}

	// biattentional model
	BiAttentionalModel<rnn_t>* p_bam = nullptr;
	if (vm.count("initialise_bam")){
		cerr << "*Loading biattentional (integrated source-to-target and target-to-source) BAM model..." << endl;
		p_bam = new BiAttentionalModel<rnn_t>(&model_bam, bidir, giza_pos, giza_markov, giza_fert, doco, fert, vm["jdec_biling_trace_alpha"].as<float>());
   		Initialise(model_bam, vm["initialise_bam"].as<string>());
		cerr << "Count of model parameters: " << model_bam.parameter_count() << endl;
	}

	// RNN language model
	RNNLanguageModel<rnn_t> rnnlm;
	if (vm.count("initialise_rnnlm")){
		cerr << "*Loading monolingual RNNLM model..." << endl;
		rnnlm.LoadModel(&model_mlm, vm["initialise_rnnlm"].as<string>());
  		cerr << "Count of model parameters: " << model_mlm.parameter_count() << endl;
	}

	// Start the decoding with relaxed optimization
	cerr << endl << "*** RelOptInference" << endl;
	
	RelOptDecoder<AttentionalModel<rnn_t>, BiAttentionalModel<rnn_t>, RNNLanguageModel<rnn_t>> relopt_decoder;
	RelOptConfig relopt_cf;

	// Read in the config
	relopt_cf.algorithm = vm["relopt_algo"].as<unsigned>();
	relopt_cf.intialization_type = vm["relopt_init"].as<unsigned>();
	relopt_cf.eta = vm["relopt_eta"].as<float>();
	relopt_cf.eta_decay = vm["relopt_eta_decay"].as<float>();
	relopt_cf.eta_power = vm["relopt_eta_power"].as<float>();
	relopt_cf.momentum = vm["relopt_momentum"].as<float>();
	relopt_cf.max_iters = vm["relopt_max_iters"].as<unsigned>();
	relopt_cf.m_weight = vm["relopt_m_weight"].as<float>();
	relopt_cf.coverage_weight = vm["relopt_coverage_weight"].as<float>();
	relopt_cf.coverage_C = vm["relopt_coverage_C"].as<float>();
	relopt_cf.glofer_weight = vm["relopt_glofer_weight"].as<float>();
	relopt_cf.add_extra_words = vm["relopt_add_extra_words"].as<unsigned>();
	relopt_cf.jdec_bidir_alpha = vm["jdec_bidir_alpha"].as<float>();
	relopt_cf.jdec_biling_alpha = vm["jdec_biling_alpha"].as<float>();
	relopt_cf.jdec_biling_trace_alpha = vm["jdec_biling_trace_alpha"].as<float>();
	relopt_cf.jdec_mlm_alpha = vm["jdec_mlm_alpha"].as<float>();
	relopt_cf.ent_gamma = vm["ent_gamma"].as<float>();

	// Read in the model file
	cerr << "Loading model(s)...";
	relopt_decoder.SetVerbose((unsigned)vm.count("verbose"));
	relopt_decoder.LoadModel(&am
		, p_am_r2l
		, p_am_t2s
		, p_bam
		, &rnnlm
		, &sd, &td);
	cerr << " done!" << endl << endl;

	// Get the source input
	shared_ptr<ifstream> src_in;
	src_in.reset(new ifstream(vm["src_in"].as<std::string>()));
	if(!*src_in)
		assert("Could not find src_in file ");// << vm["src_in"].as<std::string>() << "/" << vm["ref_in"].as<std::string>());
	
	// For target output
	shared_ptr<ofstream> trg_out, costs_out;
	trg_out.reset(new ofstream(vm["trg_out"].as<std::string>()));
	costs_out.reset(new ofstream(vm["trg_out"].as<std::string>() + ".costs"));
	if(!*trg_out && !*costs_out)
		assert("Could not find trg_out files ");// << vm["trg_out"].as<std::string>());

	// Get the references (e.g., greedy Viterbi, beam search, (human) reference)
	std::vector<std::string> strs;
	boost::split(strs, vm["ref_in"].as<std::string>(), boost::is_any_of("|"));
	std::vector<shared_ptr<ifstream>> refs_in(strs.size());
	for (auto i : boost::irange(0, (int)strs.size())){
		refs_in[i].reset(new ifstream(strs[i]));
		if(!*refs_in[i])
		assert("Could not find ref_in file(s) ");// << strs[i]);
	}

	std::string line_src;
	std::vector<std::string> lines_ref(refs_in.size());
	vector<tuple<float,float,float>> avg_scores(refs_in.size(), make_tuple(0.f, 0.f, 0.f));
	unsigned line_count = 0, line_continued = vm["cline"].as<unsigned>();
	while (getline(*src_in, line_src)){
		if ("" == line_src) break;

		for (auto i : boost::irange(0, (int)refs_in.size())){
			getline(*refs_in[i], lines_ref[i]);
			if ("" == lines_ref[i]) break;
		}

		// for continuing a crashed run
		if (line_continued > 0 && line_count < line_continued){ 
			line_count++;
			continue;
		}

		cerr << "Processing line " << line_count << "..." << endl;
		if (vm.count("verbose")) cerr << "Decoding sentence: " << line_src << endl;
		unsigned j = 0;
		for (auto& ref : lines_ref){
			if (vm.count("verbose")) cerr << "--------------------" << endl;
	  		float gcost_ref = relopt_decoder.GetNLLCost(line_src, ref);
			if (vm.count("verbose"))
				cerr << "Referencing from: " << ref << " (discrete cost=" << gcost_ref << ")" << endl;
			RelOptOutput ir = relopt_decoder.GDDecode(line_src, ref, relopt_cf);
			float gcost_inf = relopt_decoder.GetNLLCost(line_src, "<s> " + ir.decoded_sent + " </s>");
			*trg_out << ir.decoded_sent << endl;
			if (vm.count("verbose"))
				cerr << "Inference result: " << ir.decoded_sent << endl;
			std::get<0>(avg_scores[j]) += gcost_ref;// discrete cost of reference
			std::get<1>(avg_scores[j]) += ir.cost;// fractional/continuous cost of inference result
			std::get<2>(avg_scores[j]) += gcost_inf;// discrete cost of inference result
			if (vm.count("verbose")) 
				cerr << "Inference result's discrete cost=" << gcost_inf << endl;
		}
		if (vm.count("verbose")) cerr << endl;

		line_count++;
	}

	for (unsigned j = 0; j < refs_in.size(); j++){
		*costs_out << "Discrete costs of reference" << j << ": total=" << std::get<0>(avg_scores[j]) << " average=" << std::get<0>(avg_scores[j])/line_count << endl;
		*costs_out << "Continuous/Fractional costs of inference result" << j << ": total=" << std::get<1>(avg_scores[j]) << " average=" << std::get<1>(avg_scores[j])/line_count << endl;
		*costs_out << "Discrete costs of inference result " << j << ": total=" << std::get<2>(avg_scores[j]) << " average=" << std::get<2>(avg_scores[j])/line_count << endl;
	}

	//------------------------------------
	// cleaning up the allocated memory
	if (vm.count("initialise_r2l"))
		delete p_am_r2l; 
	if (vm.count("initialise_t2s"))
		delete p_am_t2s; 
	
	//dynet::cleanup();
	//------------------------------------

	return EXIT_SUCCESS;	
}

Corpus Read_Corpus(const string &filename, bool doco)
{
	ifstream in(filename);
	assert(in);
	Corpus corpus;
	string line;
	int lc = 0, stoks = 0, ttoks = 0;
	vector<int> identifiers({ -1 });
	while (getline(in, line)) {
		++lc;
		Sentence source, target;
		if (doco) 
			Read_Numbered_Sentence_Pair(line, &source, &sd, &target, &td, identifiers);
		else
			read_sentence_pair(line, source, sd, target, td);
		corpus.push_back(SentencePair(source, target, identifiers[0]));
		stoks += source.size();
		ttoks += target.size();

		if ((source.front() != kSRC_SOS && source.back() != kSRC_EOS) ||
				(target.front() != kTGT_SOS && target.back() != kTGT_EOS)) {
			cerr << "Sentence in " << filename << ":" << lc << " didn't start or end with <s>, </s>\n";
			abort();
		}
	}
	cerr << lc << " lines, " << stoks << " & " << ttoks << " tokens (s & t), " << sd.size() << " & " << td.size() << " types\n";
	return corpus;
}

std::vector<int> Read_Numbered_Sentence(const std::string& line, Dict* sd, vector<int> &identifiers) {
	std::istringstream in(line);
	std::string word;
	std::vector<int> res;
	std::string sep = "|||";
	if (in) {
		identifiers.clear();
		while (in >> word) {
			if (!in || word.empty()) break;
			if (word == sep) break;
			identifiers.push_back(atoi(word.c_str()));
		}
	}

	while(in) {
		in >> word;
		if (!in || word.empty()) break;
		res.push_back(sd->convert(word));
	}
	return res;
}


void Read_Numbered_Sentence_Pair(const std::string& line, std::vector<int>* s, Dict* sd, std::vector<int>* t, Dict* td, vector<int> &identifiers) 
{
	std::istringstream in(line);
	std::string word;
	std::string sep = "|||";
	Dict* d = sd;
	std::vector<int>* v = s; 

	if (in) {
		identifiers.clear();
		while (in >> word) {
			if (!in || word.empty()) break;
			if (word == sep) break;
			identifiers.push_back(atoi(word.c_str()));
		}
	}

	while(in) {
		in >> word;
		if (!in) break;
		if (word == sep) { d = td; v = t; continue; }
		v->push_back(d->convert(word));
	}
}

void Initialise(Model &model, const string &filename)
{
	cerr << "Initialising model parameters from file: " << filename << endl;
	//ifstream in(filename, ifstream::in);
	//boost::archive::text_iarchive ia(in);
	//ia >> model;
	dynet::load_dynet_model(filename, &model);
}

