#include "attentional-joint.h"
#include "rnnlm.h"
#include "math-utils.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <limits>

#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/program_options/parsers.hpp>
#include <boost/program_options/variables_map.hpp>

using namespace std;
using namespace dynet;
using namespace boost::program_options;

unsigned MINIBATCH_SIZE = 1;

bool DEBUGGING_FLAG = false;

unsigned TREPORT = 50;
unsigned DREPORT = 5000;

dynet::Dict sd;
dynet::Dict td;

bool verbose;

typedef vector<int> Sentence;
typedef tuple<Sentence, Sentence> SentencePair;
typedef vector<SentencePair> Corpus;

#define WTF(expression) \
	std::cout << #expression << " has dimensions " << cg.nodes[expression.i]->dim << std::endl;
#define KTHXBYE(expression) \
	std::cout << *cg.get_value(expression.i) << std::endl;

#define LOLCAT(expression) \
	WTF(expression) \
	KTHXBYE(expression)

void Load_Vocabs(const string& src_vocab_file, const string& trg_vocab_file);

void Initialise(ParameterCollection &model, const string &filename);

inline size_t Calc_Size(const Sentence & src, const Sentence & trg);
inline size_t Create_MiniBatches(const Corpus& cor
	, size_t max_size
	, std::vector<std::vector<Sentence> > & train_src_minibatch
	, std::vector<std::vector<Sentence> > & train_trg_minibatch
	, std::vector<size_t> & train_ids_minibatch);
inline size_t Create_MiniBatches(const Corpus& cor
	, size_t max_size
	, std::vector<std::vector<Sentence> > & train_src_minibatch
	, std::vector<std::vector<Sentence> > & train_trg_minibatch
	, std::vector<std::vector<Sentence> > & train_rtrg_minibatch
	, std::vector<size_t> & train_ids_minibatch);

Trainer* Create_SGDTrainer(ParameterCollection& model
				, unsigned sgd_type
				, float lr_eta
				, float g_clip_threshold
				, float sparse_updates);

template <class AM_t>
void TrainJointModel_wR2L_Batch(ParameterCollection &model, ParameterCollection &model_r2l
			, AM_t &am, AM_t &am_r2l
			, Trainer &sgd, Trainer &sgd_r2l
			, float alpha
			, Corpus &training, Corpus &devel
			, string out_file
			, bool curriculum
			, unsigned max_epochs, unsigned patience, unsigned lr_epochs, float lr_eta_decay, unsigned lr_patience);

template <class AM_t>
void TrainJointModel_wT2S_Batch(ParameterCollection &model, ParameterCollection &model_t2s
			, AM_t &am, AM_t &am_t2s
			, Trainer &sgd, Trainer &sgd_t2s
			, float alpha
			, Corpus &training, Corpus &devel
			, string out_file
			, bool curriculum
			, unsigned max_epochs, unsigned patience, unsigned lr_epochs, float lr_eta_decay, unsigned lr_patience);

Corpus Read_Corpus(const string &filename
			, bool cid=true
			, unsigned slen=0
			, unsigned eos_padding=0);
std::vector<int> Read_Numbered_Sentence(const std::string& line, Dict* sd, std::vector<int> &ids);
void Read_Numbered_Sentence_Pair(const std::string& line, std::vector<int>* s, Dict* sd, std::vector<int>* t, Dict* td, std::vector<int> &ids);

template <class rnn_t>
int main_body(variables_map vm);

int main(int argc, char** argv) {
	cerr << "*** DyNet initialization ***" << endl;
	auto dyparams = dynet::extract_dynet_params(argc, argv);
	dynet::initialize(dyparams);	

	// command line processing
	variables_map vm; 
	options_description opts("Allowed options");
	opts.add_options()
		("help", "print help message")
		("config,c", value<string>(), "config file specifying additional command line options")
		//-----------------------------------------
		("train,t", value<vector<string>>(), "file containing training sentences, with each line consisting of source ||| target.")		
		("devel,d", value<string>(), "file containing development sentences.")
		("test,T", value<string>(), "file containing testing sentences")
		("slen_limit", value<unsigned>()->default_value(0), "limit the sentence length (either source or target); none by default")
		("src_vocab", value<string>()->default_value(""), "file containing source vocabulary file; none by default (will be built from train file)")
		("trg_vocab", value<string>()->default_value(""), "file containing target vocabulary file; none by default (will be built from train file)")
		("train_percent", value<unsigned>()->default_value(100), "use <num> percent of sentences in training data; full by default")
		//-----------------------------------------
		//-----------------------------------------
		("shared_embeddings", "use shared source and target embeddings (in case that source and target use the same vocabulary; none by default")
		//-----------------------------------------
		("minibatch_size", value<unsigned>()->default_value(1), "impose the minibatch size for training (support both GPU and CPU); no by default")
		("dynet-autobatch", value<unsigned>()->default_value(0), "impose the auto-batch mode (support both GPU and CPU); no by default") //--dynet-autobatch 1		
		//-----------------------------------------
		("sgd_trainer", value<unsigned>()->default_value(0), "use specific SGD trainer (0: vanilla SGD; 1: momentum SGD; 2: Adagrad; 3: AdaDelta; 4: Adam; 5: RMSProp; 6: cyclical SGD)")
		("sparse_updates", value<bool>()->default_value(true), "enable/disable sparse update(s) for lookup parameter(s); true by default")
		("g_clip_threshold", value<float>()->default_value(5.f), "use specific gradient clipping threshold (https://arxiv.org/pdf/1211.5063.pdf); 5 by default")
		//-----------------------------------------
		("initialise,i", value<string>(), "load initial parameters from file")
		("initialise_r2l", value<string>(), "load initial parameters (from right-to-left model) from file")
		("initialise_t2s", value<string>(), "load initial parameters (from target-to-source model) from file")
		("parameters,p", value<string>(), "save best parameters to this file prefix")
		//-----------------------------------------
		("eos_padding", value<unsigned>()->default_value(0), "impose <num> of </s> padding(s) (at the end) for all training target instances; none (0) by default")
		//-----------------------------------------
		("slayers", value<unsigned>()->default_value(SLAYERS), "use <num> layers for source RNN components")
		("tlayers", value<unsigned>()->default_value(TLAYERS), "use <num> layers for target RNN components")
		("align,a", value<unsigned>()->default_value(ALIGN_DIM), "use <num> dimensions for alignment projection")
		("hidden,h", value<unsigned>()->default_value(HIDDEN_DIM), "use <num> dimensions for recurrent hidden states")
		//-----------------------------------------
		("dropout_enc", value<float>()->default_value(0.f), "use dropout (Gal et al., 2016) for RNN encoder(s), e.g., 0.5 (input=0.5;hidden=0.5;cell=0.5) for LSTM; none by default")
		("dropout_dec", value<float>()->default_value(0.f), "use dropout (Gal et al., 2016) for RNN decoder, e.g., 0.5 (input=0.5;hidden=0.5;cell=0.5) for LSTM; none by default")
		//-----------------------------------------
		("topk,k", value<unsigned>()->default_value(100), "use <num> top kbest entries, used with --kbest")
		("epochs,e", value<unsigned>()->default_value(20), "maximum number of training epochs")
		("patience", value<unsigned>()->default_value(0), "no. of times in which the model has not been improved for early stopping; default none")
		//-----------------------------------------
		("lr_eta", value<float>()->default_value(0.01f), "SGD learning rate value (e.g., 0.1 for simple SGD trainer or smaller 0.001 for ADAM trainer)")
		("lr_eta_decay", value<float>()->default_value(2.0f), "SGD learning rate decay value")
		//-----------------------------------------
		("lr_epochs", value<unsigned>()->default_value(0), "no. of epochs for starting learning rate annealing (e.g., halving)") // learning rate scheduler 1
		("lr_patience", value<unsigned>()->default_value(0), "no. of times in which the model has not been improved, e.g., for starting learning rate annealing (e.g., halving)") // learning rate scheduler 2 (which normally works better than learning rate scheduler 1, e.g., 1-2 BLEU scores better on Multi30K dataset)
		//-----------------------------------------
		("gru", "use Gated Recurrent Unit (GRU) for recurrent structure; default RNN")
		("lstm", "use Vanilla Long Short Term Memory (LSTM) for recurrent structure; default RNN")
		("dglstm", "use Depth-Gated Long Short Term Memory (DGLSTM) (Kaisheng et al., 2015; https://arxiv.org/abs/1508.03790) for recurrent structure; default RNN")
		//-----------------------------------------
		("bidirectional", "use bidirectional recurrent hidden states as source embeddings, rather than word embeddings")
		//-----------------------------------------
		// hyper-parameters for joint training
		("alpha", value<float>()->default_value(1.f), "weight balancing the importance btw the models during training; default 1.0")
		("joint_model", value<string>()->default_value("wR2L"), "joint training model type (wR2L; wT2S; wLM; wNC); default wR2L") 
		//-----------------------------------------
		("curriculum", "use 'curriculum' style learning, focusing on easy problems (e.g., shorter sentences) in earlier epochs")
		//-----------------------------------------
		("display", "just display alignments instead of training or decoding")
		//-----------------------------------------
		("treport", value<unsigned>()->default_value(50), "no. of training instances for reporting current model status on training data")
		("dreport", value<unsigned>()->default_value(5000), "no. of training instances for reporting current model status on development data (dreport = N * treport)")
		//-----------------------------------------
		("verbose,v", "be extremely chatty")
		//-----------------------------------------
		("debug", "enable/disable simpler debugging by immediate computing mode or checking validity (refers to http://dynet.readthedocs.io/en/latest/debugging.html)")// for CPU only
	;
	
	store(parse_command_line(argc, argv, opts), vm); 
	if (vm.count("config") > 0)
	{
		ifstream config(vm["config"].as<string>().c_str());
		store(parse_config_file(config, opts), vm); 
	}
	notify(vm);

	cerr << endl << "PID=" << ::getpid() << endl;
	cerr << "Command: ";
	for (int i = 0; i < argc; i++){ 
		cerr << argv[i] << " "; 
	} 
	cerr << endl;
	
	if (vm.count("help") 
		|| vm.count("train") != 1
		|| (vm.count("devel") != 1 && !(vm.count("test") == 0 || vm.count("kbest") == 0 || vm.count("fert-stats") == 0)))
	{
		cout << opts << "\n";
		return 1;
	}

	if (vm.count("lstm"))
		return main_body<LSTMBuilder>(vm);
	else if (vm.count("dglstm"))
		return main_body<DGLSTMBuilder>(vm);
	else if (vm.count("gru"))
		return main_body<GRUBuilder>(vm);
	else
		return main_body<SimpleRNNBuilder>(vm);
}

template <class rnn_t>
int main_body(variables_map vm)
{
	DEBUGGING_FLAG = vm.count("debug");

	verbose = vm.count("verbose");

	SLAYERS = vm["slayers"].as<unsigned>();
	TLAYERS = vm["tlayers"].as<unsigned>();  
	ALIGN_DIM = vm["align"].as<unsigned>(); 
	HIDDEN_DIM = vm["hidden"].as<unsigned>(); 

	TREPORT = vm["treport"].as<unsigned>(); 
	DREPORT = vm["dreport"].as<unsigned>(); 
	if (DREPORT % TREPORT != 0) assert("dreport must be divisible by treport.");// to ensure the reporting on development data

	MINIBATCH_SIZE = vm["minibatch_size"].as<unsigned>();

	bool bidir = vm.count("bidirectional");

	string flavour = "RNN";
	if (vm.count("lstm"))
		flavour = "LSTM";
	else if (vm.count("dglstm"))
		flavour = "DGLSTM";
	else if (vm.count("gru"))
		flavour = "GRU";

	// load fixed vocabularies from files if required	
	Load_Vocabs(vm["src_vocab"].as<string>(), vm["trg_vocab"].as<string>());
	kSRC_SOS = sd.convert("<s>");
	kSRC_EOS = sd.convert("</s>");
	kTGT_SOS = td.convert("<s>");
	kTGT_EOS = td.convert("</s>");

	Corpus training, devel, testing;
	vector<string> train_paths = vm["train"].as<vector<string>>();// to handle multiple training data
	if (train_paths.size() > 2) assert("Invalid -t or --train parameter. Only maximum 2 training corpora provided!");	
	//cerr << "Reading training data from " << vm["train"].as<string>() << "...\n";
	//training = Read_Corpus(vm["train"].as<string>(), doco, true, vm["slen_limit"].as<unsigned>(), r2l_target & !swap, vm["eos_padding"].as<unsigned>());
	cerr << endl << "Reading training data from " << train_paths[0] << "...\n";
	training = Read_Corpus(train_paths[0], true, vm["slen_limit"].as<unsigned>(), vm["eos_padding"].as<unsigned>());
	if ("" == vm["src_vocab"].as<string>() 
		&& "" == vm["trg_vocab"].as<string>()) // if not using external vocabularies
	{
		sd.freeze(); // no new word types allowed
		td.freeze(); // no new word types allowed
	}
	if (train_paths.size() == 2)// incremental training
	{
		training.clear();// use the next training corpus instead!	
		cerr << "Reading extra training data from " << train_paths[1] << "...\n";
		training = Read_Corpus(train_paths[1], true/*for training*/, vm["slen_limit"].as<unsigned>(), vm["eos_padding"].as<unsigned>());
		cerr << "Performing incremental training..." << endl;
	}

	// limit the percent of training data to be used
	unsigned train_percent = vm["train_percent"].as<unsigned>();
	if (train_percent < 100 
		&& train_percent > 0)
	{
		cerr << "Only use " << train_percent << "% of " << training.size() << " training instances: ";
		unsigned int rev_pos = train_percent * training.size() / 100;
		training.erase(training.begin() + rev_pos, training.end());
		cerr << training.size() << " instances remaining!" << endl;
	}
	else if (train_percent != 100){
		cerr << "Invalid --train_percent <num> used. <num> must be (0,100]" << endl;
		return EXIT_FAILURE;
	}

	if (DREPORT >= training.size())
		cerr << "WARNING: --dreport <num> (" << DREPORT << ")" << " is too large, <= training data size (" << training.size() << ")" << endl;

	// set up <s>, </s>, <unk> ids
	sd.set_unk("<unk>");
	td.set_unk("<unk>");
	kSRC_UNK = sd.get_unk_id();
	kTGT_UNK = td.get_unk_id();

	// vocabulary sizes
	SRC_VOCAB_SIZE = sd.size();
	TGT_VOCAB_SIZE = td.size();

	if (vm.count("devel")) {
		cerr << "Reading dev data from " << vm["devel"].as<string>() << "...\n";
		devel = Read_Corpus(vm["devel"].as<string>(), false/*for development/testing*/, 0/*no limit*/, vm["eos_padding"].as<unsigned>());
	}

	string fname;
	if (vm.count("parameters"))
		fname = vm["parameters"].as<string>();
	else {
		ostringstream os;
		os << "am"
			<< '_' << SLAYERS
			<< '_' << TLAYERS
			<< '_' << HIDDEN_DIM
			<< '_' << ALIGN_DIM
			<< '_' << flavour
			<< "_b" << bidir
			<< "-pid" << getpid() << ".params";
		fname = os.str();
	}
	cerr << "Parameters will be written to: " << fname << endl;

	cerr << endl << "%% Using " << flavour << " recurrent units" << endl;

	// attentional models
	// left-to-right/source-to-target model
	cerr << "Creating/Initializing left-to-right/source-to-target model..." << endl;
	ParameterCollection model;
	AttentionalModel<rnn_t> am(&model, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE
					, SLAYERS, TLAYERS, HIDDEN_DIM, ALIGN_DIM
					, bidir, false, false, false, false, false
					, vm.count("shared_embeddings"));
	am.Set_Dropout(vm["dropout_enc"].as<float>(), vm["dropout_dec"].as<float>());
	if (vm.count("initialise")) Initialise(model, vm["initialise"].as<string>());
	cerr << "Count of model parameters: " << model.parameter_count() << endl;

	// setup SGD trainer
	Trainer* sgd = Create_SGDTrainer(model, vm["sgd_trainer"].as<unsigned>()
					, vm["lr_eta"].as<float>()
					, vm["g_clip_threshold"].as<float>()
					, vm["sparse_updates"].as<bool>());

	if (vm["joint_model"].as<string>() == "wR2L"){
		// right-to-left model
		cerr << "Creating/Initializing right-to-left model..." << endl;
		ParameterCollection model_r2l;
		AttentionalModel<rnn_t> am_r2l(&model_r2l, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE
						, SLAYERS, TLAYERS, HIDDEN_DIM, ALIGN_DIM
						, bidir, false, false, false, false, false
						, vm.count("shared_embeddings"));
		am_r2l.Set_Dropout(vm["dropout_enc"].as<float>(), vm["dropout_dec"].as<float>());
		if (vm.count("initialise_r2l")) Initialise(model_r2l, vm["initialise_r2l"].as<string>());
		cerr << "Count of model parameters: " << model_r2l.parameter_count() << endl;

		// setup SGD trainer
		Trainer* sgd_r2l = Create_SGDTrainer(model_r2l, vm["sgd_trainer"].as<unsigned>()
						, vm["lr_eta"].as<float>()
						, vm["g_clip_threshold"].as<float>()
						, vm["sparse_updates"].as<bool>());

		unsigned lr_epochs = vm["lr_epochs"].as<unsigned>(), lr_patience = vm["lr_patience"].as<unsigned>();
		if (lr_epochs > 0 && lr_patience > 0)
			cerr << "[WARNING] - Conflict on learning rate scheduler; use either lr_epochs or lr_patience!" << endl;

		TrainJointModel_wR2L_Batch(model, model_r2l
					, am, am_r2l
					, *sgd, *sgd_r2l
					, vm["alpha"].as<float>()
					, training, devel
					, fname
					, vm.count("curriculum")
					, vm["epochs"].as<unsigned>(), vm["patience"].as<unsigned>(), lr_epochs, vm["lr_eta_decay"].as<float>(), lr_patience);

		cerr << "Cleaning up..." << endl;
		delete sgd_r2l;
	}
	else if (vm["joint_model"].as<string>() == "wT2S"){
		// target-to-source model
		cerr << "Creating/Initializing target-to-source model..." << endl;
		ParameterCollection model_t2s;
		AttentionalModel<rnn_t> am_t2s(&model_t2s, TGT_VOCAB_SIZE, SRC_VOCAB_SIZE
						, SLAYERS, TLAYERS, HIDDEN_DIM, ALIGN_DIM
						, bidir, false, false, false, false, false
						, vm.count("shared_embeddings"));
		am_t2s.Set_Dropout(vm["dropout_enc"].as<float>(), vm["dropout_dec"].as<float>());
		if (vm.count("initialise_t2s")) Initialise(model_t2s, vm["initialise_t2s"].as<string>());// continued training
		cerr << "Count of model parameters: " << model_t2s.parameter_count() << endl;

		// setup SGD trainers
		Trainer* sgd = Create_SGDTrainer(model, vm["sgd_trainer"].as<unsigned>()
					, vm["lr_eta"].as<float>()
					, vm["g_clip_threshold"].as<float>()
					, vm["sparse_updates"].as<bool>());
		Trainer* sgd_t2s = Create_SGDTrainer(model_t2s, vm["sgd_trainer"].as<unsigned>()
					, vm["lr_eta"].as<float>()
					, vm["g_clip_threshold"].as<float>()
					, vm["sparse_updates"].as<bool>());

		unsigned lr_epochs = vm["lr_epochs"].as<unsigned>(), lr_patience = vm["lr_patience"].as<unsigned>();
		if (lr_epochs > 0 && lr_patience > 0)
			cerr << "[WARNING] - Conflict on learning rate scheduler; use either lr_epochs or lr_patience!" << endl;

		TrainJointModel_wT2S_Batch(model, model_t2s
					, am, am_t2s
					, *sgd, *sgd_t2s
					, vm["alpha"].as<float>()
					, training, devel
					, fname
					, vm.count("curriculum")
					, vm["epochs"].as<unsigned>(), vm["patience"].as<unsigned>(), lr_epochs, vm["lr_eta_decay"].as<float>(), lr_patience);

		cerr << "Cleaning up..." << endl;
		delete sgd_t2s;
	}
	else
		cerr << "Unknown joint training mode!" << endl;
	
	delete sgd;	

	return EXIT_SUCCESS;
}

struct DoubleLength
{
  DoubleLength(const Corpus & cor_) : cor(cor_) { }
  bool operator() (int i1, int i2);
  const Corpus & cor;
};

// --------------------------------------------------------------------------------------------------------------------------------
// The following codes are referenced from lamtram toolkit (https://github.com/neubig/lamtram).
bool DoubleLength::operator() (int i1, int i2) {
  if(std::get<0>(cor[i2]).size() != std::get<0>(cor[i1]).size()) return (std::get<0>(cor[i2]).size() < std::get<0>(cor[i1]).size());
  return (std::get<1>(cor[i2]).size() < std::get<1>(cor[i1]).size());
}

inline size_t Calc_Size(const Sentence & src, const Sentence & trg) {
  return src.size()+trg.size();
}

inline size_t Create_MiniBatches(const Corpus& cor
	, size_t max_size
	, std::vector<std::vector<Sentence> > & train_src_minibatch
	, std::vector<std::vector<Sentence> > & train_trg_minibatch
	, std::vector<size_t> & train_ids_minibatch) 
{
	cerr << endl << "Creating minibatches for training data (using minibatch_size=" << max_size << ")..." << endl;

	train_src_minibatch.clear();
	train_trg_minibatch.clear();

	std::vector<size_t> train_ids(cor.size());
	std::iota(train_ids.begin(), train_ids.end(), 0);
	if(max_size > 1)
		sort(train_ids.begin(), train_ids.end(), DoubleLength(cor));

	std::vector<Sentence> train_src_next;
	std::vector<Sentence> train_trg_next;

	size_t max_len = 0;
	for(size_t i = 0; i < train_ids.size(); i++) {
		max_len = std::max(max_len, Calc_Size(std::get<0>(cor[train_ids[i]]), std::get<1>(cor[train_ids[i]])));
		train_src_next.push_back(std::get<0>(cor[train_ids[i]]));
		train_trg_next.push_back(std::get<1>(cor[train_ids[i]]));

		if((train_trg_next.size()+1) * max_len > max_size) {
			train_src_minibatch.push_back(train_src_next);
			train_src_next.clear();
			train_trg_minibatch.push_back(train_trg_next);
			train_trg_next.clear();
			max_len = 0;
		}
	}

	if(train_trg_next.size()) {
		train_src_minibatch.push_back(train_src_next);
		train_trg_minibatch.push_back(train_trg_next);
	}

	// Create a sentence list for this minibatch
	train_ids_minibatch.resize(train_src_minibatch.size());
	std::iota(train_ids_minibatch.begin(), train_ids_minibatch.end(), 0);

	return train_ids.size();
}

inline size_t Create_MiniBatches(const Corpus& cor
	, size_t max_size
	, std::vector<std::vector<Sentence> > & train_src_minibatch
	, std::vector<std::vector<Sentence> > & train_trg_minibatch
	, std::vector<std::vector<Sentence> > & train_rtrg_minibatch //with reverse target sentences
	, std::vector<size_t> & train_ids_minibatch) 
{
	cerr << endl << "Creating minibatches for training data (using minibatch_size=" << max_size << ")..." << endl;

	train_src_minibatch.clear();
	train_trg_minibatch.clear();
	train_rtrg_minibatch.clear();

	std::vector<size_t> train_ids(cor.size());
	std::iota(train_ids.begin(), train_ids.end(), 0);
	if(max_size > 1)
		sort(train_ids.begin(), train_ids.end(), DoubleLength(cor));

	std::vector<Sentence> train_src_next;
	std::vector<Sentence> train_trg_next, train_rtrg_next;

	size_t max_len = 0;
	for(size_t i = 0; i < train_ids.size(); i++) {
		max_len = std::max(max_len, Calc_Size(std::get<0>(cor[train_ids[i]]), std::get<1>(cor[train_ids[i]])));
		train_src_next.push_back(std::get<0>(cor[train_ids[i]]));
		Sentence trg(std::get<1>(cor[train_ids[i]]));
		train_trg_next.push_back(trg);
		std::reverse(trg.begin() + 1/*BOS*/, trg.end() - 1/*EOS*/);//reverse
		train_rtrg_next.push_back(trg);

		if((train_trg_next.size()+1) * max_len > max_size) {
			train_src_minibatch.push_back(train_src_next);
			train_src_next.clear();
			train_trg_minibatch.push_back(train_trg_next);
			train_trg_next.clear();
			train_rtrg_minibatch.push_back(train_rtrg_next);
			train_rtrg_next.clear();
			max_len = 0;
		}
	}

	if(train_trg_next.size()) {
		train_src_minibatch.push_back(train_src_next);
		train_trg_minibatch.push_back(train_trg_next);
		train_rtrg_minibatch.push_back(train_rtrg_next);
	}

	// Create a sentence list for this minibatch
	train_ids_minibatch.resize(train_src_minibatch.size());
	std::iota(train_ids_minibatch.begin(), train_ids_minibatch.end(), 0);

	return train_ids.size();
}
// --------------------------------------------------------------------------------------------------------------------------------

template <class AM_t>
void TrainJointModel_wR2L_Batch(ParameterCollection &model, ParameterCollection &model_r2l
			, AM_t &am, AM_t &am_r2l
			, Trainer &sgd, Trainer &sgd_r2l
			, float alpha
			, Corpus &training, Corpus &devel
			, string out_file
			, bool curriculum
			, unsigned max_epochs, unsigned patience, unsigned lr_epochs, float lr_eta_decay, unsigned lr_patience)
{
	// Create minibatches
	vector<vector<Sentence> > train_src_minibatch;
	vector<vector<Sentence> > train_trg_minibatch, train_rtrg_minibatch;
	vector<size_t> train_ids_minibatch, dev_ids_minibatch;
	size_t minibatch_size = MINIBATCH_SIZE;
	Create_MiniBatches(training, minibatch_size, train_src_minibatch, train_trg_minibatch, train_rtrg_minibatch, train_ids_minibatch);
  
	double best_loss = 9e+99, best_loss_r2l = 9e+99;
	
	unsigned report_every_i = TREPORT;//50;
	unsigned dev_every_i_reports = DREPORT;//500; 

	// shuffle minibatches
	cerr << endl << "***SHUFFLE\n";
	std::shuffle(train_ids_minibatch.begin(), train_ids_minibatch.end(), *dynet::rndeng);

	unsigned sid = 0, id = 0, last_print = 0;
	MyTimer timer_epoch("completed in"), timer_iteration("completed in");
	unsigned epoch = 0, cpt = 0, cpt_r2l = 0/*count of patience*/;
	while (epoch < max_epochs) {
		ModelStats tstats;

		am.Enable_Dropout();// enable dropout
		am_r2l.Enable_Dropout();// enable dropout
		
		for (unsigned iter = 0; iter < dev_every_i_reports;) {
			if (id == train_ids_minibatch.size()) { 
				//timing
				cerr << "***Epoch " << epoch << " is finished. ";
				timer_epoch.show();

				epoch++;

				id = 0;
				sid = 0;
				last_print = 0;

				// learning rate scheduler 1: after lr_epochs, for every next epoch, the learning rate will be decreased by a factor of eta_decay.
				if (lr_epochs > 0 && epoch >= lr_epochs)
					sgd.learning_rate /= lr_eta_decay; 

				if (epoch >= max_epochs) break;

				// Shuffle the access order
				cerr << "***SHUFFLE\n";
				std::shuffle(train_ids_minibatch.begin(), train_ids_minibatch.end(), *dynet::rndeng);

				timer_epoch.reset();
			}

			// build graph for this instance
			ComputationGraph cg;
			if (DEBUGGING_FLAG){//http://dynet.readthedocs.io/en/latest/debugging.html
				cg.set_immediate_compute(true);
				cg.set_check_validity(true);
			}
	
			ModelStats ctstats;
			std::vector<Expression> v_dist_preds;

			Expression i_xent/*main cross entropy loss*/ = am.BuildGraphWithPreds_Batch(train_src_minibatch[train_ids_minibatch[id]]/*sources*/, train_trg_minibatch[train_ids_minibatch[id]]/*targets*/, cg, ctstats, v_dist_preds);// return the distributional softmax prediction vectors
			Expression i_xent_r2l/*auxiliary cross entropy loss*/ = am_r2l.BuildGraphWithPreds_Batch(train_src_minibatch[train_ids_minibatch[id]]/*sources*/, v_dist_preds/*distributional targets*/, cg);// Note: the targets here are distributional vectors, not a real words from the target			

			// reverse
			Expression i_xent_r2l_r/*main cross entropy loss*/ = am_r2l.BuildGraphWithPreds_Batch(train_src_minibatch[train_ids_minibatch[id]]/*sources*/, train_rtrg_minibatch[train_ids_minibatch[id]]/*targets*/, cg, ctstats, v_dist_preds);// return the distributional softmax prediction vectors
			Expression i_xent_r/*auxiliary cross entropy loss*/ = am.BuildGraphWithPreds_Batch(train_src_minibatch[train_ids_minibatch[id]]/*sources*/, v_dist_preds/*distributional targets*/, cg);// Note: the targets here are distributional vectors, not a real words from the target			

			Expression i_objective = i_xent + i_xent_r2l_r + alpha * (i_xent_r2l + i_xent_r);// joint training objective			

			// perform forward computation for aggregate objective
			cg.forward(i_objective);

			// grab the parts of the objective
			float closs = as_scalar(cg.get_value(i_xent.i));
			float closs_r2l = as_scalar(cg.get_value(i_xent_r2l_r.i));
			if (!is_valid(closs) || !is_valid(closs_r2l)){
				std::cerr << "***Warning***: nan or -nan values occurred!" << std::endl;
				++id;
				continue;
			}

			//cerr << "closs=" << closs << "; closs_r2l=" << closs_r2l << endl;

			tstats.loss += closs;
			tstats.loss_aux += closs_r2l;
			tstats.words_src += ctstats.words_src/2;
			tstats.words_src_unk += ctstats.words_src_unk/2;  
			tstats.words_tgt += ctstats.words_tgt/2;
			tstats.words_tgt_unk += ctstats.words_tgt_unk/2;  

			cg.backward(i_objective);
			sgd.update();
			sgd_r2l.update();

			sid += train_trg_minibatch[train_ids_minibatch[id]].size();
			iter += train_trg_minibatch[train_ids_minibatch[id]].size();

			if (sid / report_every_i != last_print 
					|| iter >= dev_every_i_reports /*|| epoch >= max_epochs*/ 
					|| id + 1 == train_ids_minibatch.size()){
				last_print = sid / report_every_i;

				float elapsed = timer_iteration.elapsed();

				sgd.status();
				cerr << "sents=" << sid << " ";
				cerr /*<< "loss=" << tstats.loss*/ << "src_unks=" << tstats.words_src_unk << " trg_unks=" << tstats.words_tgt_unk << " E=" << (tstats.loss / tstats.words_tgt) << " ppl=" << exp(tstats.loss / tstats.words_tgt) << " E(r2l)=" << (tstats.loss_aux / tstats.words_tgt) << " ppl(r2l)=" << exp(tstats.loss_aux / tstats.words_tgt) << ' ';
				if (alpha > 0.f)
					cerr /*<< "time_elapsed=" << elapsed*/ << "(" << tstats.words_tgt * 4 * 1000 / elapsed << " words/sec)" << endl;  
				else cerr /*<< "time_elapsed=" << elapsed*/ << "(" << tstats.words_tgt * 2 * 1000 / elapsed << " words/sec)" << endl;  
	
				//if (epoch >= max_epochs) break;
			}
			   		 
			++id;
		}

		timer_iteration.reset();

		// show score on dev data?
		// disable dropout for evaluating dev data
		am.Disable_Dropout();
		am_r2l.Disable_Dropout();

		ModelStats dstats, dstats_r2l;
		for (unsigned i = 0; i < devel.size(); ++i) {
			Sentence ssent = std::get<0>(devel[i]), tsent = std::get<1>(devel[i]);
			Sentence tsent_rev(tsent);
			std::reverse(tsent_rev.begin() + 1/*BOS*/, tsent_rev.end() - 1/*EOS*/);

			ComputationGraph cg;
			auto i_xent = am.BuildGraph(ssent, tsent, cg, dstats, nullptr, nullptr, nullptr, nullptr);
			auto i_xent_r2l = am_r2l.BuildGraph(ssent, tsent_rev, cg, dstats_r2l, nullptr, nullptr, nullptr, nullptr);
			dstats.loss += as_scalar(cg.incremental_forward(i_xent));
			dstats_r2l.loss += as_scalar(cg.incremental_forward(i_xent_r2l));
		}
		
		if (dstats.loss < best_loss) {
			best_loss = dstats.loss;
			dynet::save_dynet_model(out_file + "_l2r", &model);// FIXME: use binary streaming instead for saving disk spaces?
			cpt = 0;
		}
		else cpt++;

		if (dstats_r2l.loss < best_loss_r2l) {
			best_loss_r2l = dstats_r2l.loss;
			dynet::save_dynet_model(out_file + "_r2l", &model_r2l);// FIXME: use binary streaming instead for saving disk spaces?
			cpt_r2l = 0;
		}
		else cpt_r2l++;

		cerr << "--------------------------------------------------------------------------------------------------------" << endl;
		cerr << "***DEV [epoch=" << (float)epoch + (float)sid/(float)training.size() << " eta=" << sgd.learning_rate << "]" << " sents=" << devel.size() << " src_unks=" << dstats.words_src_unk << " trg_unks=" << dstats.words_tgt_unk << " E=" << (dstats.loss / dstats.words_tgt) << " ppl=" << exp(dstats.loss / dstats.words_tgt) << " E(r2l)=" << (dstats_r2l.loss / dstats_r2l.words_tgt) << " ppl(r2l)=" << exp(dstats_r2l.loss / dstats_r2l.words_tgt) << ' ';
		if (cpt > 0) cerr << "(l2r model not improved, best ppl on dev so far = " << exp(best_loss / dstats.words_tgt) << ") ";
		if (cpt_r2l > 0) cerr << "(r2l model not improved, best ppl on dev so far = " << exp(best_loss_r2l / dstats_r2l.words_tgt) << ") ";
		timer_iteration.show();

		// learning rate scheduler 2: if the model has not been improved for lr_patience times, decrease the learning rate by lr_eta_decay factor.
		if (lr_patience > 0 && cpt > 0 && cpt % lr_patience == 0){
			cerr << "The model (l2r) has not been improved for " << lr_patience << " times. Decreasing the learning rate..." << endl;
			sgd.learning_rate /= lr_eta_decay;
		}
		if (lr_patience > 0 && cpt_r2l > 0 && cpt_r2l % lr_patience == 0){
			cerr << "The model (r2l) has not been improved for " << lr_patience << " times. Decreasing the learning rate..." << endl;
			sgd_r2l.learning_rate /= lr_eta_decay;
		}

		// another early stopping criterion
		if (patience > 0 && cpt >= patience && cpt_r2l >= patience)
		{
			cerr << "Both models have not been improved for " << patience << " times. Stopping now...!" << endl;
			cerr << "No. of epochs so far: " << epoch << "." << endl;
			cerr << "Best ppl on dev for l2r: " << exp(best_loss / dstats.words_tgt) << endl;
			cerr << "Best ppl on dev for r2l: " << exp(best_loss_r2l / dstats_r2l.words_tgt) << endl;
			cerr << "--------------------------------------------------------------------------------------------------------" << endl;
			break;
		}
		cerr << "--------------------------------------------------------------------------------------------------------" << endl;
		timer_iteration.reset();
	}

	cerr << endl << "Training completed!" << endl;
}

template <class AM_t>
void TrainJointModel_wT2S_Batch(ParameterCollection &model, ParameterCollection &model_t2s
			, AM_t &am, AM_t &am_t2s
			, Trainer &sgd, Trainer &sgd_t2s
			, float alpha
			, Corpus &training, Corpus &devel
			, string out_file
			, bool curriculum
			, unsigned max_epochs, unsigned patience, unsigned lr_epochs, float lr_eta_decay, unsigned lr_patience)
{
	// Create minibatches
	vector<vector<Sentence> > train_src_minibatch, dev_src_minibatch;
	vector<vector<Sentence> > train_trg_minibatch, dev_trg_minibatch;
	vector<size_t> train_ids_minibatch, dev_ids_minibatch;
	size_t minibatch_size = MINIBATCH_SIZE;
	Create_MiniBatches(training, minibatch_size, train_src_minibatch, train_trg_minibatch, train_ids_minibatch);
  
	double best_loss = 9e+99, best_loss_t2s = 9e+99;
	
	unsigned report_every_i = TREPORT;//50;
	unsigned dev_every_i_reports = DREPORT;//500; 

	// shuffle minibatches
	cerr << endl << "***SHUFFLE\n";
	std::shuffle(train_ids_minibatch.begin(), train_ids_minibatch.end(), *dynet::rndeng);

	unsigned sid = 0, id = 0, last_print = 0;
	MyTimer timer_epoch("completed in"), timer_iteration("completed in");
	unsigned epoch = 0, cpt = 0, cpt_t2s = 0/*count of patience*/;
	while (epoch < max_epochs) {
		ModelStats tstats;

		am.Enable_Dropout();// enable dropout
		am_t2s.Enable_Dropout();// enable dropout
		
		for (unsigned iter = 0; iter < dev_every_i_reports;) {
			if (id == train_ids_minibatch.size()) { 
				//timing
				cerr << "***Epoch " << epoch << " is finished. ";
				timer_epoch.show();

				epoch++;

				id = 0;
				sid = 0;
				last_print = 0;

				// learning rate scheduler 1: after lr_epochs, for every next epoch, the learning rate will be decreased by a factor of eta_decay.
				if (lr_epochs > 0 && epoch >= lr_epochs)
					sgd.learning_rate /= lr_eta_decay; 

				if (epoch >= max_epochs) break;

				// Shuffle the access order
				cerr << "***SHUFFLE\n";
				std::shuffle(train_ids_minibatch.begin(), train_ids_minibatch.end(), *dynet::rndeng);

				timer_epoch.reset();
			}

			// build graph for this instance
			ComputationGraph cg;
			if (DEBUGGING_FLAG){//http://dynet.readthedocs.io/en/latest/debugging.html
				cg.set_immediate_compute(true);
				cg.set_check_validity(true);
			}
	
			ModelStats ctstats;
			std::vector<Expression> v_dist_preds;
			Expression i_xent/*main cross entropy loss*/ = am.BuildGraphWithPreds_Batch(train_src_minibatch[train_ids_minibatch[id]]/*sources*/, train_trg_minibatch[train_ids_minibatch[id]]/*targets*/, cg, ctstats, v_dist_preds);// return the distributional softmax prediction vectors
			Expression i_xent_t2s/*auxiliary cross entropy loss*/ = am_t2s.BuildGraphWithPreds_Batch(v_dist_preds/*distributional targets*/, train_src_minibatch[train_ids_minibatch[id]]/*sources*/, cg);// Note: the targets here are distributional vectors, not a real words from the target

			// reverse
			Expression i_xent_t2s_r/*main cross entropy loss*/ = am_t2s.BuildGraphWithPreds_Batch(train_trg_minibatch[train_ids_minibatch[id]]/*targets*/, train_src_minibatch[train_ids_minibatch[id]]/*sources*/, cg, ctstats, v_dist_preds);// return the distributional softmax prediction vectors
			Expression i_xent_r/*auxiliary cross entropy loss*/ = am.BuildGraphWithPreds_Batch(v_dist_preds/*distributional sources*/, train_trg_minibatch[train_ids_minibatch[id]]/*targets*/, cg);// Note: the targets here are distributional vectors, not a real words from the target
				
			Expression i_objective = i_xent + i_xent_t2s_r + alpha * (i_xent_t2s + i_xent_r);// joint training objective

			// perform forward computation for aggregate objective
			cg.forward(i_objective);

			// grab the parts of the objective
			float closs = as_scalar(cg.get_value(i_xent.i));
			float closs_t2s = as_scalar(cg.get_value(i_xent_t2s_r.i));
			//cerr << "closs=" << closs << "; closs_t2s=" << closs_t2s << endl;

			tstats.loss += closs;
			tstats.loss_aux += closs_t2s;
			tstats.words_src += ctstats.words_src/2;
			tstats.words_src_unk += ctstats.words_src_unk/2;  
			tstats.words_tgt += ctstats.words_tgt/2;
			tstats.words_tgt_unk += ctstats.words_tgt_unk/2;  

			cg.backward(i_objective);
			sgd.update();
			sgd_t2s.update();

			sid += train_trg_minibatch[train_ids_minibatch[id]].size();
			iter += train_trg_minibatch[train_ids_minibatch[id]].size();

			if (sid / report_every_i != last_print 
					|| iter >= dev_every_i_reports /*|| epoch >= max_epochs*/ 
					|| id + 1 == train_ids_minibatch.size()){
				last_print = sid / report_every_i;

				float elapsed = timer_iteration.elapsed();

				sgd.status();
				cerr << "sents=" << sid << " ";
				cerr /*<< "loss=" << tstats.loss*/ << "src_unks=" << tstats.words_src_unk << " trg_unks=" << tstats.words_tgt_unk << " E=" << (tstats.loss / tstats.words_tgt) << " ppl=" << exp(tstats.loss / tstats.words_tgt) << " E(t2s)=" << (tstats.loss_aux / tstats.words_tgt) << " ppl(t2s)=" << exp(tstats.loss_aux / tstats.words_tgt) << ' ';
				if (alpha > 0.f)
					cerr /*<< "time_elapsed=" << elapsed*/ << "(" << tstats.words_tgt * 4 * 1000 / elapsed << " words/sec)" << endl;  
				else cerr /*<< "time_elapsed=" << elapsed*/ << "(" << tstats.words_tgt * 2 * 1000 / elapsed << " words/sec)" << endl;  
	
				//if (epoch >= max_epochs) break;
			}
			   		 
			++id;
		}

		timer_iteration.reset();

		// show score on dev data?
		// disable dropout for evaluating dev data
		am.Disable_Dropout();
		am_t2s.Disable_Dropout();

		ModelStats dstats, dstats_t2s;
		for (unsigned i = 0; i < devel.size(); ++i) {
			Sentence ssent = std::get<0>(devel[i]), tsent = std::get<1>(devel[i]);
			
			ComputationGraph cg;
			auto i_xent = am.BuildGraph(ssent, tsent, cg, dstats, nullptr, nullptr, nullptr, nullptr);
			auto i_xent_t2s = am_t2s.BuildGraph(tsent, ssent, cg, dstats_t2s, nullptr, nullptr, nullptr, nullptr);
			dstats.loss += as_scalar(cg.incremental_forward(i_xent));
			dstats_t2s.loss += as_scalar(cg.incremental_forward(i_xent_t2s));
		}
		
		if (dstats.loss < best_loss) {
			best_loss = dstats.loss;
			dynet::save_dynet_model(out_file + "_s2t", &model);// FIXME: use binary streaming instead for saving disk spaces?
			cpt = 0;
		}
		else cpt++;

		if (dstats_t2s.loss < best_loss_t2s) {
			best_loss_t2s = dstats_t2s.loss;
			dynet::save_dynet_model(out_file + "_t2s", &model_t2s);// FIXME: use binary streaming instead for saving disk spaces?
			cpt_t2s = 0;
		}
		else cpt_t2s++;

		cerr << "--------------------------------------------------------------------------------------------------------" << endl;
		cerr << "***DEV [epoch=" << (float)epoch + (float)sid/(float)training.size() << " eta=" << sgd.learning_rate << "]" << " sents=" << devel.size() << " src_unks=" << dstats.words_src_unk << " trg_unks=" << dstats.words_tgt_unk << " E=" << (dstats.loss / dstats.words_tgt) << " ppl=" << exp(dstats.loss / dstats.words_tgt) << " E(t2s)=" << (dstats_t2s.loss / dstats_t2s.words_tgt) << " ppl(t2s)=" << exp(dstats_t2s.loss / dstats_t2s.words_tgt) << ' ';
		if (cpt > 0) cerr << "(s2t model not improved, best ppl on dev so far = " << exp(best_loss / dstats.words_tgt) << ") ";
		if (cpt_t2s > 0) cerr << "(t2s model not improved, best ppl on dev so far = " << exp(best_loss_t2s / dstats_t2s.words_tgt) << ") ";
		timer_iteration.show();

		// learning rate scheduler 2: if the model has not been improved for lr_patience times, decrease the learning rate by lr_eta_decay factor.
		if (lr_patience > 0 && cpt > 0 && cpt % lr_patience == 0){
			cerr << "The model (s2t) has not been improved for " << lr_patience << " times. Decreasing the learning rate..." << endl;
			sgd.learning_rate /= lr_eta_decay;
		}
		if (lr_patience > 0 && cpt_t2s > 0 && cpt_t2s % lr_patience == 0){
			cerr << "The model (t2s) has not been improved for " << lr_patience << " times. Decreasing the learning rate..." << endl;
			sgd_t2s.learning_rate /= lr_eta_decay;
		}

		// another early stopping criterion
		if (patience > 0 && cpt >= patience && cpt_t2s >= patience)
		{
			cerr << "Both models have not been improved for " << patience << "+ times. Stopping now...!" << endl;
			cerr << "No. of epochs so far: " << epoch << "." << endl;
			cerr << "Best ppl on dev for s2t: " << exp(best_loss / dstats.words_tgt) << endl;
			cerr << "Best ppl on dev for t2s: " << exp(best_loss_t2s / dstats_t2s.words_tgt) << endl;
			cerr << "--------------------------------------------------------------------------------------------------------" << endl;
			break;
		}
		cerr << "--------------------------------------------------------------------------------------------------------" << endl;
		timer_iteration.reset();
	}

	cerr << endl << "Training completed!" << endl;
}

Corpus Read_Corpus(const string &filename
			, bool cid
			, unsigned slen
			, unsigned eos_padding)
{
	ifstream in(filename);
	assert(in);

	Corpus corpus;
	string line;
	int lc = 0, stoks = 0, ttoks = 0;
	while (getline(in, line)) {
		Sentence source, target;
		read_sentence_pair(line, source, sd, target, td);

		// constrain sentence length(s)
		if (cid/*train only*/ && slen > 0/*length limit*/){
			if (source.size() > slen || target.size() > slen)
				continue;// ignore this sentence
		}

		// add additional </s> paddings at the end of target
		if (eos_padding > 0)
		{
			for (unsigned i = 0; i < eos_padding; i++){
				target.push_back(td.convert("</s>"));
			}
		}

		corpus.push_back(SentencePair(source, target));

		stoks += source.size();
		ttoks += target.size();

		if ((source.front() != kSRC_SOS && source.back() != kSRC_EOS) ||
				(target.front() != kTGT_SOS && target.back() != kTGT_EOS)) {
			cerr << "Sentence in " << filename << ":" << lc << " didn't start or end with <s>, </s>\n";
			abort();
		}

		++lc;
	}

	if (cid)
		cerr << lc << " lines, " << stoks << " & " << ttoks << " tokens (s & t), " << sd.size() << " & " << td.size() << " types\n";
	else 
		cerr << lc << " lines, " << stoks << " & " << ttoks << " tokens (s & t)\n" ;

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

void Load_Vocabs(const string& src_vocab_file, const string& trg_vocab_file)
{
	if ("" == src_vocab_file || "" == trg_vocab_file) return;

	cerr << endl << "Loading vocabularies from files..." << endl;
	cerr << "Source vocabulary file: " << src_vocab_file << endl;
	cerr << "Target vocabulary file: " << trg_vocab_file << endl;
	ifstream if_src_vocab(src_vocab_file), if_trg_vocab(trg_vocab_file);
	string sword, tword;
	while (getline(if_src_vocab, sword)) sd.convert(sword);
	while (getline(if_trg_vocab, tword)) td.convert(tword);
	
	cerr << "Source vocabluary: " << sd.size() << endl;
	cerr << "Target vocabluary: " << td.size() << endl;

	sd.freeze();
	td.freeze();
}

void Initialise(ParameterCollection &model, const string &filename)
{
	cerr << "Initialising model parameters from file: " << filename << endl;
	//ifstream in(filename, ifstream::in);
	//boost::archive::text_iarchive ia(in);
	//ia >> model;
	dynet::load_dynet_model(filename, &model);// FIXME: use binary streaming instead for saving disk spaces
}

Trainer* Create_SGDTrainer(ParameterCollection& model
				, unsigned sgd_type
				, float lr_eta
				, float g_clip_threshold
				, float sparse_updates){
	Trainer* sgd = nullptr;
	if (sgd_type == 1)
		sgd = new MomentumSGDTrainer(model, lr_eta);
	else if (sgd_type == 2)
		sgd = new AdagradTrainer(model, lr_eta);
	else if (sgd_type == 3)
		sgd = new AdadeltaTrainer(model);
	else if (sgd_type == 4)
		sgd = new AdamTrainer(model, lr_eta);
	else if (sgd_type == 5)
		sgd = new RMSPropTrainer(model, lr_eta);
	else if (sgd_type == 0)//Vanilla SGD trainer
		sgd = new SimpleSGDTrainer(model, lr_eta);
	else
	 	assert("Unknown SGD trainer type! (0: vanilla SGD; 1: momentum SGD; 2: Adagrad; 3: AdaDelta; 4: Adam)");
	sgd->clip_threshold = g_clip_threshold;// * MINIBATCH_SIZE;// use larger gradient clipping threshold if training with mini-batching???
	sgd->sparse_updates_enabled = sparse_updates;
	if (!sgd->sparse_updates_enabled)
		cerr << "Sparse updates for lookup parameter(s) to be disabled!" << endl;

	return sgd;
}
