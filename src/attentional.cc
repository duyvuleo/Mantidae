#include "attentional.h"
#include "ensemble-decoder.h"
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

unsigned TREPORT = 100;
unsigned DREPORT = 50000;

dynet::Dict sd;
dynet::Dict td;

bool verbose;

typedef vector<int> Sentence;
//typedef pair<Sentence, Sentence> SentencePair;
typedef tuple<Sentence, Sentence, int> SentencePair; // includes document id (optional)
typedef vector<SentencePair> Corpus;

#define WTF(expression) \
	std::cout << #expression << " has dimensions " << cg.nodes[expression.i]->dim << std::endl;
#define KTHXBYE(expression) \
	std::cout << *cg.get_value(expression.i) << std::endl;

#define LOLCAT(expression) \
	WTF(expression) \
	KTHXBYE(expression) 

void Initialise(Model &model, const string &filename);

inline size_t Calc_Size(const Sentence & src, const Sentence & trg);
inline size_t Create_MiniBatches(const Corpus& cor
	, size_t max_size
	, std::vector<std::vector<Sentence> > & train_src_minibatch
	, std::vector<std::vector<Sentence> > & train_trg_minibatch
	, std::vector<size_t> & train_ids_minibatch);

template <class AM_t>
void TrainModel(Model &model, AM_t &am, Corpus &training, Corpus &devel, 
	Trainer &sgd, string out_file, bool curriculum, int max_epochs, int lr_epochs,
	bool doco, float coverage, bool display, bool fert, float fert_weight);
template <class AM_t>
void TrainModel_Batch(Model &model, AM_t &am, Corpus &training, Corpus &devel, 
	Trainer &sgd, string out_file, bool curriculum, int max_epochs, int lr_epochs,
	bool doco, float coverage, bool display, bool fert, float fert_weight);

template <class AM_t> void Test_Rescore(Model &model
	, AM_t &am
	, Corpus &testing
	, bool doco);
template <class AM_t> void Test_Decode(Model &model
	//, AM_t &am
	, std::vector<std::shared_ptr<AM_t>>& ams
	, std::string test_file
	, bool doco
	, unsigned beam
	, bool r2l_target=false);
template <class AM_t> void Test_Decode_Nbest(Model &model
	//, AM_t &am
	, std::vector<std::shared_ptr<AM_t>>& ams
	, string test_file
	, unsigned beam
	, unsigned nbest_size
	, bool print_nbest_scores
	, bool r2l_target=false);
template <class AM_t> void Test_Kbest_Arcs(Model &model, AM_t &am, string test_file, unsigned top_k);
template <class AM_t> void Show_Fert_Stats(Model &model, AM_t &am, Corpus &devel, bool global_fert);

template <class AM_t> void LoadEnsembleModels(const string& conf_file
	, std::vector<std::shared_ptr<AM_t>>& ams
	, std::vector<std::shared_ptr<Model>>& mods);

const Sentence* GetContext(const Corpus &corpus, unsigned i);

Corpus Read_Corpus(const string &filename, bool doco
		, bool cid=true/*corpus id, 1:train;0:otherwise*/
		, unsigned slen=0, bool r2l_target=false
		, unsigned eos_padding=0
		, bool swap=false);
std::vector<int> Read_Numbered_Sentence(const std::string& line, Dict* sd, std::vector<int> &ids);
void Read_Numbered_Sentence_Pair(const std::string& line, std::vector<int>* s, Dict* sd, std::vector<int>* t, Dict* td, std::vector<int> &ids);

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
		("train,t", value<vector<string>>(), "file containing training sentences, with "
			"each line consisting of source ||| target.")		
		("devel,d", value<string>(), "file containing development sentences.")
		("test,T", value<string>(), "file containing testing sentences")
		("slen_limit", value<unsigned>()->default_value(0), "limit the sentence length (either source or target); no by default")
		//-----------------------------------------
		("rescore,r", "rescore (source, target) pairs in testing, default: translate source only")
		("beam,b", value<unsigned>()->default_value(0), "size of beam in decoding; 0=greedy")
		("nbest_size", value<unsigned>()->default_value(1), "nbest size of translation generation/decoding; 1 by default")
		("print_nbest_scores", "print nbest translations with their scores; no by default")
		("kbest,K", value<string>(), "test on kbest inputs using monolingual Markov model")
		("ensemble_conf", value<string>(), "specify the configuration of different AM models for ensemble decoding")
		//-----------------------------------------
		("minibatch_size", value<unsigned>()->default_value(1), "impose the minibatch size for training (support both GPU and CPU); no by default")
		("dynet-autobatch", value<unsigned>()->default_value(0), "impose the auto-batch mode (support both GPU and CPU); no by default") //--dynet-autobatch 1		
		//-----------------------------------------
		("sgd_trainer", value<unsigned>()->default_value(0), "use specific SGD trainer (0: vanilla SGD; 1: momentum SGD; 2: Adagrad; 3: AdaDelta; 4: Adam; 5: RMSProp; 6: cyclical SGD)")
		("sparse_updates", value<bool>()->default_value(true), "enable/disable sparse update(s) for lookup parameter(s); true by default")
		("g_clip_threshold", value<float>()->default_value(5.f), "use specific gradient clipping threshold (https://arxiv.org/pdf/1211.5063.pdf); 5 by default")
		//-----------------------------------------
		("initialise,i", value<string>(), "load initial parameters from file")
		("parameters,p", value<string>(), "save best parameters to this file")
		//-----------------------------------------
		("eos_padding", value<unsigned>()->default_value(0), "impose the </s> padding for all training target instances; none (0) by default")
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
		("epochs,e", value<int>()->default_value(20), "maximum number of training epochs")
		//-----------------------------------------
		("lr_eta", value<float>()->default_value(0.01f), "SGD learning rate value (e.g., 0.1 for simple SGD trainer or smaller 0.001 for ADAM trainer)")
		("lr_eta_decay", value<float>()->default_value(2.0f), "SGD learning rate decay value")
		//-----------------------------------------
		("lr_epochs", value<int>()->default_value(0), "no. of epochs for starting learning rate annealing (e.g., halving)")
		//-----------------------------------------
		("r2l_target", "use right-to-left direction for target during training; default not")
		//-----------------------------------------
		("gru", "use Gated Recurrent Unit (GRU) for recurrent structure; default RNN")
		("lstm", "use Long Short Term Memory (LSTM) for recurrent structure; default RNN")
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
		("fert-weight", value<float>()->default_value(1.0f), "impose a weight for fertility model, 1.0 by default")
		("fert-stats,F", "display computed fertility values on the development set")
		//-----------------------------------------
		("curriculum", "use 'curriculum' style learning, focusing on easy problems (e.g., shorter sentences) in earlier epochs")
		//-----------------------------------------
		("swap", "swap roles of source and target, i.e., learn p(source|target)")
		("swap_T", "swap roles of source and target on --test or -T, applied for rescoring only")
		//-----------------------------------------
		("document,D", "use previous sentence as document context; requires document id prefix in input files")
		//-----------------------------------------
		("coverage,C", value<float>()->default_value(0.0f), "impose alignment coverage penalty in training, with given coefficient")		
		//-----------------------------------------
		("display", "just display alignments instead of training or decoding")
		//-----------------------------------------
		("treport", value<unsigned>()->default_value(100), "no. of training instances for reporting current model status on training data")
		("dreport", value<unsigned>()->default_value(50000), "no. of training instances for reporting current model status on development data (dreport = N * treport)")
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

	cerr << "PID=" << ::getpid() << endl;
	
	if (vm.count("help") || vm.count("train") != 1 || (vm.count("devel") != 1 && !(vm.count("test") == 0 || vm.count("kbest") == 0 || vm.count("fert-stats") == 0))) {
		cout << opts << "\n";
		return 1;
	}

	if (vm.count("lstm"))
		return main_body<LSTMBuilder>(vm);
	else if (vm.count("vlstm"))
		return main_body<VanillaLSTMBuilder>(vm);
	//else if (vm.count("dglstm"))
		//return main_body<DGLSTMBuilder>(vm);
	else if (vm.count("gru"))
		return main_body<GRUBuilder>(vm);
	else
		return main_body<SimpleRNNBuilder>(vm);
}

template <class rnn_t>
int main_body(variables_map vm)
{
	DEBUGGING_FLAG = vm.count("debug");

	kSRC_SOS = sd.convert("<s>");
	kSRC_EOS = sd.convert("</s>");
	kTGT_SOS = td.convert("<s>");
	kTGT_EOS = td.convert("</s>");

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
	bool giza = vm.count("giza");
	bool giza_pos = giza || vm.count("gz-position");
	bool giza_markov = giza || vm.count("gz-markov");
	bool giza_fert = giza || vm.count("gz-fertility");// local fertility
	bool fert = vm.count("fertility");// global fertility
	bool swap = vm.count("swap");
	bool doco = vm.count("document");
	bool r2l_target = vm.count("r2l_target");
	
	string flavour = "RNN";
	if (vm.count("lstm"))
		flavour = "LSTM";
	else if (vm.count("vlstm"))
		flavour = "VanillaLSTM";
	else if (vm.count("dglstm"))
		flavour = "DGLSTM";
	else if (vm.count("gru"))
		flavour = "GRU";

	Corpus training, devel, testing;
	vector<string> train_paths = vm["train"].as<vector<string>>();// to handle multiple training data
	if (train_paths.size() > 2) assert("Invalid -t or --train parameter. Only maximum 2 training corpora provided!");	
	//cerr << "Reading training data from " << vm["train"].as<string>() << "...\n";
	//training = Read_Corpus(vm["train"].as<string>(), doco, true, vm["slen_limit"].as<unsigned>(), r2l_target & !swap, vm["eos_padding"].as<unsigned>());
	cerr << "Reading training data from " << train_paths[0] << "...\n";
	training = Read_Corpus(train_paths[0], doco, true, vm["slen_limit"].as<unsigned>(), r2l_target & !swap, vm["eos_padding"].as<unsigned>());
	kSRC_UNK = sd.convert("<unk>");// add <unk> if does not exist!
	kTGT_UNK = td.convert("<unk>");
	sd.freeze(); // no new word types allowed
	td.freeze(); // no new word types allowed
	if (train_paths.size() == 2)// incremental training
	{
		training.clear();// use the next training corpus instead!	
		cerr << "Reading extra training data from " << train_paths[1] << "...\n";
		training = Read_Corpus(train_paths[1], doco, true/*for training*/, vm["slen_limit"].as<unsigned>(), r2l_target & !swap, vm["eos_padding"].as<unsigned>());
		cerr << "Performing incremental training..." << endl;
	}

	SRC_VOCAB_SIZE = sd.size();
	TGT_VOCAB_SIZE = td.size();

	if (vm.count("devel")) {
		cerr << "Reading dev data from " << vm["devel"].as<string>() << "...\n";
		devel = Read_Corpus(vm["devel"].as<string>(), doco, false/*for development/testing*/, 0, r2l_target & !swap, vm["eos_padding"].as<unsigned>());
	}

	if (vm.count("test") && vm.count("rescore")) {
		// otherwise "test" file is assumed just to contain source language strings
		cerr << "Reading test examples from " << vm["test"].as<string>() << endl;
		testing = Read_Corpus(vm["test"].as<string>(), doco, false/*for development/testing*/, 0, r2l_target & !swap, 0, vm.count("swap_T"));
	}

	if (swap) {
		cerr << "Swapping role of source and target\n";
		std::swap(sd, td);
		std::swap(kSRC_SOS, kTGT_SOS);
		std::swap(kSRC_EOS, kTGT_EOS);
		std::swap(kSRC_UNK, kTGT_UNK);
		std::swap(SRC_VOCAB_SIZE, TGT_VOCAB_SIZE);

		for (auto &sent: training){
			std::swap(get<0>(sent), get<1>(sent));
			if (r2l_target){
				Sentence &tsent = get<1>(sent);
				std::reverse(tsent.begin() + 1, tsent.end() - 1);
			}
		}
		
		for (auto &sent: devel){
			std::swap(get<0>(sent), get<1>(sent));
			if (r2l_target){
				Sentence &tsent = get<1>(sent);
				std::reverse(tsent.begin() + 1, tsent.end() - 1);
			}
		}
	
		for (auto &sent: testing){
			if (!vm.count("swap_T"))
				std::swap(get<0>(sent), get<1>(sent));
			if (r2l_target){
				Sentence &tsent = get<1>(sent);
				std::reverse(tsent.begin() + 1, tsent.end() - 1);
			}
		}
	}

	if (vm.count("rescore") && !swap && vm.count("swap_T")){
		for (auto &sent: testing)
			std::swap(get<0>(sent), get<1>(sent));			
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
			<< "_g" << (int)giza_pos << (int)giza_markov << (int)giza_fert
			<< "_d" << doco
			<< "-pid" << getpid() << ".params";
		fname = os.str();
	}

	if (!vm.count("test") && !vm.count("kbest")) // training phase
		cerr << "Parameters will be written to: " << fname << endl;

	Model model;
	Trainer* sgd = nullptr;
	if (!vm.count("test") && !vm.count("kbest") && !vm.count("fert-stats")){
		unsigned sgd_type = vm["sgd_trainer"].as<unsigned>();
		if (sgd_type == 1)
			sgd = new MomentumSGDTrainer(model, vm["lr_eta"].as<float>());
		else if (sgd_type == 2)
			sgd = new AdagradTrainer(model, vm["lr_eta"].as<float>());
		else if (sgd_type == 3)
			sgd = new AdadeltaTrainer(model);
		else if (sgd_type == 4)
			sgd = new AdamTrainer(model, vm["lr_eta"].as<float>());
		else if (sgd_type == 5)
			sgd = new RMSPropTrainer(model, vm["lr_eta"].as<float>());
		else if (sgd_type == 6)
			sgd = new CyclicalSGDTrainer(model, vm["lr_eta"].as<float>()/10, vm["lr_eta"].as<float>(), (float)8 * (float)training.size()/MINIBATCH_SIZE , 0.99994f);// FIXME: these hyperparameters are empirically set. Also see the original paper! 
		else if (sgd_type == 0)//Vanilla SGD trainer
			sgd = new SimpleSGDTrainer(model, vm["lr_eta"].as<float>());
		else
	   	assert("Unknown SGD trainer type! (0: vanilla SGD; 1: momentum SGD; 2: Adagrad; 3: AdaDelta; 4: Adam)");
		sgd->eta_decay = vm["lr_eta_decay"].as<float>();
		sgd->clip_threshold = vm["g_clip_threshold"].as<float>();// * MINIBATCH_SIZE;// use larger gradient clipping threshold if training with mini-batching???
		sgd->sparse_updates_enabled = vm["sparse_updates"].as<bool>();
		if (!sgd->sparse_updates_enabled)
			cerr << "Sparse updates for lookup parameter(s) to be disabled!" << endl;
	}

	// FIXME: to support different models with different RNN structures?
	std::vector<std::shared_ptr<AttentionalModel<rnn_t>>> v_ams;
	std::vector<std::shared_ptr<Model>> v_mods;
	std::shared_ptr<AttentionalModel<rnn_t>> pam(nullptr);// for training only
	if (vm.count("ensemble_conf"))//ensemble decoding
		LoadEnsembleModels(vm["ensemble_conf"].as<string>(), v_ams, v_mods);
	else{
		cerr << "%% Using " << flavour << " recurrent units" << endl;
		
		//AttentionalModel<rnn_t> am(&model, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE
		//	, SLAYERS, TLAYERS, HIDDEN_DIM, ALIGN_DIM
		//	, bidir
		//	, giza_pos, giza_markov, giza_fert
		//	, doco
		//	, fert);
		pam.reset(new AttentionalModel<rnn_t>(&model, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE
			, SLAYERS, TLAYERS, HIDDEN_DIM, ALIGN_DIM
			, bidir
			, giza_pos, giza_markov, giza_fert
			, doco
			, fert));

		pam->Set_Dropout(vm["dropout_enc"].as<float>(), vm["dropout_dec"].as<float>());
		
		bool add_fert = false;
		if (vm.count("test") || vm.count("kbest")){
			pam->Add_Global_Fertility_Params(&model, HIDDEN_DIM, bidir);// testing/rescoring phase: add extra global fertility parameters (uninitialized model parameters)
			add_fert = true;
		}

		if (vm.count("initialise")) Initialise(model, vm["initialise"].as<string>());

		if (fert && !add_fert) pam->Add_Global_Fertility_Params(&model, HIDDEN_DIM, bidir);// training phase: add extra global fertility parameters to already-initialized model parameters

		cerr << "Count of model parameters: " << model.parameter_count() << endl << endl;

		v_ams.push_back(pam);
	}

	if (!vm.count("test") && !vm.count("kbest") && !vm.count("fert-stats"))
		TrainModel_Batch(model, *v_ams[0], training, devel, *sgd, fname, vm.count("curriculum"),
		vm["epochs"].as<int>(), vm["lr_epochs"].as<int>()
		, doco, vm["coverage"].as<float>()
		, vm.count("display")
		, fert, vm["fert-weight"].as<float>());
	else if (vm.count("kbest"))
		Test_Kbest_Arcs(model, *v_ams[0], vm["kbest"].as<string>(), vm["topk"].as<unsigned>());
	else if (vm.count("test")) {
		if (vm.count("rescore")){//e.g., compute perplexity scores
			cerr << "Rescoring..." << endl;
			Test_Rescore(model, *v_ams[0], testing, doco);
		}
		else{ // test/decode
			if (vm["nbest_size"].as<unsigned>() > 1){
				cerr <<  vm["nbest_size"].as<unsigned>() << "-best Decoding..." << endl;
				Test_Decode_Nbest(model
					//, *v_ams[0]
					, v_ams
					, vm["test"].as<string>()
					, vm["beam"].as<unsigned>()
					, vm["nbest_size"].as<unsigned>()
					, vm.count("print_nbest_scores")
					, r2l_target);
			}
			else{
				cerr << "Decoding..." << endl;			
				Test_Decode(model
					//, *v_ams[0]
					, v_ams
					, vm["test"].as<string>()
					, doco
					, vm["beam"].as<unsigned>()
					, r2l_target);
			}
		}
	
		cerr << "Decoding completed!" << endl;
	}
	else if (vm.count("fert-stats"))
		Show_Fert_Stats(model, *v_ams[0], devel, vm.count("fertility"));

	cerr << "Cleaning up..." << endl;
	delete sgd;
	//dynet::cleanup();

	return EXIT_SUCCESS;
}

template <class AM_t>
void Test_Rescore(Model &model, AM_t &am, Corpus &testing, bool doco)
{
	//double tloss = 0;
	//int tchars = 0;
	int lno = 0;

	Sentence ssent, tsent;
	int docid;
	ModelStats tstats;
	for (unsigned i = 0; i < testing.size(); ++i) {
		tie(ssent, tsent, docid) = testing[i];

		ComputationGraph cg;
		auto i_xent = am.BuildGraph(ssent, tsent, cg, tstats, nullptr, (doco) ? GetContext(testing, i) : nullptr);

		double loss = as_scalar(cg.forward(i_xent));
		//cout << i << " |||";
		for (auto &w: ssent)
			cout << " " << sd.convert(w);
		cout << " |||";
		for (auto &w: tsent)
			cout << " " << td.convert(w);
		
		cout << " ||| " << (loss / (tsent.size()-1)) << endl;

		tstats.loss += loss;
		tstats.words_tgt += tsent.size() - 1;

		if (verbose)
			cerr << "chug " << lno++ << "\r" << flush;
	}

	if (verbose)
		cout << "\n***TEST E=" << (tstats.loss / tstats.words_tgt) << " ppl=" << exp(tstats.loss / tstats.words_tgt) << ' ' << endl;

	return;
}

template <class AM_t>
//void Test_Decode(Model &model, AM_t &am, string test_file, bool doco, unsigned beam, bool r2l_target)
void Test_Decode(Model &model, std::vector<std::shared_ptr<AM_t>>& ams, string test_file, bool doco, unsigned beam, bool r2l_target)
{
	int lno = 0;

	//EnsembleDecoder<AM_t> edec(std::vector<AM_t*>({&am}), &td);//FIXME: single decoder only
	EnsembleDecoder<AM_t> edec(ams, &td);//FIXME: multiple decoders
	edec.SetBeamSize(beam);

	cerr << "Reading test examples from " << test_file << endl;

	Timer timer_dec("completed in");

	ifstream in(test_file);
	assert(in);

	string line;
	Sentence last_source, source;
	int last_docid = -1;
	while (getline(in, line)) {
		vector<int> num;
		if (doco)
			source = Read_Numbered_Sentence(line, &sd, num);
		else 
			source = read_sentence(line, sd);

		if (source.front() != kSRC_SOS && source.back() != kSRC_EOS) {
			cerr << "Sentence in " << test_file << ":" << lno << " didn't start or end with <s>, </s>\n";
			abort();
		}

		ComputationGraph cg;
		std::vector<int> target;

		if (beam > 0){
			// Trevor's beam search implementation		
			//target = am.Beam_Decode(source, cg, beam, td, (doco && num[0] == last_docid) ? &last_source : nullptr);// ensemble decoding not supported yet!

			// Vu's beam search implementation
			EnsembleDecoderHypPtr trg_hyp = edec.Generate(source, cg);//1-best
			if (trg_hyp.get() == nullptr) {
				target.clear();
				//align.clear();
			} 
			else {
				target = trg_hyp->GetSentence();
				//align = trg_hyp->GetAlignment();
				//str_trg = Convert2iStr(*vocab_trg, sent_trg, false);
				//MapWords(str_src, sent_trg, align, mapping, str_trg);
			}
		}

		if (r2l_target)
			std::reverse(target.begin() + 1, target.end() - 1);

		bool first = true;
		for (auto &w: target) {
			if (!first) cout << " ";
			cout << td.convert(w);
			first = false;
		}
		cout << endl;

		if (verbose) cerr << "chug " << lno << "\r" << flush;

		if (doco) {
			last_source = source;
			last_docid = num[0];
		}

		//break;//for debug only

		lno++;
	}

	double elapsed = timer_dec.elapsed();
	cerr << "Decoding is finished!" << endl;
	cerr << "Decoded " << lno << " sentences, completed in " << elapsed/1000 << "(s)" << endl;
}

template <class AM_t>
void Test_Decode_Nbest(Model &model
		//, AM_t &am
		, std::vector<std::shared_ptr<AM_t>>& ams
		, string test_file
		, unsigned beam
		, unsigned nbest_size
		, bool print_nbest_scores
		, bool r2l_target)
{
	//EnsembleDecoder<AM_t> edec(std::vector<AM_t*>({&am}), &td);//FIXME: single decoder for now!
	EnsembleDecoder<AM_t> edec(ams, &td);//FIXME: multiple decoders
	edec.SetBeamSize(beam);

	cerr << "Reading test examples from " << test_file << endl;
	
	Timer timer_dec("completed in");
	
	ifstream in(test_file);
	assert(in);

	int lno = 0;	
	string line;
	Sentence source;
	while (getline(in, line)) {
		if (verbose)
			cerr << "Decoding sentence " << lno << "..." << endl;	

		source = read_sentence(line, sd);

		if (source.front() != kSRC_SOS && source.back() != kSRC_EOS) {
			cerr << "Sentence in " << test_file << ":" << lno << " didn't start or end with <s>, </s>\n";
			abort();
		}

		ComputationGraph cg;
		std::vector<int> target;
		float score = 0.f;
		if (verbose) cerr << "Executing GenerateNbest(...)... ";
		std::vector<EnsembleDecoderHypPtr> v_trg_hyps = edec.GenerateNbest(source, nbest_size, cg);
		if (verbose) cerr << "Done!" << endl;
		for (auto& trg_hyp : v_trg_hyps){
			if (trg_hyp.get() == nullptr) {
				target.clear();
				//align.clear();
				continue;
			} 
			else {
				target = trg_hyp->GetSentence();
				score = trg_hyp->GetScore();
				//align = trg_hyp->GetAlignment();
				//str_trg = Convert2iStr(*vocab_trg, sent_trg, false);
				//MapWords(str_src, sent_trg, align, mapping, str_trg);
			}

			if (target.size() < 2) continue;// FIXME: <=2, e.g., <s> ... </s>?
		
			if (r2l_target)
		   		std::reverse(target.begin() + 1, target.end() - 1);

			// n-best with Moses's format 
			// <line_number1> ||| source ||| target1 ||| AM=score1 || score1
			// <line_number2> ||| source ||| target2 ||| AM=score2 || score2
			//...

			// follows Moses's nbest file format
			stringstream ss;

			// source text
			ss /*<< lno << " ||| "*/ << line << " ||| ";
		   
			// target text
			bool first = true;
			for (auto &w: target) {
				if (!first) ss << " ";
				ss << td.convert(w);
				first = false;
			}
		
			// score
			if (print_nbest_scores){
				ss << " ||| " << "AM=" << -score / (target.size() - 1) << " ||| " << -score / (target.size() - 1);//normalize by target length, following Moses's N-best format.
			}
		
			ss << endl;

			cout << ss.str();
			
			/*
			// simple format with target1 ||| target2 ||| ...
			stringstream ss;
			bool first = true;
			for (auto &w: target) {
				if (!first) ss << " ";
				ss << td.convert(w);
				first = false;
			}
			ss << " ||| ";
			cout << ss.str();
			*/
		}

		//cout << endl;

		if (verbose)
			cerr << "chug " << lno << "\r" << flush;

		lno++;
	}

	double elapsed = timer_dec.elapsed();
	cerr << "Nbest decoding is finished!" << endl;
	cerr << "Decoded " << lno << " sentences, completed in " << elapsed/1000 << "(s)" << endl;
	
	return;
}

template <class AM_t>
void Test_Kbest_Arcs(Model &model, AM_t &am, string test_file, unsigned top_k)
{
	// only suitable for monolingual setting, of predicting a sentence given preceeding sentence
	cerr << "Reading test examples from " << test_file << endl;
	unsigned lno = 0;
	ifstream in(test_file);
	assert(in);
	string line, last_id;
	const std::string sep = "|||";
	vector<SentencePair> items, last_items;
	last_items.push_back(SentencePair(Sentence({ kSRC_SOS, kSRC_EOS }), Sentence({ kTGT_SOS, kTGT_EOS }), -1));
	unsigned snum = 0;
	unsigned count = 0;

	auto process = [&am, &snum, &last_items, &items, &count]() {
		for (unsigned i = 0; i < last_items.size(); ++i) {
			ComputationGraph cg;
			auto &source = get<0>(last_items[i]);
			am.StartNewInstance(source, cg);

			for (unsigned j = 0; j < items.size(); ++j) {
				std::vector<Expression> errs;
				auto &target = get<1>(items[j]);
				const unsigned tlen = target.size() - 1;
				for (unsigned t = 0; t < tlen; ++t) {
					Expression i_r_t = am.AddInput(target[t], t, cg);
					Expression i_err = pickneglogsoftmax(i_r_t, target[t+1]);
					errs.push_back(i_err);
				}
				Expression i_nerr = sum(errs);
				double loss = as_scalar(cg.incremental_forward(i_nerr));

				//cout << last_last_id << ":" << last_id << " |||";
				//for (auto &w: source) cout << " " << sd.convert(w);
				//cout << " |||";
				//for (auto &w: target) cout << " " << td.convert(w);
				//cout << " ||| " << loss << "\n";
				cout << snum << '\t' << i << '\t' << j << '\t' << loss << '\n';
				++count;
			}
		}
	};

	while (getline(in, line)) {
		Sentence source, target;

		istringstream in(line);
		string id, word;
		in >> id >> word;
		assert(word == sep);
		while(in) {
			in >> word;
			if (word.empty() || word == sep) break;
			source.push_back(sd.convert(word));
			target.push_back(td.convert(word));
		}

		if ((source.front() != kSRC_SOS && source.back() != kSRC_EOS) ||
			(target.front() != kTGT_SOS && target.back() != kTGT_EOS)) {
			cerr << "Sentence in " << test_file << ":" << lno << " didn't start or end with <s>, </s>\n";
			abort();
		}

		if (id != last_id && !items.empty()) {
			if (items.size() > top_k)
			items.resize(top_k);

			process();

			last_items = items;
			last_id = id;
			items.clear();
			snum++;

			if (verbose)
			cerr << "chug " << lno++ << " [" << count << " pairs]\r" << flush;
		}

		last_id = id;
		items.push_back(SentencePair(source, target, -1));
	}
	
	if (!items.empty())
		process();

	return;
}

template <class AM_t> 
void Show_Fert_Stats(Model &model, AM_t &am, Corpus &devel, bool global_fert)
{
	Sentence ssent, tsent;
	int docid;

	if (global_fert) {
		std::cout << "==== FERTILITY ESTIMATES ====\n";
		for (unsigned i = 0; i < devel.size(); ++i) {
			tie(ssent, tsent, docid) = devel[i];
			std::cout << "=== sentence " << i << " (" << docid << ") ===\n";
			am.Display_Fertility(ssent, sd);
		}
	}

	std::cout << "==== EMPIRICAL FERTILITY VALUES ====\n";
	for (unsigned i = 0; i < devel.size(); ++i) {
		tie(ssent, tsent, docid) = devel[i];
		std::cout << "=== sentence " << i << " (" << docid << ") ===\n";
		am.Display_Empirical_Fertility(ssent, tsent, sd);
	}
}

template <class AM_t> void LoadEnsembleModels(const string& conf_file
	, std::vector<std::shared_ptr<AM_t>>& ams
	, std::vector<std::shared_ptr<Model>>& mods)
{
	cerr << "Loading multiple models..." << endl;	

	unsigned nModels = 0;
	ams.clear();

	ifstream inpf(conf_file);
	assert(inpf);

	string line;
	getline(inpf, line);
	nModels = atoi(line.c_str());

	ams.resize(nModels);
	mods.resize(nModels);

	unsigned i = 0;
	while (getline(inpf, line)){
		if ("" == line) break;

		// each line has the format: SLAYERS \t TLAYERS \t HIDDEN_DIM \t ALIGN_DIM \t bidir \t giza_pos \t giza_markov \t giza_fert \t fert
		cerr << "Loading model " << i+1 << "..." << endl;
		stringstream ss(line);
		unsigned sl, tl, hd, ad;
		bool bidir, giza_pos, giza_markov, giza_fert, glo_fert;
		string model_file;
		ss >> sl >> tl >> hd >> ad >> bidir >> giza_pos >> giza_markov >> giza_fert >> glo_fert;
		ss >> model_file;
		mods[i].reset(new Model());
		ams[i].reset(new AM_t(mods[i].get(), SRC_VOCAB_SIZE, TGT_VOCAB_SIZE
			, sl, tl, hd, ad
			, bidir
			, giza_pos, giza_markov, giza_fert
			, false
			, glo_fert));
		ams[i]->Add_Global_Fertility_Params(mods[i].get(), hd, bidir);
		Initialise(*mods[i], model_file);

		cerr << "Done!" << endl;
		cerr << "Count of model parameters: " << mods[i++]->parameter_count() << endl;
	}
	cerr << endl;
}

template <class AM_t>
void TrainModel(Model &model, AM_t &am, Corpus &training, Corpus &devel, 
	Trainer &sgd, string out_file, bool curriculum, int max_epochs, int lr_epochs
		, bool doco, float cov_weight, bool display, bool fert, float fert_weight)
{
	double best_loss = 9e+99;
	
	unsigned report_every_i = TREPORT;//50;
	unsigned dev_every_i_reports = DREPORT;//500; 
	
	unsigned si = 0;//training.size();
	vector<unsigned> order(training.size());
	for (unsigned i = 0; i < order.size(); ++i) order[i] = i;

	vector<vector<unsigned>> order_by_length; 
	const unsigned curriculum_steps = 10;
	if (curriculum) {
		// simple form of curriculum learning: for the first K epochs, use only
		// the shortest examples from the training set. E.g., K=10, then in
		// epoch 0 using the first decile, epoch 1 use the first & second
		// deciles etc. up to the full dataset in k >= 9.
		multimap<size_t, unsigned> lengths;
		for (unsigned i = 0; i < training.size(); ++i) 
			lengths.insert(make_pair(get<0>(training[i]).size(), i));

		order_by_length.resize(curriculum_steps);
		unsigned i = 0;
		for (auto& landi: lengths) {
			for (unsigned k = i * curriculum_steps / lengths.size(); k < curriculum_steps; ++k)  
			order_by_length[k].push_back(landi.second);
			++i;
		}
	}

	unsigned report = 0;
	unsigned lines = 0;
	unsigned epoch = 0;
	Sentence ssent, tsent;
	int docid;

	// FIXME: move this into sep function
	if (display) {
		// only display the alignments instead of training or decoding/rescoring
		for (unsigned i = 0; i < devel.size(); ++i) {
			tie(ssent, tsent, docid) = devel[i];
			ComputationGraph cg;
			Expression alignment;
			ModelStats stats;
			auto i_loss = am.BuildGraph(ssent, tsent, cg, stats, &alignment, (doco) ? GetContext(devel, i) : nullptr);
			cg.forward(i_loss);

			cout << "\n====== SENTENCE " << i << " =========\n";
			am.Display_ASCII(ssent, tsent, cg, alignment, sd, td);
			cout << "\n";

			am.Display_TIKZ(ssent, tsent, cg, alignment, sd, td);
			cout << "\n";
		}

		return;
	}

	cerr << "**SHUFFLE\n";
	shuffle(order.begin(), order.end(), *rndeng);

	Timer timer_epoch("completed in"), timer_iteration("completed in");

	while (sgd.epoch < max_epochs) {
		double cov_penalty = 0.f;
		double loss_fert = 0.f;
		ModelStats tstats;

		am.Enable_Dropout();// enable dropout

		for (unsigned iter = 0; iter < report_every_i; ++iter) {
			if (si == training.size()) {
				//timing
				cerr << "***Epoch " << sgd.epoch << " is finished. ";
				timer_epoch.show();

				si = 0;

				if (lr_epochs == 0)
					sgd.update_epoch(); 
				else sgd.update_epoch(1, lr_epochs); // @vhoang2: learning rate annealing (after lr_epochs, for every next epoch, the learning rate will be decreased by a factor of eta_decay.

				if (sgd.epoch >= max_epochs) break;

				cerr << "**SHUFFLE\n";
				shuffle(order.begin(), order.end(), *rndeng);

				// for curriculum learning
				if (curriculum && epoch < order_by_length.size()) {
					order = order_by_length[epoch++];
					cerr << "Curriculum learning, with " << order.size() << " examples\n";
				} 

				timer_epoch.reset();
			}

			if (verbose && iter+1 == report_every_i) {
				tie(ssent, tsent, docid) = training[order[si % order.size()]];

				ComputationGraph cg;
				cerr << "\nDecoding source, greedy Viterbi: ";
				am.Greedy_Decode(ssent, cg, td, (doco) ? GetContext(training, order[si % order.size()]) : nullptr);
				cerr << "\nDecoding source, sampling: ";
				am.Sample(ssent, cg, td, (doco) ? GetContext(training, order[si % order.size()]) : nullptr);
			}

			// build graph for this instance
			tie(ssent, tsent, docid) = training[order[si % order.size()]];
			ComputationGraph cg;
			if (DEBUGGING_FLAG){// see http://dynet.readthedocs.io/en/latest/debugging.html
				cg.set_immediate_compute(true);
				cg.set_check_validity(true);
			}
			
			++si;

			Expression i_alignment, i_coverage_penalty, i_fertility_nll;
			ModelStats ctstats;
			Expression i_xent = am.BuildGraph(ssent, tsent, cg, ctstats, &i_alignment, 
				(doco) ? GetContext(training, order[si % order.size()]) : nullptr, 
				(cov_weight > 0) ? &i_coverage_penalty : nullptr,
				(fert) ? &i_fertility_nll : nullptr);

			Expression i_objective = i_xent;
			if (cov_weight > 0) 
				i_objective = i_objective + cov_weight * i_coverage_penalty;
			if (fert) 
				i_objective = i_objective + fert_weight * i_fertility_nll;

			// perform forward computation for aggregate objective
			cg.forward(i_objective);

			// grab the parts of the objective
			//tstats.loss += as_scalar(cg.get_value(i_xent.i));
			float closs = as_scalar(cg.get_value(i_xent.i));
			if (!is_valid(closs)){
				std::cerr << "***Warning***: nan or -nan values occurred!" << std::endl;
				continue;
			}

			tstats.loss += closs;
			tstats.words_src += ctstats.words_src;
			tstats.words_src_unk += ctstats.words_src_unk;  
			tstats.words_tgt += ctstats.words_tgt;
			tstats.words_tgt_unk += ctstats.words_tgt_unk;  

			if (cov_weight > 0){
				//cov_penalty += as_scalar(cg.get_value(i_cov_penalty.i));
				closs = as_scalar(cg.get_value(i_coverage_penalty.i));
				if (!is_valid(closs))
					std::cerr << "***Warning***: nan or -nan values occurred!" << std::endl;
				else cov_penalty += closs;
			}
			if (fert){
				//loss_fert += as_scalar(cg.get_value(i_fertility_nll.i));
				closs = as_scalar(cg.get_value(i_fertility_nll.i));
				if (!is_valid(closs))
					std::cerr << "***Warning***: nan or -nan values occurred!" << std::endl;
				else loss_fert += closs;
			}

			cg.backward(i_objective);
			sgd.update();

			++lines;

			if (verbose) {
				cerr << "chug " << iter << "\r" << flush;
				if (iter+1 == report_every_i) {
					// display the alignment
					am.Display_ASCII(ssent, tsent, cg, i_alignment, sd, td);
							cout << "\n";
					am.Display_TIKZ(ssent, tsent, cg, i_alignment, sd, td);
							cout << "\n";
				}
			}
		}

		if (sgd.epoch >= max_epochs) continue;
	
		sgd.status();
		double elapsed = timer_iteration.elapsed();
		cerr << "sents=" << si << " src_unks=" << tstats.words_src_unk << " trg_unks=" << tstats.words_tgt_unk << " E=" << (tstats.loss / tstats.words_tgt) << " ppl=" << exp(tstats.loss / tstats.words_tgt) << ' ';
		if (cov_weight > 0) 
			cerr << "cov_penalty=" << cov_penalty / tstats.words_src << ' ';
		if (fert)
			cerr << "fert_ppl=" << exp(loss_fert / tstats.words_src) << ' ';
		cerr << "[time_elapsed=" << elapsed << "(msec)" << " (" << tstats.words_tgt * 1000 / elapsed << " words/sec)]" << endl;  

		timer_iteration.reset();	

		// show score on dev data?
		report += report_every_i;
		if (report % dev_every_i_reports == 0) {
			am.Disable_Dropout();// disable dropout for evaluating on dev data

			ModelStats dstats;
			for (unsigned i = 0; i < devel.size(); ++i) {
				tie(ssent, tsent, docid) = devel[i];
				ComputationGraph cg;
				auto i_xent = am.BuildGraph(ssent, tsent, cg, dstats, nullptr, (doco) ? GetContext(devel, i) : nullptr, nullptr, nullptr);
				dstats.loss += as_scalar(cg.forward(i_xent));
			}
			if (dstats.loss < best_loss) {
				best_loss = dstats.loss;
				//ofstream out(out_file, ofstream::out);
				//boost::archive::text_oarchive oa(out);
				//oa << model;
				dynet::save_dynet_model(out_file, &model);// FIXME: use binary streaming instead for saving disk spaces
			}
			//else{
			//	sgd.eta *= 0.5;
			//}

			cerr << "--------------------------------------------------------------------------------------------------------" << endl;
			cerr << "***DEV [epoch=" << (lines / (double)training.size()) << " eta=" << sgd.eta << "]" << " sents=" << devel.size() << " src_unks=" << dstats.words_src_unk << " trg_unks=" << dstats.words_tgt_unk << " E=" << (dstats.loss / dstats.words_tgt) << " ppl=" << exp(dstats.loss / dstats.words_tgt) << ' ';
			timer_iteration.show();	
			cerr << "--------------------------------------------------------------------------------------------------------" << endl;
		}

		timer_iteration.reset();
	}

	cerr << endl << "Training completed!" << endl;
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
// --------------------------------------------------------------------------------------------------------------------------------

template <class AM_t>
void TrainModel_Batch(Model &model, AM_t &am, Corpus &training, Corpus &devel, 
	Trainer &sgd, string out_file, bool curriculum, int max_epochs, int lr_epochs
		, bool doco, float cov_weight, bool display, bool fert, float fert_weight)
{
	if (MINIBATCH_SIZE == 1){
		TrainModel(model, am, training, devel, sgd, out_file, curriculum, max_epochs, lr_epochs, doco, cov_weight, display, fert, fert_weight);
		return;
	}
	
	// Create minibatches
	vector<vector<Sentence> > train_src_minibatch, dev_src_minibatch;
	vector<vector<Sentence> > train_trg_minibatch, dev_trg_minibatch;
	vector<size_t> train_ids_minibatch, dev_ids_minibatch;
	size_t minibatch_size = MINIBATCH_SIZE;
	Create_MiniBatches(training, minibatch_size, train_src_minibatch, train_trg_minibatch, train_ids_minibatch);
  
	double best_loss = 9e+99;
	
	unsigned report_every_i = TREPORT;//50;
	unsigned dev_every_i_reports = DREPORT;//500; 

	// FIXME: move this into sep function
	if (display) {
		// only display the alignments instead of training or decoding/rescoring
		for (unsigned i = 0; i < devel.size(); ++i) {
			Sentence ssent, tsent;
			int docid;
			tie(ssent, tsent, docid) = devel[i];

			ComputationGraph cg;
			Expression alignment;
			ModelStats stats;	
			auto iloss = am.BuildGraph(ssent, tsent, cg, stats, &alignment, (doco) ? GetContext(devel, i) : nullptr);

			cg.forward(iloss);

			cout << "\n====== SENTENCE " << i << " =========\n";
			am.Display_ASCII(ssent, tsent, cg, alignment, sd, td);
			cout << "\n";

			am.Display_TIKZ(ssent, tsent, cg, alignment, sd, td);
			cout << "\n";
		}

		return;
	}

	// shuffle minibatches
	cerr << "***SHUFFLE\n";
	std::shuffle(train_ids_minibatch.begin(), train_ids_minibatch.end(), *dynet::rndeng);

	unsigned sid = 0, id = 0, last_print = 0;
	Timer timer_epoch("completed in"), timer_iteration("completed in");
	
	while (sgd.epoch < max_epochs) {
		double cov_penalty = 0;
		double loss_fert = 0;
		ModelStats tstats;

		am.Enable_Dropout();// enable dropout

		for (unsigned iter = 0; iter < dev_every_i_reports;) {
			if (id == train_ids_minibatch.size()) { 
				//timing
				cerr << "***Epoch " << sgd.epoch << " is finished. ";
				timer_epoch.show();

				id = 0;
				sid = 0;
				last_print = 0;

				if (lr_epochs == 0)
					sgd.update_epoch(); 
				else sgd.update_epoch(1, lr_epochs); // @vhoang2: learning rate annealing (after lr_epochs, for every next epoch, the learning rate will be decreased by a factor of eta_decay.

				if (sgd.epoch >= max_epochs) break;

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
	
			Expression i_cov_penalty, i_fertility_nll;
			ModelStats ctstats;
			Expression i_xent = am.BuildGraph_Batch(train_src_minibatch[train_ids_minibatch[id]], train_trg_minibatch[train_ids_minibatch[id]]
				, cg, ctstats
				, (cov_weight > 0) ? &i_cov_penalty : nullptr
				, (fert) ? &i_fertility_nll : nullptr);

			Expression i_objective = i_xent;
			if (cov_weight > 0) 
				i_objective = i_objective + cov_weight * i_cov_penalty;
			if (fert) 
				i_objective = i_objective + fert_weight * i_fertility_nll;

			// perform forward computation for aggregate objective
			cg.forward(i_objective);

			// grab the parts of the objective
			//tstats.loss += as_scalar(cg.get_value(i_xent.i));
			float closs = as_scalar(cg.get_value(i_xent.i));
			if (!is_valid(closs)){
				std::cerr << "***Warning***: nan or -nan values occurred!" << std::endl;
				++id;
				continue;
			}

			tstats.loss += closs;
			tstats.words_src += ctstats.words_src;
			tstats.words_src_unk += ctstats.words_src_unk;  
			tstats.words_tgt += ctstats.words_tgt;
			tstats.words_tgt_unk += ctstats.words_tgt_unk;  

			if (cov_weight > 0){
				//cov_penalty += as_scalar(cg.get_value(i_cov_penalty.i));
				closs = as_scalar(cg.get_value(i_cov_penalty.i));
				if (!is_valid(closs))
					std::cerr << "***Warning***: nan or -nan values occurred!" << std::endl;
				else cov_penalty += closs;
			}
			if (fert){
				//loss_fert += as_scalar(cg.get_value(i_fertility_nll.i));
				closs = as_scalar(cg.get_value(i_fertility_nll.i));
				if (!is_valid(closs))
					std::cerr << "***Warning***: nan or -nan values occurred!" << std::endl;
				else loss_fert += closs;
			}

			cg.backward(i_objective);
			sgd.update();

			sid += train_trg_minibatch[train_ids_minibatch[id]].size();
			iter += train_trg_minibatch[train_ids_minibatch[id]].size();

			if (sid / report_every_i != last_print 
					|| iter >= dev_every_i_reports /*|| sgd.epoch >= max_epochs*/ 
					|| id + 1 == train_ids_minibatch.size()){
				last_print = sid / report_every_i;

				float elapsed = timer_iteration.elapsed();

				sgd.status();
				cerr << "sents=" << sid << " ";
				cerr /*<< "loss=" << tstats.loss*/ << "src_unks=" << tstats.words_src_unk << " trg_unks=" << tstats.words_tgt_unk << " E=" << (tstats.loss / tstats.words_tgt) << " ppl=" << exp(tstats.loss / tstats.words_tgt) << ' ';
				if (cov_weight > 0)
					cerr << "cov=" << cov_penalty / tstats.words_src << ' ';
				if (fert)
					cerr << "fert_ppl=" << exp(loss_fert / tstats.words_src) << ' ';	  
				cerr /*<< "time_elapsed=" << elapsed*/ << "(" << tstats.words_tgt * 1000 / elapsed << " words/sec)" << endl;  

				//if (sgd.epoch >= max_epochs) break;
			}
			   		 
			++id;
		}

		timer_iteration.reset();

		// show score on dev data?
		am.Disable_Dropout();// disable dropout for evaluating dev data
		ModelStats dstats;
		for (unsigned i = 0; i < devel.size(); ++i) {
			Sentence ssent, tsent;
			int docid;
			tie(ssent, tsent, docid) = devel[i];  

			ComputationGraph cg;
			auto i_xent = am.BuildGraph(ssent, tsent, cg, dstats, nullptr, (doco) ? GetContext(devel, i) : nullptr, nullptr, nullptr);
			dstats.loss += as_scalar(cg.forward(i_xent));
		}

		if (dstats.loss < best_loss) {
			best_loss = dstats.loss;
			//ofstream out(out_file, ofstream::out);
			//boost::archive::text_oarchive oa(out);
			//oa << model;
			dynet::save_dynet_model(out_file, &model);// FIXME: use binary streaming instead for saving disk spaces
		}
		//else{
		//	sgd.eta *= 0.5;
		//}

		cerr << "--------------------------------------------------------------------------------------------------------" << endl;
		cerr << "***DEV [epoch=" << (float)sgd.epoch + (float)sid/(float)training.size() << " eta=" << sgd.eta << "]" << " sents=" << devel.size() << " src_unks=" << dstats.words_src_unk << " trg_unks=" << dstats.words_tgt_unk << " E=" << (dstats.loss / dstats.words_tgt) << " ppl=" << exp(dstats.loss / dstats.words_tgt) << ' ';
		timer_iteration.show();
		cerr << "--------------------------------------------------------------------------------------------------------" << endl;

		timer_iteration.reset();
	}

	cerr << endl << "Training completed!" << endl;
}

Corpus Read_Corpus(const string &filename, bool doco, bool cid, unsigned slen, bool r2l_target, unsigned eos_padding, bool swap)
{
	ifstream in(filename);
	assert(in);
	Corpus corpus;
	string line;
	int lc = 0, stoks = 0, ttoks = 0;
	vector<int> identifiers({ -1 });
	while (getline(in, line)) {
		Sentence source, target;
		if (!swap){
			if (doco) 
				Read_Numbered_Sentence_Pair(line, &source, &sd, &target, &td, identifiers);
			else
				read_sentence_pair(line, source, sd, target, td);
		}
		else{
			if (doco) 
				Read_Numbered_Sentence_Pair(line, &source, &td, &target, &sd, identifiers);
			else
				read_sentence_pair(line, source, td, target, sd);
		}

		// reverse the target if required
		if (r2l_target) 
			std::reverse(target.begin() + 1/*BOS*/,target.end() - 1/*EOS*/);

		// constrain sentence length(s)
		if (cid/*train only*/ && slen > 0/*length limit*/){
			if (source.size() > slen || target.size() > slen)
				continue;// ignore this sentence
		}

		// add additional </s> paddings
		if (eos_padding > 0)
		{
			for (unsigned i = 0; i < eos_padding; i++){
				target.push_back(td.convert("</s>"));
			}
		}

		corpus.push_back(SentencePair(source, target, identifiers[0]));

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

void Initialise(Model &model, const string &filename)
{
	cerr << "Initialising model parameters from file: " << filename << endl;
	//ifstream in(filename, ifstream::in);
	//boost::archive::text_iarchive ia(in);
	//ia >> model;
	dynet::load_dynet_model(filename, &model);// FIXME: use binary streaming instead for saving disk spaces
}

const Sentence* GetContext(const Corpus &corpus, unsigned i)
{
	if (i > 0) {
		int docid = get<2>(corpus.at(i));
		int prev_docid = get<2>(corpus.at(i-1));
		if (docid == prev_docid) 
			return &get<0>(corpus.at(i-1));
	} 

	return nullptr;
}
