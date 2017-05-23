#include "rnnlm.h"

#include "math-utils.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>

#include <boost/program_options/parsers.hpp>
#include <boost/program_options/variables_map.hpp>
#include <boost/algorithm/string.hpp>

using namespace std;
using namespace dynet;
using namespace boost::program_options;

unsigned LAYERS = 2;
unsigned INPUT_DIM = 256;
unsigned HIDDEN_DIM = 512;
unsigned BATCH_SIZE = 1;
unsigned VOCAB_SIZE = 0;
unsigned ETA_EPOCH = 0;
unsigned MAX_EPOCH = 20;
dynet::real LAMBDA;
dynet::real ETA = 0.1f;
dynet::real ETA_DECAY = 2;
unsigned TREPORT = 20;
unsigned DREPORT = 200;
double MAX_LOSS = 9e+99;

dynet::Dict d;// vocabulary

bool verbose = false;

typedef std::vector<int> Sentence;
typedef std::vector<Sentence> Corpus;

Corpus Read_Corpus(const string &fp, bool dir=true);

inline size_t CalcSize(const Sentence & src, int trg);
inline void Create_MiniBatches(const Corpus& traincor, size_t max_size,
	std::vector<std::vector<Sentence> > & traincor_minibatch);

template <class RNNLM_t>
void TrainModel(Model &model, RNNLM_t &rnn
	, Corpus &traincor, Corpus &devcor
	, Trainer* sgd
	, const string& param_file, float dropout_p);
template <class RNNLM_t>
void TrainModel_Batch1(Model &model, RNNLM_t &rnn
	, Corpus &traincor, Corpus &devcor
	, Trainer* sgd
	, const string& param_file, float dropout_p);
template <class RNNLM_t>
void TrainModel_Batch2(Model &model, RNNLM_t &rnn
	, Corpus &traincor, Corpus &devcor
	, Trainer* sgd
	, const string& param_file, float dropout_p);

template <class RNNLM_t>
void TestModel(Model &model, RNNLM_t &rnn
	, const Corpus &testcor);

template <class Builder>
int main_body(variables_map vm);

// Sort in descending order of length
struct CompareLen {
	bool operator()(const std::vector<int>& first, const std::vector<int>& second) {
		return first.size() > second.size();
	}
};

int main(int argc, char** argv) {
	dynet::initialize(argc, argv);
	
	// command line processing
	using namespace boost::program_options;
	variables_map vm;
	options_description opts("Allowed options");
	opts.add_options()
		("help", "print help message")
		("train,t", value<string>(), "file containing training sentences")
		("dev,d", value<string>(), "file containing development sentences")
		("test,T", value<string>(), "file containing testing source sentences for computing perplexity scores")
		("lang", value<bool>()->default_value(true), "use source or target language from a corpus, e.g., source ||| target")
		("initialise,l", value<string>(), "load initial parameters from file")
		("parameters,s", value<string>(), "save best parameters to this file")
		("layers", value<unsigned>()->default_value(LAYERS), "use <num> layers for RNN components")
		("hidden_dim,h", value<unsigned>()->default_value(HIDDEN_DIM), "use <num> dimensions for recurrent hidden states")
		("input_dim,i", value<unsigned>()->default_value(INPUT_DIM), "use <num> dimensions for input embedding")
		("batch_size,b", value<unsigned>()->default_value(BATCH_SIZE), "impose mini-batch size for training")
		("gru", "use Gated Recurrent Unit (GRU) for recurrent structure; default RNN")
		("lstm", "use Long Short Term Memory (LSTM) for recurrent structure; default RNN")
		("vlstm", "use Vanilla Long Short Term Memory (VLSTM) for recurrent structure; default RNN")
		("dglstm", "use Depth-Gated Long Short Term Memory (DGLSTM) for recurrent structure; default RNN")
		("dropout", value<float>(), "apply dropout technique (Gal et al., 2015) for RNN")
		("sgd_trainer", value<unsigned>()->default_value(0), "use specific SGD trainer (0: vanilla SGD; 1: momentum SGD; 2: Adagrad; 3: AdaDelta; 4: Adam)")
		("sparse_updates", value<bool>()->default_value(true), "enable/disable sparse update(s) for lookup parameter(s)")
		("r2l_target", "use right-to-left direction for target during training; default not")
		("max_loss", value<double>()->default_value(MAX_LOSS), "impose max loss if possible (for interruped running jobs)")
		("treport", value<unsigned>()->default_value(TREPORT), "impose instance reporting value (during training); default 50")
		("dreport", value<unsigned>()->default_value(DREPORT), "impose instance reporting value (during training) on development/test sets; default 500")
		("eta_epoch", value<unsigned>()->default_value(ETA_EPOCH), "limit number of epochs for scheduling the learning rate, e.g., decreasing by a factor of eta_decay; default 4")
		("max_epoch", value<unsigned>()->default_value(MAX_EPOCH), "limit number of epochs for training; default 20")
		("lambda", value<dynet::real>()->default_value(1e-6), "the L2 regularization coefficient; default 1e-6.")
		("eta", value<dynet::real>()->default_value(ETA), "the learning rate of neural network model; default 0.1")
		("eta_decay", value<dynet::real>()->default_value(ETA_DECAY), "decay value for decreasing the learning rate; default 2.")
		("seed", value<unsigned>()->default_value(1), "value for random initialization; default 1")
		("verbose,v", "be extremely noisy");

	store(parse_command_line(argc, argv, opts), vm);

	notify(vm);

	verbose = vm.count("verbose");
	ETA_EPOCH = vm["eta_epoch"].as<unsigned>();
	MAX_EPOCH = vm["max_epoch"].as<unsigned>();
	MAX_LOSS = vm["max_loss"].as<double>();
	TREPORT = vm["treport"].as<unsigned>();
	DREPORT = vm["dreport"].as<unsigned>();
	LAMBDA = vm["lambda"].as<dynet::real>();
	ETA_DECAY = vm["eta_decay"].as<dynet::real>();
	ETA = vm["eta"].as<dynet::real>();
	LAYERS = vm["layers"].as<unsigned>();
	INPUT_DIM = vm["input_dim"].as<unsigned>();
	HIDDEN_DIM = vm["hidden_dim"].as<unsigned>();

	if (vm.count("help") 
		|| vm.count("train") != 1 
		|| (vm.count("devel") != 1 
			&& !(vm.count("test") == 0))) {
		cout << opts << "\n";
			return 1;
	}

	cerr << "PID=" << ::getpid() << endl;
	
	if (vm.count("lstm"))
		return main_body<LSTMBuilder>(vm);
	else if (vm.count("vlstm"))
		return main_body<VanillaLSTMBuilder>(vm);
	//else if (vm.count("dglstm"))
	//	return main_body<DGLSTMBuilder>(vm);
   	else if (vm.count("gru"))
		return main_body<GRUBuilder>(vm);
   	else
		return main_body<SimpleRNNBuilder>(vm);
}

template <class Builder>
int main_body(variables_map vm)
{
	Corpus traincor, devcor, testcor;
	cerr << "Reading training data from " << vm["train"].as<string>() << "...\n";
	kSOS = d.convert("<s>");
	kEOS = d.convert("</s>");
	kUNK = d.convert("<unk>");
	traincor = Read_Corpus(vm["train"].as<string>(), vm["lang"].as<bool>());
	d.freeze(); // no new word types allowed

	VOCAB_SIZE = d.size();
	
	if (vm.count("dev")) {
		cerr << "Reading development data from " << vm["dev"].as<string>() << "...\n";
		devcor = Read_Corpus(vm["dev"].as<string>(), vm["lang"].as<bool>());
	}

	if (vm.count("test")){
		cerr << "Reading testing data from " << vm["test"].as<string>() << "...\n";
		testcor = Read_Corpus(vm["test"].as<string>(), vm["lang"].as<bool>());
	}

	float dropout_p = 0.0f;
	if (vm.count("dropout"))//apply dropout technique
	{
		dropout_p = vm["dropout"].as<float>();
		cerr << "Applying dropout technique with dropout probability p=" << dropout_p << "..." << endl;
	}

	string param_file = "";
	if (vm.count("parameters"))
		param_file = vm["parameters"].as<string>();
	else if (vm.count("initialise"))
		param_file = vm["initialise"].as<string>();
	else {
		ostringstream os;
		os << "rnnlm" << '_' << LAYERS << '_' << HIDDEN_DIM << '_' << INPUT_DIM << '_' << VOCAB_SIZE  << "_pid" << getpid() << ".params";
		param_file = os.str();
	}
	cerr << "Parameters will be updated to: " << param_file << endl;

	BATCH_SIZE = vm["batch_size"].as<unsigned>();
		
	Model model;
	unsigned sgd_trainer_type = vm["sgd_trainer"].as<unsigned>();
	Trainer* sgd_trainer = nullptr;
	if (sgd_trainer_type == 1)
		sgd_trainer = new MomentumSGDTrainer(model, vm["eta"].as<float>());
	else if (sgd_trainer_type == 2)
		sgd_trainer = new AdagradTrainer(model, vm["eta"].as<float>());
	else if (sgd_trainer_type == 3)
		sgd_trainer = new AdadeltaTrainer(model);
	else if (sgd_trainer_type == 4)
		sgd_trainer = new AdamTrainer(model, vm["eta"].as<float>());
	else if (sgd_trainer_type == 0)//Vanilla SGD trainer
		sgd_trainer = new SimpleSGDTrainer(model, vm["eta"].as<float>());
	else
		assert("Unknown SGD trainer type! (0: vanilla SGD; 1: momentum SGD; 2: Adagrad; 3: AdaDelta; 4: Adam)");
	sgd_trainer->eta_decay = ETA_DECAY;
	sgd_trainer->sparse_updates_enabled = vm["sparse_updates"].as<bool>();
	if (!sgd_trainer->sparse_updates_enabled)
		cerr << "Sparse updates for lookup parameter(s) to be disabled!" << endl;
	sgd_trainer->clip_threshold *= BATCH_SIZE;

	if (vm.count("r2l_target")) cerr << "Using right-to-left direction..." << endl;

	RNNLanguageModel<Builder> lm;
	if (vm.count("initialise"))// load from pre-trained model
		lm.LoadModel(&model, vm["initialise"].as<string>());
	else// create new model
		lm.CreateModel(&model, LAYERS, INPUT_DIM, HIDDEN_DIM, VOCAB_SIZE, vm.count("r2l_target"));

	if (vm.count("test")){
		TestModel(model, lm, testcor);
	}// otherwise, do training.
	else
		TrainModel_Batch2(model, lm, traincor, devcor, sgd_trainer, param_file, dropout_p);
	
	// cleaning up
	delete sgd_trainer;

	return EXIT_SUCCESS;
}


//====================================== FUNCTION IMPLEMENTATION ======================================

template <class RNNLM_t>
void TestModel(Model &model, RNNLM_t &rnn
	, const Corpus &testcor)
{
	rnn.DisableDropout();

	double dloss = 0;
	unsigned dtokens = 0;
	unsigned id = 0;
	unsigned unk_dtokens = 0;
	for (auto& sent: testcor)
	{
		ComputationGraph cg;
		unsigned tokens = 0, unk_tokens = 0;
		Expression i_xent = rnn.BuildLMGraph(sent, tokens, unk_tokens, cg);

		//more score details for each sentence
		double loss = as_scalar(cg.forward(i_xent));
		//int tokens = sent.size() - 1;
		cerr << id++ << "\t" << loss << "\t" << tokens << "\t" << exp(loss / tokens) << endl;

		dloss += loss;
		dtokens += tokens;
		unk_dtokens += unk_tokens;
	}

	cerr << "-------------------------------------------------------------------------" << endl;
	cerr << "***TEST " << "sentences=" << testcor.size() << " unks=" << unk_dtokens << " E=" << (dloss / dtokens) << " ppl=" << exp(dloss / dtokens) << ' ';
	cerr << "\n-------------------------------------------------------------------------\n" << endl;
}

template <class RNNLM_t>
void TrainModel(Model &model, RNNLM_t &rnn
	, Corpus &traincor, Corpus &devcor
	, Trainer* sgd_trainer
	, const string& param_file, float dropout_p)
{
	unsigned report_every_i = TREPORT;
	unsigned devcor_every_i_reports = DREPORT;
	double best = 9e+99;
	
	vector<unsigned> order((traincor.size() + BATCH_SIZE - 1) / BATCH_SIZE);
	for (unsigned i = 0; i < order.size(); ++i) order[i] = i * BATCH_SIZE;

	cerr << "**SHUFFLE\n";
	shuffle(order.begin(), order.end(), *rndeng);

	unsigned si = 0;//order.size();
	Timer timer_epoch("completed in"), timer_iteration("completed in");
	
	int report = 0;
	unsigned lines = 0;
	while (sgd_trainer->epoch < MAX_EPOCH) {
		rnn.EnableDropout(dropout_p);

		double loss = 0;
		unsigned tokens = 0, unk_tokens = 0;
		for (unsigned i = 0; i < report_every_i; ++i, ++si) {
			if (si == order.size()) {
				//timing
				cerr << "***Epoch " << sgd_trainer->epoch << " is finished. ";
				timer_epoch.show();

				si = 0;

				if (ETA_EPOCH == 0)
					sgd_trainer->update_epoch(); 
				else sgd_trainer->update_epoch(1, ETA_EPOCH); // @vhoang2: learning rate annealing (after lr_epochs, for every next epoch, the learning rate will be decreased by a factor of eta_decay.

				if (sgd_trainer->epoch >= MAX_EPOCH) break;

				cerr << "**SHUFFLE\n";
				shuffle(order.begin(), order.end(), *rndeng);

				timer_epoch.reset();
			}
		
			// build graph for this instance
			ComputationGraph cg;
			unsigned c1 = 0, c2 = 0;
			Expression i_xent = rnn.BuildLMGraph(traincor[order[si]], c1/*tokens*/, c2/*unk_tokens*/, cg);
		
			float closs = as_scalar(cg.forward(i_xent));// consume the loss
			if (!is_valid(closs)){
				std::cerr << "***Warning***: nan or -nan values occurred!" << std::endl;
				continue;
			}

			loss += closs;
			tokens += c1;
			unk_tokens += c2;  

			cg.backward(i_xent);
			sgd_trainer->update();			

			lines++;
		}

		if (sgd_trainer->epoch >= MAX_EPOCH) continue;
	
		sgd_trainer->status();
		cerr << "sents=" << lines << " unks=" << unk_tokens << " E=" << (loss / tokens) << " ppl=" << exp(loss / tokens) << ' ';
		double elapsed = timer_iteration.elapsed();		
		cerr << "[time_elapsed=" << elapsed << "(msec)" << " (" << tokens * 1000.f / elapsed << " words/sec)]" << endl;  
		timer_iteration.reset();

		//rnn.RandomSample(); // uncomment this to see some randomly-generated sentences

		// show score on devcor data?
		//report++;		
		report += report_every_i;
		if (report % devcor_every_i_reports == 0) {
			rnn.DisableDropout();
			
			double dloss = 0;
			unsigned dtokens = 0, unk_dtokens = 0;
			for (auto& sent: devcor){
				ComputationGraph cg;
				Expression i_xent = rnn.BuildLMGraph(sent, dtokens, unk_dtokens, cg);
				dloss += as_scalar(cg.forward(i_xent));
			}
		
			if (dloss < best) {
				best = dloss;
				//ofstream out(param_file);
				//boost::archive::text_oarchive oa(out);
				//oa << lm;
				rnn.SaveModel(param_file);
			}
			//else
			//	sgd_trainer->eta *= 0.5;
		
			cerr << "\n***DEV [epoch=" << (lines / (double)traincor.size()) << " eta=" << sgd_trainer->eta << "]" << " sents=" << devcor.size() << " unks=" << unk_dtokens << " E=" << (dloss / dtokens) << " ppl=" << exp(dloss / dtokens) << ' ';
			timer_iteration.show();
			timer_iteration.reset();
		}
	}	

	cerr << endl << "Training completed!" << endl;
}

//Note: This version works increasingly worse when batch_size > 16.
template <class RNNLM_t>
void TrainModel_Batch1(Model &model, RNNLM_t &rnn
	, Corpus &traincor, Corpus &devcor
	, Trainer* sgd_trainer
	, const string& param_file, float dropout_p)
{
	if (BATCH_SIZE == 1){
		TrainModel(model, rnn, traincor, devcor, sgd_trainer, param_file, dropout_p);
		return;
	}		

	// Sort the traincor sentences in descending order of length
	CompareLen comp;
	sort(traincor.begin(), traincor.end(), comp);
	// Pad the sentences in the same batch with EOS so they are the same length
	// This modifies the traincor objective a bit by making it necessary to
	// predict EOS multiple times, but it's easy and not so harmful
	// Potential risk: if there is a very long sentence, computation will be significantly increased!
	for(size_t i = 0; i < traincor.size(); i += BATCH_SIZE)
		for(size_t j = 1; j < BATCH_SIZE; ++j)
			while(traincor[i+j].size() < traincor[i].size())
				traincor[i+j].push_back(kEOS);

	unsigned report_every_i = TREPORT;
	unsigned devcor_every_i_reports = DREPORT;
	double best = 9e+99;
	
	vector<unsigned> order((traincor.size() + BATCH_SIZE - 1) / BATCH_SIZE);
	for (unsigned i = 0; i < order.size(); ++i) order[i] = i * BATCH_SIZE;

	cerr << "**SHUFFLE\n";
	shuffle(order.begin(), order.end(), *rndeng);

	unsigned si = 0;//order.size();
	Timer timer_epoch("completed in"), timer_iteration("completed in");
	
	int report = 0;
	unsigned lines = 0;
	while (sgd_trainer->epoch < MAX_EPOCH) {
		rnn.EnableDropout(dropout_p);

		double loss = 0;
		unsigned tokens = 0, unk_tokens = 0;
		for (unsigned i = 0; i < report_every_i; /*++i,*/ ++si) {
			if (si == order.size()) {
				//timing
				cerr << "***Epoch " << sgd_trainer->epoch << " is finished. ";
				timer_epoch.show();

				si = 0;

				if (ETA_EPOCH == 0)
					sgd_trainer->update_epoch(); 
				else sgd_trainer->update_epoch(1, ETA_EPOCH); // @vhoang2: learning rate annealing (after lr_epochs, for every next epoch, the learning rate will be decreased by a factor of eta_decay.

				if (sgd_trainer->epoch >= MAX_EPOCH) break;

				cerr << "**SHUFFLE\n";
				shuffle(order.begin(), order.end(), *rndeng);

				timer_epoch.reset();
			}
		
			// build graph for this instance
			ComputationGraph cg;
			unsigned bsize = std::min((unsigned)traincor.size()-order[si], BATCH_SIZE); // batch size
			unsigned c1 = 0, c2 = 0;
			Expression i_xent = rnn.BuildLMGraph(traincor, order[si], bsize, c1/*tokens*/, c2/*unk_tokens*/, cg);
		
			float closs = as_scalar(cg.forward(i_xent));// consume the loss
			if (!is_valid(closs)){
				std::cerr << "***Warning***: nan or -nan values occurred!" << std::endl;
				continue;
			}
		
			loss += closs;
			tokens += c1;
			unk_tokens += c2;

			cg.backward(i_xent);
			sgd_trainer->update();

			lines += bsize;
			i += bsize;			
		}

		if (sgd_trainer->epoch >= MAX_EPOCH) continue;
	
		sgd_trainer->status();
		cerr << "sents=" << lines << " unks=" << unk_tokens << " E=" << (loss / tokens) << " ppl=" << exp(loss / tokens) << ' ';
		double elapsed = timer_iteration.elapsed();		
		cerr << "[time_elapsed=" << elapsed << "(msec)" << " (" << tokens * 1000.f / elapsed << " words/sec)]" << endl;  
		timer_iteration.reset();

		//rnn.RandomSample(); // uncomment this to see some randomly-generated sentences

		// show score on devcor data?
		//report++;		
		report += report_every_i;
		if (report % devcor_every_i_reports == 0) {
			rnn.DisableDropout();
			
			double dloss = 0;
			unsigned dtokens = 0, unk_dtokens = 0;
			for (auto& sent: devcor){
				ComputationGraph cg;
				Expression i_xent = rnn.BuildLMGraph(sent, dtokens, unk_dtokens, cg);
				dloss += as_scalar(cg.forward(i_xent));
			}
		
			if (dloss < best) {
				best = dloss;
				//ofstream out(param_file);
				//boost::archive::text_oarchive oa(out);
				//oa << lm;
				rnn.SaveModel(param_file);
			}
			//else
			//	sgd_trainer->eta *= 0.5;
		
			cerr << "\n***DEV [epoch=" << (lines / (double)traincor.size()) << " eta=" << sgd_trainer->eta << "]" << " sents=" << devcor.size() << " unks=" << unk_dtokens << " E=" << (dloss / dtokens) << " ppl=" << exp(dloss / dtokens) << ' ';
			timer_iteration.show();
			timer_iteration.reset();
		}
	}	

	cerr << endl << "Training completed!" << endl;
}

// --------------------------------------------------------------------------------------------------------------------------------
// The following codes are referenced from lamtram toolkit (https://github.com/neubig/lamtram).
struct SingleLength
{
	SingleLength(const vector<Sentence> & v) : vec(v) { }
	inline bool operator() (int i1, int i2)
	{
		return (vec[i2].size() < vec[i1].size());
	}
	const vector<Sentence> & vec;
};

inline size_t CalcSize(const Sentence & src, int trg) {
	return src.size()+1;
}

inline void Create_MiniBatches(const Corpus& traincor,
	size_t max_size,
	std::vector<std::vector<Sentence> > & traincor_minibatch) 
{
	std::vector<int> train_ids(traincor.size());
	std::iota(train_ids.begin(), train_ids.end(), 0);

	if(max_size > 1)
		sort(train_ids.begin(), train_ids.end(), SingleLength(traincor));

	std::vector<Sentence> traincor_next;
	size_t first_size = 0;
	for(size_t i = 0; i < train_ids.size(); i++) {
		if (traincor_next.size() == 0)
			first_size = traincor[train_ids[i]].size();

		traincor_next.push_back(traincor[train_ids[i]]);

		if ((traincor_next.size()+1) * first_size > max_size) {
			traincor_minibatch.push_back(traincor_next);
			traincor_next.clear();
		}
	}
	
	if (traincor_next.size()) traincor_minibatch.push_back(traincor_next);
}
// --------------------------------------------------------------------------------------------------------------------------------

template <class RNNLM_t>
void TrainModel_Batch2(Model &model, RNNLM_t &rnn
	, Corpus &traincor, Corpus &devcor
	, Trainer* sgd_trainer
	, const string& param_file, float dropout_p)
{
	if (BATCH_SIZE == 1){
		TrainModel(model, rnn, traincor, devcor, sgd_trainer, param_file, dropout_p);
		return;
	}	

	// create minibatches
	std::vector<std::vector<Sentence> > train_minibatch;
	size_t minibatch_size = BATCH_SIZE;
	Create_MiniBatches(traincor, minibatch_size, train_minibatch);

	std::vector<int> train_ids_minibatch(train_minibatch.size());
	std::iota(train_ids_minibatch.begin(), train_ids_minibatch.end(), 0);
	cerr << "**SHUFFLE\n";
	std::shuffle(train_ids_minibatch.begin(), train_ids_minibatch.end(), *dynet::rndeng);

	unsigned report_every_i = TREPORT;
	unsigned dev_every_i_reports = DREPORT;
	double best = 9e+99;
	unsigned si = 0, last_print = 0, lines = 0;
	Timer timer_epoch("completed in"), timer_iteration("completed in");
	while (sgd_trainer->epoch < MAX_EPOCH) {
		rnn.EnableDropout(dropout_p);

		double loss = 0;
		unsigned tokens = 0, unk_tokens = 0;
		for (unsigned i = 0; i < dev_every_i_reports; ++si) {
			if (si == train_ids_minibatch.size()) {
				//timing
				cerr << "***Epoch " << sgd_trainer->epoch << " is finished. ";
				timer_epoch.show();

				si = 0;
				last_print = 0;								
				lines = 0;

				if (ETA_EPOCH == 0)
					sgd_trainer->update_epoch(); 
				else sgd_trainer->update_epoch(1, ETA_EPOCH); // @vhoang2: learning rate annealing (after lr_epochs, for every next epoch, the learning rate will be decreased by a factor of eta_decay.

				if (sgd_trainer->epoch >= MAX_EPOCH) break;

				cerr << "**SHUFFLE\n";
				shuffle(train_ids_minibatch.begin(), train_ids_minibatch.end(), *dynet::rndeng);

				timer_epoch.reset();
			}
		
			// build graph for this instance
			ComputationGraph cg;
			unsigned c1 = 0, c2 = 0;// intermediate variables
			Expression i_xent = rnn.BuildLMGraph(train_minibatch[train_ids_minibatch[si]], c1/*tokens*/, c2/*unk_tokens*/, cg);

			cg.forward(i_xent);// forward step
			float closs = as_scalar(cg.get_value(i_xent.i));// consume the loss
			if (!is_valid(closs)){
				std::cerr << "***Warning***: nan or -nan values occurred!" << std::endl;
				continue;
			}

			// update returned values
			loss += closs;
			tokens += c1;
			unk_tokens += c2;

			cg.backward(i_xent);// backward step
			sgd_trainer->update();// SGD update step

			lines += train_minibatch[train_ids_minibatch[si]].size();
			i += train_minibatch[train_ids_minibatch[si]].size();

			if (lines / report_every_i != last_print 
					|| i >= dev_every_i_reports /*|| sgd.epoch >= max_epochs*/ 
					|| si + 1 == train_ids_minibatch.size()){
				last_print = lines / report_every_i;		

				sgd_trainer->status();
				cerr << "sents=" << lines << " unks=" << unk_tokens << " E=" << (loss / tokens) << " ppl=" << exp(loss / tokens) << ' ';
				double elapsed = timer_iteration.elapsed();		
				cerr << "[time_elapsed=" << elapsed << "(msec)" << " (" << tokens * 1000.f / elapsed << " words/sec)]" << endl;  
			}
		}

		timer_iteration.reset();

		if (sgd_trainer->epoch >= MAX_EPOCH) continue;

		//rnn.RandomSample(); // uncomment this to see some randomly-generated sentences

		// show score on devcor data?
		rnn.DisableDropout();
			
		double dloss = 0;
		unsigned dtokens = 0, unk_dtokens = 0;
		for (auto& sent: devcor){
			ComputationGraph cg;
			Expression i_xent = rnn.BuildLMGraph(sent, dtokens, unk_dtokens, cg);
			dloss += as_scalar(cg.forward(i_xent));
		}
		
		if (dloss < best) {
			best = dloss;
			//ofstream out(param_file);
			//boost::archive::text_oarchive oa(out);
			//oa << lm;
			rnn.SaveModel(param_file);
		}
		//else
		//	sgd_trainer->eta *= 0.5;
	
		cerr << "--------------------------------------------------------------------------------------------------------" << endl;
		cerr << "***DEV [epoch=" << sgd_trainer->epoch + lines / (double)traincor.size() << " eta=" << sgd_trainer->eta << "]" << " sents=" << devcor.size() << " unks=" << unk_dtokens << " E=" << (dloss / dtokens) << " ppl=" << exp(dloss / dtokens) << ' ';
		timer_iteration.show();
		cerr << "--------------------------------------------------------------------------------------------------------" << endl;
		timer_iteration.reset();
	}	

	cerr << endl << "Training completed!" << endl;	
}

Corpus Read_Corpus(const string &fp, bool dir)
{
	Corpus cor;

	std::ifstream f(fp);

	string line;
	int lc = 0, toks = 0;
	while (std::getline(f, line))
	{
		// parse the line which can be: source ||| target
		size_t pos_sep = line.find("|||");
		if (pos_sep != std::string::npos){
			if (dir)// source
				line = line.substr(0, pos_sep);
			else// target
				line = line.substr(pos_sep + 3);
		}

		boost::trim(line);
		
		Sentence source;
		source = read_sentence(line, d);
		cor.push_back(Sentence(source));

		toks += source.size();

		if ((source.front() != kSOS && source.back() != kEOS))
		{
			cerr << "Sentence in " << fp << ":" << lc << " didn't start or end with <s> and </s>\n";
			abort();
		}

		lc++;
	}

	cerr << lc << " lines, " << toks << " tokens, " << d.size() << " types\n";

	return cor;
}


