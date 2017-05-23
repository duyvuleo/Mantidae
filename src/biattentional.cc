#include "biattentional.h"
#include "ensemble-decoder.h"

typedef vector<int> Sentence;
typedef pair<Sentence, Sentence> SentencePair;
typedef vector<SentencePair> Corpus;

template <class rnn_t>
int main_body(variables_map vm);

unsigned TREPORT = 100;
unsigned DREPORT = 50000;

dynet::Dict sd;
dynet::Dict td;

Corpus read_corpus(const string &filename)
{
	ifstream in(filename);
	assert(in);
	Corpus corpus;
	string line;
	int lc = 0, stoks = 0, ttoks = 0;
	while(getline(in, line)) {
		++lc;
		Sentence source, target;
		read_sentence_pair(line, source, sd, target, td);
		corpus.push_back(SentencePair(source, target));
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
		("devel,d", value<string>(), "file containing development sentences.")
		("test,T", value<string>(), "file containing testing sentences (for decoding).")
		("rescore,r", value<string>(), "rescore (source, target) sentence pairs")
		//-----------------------------------------
		("parameters,p", value<string>(), "save best parameters to this file")
		("initialise,i", value<vector<string>>(), "load initial parameters from file")
		//-----------------------------------------
		("slayers", value<unsigned>()->default_value(SLAYERS), "use <num> layers for source RNN components")
		("tlayers", value<unsigned>()->default_value(TLAYERS), "use <num> layers for target RNN components")
		("align,a", value<unsigned>()->default_value(ALIGN_DIM), "use <num> dimensions for alignment projection")
		("hidden,h", value<unsigned>()->default_value(HIDDEN_DIM), "use <num> dimensions for recurrent hidden states")
		//-----------------------------------------
		("dir", value<bool>()->default_value(true), "enable/disable translation direction; source-to-target by default")
		("beam,b", value<unsigned>()->default_value(0), "size of beam in decoding; 0=greedy")
		("nbest_size", value<unsigned>()->default_value(1), "nbest size of translation generation/decoding; 1 by default")
		//-----------------------------------------
	   	("treport", value<unsigned>()->default_value(50), "no. of training instances for reporting training status")
		("dreport", value<unsigned>()->default_value(500), "no. of training instances for reporting training status on development data")
		//-----------------------------------------
		("sgd_trainer", value<unsigned>()->default_value(0), "use specific SGD trainer (0: vanilla SGD; 1: momentum SGD; 2: Adagrad; 3: AdaDelta; 4: Adam)")
		("sparse_updates", value<bool>()->default_value(true), "enable/disable sparse update(s) for lookup parameter(s); true by default")
		//-----------------------------------------
		("gru", "use Gated Recurrent Unit (GRU) for recurrent structure; default RNN")
		("lstm", "use Long Short Term Memory (LSTM) for recurrent structure; default RNN")
		("vlstm", "use Vanilla Long Short Term Memory (VLSTM) for recurrent structure; default RNN")
		("dglstm", "use Depth-Gated Long Short Term Memory (DGLSTM) (Kaisheng et al., 2015; https://arxiv.org/abs/1508.03790) for recurrent structure; default RNN") // FIXME: add this to dynet?
		//-----------------------------------------
		("epochs,e", value<unsigned>()->default_value(20), "maximum number of training epochs")
		//-----------------------------------------
		("lr_eta", value<float>()->default_value(0.01f), "SGD learning rate value (e.g., 0.01 for simple SGD trainer)")
		("lr_eta_decay", value<float>()->default_value(2.0f), "SGD learning rate decay value")
		("lr_epochs", value<unsigned>()->default_value(0), "no. of epochs for starting learning rate annealing (e.g., halving)")
		//-----------------------------------------
		("trace-weight", value<float>()->default_value(0.1f), "indicate weight for trace bonus term")
		//-----------------------------------------
		("embedding_shared", "use word embedding sharing between the models")
		//-----------------------------------------
		("bidirectional", "use bidirectional recurrent hidden states as source embeddings, rather than word embeddings")
		//-----------------------------------------
		("giza", "use GIZA++ style features in attentional components (corresponds to all of the 'gz' options)")
		("gz-position", "use GIZA++ positional index features")
		("gz-markov", "use GIZA++ markov context features")
		("gz-fertility", "use GIZA++ fertility type features")
		("fertility,f", "learn Normal model of word fertility values")
		//-----------------------------------------
		("document,D", "use previous sentence as document context; requires document id prefix in input files")
		//-----------------------------------------
		("curriculum", "use 'curriculum' style learning, focusing on easy problems in earlier epochs")
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
	
	if (vm.count("help") || vm.count("train") != 1 || (vm.count("test") != 1 && vm.count("devel") != 1 && vm.count("rescore") != 1)) {
		cout << opts << "\n";
		return EXIT_FAILURE;
	}

	if (vm.count("lstm"))
		return main_body<LSTMBuilder>(vm);
	else if (vm.count("vlstm"))
		return main_body<VanillaLSTMBuilder>(vm);
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
	kSRC_SOS = sd.convert("<s>");
	kSRC_EOS = sd.convert("</s>");
	kTGT_SOS = td.convert("<s>");
	kTGT_EOS = td.convert("</s>");

	TREPORT = vm["treport"].as<unsigned>(); 
	DREPORT = vm["dreport"].as<unsigned>(); 
	if (DREPORT % TREPORT != 0) assert("dreport must be divisible by treport.");// to ensure the reporting on development data

	typedef vector<int> Sentence;
	typedef pair<Sentence, Sentence> SentencePair;
	vector<SentencePair> training, dev, testing;
	string line;
	cerr << "Reading training data from " << vm["train"].as<string>() << "...\n";
	training = read_corpus(vm["train"].as<string>());
	sd.freeze(); // no new word types allowed
	td.freeze(); // no new word types allowed
	SRC_VOCAB_SIZE = sd.size();
	TGT_VOCAB_SIZE = td.size();

	if (vm.count("devel")) {
		cerr << "Reading dev data from " << vm["devel"].as<string>() << "...\n";
		dev = read_corpus(vm["devel"].as<string>());
	}

	if (vm.count("rescore")) {
		cerr << "Reading test data from " << vm["rescore"].as<string>() << "...\n";
		testing = read_corpus(vm["rescore"].as<string>());
	}

	SLAYERS = vm["slayers"].as<unsigned>();
	TLAYERS = vm["tlayers"].as<unsigned>();  
	ALIGN_DIM = vm["align"].as<unsigned>(); 
	HIDDEN_DIM = vm["hidden"].as<unsigned>(); 

	bool BIDIR = vm.count("bidirectional");

	bool giza = vm.count("giza");
	bool GIZA_P = giza || vm.count("gz-position");
	bool GIZA_M = giza || vm.count("gz-markov");
	bool GIZA_F = giza || vm.count("gz-fertility");
	bool FERT = vm.count("fertility");

	bool doco = vm.count("document");

	string fname;
	if (!vm.count("parameters")) {
		ostringstream os;
		os << "bam"
			<< '_' << SLAYERS
			<< '_' << TLAYERS
			<< '_' << HIDDEN_DIM
			<< '_' << ALIGN_DIM
			<< "_lstm"
			<< "_b" << BIDIR
		<< "_g" << (int)GIZA_P << (int)GIZA_M << (int)GIZA_F
			<< "-pid" << getpid() << ".params";
		fname = os.str();
	}
	else {
		fname = vm["parameters"].as<string>();
	}

	if (!vm.count("initialise")) cerr << "Parameters will be written to: " << fname << endl;

	double best = 9e+99;

	Model model;
	Trainer* sgd = nullptr;
	unsigned sgd_type = vm["sgd_trainer"].as<unsigned>();
	if (sgd_type == 1)
		sgd = new MomentumSGDTrainer(model, vm["lr_eta"].as<float>());
	else if (sgd_type == 2)
		sgd = new AdagradTrainer(model, vm["lr_eta"].as<float>());
	else if (sgd_type == 3)
		sgd = new AdadeltaTrainer(model);
	else if (sgd_type == 4)
		sgd = new AdamTrainer(model, vm["lr_eta"].as<float>());
	else if (sgd_type == 0)//Vanilla SGD trainer
		sgd = new SimpleSGDTrainer(model, vm["lr_eta"].as<float>());
	else
		assert("Unknown SGD trainer type! (0: vanilla SGD; 1: momentum SGD; 2: Adagrad; 3: AdaDelta; 4: Adam)");
	sgd->eta_decay = vm["lr_eta_decay"].as<float>();
	sgd->sparse_updates_enabled = vm["sparse_updates"].as<bool>();
	if (!sgd->sparse_updates_enabled)
		cerr << "Sparse updates for lookup parameter(s) to be disabled!" << endl;

	BiAttentionalModel<rnn_t> bam(&model, BIDIR, GIZA_P, GIZA_M, GIZA_F, doco, FERT, vm["trace-weight"].as<float>(), vm.count("embedding_shared"));

	bool add_fer = false;
	if (vm.count("rescore"))
	{
		bam.Add_Global_Fertility_Params(&model);
		add_fer = true;
	}

	if (vm.count("initialise")) {
		vector<string> init_files = vm["initialise"].as<vector<string>>();
		if (init_files.size() == 1) {
			cerr << "Parameters will be loaded from: " << init_files[0] << endl;

			ifstream in(init_files[0]);
			boost::archive::text_iarchive ia(in);
			ia >> model;
		} else if (init_files.size() == 2) {
			cerr << "initialising from " << init_files[0] << " and " << init_files[1] << endl;
			bam.Initialise(init_files[0], init_files[1], model);
		} else {
			assert(false);
		}
	}

	if (FERT && !add_fer) bam.Add_Global_Fertility_Params(&model);

	// rescoring
	if (vm.count("rescore")) {
		double dloss = 0, dloss_s2t = 0, dloss_t2s = 0, dloss_trace = 0;
		unsigned dchars_s = 0, dchars_t = 0, dchars_tt = 0;
		for (unsigned i = 0; i < testing.size(); ++i) {
			ComputationGraph cg;
			auto idloss = bam.BuildGraph(testing[i].first, testing[i].second, cg);

			dchars_s += testing[i].first.size() - 1;
			dchars_t += testing[i].second.size() - 1;
			dchars_tt += std::max(testing[i].first.size(), testing[i].second.size()) - 1; // max or min?

			dloss += as_scalar(cg.forward(idloss));

			double loss_s2t = as_scalar(cg.get_value(bam.s2t_xent.i));
			double loss_t2s = as_scalar(cg.get_value(bam.t2s_xent.i));
			double loss_trace = as_scalar(cg.get_value(bam.trace_bonus.i));

			dloss_s2t += loss_s2t;
			dloss_t2s += loss_t2s;
			dloss_trace += loss_trace;

			cout << i << " |||";
			for (auto &w: testing[i].first)
				cout << " " << sd.convert(w);
			cout << " |||";
			for (auto &w: testing[i].second)
				cout << " " << td.convert(w);
			cout << " ||| " << (loss_s2t / (testing[i].second.size()-1))
				<< " " << (loss_t2s / (testing[i].first.size()-1))
				<< " " << (loss_trace / (std::max(testing[i].first.size(), testing[i].second.size())-1)) 
				<< std::endl;
		}

		cerr <<endl << " E = " << (dloss / (dchars_s + dchars_t)) << " ppl=" << exp(dloss / (dchars_s + dchars_t)) << ' '; // FIXME: kind of hacky, as trace should be normalised differently
		cerr << " ppl_s=" << exp(dloss_t2s / dchars_s) << ' ';
		cerr << " ppl_t=" << exp(dloss_s2t / dchars_t) << ' ';
		cerr << " trace=" << exp(dloss_trace / dchars_tt) << endl;

		//dynet::cleanup();

		return EXIT_SUCCESS;
	}

	// decoding with either of models
	if (vm.count("test")){
		// use vm["dir"].as<bool>() for choosing either source-to-target or target-to-source models
		// input test file: vm["test"].as<string>()
		// output test file: cout stream

		int lno = 0;

		//std::vector<AttentionalModel<rnn_t>*> v_ams;
		std::vector<std::shared_ptr<AttentionalModel<rnn_t>>> v_ams;
		//AttentionalModel<rnn_t>* pam = nullptr;
		std::shared_ptr<AttentionalModel<rnn_t>> pam(nullptr);
		if (vm["dir"].as<bool>() == true)
			//pam = &bam.s2t_model;
			v_ams.push_back(std::make_shared<AttentionalModel<rnn_t>>(bam.s2t_model));//FIXME: single decoder for now
		else{
			//pam = &bam.t2s_model;
			v_ams.push_back(std::make_shared<AttentionalModel<rnn_t>>(bam.t2s_model));//FIXME: single decoder for now

			//swapping
			std::swap(sd, td);
			std::swap(kSRC_SOS, kTGT_SOS);
			std::swap(kSRC_EOS, kTGT_EOS);
			std::swap(kSRC_UNK, kTGT_UNK);
			std::swap(SRC_VOCAB_SIZE, TGT_VOCAB_SIZE);
		}
		//v_ams.push_back(pam);//FIXME: single decoder for now

		unsigned beam = vm["beam"].as<unsigned>();

		EnsembleDecoder<AttentionalModel<rnn_t>> edec(v_ams, &td);
		edec.SetBeamSize(beam);

		string test_file = vm["test"].as<string>();
		cerr << "Reading test examples from " << test_file << endl;
		ifstream in(test_file);
		assert(in);
		string line;
		Sentence last_source;
		Sentence source;
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
				//target = bam.Beam_Decode(source, cg, beam, td, (doco && num[0] == last_docid) ? &last_source : nullptr);

				// Vu's beam search implementation
				EnsembleDecoderHypPtr trg_hyp = edec.Generate(source, cg);//1-best
				if (trg_hyp.get() == nullptr) {
					target.clear();
					//align.clear();
				} 
				else {
					target = trg_hyp->GetSentence();
					//align = trg_hyp->GetAlignment();
					//str_trg = ConvertWords(*vocab_trg, sent_trg, false);
					//MapWords(str_src, sent_trg, align, mapping, str_trg);
				}
			}
			else
				target = pam->Greedy_Decode(source, cg, td, (doco && num[0] == last_docid) ? &last_source : nullptr);

			bool first = true;
			for (auto &w: target) {
				if (!first) cout << " ";
				cout << td.convert(w);
				first = false;
			}
			cout << endl;

			if (vm.count("verbose")) cerr << "chug " << lno++ << "\r" << flush;

			if (doco) {
				last_source = source;
				last_docid = num[0];
			}

			//break;//for debug only
		}

		// clean up
		//dynet::cleanup();

		return EXIT_SUCCESS;
	}

	// training
	unsigned report_every_i = TREPORT;//50
	unsigned dev_every_i_reports = DREPORT; // 500
	unsigned si = 0;
	vector<unsigned> order(training.size());
	for (unsigned i = 0; i < order.size(); ++i) order[i] = i;
	int report = 0;
	unsigned lines = 0;
	unsigned epochs = vm["epochs"].as<unsigned>(), lr_epochs = vm["lr_epochs"].as<unsigned>();
	Timer timer_epoch("completed in"), timer_iteration("completed in");
	while (sgd->epoch < epochs) {
		double loss = 0, loss_s2t = 0, loss_t2s = 0, loss_trace = 0;
		unsigned chars_s = 0, chars_t = 0, chars_tt = 0;
		for (unsigned i = 0; i < report_every_i; ++i) {
			if (si == training.size()) {
				//timing
				cerr << "***Epoch " << sgd->epoch << " is finished. ";
				timer_epoch.show();

				si = 0;

				if (lr_epochs == 0)
					sgd->update_epoch(); 
				else sgd->update_epoch(1, lr_epochs); // @vhoang2: learning rate annealing (after lr_epochs, for every next epoch, the learning rate will be decreased by a factor of eta_decay.

				if (sgd->epoch >= epochs) break;

				cerr << "**SHUFFLE\n";
				shuffle(order.begin(), order.end(), *rndeng);

				timer_epoch.reset();
			}

			// build graph for this instance
			ComputationGraph cg;
			auto& spair = training[order[si]];
			chars_s += spair.first.size() - 1;
			chars_t += spair.second.size() - 1;
			chars_tt += std::max(spair.first.size(), spair.second.size()) - 1; // max or min?
			++si;

			auto iloss = bam.BuildGraph(spair.first, spair.second, cg);
			loss += as_scalar(cg.forward(iloss));
			loss_s2t += as_scalar(cg.get_value(bam.s2t_xent.i));
			loss_t2s += as_scalar(cg.get_value(bam.t2s_xent.i));
			loss_trace += as_scalar(cg.get_value(bam.trace_bonus.i));
			cg.backward(iloss);
			sgd->update(1.0f);

			++lines;

			if (vm.count("verbose") && (i+1) == report_every_i) {//display the last sentence pair' alignments
				//if (si == 1) {
				//Expression aligns = concatenate({transpose(bam.src_align), bam.tgt_align});
				//cerr << cg.get_value(aligns.i) << "\n";
				bam.s2t_model.Display_ASCII(spair.first, spair.second, cg, bam.s2t_align, sd, td);
				bam.t2s_model.Display_ASCII(spair.second, spair.first, cg, bam.t2s_align, td, sd);
				cerr << "\txent_s2t " << as_scalar(cg.get_value(bam.s2t_xent.i))
					 << "\txent_t2s " << as_scalar(cg.get_value(bam.t2s_xent.i))
					 << "\ttrace " << as_scalar(cg.get_value(bam.trace_bonus.i)) << endl;
			}
		}

		if (sgd->epoch >= epochs) continue;

		sgd->status();
		cerr << " E=" << (loss / (chars_s + chars_t)) << " ppl=" << exp(loss / (chars_s + chars_t)) << ' '; // FIXME: kind of hacky, as trace should be normalised differently
		cerr << " ppl_t2s=" << exp(loss_t2s / chars_s) << ' ';
		cerr << " ppl_s2t=" << exp(loss_s2t / chars_t) << ' ';
		cerr << " trace=" << exp(loss_trace / chars_tt) << ' ';
		timer_iteration.show();
		
		timer_iteration.reset();

		// show score on dev data?
		//report++;
		report += report_every_i;
		if (report % dev_every_i_reports == 0) {
			double dloss = 0, dloss_s2t = 0, dloss_t2s = 0, dloss_trace = 0;
			unsigned dchars_s = 0, dchars_t = 0, dchars_tt = 0;
			for (auto& spair : dev) {
				ComputationGraph cg;
				auto idloss = bam.BuildGraph(spair.first, spair.second, cg);
				dloss += as_scalar(cg.incremental_forward(idloss));
				dloss_s2t += as_scalar(cg.get_value(bam.s2t_xent.i));
				dloss_t2s += as_scalar(cg.get_value(bam.t2s_xent.i));
				dloss_trace += as_scalar(cg.get_value(bam.trace_bonus.i));
				dchars_s += spair.first.size() - 1;
				dchars_t += spair.second.size() - 1;
				dchars_tt += std::max(spair.first.size(), spair.second.size()) - 1;
			}

			if (dloss < best) {
				best = dloss;
				//ofstream out(fname);
				//boost::archive::text_oarchive oa(out);
				//oa << model;
				dynet::save_dynet_model(fname, &model);// FIXME: use binary streaming instead for saving disk spaces
			}

			cerr << "--------------------------------------------------------------------------------------------------------" << endl;
			cerr << "***DEV [epoch=" << (lines / (double)training.size()) << " eta=" << sgd->eta << "]";
			cerr << " sents=" << dev.size() << " E=" << (dloss / (dchars_s + dchars_t)) << " ppl=" << exp(dloss / (dchars_s + dchars_t)) << ' '; // FIXME: kind of hacky, as trace should be normalised differently
			cerr << " ppl_t2s=" << exp(dloss_t2s / dchars_s) << ' ';
			cerr << " ppl_s2t=" << exp(dloss_s2t / dchars_t) << ' ';
			cerr << " trace=" << exp(dloss_trace / dchars_tt) << ' ';
			timer_iteration.show();	
			cerr << "--------------------------------------------------------------------------------------------------------" << endl;
		}

		timer_iteration.reset();
	}

	// cleaning up
	delete sgd;
	//dynet::cleanup();

	return EXIT_SUCCESS;
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


