/* This is an implementation of ensemble decoder for Mantidae, adapted from ensemble-decoder.{h,cc} of lamtram toolkit (https://github.com/neubig/lamtram).
 * Adapted by Cong Duy Vu Hoang (duyvuleo@gmail.com, vhoang2@student.unimelb.edu.au) 
 */

#pragma once

#include <dynet/tensor.h>
#include <dynet/dynet.h>

#include <vector>

#include <cfloat>

using namespace dynet;
using namespace std;
using namespace dynet::expr;

typedef vector<int> Sentence;
typedef int WordId;

namespace dynet {

class EnsembleDecoderHyp {
public:
	EnsembleDecoderHyp(float score, const RNNPointer & state, const Sentence & sent, const Sentence & align) :
		score_(score), state_(state), sent_(sent), align_(align) { }

	float GetScore() const { return score_; }
	RNNPointer& GetState() { return state_; }
	const Sentence & GetSentence() const { return sent_; }
	const Sentence & GetAlignment() const { return align_; }

protected:

	float score_;
	RNNPointer state_;
	Sentence sent_;
	Sentence align_;

};

typedef std::shared_ptr<EnsembleDecoderHyp> EnsembleDecoderHypPtr;
inline bool operator<(const EnsembleDecoderHypPtr & lhs, const EnsembleDecoderHypPtr & rhs) {
	if(lhs->GetScore() != rhs->GetScore()) return lhs->GetScore() > rhs->GetScore();
	return lhs->GetSentence() < rhs->GetSentence();
}

template <class AM_t>
class EnsembleDecoder {

public:
	EnsembleDecoder(const std::vector<AM_t*> ams, dynet::Dict* td);
	~EnsembleDecoder() {}

	template <class OutSent, class OutLL, class OutWords>
	void CalcSentLL(const Sentence & sent_src, const OutSent & sent_trg, OutLL & ll, OutWords & words);

	EnsembleDecoderHypPtr Generate(const Sentence & sent_src, dynet::ComputationGraph& cg);
	std::vector<EnsembleDecoderHypPtr> GenerateNbest(const Sentence & sent_src, size_t nbest, dynet::ComputationGraph& cg);

	// Ensemble together probabilities or log probabilities for a single word
	dynet::expr::Expression EnsembleProbs(const std::vector<dynet::expr::Expression> & in, dynet::ComputationGraph & cg);
	dynet::expr::Expression EnsembleLogProbs(const std::vector<dynet::expr::Expression> & in, dynet::ComputationGraph & cg);
	
	float GetWordPen() const { return word_pen_; }
	float GetUnkPen() const { return unk_pen_; }
	std::string GetEnsembleOperation() const { return ensemble_operation_; }
	void SetWordPen(float word_pen) { word_pen_ = word_pen; }
	void SetUnkPen(float unk_pen) { unk_pen_ = unk_pen; }
	void SetEnsembleOperation(const std::string & ensemble_operation) { ensemble_operation_ = ensemble_operation; }

	int GetBeamSize() const { return beam_size_; }
	void SetBeamSize(int beam_size) { beam_size_ = beam_size; }
	int GetSizeLimit() const { return size_limit_; }
	void SetSizeLimit(int size_limit) { size_limit_ = size_limit; }

protected:
	std::vector<AM_t*> ams_;
	dynet::Dict* ptd_;
	float word_pen_;
	float unk_pen_, unk_log_prob_;
	WordId unk_id_;
	int size_limit_;
	int beam_size_;
	std::string ensemble_operation_;
	bool verbose_;
};

template <class AM_t>
EnsembleDecoder<AM_t>::EnsembleDecoder(const std::vector<AM_t*> ams, dynet::Dict* td)
	: word_pen_(0.f), unk_pen_(0.f), size_limit_(2000), beam_size_(1), ensemble_operation_("sum"), verbose_(false) 
{
	if(ams.size() == 0)
		assert("Cannot decode without models!");

	ptd_ = td;
  
	for(auto & am : ams) { 
		ams_.push_back(am);
	}
  
	unk_id_ = ptd_->convert("<unk>");
	unk_log_prob_ = -std::log(ptd_->size());// penalty score for <unk>
}

template <class AM_t>
Expression EnsembleDecoder<AM_t>::EnsembleProbs(const std::vector<Expression> & in, dynet::ComputationGraph & cg) {
	if(in.size() == 1) return in[0];
	return average(in);
}

template <class AM_t>
Expression EnsembleDecoder<AM_t>::EnsembleLogProbs(const std::vector<Expression> & in, dynet::ComputationGraph & cg) {
	if(in.size() == 1) return in[0];
	Expression i_average = average(in);
	return log_softmax({i_average});
}

inline int MaxLen(const Sentence & sent) { return sent.size(); }
inline int MaxLen(const vector<Sentence> & sent) {
	size_t val = 0;
	for (const auto & s : sent){
		val = std::max(val, s.size()); 
	}
	return val;
}

inline int GetWord(const vector<Sentence> & vec, int t) { return vec[0][t]; }
inline int GetWord(const Sentence & vec, int t) { return vec[t]; }

template <class AM_t>
EnsembleDecoderHypPtr EnsembleDecoder<AM_t>::Generate(const Sentence & sent_src
	, dynet::ComputationGraph& cg) 
{
	auto nbest = GenerateNbest(sent_src, 1, cg);
	return (nbest.size() > 0 ? nbest[0] : EnsembleDecoderHypPtr());
}

template <class AM_t>
std::vector<EnsembleDecoderHypPtr> EnsembleDecoder<AM_t>::GenerateNbest(const Sentence & sent_src
	, size_t nbest_size //FIXME: segmentation fault error with nbest_size < 40
	, dynet::ComputationGraph& cg) 
{ 
	// Sentinel symbols
	WordId sos_sym = ptd_->convert("<s>");
	WordId eos_sym = ptd_->convert("</s>");
  
	// Initialize the computation graph including source embedding(s)
	for(auto & am : ams_)
		am->StartNewInstance(sent_src, cg, nullptr);

	// The n-best hypotheses
	vector<EnsembleDecoderHypPtr> nbest;

	// Create the initial hypothesis
	vector<RNNPointer> last_states(beam_size_);
	vector<EnsembleDecoderHypPtr> curr_beam(1, 
	EnsembleDecoderHypPtr(new EnsembleDecoderHyp(0.0, RNNPointer(-1), Sentence(1, sos_sym), Sentence(1, 0))));

	int bid;
	Expression empty_idx;

	size_limit_ = sent_src.size() * 3/*x*/;// not generating target with length "x times" the source length

	// Perform decoding
	for(int sent_len = 0; sent_len <= size_limit_; sent_len++) {
		//if (verbose_) cerr << endl << "t=" << sent_len << endl;

		// This vector will hold the best IDs
		vector<tuple<dynet::real,int,int,int> > next_beam_id(beam_size_+1, tuple<dynet::real,int,int,int>(-DBL_MAX,-1,-1,-1));

		// Go through all the hypothesis IDs
		//cerr << "l=" << sent_len << " - " << "(1) ";
		for(int hypid = 0; hypid < (int)curr_beam.size(); hypid++) {
			EnsembleDecoderHypPtr curr_hyp = curr_beam[hypid];
			const Sentence & sent = curr_beam[hypid]->GetSentence();

			if(sent_len != 0 && *sent.rbegin() == eos_sym) continue;

			// Perform the forward step on all models
			//cerr << "(a,Forward) ";
			vector<Expression> i_softmaxes, i_aligns;
			for(int j : boost::irange(0, (int)ams_.size())){
				//if (verbose_) cerr << "Forward(); state=" << curr_hyp->GetState() << endl;
				i_softmaxes.push_back( ams_[j]->Forward(sent
					, sent_len
					, ensemble_operation_ == "logsum"
					, curr_hyp->GetState()
					, last_states[hypid]
					, cg
					, i_aligns) );		
			}

			// Ensemble and calculate the likelihood
			//cerr << "(b,Ensemble) ";
			Expression i_softmax, i_logprob;
			if(ensemble_operation_ == "sum") {
				i_softmax = EnsembleProbs(i_softmaxes, cg);
				i_logprob = log({i_softmax});
			}
			else if(ensemble_operation_ == "logsum") {
				i_logprob = EnsembleLogProbs(i_softmaxes, cg);
			}
			else
				assert(string("Bad ensembling operation: " + ensemble_operation_).c_str());

			// Get the (log) softmax predictions
			//cerr << "(c,softmax) ";
			vector<dynet::real> softmax = as_vector(cg.incremental_forward(i_logprob));

			// Add the word/unk penalty
			// word penalty
			if(word_pen_ != 0.f) {
				for(size_t i = 0; i < softmax.size(); i++)
					softmax[i] += word_pen_;
			}

			// unk penalty
			if(unk_id_ >= 0) softmax[unk_id_] += unk_pen_ * unk_log_prob_;

			// Find the best aligned source, if any alignments exists
			//cerr << "(d,Align) ";
			WordId best_align = -1;
			if(i_aligns.size() != 0) {
				dynet::expr::Expression ens_align = sum(i_aligns);
				vector<dynet::real> align = as_vector(cg.incremental_forward(ens_align));
				best_align = 0;
				for(size_t aid = 0; aid < align.size(); aid++)
				  if(align[aid] > align[best_align])
					best_align = aid;
			}

			// Find the best IDs in the beam
			//cerr << "(e,ID) ";
			for(int wid = 0; wid < (int)softmax.size(); wid++) {
				dynet::real my_score = curr_hyp->GetScore() + softmax[wid];
				for (bid = beam_size_; bid > 0 && my_score > std::get<0>(next_beam_id[bid-1]); bid--)
					next_beam_id[bid] = next_beam_id[bid-1];
				next_beam_id[bid] = tuple<dynet::real,int,int,int>(my_score, hypid, wid, best_align);
			}
		}

		// Create the new hypotheses
		//cerr << endl << "(2) " << endl;
		vector<EnsembleDecoderHypPtr> next_beam;
		for(int i = 0; i < beam_size_; i++) {
			dynet::real score = std::get<0>(next_beam_id[i]);
			int hypid = std::get<1>(next_beam_id[i]);
			int wid = std::get<2>(next_beam_id[i]);
			int aid = std::get<3>(next_beam_id[i]);

			//if (verbose_) 
			//	cerr << "Adding " << ptd_->convert(wid) << "(" << wid << ")" << " @ beam " << i << ": score=" << std::get<0>(next_beam_id[i]) - curr_beam[hypid]->GetScore() << " state=" << last_states[hypid] << endl;

			if(hypid == -1) break;

			Sentence next_sent = curr_beam[hypid]->GetSentence();
			next_sent.push_back(wid);

			Sentence next_align = curr_beam[hypid]->GetAlignment();
			next_align.push_back(aid);

			EnsembleDecoderHypPtr hyp(new EnsembleDecoderHyp(score, last_states[hypid], next_sent, next_align));

			if(wid == eos_sym && hyp->GetSentence().size() == 2) //as of 26 April 2017: excluding: <s> </s>
				continue;

			if(wid == eos_sym || sent_len == size_limit_)
				nbest.push_back(hyp);

			next_beam.push_back(hyp);
		}

		curr_beam = next_beam;
		//if (verbose_)
		//	cerr << "Current beam size=" << curr_beam.size() << endl;

		// Check if we're done with search
		//cerr << "(3) " << endl;
		if(nbest.size() != 0) {
			sort(nbest.begin(), nbest.end());

			if(nbest.size() > nbest_size) nbest.resize(nbest_size);
			if(nbest.size() == nbest_size && (curr_beam.size() == 0 || (*nbest.rbegin())->GetScore() >= next_beam[0]->GetScore()))
				return nbest;
		}

		//if current beam size is 0, stop!
		if(curr_beam.size() == 0) break;
	}

	if (verbose_) cerr << "WARNING: Generated sentence size exceeded " << size_limit_ << ". Truncating." << endl;

	return nbest;
	// return vector<EnsembleDecoderHypPtr>(0);
}

}
