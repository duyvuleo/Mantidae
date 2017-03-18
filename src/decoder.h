/*
 * decoder.h
 *
 *  Created on: 9 Jan 2017
 *      Author: vhoang2
 */

#ifndef DECODER_H_
#define DECODER_H_

#include "attentional.h"
#include "dict-utils.h"

#include <dynet/dict.h>
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

template <class AM_t>
class Decoder {

public:
	void LoadModel(AM_t *am
		, dynet::Dict* vocab_src
		, dynet::Dict* vocab_trg);
	void Decode(const string& src_sent
			, std::string& trg_sent);

public:
	Decoder();
	virtual ~Decoder();

protected:
	Dict *vocab_src_, *vocab_trg_;// vocabularies
	AM_t* am_;// attentional encoder-decoder object pointer (including encoders, decoder, attention mechanism)
};

template <class AM_t>
RelOptDecoder<AM_t>::RelOptDecoder() : vocab_src_(0), vocab_trg_(0), am_(0), am_r2l_(0)
{
	// TODO Auto-generated constructor stub
}

template <class AM_t>
RelOptDecoder<AM_t>::~RelOptDecoder() {
	// TODO Auto-generated destructor stub
}

template <class AM_t>
void RelOptDecoder<AM_t>::LoadModel(AM_t *am
	, dynet::Dict* vocab_src
	, dynet::Dict* vocab_trg)
{
	am_ = am;
	vocab_src_ = vocab_src;
	vocab_trg_ = vocab_trg;
}
