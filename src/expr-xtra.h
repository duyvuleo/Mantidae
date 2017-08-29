#pragma once

#include "dynet/nodes.h"
#include "dynet/dynet.h"
#include "dynet/training.h"
#include "dynet/timing.h"
#include "dynet/rnn.h"
#include "dynet/gru.h"
#include "dynet/lstm.h"
#include "dynet/dict.h"
#include "dynet/expr.h"

using namespace dynet;

// Chris -- this should be a library function
Expression arange(ComputationGraph &cg, unsigned begin, unsigned end, bool log_transform, std::vector<float> *aux_mem) 
{
	aux_mem->clear();
	for (unsigned i = begin; i < end; ++i) 
		aux_mem->push_back((log_transform) ? log(1.0 + i) : i);
	//return Expression(&cg, cg.add_input(Dim({end-begin}), aux_mem));
	return input(cg, Dim({end-begin}), aux_mem, dynet::default_device);// dynet v2
}

// Chris -- this should be a library function
Expression repeat(ComputationGraph &cg, unsigned num, float value, std::vector<float> *aux_mem) 
{
	aux_mem->clear();
	aux_mem->resize(num, value);
	//return Expression(&cg, cg.add_input(Dim({num}), aux_mem));
	return input(cg, Dim({num}), aux_mem, dynet::default_device);// dynet v2
}

// Chris -- this should be a library function
Expression dither(ComputationGraph &cg, const Expression &expr, float pad_value, std::vector<float> *aux_mem)
{
	const auto& shape = cg.nodes[expr.i]->dim;
	aux_mem->clear();
	aux_mem->resize(shape.cols(), pad_value);
	//Expression padding(&cg, cg.add_input(Dim({shape.cols()}), aux_mem));
	Expression padding = input(cg, Dim({shape.cols()}), aux_mem, dynet::default_device);// dynet v2
	Expression padded = concatenate(std::vector<Expression>({padding, expr, padding}));
	Expression left_shift = pickrange(padded, 2, shape.rows()+2);
	Expression right_shift = pickrange(padded, 0, shape.rows());
	return concatenate_cols(std::vector<Expression>({left_shift, expr, right_shift}));
}

// these expressions can surely be implemented much more efficiently than this
Expression abs(const Expression &expr) 
{
	return rectify(expr) + rectify(-expr); 
}

// binary boolean functions, is it better to use a sigmoid?
Expression eq(const Expression &expr, float value, float epsilon=0.1) 
{
	return min(rectify(expr - (value - epsilon)), rectify(-expr + (value + epsilon))) / epsilon; 
}

Expression geq(const Expression &expr, float value, Expression &one, float epsilon=0.01) 
{
	return min(one, rectify(expr - (value - epsilon)) / epsilon);
		//rectify(1 - rectify(expr - (value - epsilon)));
}

Expression leq(const Expression &expr, float value, Expression &one, float epsilon=0.01) 
{
	return min(one, rectify((value + epsilon) - expr) / epsilon);
	//return rectify(1 - rectify((value + epsilon) - expr));
}
