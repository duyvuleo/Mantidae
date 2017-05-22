#pragma once

#include <memory>
#include <iostream>
#include <sstream>
#include <vector>

#include "dynet/dict.h"

using namespace std;

namespace dynet {

typedef int WordId;
typedef std::vector<WordId> Sentence;

std::vector<std::string> SplitWords(const std::string & str);
Sentence ParseWords(Dict & dict, const std::string & str, bool sent_end);
std::string PrintWords(Dict & dict, const Sentence & sent);
std::string PrintWords(const std::vector<std::string> & sent);
std::vector<std::string> Convert2iStr(Dict & sd, const Sentence & sent, bool smarker = true);
std::string Convert2Str(Dict & sd, const Sentence & sent, bool smarker = true);

vector<string> SplitWords(const std::string & line) {
	std::istringstream in(line);
	std::string word;
	std::vector<std::string> res;
	while(in) {
		in >> word;
		if (!in || word.empty()) break;
		res.push_back(word);
	}
	return res;
}

Sentence ParseWords(Dict & sd, const std::string& line) {
	std::istringstream in(line);
	std::string word;
	std::vector<int> res;
	int unk_ind = sd.convert("<unk>");
	while(in) {
		in >> word;
		if (!in || word.empty()) break;
		//res.push_back(sd.convert(word));
		if (sd.contains(word)) 
			res.push_back(sd.convert(word));
		else
			res.push_back(unk_ind);
	}
	return res;
}

std::string PrintWords(Dict & sd, const Sentence & sent) {
	ostringstream oss;
	if(sent.size())
		oss << sd.convert(sent[0]);
	for(size_t i = 1; i < sent.size(); i++)
		oss << ' ' << sd.convert(sent[i]);
	return oss.str();
}

std::string PrintWords(const std::vector<std::string> & sent) {
	ostringstream oss;
	if(sent.size())
		oss << sent[0];
	for(size_t i = 1; i < sent.size(); i++)
		oss << ' ' << sent[i];
	return oss.str();
}

vector<string> Convert2iStr(Dict & sd, const Sentence & sent, bool smarker) {
	vector<string> ret;
	WordId wid_bos = sd.convert("<s>");
	WordId wid_eos = sd.convert("</s>");
	for(WordId wid : sent) {
		if (smarker || (wid != wid_bos && wid != wid_eos))
			ret.push_back(sd.convert(wid));
	}
	return ret;
}

string Convert2Str(Dict & sd, const Sentence & sent, bool smarker) {
	stringstream ss;
	WordId wid_bos = sd.convert("<s>");
	WordId wid_eos = sd.convert("</s>");
	for(WordId wid : sent) {
		if (smarker || (wid != wid_bos && wid != wid_eos))
			ss << sd.convert(wid) << " ";
	}
	return ss.str();
}

}

