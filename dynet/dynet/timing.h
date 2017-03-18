#ifndef _TIMING_H_
#define _TIMING_H_

#include <iostream>
#include <string>
#include <chrono>

namespace dynet {

struct Timer {
  Timer(const std::string& msg) : msg(msg), start(std::chrono::high_resolution_clock::now()) {}
  ~Timer() {
  }
  void Show(){
    auto stop = std::chrono::high_resolution_clock::now();
    std::cerr << '[' << msg << ' ' << std::chrono::duration<double, std::milli>(stop-start).count() << " ms]\n";
  }
  void Reset(){
    start = std::chrono::high_resolution_clock::now();
  }	
  double Elapsed() {
    auto stop = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(stop-start).count();// in ms
  }
  std::string msg;
  std::chrono::high_resolution_clock::time_point start;
};

} // namespace dynet

#endif
