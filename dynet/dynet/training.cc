#include "dynet/training.h"

#include <boost/serialization/vector.hpp>
#include <boost/serialization/export.hpp>

// #include "dynet/gpu-ops.h"
#include "dynet/param-nodes.h"
#include "dynet/weight-decay.h"

// Macros for defining parameter update functions
#ifdef __CUDACC__
#define DYNET_TRAINER_INST_DEV_IMPL(MyTrainer) \
  template void MyTrainer::update_rule_dev<Device_GPU>(const Device_GPU & dev, real scale, real gscale, const std::vector<Tensor*> & values);
#elif defined(HAVE_CUDA)
// This is correct, but dying when models are read and written.
// if(values[0]->device->type == DeviceType::CPU) { update_rule_dev(*(Device_CPU*)values[0]->device,scale,gscale,values); } 
// else if(values[0]->device->type == DeviceType::GPU) { update_rule_dev(*(Device_GPU*)values[0]->device,scale,gscale,values); } 
// else { throw std::runtime_error("Bad device in MyTrainer::update_rule"); }
#define DYNET_TRAINER_INST_DEV_IMPL(MyTrainer) \
  extern template void MyTrainer::update_rule_dev<Device_GPU>(const Device_GPU & dev, real scale, real gscale, const std::vector<Tensor*> & values); \
  template void MyTrainer::update_rule_dev<Device_CPU>(const Device_CPU & dev, real scale, real gscale, const std::vector<Tensor*> & values); \
  void MyTrainer::update_rule(real scale, real gscale, const std::vector<Tensor*> & values) { \
    if(default_device->type == DeviceType::CPU) { update_rule_dev(*(Device_CPU*)default_device,scale,gscale,values); } \
    else if(default_device->type == DeviceType::GPU) { update_rule_dev(*(Device_GPU*)default_device,scale,gscale,values); } \
    else { throw std::runtime_error("Bad device in MyTrainer::update_rule"); } \
  }
#else
#define DYNET_TRAINER_INST_DEV_IMPL(MyTrainer) \
  template void MyTrainer::update_rule_dev<Device_CPU>(const Device_CPU & dev, real scale, real gscale, const std::vector<Tensor*> & values); \
  void MyTrainer::update_rule(real scale, real gscale, const std::vector<Tensor*> & values) { \
    if(default_device->type == DeviceType::CPU) { update_rule_dev(*(Device_CPU*)default_device,scale,gscale,values); } \
    else { throw std::runtime_error("Bad device in MyTrainer::update_rule"); } \
  }
#endif

namespace dynet {

using namespace std;

template <class Derived>
bool is_valid(const Eigen::MatrixBase<Derived>& x) {
  return ((x - x).array() == (x - x).array()).all();
}

// --- The actual update code for each operation, implemented on various devices

// Trainer base class is run on CPUs
#ifndef __CUDACC__

Trainer::~Trainer() {}

void Trainer::rescale_and_reset_weight_decay() {
  const float weight_decay = model->weight_decay.current_weight_decay();
  const auto params = model->parameters_list();
  for (auto p : model->updated_parameters_list())
    params[p]->scale_parameters(weight_decay);
  const auto lookup_params = model->lookup_parameters_list();
  for (auto p : model->updated_lookup_parameters_list())
    lookup_params[p]->scale_parameters(weight_decay);
  model->weight_decay.reset_weight_decay();
}

float Trainer::clip_gradients(real scale) {
  float gscale = 1;
  if (clipping_enabled) {
    // TODO should I handle updatebale differently?
    float gg = model->gradient_l2_norm();
    if (isnan(gg) || isinf(gg)) {
      ostringstream oss; oss << "Magnitude of gradient is bad: " << gg;
      throw std::runtime_error(oss.str());
    }
    if (scale * gg > clip_threshold) {
      ++clips;
      ++clips_since_status;
      gscale = clip_threshold / (scale * gg);
    }
  }
  return gscale;
}

// this calls update on all of the parameters that are supposed to be updated
void Trainer::update(real scale) {
  update(model->updated_parameters_list(), model->updated_lookup_parameters_list(), scale);
}

// this calls the rule-specific updates over all updated parameters
void Trainer::update(const std::vector<unsigned> & upd_params, const std::vector<unsigned> & upd_lookup_params, real scale) {
  // Allocate if necessary
  if(!aux_allocated) {
    alloc_impl();
    aux_allocated = true;
  }

  // Perform gradient clipping and cycle through parameters
  const float gscale = clip_gradients(scale);
  const auto & params = model->parameters_list();
  for(auto i : upd_params) {
    update_params(scale, gscale, i);
    params[i]->clear();
  }
  const auto & lookup_params = model->lookup_parameters_list();
  for(auto i : upd_lookup_params) {
    if(sparse_updates_enabled && !lookup_params[i]->all_updated) {
      for (auto j : lookup_params[i]->non_zero_grads)
        update_lookup_params(scale, gscale, i, j);
    } else {
      update_lookup_params(scale, gscale, i);
    }
    lookup_params[i]->clear();
  }
  ++updates;
  ++updates_since_status;

  model->weight_decay.update_weight_decay(); // update global weight scale
  if (model->weight_decay.parameters_need_rescaled())
    rescale_and_reset_weight_decay();  // if wdscale is getting to small multiply all weights by wdscale, and set wdscale to 1
}

#endif

// --- SimpleSGDTrainer

// Perform update of ts[0]=parameters, ts[1]=gradients
template <class MyDevice>
void SimpleSGDTrainer::update_rule_dev(const MyDevice & dev, real scale, real gscale, const std::vector<Tensor*> & ts) {
  ts[0]->tvec().device(*dev.edevice) -= ts[1]->tvec() * (eta * scale * gscale / model->weight_decay.current_weight_decay());
}
DYNET_TRAINER_INST_DEV_IMPL(SimpleSGDTrainer)

#ifndef __CUDACC__
void SimpleSGDTrainer::update_params(real scale, real gscale, size_t idx) {
  auto & p = model->parameters_list()[idx];
  update_rule(scale, gscale, {&p->values, &p->g});
}
void SimpleSGDTrainer::update_lookup_params(real scale, real gscale, size_t idx, size_t lidx) {
  auto & p = model->lookup_parameters_list()[idx];
  update_rule(scale, gscale, {&p->values[lidx], &p->grads[lidx]});
}
void SimpleSGDTrainer::update_lookup_params(real scale, real gscale, size_t idx) {
  auto & p = model->lookup_parameters_list()[idx];
  update_rule(scale, gscale, {&p->all_values, &p->all_grads});
}
#endif

// --- CyclicalSGDTrainer

// Perform update of ts[0]=parameters, ts[1]=gradients
template <class MyDevice>
void CyclicalSGDTrainer::update_rule_dev(const MyDevice & dev, real scale, real gscale, const std::vector<Tensor*> & ts) {
  ts[0]->tvec().device(*dev.edevice) -= ts[1]->tvec() * (eta * scale * gscale / model->weight_decay.current_weight_decay());
}
DYNET_TRAINER_INST_DEV_IMPL(CyclicalSGDTrainer)

#ifndef __CUDACC__
void CyclicalSGDTrainer::update_params(real scale, real gscale, size_t idx) {
  auto & p = model->parameters_list()[idx];
  update_rule(scale, gscale, {&p->values, &p->g});
}
void CyclicalSGDTrainer::update_lookup_params(real scale, real gscale, size_t idx, size_t lidx) {
  auto & p = model->lookup_parameters_list()[idx];
  update_rule(scale, gscale, {&p->values[lidx], &p->grads[lidx]});
}
void CyclicalSGDTrainer::update_lookup_params(real scale, real gscale, size_t idx) {
  auto & p = model->lookup_parameters_list()[idx];
  update_rule(scale, gscale, {&p->all_values, &p->all_grads});
}
#endif

// --- MomentumSGDTrainer

// Perform update of ts[0]=parameters, ts[1]=gradients, ts[2]=momentum
template <class MyDevice>
void MomentumSGDTrainer::update_rule_dev(const MyDevice & dev, real scale, real gscale, const std::vector<Tensor*> & ts) {
  ts[2]->tvec().device(*dev.edevice) = ts[2]->tvec() * momentum - ts[1]->tvec() * (eta * scale * gscale);
  ts[0]->tvec().device(*dev.edevice) += ts[2]->tvec() / model->weight_decay.current_weight_decay();
}
DYNET_TRAINER_INST_DEV_IMPL(MomentumSGDTrainer)

#ifndef __CUDACC__
void MomentumSGDTrainer::update_params(real scale, real gscale, size_t idx) {
  auto & p = model->parameters_list()[idx];
  update_rule(scale, gscale, {&p->values, &p->g, &vp[idx].h});
}
void MomentumSGDTrainer::update_lookup_params(real scale, real gscale, size_t idx, size_t lidx) {
  auto & p = model->lookup_parameters_list()[idx];
  update_rule(scale, gscale, {&p->values[lidx], &p->grads[lidx], &vlp[idx].h[lidx]});
}
void MomentumSGDTrainer::update_lookup_params(real scale, real gscale, size_t idx) {
  auto & p = model->lookup_parameters_list()[idx];
  update_rule(scale, gscale, {&p->all_values, &p->all_grads, &vlp[idx].all_h});
}
void MomentumSGDTrainer::alloc_impl() {
  vp = allocate_shadow_parameters(*model);
  vlp = allocate_shadow_lookup_parameters(*model);
}
#endif

// --- AdagradTrainer

// Perform update of ts[0]=parameters, ts[1]=gradients, ts[2]=stddev
template <class MyDevice>
void AdagradTrainer::update_rule_dev(const MyDevice & dev, real scale, real gscale, const std::vector<Tensor*> & ts) {
  ts[1]->tvec().device(*dev.edevice) = ts[1]->tvec() * (scale * gscale);
  ts[2]->tvec().device(*dev.edevice) += ts[1]->tvec().square();
  ts[0]->tvec().device(*dev.edevice) += ts[1]->tvec() / (ts[2]->tvec() + epsilon).sqrt() * (-eta / model->weight_decay.current_weight_decay());
}
DYNET_TRAINER_INST_DEV_IMPL(AdagradTrainer)

#ifndef __CUDACC__
void AdagradTrainer::update_params(real scale, real gscale, size_t idx) {
  auto & p = model->parameters_list()[idx];
  update_rule(scale, gscale, {&p->values, &p->g, &vp[idx].h});
}
void AdagradTrainer::update_lookup_params(real scale, real gscale, size_t idx, size_t lidx) {
  auto & p = model->lookup_parameters_list()[idx];
  update_rule(scale, gscale, {&p->values[lidx], &p->grads[lidx], &vlp[idx].h[lidx]});
}
void AdagradTrainer::update_lookup_params(real scale, real gscale, size_t idx) {
  auto & p = model->lookup_parameters_list()[idx];
  update_rule(scale, gscale, {&p->all_values, &p->all_grads, &vlp[idx].all_h});
}
void AdagradTrainer::alloc_impl() {
  vp = allocate_shadow_parameters(*model);
  vlp = allocate_shadow_lookup_parameters(*model);
}
#endif

// --- AdadeltaTrainer

// Perform update of ts[0]=parameters, ts[1]=gradients, ts[2]=hg, ts[3]=hd
template <class MyDevice>
void AdadeltaTrainer::update_rule_dev(const MyDevice & dev, real scale, real gscale, const std::vector<Tensor*> & ts) {
  ts[1]->tvec().device(*dev.edevice) = ts[1]->tvec() * (scale * gscale);
  ts[2]->tvec().device(*dev.edevice) = ts[2]->tvec() * rho + ts[1]->tvec().square() * (1.f - rho);
  ts[1]->tvec().device(*dev.edevice) = - ts[1]->tvec() * (ts[3]->tvec() + epsilon).sqrt() / (ts[2]->tvec() + epsilon).sqrt();
  ts[3]->tvec().device(*dev.edevice) = ts[3]->tvec() * rho + ts[1]->tvec().square() * (1.f - rho);
  ts[0]->tvec().device(*dev.edevice) += ts[1]->tvec() / model->weight_decay.current_weight_decay();
}
DYNET_TRAINER_INST_DEV_IMPL(AdadeltaTrainer)

#ifndef __CUDACC__
void AdadeltaTrainer::update_params(real scale, real gscale, size_t idx) {
  auto & p = model->parameters_list()[idx];
  update_rule(scale, gscale, {&p->values, &p->g, &hg[idx].h, &hd[idx].h});
}
void AdadeltaTrainer::update_lookup_params(real scale, real gscale, size_t idx, size_t lidx) {
  auto & p = model->lookup_parameters_list()[idx];
  update_rule(scale, gscale, {&p->values[lidx], &p->grads[lidx], &hlg[idx].h[lidx], &hld[idx].h[lidx]});
}
void AdadeltaTrainer::update_lookup_params(real scale, real gscale, size_t idx) {
  auto & p = model->lookup_parameters_list()[idx];
  update_rule(scale, gscale, {&p->all_values, &p->all_grads, &hlg[idx].all_h, &hld[idx].all_h});
}
void AdadeltaTrainer::alloc_impl() {
  hg = allocate_shadow_parameters(*model);
  hlg = allocate_shadow_lookup_parameters(*model);
  hd = allocate_shadow_parameters(*model);
  hld = allocate_shadow_lookup_parameters(*model);
}
#endif

// --- RMSPropTrainer
// TODO: This is not finished yet, because it memorizes a scalar for each set of parameters, not each parameter itself.
//       We could implement this with one tensor for each scalar, but this is pretty wasteful

// Perform update of ts[0]=parameters, ts[1]=gradients
template <class MyDevice>
void RMSPropTrainer::update_rule_dev(const MyDevice & dev, real scale, real gscale, const std::vector<Tensor*> & ts) {
  ts[1]->tvec().device(*dev.edevice) = ts[1]->tvec() * (scale * gscale); // Scale gradient
  ts[2]->tvec().device(*dev.edevice) = ts[2]->tvec() * rho + ts[1]->tvec().square() * (1.f - rho); // Update square gradient exponential average
  ts[1]->tvec().device(*dev.edevice) = - ts[1]->tvec() / (ts[2]->tvec() + epsilon).sqrt(); // Divide by the RMS
  ts[0]->tvec().device(*dev.edevice) += eta * ts[1]->tvec() / model->weight_decay.current_weight_decay(); // Apply weight decay (should we do this?)
  // real& d2 = hg[pi++];
  // real g2 = p->g.vec().squaredNorm();
  // d2 = rho * d2 + (1.f - rho) * g2;
  // p->values.vec() -= ((eta * scale * gscale / sqrt(d2 + epsilon)) * p->g.vec()) / model->weight_decay.current_weight_decay();
}
DYNET_TRAINER_INST_DEV_IMPL(RMSPropTrainer)

#ifndef __CUDACC__
void RMSPropTrainer::update_params(real scale, real gscale, size_t idx) {
  auto & p = model->parameters_list()[idx];
  update_rule(scale, gscale, {&p->values, &p->g, &hmsg[idx].h});
}
void RMSPropTrainer::update_lookup_params(real scale, real gscale, size_t idx, size_t lidx) {
  auto & p = model->lookup_parameters_list()[idx];
  update_rule(scale, gscale, {&p->values[lidx], &p->grads[lidx], &hlmsg[idx].h[lidx]});
}
void RMSPropTrainer::update_lookup_params(real scale, real gscale, size_t idx) {
  auto & p = model->lookup_parameters_list()[idx];
  update_rule(scale, gscale, {&p->all_values, &p->all_grads, &hlmsg[idx].all_h});
}
void RMSPropTrainer::alloc_impl() {
  hmsg = allocate_shadow_parameters(*model);
  hlmsg = allocate_shadow_lookup_parameters(*model);
}
#endif

// --- AdamTrainer

// Perform update of ts[0]=parameters, ts[1]=gradients, ts[2]=mean, ts[3]=variance
template <class MyDevice>
void AdamTrainer::update_rule_dev(const MyDevice & dev, real scale, real gscale, const std::vector<Tensor*> & ts) {
  ts[1]->tvec().device(*dev.edevice) = ts[1]->tvec() * (scale * gscale);
  ts[2]->tvec().device(*dev.edevice) = ts[2]->tvec() * beta_1 + ts[1]->tvec() * (1.f - beta_1);
  ts[3]->tvec().device(*dev.edevice) = ts[3]->tvec() * beta_2 + ts[1]->tvec().square() * (1.f - beta_2);
  float lr_t = eta * sqrt(1-pow(beta_2, updates+1))/(1-pow(beta_1, updates+1))/ model->weight_decay.current_weight_decay();
  ts[0]->tvec().device(*dev.edevice) -= ts[2]->tvec() / (ts[3]->tvec().sqrt() + epsilon) * lr_t;
}
DYNET_TRAINER_INST_DEV_IMPL(AdamTrainer)

#ifndef __CUDACC__
void AdamTrainer::update_params(real scale, real gscale, size_t idx) {
  auto & p = model->parameters_list()[idx];
  update_rule(scale, gscale, {&p->values, &p->g, &m[idx].h, &v[idx].h});
}
void AdamTrainer::update_lookup_params(real scale, real gscale, size_t idx, size_t lidx) {
  auto & p = model->lookup_parameters_list()[idx];
  update_rule(scale, gscale, {&p->values[lidx], &p->grads[lidx], &lm[idx].h[lidx], &lv[idx].h[lidx]});
}
void AdamTrainer::update_lookup_params(real scale, real gscale, size_t idx) {
  auto & p = model->lookup_parameters_list()[idx];
  update_rule(scale, gscale, {&p->all_values, &p->all_grads, &lm[idx].all_h, &lv[idx].all_h});
}
void AdamTrainer::alloc_impl() {
  m = allocate_shadow_parameters(*model);
  lm = allocate_shadow_lookup_parameters(*model);
  v = allocate_shadow_parameters(*model);
  lv = allocate_shadow_lookup_parameters(*model);
}
#endif

// ===== Auxiliary functions for both CPU and GPU
// This function already exists in nodes.cc.
template <class MyDevice>
EIGEN_STRONG_INLINE void logsumexp(const MyDevice & dev, const Tensor& x, Tensor & m, Tensor& z) {
  if(x.d.bd == 1 && x.d[1] == 1) {
    m.t<0>().device(*dev.edevice) = x.t<1>().maximum();
#ifdef __CUDACC__
    Eigen::array<int, 1> bcast;
    bcast[0] = x.d[0];
    // This needs to be split into two lines to prevent memory allocation
    // TODO? Here and in logsoftmax: Is there a better way to subtract a scalar that is already on the GPU without using broadcasting (and without copying the scalar back to the host first)
    z.t<0>().device(*dev.edevice) = (x.t<1>() - m.t<1>().broadcast(bcast)).exp().sum();
    z.t<0>().device(*dev.edevice) = z.t<0>().log() + m.t<0>();
#else
    float mval = as_scalar(m);
    // This needs to be split into two lines to prevent memory allocation
    z.t<0>().device(*dev.edevice) = (x.t<1>() - mval).exp().sum();
    z.t<0>().device(*dev.edevice) = z.t<0>().log() + mval;
#endif
  } else {
    Eigen::array<int, 1> red_axis; red_axis[0] = 0;
    m.tb<1>().device(*dev.edevice) = x.tb<2>().maximum(red_axis);
    // TODO: Currently, the first version is slower on CPU, hence the switch
#ifdef __CUDACC__
    Eigen::array<int, 3> bcast({(int)x.d.rows(), 1, 1});
    Eigen::array<int, 3> morph({1, (int)m.d[0], (int)m.d.bd});
    // This needs to be split into two lines to prevent memory allocation
    z.tb<1>().device(*dev.edevice) = (x.tb<2>() - m.tb<2>().reshape(morph).broadcast(bcast)).exp().sum(red_axis);
    z.tb<1>().device(*dev.edevice) = z.tb<1>().log() + m.tb<1>();
#else
    auto miter = m.v;
    for(size_t b = 0; b < x.d.bd; ++b) {
      for(size_t i = 0; i < x.d[1]; ++i, ++miter) {
        z.tb<1>().chip<1>(b).chip<0>(i).device(*dev.edevice) = (x.tb<2>().chip<2>(b).chip<1>(i) - *miter).exp().sum();
        z.tb<1>().chip<1>(b).chip<0>(i).device(*dev.edevice) = z.tb<1>().chip<1>(b).chip<0>(i).log() + *miter;
      }
    }
#endif
  }
}

// --- EGTrainer
template <class MyDevice>
void EGTrainer::update_rule_dev(const MyDevice & dev, real scale, real gscale, const std::vector<Tensor*> & ts) {
  //------------------------------------------------------------------
  // Add momentum
  ts[2]->tvec().device(*dev.edevice) = ts[2]->tvec() * momentum - ts[1]->tvec() * (eta * scale * gscale);

  //------------------------------------------------------------------
  // Initialize white Gaussian noise using stddev as given noise_eta
  if (noise_eta != 0.f){
    //cerr << "noise_eta=" << noise_eta << endl;
    TensorTools::randomize_normal(*ts[5], 0, noise_eta);
  }

/*
  //------------------------------------------------------------------
  // METHOD 1: naive implementation (slow?)
  std::vector<real> v_params = dynet::as_vector(*ts[0]);// FIXME: not a smart way but it works on both GPU and CPU!
  std::vector<real> v_grads = dynet::as_vector(*ts[2]);// gradients with momentum
  Eigen::TensorMap<Eigen::Tensor<real,1>> t_params(&v_params[0], v_params.size());
  Eigen::TensorMap<Eigen::Tensor<real,1>> t_grads(&v_grads[0], v_grads.size());
  Eigen::Tensor<real,1> v = t_params.log() + t_grads / model->weight_decay.current_weight_decay();// with momentum
  
  Eigen::Tensor<float,0> m = v.maximum();// max
  Eigen::Tensor<float,0> z = m.coeff() + (v - m.coeff()).exp().sum().log();// Z
  Eigen::Tensor<float,1> u = (v - z.coeff()).exp();// result
  TensorTools::set_elements(*ts[0], u.data(), u.dimensions()[0]);// for both GPU and CPU
  //------------------------------------------------------------------
*/

  //------------------------------------------------------------------
  // METHOD 2: advanced implementation (supposedly faster?)
  if (noise_eta != 0.f)
    ts[0]->tvec().device(*dev.edevice) = ts[0]->tvec().log() + ts[2]->tvec() / model->weight_decay.current_weight_decay() + ts[5]->tvec();// with noise-injected momentum
  else ts[0]->tvec().device(*dev.edevice) = ts[0]->tvec().log() + ts[2]->tvec() / model->weight_decay.current_weight_decay();// with momentum only

  logsumexp(dev, *ts[0], *ts[3], *ts[4]);// z refers to logZ
  //cerr << "max_element=" << as_scalar(m) << endl;
  //cerr << "logZ=" << as_scalar(z) << endl;
    
  ts[0]->tvec().device(*dev.edevice) = (ts[0]->tvec() - as_scalar(*ts[4])).exp();// FIXME: other way(s) of not using as_scalar(z)?
  //------------------------------------------------------------------
}
DYNET_TRAINER_INST_DEV_IMPL(EGTrainer)

#ifndef __CUDACC__
void EGTrainer::update_params(real scale, real gscale, size_t idx) {
  auto & p = model->parameters_list()[idx];
  update_rule(scale, gscale, {&p->values, &p->g, &hp[idx].h, &meg, &zeg, &np[idx].h});
}
void EGTrainer::update_lookup_params(real scale, real gscale, size_t idx, size_t lidx) {
  auto & p = model->lookup_parameters_list()[idx];
  update_rule(scale, gscale, {&p->values[lidx], &p->grads[lidx], &hlp[idx].h[lidx], &meg, &zeg, &nlp[idx].h[lidx]});
}
void EGTrainer::update_lookup_params(real scale, real gscale, size_t idx) {
  auto & p = model->lookup_parameters_list()[idx];
  update_rule(scale, gscale, {&p->all_grads, &p->all_grads, &hlp[idx].all_h, &meg, &zeg, &nlp[idx].all_h});
}
void EGTrainer::alloc_impl() {
  hp = allocate_shadow_parameters(*model);
  hlp = allocate_shadow_lookup_parameters(*model);
  if (noise_eta != 0.f){
    np = allocate_shadow_parameters(*model);
    nlp = allocate_shadow_lookup_parameters(*model);  
  }
}
#endif

// --- AdamEGTrainer

// Perform update of ts[0]=parameters, ts[1]=gradients, ts[2]=accumulated_gradients
template <class MyDevice>
void AdamEGTrainer::update_rule_dev(const MyDevice & dev, real scale, real gscale, const std::vector<Tensor*> & ts) {
  // --------------------------
  // Adam step
  ts[1]->tvec().device(*dev.edevice) = ts[1]->tvec() * (scale * gscale);
  ts[2]->tvec().device(*dev.edevice) = ts[2]->tvec() * beta_1 + ts[1]->tvec() * (1.f - beta_1);
  ts[3]->tvec().device(*dev.edevice) = ts[3]->tvec() * beta_2 + ts[1]->tvec().square() * (1.f - beta_2);
  float lr_t = eta * sqrt(1-pow(beta_2, updates+1))/(1-pow(beta_1, updates+1))/ model->weight_decay.current_weight_decay();
  //ts[0]->tvec().device(*dev.edevice) -= ts[2]->tvec() / (ts[3]->tvec().sqrt() + epsilon) * lr_t;
  // --------------------------
  
/*
  //------------------------------------------------------------------
  // METHOD 1: naive implementation (slow?)
  std::vector<real> v_params = dynet::as_vector(*ts[0]);
  //std::vector<real> v_grads = dynet::as_vector(*ts[1]);
  std::vector<real> v_m1 = dynet::as_vector(*ts[2]);// for adam step
  std::vector<real> v_m2 = dynet::as_vector(*ts[3]);// for adam step

  Eigen::TensorMap<Eigen::Tensor<real,1>> t_params(&v_params[0], v_params.size());
  //Eigen::TensorMap<Eigen::Tensor<real,1>> t_grads(&v_grads[0], v_grads.size());
  Eigen::TensorMap<Eigen::Tensor<real,1>> t_m1(&v_m1[0], v_m1.size());// for adam step
  Eigen::TensorMap<Eigen::Tensor<real,1>> t_m2(&v_m2[0], v_m2.size());// for adam step
  Eigen::Tensor<real,1> v = t_params.log() - t_m1 / (t_m2.sqrt() + eps) * lr_t;// for adam step
  
  Eigen::Tensor<float,0> m = v.maximum() ;// max
  Eigen::Tensor<float,0> z = m.coeff() + (v - m.coeff()).exp().sum().log();// Z
  Eigen::Tensor<float,1> u = (v - z.coeff()).exp();// result
  TensorTools::set_elements(*ts[0], u.data(), u.dimensions()[0]);
  //------------------------------------------------------------------   
*/

  //------------------------------------------------------------------
  // METHOD 2: advanced implementation (supposedly faster?)
  ts[0]->tvec().device(*dev.edevice) = ts[0]->tvec().log() - ts[2]->tvec() / (ts[3]->tvec().sqrt() + epsilon) * lr_t;// Adam update

  logsumexp(dev, *ts[0], *ts[4], *ts[5]);// z refers to logZ
  //cerr << "max_element=" << as_scalar(m) << endl;
  //cerr << "logZ=" << as_scalar(z) << endl;
    
  ts[0]->tvec().device(*dev.edevice) = (ts[0]->tvec() - as_scalar(*ts[5])).exp();// FIXME: other way(s) of not using as_scalar(z)?
  //------------------------------------------------------------------
}
DYNET_TRAINER_INST_DEV_IMPL(AdamEGTrainer)

#ifndef __CUDACC__
void AdamEGTrainer::update_params(real scale, real gscale, size_t idx) {
  auto & p = model->parameters_list()[idx];
  update_rule(scale, gscale, {&p->values, &p->g, &m[idx].h, &v[idx].h, &meg, &zeg});
}
void AdamEGTrainer::update_lookup_params(real scale, real gscale, size_t idx, size_t lidx) {
  auto & p = model->lookup_parameters_list()[idx];
  update_rule(scale, gscale, {&p->values[lidx], &p->grads[lidx], &lm[idx].h[lidx], &lv[idx].h[lidx], &meg, &zeg});
}
void AdamEGTrainer::update_lookup_params(real scale, real gscale, size_t idx) {
  auto & p = model->lookup_parameters_list()[idx];
  update_rule(scale, gscale, {&p->all_grads, &p->all_grads, &lm[idx].all_h, &lv[idx].all_h, &meg, &zeg});
}
void AdamEGTrainer::alloc_impl() {
  m = allocate_shadow_parameters(*model);
  lm = allocate_shadow_lookup_parameters(*model);
  v = allocate_shadow_parameters(*model);
  lv = allocate_shadow_lookup_parameters(*model);  
}
#endif

// --- RMSPropEGTrainer

// Perform update of ts[0]=parameters, ts[1]=gradients, ts[2]=accumulated_gradients
template <class MyDevice>
void RMSPropEGTrainer::update_rule_dev(const MyDevice & dev, real scale, real gscale, const std::vector<Tensor*> & ts) {
  // --------------------------
  // RMSProp step
  ts[1]->tvec().device(*dev.edevice) = ts[1]->tvec() * (scale * gscale); // Scale gradient
  ts[2]->tvec().device(*dev.edevice) = ts[2]->tvec() * rho + ts[1]->tvec().square() * (1.f - rho); // Update square gradient exponential average
  ts[1]->tvec().device(*dev.edevice) = - ts[1]->tvec() / (ts[2]->tvec() + epsilon).sqrt(); // Divide by the RMS
  //ts[0]->tvec().device(*dev.edevice) += eta * ts[1]->tvec() / model->weight_decay.current_weight_decay(); // Apply weight decay (should we do this?)
  // --------------------------
  
/*
  //------------------------------------------------------------------
  // METHOD 1: naive implementation (slow?)
  std::vector<real> v_params = dynet::as_vector(*ts[0]);
  //std::vector<real> v_grads = dynet::as_vector(*ts[1]);
  std::vector<real> v_m1 = dynet::as_vector(*ts[2]);// for RMSProp step
  std::vector<real> v_m2 = dynet::as_vector(*ts[3]);// for RMSProp step

  Eigen::TensorMap<Eigen::Tensor<real,1>> t_params(&v_params[0], v_params.size());
  //Eigen::TensorMap<Eigen::Tensor<real,1>> t_grads(&v_grads[0], v_grads.size());
  Eigen::TensorMap<Eigen::Tensor<real,1>> t_m1(&v_m1[0], v_m1.size());// for RMSProp step
  Eigen::TensorMap<Eigen::Tensor<real,1>> t_m2(&v_m2[0], v_m2.size());// for RMSProp step
  Eigen::Tensor<real,1> v = t_params.log() - t_m1 / (t_m2.sqrt() + eps) * lr_t;// for RMSProp step
  
  Eigen::Tensor<float,0> m = v.maximum() ;// max
  Eigen::Tensor<float,0> z = m.coeff() + (v - m.coeff()).exp().sum().log();// Z
  Eigen::Tensor<float,1> u = (v - z.coeff()).exp();// result
  TensorTools::set_elements(*ts[0], u.data(), u.dimensions()[0]);
  //------------------------------------------------------------------   
*/

  //------------------------------------------------------------------
  // METHOD 2: advanced implementation (supposedly faster?)
  ts[0]->tvec().device(*dev.edevice) = ts[0]->tvec().log() + eta * ts[1]->tvec() / model->weight_decay.current_weight_decay(); // RMSProp update

  logsumexp(dev, *ts[0], *ts[3], *ts[4]);// z refers to logZ
  //cerr << "max_element=" << as_scalar(m) << endl;
  //cerr << "logZ=" << as_scalar(z) << endl;
    
  ts[0]->tvec().device(*dev.edevice) = (ts[0]->tvec() - as_scalar(*ts[4])).exp();// FIXME: other way(s) of not using as_scalar(z)?
  //------------------------------------------------------------------
}
DYNET_TRAINER_INST_DEV_IMPL(RMSPropEGTrainer)

#ifndef __CUDACC__
void RMSPropEGTrainer::update_params(real scale, real gscale, size_t idx) {
  auto & p = model->parameters_list()[idx];
  update_rule(scale, gscale, {&p->values, &p->g, &hmsg[idx].h, &meg, &zeg});
}
void RMSPropEGTrainer::update_lookup_params(real scale, real gscale, size_t idx, size_t lidx) {
  auto & p = model->lookup_parameters_list()[idx];
  update_rule(scale, gscale, {&p->values[lidx], &p->grads[lidx], &hlmsg[idx].h[lidx], &meg, &zeg});
}
void RMSPropEGTrainer::update_lookup_params(real scale, real gscale, size_t idx) {
  auto & p = model->lookup_parameters_list()[idx];
  update_rule(scale, gscale, {&p->all_values, &p->all_grads, &hlmsg[idx].all_h, &meg, &zeg});
}
void RMSPropEGTrainer::alloc_impl() {
  hmsg = allocate_shadow_parameters(*model);
  hlmsg = allocate_shadow_lookup_parameters(*model);
}
#endif

// --- SGLDTrainer

// Perform update of ts[0]=parameters, ts[1]=gradients
template <class MyDevice>
void SGLDTrainer::update_rule_dev(const MyDevice & dev, real scale, real gscale, const std::vector<Tensor*> & ts) {
  //------------------------------------------------------------------
  // Initialize white Gaussian noise using stddev as given noise_eta
  TensorTools::randomize_normal(*ts[2], 0, eta);

  ts[0]->tvec().device(*dev.edevice) += ts[1]->tvec() * (0.5f * eta * scale * gscale / model->weight_decay.current_weight_decay()) + ts[2]->tvec();// FIXME: requires weight_decay? gscale?
}
DYNET_TRAINER_INST_DEV_IMPL(SGLDTrainer)

#ifndef __CUDACC__
void SGLDTrainer::update_params(real scale, real gscale, size_t idx) {
  auto & p = model->parameters_list()[idx];
  update_rule(scale, gscale, {&p->values, &p->g, &np[idx].h});
}
void SGLDTrainer::update_lookup_params(real scale, real gscale, size_t idx, size_t lidx) {
  auto & p = model->lookup_parameters_list()[idx];
  update_rule(scale, gscale, {&p->values[lidx], &p->grads[lidx], &nlp[idx].h[lidx]});
}
void SGLDTrainer::update_lookup_params(real scale, real gscale, size_t idx) {
  auto & p = model->lookup_parameters_list()[idx];
  update_rule(scale, gscale, {&p->all_values, &p->all_grads, &nlp[idx].all_h});
}
void SGLDTrainer::alloc_impl() {
  np = allocate_shadow_parameters(*model);
  nlp = allocate_shadow_lookup_parameters(*model); 
}
#endif

// --- pSGLDTrainer
// Perform update of ts[0]=parameters, ts[1]=gradients
template <class MyDevice>
void pSGLDTrainer::update_rule_dev(const MyDevice & dev, real scale, real gscale, const std::vector<Tensor*> & ts) {
  ts[1]->tvec().device(*dev.edevice) = ts[1]->tvec() * (scale * gscale); // Scale gradient
  ts[2]->tvec().device(*dev.edevice) = ts[2]->tvec() * rho + ts[1]->tvec().square() * (1.f - rho); // Update square gradient exponential average
  ts[1]->tvec().device(*dev.edevice) = - ts[1]->tvec() / (ts[2]->tvec() + epsilon).sqrt(); // Divide by the RMS
  ts[0]->tvec().device(*dev.edevice) += eta * ts[1]->tvec() / model->weight_decay.current_weight_decay(); // Apply weight decay (should we do this?)
}
DYNET_TRAINER_INST_DEV_IMPL(pSGLDTrainer)

#ifndef __CUDACC__
void pSGLDTrainer::update_params(real scale, real gscale, size_t idx) {
  auto & p = model->parameters_list()[idx];
  update_rule(scale, gscale, {&p->values, &p->g, &hmsg[idx].h});
}
void pSGLDTrainer::update_lookup_params(real scale, real gscale, size_t idx, size_t lidx) {
  auto & p = model->lookup_parameters_list()[idx];
  update_rule(scale, gscale, {&p->values[lidx], &p->grads[lidx], &hlmsg[idx].h[lidx]});
}
void pSGLDTrainer::update_lookup_params(real scale, real gscale, size_t idx) {
  auto & p = model->lookup_parameters_list()[idx];
  update_rule(scale, gscale, {&p->all_values, &p->all_grads, &hlmsg[idx].all_h});
}
void pSGLDTrainer::alloc_impl() {
  hmsg = allocate_shadow_parameters(*model);
  hlmsg = allocate_shadow_lookup_parameters(*model);
}
#endif

#ifndef __CUDACC__
// BOOST_CLASS_EXPORT_IMPLEMENT(dynet::SimpleSGDTrainer)
// BOOST_CLASS_EXPORT_IMPLEMENT(dynet::MomentumSGDTrainer)
// BOOST_CLASS_EXPORT_IMPLEMENT(dynet::AdagradTrainer)
// BOOST_CLASS_EXPORT_IMPLEMENT(dynet::AdadeltaTrainer)
// BOOST_CLASS_EXPORT_IMPLEMENT(dynet::RMSPropTrainer)
// BOOST_CLASS_EXPORT_IMPLEMENT(dynet::AdamTrainer)

DYNET_SERIALIZE_COMMIT(Trainer, DYNET_SERIALIZE_DEFINE(eta0, eta, eta_decay, epoch,
						       clipping_enabled, clip_threshold, clips, updates,
						       aux_allocated, model))
DYNET_SERIALIZE_IMPL(Trainer)

DYNET_SERIALIZE_COMMIT(SimpleSGDTrainer, DYNET_SERIALIZE_DERIVED_EQ_DEFINE(Trainer))
DYNET_SERIALIZE_IMPL(SimpleSGDTrainer)

DYNET_SERIALIZE_COMMIT(CyclicalSGDTrainer, DYNET_SERIALIZE_DERIVED_EQ_DEFINE(Trainer))
DYNET_SERIALIZE_IMPL(CyclicalSGDTrainer)

DYNET_SERIALIZE_COMMIT(MomentumSGDTrainer, DYNET_SERIALIZE_DERIVED_DEFINE(Trainer, momentum, vp, vlp))
DYNET_SERIALIZE_IMPL(MomentumSGDTrainer)

DYNET_SERIALIZE_COMMIT(AdagradTrainer, DYNET_SERIALIZE_DERIVED_DEFINE(Trainer, epsilon, vp, vlp))
DYNET_SERIALIZE_IMPL(AdagradTrainer)

DYNET_SERIALIZE_COMMIT(AdadeltaTrainer, DYNET_SERIALIZE_DERIVED_DEFINE(Trainer, epsilon, rho, hg, hlg, hd, hld))
DYNET_SERIALIZE_IMPL(AdadeltaTrainer)

DYNET_SERIALIZE_COMMIT(RMSPropTrainer, DYNET_SERIALIZE_DERIVED_DEFINE(Trainer, epsilon, rho, hmsg, hlmsg))
DYNET_SERIALIZE_IMPL(RMSPropTrainer)

DYNET_SERIALIZE_COMMIT(AdamTrainer, DYNET_SERIALIZE_DERIVED_DEFINE(Trainer, beta_1, beta_2, epsilon, m, lm, v, lv))
DYNET_SERIALIZE_IMPL(AdamTrainer)

//---
DYNET_SERIALIZE_COMMIT(SGLDTrainer, DYNET_SERIALIZE_DERIVED_EQ_DEFINE(Trainer))
DYNET_SERIALIZE_IMPL(SGLDTrainer)

DYNET_SERIALIZE_COMMIT(pSGLDTrainer, DYNET_SERIALIZE_DERIVED_DEFINE(Trainer, epsilon, rho, hmsg, hlmsg))
DYNET_SERIALIZE_IMPL(pSGLDTrainer)
//---

//---
DYNET_SERIALIZE_COMMIT(EGTrainer, DYNET_SERIALIZE_DERIVED_DEFINE(Trainer, momentum, noise_eta, hp, hlp))
DYNET_SERIALIZE_IMPL(EGTrainer)

DYNET_SERIALIZE_COMMIT(AdamEGTrainer, DYNET_SERIALIZE_DERIVED_DEFINE(Trainer, beta_1, beta_2, epsilon, m, lm, v, lv))
DYNET_SERIALIZE_IMPL(AdamEGTrainer)

DYNET_SERIALIZE_COMMIT(RMSPropEGTrainer, DYNET_SERIALIZE_DERIVED_DEFINE(Trainer, epsilon, rho, hmsg, hlmsg))
DYNET_SERIALIZE_IMPL(RMSPropEGTrainer)
//---

#endif

} // namespace dynet

#ifndef __CUDACC__
BOOST_CLASS_EXPORT_IMPLEMENT(dynet::SimpleSGDTrainer)
BOOST_CLASS_EXPORT_IMPLEMENT(dynet::CyclicalSGDTrainer)
BOOST_CLASS_EXPORT_IMPLEMENT(dynet::MomentumSGDTrainer)
BOOST_CLASS_EXPORT_IMPLEMENT(dynet::AdagradTrainer)
BOOST_CLASS_EXPORT_IMPLEMENT(dynet::AdadeltaTrainer)
BOOST_CLASS_EXPORT_IMPLEMENT(dynet::RMSPropTrainer)
BOOST_CLASS_EXPORT_IMPLEMENT(dynet::AdamTrainer)
//---
BOOST_CLASS_EXPORT_IMPLEMENT(dynet::SGLDTrainer)
BOOST_CLASS_EXPORT_IMPLEMENT(dynet::pSGLDTrainer)
//---
BOOST_CLASS_EXPORT_IMPLEMENT(dynet::EGTrainer)
BOOST_CLASS_EXPORT_IMPLEMENT(dynet::AdamEGTrainer)
BOOST_CLASS_EXPORT_IMPLEMENT(dynet::RMSPropEGTrainer)
//---
#endif
