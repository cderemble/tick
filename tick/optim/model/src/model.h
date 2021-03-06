//
// Created by Martin Bompaire on 22/10/15.
//

#ifndef TICK_OPTIM_MODEL_SRC_MODEL_H_
#define TICK_OPTIM_MODEL_SRC_MODEL_H_

#include <iostream>
#include <cereal/cereal.hpp>

#include "base.h"


// TODO: Model "data" : ModeLabelsFeatures, Model,Model pour les Hawkes

/**
 * @class Model
 * @brief The main Model class from which all models inherit.
 * @note This class has all methods ever used by any model, hence solvers which are using a
 * pointer on a model should be able to call all methods they need. This is certainly not the
 * best possible design but it is sufficient at the moment.
 */
class Model {
 public:
  Model() {}

  virtual const char *get_class_name() const {
    return "Model";
  }

  virtual double loss_i(const ulong i, const ArrayDouble &coeffs) {
    TICK_CLASS_DOES_NOT_IMPLEMENT(get_class_name());
  }

  virtual void grad_i(const ulong i, const ArrayDouble &coeffs, ArrayDouble &out) {
    TICK_CLASS_DOES_NOT_IMPLEMENT(get_class_name());
  }

  virtual void grad(const ArrayDouble &coeffs, ArrayDouble &out) {
    TICK_CLASS_DOES_NOT_IMPLEMENT(get_class_name());
  }

  virtual double loss(const ArrayDouble &coeffs) {
    TICK_CLASS_DOES_NOT_IMPLEMENT(get_class_name());
  }

  virtual ulong get_epoch_size() const {
    TICK_CLASS_DOES_NOT_IMPLEMENT(get_class_name());
  }

  virtual ulong get_n_samples() const {
    TICK_CLASS_DOES_NOT_IMPLEMENT(get_class_name());
  }

  virtual ulong get_n_features() const {
    TICK_CLASS_DOES_NOT_IMPLEMENT(get_class_name());
  }

  // Number of parameters to be estimated. Can differ from the number of
  // features, e.g. when including an intercept.
  virtual ulong get_n_coeffs() const {
    TICK_CLASS_DOES_NOT_IMPLEMENT(get_class_name());
  }

  virtual double sdca_dual_min_i(ulong i,
                                 const ArrayDouble &dual_vector,
                                 const ArrayDouble &primal_vector,
                                 const ArrayDouble &previous_delta_dual,
                                 double l_l2sq) {
    TICK_CLASS_DOES_NOT_IMPLEMENT(get_class_name());
  }

  virtual BaseArrayDouble get_features(const ulong i) const {
    TICK_CLASS_DOES_NOT_IMPLEMENT(get_class_name());
  }

  virtual bool is_sparse() const {
    return false;
  }

  virtual bool use_intercept() const {
    return false;
  }

  virtual double grad_i_factor(const ulong i, const ArrayDouble &coeffs) {
    TICK_CLASS_DOES_NOT_IMPLEMENT(get_class_name());
  }

  virtual void compute_lip_consts() {
    TICK_CLASS_DOES_NOT_IMPLEMENT(get_class_name());
  }

  /**
   * @brief Get the maximum of all Lipschits constants
   * @note This will cache the obtained value for later calls
   */
  virtual double get_lip_max() {
    TICK_CLASS_DOES_NOT_IMPLEMENT(get_class_name());
  }

  /**
   * @brief Get the mean of all Lipschits constants
   * @note This will cache the obtained value for later calls
   */
  virtual double get_lip_mean() {
    TICK_CLASS_DOES_NOT_IMPLEMENT(get_class_name());
  }
};

typedef std::shared_ptr<Model> ModelPtr;

#endif  // TICK_OPTIM_MODEL_SRC_MODEL_H_

