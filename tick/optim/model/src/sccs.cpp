//
// Created by Maryan Morel on 18/05/2017.
//

#include "sccs.h"
#include <base.h>
#include <cmath>

// Remark: in this model, i represents a patient, not an observation (i, b)

ModelSCCS::ModelSCCS(const SBaseArrayDouble2dPtrList1D &features,
                                                 const SArrayIntPtrList1D &labels,
                                                 const SBaseArrayULongPtr censoring,
                                                 ulong n_lags)
    : n_intervals(features[0]->n_rows()),
      n_lags(n_lags),
      n_samples(features.size()),
      n_observations(n_samples * n_intervals),
      n_lagged_features(features[0]->n_cols()),
      n_features(n_lags > 0 ? n_lagged_features / n_lags : n_lagged_features),
      labels(labels),
      features(features),
      censoring(censoring) {}  // TODO: add checks here?

double ModelSCCS::loss(const ArrayDouble &coeffs) {
  double loss = 0;
  for (ulong i = 0; i < n_samples; ++i)
    loss += loss_i(i, coeffs);

  return loss / n_samples;
}

double ModelSCCS::loss_i(const ulong i,
                                       const ArrayDouble &coeffs) {
  double loss = 0;
  ArrayDouble inner_prod(n_intervals), softmax(n_intervals);
  ulong max_interval = get_max_interval(i);

  for (ulong t = 0; t < max_interval; t++)
    inner_prod[t] = get_inner_prod(i, t, coeffs);
  for (ulong t = max_interval; t < n_intervals; t++)
    inner_prod[t] = 0;

  softMax(inner_prod, softmax);

  for (ulong t = 0; t < max_interval; t++)
    loss -= get_longitudinal_label(i, t) * log(softmax[t]);

  return loss;
}

void ModelSCCS::grad(const ArrayDouble &coeffs,
                                   ArrayDouble &out) {
  out.init_to_zero();
  ArrayDouble buffer(out.size());  // is init to 0 in grad_i

  for (ulong i = 0; i < n_samples; ++i) {
    grad_i(i, coeffs, buffer);
    out.mult_incr(buffer, 1);
  }

  for (ulong j = 0; j < out.size(); ++j) {
    out[j] /= n_samples;
  }
}

void ModelSCCS::grad_i(const ulong i,
                                     const ArrayDouble &coeffs,
                                     ArrayDouble &out) {
  out.init_to_zero();
  ArrayDouble inner_prod(n_intervals);
  ArrayDouble buffer(n_lagged_features);
  buffer.init_to_zero();
  ulong max_interval = get_max_interval(i);

  for (ulong t = 0; t < max_interval; t++)
    inner_prod[t] = get_inner_prod(i, t, coeffs);
  for (ulong t = max_interval; t < n_intervals; t++)
    inner_prod[t] = 0;

  double x_max = inner_prod.max();
  long double sum_exp = sumExpMinusMax(inner_prod, x_max);

  double multiplier = 0;  // need a double instead of long double for mult_incr
  for (ulong t = 0; t < max_interval; t++) {
    multiplier =
        exp(inner_prod[t] - x_max) / sum_exp;  // overflow-proof
    buffer.mult_incr(get_longitudinal_features(i, t), multiplier);
  }

  double label = 0;
  for (ulong t = 0; t < max_interval; t++) {
    label = get_longitudinal_label(i, t);
    if (label != 0) {
      out.mult_add_mult_incr(get_longitudinal_features(i, t),
                             -label,
                             buffer,
                             label);
    }
  }
}

void ModelSCCS::compute_lip_consts() {
  lip_consts = ArrayDouble(n_samples);
  lip_consts.init_to_zero();
  BaseArrayDouble row, other_row;

  double max_sq_norm, sq_norm;

  for (ulong sample = 0; sample < n_samples; sample++) {
    max_sq_norm = 0;
    ulong max_interval = get_max_interval(sample);
    for (ulong t=0; t < max_interval; t++) {
      row = get_longitudinal_features(sample, t);
      // Lipschitz constant = 0 if Y_{sample, t} = 0
      if (get_longitudinal_label(sample, t) > 0) {
        for (ulong k = 0; k < max_interval; k++) {
          other_row = get_longitudinal_features(sample, k);
          sq_norm = 0;
          for (ulong feature = 0; feature < n_lagged_features; feature++) {
            sq_norm += pow(row.value(feature) - other_row.value(feature), 2.0);
          }
          max_sq_norm = sq_norm > max_sq_norm ? sq_norm : max_sq_norm;
        }
      }
    }
    lip_consts[sample] = max_sq_norm / 4;
  }
}

double ModelSCCS::get_inner_prod(const ulong i,
                                               const ulong t,
                                               const ArrayDouble &coeffs) const {
  BaseArrayDouble sample = get_longitudinal_features(i, t);
  return sample.dot(coeffs);
}