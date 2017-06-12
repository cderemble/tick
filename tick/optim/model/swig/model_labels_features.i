
%{
#include "model_labels_features.h"
%}


class ModelLabelsFeatures : public virtual Model {

 public:
  ModelLabelsFeatures(const SBaseArrayDouble2dPtr features,
                      const SArrayDoublePtr labels);

  virtual ulong get_n_samples() const;
  virtual ulong get_n_features() const;
};
