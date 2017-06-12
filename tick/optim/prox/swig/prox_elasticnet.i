%{
#include "prox_elasticnet.h"
%}


class ProxElasticNet : public ProxSeparable {


public:

    ProxElasticNet(double strength, double ratio, bool positive);

    ProxElasticNet(double strength, double ratio, ulong start, ulong end, bool positive);

    inline virtual void set_positive(bool positive);

    inline virtual void set_ratio(double ratio);

    inline virtual double get_ratio() const;

};
