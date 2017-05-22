

#include "prox_tv.h"
#include "prox_binarsity.h"


ProxBinarsity::ProxBinarsity(double strength,
                             SArrayULongPtr blocks_start,
                             SArrayULongPtr blocks_length,
                             bool positive)
        : Prox(strength)
{
    this->blocks_start = blocks_start;
    this->blocks_length = blocks_length;
    this->positive = positive;
    // blocks_start and blocks_end have the same size
    n_blocks = blocks_start->size();
    // The object is not ready (prepare has not been called)
    ready = false;
}


ProxBinarsity::ProxBinarsity(double strength,
                             SArrayULongPtr blocks_start,
                             SArrayULongPtr blocks_length,
                             ulong start,
                             ulong end, bool positive)
        : Prox(strength, start, end)
{
    this->blocks_start = blocks_start;
    this->blocks_length = blocks_length;
    this->positive = positive;
    // blocks_start and blocks_end have the same size
    n_blocks = blocks_start->size();
    // The object is not ready (prepare has not been called)
    ready = false;
}


void ProxBinarsity::prepare()
{
    if(!ready) {
        blocks_prox.clear();
        for(ulong k=0; k < n_blocks; k++) {
            ulong start = (*blocks_start)[k];
            if(has_range) {
                // If there is a range, we apply the global start
                start += this->start;
            }
            ulong end = start + (*blocks_length)[k];
            blocks_prox.push_back(ProxTV(strength, start, end, positive));
        }
        ready = true;
    }
}


const std::string ProxBinarsity::get_class_name() const
{
    return "ProxBinarsity";
}


double ProxBinarsity::_value(ArrayDouble &coeffs,
                             ulong start,
                             ulong end)
{
    prepare();
    double val = 0.;
    for(ulong k=0; k < n_blocks; k++) {
        val += blocks_prox[k].value(coeffs);
    }
    return val;
}

void ProxBinarsity::_call(ArrayDouble &coeffs,
                          double step,
                          ArrayDouble &out,
                          ulong start,
                          ulong end)
{
    prepare();
    for(ulong k=0; k < n_blocks; k++) {
        blocks_prox[k].call(coeffs, step, out);
        ulong start_k = blocks_prox[k].get_start();
        ulong end_k = blocks_prox[k].get_end();
        ArrayDouble out_block_k = view(out, start_k, end_k);
        double mean_k = out_block_k.sum() / (end_k - start_k);
        for (ulong j = 0; j < end_k - start_k; j++) {
            out_block_k[j] -= mean_k;
        }
    }
}
