#include <random>
#include <algorithm>
#include <vector>
#include <iostream>
using namespace std;


double dis_st(double* fm1, int pos1, double* fm2, int pos2, int flen) {
    double sum = 0.0;
    for (int l=0; l<flen; l++){
        sum += pow(fm1[pos1*flen+l]-fm2[pos2*flen+l],2.0);
    }
    return sum;
}

extern "C" void calc(double* fms, double* fmo, int flen, int sample_total, int original_total, int sample_select, int* result, double* stdvars = NULL) {
    double* mindiss = new double[sample_total];
    #pragma omp parallel for
    for (int j=0; j<sample_total; j++) {
        mindiss[j] = 1e200;
        for (int k=0; k<original_total; k++){
            double dis = dis_st(fms, j, fmo, k, flen);
            if (dis < mindiss[j]) mindiss[j] = dis;
        }
    }

    double mindis_mean;
    double var_mean;
    if (stdvars != NULL) {
        double mindis_sum = 0.0;
        double var_sum = 0.0;
        for (int j=0; j<sample_total; j++) {
            mindis_sum += mindiss[j];
            var_sum += stdvars[j];
        }
        mindis_mean = mindis_sum / sample_total;
        var_mean = var_sum / sample_total;
    }

    for (int i=0; i<sample_select; i++){
        if (i>0){
            #pragma omp parallel for
            for (int j=0; j<sample_total; j++){
                if (mindiss[j] > 0.0){
                    double dis = dis_st(fms, j, fms, result[i-1], flen);
                    if (dis < mindiss[j]) mindiss[j] = dis;
                }
            }
        }
        int current_select = -1;
        double current_d = -1.0;
        if (stdvars == NULL) {
            for (int j=0; j<sample_total; j++) {
                if (mindiss[j]>current_d){
                    current_select = j;
                    current_d = mindiss[j];
                }
            }
        }
        else {
            for (int j=0; j<sample_total; j++) {
                if (mindiss[j] > 0.0) {
                    double wdis = 1.0 / (mindis_mean / mindiss[j] + var_mean / stdvars[j]);
                    if (wdis>current_d){
                        current_select = j;
                        current_d = wdis;
                    }
                }
            }
        }
        result[i] = current_select;
        mindiss[current_select] = 0.0;
    }
    delete[] mindiss;
}
