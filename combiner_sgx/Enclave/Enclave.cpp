#include "string.h"
#include "Enclave_t.h"
#include <stdio.h>
#include "math.h"
// trusted libraries for SGX
#include "sgx_trts.h"
#include "sgx_tseal.h"

#define SECRET_FILE "enclave_secret"
#define RANDOM_FILE "enclave_secret_rand"


void printf(const char *fmt, ...)
{
    char buf[BUFSIZ] = {'\0'};
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, BUFSIZ, fmt, ap);
    va_end(ap);
    ocall_print(buf);
}

int init_model_inputs( ModelInput *inputs){
	for(int i =0; i < MODEL_NUM; ++i){
		inputs[i].weight = 1;
	}
}



// write to file/read from file using ocalls
// void write_to_file() {
// 	ocall_write_file ("test_file.txt", "horiahoria", 10);
// }

// void read_from_file() {
// 	char *buf = (char *) malloc(5);
// 	if (buf == NULL) {
// 		ocall_print("Failed to allocate space for buf\n");
// 	}
// 	ocall_read_file("test_file.txt", buf, 5);
// 	free(buf);
// }


void softmax(ModelInput *input, int length) {

    double denominator = 0;

    for (int i = 0; i < length; ++i) {
        denominator += exp(input[i].weight);
    }


    for (int i = 0; i < length; ++i) {
        input[i].weight = exp(input[i].weight) / denominator;
    }

}

void cal_output(ModelInput *inputs, double outputs[QUERY_SIZE][CATEGORY_NUM]){


    for (int j = 0; j < QUERY_SIZE; ++j) {

        double denominator = 0;
        double temp_output[CATEGORY_NUM] = {0};


        for (int i = 0; i < MODEL_NUM; ++i) {
            denominator += inputs[0].weight;
        }

        for (int i = 0; i < MODEL_NUM; ++i) {
            for (int k = 0; k < CATEGORY_NUM; ++k) {
                temp_output[k] += inputs[i].weight *  inputs[i].preds[j][k];
            }
        }

        for (int k = 0; k  < CATEGORY_NUM; ++k) {
            outputs[j][k]  = temp_output[k] / denominator;
        }
//        cout << "" << endl;
    }

}


double get_L2_distance(double *vec1, double *vec2, int size){
    double distance = 0;

    for (int i = 0; i < size; ++i) {
        distance += pow(vec1[i] - vec2[i], 2);
    }

    distance = sqrt(distance);
    return distance;
}

double get_KL_divergence_distance(double *vec1, double *vec2, int size){

    // KL(vec1,cev2)
    double distance = 0;

    for (int i = 0; i < size; ++i) {
        int addition = 0;
        // if( !(IS_DOUBLE_ZERO(vec1[i]) || IS_DOUBLE_ZERO(vec2[i])))
            addition = vec1[i] * log(vec1[i]/vec2[i]);
        distance += addition;
    }

    return distance;
}


void update_weights_new(ModelInput *inputs, double outputs[QUERY_SIZE][CATEGORY_NUM], double (*loss_func)(double*,double*,int)){
    double old_weights[MODEL_NUM] = {0};

    for (int i = 0; i < MODEL_NUM; ++i) {
        old_weights[i] = inputs[i].weight;
    }

    // double numerator[QUERY_SIZE] = {0};
    // double denominators[QUERY_SIZE] = {0};

    double numerators[MODEL_NUM] = {0};
    double denominator = 0;
    double distances[MODEL_NUM][QUERY_SIZE] = {0};
    for (int i = 0; i < MODEL_NUM; ++i) {
        for (int j = 0; j < QUERY_SIZE; ++j) {
            distances[i][j] = loss_func(inputs[i].preds[j], outputs[j], CATEGORY_NUM);
            numerators[i] += distances[i][j]; // numerator of a model is the sum of the losses across all the query(j)
        }
        denominator += numerators[i];
    }

    // update weights
    for (int i = 0; i < MODEL_NUM; ++i) {
        double temp = numerators[i] / denominator;
        if( isnan(temp) || temp == 0 ){
            inputs[i].weight = old_weights[i];
        }else{
            inputs[i].weight = -log(temp);
        }
    }
    // normalize weights, do the softmax
    softmax(inputs,MODEL_NUM);
}



void update_weights(ModelInput *inputs, double outputs[QUERY_SIZE][CATEGORY_NUM], double (*loss_func)(double*,double*,int)){
    double old_weights[MODEL_NUM] = {0};

    for (int i = 0; i < MODEL_NUM; ++i) {
        old_weights[i] = inputs[i].weight;
    }

    double denominators[QUERY_SIZE] = {0};
    double distances[MODEL_NUM][QUERY_SIZE] = {0};
    for (int j = 0; j < QUERY_SIZE; ++j) {
        for (int i = 0; i < MODEL_NUM; ++i) {
            distances[i][j] = loss_func(inputs[i].preds[j], outputs[j], CATEGORY_NUM);
            denominators[j] += distances[i][j];
        }
    }

    // update weights
    for (int i = 0; i < MODEL_NUM; ++i) {
        double temp = 0;
        for (int j = 0; j < QUERY_SIZE; ++j) {
            // denominators[j] will be 0 if all the distances between inputs and outputs are identical.
            denominators[j] = (denominators[j] == 0) ? 1 : denominators[j];
            temp += (distances[i][j]/denominators[j]);
//            temp = isfinite(temp)? 0.5 : temp;
        }
        if( isnan(temp) || temp == 0 ){
            inputs[i].weight = old_weights[i];
        }else{
            inputs[i].weight = -log(temp);
        }
    }
    // normalize weights, do the softmax
    softmax(inputs,MODEL_NUM);
}

void softmax_2(double output[CATEGORY_NUM], int length) {

    double denominator = 0;

    for (int i = 0; i < length; ++i) {
        denominator += exp(output[i]);
    }

    for (int i = 0; i < length; ++i) {
        output[i] = exp(output[i]) / denominator;
    }

}

void turth_discovery(ModelInput *inputs, double outputs[QUERY_SIZE][CATEGORY_NUM],double (*loss_func)(double*,double*,int)){

    int r = 0;
    while(r < ITERATION_NUM){
        // calculate current outputs
        cal_output(inputs, outputs);

        if( DISTANCE == "KL" ){
            for (int j = 0; j < QUERY_SIZE; ++j) {
               softmax_2(outputs[j],CATEGORY_NUM);
            }
        }


        // update weights
        // ocall_print("looping...");
        // update_weights(inputs,outputs, loss_func);
        update_weights_new(inputs,outputs, loss_func);
        r += 1;
    }

}



int combine_result(ModelInput *inputs, CombinerOutput *output) {
	// double outputs[QUERY_SIZE][CATEGORY_NUM] = {{0}};
	ocall_print("Combining results inside enclave...");
	init_model_inputs(inputs);

    if(DISTANCE == "KL"){
	    turth_discovery(inputs, output->preds, get_KL_divergence_distance);
    }else{
    	turth_discovery(inputs,output->preds, get_L2_distance);
    }

	ocall_print("Results combined! Exiting the enclave");
	
	return 0;
}