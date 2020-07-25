#include <stdio.h>
#include <ios>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <time.h>
#include "Enclave_u.h"
#include "sgx_urts.h"
#include "sgx_utils/sgx_utils.h"

using namespace std;
const int DATA_TYPE_INT = 0;
const int DATA_TYPE_DOUBLE = 1;

/* Global Enclave ID */
sgx_enclave_id_t global_eid;

/* OCall implementations */
void ocall_print(const char* str) {
    printf("%s\n", str);
}

// implement write to file - use fopen, fwrite
void ocall_write_file(const char* filename, const char* buf, size_t buf_len) {
    FILE *fp;

    // write to file - filename the buf string with len = buf_len
    fp = fopen(filename, "w+");
    if (fp != NULL) {
	int ret_val = fwrite(buf, sizeof(char), buf_len, fp);
	if (ret_val != buf_len) {
		printf ("Failed to write to file - returned value: %d\n", ret_val);
		// not returning anything as we need to close the stream
	}

	int out_fclose = fclose(fp);
	if (out_fclose != 0) {
		printf ("Failed to close the file - %s\n", filename);
		exit(0);
	}
    } else {
	printf ("Failed to open the file - %s - for writing\n", filename);
	exit(0);
    }
}

// implement read from fil - use fopen, fread
void ocall_read_file(const char* filename, char* buf, size_t buf_len) {

    int result;
    FILE *fp;

    fp = fopen(filename, "r");
    if (fp != NULL) {
	result = fread(buf, sizeof(char), buf_len, fp);
	buf[buf_len] = '\0';
	if (result != buf_len) {
		printf ("Reading error from the file - %s\n", filename);
	} else {
		printf ("The first %d chars from the file %s: %s\n", int(buf_len), filename,buf);
	}
	fclose(fp);

    } else {
	printf ("Failed to open the file - %s - for reading\n", filename);
    	exit(1);
    }

}

void readResultsFromFiles(string file_name, double results[QUERY_SIZE][CATEGORY_NUM], int DATA_TYPE){
    // cout << file_name << endl;
    ifstream in(file_name);

    if(!in){
        cout << "File not exists !!!" << endl;
    }

    int i = 0;

    while(in.good()){
        string line;
        int j = 0;
        getline(in,line);

//        cout << line.length() << endl;
        if(line.length() == 0){
            continue;
        }

        stringstream ss(line);
        while(ss.good()){
            string r;
            getline( ss, r, ',' );
            
            results[i][j++] = DATA_TYPE == DATA_TYPE_INT ? stoi(r) : stod(r);
            // cout << " -> " << results[i][j-1];
        }

        i+=1;
        if( i == QUERY_SIZE) break;
    }
    
    in.close();
}



void saveResults(string save_path, double output[QUERY_SIZE][CATEGORY_NUM]){

    ofstream out;
    out.open (save_path);

    for (int j = 0; j < QUERY_SIZE; ++j) {
        for (int k = 0; k < CATEGORY_NUM; ++k) {
            out << output[j][k];
            if (k != CATEGORY_NUM -1){
                out << ",";
            }
        }
        out << "\n";
    }

    out.close();

}


void print_array(double *array, int size){
    for (int i = 0; i < size ; ++i) {
        cout<< array[i] << " ";
    }
    cout << "" << endl;
}






int exp_mia(int is_shadow){
    string prefix = "shadow";
    if(!is_shadow){
        prefix = "target";
    }

    int sum_result;
    sgx_status_t status;
  
    /* Enclave Initialization */ 
    if (initialize_enclave(&global_eid, "enclave.token", "enclave.signed.so") < 0) {
        printf("Fail to initialize enclave.\n");
        return 1;
    }
    // add your codes here
    static ModelInput inputs[3];
    static CombinerOutput output;

    // MNIST
    string model_results[3] = {
        prefix+"_cnn_member.csv",    
        prefix+"_mlp_member.csv",    
        prefix+"_rnn_member.csv"
    };

    // string model_results[3] = {
    //     prefix+"_cnn_non_member.csv",    
    //     prefix+"_mlp_non_member.csv",    
    //     prefix+"_rnn_non_member.csv"
    // };


    clock_t r_start,r_end;
    r_start=clock();
    for(int i=0; i< MODEL_NUM; ++i){
        readResultsFromFiles("../image/mia/"+prefix+"_outputs/"+model_results[i],  inputs[i].preds, DATA_TYPE_DOUBLE);
    }
    r_end = clock();
    cout << "Data reading takes " <<  double(r_end-r_start)/CLOCKS_PER_SEC << "s" << endl;

    
    clock_t start,end;
    start=clock();
    // use sgx to combine the results
    status = combine_result(global_eid, &sum_result, inputs, &output);
    end=clock();

    if (status != SGX_SUCCESS) {
        printf("ECall failed.\n");
        return 1;
    }

    cout << "\nweights:" << endl;
    for(int i=0; i< MODEL_NUM; ++i){
        cout << "model "<< (i+1) <<":"<< inputs[i].weight << endl;
    }

    // print_array(output.preds[0], CATEGORY_NUM);
    // print_array(inputs[0].preds[0], CATEGORY_NUM);

    cout << "Combination takes " <<  double(end-start)/CLOCKS_PER_SEC << "s" << endl;
    
    if(DISTANCE == "KL"){
        saveResults( "" TASK "_" STR(QUERY_SIZE) "_" OTYPE "_iter_" STR(ITERATION_NUM) "_dis_" DISTANCE "_.csv", output.preds);
    }else{
        saveResults( "../image/mia/"+prefix+"_combined_results/"+model_results[0], output.preds);
    }

    // cout.rdbuf(coutbuf);
    return 0;
}

int exp_mia_adv2(){

    int sum_result;
    sgx_status_t status;
  
    /* Enclave Initialization */ 
    if (initialize_enclave(&global_eid, "enclave.token", "enclave.signed.so") < 0) {
        printf("Fail to initialize enclave.\n");
        return 1;
    }
    // add your codes here
    static ModelInput inputs[3];
    static CombinerOutput output;


    // string model_results[3] = {
    //     "dnn_member.csv",    
    //     "rnn_member.csv",    
    //     "svm_member.csv"
    // };

    string model_results[3] = {
        "dnn_non_member.csv",    
        "rnn_non_member.csv",    
        "svm_non_member.csv"
    };


    clock_t r_start,r_end;
    r_start=clock();
    for(int i=0; i< MODEL_NUM; ++i){
        string f_path = "../image/mia/adv2/local_results/"+model_results[i] ;
        cout << "read from :" << f_path << endl;
        readResultsFromFiles(f_path,  inputs[i].preds, DATA_TYPE_DOUBLE);
    }
    r_end = clock();
    cout << "Data reading takes " <<  double(r_end-r_start)/CLOCKS_PER_SEC << "s" << endl;

    
    clock_t start,end;
    start=clock();
    // use sgx to combine the results
    status = combine_result(global_eid, &sum_result, inputs, &output);
    end=clock();

    if (status != SGX_SUCCESS) {
        printf("ECall failed.\n");
        return 1;
    }

    cout << "\nweights:" << endl;
    for(int i=0; i< MODEL_NUM; ++i){
        cout << "model "<< (i+1) <<":"<< inputs[i].weight << endl;
    }

    // print_array(output.preds[0], CATEGORY_NUM);
    // print_array(inputs[0].preds[0], CATEGORY_NUM);

    cout << "Combination takes " <<  double(end-start)/CLOCKS_PER_SEC << "s" << endl;
    
    if(DISTANCE == "KL"){
        saveResults( "" TASK "_" STR(QUERY_SIZE) "_" OTYPE "_iter_" STR(ITERATION_NUM) "_dis_" DISTANCE "_.csv", output.preds);
    }else{
        saveResults( "../image/mia/adv2/combined_result/"+model_results[0], output.preds);
    }

    // cout.rdbuf(coutbuf);
    return 0;
}

int exp_adv(string attack_method){
    
    cout << "attack:" << attack_method << endl;

    int sum_result;
    sgx_status_t status;
  
    /* Enclave Initialization */ 
    if (initialize_enclave(&global_eid, "enclave.token", "enclave.signed.so") < 0) {
        printf("Fail to initialize enclave.\n");
        return 1;
    }
    // add your codes here
    static ModelInput inputs[6];
    static CombinerOutput output;

    // MNIST
    string model_results[6] = {
        "cnn.csv",
        "mlp.csv",
        "rnn.csv",
        "lr.csv" ,
        "knn.csv",
        "svm.csv"
    };

    clock_t r_start,r_end;
    r_start=clock();
    for(int i=0; i< MODEL_NUM; ++i){
        string f_path = "../image/adv/adv_outputs/"+attack_method+"/"+model_results[i];
        cout << "read from :" << f_path << endl;
        readResultsFromFiles(f_path,  inputs[i].preds, DATA_TYPE_DOUBLE);
        readResultsFromFiles(f_path,  inputs[i].preds, DATA_TYPE_DOUBLE);
    }
    r_end = clock();
    cout << "Data reading takes " <<  double(r_end-r_start)/CLOCKS_PER_SEC << "s" << endl;

    
    clock_t start,end;
    start=clock();
    // use sgx to combine the results
    status = combine_result(global_eid, &sum_result, inputs, &output);
    end=clock();

    if (status != SGX_SUCCESS) {
        printf("ECall failed.\n");
        return 1;
    }

    cout << "\nweights:" << endl;
    for(int i=0; i< MODEL_NUM; ++i){
        cout << "model "<< (i+1) <<":"<< inputs[i].weight << endl;
    }

    // print_array(output.preds[0], CATEGORY_NUM);
    // print_array(inputs[0].preds[0], CATEGORY_NUM);

    cout << "Combination takes " <<  double(end-start)/CLOCKS_PER_SEC << "s" << endl;
    
    if(DISTANCE == "KL"){
        saveResults( "" TASK "_" STR(QUERY_SIZE) "_" OTYPE "_iter_" STR(ITERATION_NUM) "_dis_" DISTANCE "_.csv", output.preds);
    }else{
        saveResults( "../image/adv/adv_combined_results/"+attack_method+"/adv_combined.csv", output.preds);
    }

    // cout.rdbuf(coutbuf);
    return 0;
}



int exp1(){
    int sum_result;
    sgx_status_t status;
  

    // LOG redirection
    // ofstream out("LOG_" TASK "_" STR(QUERY_SIZE) "_" OTYPE "_iter_" STR(ITERATION_NUM) "_noise_" STR(NOISE_LEVEL) "_dis_" DISTANCE "_.log");  
    // streambuf *coutbuf=cout.rdbuf(); 
    // cout.rdbuf(out.rdbuf());


    clock_t i_start,i_end;
    i_start=clock();
    /* Enclave Initialization */ 
    if (initialize_enclave(&global_eid, "enclave.token", "enclave.signed.so") < 0) {
        printf("Fail to initialize enclave.\n");
        return 1;
    }
    i_end=clock();
    cout << "Initialization takes " <<  double(i_end-i_start)/CLOCKS_PER_SEC << "s" << endl;

    // add your codes here
    static ModelInput inputs[MODEL_NUM];
    static CombinerOutput output;
    string model_results[MODEL_NUM];

    cout << "Task:" << TASK << endl;
    if (TASK  == "MNIST"){
        model_results[0] = "KNN.csv";    // 3,5
        model_results[1] = "SVM.csv";    // 3,5
        model_results[2] = "CNN.csv";    // 3,5
        model_results[3] = "RNN.csv";    // 5
        model_results[4] = "LR.csv" ;    // 5
        model_results[5] = "MLP.csv";
    }else if(TASK  == "20New"){
        // model_results[MODEL_NUM] = {
        model_results[0] = "knn.csv";                  // 5,9
        model_results[1] = "svm.csv";                  // 5,9
        model_results[2] = "decision_tree.csv";        // 5,9
        model_results[3] = "srandom_forest.csv";       // 5,9
        model_results[4] = "bagging.csv";              // 5,9
        model_results[5] = "boost.csv";                // 9
        model_results[6] = "DNN.csv";                  // 9
        model_results[7] = "CNN.csv";                  // 9
        model_results[8] = "RNN.csv";                  // 9
        model_results[9] = "RCNN.csv";                
        // }
    }else if( TASK == "ImageNet"){
         model_results[0] = "pytorch_AlexNet.csv";        // 7,14
         model_results[1] = "pytorch_DenseNet.csv";       // 7,14    
         model_results[2] = "pytorch_GoogLenet.csv";      // 7,14
         model_results[3] = "pytorch_InceptionV3.csv";    // 7,14
         model_results[4] = "pytorch_MobileNetV2.csv";    // 7,14
         model_results[5] = "pytorch_ResNet101.csv";      // 7,14 
         model_results[6] = "pytorch_ResNet50.csv";       // 7,14
         model_results[7] = "pytorch_VGG16.csv";          // 14
         model_results[8] = "pytorch_VGG19.csv";          // 14
         model_results[9] = "keras_DenseNet.csv";         // 14
         model_results[10] = "keras_InceptionV3.csv";     // 14
         model_results[11] = "keras_MobileNetV2.csv";     // 14
         model_results[12] = "keras_ResNet50.csv";        // 14
         model_results[13] = "keras_VGG16.csv";           // 14
         model_results[14] = "keras_VGG19.csv";
    }

    clock_t r_start,r_end;
    r_start=clock();
    for(int i=0; i< MODEL_NUM; ++i){
        string f_path = "../outputs/" TASK "/query_" STR(QUERY_SIZE) "/type_" OTYPE "/"+model_results[i];
        cout << "read data from: " << f_path << endl;
        readResultsFromFiles( f_path,  inputs[i].preds, DATA_TYPE_DOUBLE);
    }
    r_end = clock();
    cout << "Data reading takes " <<  double(r_end-r_start)/CLOCKS_PER_SEC << "s" << endl;

    
    clock_t start,end;
    start=clock();
    // use sgx to combine the results
    status = combine_result(global_eid, &sum_result, inputs, &output);
    end=clock();

    if (status != SGX_SUCCESS) {
        printf("ECall failed.\n");
        return 1;
    }

    // cout << "\nweights:" << endl;
    // for(int i=0; i< MODEL_NUM; ++i){
    //     cout << "model "<< (i+1) <<":"<< inputs[i].weight << endl;
    // }

    cout << "Combination takes " <<  double(end-start)/CLOCKS_PER_SEC << "s" << endl;
    
    if(DISTANCE == "KL"){
        saveResults( "" TASK "_" STR(QUERY_SIZE) "_" OTYPE "_iter_" STR(ITERATION_NUM) "_dis_" DISTANCE "_.csv", output.preds);
    }else{
        string s_path = "./../outputs_combined/combination_results_new/" TASK "/noise_" STR(NOISE_LEVEL) "/" TASK "_" STR(QUERY_SIZE) "_" OTYPE "_iter_" STR(ITERATION_NUM) ".csv";
        saveResults( s_path, output.preds);
        cout << "Results are saved to " << s_path << endl;
    }
    // cout.rdbuf(coutbuf);
    return 0;
}




int main(int argc, char const *argv[]) {
    exp1();

    // exp for attacks
    // int is_shadow = 0;
    // exp_mia(is_shadow);
    // exp_mia_adv2();
    // exp_adv("cw");
    return 0;
}