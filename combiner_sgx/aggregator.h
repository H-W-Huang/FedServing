#define TASK "MNIST" // 1. MNIST; 2. 20News; 3. ImageNet
#define OTYPE "label" // 1. label; 2. rank; 3.probs
// #define OTYPE "rank"
// #define OTYPE "probs"
#define DISTANCE "L2"  // L2 or KL. KL for OTYPE of probs ONLY.
#define NOISE_LEVEL 0  
// default NOISE_LEVEL is 0
// with NOISE_LEVELincreases, amount of models providing noise outputs ascends.
// If you want to change the NOISE_LEVEL, please modify the '*.csv" of "model_results" in App.cpp first.
// for examples, change "KNN.csv" to "KNN_noise.csv" to turn it to noise outputs.
#define QUERY_SIZE  4000 
// QUERY_SIZE :
// 1. MNIST: (1000, 4000, 7000)
// 2. 20News: (1000, 3000, 5000)
// 3. ImageNet: (5000, 10000, 16000)
#define CATEGORY_NUM 10 // 1. MNIST: 10; 2. 20News: 20; 3. ImageNet: 1000
#define MODEL_NUM 6 // 1. MNIST: 6; 2. 20News: 10; 3. ImageNet: 15
#define ITERATION_NUM 16  
#define MIN_VALUE 1e-9
#define IS_DOUBLE_ZERO(d)  (abs(d) < MIN_VALUE)
#define STR1(R) #R
#define STR(R) STR1(R)


typedef struct ModelInput{
    double preds[QUERY_SIZE][CATEGORY_NUM];
    double weight;
}ModelInput;


typedef struct CombinerOutput{
    double preds[QUERY_SIZE][CATEGORY_NUM];
    int iteration;
}CombinerOutput;
