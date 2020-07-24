#define TASK "20News"
#define OTYPE "label"
// #define OTYPE "rank"
// #define OTYPE "probs"
#define DISTANCE "L2"
#define NOISE_LEVEL 0
#define QUERY_SIZE  5000
#define CATEGORY_NUM 20
#define MODEL_NUM 10
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
