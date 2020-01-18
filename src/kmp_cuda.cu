#include <algorithm> 
#include <iostream>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include "time.h"

using namespace std;

void mpTable(char* pattern, int *P) {
    int m = strlen(pattern);
    int i, j;
 
    i = 0;
    j = P[0] = -1;

    while (i < m) {
       while (j > -1 && pattern[i] != pattern[j])
          j = P[j];
       P[++i] = ++j;
    }
}

void kmpTable(char* pattern, int *P) {
    int m = strlen(pattern);
    int k;
    
    P[0] = -1;

    for (int i = 1; i < m; i++)
    {
        k = P[i - 1];
        while (k >= 0)
        {
            if (pattern[k] == pattern[i - 1])
                break;
            else
                k = P[k];
        }
        P[i] = k + 1;
    }
}

//check whether target string contains pattern 
__global__ void kmpAlgorithm(char *text, char *pattern, int *P,int *pat_positions, int pattern_length, int text_length) {
    // get current cuda thread id
    int index = blockIdx.x*blockDim.x + threadIdx.x;

    // assign "pointer" to beginning and end of this thread data chunk
    int i = pattern_length * index;
    int j = pattern_length * (index + 2) - 1;

    // if this is end of text then return
    if(i > text_length)
        return;

    // snap end pointer to end of text if it falls beyond
    if(j > text_length)
        j = text_length;

    int k = 0;        

    // do kmp algorithm for chunk of text
    while (i < j)
    {
        if (k == -1)
        {
            i++;
            k = 0;
        }
        else if (text[i] == pattern[k])
        {
            i++;
            k++;
            if (k == pattern_length)
            {
                pat_positions[i - pattern_length] = i - pattern_length;
                i = i - k + 1;
            }
        }
        else
            k = P[k];
    }
    return;
}
 
int main(int argc, char* argv[]) {   
    bool is_kmp = 1;

    int n_cuda_threads = 1024;

    char *text_data;
    char *pattern;
    int *P;
    int *pat_positions;

    char *d_text_data;
    char *d_pattern;
    int *d_P;
    int *d_pat_positions;


    std::ifstream file(argv[1]);

    if (file.fail()) {
        std::cout << argv[1] << " file does not exist, exiting" << std::endl;
        exit(1);
    }

    std::stringstream buffer;
    buffer << file.rdbuf();
    std::string string_buffer = buffer.str();
    text_data = (char*)string_buffer.c_str();


    pattern = argv[2];
    is_kmp = argv[3];

    int text_length = strlen(text_data);
    int pattern_length = strlen(pattern);

    P = new int[text_length];
    pat_positions = new int[text_length];

    std::fill_n(pat_positions, text_length, -1);   

    if(is_kmp)
        kmpTable(pattern, P);
    else
        mpTable(pattern, P);
 
    cudaMalloc((void **)&d_text_data, text_length*sizeof(char));
    cudaMalloc((void **)&d_pattern, pattern_length*sizeof(char));
    cudaMalloc((void **)&d_P, text_length*sizeof(int));
    cudaMalloc((void **)&d_pat_positions, text_length*sizeof(int));

    cudaMemcpy(d_text_data, text_data, text_length*sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pattern, pattern, pattern_length*sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_P, P, text_length*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pat_positions, pat_positions, text_length*sizeof(int), cudaMemcpyHostToDevice);

    float elapsed_time = 0;
    cudaEvent_t start_time, stop_time;

    cudaEventCreate(&start_time);
    cudaEventCreate(&stop_time);

    cudaEventRecord(start_time, 0); 

    kmpAlgorithm<<<(text_length/pattern_length + n_cuda_threads)/n_cuda_threads, n_cuda_threads>>>(d_text_data, d_pattern, d_P, d_pat_positions, pattern_length, text_length);

    cudaEventRecord(stop_time, 0); 
 
    cudaEventSynchronize(start_time);    
    cudaEventSynchronize(stop_time);    
    cudaEventElapsedTime(&elapsed_time, start_time, stop_time);  


    printf("KMP algorithm finished, elapsed time: %f s \n", elapsed_time/1000);  
    
    cudaMemcpy(pat_positions, d_pat_positions, text_length*sizeof(int), cudaMemcpyDeviceToHost);

    // Count all occurrences
    int occurrences = 0;
    for(int i = 0; i < text_length; i++)
    { 
        if(pat_positions[i] != -1)
        {
            occurrences++;
        }
    }

    printf("Number of occurences: %d\n", occurrences);

    cudaFree(d_text_data); 
    cudaFree(d_pattern);
    cudaFree(d_P);
    cudaFree(pat_positions);

    return 0;
}
