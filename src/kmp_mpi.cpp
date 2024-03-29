#include "mpi.h"
#include "random"
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>

#define MASTER_CORE 0

void findDomainDecomposition(int text_length, int core_count, int* number_of_char_per_core, int* number_of_char_on_last_core);
void computePrefixSuffix(const char *text, int *P, int textLength);
void computeStrongPrefixSuffix(char *pattern, int *Pp, int textLength);
int kmpAlgorithm(char *text, char *pattern, bool isPSStrong, int *patPositions);
std::string readTextFile(std::string fileName);

int main(int argc, char **argv) {

    // declare basic variables
    int process_rank, number_of_processes;
    int text_length;
    int number_of_char_per_core, number_of_char_on_last_core;
    int length_of_pattern;
    int occurrences;
    std::string received_string;
    std::string pattern;
    std::string data_file_name;
    std::string pattern_file_name;

    bool is_ps_strong;
    double start, end;

    // initialize MPI
    MPI_Init(NULL, NULL);
    MPI_Status status;
    MPI_Comm_rank(MPI_COMM_WORLD, &process_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &number_of_processes);
    MPI_Barrier(MPI_COMM_WORLD);



    // master instructions
    if(process_rank == MASTER_CORE) {

        
        std::string text_data = readTextFile(argv[1]);
        text_length = text_data.length();

        pattern = argv[2];
        length_of_pattern = pattern.length();

        is_ps_strong = argv[3];
        
        start = MPI_Wtime();

        // decompose data
        findDomainDecomposition(text_length, number_of_processes, &number_of_char_per_core, &number_of_char_on_last_core);

        // get masters part of data to process
        std::string tmp_string;
        received_string =
                text_data.substr(0, number_of_char_per_core + length_of_pattern - 1);

        // send all other chunks of data to slave processes
        for (int i = 1; i < number_of_processes; i++) {
            if (i < number_of_processes - 1) {
                tmp_string =
                        text_data.substr(i * number_of_char_per_core, number_of_char_per_core + length_of_pattern - 1);
                MPI_Send(tmp_string.c_str(),  tmp_string.length(), MPI_CHAR, i, 0, MPI_COMM_WORLD);

                // if it is last chunk (which might have different size than other chunks)
            } else {
                tmp_string =
                        text_data.substr(i * number_of_char_per_core, number_of_char_on_last_core);
                MPI_Send(tmp_string.c_str(), tmp_string.length(), MPI_CHAR, i, 0, MPI_COMM_WORLD);

            }

        }

        // slave processes instruction
    } else {
        // probe size of incoming message
        MPI_Probe(MASTER_CORE, 0, MPI_COMM_WORLD, &status);
        int count;
        MPI_Get_count(&status, MPI_CHAR, &count);

        // allocate array to receive message
        char* buf;
        buf = (char*) malloc(count*sizeof(char));

        // receive message and cast it on string
        MPI_Recv(buf, count, MPI_CHAR, MASTER_CORE, 0, MPI_COMM_WORLD, &status);
        std::string buf_string(buf, count);
        received_string = buf_string;

    }

    if(process_rank == MASTER_CORE) {
        for (int i = 1; i < number_of_processes; i++) {
            MPI_Send(pattern.c_str(),  pattern.length(), MPI_CHAR, i, 2, MPI_COMM_WORLD);
        }
    } else {

        MPI_Probe(MASTER_CORE, 2, MPI_COMM_WORLD, &status);
        int count;
        MPI_Get_count(&status, MPI_CHAR, &count);

        // allocate array to receive message
        char* buf;
        buf = (char*) malloc(count*sizeof(char));

        // receive message and cast it on string
        MPI_Recv(buf, count, MPI_CHAR, MASTER_CORE, 2, MPI_COMM_WORLD, &status);
        std::string buf_string(buf, count);
        pattern = buf_string;
    }

    // allocate memory for pattern position arrays
    int *P = new int[strlen(received_string.c_str()) + 1];
    int *patPositions = new int[received_string.length() + 1] {};

    // run kmp algorithm
    occurrences = kmpAlgorithm((char*)received_string.c_str(), (char*)pattern.c_str(), is_ps_strong, patPositions);


    // if this process is not master then send pattern position array to master process
    if(process_rank != MASTER_CORE) {
        MPI_Request request;
        MPI_Isend(patPositions, received_string.length() + 1, MPI_INT, MASTER_CORE, 1, MPI_COMM_WORLD, &request);
        MPI_Request request2;
        MPI_Isend(&occurrences, 1, MPI_INT, MASTER_CORE, 3, MPI_COMM_WORLD, &request2);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    if(process_rank == MASTER_CORE) {
        // this process is master
        std::vector<int> indexes;

        for (int j = 0; j < received_string.length() + 1; j++) {
            if (patPositions[j] != 0) {
                indexes.push_back(patPositions[j]);

            } else {
                break;
            }
        }

        // receive pattern position arrays from slave processes
        for (int i = 1; i < number_of_processes; i++) {
            // probe size of array
            MPI_Probe(i, 1, MPI_COMM_WORLD, &status);
            int count;
            MPI_Get_count(&status, MPI_INT, &count);

            // allocate memory for buffer array
            int* buf;
            buf = (int*) malloc(count*(sizeof(int) + 1));
            MPI_Recv(buf, count, MPI_INT, i, 1, MPI_COMM_WORLD, &status);

            int occurences_buf;
            MPI_Recv(&occurences_buf, 1, MPI_INT, i, 3, MPI_COMM_WORLD, &status);
            occurrences += occurences_buf;
            // rewrite indexes of pattern occurrence from local arrays to global vector
            // with proper offset
            for (int j = 0; j < count; j++) {
                if(buf[j] != 0) {
                    indexes.push_back(i*count + buf[j] - 1);
                } else {
                    break;
                }

            }
        }
        std::cout << " Number of occurrences: " << occurrences << std::endl;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    end = MPI_Wtime();

    // finalize mpi
    MPI_Finalize();

    if (process_rank == MASTER_CORE) {
        printf("Runtime = %f\n", end-start);
    }

    return 0;
}

void findDomainDecomposition(int text_length, int core_count, int* number_of_char_per_core, int* number_of_char_on_last_core) {
    // this function decompose domain taking into account declared number of processes
    *number_of_char_per_core = text_length/core_count;
    *number_of_char_on_last_core = text_length/core_count + text_length%core_count;
}

void computePrefixSuffix(const char *text, int *P, int textLength) {
    // this function computes prefix-suffix table for MP algorithm
    P[0] = -1;
    int t = -1;
    for (int j = 1; j <= textLength; j++) {
        while (t >= 0 && text[t] != text[j - 1]) {
            t = P[t];
        }
        t++;
        P[j] = t;
    }
}

void computeStrongPrefixSuffix(char *pattern, int *Pp, int textLength) {
    // this function computes strong prefix-suffix table for KMP algorithm
    Pp[0] = -1;
    int t  = -1;
    for (int j = 1; j <= textLength; j++) {
        while (t >= 0 && pattern[t] != pattern[j - 1]) {
            t = Pp[t];
        }
        t++;
        if (j == textLength | pattern[t] != pattern[j]) {
            Pp[j] = t;
        } else {
            Pp[j] = Pp[t];
        }

    }

}

int kmpAlgorithm(char *text, char *pattern, bool isPSStrong, int *patPositions) {
    // MP and KMP algorithm function, MP and KMP only differs with prefix-suffix tables
    int textLength = strlen(text);
    int patLength = strlen(pattern);
    int pLength = textLength + 1;
    int *P = new int[pLength];
    int occurrences = 0;

    int i = 0;
    int j = 0;
    if (isPSStrong)
        computeStrongPrefixSuffix(pattern, P, patLength);
    else
        computePrefixSuffix(pattern, P, patLength);

    for (i = 0; i < textLength - patLength + 1; i = i + j - P[j]){
        if (i > 0)
            j = std::max(0, P[j]);
        while (j < patLength && pattern[j] == text[i + j])
            j++;
        // pattern found
        if (j == patLength) {
            patPositions[occurrences] = i;
            occurrences++;
        }
    }
    delete[] P;
    return occurrences;
}

std::string readTextFile(std::string fileName) {
    // utility function for reading text files
    std::ifstream file(fileName);
    if (file.fail()) {
        std::cout << fileName << " file does not exist, exiting" << std::endl;
        exit(1);
    }
    std::stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}
