#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define INPUT_SIZE 2
#define HIDDEN_SIZE 2
#define OUTPUT_SIZE 1
#define LEARNING_RATE 0.1
#define ITERATIONS 10000

double sigmoid(double x) {
    return 1 / (1 + exp(-x));
}

void forward(double input[INPUT_SIZE], double hidden[HIDDEN_SIZE], double output[OUTPUT_SIZE], double weights_ih[INPUT_SIZE][HIDDEN_SIZE], double weights_ho[HIDDEN_SIZE][OUTPUT_SIZE]) {
    // Calculate hidden layer
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        hidden[i] = 0;
        for (int j = 0; j < INPUT_SIZE; j++) {
            hidden[i] += input[j] * weights_ih[j][i];
        }
        hidden[i] = sigmoid(hidden[i]);
    }

    // Calculate output layer
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        output[i] = 0;
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            output[i] += hidden[j] * weights_ho[j][i];
        }
        output[i] = sigmoid(output[i]);
    }
}

void train(double input[INPUT_SIZE], double target[OUTPUT_SIZE], double weights_ih[INPUT_SIZE][HIDDEN_SIZE], double weights_ho[HIDDEN_SIZE][OUTPUT_SIZE]) {
    double hidden[HIDDEN_SIZE];
    double output[OUTPUT_SIZE];

    // Forward pass
    forward(input, hidden, output, weights_ih, weights_ho);

    // Backpropagation
    double output_error[OUTPUT_SIZE];
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        output_error[i] = target[i] - output[i];
    }

    double hidden_error[HIDDEN_SIZE];
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        hidden_error[i] = 0;
        for (int j = 0; j < OUTPUT_SIZE; j++) {
            hidden_error[i] += output_error[j] * weights_ho[i][j];
        }
    }

    // Update weights between hidden and output layer
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        for (int j = 0; j < OUTPUT_SIZE; j++) {
            weights_ho[i][j] += LEARNING_RATE * output_error[j] * output[j] * (1 - output[j]) * hidden[i];
        }
    }

    // Update weights between input and hidden layer
    for (int i = 0; i < INPUT_SIZE; i++) {
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            weights_ih[i][j] += LEARNING_RATE * hidden_error[j] * hidden[j] * (1 - hidden[j]) * input[i];
        }
    }
}

int main() {
    // Initialize weights randomly
    double weights_ih[INPUT_SIZE][HIDDEN_SIZE];
    double weights_ho[HIDDEN_SIZE][OUTPUT_SIZE];
    for (int i = 0; i < INPUT_SIZE; i++) {
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            weights_ih[i][j] = ((double)rand() / RAND_MAX) * 2 - 1; // Random value between -1 and 1
        }
    }
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        for (int j = 0; j < OUTPUT_SIZE; j++) {
            weights_ho[i][j] = ((double)rand() / RAND_MAX) * 2 - 1; // Random value between -1 and 1
        }
    }

    // Training data (XOR)
    double inputs[4][INPUT_SIZE] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    double targets[4][OUTPUT_SIZE] = {{0}, {1}, {1}, {0}};

    // Training loop
    for (int i = 0; i < ITERATIONS; i++) {
        int index = rand() % 4; // Randomly choose one sample
        train(inputs[index], targets[index], weights_ih, weights_ho);
    }

    // Test the trained model
    printf("Testing trained model:\n");
    for (int i = 0; i < 4; i++) {
        double hidden[HIDDEN_SIZE];
        double output[OUTPUT_SIZE];
        forward(inputs[i], hidden, output, weights_ih, weights_ho);
        printf("Input: %d %d, Output: %f\n", (int)inputs[i][0], (int)inputs[i][1], output[0]);
    }

    return 0;
}
