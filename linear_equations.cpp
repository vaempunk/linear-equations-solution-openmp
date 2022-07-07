#include <omp.h>
#include <stdio.h>
#include <fstream>
#include <string>

int size;
int num_threads;

void reduce(long double **matrix, long double *vector)
{
    for (int placing = 0; placing < size - 1 && placing < size; placing++)
    {
        int i, j;
        long double reduce;
#pragma omp parallel for num_threads(num_threads) default(none) shared(size, placing, matrix, vector) private(i, j, reduce)
        for (i = placing + 1; i < size; i++)
        {
            reduce = matrix[i][placing] / matrix[placing][placing];
            vector[i] -= vector[placing] * reduce;
            for (j = placing; j < size; j++)
            {
                matrix[i][j] -= matrix[placing][j] * reduce;
            }
        }
    }
}

long double *find_solution(long double **matrix, long double *b)
{
    long double *res = new long double[size]{1};

    for (int i = size - 1; i >= 0; i--)
    {
        if (matrix[i][i] != 0)
        {
            long double temp = b[i];
            for (int j = i + 1; j < size; j++)
                temp -= matrix[i][j] * res[j];
            res[i] = temp / matrix[i][i];
        }
    }

    return res;
}

long double *gauss(long double **matrix, long double *vector)
{
    reduce(matrix, vector);
    return find_solution(matrix, vector);
}

bool check_solution(long double **matrix, long double *b, long double *x)
{
    long double *tempb = new long double[size]{0};
    for (int i = 0; i < size; i++)
        for (int j = 0; j < size; j++)
            tempb[i] += matrix[i][j] * x[j];

    for (int i = 0; i < size; i++)
    {
        if ((b[i] != 0) && (abs(tempb[i] - b[i]) / b[i] > 1e-6) ||
            (tempb[i] != 0) && (abs(tempb[i] - b[i]) / tempb[i] > 1e-6))
        {
            printf("error:\n%.20f: %.20f\n", b[i], tempb[i]);
            return false;
        }
    }

    return true;
}

/* generate random matrix and vector in range [0, 99] */
void generate_input(int size)
{
    long double **matrix = new long double *[size];
    std::ofstream matrix_output{std::string{"data/matrix"} + std::to_string(size) + ".txt"};
    for (int i = 0; i < size; i++)
    {
        matrix[i] = new long double[size];
        for (int j = 0; j < size; j++)
        {
            matrix[i][j] = rand() % 100;
            matrix_output << matrix[i][j] << ' ';
        }
        matrix_output << '\n';
    }
    matrix_output.close();

    long double *x = new long double[size];
    for (int i = 0; i < size; i++)
        x[i] = rand() % 100;

    long double *right = new long double[size]{0};
    for (int i = 0; i < size; i++)
        for (int j = 0; j < size; j++)
            right[i] += matrix[i][j] * x[j];
    for (int i = 0; i < size; i++)
        delete[] matrix[i];
    delete[] matrix;
    delete[] x;

    std::ofstream vector_output{std::string{"data/vector"} + std::to_string(size) + ".txt"};
    for (int i = 0; i < size; i++)
        vector_output << right[i] << ' ';
    delete[] right;
    vector_output.close();
}

int main()
{
    printf("enter size: ");
    scanf("%d", &size);

    printf("enter number of threads: ");
    scanf("%d", &num_threads);

    printf("reading matrix...\n");
    long double **matrix = new long double *[size];
    std::ifstream matrix_input{std::string{"data/matrix.txt"}};
    for (int i = 0; i < size; i++)
    {
        matrix[i] = new long double[size];
        for (int j = 0; j < size; j++)
        {
            matrix_input >> matrix[i][j];
        }
    }
    matrix_input.close();

    printf("reading vector...\n");
    long double *vector = new long double[size];
    std::ifstream vector_input{std::string{"data/vector"} + std::to_string(size) + ".txt"};
    for (int i = 0; i < size; i++)
        vector_input >> vector[i];
    vector_input.close();

    printf("finding solution...\n");
    double start = omp_get_wtime();
    long double *solution = gauss(matrix, vector);
    double end = omp_get_wtime();

    printf("checking solution...\n");
    if (check_solution(matrix, vector, solution))
        printf("\n\nSolution is right\n");
    else
        printf("\n\nSolution is wrong\n");

    printf("time: %f\n", end - start);

    printf("writing solution...\n");
    std::ofstream solution_output{"data/solution.txt"};
    for (int i = 0; i < size; i++)
        solution_output << solution[i] << ' ';
    solution_output << ' ';
    solution_output.close();

    for (int i = 0; i < size; i++)
        delete[] matrix[i];
    delete[] matrix;
    delete[] vector;
    delete[] solution;

    return 0;
}