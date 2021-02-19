#include <iostream>
#include "C:\Program Files (x86)\Microsoft SDKs\MPI\Include/mpi.h"
#include <cmath>

const double REQUIRED_ACCURACY = 0.00000001;


double *mulpMatrixToVector(double *Matrix, double *vector, int N, int stringsNumber) {
    auto resVector = new double[N];
    //for(int i = 0;i < stringsNumber;i++) std::cout << vector[i] << " ";
    for (int i = 0; i < N; i++) {
        double sum = 0;
        for (int j = 0; j < stringsNumber; j++) {
            sum += vector[j] * Matrix[i + j * N];
        }

        resVector[i] = sum;
    }
    return resVector;
}

double *
subtractVectorPartsAndFullVector(double *deductedVectorPart, double *reducedFullVector, int partSize, int startIndex) {
    auto *result = new double[partSize];
    for (int i = 0; i < partSize; i++) {
        result[i] = reducedFullVector[startIndex + i] - deductedVectorPart[i];
    }
    return result;
}

void subtractVectorParts(double *firstPart, double *secondPart, int columnSize) {
    for (int i = 0; i < columnSize; i++) {
        firstPart[i] -= secondPart[i];
    }
}

void mulpVectorByNumber(double *vector, int partSize, double number) {
    for (int i = 0; i < partSize; i++) {
        vector[i] *= number;
    }
}

double getVectorPartLength(double *vectorPart, int partSize) {
    double sum = 0;
    for (int i = 0; i < partSize; i++) sum += (vectorPart[i] * vectorPart[i]);
    return sum;
}

void transposeMatrix(double *matrix, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < i; j++) {
            double t = matrix[j * N + i];
            matrix[j * N + i] = matrix[i * N + j];
            matrix[i * N + j] = t;
        }
    }
}

int main(int argc, char **argv) {
    int N = 7;
    double t = 0.01;

    MPI_Init(&argc, &argv);
    int procNum, rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &procNum);
    if (rank == 0) {
        std::cout << "Number of process : " << procNum << std::endl;
    }
    auto A = new double[N * N];
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A[i * N + j] = (i == j) ? 2.0 : 1.0;
        }
    }


    double *strings;
    int stringCount = N / procNum;

    strings = new double[N * stringCount];

    double startTime = MPI_Wtime();
    if (rank == 0) {
        transposeMatrix(A, N);
    }

    MPI_Scatter(A, N * stringCount, MPI_DOUBLE, strings, N * stringCount, MPI_DOUBLE, 0, MPI_COMM_WORLD);


    if (N % procNum != 0) {
        if (rank == 0) {
            for (int i = 0; i < (N % procNum); i++) {
                MPI_Send(A + N * stringCount * procNum + i * N, N, MPI_DOUBLE, procNum - 1 - i, 123, MPI_COMM_WORLD);
            }
        }
        if (rank >= procNum - (N % procNum)) {
            stringCount++;
            strings = (double *) realloc(strings, sizeof(double) * N * stringCount);

            MPI_Recv(strings + (stringCount - 1) * N, N, MPI_DOUBLE, 0, 123, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }

    int *recvcounts = new int[procNum];
    int *displs = new int[procNum];
    if (rank == 0) {
        for (int i = 0; i < procNum; i++) recvcounts[i] = N / procNum;
        if (N % procNum != 0) {
            for (int j = procNum - (N % procNum); j < procNum; j++) recvcounts[j]++;
        }

        for (int i = 0; i < procNum; i++) displs[i] = i * stringCount;
        if (N % procNum != 0) {
            for (int j = procNum - (N % procNum), i = 0; j < procNum; j++, i++) {
                displs[j] += i;
            }
        }
    }

    auto x = new double[stringCount];
    for (int i = 0; i < stringCount; i++)
        x[i] = 0.;

    auto b = new double[stringCount];
    for (int i = 0; i < stringCount; i++) b[i] = N + 1;
    double partsumB = getVectorPartLength(b, stringCount);
    double lengthB;

    MPI_Allreduce(&partsumB, &lengthB, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);


    double flaw = 1;
    double prevFlaw = 1;

    while (flaw > REQUIRED_ACCURACY) {

//// A*xn
        double *mulpMatrixPartSum = mulpMatrixToVector(strings, x, N, stringCount);


        double *mulpMatrixRes = new double[N];
        MPI_Allreduce(mulpMatrixPartSum, mulpMatrixRes, N, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        delete[] mulpMatrixPartSum;
//
//// A*Xn - b
        double *tempRes;
        if (rank < procNum - (N % procNum)) {

            tempRes = subtractVectorPartsAndFullVector(b, mulpMatrixRes, stringCount, rank * stringCount);
        } else
            tempRes = subtractVectorPartsAndFullVector(b, mulpMatrixRes, stringCount,
                                                       ((procNum - (N % procNum)) * (stringCount - 1)));

        double partLength = getVectorPartLength(tempRes, stringCount);

        double length;
        MPI_Allreduce(&partLength, &length, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        flaw = sqrt(length / lengthB);

        if (flaw < REQUIRED_ACCURACY) {

            double endTime = MPI_Wtime();
            auto* result = new double[N];
            MPI_Gatherv(x,stringCount,MPI_DOUBLE,result,recvcounts,displs,MPI_DOUBLE,0,MPI_COMM_WORLD);
            if(rank == 0) {
                std::cout << "\nTIME: " << endTime - startTime << " sec" << std::endl;
                std::cout << "X: ";
                for(int i = 0;i < N;i++) std::cout << result[i] << " ";
            }
            delete[] result;
            delete[] tempRes;
            break;
        }

        if (flaw > prevFlaw) {
            t = -t * 0.98;
            prevFlaw = flaw;
        }

        mulpVectorByNumber(tempRes, stringCount, t);

        subtractVectorParts(tempRes, x, stringCount);

        mulpVectorByNumber(tempRes, stringCount, -1.0);

        memcpy(x,tempRes,sizeof(double)*stringCount);


    }

    MPI_Finalize();
    return 0;
}
