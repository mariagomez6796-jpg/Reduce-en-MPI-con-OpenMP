#include "OperacionesArreglos.h"

#include <random>
#include <omp.h>


// ============================================================
// 6. LLENAR SECUENCIAL
// ============================================================

void OperacionesArreglos::llenarSecuencialMPI(
    int* A,
    int n,
    char* nombrePC,
    int nodo,
    int inicioGlobal
) {

    #pragma omp parallel for
    for (int i = 0; i < n; i++) {

        A[i] = inicioGlobal + i + 1;
    }
}


// ============================================================
// 7. LLENAR ALEATORIO
// ============================================================

void OperacionesArreglos::llenarAleatorioMPI(
    int* A,
    int n,
    char* nombrePC,
    int nodo,
    int inicioGlobal
) {

    #pragma omp parallel
    {

        int hilo = omp_get_thread_num();

        std::random_device rd;

        unsigned int semilla =
            rd()
            ^ (nodo * 10007)
            ^ (hilo * 7919);

        std::mt19937 generador(semilla);

        std::uniform_int_distribution<int>
            distribucion(1, 1000000);


        #pragma omp for
        for (int i = 0; i < n; i++) {

            A[i] = distribucion(generador);
        }
    }
}


// ============================================================
// 8. SUMATORIA
// ============================================================

long long OperacionesArreglos::sumatoriaMPI(
    int* A,
    int n,
    char* nombrePC,
    int nodo,
    int inicioGlobal
) {

    long long suma = 0;

    #pragma omp parallel for reduction(+:suma)
    for (int i = 0; i < n; i++) {

        suma += A[i];
    }

    return suma;
}


// ============================================================
// 9. PROMEDIO
// ============================================================

double OperacionesArreglos::promedioMPI(
    int* A,
    int n,
    char* nombrePC,
    int nodo,
    int inicioGlobal
) {

    long long suma = 0;

    #pragma omp parallel for reduction(+:suma)
    for (int i = 0; i < n; i++) {

        suma += A[i];
    }

    return static_cast<double>(suma) / n;
}


// ============================================================
// 10. MAXIMO
// ============================================================

int OperacionesArreglos::maximoMPI(
    int* A,
    int n,
    char* nombrePC,
    int nodo,
    int inicioGlobal
) {

    int maximo = A[0];

    #pragma omp parallel for reduction(max:maximo)
    for (int i = 0; i < n; i++) {

        if (A[i] > maximo) {

            maximo = A[i];
        }
    }

    return maximo;
}


// ============================================================
// 11. MINIMO
// ============================================================

int OperacionesArreglos::minimoMPI(
    int* A,
    int n,
    char* nombrePC,
    int nodo,
    int inicioGlobal
) {

    int minimo = A[0];

    #pragma omp parallel for reduction(min:minimo)
    for (int i = 0; i < n; i++) {

        if (A[i] < minimo) {

            minimo = A[i];
        }
    }

    return minimo;
}