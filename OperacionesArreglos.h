#ifndef OPERACIONES_ARREGLOS_H
#define OPERACIONES_ARREGLOS_H

class OperacionesArreglos {

public:

    void llenarSecuencialMPI(
        int* A,
        int n,
        char* nombrePC,
        int nodo,
        int inicioGlobal
    );

    void llenarAleatorioMPI(
        int* A,
        int n,
        char* nombrePC,
        int nodo,
        int inicioGlobal
    );

    long long sumatoriaMPI(
        int* A,
        int n,
        char* nombrePC,
        int nodo,
        int inicioGlobal
    );

    double promedioMPI(
        int* A,
        int n,
        char* nombrePC,
        int nodo,
        int inicioGlobal
    );

    int maximoMPI(
        int* A,
        int n,
        char* nombrePC,
        int nodo,
        int inicioGlobal
    );

    int minimoMPI(
        int* A,
        int n,
        char* nombrePC,
        int nodo,
        int inicioGlobal
    );
};

#endif