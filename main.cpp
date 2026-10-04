#include <mpi.h>
#include <omp.h>

#include <iostream>
#include <iomanip>
#include <climits>

#include "OperacionesArreglos.h"


int main(int argc, char* argv[]) {

    // ========================================================
    // INICIAR MPI
    // ========================================================

    MPI_Init(&argc, &argv);

    int nodo;
    int totalNodos;

    MPI_Comm_rank(MPI_COMM_WORLD, &nodo);
    MPI_Comm_size(MPI_COMM_WORLD, &totalNodos);


    // ========================================================
    // OBTENER NOMBRE DE LA COMPUTADORA
    // ========================================================

    char nombrePC[MPI_MAX_PROCESSOR_NAME];
    int longitudNombre;

    MPI_Get_processor_name(
        nombrePC,
        &longitudNombre
    );


    // ========================================================
    // DATOS GENERALES
    // ========================================================

    OperacionesArreglos operaciones;

    int opcion = -1;

    const int tamanio = 4000000;

    int* A = nullptr;

    int trabajadores = totalNodos - 1;


    // ========================================================
    // VALIDAR 3 PROCESOS
    // ========================================================

    if (totalNodos != 3) {

        if (nodo == 0) {

            std::cout
                << "\nERROR: La ejecucion distribuida "
                << "requiere 3 procesos MPI.\n";

            std::cout
                << "MPI 0 = Maestro\n";

            std::cout
                << "MPI 1 = Trabajador 1\n";

            std::cout
                << "MPI 2 = Trabajador 2\n";
        }

        MPI_Finalize();

        return 0;
    }


    // ========================================================
    // DISTRIBUCION
    // ========================================================

    int elementosPorTrabajador =
        tamanio / trabajadores;


    // ========================================================
    // IDENTIFICAR COMPUTADORAS
    // ========================================================

    MPI_Barrier(MPI_COMM_WORLD);

    for (int i = 0; i < totalNodos; i++) {

        if (nodo == i) {

            std::cout
                << "\n----------------------------------------\n";

            std::cout
                << "PC: "
                << nombrePC
                << std::endl;

            std::cout
                << "Proceso MPI: "
                << nodo
                << " de "
                << totalNodos
                << std::endl;

            std::cout
                << "Hilos OpenMP disponibles: "
                << omp_get_max_threads()
                << std::endl;


            if (nodo == 0) {

                std::cout
                    << "ROL: MAESTRO"
                    << std::endl;
            }

            else {

                int inicioGlobal =
                    (nodo - 1)
                    * elementosPorTrabajador;

                int finGlobal =
                    inicioGlobal
                    + elementosPorTrabajador
                    - 1;


                std::cout
                    << "ROL: TRABAJADOR"
                    << std::endl;

                std::cout
                    << "Rango asignado: A["
                    << inicioGlobal
                    << "] - A["
                    << finGlobal
                    << "]"
                    << std::endl;

                std::cout
                    << "Elementos asignados: "
                    << elementosPorTrabajador
                    << std::endl;
            }


            std::cout
                << "----------------------------------------\n";
        }


        MPI_Barrier(MPI_COMM_WORLD);
    }


    // ========================================================
    // INFORMACION GENERAL
    // ========================================================

    if (nodo == 0) {

        std::cout
            << "\n========================================\n";

        std::cout
            << "   MPI + OPENMP DISTRIBUIDO\n";

        std::cout
            << "========================================\n";

        std::cout
            << "Computadora maestra: "
            << nombrePC
            << std::endl;

        std::cout
            << "Procesos MPI: "
            << totalNodos
            << std::endl;

        std::cout
            << "Trabajadores: "
            << trabajadores
            << std::endl;

        std::cout
            << "Elementos totales: "
            << tamanio
            << std::endl;

        std::cout
            << "Elementos por trabajador: "
            << elementosPorTrabajador
            << std::endl;

        std::cout
            << "Valores aleatorios: 1 - 1000000\n";

        std::cout
            << "========================================\n";
    }


    // ========================================================
    // MENU
    // ========================================================

    do {

        if (nodo == 0) {

            std::cout
                << "\n============== MENU ==============\n";

            std::cout
                << "1. Crear arreglo\n";

            std::cout
                << "6. Llenar secuencial\n";

            std::cout
                << "7. Llenar aleatorio\n";

            std::cout
                << "8. Sumatoria\n";

            std::cout
                << "9. Promedio\n";

            std::cout
                << "10. Maximo\n";

            std::cout
                << "11. Minimo\n";

            std::cout
                << "0. Salir\n";

            std::cout
                << "==================================\n";

            std::cout
                << "Seleccione una opcion: ";

            std::cin >> opcion;
        }


        // ====================================================
        // ENVIAR OPCION A TODOS
        // ====================================================

        MPI_Bcast(
            &opcion,
            1,
            MPI_INT,
            0,
            MPI_COMM_WORLD
        );


        // ====================================================
        // 1. CREAR ARREGLO
        // ====================================================

        if (opcion == 1) {

            MPI_Barrier(MPI_COMM_WORLD);

            double inicio =
                MPI_Wtime();


            if (nodo == 0) {

                if (A != nullptr) {

                    delete[] A;

                    A = nullptr;
                }


                A =
                    new int[tamanio];


                #pragma omp parallel for
                for (int i = 0; i < tamanio; i++) {

                    A[i] = 0;
                }
            }


            MPI_Barrier(MPI_COMM_WORLD);

            double fin =
                MPI_Wtime();


            if (nodo == 0) {

                std::cout
                    << "\nArreglo A creado correctamente.\n";

                std::cout
                    << "Elementos: "
                    << tamanio
                    << std::endl;

                std::cout
                    << "Memoria aproximada: "
                    << (
                        static_cast<double>(
                            tamanio * sizeof(int)
                        ) / (1024.0 * 1024.0)
                    )
                    << " MB\n";

                std::cout
                    << "Tiempo: "
                    << (fin - inicio)
                    << " segundos\n";
            }
        }


        // ====================================================
        // 6. LLENAR SECUENCIAL
        // ====================================================

        else if (opcion == 6) {

            MPI_Barrier(MPI_COMM_WORLD);

            double inicio =
                MPI_Wtime();


            if (nodo != 0) {

                int cantidad =
                    elementosPorTrabajador;

                int inicioGlobal =
                    (nodo - 1)
                    * elementosPorTrabajador;

                int finGlobal =
                    inicioGlobal
                    + cantidad
                    - 1;


                std::cout
                    << "\nPC: "
                    << nombrePC
                    << " | MPI: "
                    << nodo
                    << " | Procesando A["
                    << inicioGlobal
                    << "] - A["
                    << finGlobal
                    << "]"
                    << std::endl;


                int* parte =
                    new int[cantidad];


                operaciones.llenarSecuencialMPI(
                    parte,
                    cantidad,
                    nombrePC,
                    nodo,
                    inicioGlobal
                );


                MPI_Send(
                    parte,
                    cantidad,
                    MPI_INT,
                    0,
                    100,
                    MPI_COMM_WORLD
                );


                delete[] parte;
            }

            else {

                if (A == nullptr) {

                    A =
                        new int[tamanio];
                }


                for (
                    int trabajador = 1;
                    trabajador < totalNodos;
                    trabajador++
                ) {

                    int inicioGlobal =
                        (trabajador - 1)
                        * elementosPorTrabajador;


                    MPI_Recv(
                        A + inicioGlobal,
                        elementosPorTrabajador,
                        MPI_INT,
                        trabajador,
                        100,
                        MPI_COMM_WORLD,
                        MPI_STATUS_IGNORE
                    );
                }
            }


            MPI_Barrier(MPI_COMM_WORLD);

            double fin =
                MPI_Wtime();


            if (nodo == 0) {

                std::cout
                    << "\nArreglo secuencial generado.\n";

                std::cout
                    << "Primer elemento: "
                    << A[0]
                    << std::endl;

                std::cout
                    << "Ultimo elemento: "
                    << A[tamanio - 1]
                    << std::endl;

                std::cout
                    << "Tiempo distribuido: "
                    << (fin - inicio)
                    << " segundos\n";
            }
        }


        // ====================================================
        // 7. LLENAR ALEATORIO
        // ====================================================

        else if (opcion == 7) {

            MPI_Barrier(MPI_COMM_WORLD);

            double inicio =
                MPI_Wtime();


            if (nodo != 0) {

                int cantidad =
                    elementosPorTrabajador;

                int inicioGlobal =
                    (nodo - 1)
                    * elementosPorTrabajador;

                int finGlobal =
                    inicioGlobal
                    + cantidad
                    - 1;


                std::cout
                    << "\nPC: "
                    << nombrePC
                    << " | MPI: "
                    << nodo
                    << " | Generando A["
                    << inicioGlobal
                    << "] - A["
                    << finGlobal
                    << "]"
                    << std::endl;


                int* parte =
                    new int[cantidad];


                operaciones.llenarAleatorioMPI(
                    parte,
                    cantidad,
                    nombrePC,
                    nodo,
                    inicioGlobal
                );


                MPI_Send(
                    parte,
                    cantidad,
                    MPI_INT,
                    0,
                    200,
                    MPI_COMM_WORLD
                );


                delete[] parte;
            }

            else {

                if (A == nullptr) {

                    A =
                        new int[tamanio];
                }


                for (
                    int trabajador = 1;
                    trabajador < totalNodos;
                    trabajador++
                ) {

                    int inicioGlobal =
                        (trabajador - 1)
                        * elementosPorTrabajador;


                    MPI_Recv(
                        A + inicioGlobal,
                        elementosPorTrabajador,
                        MPI_INT,
                        trabajador,
                        200,
                        MPI_COMM_WORLD,
                        MPI_STATUS_IGNORE
                    );
                }
            }


            MPI_Barrier(MPI_COMM_WORLD);

            double fin =
                MPI_Wtime();


            if (nodo == 0) {

                std::cout
                    << "\nArreglo aleatorio generado correctamente.\n";

                std::cout
                    << "Elementos: "
                    << tamanio
                    << std::endl;

                std::cout
                    << "Rango: 1 - 1000000\n";

                std::cout
                    << "Tiempo distribuido: "
                    << (fin - inicio)
                    << " segundos\n";
            }
        }


        // ====================================================
        // 8. SUMATORIA
        // ====================================================

        else if (opcion == 8) {

            MPI_Barrier(MPI_COMM_WORLD);

            double inicio =
                MPI_Wtime();


            long long sumaParcial = 0;
            long long sumaTotal = 0;


            if (nodo == 0) {

                for (
                    int trabajador = 1;
                    trabajador < totalNodos;
                    trabajador++
                ) {

                    int inicioGlobal =
                        (trabajador - 1)
                        * elementosPorTrabajador;


                    MPI_Send(
                        A + inicioGlobal,
                        elementosPorTrabajador,
                        MPI_INT,
                        trabajador,
                        300,
                        MPI_COMM_WORLD
                    );
                }
            }

            else {

                int cantidad =
                    elementosPorTrabajador;

                int inicioGlobal =
                    (nodo - 1)
                    * elementosPorTrabajador;


                int* parte =
                    new int[cantidad];


                MPI_Recv(
                    parte,
                    cantidad,
                    MPI_INT,
                    0,
                    300,
                    MPI_COMM_WORLD,
                    MPI_STATUS_IGNORE
                );


                sumaParcial =
                    operaciones.sumatoriaMPI(
                        parte,
                        cantidad,
                        nombrePC,
                        nodo,
                        inicioGlobal
                    );


                std::cout
                    << "\nPC: "
                    << nombrePC
                    << " | MPI: "
                    << nodo
                    << " | SUMA PARCIAL: "
                    << sumaParcial
                    << std::endl;


                delete[] parte;
            }


            MPI_Reduce(
                &sumaParcial,
                &sumaTotal,
                1,
                MPI_LONG_LONG,
                MPI_SUM,
                0,
                MPI_COMM_WORLD
            );


            MPI_Barrier(MPI_COMM_WORLD);

            double fin =
                MPI_Wtime();


            if (nodo == 0) {

                std::cout
                    << "\n====================================\n";

                std::cout
                    << "SUMATORIA TOTAL: "
                    << sumaTotal
                    << std::endl;

                std::cout
                    << "Tiempo distribuido: "
                    << (fin - inicio)
                    << " segundos\n";

                std::cout
                    << "====================================\n";
            }
        }


        // ====================================================
        // 9. PROMEDIO
        // ====================================================

        else if (opcion == 9) {

            MPI_Barrier(MPI_COMM_WORLD);

            double inicio =
                MPI_Wtime();


            long long sumaParcial = 0;
            long long sumaTotal = 0;


            if (nodo == 0) {

                for (
                    int trabajador = 1;
                    trabajador < totalNodos;
                    trabajador++
                ) {

                    int inicioGlobal =
                        (trabajador - 1)
                        * elementosPorTrabajador;


                    MPI_Send(
                        A + inicioGlobal,
                        elementosPorTrabajador,
                        MPI_INT,
                        trabajador,
                        400,
                        MPI_COMM_WORLD
                    );
                }
            }

            else {

                int cantidad =
                    elementosPorTrabajador;

                int inicioGlobal =
                    (nodo - 1)
                    * elementosPorTrabajador;


                int* parte =
                    new int[cantidad];


                MPI_Recv(
                    parte,
                    cantidad,
                    MPI_INT,
                    0,
                    400,
                    MPI_COMM_WORLD,
                    MPI_STATUS_IGNORE
                );


                sumaParcial =
                    operaciones.sumatoriaMPI(
                        parte,
                        cantidad,
                        nombrePC,
                        nodo,
                        inicioGlobal
                    );


                std::cout
                    << "\nPC: "
                    << nombrePC
                    << " | MPI: "
                    << nodo
                    << " | Suma para promedio: "
                    << sumaParcial
                    << std::endl;


                delete[] parte;
            }


            MPI_Reduce(
                &sumaParcial,
                &sumaTotal,
                1,
                MPI_LONG_LONG,
                MPI_SUM,
                0,
                MPI_COMM_WORLD
            );


            MPI_Barrier(MPI_COMM_WORLD);

            double fin =
                MPI_Wtime();


            if (nodo == 0) {

                double promedioTotal =
                    static_cast<double>(
                        sumaTotal
                    ) / tamanio;


                std::cout
                    << "\n====================================\n";

                std::cout
                    << "PROMEDIO TOTAL: "
                    << std::fixed
                    << std::setprecision(6)
                    << promedioTotal
                    << std::endl;

                std::cout
                    << "Tiempo distribuido: "
                    << (fin - inicio)
                    << " segundos\n";

                std::cout
                    << "====================================\n";
            }
        }


        // ====================================================
        // 10. MAXIMO
        // ====================================================

        else if (opcion == 10) {

            MPI_Barrier(MPI_COMM_WORLD);

            double inicio =
                MPI_Wtime();


            int maximoParcial =
                INT_MIN;

            int maximoTotal =
                INT_MIN;


            if (nodo == 0) {

                for (
                    int trabajador = 1;
                    trabajador < totalNodos;
                    trabajador++
                ) {

                    int inicioGlobal =
                        (trabajador - 1)
                        * elementosPorTrabajador;


                    MPI_Send(
                        A + inicioGlobal,
                        elementosPorTrabajador,
                        MPI_INT,
                        trabajador,
                        500,
                        MPI_COMM_WORLD
                    );
                }
            }

            else {

                int cantidad =
                    elementosPorTrabajador;

                int inicioGlobal =
                    (nodo - 1)
                    * elementosPorTrabajador;


                int* parte =
                    new int[cantidad];


                MPI_Recv(
                    parte,
                    cantidad,
                    MPI_INT,
                    0,
                    500,
                    MPI_COMM_WORLD,
                    MPI_STATUS_IGNORE
                );


                maximoParcial =
                    operaciones.maximoMPI(
                        parte,
                        cantidad,
                        nombrePC,
                        nodo,
                        inicioGlobal
                    );


                std::cout
                    << "\nPC: "
                    << nombrePC
                    << " | MPI: "
                    << nodo
                    << " | MAXIMO PARCIAL: "
                    << maximoParcial
                    << std::endl;


                delete[] parte;
            }


            MPI_Reduce(
                &maximoParcial,
                &maximoTotal,
                1,
                MPI_INT,
                MPI_MAX,
                0,
                MPI_COMM_WORLD
            );


            MPI_Barrier(MPI_COMM_WORLD);

            double fin =
                MPI_Wtime();


            if (nodo == 0) {

                std::cout
                    << "\n====================================\n";

                std::cout
                    << "MAXIMO TOTAL: "
                    << maximoTotal
                    << std::endl;

                std::cout
                    << "Tiempo distribuido: "
                    << (fin - inicio)
                    << " segundos\n";

                std::cout
                    << "====================================\n";
            }
        }


        // ====================================================
        // 11. MINIMO
        // ====================================================

        else if (opcion == 11) {

            MPI_Barrier(MPI_COMM_WORLD);

            double inicio =
                MPI_Wtime();


            int minimoParcial =
                INT_MAX;

            int minimoTotal =
                INT_MAX;


            if (nodo == 0) {

                for (
                    int trabajador = 1;
                    trabajador < totalNodos;
                    trabajador++
                ) {

                    int inicioGlobal =
                        (trabajador - 1)
                        * elementosPorTrabajador;


                    MPI_Send(
                        A + inicioGlobal,
                        elementosPorTrabajador,
                        MPI_INT,
                        trabajador,
                        600,
                        MPI_COMM_WORLD
                    );
                }
            }

            else {

                int cantidad =
                    elementosPorTrabajador;

                int inicioGlobal =
                    (nodo - 1)
                    * elementosPorTrabajador;


                int* parte =
                    new int[cantidad];


                MPI_Recv(
                    parte,
                    cantidad,
                    MPI_INT,
                    0,
                    600,
                    MPI_COMM_WORLD,
                    MPI_STATUS_IGNORE
                );


                minimoParcial =
                    operaciones.minimoMPI(
                        parte,
                        cantidad,
                        nombrePC,
                        nodo,
                        inicioGlobal
                    );


                std::cout
                    << "\nPC: "
                    << nombrePC
                    << " | MPI: "
                    << nodo
                    << " | MINIMO PARCIAL: "
                    << minimoParcial
                    << std::endl;


                delete[] parte;
            }


            MPI_Reduce(
                &minimoParcial,
                &minimoTotal,
                1,
                MPI_INT,
                MPI_MIN,
                0,
                MPI_COMM_WORLD
            );


            MPI_Barrier(MPI_COMM_WORLD);

            double fin =
                MPI_Wtime();


            if (nodo == 0) {

                std::cout
                    << "\n====================================\n";

                std::cout
                    << "MINIMO TOTAL: "
                    << minimoTotal
                    << std::endl;

                std::cout
                    << "Tiempo distribuido: "
                    << (fin - inicio)
                    << " segundos\n";

                std::cout
                    << "====================================\n";
            }
        }


        // ====================================================
        // OPCION INVALIDA
        // ====================================================

        else if (opcion != 0) {

            if (nodo == 0) {

                std::cout
                    << "\nOpcion no valida.\n";
            }
        }


    } while (opcion != 0);


    // ========================================================
    // LIBERAR MEMORIA
    // ========================================================

    if (A != nullptr) {

        delete[] A;

        A = nullptr;
    }


    // ========================================================
    // FINALIZAR MPI
    // ========================================================

    MPI_Finalize();

    return 0;
}