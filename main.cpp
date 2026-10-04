#include <mpi.h>
#include <omp.h>

#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <string>
#include <climits>
#include <cstdlib>
#include <ctime>

#include "Busquedas.h"

// ============================================================
// VARIABLES PARA RESULTADOS
// ============================================================

struct Tiempos
{
    double ordenamientoSecuencial = 0.0;
    double ordenamientoParalelo = 0.0;

    double busquedaSecuencialLocal = 0.0;
    double busquedaSecuencialDistribuida = 0.0;

    double busquedaBinariaLocal = 0.0;
    double busquedaBinariaDistribuida = 0.0;
};

// ============================================================
// ENCABEZADO DEL LOG
// ============================================================

void encabezadoLog(
    std::ofstream& log,
    const std::string& equipo,
    int rank,
    int procesos
)
{
    std::ostringstream salida;

    salida
        << "\n============================================\n"
        << " BUSQUEDA PARALELA DISTRIBUIDA MPI + OPENMP\n"
        << " PRACTICA 1.5\n"
        << "============================================\n"
        << "Equipo / Host: " << equipo << "\n"
        << "Nodo MPI: " << rank << "\n"
        << "Procesos MPI: " << procesos << "\n"
        << "Hilos OpenMP disponibles: "
        << omp_get_max_threads() << "\n"
        << "Fecha y hora: " << obtenerFechaHora() << "\n"
        << "============================================";

    imprimirYLog(log, salida.str());
}

// ============================================================
// CALCULAR DISTRIBUCION
// ============================================================

void calcularDistribucion(
    long long n,
    int procesos,
    int* cantidades,
    int* desplazamientos
)
{
    long long base = n / procesos;
    long long sobrante = n % procesos;

    long long offset = 0;

    for (int i = 0; i < procesos; i++)
    {
        long long cantidad =
            base + (i < sobrante ? 1 : 0);

        cantidades[i] =
            static_cast<int>(cantidad);

        desplazamientos[i] =
            static_cast<int>(offset);

        offset += cantidad;
    }
}

// ============================================================
// BUSQUEDA DISTRIBUIDA LINEAL
// ============================================================

long long ejecutarBusquedaSecuencialDistribuida(
    int* arregloGlobal,
    long long n,
    int target,
    int rank,
    int procesos,
    bool detallado,
    std::ofstream& log,
    const std::string& equipo,
    double& tiempo
)
{
    int* cantidades = new int[procesos];
    int* desplazamientos = new int[procesos];

    calcularDistribucion(
        n,
        procesos,
        cantidades,
        desplazamientos
    );

    int cantidadLocal =
        cantidades[rank];

    int* bloqueLocal =
        new int[cantidadLocal];

    MPI_Barrier(MPI_COMM_WORLD);

    double inicio = MPI_Wtime();

    MPI_Scatterv(
        arregloGlobal,
        cantidades,
        desplazamientos,
        MPI_INT,
        bloqueLocal,
        cantidadLocal,
        MPI_INT,
        0,
        MPI_COMM_WORLD
    );

    long long resultadoLocal =
        busquedaLinealOpenMP(
            bloqueLocal,
            cantidadLocal,
            target,
            desplazamientos[rank],
            detallado,
            log,
            rank,
            equipo
        );

    long long valorReducir =
        (resultadoLocal == -1)
        ? LLONG_MAX
        : resultadoLocal;

    long long resultadoGlobal =
        LLONG_MAX;

    MPI_Reduce(
        &valorReducir,
        &resultadoGlobal,
        1,
        MPI_LONG_LONG,
        MPI_MIN,
        0,
        MPI_COMM_WORLD
    );

    MPI_Barrier(MPI_COMM_WORLD);

    double fin = MPI_Wtime();

    tiempo = fin - inicio;

    delete[] bloqueLocal;
    delete[] cantidades;
    delete[] desplazamientos;

    if (rank == 0)
    {
        if (resultadoGlobal == LLONG_MAX)
            return -1;

        return resultadoGlobal;
    }

    return -1;
}

// ============================================================
// BUSQUEDA DISTRIBUIDA BINARIA
// ============================================================

long long ejecutarBusquedaBinariaDistribuida(
    int* arregloGlobal,
    long long n,
    int target,
    int rank,
    int procesos,
    bool detallado,
    std::ofstream& log,
    const std::string& equipo,
    double& tiempo
)
{
    int* cantidades = new int[procesos];
    int* desplazamientos = new int[procesos];

    calcularDistribucion(
        n,
        procesos,
        cantidades,
        desplazamientos
    );

    int cantidadLocal =
        cantidades[rank];

    int* bloqueLocal =
        new int[cantidadLocal];

    MPI_Barrier(MPI_COMM_WORLD);

    double inicio = MPI_Wtime();

    MPI_Scatterv(
        arregloGlobal,
        cantidades,
        desplazamientos,
        MPI_INT,
        bloqueLocal,
        cantidadLocal,
        MPI_INT,
        0,
        MPI_COMM_WORLD
    );

    long long resultadoLocal = -1;

    if (cantidadLocal > 0)
    {
        // Primero comprobamos si el target puede estar
        // dentro de los valores almacenados en este bloque.
        if (target >= bloqueLocal[0] &&
            target <= bloqueLocal[cantidadLocal - 1])
        {
            resultadoLocal =
                busquedaBinariaOpenMP(
                    bloqueLocal,
                    cantidadLocal,
                    target,
                    desplazamientos[rank],
                    detallado,
                    log,
                    rank,
                    equipo
                );
        }
        else if (detallado)
        {
            std::ostringstream salida;

            salida
                << "[Equipo: " << equipo << "] "
                << "[Nodo MPI: " << rank << "] "
                << "[Bloque global: "
                << desplazamientos[rank]
                << " - "
                << desplazamientos[rank]
                + cantidadLocal - 1
                << "] "
                << "[Busqueda Binaria] "
                << "[Target fuera del rango de valores]";

            imprimirYLog(log, salida.str());
        }
    }

    long long valorReducir =
        resultadoLocal == -1
        ? LLONG_MAX
        : resultadoLocal;

    long long resultadoGlobal =
        LLONG_MAX;

    MPI_Reduce(
        &valorReducir,
        &resultadoGlobal,
        1,
        MPI_LONG_LONG,
        MPI_MIN,
        0,
        MPI_COMM_WORLD
    );

    MPI_Barrier(MPI_COMM_WORLD);

    double fin = MPI_Wtime();

    tiempo = fin - inicio;

    delete[] bloqueLocal;
    delete[] cantidades;
    delete[] desplazamientos;

    if (rank == 0)
    {
        if (resultadoGlobal == LLONG_MAX)
            return -1;

        return resultadoGlobal;
    }

    return -1;
}

// ============================================================
// MOSTRAR COMPARATIVA
// ============================================================

void mostrarComparativa(
    const Tiempos& t,
    int procesos
)
{
    std::cout
        << "\n============================================\n"
        << " COMPARATIVA DE RENDIMIENTO\n"
        << "============================================\n";

    std::cout << std::fixed << std::setprecision(8);

    std::cout
        << "\nORDENAMIENTO\n"
        << "Secuencial: "
        << t.ordenamientoSecuencial << " s\n"
        << "OpenMP:     "
        << t.ordenamientoParalelo << " s\n";

    if (t.ordenamientoParalelo > 0)
    {
        std::cout
            << "Speedup OpenMP: "
            << t.ordenamientoSecuencial /
            t.ordenamientoParalelo
            << "x\n";
    }

    std::cout
        << "\nBUSQUEDA SECUENCIAL\n"
        << "Local OpenMP:       "
        << t.busquedaSecuencialLocal << " s\n"
        << "Distribuida MPI+OMP:"
        << t.busquedaSecuencialDistribuida
        << " s\n";

    if (t.busquedaSecuencialDistribuida > 0)
    {
        double speedup =
            t.busquedaSecuencialLocal /
            t.busquedaSecuencialDistribuida;

        double eficiencia =
            (speedup / procesos) * 100.0;

        std::cout
            << "Speedup distribuido: "
            << speedup << "x\n"
            << "Eficiencia MPI: "
            << eficiencia << "%\n";
    }

    std::cout
        << "\nBUSQUEDA BINARIA\n"
        << "Local OpenMP:        "
        << t.busquedaBinariaLocal << " s\n"
        << "Distribuida MPI+OMP: "
        << t.busquedaBinariaDistribuida
        << " s\n";

    if (t.busquedaBinariaDistribuida > 0)
    {
        double speedup =
            t.busquedaBinariaLocal /
            t.busquedaBinariaDistribuida;

        double eficiencia =
            (speedup / procesos) * 100.0;

        std::cout
            << "Speedup distribuido: "
            << speedup << "x\n"
            << "Eficiencia MPI: "
            << eficiencia << "%\n";
    }

    std::cout
        << "============================================\n";
}

// ============================================================
// MAIN
// ============================================================

int main(int argc, char** argv)
{
    // ========================================================
    // INICIALIZAR MPI
    // ========================================================

    int soporteHilos = 0;

    MPI_Init_thread(
        &argc,
        &argv,
        MPI_THREAD_FUNNELED,
        &soporteHilos
    );

    int rank;
    int procesos;

    MPI_Comm_rank(
        MPI_COMM_WORLD,
        &rank
    );

    MPI_Comm_size(
        MPI_COMM_WORLD,
        &procesos
    );

    // ========================================================
    // INFORMACION DEL EQUIPO
    // ========================================================

    std::string equipo =
        obtenerNombreEquipo();

    std::ostringstream nombreLog;

    nombreLog
        << "log_equipo_"
        << equipo
        << "_nodo_"
        << rank
        << ".txt";

    std::ofstream log(
        nombreLog.str(),
        std::ios::app
    );

    encabezadoLog(
        log,
        equipo,
        rank,
        procesos
    );

    // ========================================================
    // VARIABLES PRINCIPALES
    // ========================================================

    int* arreglo = nullptr;

    long long n = 0;

    int maxValor = 1000;

    bool arregloGenerado = false;
    bool arregloOrdenado = false;

    Tiempos tiempos;

    int opcion = 0;

    // ========================================================
    // CICLO PRINCIPAL
    // ========================================================

    do
    {
        if (rank == 0)
        {
            std::cout
                << "\n==============================================\n"
                << " BUSQUEDA PARALELA DISTRIBUIDA MPI + OPENMP\n"
                << "==============================================\n"
                << "1. Generar arreglo dinamico\n"
                << "2. Ordenar arreglo con OpenMP\n"
                << "3. Busqueda secuencial (Local vs Distribuida)\n"
                << "4. Busqueda binaria (Local vs Distribuida)\n"
                << "5. Mostrar tiempos y comparativa\n"
                << "6. Salir\n"
                << "==============================================\n"
                << "Seleccione una opcion: ";

            std::cin >> opcion;
        }

        MPI_Bcast(
            &opcion,
            1,
            MPI_INT,
            0,
            MPI_COMM_WORLD
        );

        // ====================================================
        // OPCION 1
        // ====================================================

        if (opcion == 1)
        {
            if (rank == 0)
            {
                if (arreglo != nullptr)
                {
                    delete[] arreglo;
                    arreglo = nullptr;
                }

                std::cout
                    << "\nCantidad de elementos: ";

                std::cin >> n;

                std::cout
                    << "Valor maximo aleatorio: ";

                std::cin >> maxValor;

                if (n <= 0)
                {
                    std::cout
                        << "Tamano invalido.\n";

                    arregloGenerado = false;
                }
                else
                {
                    arreglo =
                        new int[n];

                    llenarArregloAleatorio(
                        arreglo,
                        n,
                        maxValor
                    );

                    arregloGenerado = true;
                    arregloOrdenado = false;

                    std::cout
                        << "\nArreglo generado correctamente.\n"
                        << "Elementos: "
                        << n << "\n"
                        << "Memoria aproximada: "
                        << (n * sizeof(int))
                        / (1024.0 * 1024.0)
                        << " MB\n";

                    escribirLog(
                        log,
                        "Arreglo dinamico generado con "
                        + std::to_string(n)
                        + " elementos."
                    );

                    // Para arreglo pequeño mostramos todo.
                    if (n <= 100)
                    {
                        mostrarArreglo(
                            arreglo,
                            n,
                            log
                        );
                    }
                }
            }

            int generado =
                arregloGenerado ? 1 : 0;

            MPI_Bcast(
                &generado,
                1,
                MPI_INT,
                0,
                MPI_COMM_WORLD
            );

            MPI_Bcast(
                &n,
                1,
                MPI_LONG_LONG,
                0,
                MPI_COMM_WORLD
            );

            arregloGenerado =
                generado == 1;
        }

        // ====================================================
        // OPCION 2
        // ====================================================

        else if (opcion == 2)
        {
            int generado =
                arregloGenerado ? 1 : 0;

            MPI_Bcast(
                &generado,
                1,
                MPI_INT,
                0,
                MPI_COMM_WORLD
            );

            if (!generado)
            {
                if (rank == 0)
                {
                    std::cout
                        << "\nPrimero debe generar el arreglo.\n";
                }

                continue;
            }

            if (rank == 0)
            {
                int* copiaSecuencial =
                    new int[n];

                int* copiaParalela =
                    new int[n];

                for (long long i = 0; i < n; i++)
                {
                    copiaSecuencial[i] = arreglo[i];
                    copiaParalela[i] = arreglo[i];
                }

                // --------------------------------------------
                // ORDENAMIENTO SECUENCIAL
                // --------------------------------------------

                int* auxiliar =
                    new int[n];

                double inicioSec =
                    MPI_Wtime();

                mergeSortSecuencial(
                    copiaSecuencial,
                    auxiliar,
                    0,
                    n - 1
                );

                double finSec =
                    MPI_Wtime();

                tiempos.ordenamientoSecuencial =
                    finSec - inicioSec;

                delete[] auxiliar;

                // --------------------------------------------
                // ORDENAMIENTO PARALELO
                // --------------------------------------------

                double inicioPar =
                    MPI_Wtime();

                mergeSortParalelo(
                    copiaParalela,
                    n
                );

                double finPar =
                    MPI_Wtime();

                tiempos.ordenamientoParalelo =
                    finPar - inicioPar;

                // --------------------------------------------
                // CONSERVAR ARREGLO PARALELO
                // --------------------------------------------

                delete[] arreglo;

                arreglo = copiaParalela;

                copiaParalela = nullptr;

                arregloOrdenado = true;

                std::cout
                    << std::fixed
                    << std::setprecision(8)
                    << "\nOrdenamiento terminado.\n"
                    << "Merge Sort Secuencial: "
                    << tiempos.ordenamientoSecuencial
                    << " s\n"
                    << "Merge Sort OpenMP:     "
                    << tiempos.ordenamientoParalelo
                    << " s\n";

                if (tiempos.ordenamientoParalelo > 0)
                {
                    std::cout
                        << "Speedup: "
                        << tiempos.ordenamientoSecuencial /
                        tiempos.ordenamientoParalelo
                        << "x\n";
                }

                if (n <= 100)
                {
                    std::cout
                        << "\nArreglo ordenado:\n";

                    mostrarArreglo(
                        arreglo,
                        n,
                        log
                    );
                }

                delete[] copiaSecuencial;
            }

            int ordenado =
                arregloOrdenado ? 1 : 0;

            MPI_Bcast(
                &ordenado,
                1,
                MPI_INT,
                0,
                MPI_COMM_WORLD
            );

            arregloOrdenado =
                ordenado == 1;
        }

        // ====================================================
        // OPCION 3
        // BUSQUEDA SECUENCIAL
        // ====================================================

        else if (opcion == 3)
        {
            int generado =
                arregloGenerado ? 1 : 0;

            MPI_Bcast(
                &generado,
                1,
                MPI_INT,
                0,
                MPI_COMM_WORLD
            );

            if (!generado)
            {
                if (rank == 0)
                {
                    std::cout
                        << "\nPrimero debe generar el arreglo.\n";
                }

                continue;
            }

            int target = 0;

            if (rank == 0)
            {
                std::cout
                    << "\nValor a buscar: ";

                std::cin >> target;
            }

            MPI_Bcast(
                &target,
                1,
                MPI_INT,
                0,
                MPI_COMM_WORLD
            );

            bool detallado =
                n <= 100;

            // --------------------------------------------
            // LOCAL OPENMP
            // --------------------------------------------

            long long resultadoLocal = -1;

            if (rank == 0)
            {
                double inicio =
                    MPI_Wtime();

                resultadoLocal =
                    busquedaLinealOpenMP(
                        arreglo,
                        n,
                        target,
                        0,
                        detallado,
                        log,
                        rank,
                        equipo
                    );

                double fin =
                    MPI_Wtime();

                tiempos.busquedaSecuencialLocal =
                    fin - inicio;

                std::cout
                    << "\n----- BUSQUEDA SECUENCIAL LOCAL -----\n";

                if (resultadoLocal != -1)
                {
                    std::cout
                        << "Elemento encontrado en indice: "
                        << resultadoLocal << "\n";
                }
                else
                {
                    std::cout
                        << "Elemento NO encontrado.\n";
                }

                std::cout
                    << "Tiempo: "
                    << tiempos.busquedaSecuencialLocal
                    << " s\n";
            }

            MPI_Barrier(MPI_COMM_WORLD);

            // --------------------------------------------
            // DISTRIBUIDA
            // --------------------------------------------

            double tiempoDistribuido = 0;

            long long resultadoDistribuido =
                ejecutarBusquedaSecuencialDistribuida(
                    arreglo,
                    n,
                    target,
                    rank,
                    procesos,
                    detallado,
                    log,
                    equipo,
                    tiempoDistribuido
                );

            if (rank == 0)
            {
                tiempos.busquedaSecuencialDistribuida =
                    tiempoDistribuido;

                std::cout
                    << "\n----- BUSQUEDA SECUENCIAL DISTRIBUIDA -----\n";

                if (resultadoDistribuido != -1)
                {
                    std::cout
                        << "Elemento encontrado en indice global: "
                        << resultadoDistribuido << "\n";
                }
                else
                {
                    std::cout
                        << "Elemento NO encontrado.\n";
                }

                std::cout
                    << "Tiempo MPI + OpenMP: "
                    << tiempoDistribuido
                    << " s\n";

                if (tiempoDistribuido > 0)
                {
                    std::cout
                        << "Speedup Local/Distribuido: "
                        << tiempos.busquedaSecuencialLocal /
                        tiempoDistribuido
                        << "x\n";
                }
            }
        }

        // ====================================================
        // OPCION 4
        // BUSQUEDA BINARIA
        // ====================================================

        else if (opcion == 4)
        {
            int generado =
                arregloGenerado ? 1 : 0;

            int ordenado =
                arregloOrdenado ? 1 : 0;

            MPI_Bcast(
                &generado,
                1,
                MPI_INT,
                0,
                MPI_COMM_WORLD
            );

            MPI_Bcast(
                &ordenado,
                1,
                MPI_INT,
                0,
                MPI_COMM_WORLD
            );

            if (!generado || !ordenado)
            {
                if (rank == 0)
                {
                    std::cout
                        << "\nDebe generar y ordenar "
                        << "el arreglo antes de usar "
                        << "busqueda binaria.\n";
                }

                continue;
            }

            int target = 0;

            if (rank == 0)
            {
                std::cout
                    << "\nValor a buscar: ";

                std::cin >> target;
            }

            MPI_Bcast(
                &target,
                1,
                MPI_INT,
                0,
                MPI_COMM_WORLD
            );

            bool detallado =
                n <= 100;

            // --------------------------------------------
            // BINARIA LOCAL
            // --------------------------------------------

            long long resultadoLocal = -1;

            if (rank == 0)
            {
                double inicio =
                    MPI_Wtime();

                resultadoLocal =
                    busquedaBinariaOpenMP(
                        arreglo,
                        n,
                        target,
                        0,
                        detallado,
                        log,
                        rank,
                        equipo
                    );

                double fin =
                    MPI_Wtime();

                tiempos.busquedaBinariaLocal =
                    fin - inicio;

                std::cout
                    << "\n----- BUSQUEDA BINARIA LOCAL -----\n";

                if (resultadoLocal != -1)
                {
                    std::cout
                        << "Elemento encontrado en indice: "
                        << resultadoLocal << "\n";
                }
                else
                {
                    std::cout
                        << "Elemento NO encontrado.\n";
                }

                std::cout
                    << "Tiempo: "
                    << tiempos.busquedaBinariaLocal
                    << " s\n";
            }

            MPI_Barrier(MPI_COMM_WORLD);

            // --------------------------------------------
            // BINARIA DISTRIBUIDA
            // --------------------------------------------

            double tiempoDistribuido = 0;

            long long resultadoDistribuido =
                ejecutarBusquedaBinariaDistribuida(
                    arreglo,
                    n,
                    target,
                    rank,
                    procesos,
                    detallado,
                    log,
                    equipo,
                    tiempoDistribuido
                );

            if (rank == 0)
            {
                tiempos.busquedaBinariaDistribuida =
                    tiempoDistribuido;

                std::cout
                    << "\n----- BUSQUEDA BINARIA DISTRIBUIDA -----\n";

                if (resultadoDistribuido != -1)
                {
                    std::cout
                        << "Elemento encontrado en indice global: "
                        << resultadoDistribuido << "\n";
                }
                else
                {
                    std::cout
                        << "Elemento NO encontrado.\n";
                }

                std::cout
                    << "Tiempo MPI + OpenMP: "
                    << tiempoDistribuido
                    << " s\n";

                if (tiempoDistribuido > 0)
                {
                    std::cout
                        << "Speedup Local/Distribuido: "
                        << tiempos.busquedaBinariaLocal /
                        tiempoDistribuido
                        << "x\n";
                }
            }
        }

        // ====================================================
        // OPCION 5
        // ====================================================

        else if (opcion == 5)
        {
            if (rank == 0)
            {
                mostrarComparativa(
                    tiempos,
                    procesos
                );
            }
        }

        // ====================================================
        // OPCION 6
        // ====================================================

        else if (opcion == 6)
        {
            if (rank == 0)
            {
                std::cout
                    << "\nFinalizando programa...\n";
            }

            escribirLog(
                log,
                "Proceso MPI finalizado."
            );
        }

        else
        {
            if (rank == 0)
            {
                std::cout
                    << "\nOpcion invalida.\n";
            }
        }

        MPI_Barrier(MPI_COMM_WORLD);

    } while (opcion != 6);

    // ========================================================
    // LIBERAR MEMORIA
    // ========================================================

    if (rank == 0 &&
        arreglo != nullptr)
    {
        delete[] arreglo;
        arreglo = nullptr;
    }

    if (log.is_open())
    {
        log.close();
    }

    MPI_Finalize();

    return 0;
}