#include "cuda_runtime.h"
#include "cuda_runtime_api.h"
#include "device_launch_parameters.h"
#include "math.h"
#include <time.h>
#include <stdio.h>
#include <stdlib.h>

void mostrarArray(int* array, int tamanho) {

    int* guardado = array;
    for (int i = 0; i < tamanho; i++) {
        printf("%d, ", *guardado++);
    }
    printf("\n");
}

int* generarArrayAleatorio(int n, int min, int max) {
    int* array = (int*)malloc(n * sizeof(int));
    if (array == NULL) {
        printf("Error al asignar memoria para el array.\n");
        return NULL;
    }

    for (int i = 0; i < n; i++) {
        array[i] = min + (int)(((double)rand() / RAND_MAX) * (max - min + 1));
    }

    return array;
}

__global__ void tester1OrdenadoParalelokernel(int* sorted_device, int sortedSize, int* first_last_device, int first_last_Size, int nHilosUsados, bool* toret) {
    int idHilo = threadIdx.x + blockIdx.x * blockDim.x;
    int nHilosTotal = blockDim.x * gridDim.x;
    if (idHilo < nHilosUsados)
    {

        int inicio, fin;
        int nposRecorrer = (sortedSize / nHilosUsados);

        inicio = idHilo * nposRecorrer;
        fin = (idHilo * nposRecorrer) + nposRecorrer;

        if (idHilo == nHilosUsados - 1) { fin += (sortedSize % nHilosUsados); }

        for (int i = inicio; i < fin - 1; i++)
        {
            if (sorted_device[i] > sorted_device[i + 1]) { *toret = false; }
        }
        first_last_device[(idHilo * 2)] = sorted_device[inicio];
        first_last_device[(idHilo * 2) + 1] = sorted_device[fin - 1];

    }

}

__global__ void tester2OrdenadoParalelokernel( int* first_last_device, int first_last_Size, bool* toret) {
    int idHilo = threadIdx.x + blockIdx.x * blockDim.x;
    int nHilosTotal = blockDim.x * gridDim.x;
    
    if (idHilo == 0)
    {
        for (int i = 0; i < first_last_Size - 1; i++)
        {
            if (first_last_device[i] > first_last_device[i + 1]) { *toret = false; }
        }
    }

}

int testerOrdenadoParalelo(int* sorted_device, int sortedSize, int numBloques, int numHilos) {
    int nHilosUsados;
    if (numBloques * numHilos > sortedSize) {
        nHilosUsados = sortedSize;
    }
    else {

        nHilosUsados = numBloques * numHilos;
    }

    int* first_last_device;
    int first_last_Size = nHilosUsados * 2;
    cudaError_t cudaStatusMalloc = cudaMalloc(&first_last_device, first_last_Size * sizeof(int));
    if (cudaStatusMalloc != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed! (Testeando)");
        cudaFree(first_last_device);
        return 2;
    }
    cudaError_t cudaStatusMemcpy;

    bool verdad = true;
    bool* ordenado;
    cudaStatusMalloc = cudaMalloc(&ordenado, sizeof(bool));
    if (cudaStatusMalloc != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed! (Testeando)");
        cudaFree(ordenado);
        return 2;
    }
    cudaStatusMemcpy = cudaMemcpy(ordenado, &verdad, sizeof(bool), cudaMemcpyHostToDevice);
    if (cudaStatusMemcpy != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed! (Array: verdad)");
        cudaFree(&verdad);
        return 2;
    }

    tester1OrdenadoParalelokernel << <numBloques, numHilos >> > (sorted_device, sortedSize, first_last_device, first_last_Size, nHilosUsados, ordenado);
    cudaDeviceSynchronize();
    tester2OrdenadoParalelokernel << <numBloques, numHilos >> > (first_last_device, first_last_Size, ordenado);
    cudaDeviceSynchronize();

    bool* ordenadoResult = (bool*)malloc(sizeof(bool));
    cudaStatusMemcpy = cudaMemcpy(ordenadoResult, ordenado, sizeof(bool), cudaMemcpyDeviceToHost);
    if (cudaStatusMemcpy != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed! (Array: ordenado)");
        cudaFree(ordenado);
        return 2;
    }


    cudaFree(ordenado);
    cudaFree(first_last_device);

    if (*ordenadoResult) {
        return 0;
    }
    else {
        return 1;
    }

}

void presentacion() {
    int ndisp = 0;
    cudaError_t errorcudaGetDevice = cudaGetDevice(&ndisp);
    if (errorcudaGetDevice != cudaSuccess) {
        fprintf(stderr, "cudaGetDevice failed!\n");
        printf("Error en la presentacion\n");
    }
    else {
        cudaDeviceProp propiedades;
        cudaDeviceProp* propiedadesPuntero = &propiedades;
        cudaGetDeviceProperties(propiedadesPuntero, ndisp);
        printf("Usando la grafica: %s\n", propiedades.name);
        //printf("Numero de multiprocesadores: %d\n", propiedades.multiProcessorCount);
        printf("Maximo de bloques por multiprocesador : %d\n", propiedades.maxBlocksPerMultiProcessor);
        printf("Maximo de hilos por bloque: %d\n", propiedades.maxThreadsPerBlock);
        printf("Maximo de hilos posibles: %d\n", propiedades.maxThreadsPerMultiProcessor);
        printf("Warp: %d\n", propiedades.warpSize);
        size_t free_mem, total_mem;
        cudaMemGetInfo(&free_mem, &total_mem);
        printf("Memoria disponible en la GPU: %zu bytes\n\n", free_mem);
    }
}

//METODOS USADOS POR EL METODO PRINCIPAL COUNTING SORT
__global__ void findMinMax(int* array, int size, int* min, int* max, int* findmin, int* findmax) {

    int idHilo = threadIdx.x + blockIdx.x * blockDim.x;
    int nHilosTotal = blockDim.x * gridDim.x;

    int nHilosUsados;

    if (nHilosTotal > size)
    {
        nHilosUsados = size;
    }
    else {
        nHilosUsados = nHilosTotal;
    }

    if (idHilo < nHilosUsados)
    {
        int inicio, fin;
        int nposRecorrer = (size / nHilosUsados);

        inicio = idHilo * nposRecorrer;
        fin = (idHilo * nposRecorrer) + nposRecorrer;
        if (idHilo == nHilosUsados - 1) { fin += (size % nHilosUsados); }
        int localmin, localmax;


        localmin = array[inicio];
        localmax = array[inicio];
        for (int i = inicio; i < fin; i++)
        {
            if (localmin > array[i]) { localmin = array[i]; }
            if (localmax < array[i]) { localmax = array[i]; }
        }

        findmin[idHilo] = localmin;
        findmax[idHilo] = localmax;
    }
}

__global__ void calcularMinMax(int size, int* min, int* max, int* findmin, int* findmax) {
    int idHilo = threadIdx.x + blockIdx.x * blockDim.x;
    int nHilosTotal = blockDim.x * gridDim.x;

    int nHilosUsados;

    if (nHilosTotal > size)
    {
        nHilosUsados = size;
    }
    else {
        nHilosUsados = nHilosTotal;
    }

    if (idHilo == 0) {
        *min = findmin[0];
        *max = findmax[0];
        for (int i = 1; i < nHilosUsados; i++)
        {
            if (*min > findmin[i]) { *min = findmin[i]; }
            if (*max < findmax[i]) { *max = findmax[i]; }
        }
    }

}

int findMinMaxDeInput(int* input_device, int inputSize, int* minimo, int* maximo, int numBloques, int numHilos) {

    int* min_device;
    cudaMalloc(&min_device, sizeof(int));
    int* max_device;
    cudaMalloc(&max_device, sizeof(int));
    int nHilosUsados;
    if (numBloques * numHilos > inputSize)
    {
        nHilosUsados = inputSize;
    }
    else {
        nHilosUsados = numBloques * numHilos;
    }

    int* findmin;
    cudaError_t cudaStatusMalloc = cudaMalloc(&findmin, nHilosUsados * sizeof(int));
    if (cudaStatusMalloc != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed! (Array: findmin)");
        cudaFree(findmin);
        return 1;
    }
    int* findmax;
    cudaStatusMalloc = cudaMalloc(&findmax, nHilosUsados * sizeof(int));
    if (cudaStatusMalloc != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed! (Array: findmax)");
        cudaFree(findmax);
        return 1;
    }

    findMinMax << <numBloques, numHilos >> > (input_device, inputSize, min_device, max_device, findmin, findmax);
    cudaDeviceSynchronize();
    calcularMinMax << <numBloques, numHilos >> > ( inputSize, min_device, max_device, findmin, findmax);
    cudaDeviceSynchronize();

    int* min = (int*)malloc(sizeof(int));
    cudaError_t cudaStatusMemcpy = cudaMemcpy(min, min_device, sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatusMemcpy != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed! (Array: min_device)");
        cudaFree(min_device);
        return 1;
    }
    int* max = (int*)malloc(sizeof(int));
    cudaStatusMemcpy = cudaMemcpy(max, max_device, sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatusMemcpy != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed! (Array: max_device)");
        cudaFree(max_device);
        return 1;
    }

    *minimo = *min;
    *maximo = *max;

    return 0;
}

__global__ void inicializamosOcurrencias(int* ocurrencias, int ocurrenciasSize) {
    int idHilo = threadIdx.x + blockIdx.x * blockDim.x;
    int nHilosTotal = blockDim.x * gridDim.x;

    //INICIALIZAMOS OCURRENCIAS
    int nhilosInicializaOcurrencias;
    if (nHilosTotal > ocurrenciasSize)
    {
        nhilosInicializaOcurrencias = ocurrenciasSize;
    }
    else {
        nhilosInicializaOcurrencias = nHilosTotal;
    }

    if (idHilo < nhilosInicializaOcurrencias) {

        int nElementosAcalcular = (ocurrenciasSize / nhilosInicializaOcurrencias);
        if ((ocurrenciasSize % nhilosInicializaOcurrencias) > idHilo)
        {
            nElementosAcalcular += 1;
        }

        for (int i = 0; i < nElementosAcalcular; i++)
        {
            int posOcurr = idHilo + (nhilosInicializaOcurrencias * i);
            ocurrencias[posOcurr] = 0;

        }
    }
}

__global__ void contarOcurrencias(int* input, int inputSize, int* ocurrencias, int ocurrenciasSize, int min, int max) {
    int idHilo = threadIdx.x + blockIdx.x * blockDim.x;
    int nHilosTotal = blockDim.x * gridDim.x;

    //CONTAMOS OCURRENCIAS VERTICAL
    int nHilosUsados;
    if (nHilosTotal > inputSize)
    {
        nHilosUsados = inputSize;
    }
    else {
        nHilosUsados = nHilosTotal;
    }
    if (idHilo < nHilosUsados) {

        int nPosASumar = (inputSize / nHilosUsados);
        if (idHilo < (inputSize % nHilosUsados)) { nPosASumar++; }
        for (int i = 0; i < nPosASumar; i++)
        {
            int valorInput = input[idHilo + (i * nHilosUsados)];
            int posEnMinMax = valorInput - min;
            ocurrencias[idHilo + (nHilosUsados * posEnMinMax)] += 1;
        }
    }
}

__global__ void sumarOcurrencias(int inputSize, int* ocurrencias, int ocurrenciasSize, int* aux, int* auxsumado, int min, int max) {
    int idHilo = threadIdx.x + blockIdx.x * blockDim.x;
    int nHilosTotal = blockDim.x * gridDim.x;

    int nHilosUsados;
    if (nHilosTotal > inputSize)
    {
        nHilosUsados = inputSize;
    }
    else {
        nHilosUsados = nHilosTotal;
    }
    //SUMAMOS COLUMNAS OCURRENCIA
    int nHilosUsadosSumaOcurrencias;
    int minmaxSize = max + 1 - min;
    if (nHilosTotal > minmaxSize)
    {
        nHilosUsadosSumaOcurrencias = minmaxSize;
    }
    else {
        nHilosUsadosSumaOcurrencias = nHilosTotal;
    }

    if (idHilo < nHilosUsadosSumaOcurrencias) {

        int nfilasAcalcular = (minmaxSize / nHilosUsadosSumaOcurrencias);
        if ((minmaxSize % nHilosUsadosSumaOcurrencias) > idHilo)
        {
            nfilasAcalcular += 1;
        }

        for (int i = 0; i < nfilasAcalcular; i++)
        {
            for (int j = 0; j < nHilosUsados; j++)
            {
                aux[idHilo + (nHilosUsadosSumaOcurrencias * i)] += ocurrencias[(((idHilo + (nHilosUsadosSumaOcurrencias * i)) * nHilosUsados) + j)];
            }
        }

    }
}

__global__ void calcularAuxEscalera(int* aux, int* auxsumado, int min, int max) {
    int idHilo = threadIdx.x + blockIdx.x * blockDim.x;

    //CALCULAMOS AUXESCALERA
    if (idHilo == 0) {

        auxsumado[0] = aux[0];
        for (int i = 1; i < max + 1 - min; i++)
        {
            auxsumado[i] = aux[i] + auxsumado[i - 1];
        }

    }
}

__global__ void crearSorted1(int* input, int inputSize, int* aux, int* auxsumado, int* sorted, int min, int max) {

    int idHilo = threadIdx.x + blockIdx.x * blockDim.x;
    int nHilosTotal = blockDim.x * gridDim.x;
    int minmaxSize = max + 1 - min;
    //CREAMOS EL ARRAY SORTED
    int nHilosUsadosSorting;
    if (nHilosTotal > minmaxSize)
    {
        nHilosUsadosSorting = minmaxSize;
    }
    else {
        nHilosUsadosSorting = nHilosTotal;
    }

    if (idHilo < nHilosUsadosSorting) {

        int nfilasAcalcular = (minmaxSize / nHilosUsadosSorting);
        if ((minmaxSize % nHilosUsadosSorting) > idHilo)
        {
            nfilasAcalcular += 1;
        }

        for (int i = 0; i < nfilasAcalcular; i++)
        {

            int posAux = idHilo + (nHilosUsadosSorting * i);
            int valor = min + posAux;

            for (int j = 0; j < aux[posAux]; j++)
            {
                sorted[(auxsumado[posAux] - aux[posAux]) + j] = valor;
            }
        }
    }
}

__global__ void crearSorted2(int* input, int inputSize, int* aux, int* auxsumado, int* sorted, int* indicesNuevos, int min, int max) {

    int idHilo = threadIdx.x + blockIdx.x * blockDim.x;
    int nHilosTotal = blockDim.x * gridDim.x;
    int minmaxSize = max + 1 - min;
    //CREAMOS EL ARRAY SORTED
    int nHilosUsadosSorting;
    if (nHilosTotal > inputSize)
    {
        nHilosUsadosSorting = inputSize;
    }
    else {
        nHilosUsadosSorting = nHilosTotal;
    }

    if (idHilo < nHilosUsadosSorting) {

        int nfilasAcalcular = (inputSize / nHilosUsadosSorting);
        if ((inputSize % nHilosUsadosSorting) > idHilo)
        {
            nfilasAcalcular += 1;
        }

        for (int i = 0; i < nfilasAcalcular; i++)
        {

            int num_A_ordenar = input[idHilo + (nHilosUsadosSorting * i)];
            int posEnMinMax = num_A_ordenar - min;
            int posEnAux = posEnMinMax;
            int posEnSorted = atomicSub(&auxsumado[posEnAux], 1);
            posEnSorted--;
            sorted[posEnSorted] = num_A_ordenar;
            indicesNuevos[idHilo + (nHilosUsadosSorting * i)] = posEnSorted;
        }
    }


}

//METODO PRINCIPAL
int CountingSortParalelo(int* input, int inputSize, int numBloques, int numHilos, char metodoCrearSort) {

    cudaError_t cudaStatusMalloc;
    cudaError_t cudaStatusMemcpy;

    //INICIAMOS CRONOMETRO TIEMPO
    cudaEvent_t inicio, fin;
    cudaEventCreate(&inicio);
    cudaEventCreate(&fin);
    cudaEventRecord(inicio, 0);

    //PASAMOS INPUT A LA GRAFICA 
    int* input_device;
    cudaStatusMalloc = cudaMalloc(&input_device, inputSize * sizeof(int));
    if (cudaStatusMalloc != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed! (Array: INPUT)");
        cudaFree(input_device);
        goto Error;
    }
    cudaStatusMemcpy = cudaMemcpy(input_device, input, inputSize * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatusMemcpy != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed! (Array: INPUT)");
        cudaFree(input_device);
        goto Error;
    }

    //RECOJEMOS EL VALOR MIN Y EL VALOR MAX QUE APARECE EN EL ARRAY
    int minimo;
    int maximo;
    int toret = findMinMaxDeInput(input_device, inputSize, &minimo, &maximo, numBloques, numHilos);
    if (toret == 1) {
        goto Error;
    }
    //CREAMOS EL ARRAY DE OCURRENCIAS  
    int ocurrenciasSize;
    if (numBloques * numHilos > inputSize)
    {
        ocurrenciasSize = inputSize;
    }
    else {
        ocurrenciasSize = numBloques * numHilos;
    }

    ocurrenciasSize *= ((maximo + 1) - minimo);

    ////Pasamos el array OCURRENCIAS a la GPU
    int* ocurrencias_device;
    cudaStatusMalloc = cudaMalloc(&ocurrencias_device, ocurrenciasSize * sizeof(int));
    if (cudaStatusMalloc != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!(Array: Ocurrencias)");
        cudaFree(ocurrencias_device);
        goto Error;
    }

    //CREAMOS EL ARRAY AUXILIAR Y LO PASAMOS LA GRAFICA
    int* aux_device;
    int aux_size = (maximo + 1) - minimo;
    cudaStatusMalloc = cudaMalloc(&aux_device, aux_size * sizeof(int));
    if (cudaStatusMalloc != cudaSuccess) {
        fprintf(stderr, "cudaMalloc  failed! (Array: Aux)");
        cudaFree(aux_device);
        goto Error;
    }

    //CREAMOS EL ARRAY ESCALERA Y LO PASAMOS LA GRAFICA
    int* auxEscalera_device;
    cudaStatusMalloc = cudaMalloc(&auxEscalera_device, aux_size * sizeof(int));
    if (cudaStatusMalloc != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed! (Array: AuxEscalera)");
        cudaFree(auxEscalera_device);
        goto Error;
    }

    //CREAMOS EL ARRAY SORTED Y LO PASAMOS LA GRAFICA
    int* sorted_device;
    int sorted_size = inputSize;
    cudaStatusMalloc = cudaMalloc(&sorted_device, sorted_size * sizeof(int));
    if (cudaStatusMalloc != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed! (Array: Sorted)");
        cudaFree(sorted_device);
        goto Error;
    }

    //CREAMOS EL ARRAY INDICES_NUEVOS Y LO PASAMOS LA GRAFICA
    int* indicesNuevos_device;
    if (metodoCrearSort == '2') {
        int indicesNuevos_size = inputSize;
        cudaStatusMalloc = cudaMalloc(&indicesNuevos_device, indicesNuevos_size * sizeof(int));
        if (cudaStatusMalloc != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed! (Array: indicesNuevos)");
            cudaFree(indicesNuevos_device);
            goto Error;
        }
    }

    //ORDENAMOS DE FORMA PARALELA
    inicializamosOcurrencias << <numBloques, numHilos >> > (ocurrencias_device, ocurrenciasSize);
    cudaDeviceSynchronize();
    contarOcurrencias << <numBloques, numHilos >> > (input_device, inputSize, ocurrencias_device, ocurrenciasSize, minimo, maximo);
    cudaDeviceSynchronize();
    sumarOcurrencias << <numBloques, numHilos >> > (inputSize, ocurrencias_device, ocurrenciasSize, aux_device, auxEscalera_device, minimo, maximo);
    cudaDeviceSynchronize();
    calcularAuxEscalera << <numBloques, numHilos >> > (aux_device, auxEscalera_device, minimo, maximo);
    cudaDeviceSynchronize();

    if (metodoCrearSort == '1') {
        crearSorted1 << <numBloques, numHilos >> > (input_device, inputSize, aux_device, auxEscalera_device, sorted_device, minimo, maximo);
        cudaDeviceSynchronize();
    }
    else {
        crearSorted2 << <numBloques, numHilos >> > (input_device, inputSize, aux_device, auxEscalera_device, sorted_device, indicesNuevos_device, minimo, maximo);
        cudaDeviceSynchronize();
    }

    //PARAMOS CRONOMETRO
    cudaEventRecord(fin, 0);
    cudaEventSynchronize(fin);
    float tiempo;
    cudaEventElapsedTime(&tiempo, inicio, fin);

    //RECOJEMOS AUX POR SI LO NECESITA
    int* aux = (int*)malloc(aux_size * sizeof(int));
    cudaStatusMemcpy = cudaMemcpy(aux, aux_device, aux_size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatusMemcpy != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed! (Array: Aux)");
        cudaFree(aux_device);
        goto Error;
    }
    
    //RECOGEMOS EL RESULTADO de SORTED_DEVICE TRAS LA EJECUCION DEL KERNEL
    int* sorted = (int*)malloc(sorted_size * sizeof(int));
    cudaStatusMemcpy = cudaMemcpy(sorted, sorted_device, sorted_size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatusMemcpy != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed! (Array: Sorted)");
        cudaFree(sorted_device);
        goto Error;
    }

    int* indicesNuevos;
    if (metodoCrearSort != '1') {
        indicesNuevos = (int*)malloc(sorted_size * sizeof(int));
        cudaStatusMemcpy = cudaMemcpy(indicesNuevos, indicesNuevos_device, sorted_size * sizeof(int), cudaMemcpyDeviceToHost);
        if (cudaStatusMemcpy != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed! (Array: indicesNuevos)");
            cudaFree(indicesNuevos_device);
            goto Error;
        }
    }


    //TESTS 
    printf("Testeando... ");
    int testResult = testerOrdenadoParalelo(sorted_device, inputSize, numBloques, numHilos);
    if (testResult == 0) {
        printf("Ordenado Paralelo Correcto\n");
    }
    else if(testResult == 1){
        printf("Ordenado Paralelo NO Correcto\n");
    }
    else {
        goto Error;
    }

    //MOSTRAR DATOS
    printf("Bloques,Hilos: %d,%d\t", numBloques, numHilos);
    printf("Tamano array: %d\t", inputSize);
    printf("Valores: %d, [%d,%d]\t", maximo + 1 - minimo, minimo, maximo);
    printf("Tiempo tardado: %f milisegundos\n\n", tiempo);

    char option;
    printf("¿Quieres ver el array input? (s/n): ");
    scanf(" %c", &option);

    if (option == 's' || option == 'S') {
        mostrarArray(input, inputSize);
        printf("\n");
    }

    if (metodoCrearSort != '1') {
        printf("¿Quieres ver el array indicesNuevos? (s/n): ");
        scanf(" %c", &option);

        if (option == 's' || option == 'S') {
            mostrarArray(indicesNuevos, inputSize);
            printf("\n");
        }
    }

    printf("¿Quieres ver el array sorted? (s/n): ");
    scanf(" %c", &option);

    if (option == 's' || option == 'S') {
        mostrarArray(sorted, inputSize);
        printf("\n");
    }

    printf("¿Quieres ver el array aux? (s/n): ");
    scanf(" %c", &option);

    if (option == 's' || option == 'S') {
        mostrarArray(aux, aux_size);
        printf("\n");
    }
    

    //LIBERAMOS MEMORIA
    free(aux);
    free(sorted);
    if (metodoCrearSort != '1') { free(indicesNuevos); }
    cudaFree(input_device);
    cudaFree(aux_device);
    cudaFree(auxEscalera_device);
    cudaFree(ocurrencias_device);
    cudaFree(sorted_device);
    if (metodoCrearSort != '1') { cudaFree(indicesNuevos_device);}

    cudaError_t cudaStatusFinal = cudaDeviceReset();
    if (cudaStatusFinal != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
    //FIN

    //MANEJO DE ERRORES
Error:
    
    fprintf(stderr, "\nSe ha producido algun error\nSeguramente al reservar memoria debido al gran tamano de algun array\nReseteando device...\n\n");
    cudaError_t cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return -1;
}

void prueba1() {
    int size = 10;
    int* input = generarArrayAleatorio(size, 1, size);

    CountingSortParalelo(input, size, 1, 3, '1');
    CountingSortParalelo(input, size, 1, 3, '2');
}

void prueba2() {
    //100
    int size100 = 100;
    int* input100 = generarArrayAleatorio(size100, 1, size100);
    CountingSortParalelo(input100, size100, 32, 32, '1');
    //500
    int size500 = 500;
    int* input500 = generarArrayAleatorio(size500, 1, size500);
    CountingSortParalelo(input500, size500, 32, 32, '2');
    //1000
    int size1000 = 1000;
    int* input1000 = generarArrayAleatorio(size1000, 1, size1000);
    CountingSortParalelo(input1000, size1000, 32, 32, '1');
    //10000
    int size10000 = 10000;
    int* input10000 = generarArrayAleatorio(size10000, 1, size10000);
    CountingSortParalelo(input10000, size10000, 32, 32, '2');
    //50000
    int size50000 = 50000;
    int* input50000 = generarArrayAleatorio(size50000, 1, size50000);
    CountingSortParalelo(input50000, size50000, 32, 32, '1');
    //200000
    int size200000 = 200000;
    int* input200000 = generarArrayAleatorio(size200000, 1, size200000);
    CountingSortParalelo(input200000, size200000, 32, 32, '2');
    //500000
     int size500000 = 500000;
    int* input500000 = generarArrayAleatorio(size500000, 1, size500000);
    CountingSortParalelo(input500000, size500000, 32, 32, '1');
}

void prueba3() {
    
    //Size demasiado grande
    int sizeDemasiadoGrande = 1000000000;
    int* inputDemasiadoGrande = generarArrayAleatorio(sizeDemasiadoGrande, 1, 10000);
    CountingSortParalelo(inputDemasiadoGrande, sizeDemasiadoGrande, 32, 32, '1');

    //Size viable
    int sizeViable = 100;
    int* inputViable = generarArrayAleatorio(sizeViable, 1, 10000);
    CountingSortParalelo(inputViable, sizeViable, 32, 32, '1');

}

void prueba4() {
    
    int size = 50000000;
    int* input = generarArrayAleatorio(size, 1, 65535);
    CountingSortParalelo(input, size, 1, 1, '1');
    CountingSortParalelo(input, size, 1, 2, '1');
    CountingSortParalelo(input, size, 1, 4, '1');
    CountingSortParalelo(input, size, 1, 8, '1');
    CountingSortParalelo(input, size, 1, 16, '1');
    CountingSortParalelo(input, size, 1, 32, '1');
    CountingSortParalelo(input, size, 1, 64, '1');
    CountingSortParalelo(input, size, 1, 128, '1');
    CountingSortParalelo(input, size, 1, 256, '1');
    CountingSortParalelo(input, size, 1, 512, '1');
    CountingSortParalelo(input, size, 1, 1024, '1');
    CountingSortParalelo(input, size, 5, 256, '1');
    CountingSortParalelo(input, size, 6, 256, '1');
    CountingSortParalelo(input, size, 7, 256, '1');
    CountingSortParalelo(input, size, 2, 1024, '1');
}


int main(int argc, char* argv[]) {

    //PRESENTACION
    presentacion();
    
    int size = 10;
    int* input = generarArrayAleatorio(size, 1, 65535);
    int numBloques = 1;
    int numHilosPorBloque = 32;
    char metodoCreaSorted = '1';

    CountingSortParalelo(input, size, numBloques, numHilosPorBloque, metodoCreaSorted);
        
    //ANTES DE EJECUTAR LAS PRUEBAS, COMENTA LOS PRINTS DE SALIDA LINEAS [657,670]
    //prueba1();
    //prueba2();
    //prueba3();
    //prueba4();

    return 0;
}