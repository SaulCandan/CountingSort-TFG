# 🚀 Counting Sort Paralelo con CUDA

Este repositorio contiene el Trabajo de Fin de Grado titulado **"Implementación Paralela del Algoritmo Counting Sort usando CUDA"**, desarrollado en el entorno de Visual Studio con C/C++ y arquitectura CUDA, orientado al aprovechamiento de la capacidad de procesamiento paralelo de las GPUs.

## 📚 Introducción

Con el avance constante en la potencia de cómputo y el auge de aplicaciones que requieren procesamiento intensivo, surge la necesidad de algoritmos eficientes y paralelizables. Un caso relevante es el **procesamiento de imágenes**, donde algoritmos de ordenación como **Counting Sort** pueden jugar un papel clave.

**Counting Sort** es un algoritmo de ordenación no comparativo diseñado para ordenar números enteros en un rango conocido, lo que lo hace ideal para su paralelización.

Este proyecto explora la implementación de **Counting Sort en CUDA**, analizando su funcionamiento, rendimiento y ventajas frente a una versión secuencial tradicional.

## 🎯 Objetivos

### Objetivo general
- Implementar el algoritmo **Counting Sort** de manera paralela utilizando **CUDA** en C/C++ desde **Visual Studio**.

### Objetivos específicos
- ✅ Diseñar un algoritmo capaz de ordenar arrays de enteros positivos en paralelo.
- ✅ Optimizar el código para maximizar rendimiento y minimizar consumo de recursos.
- ✅ Asegurar la mantenibilidad del código mediante buenas prácticas de programación.
- ✅ Gestionar adecuadamente errores, especialmente en casos de falta de memoria.
- ✅ Realizar pruebas sobre diferentes tamaños de arrays y rangos de valores.
- ✅ Analizar los resultados mediante gráficas de rendimiento.

## 🛠️ Tecnologías utilizadas

- 💻 **Lenguaje**: C / CUDA C++
- 🧠 **Paralelización**: NVIDIA CUDA

## 📈 Resultados esperados

- Aceleración significativa en el tiempo de ejecución respecto a versiones secuenciales.
- Pruebas que validan la escalabilidad y robustez del algoritmo.
- Gráficas de rendimiento en función del tamaño del array y del rango de valores.
