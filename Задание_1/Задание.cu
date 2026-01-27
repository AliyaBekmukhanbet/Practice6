%%cuda
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#define N 20000000         // 20 миллионов элементов
#define CHECK(err) do { if(err != CL_SUCCESS) { \
    fprintf(stderr, "OpenCL error %d at line %d\n", err, __LINE__); \
    exit(1); }} while(0)

int main()
{
    cl_int err;
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel kernel;

    // 1. Инициализация платформы и устройства
    CHECK(clGetPlatformIDs(1, &platform, NULL));
    
    // Пробуем сначала GPU, если нет — CPU
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    if (err != CL_SUCCESS) {
        printf("GPU не найдено, использую CPU\n");
        CHECK(clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &device, NULL));
    }

    char device_name[128];
    clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(device_name), device_name, NULL);
    printf("Используемое устройство: %s\n", device_name);

    // 2. Контекст и очередь
    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    CHECK(err);
    queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err);
    CHECK(err);

    // 3. Загрузка и компиляция ядра
    FILE *f = fopen("kernel.cl", "rb");
    if (!f) { perror("kernel.cl"); exit(1); }
    fseek(f, 0, SEEK_END);
    size_t source_size = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *source = malloc(source_size + 1);
    fread(source, 1, source_size, f);
    source[source_size] = 0;
    fclose(f);

    program = clCreateProgramWithSource(context, 1, (const char**)&source, &source_size, &err);
    CHECK(err);
    free(source);

    err = clBuildProgram(program, 1, &device, "-cl-fast-relaxed-math", NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t log_size;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        char *log = malloc(log_size);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
        fprintf(stderr, "Build error:\n%s\n", log);
        free(log);
        exit(1);
    }

    kernel = clCreateKernel(program, "vector_add", &err);
    CHECK(err);

    // 4. Подготовка данных
    size_t bytes = N * sizeof(float);
    float *h_A = (float*)malloc(bytes);
    float *h_B = (float*)malloc(bytes);
    float *h_C = (float*)malloc(bytes);

    for(int i = 0; i < N; i++) {
        h_A[i] = (float)i * 0.1f;
        h_B[i] = (float)i * 0.01f + 1.5f;
    }

    cl_mem d_A = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,  bytes, h_A, &err);
    cl_mem d_B = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,  bytes, h_B, &err);
    cl_mem d_C = clCreateBuffer(context, CL_MEM_WRITE_ONLY, bytes, NULL, &err);
    CHECK(err);

    // 5. Настройка запуска ядра
    size_t global_size = ((N + 255) / 256) * 256;   // кратно 256
    size_t local_size  = 256;

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_A);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_B);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_C);
    clSetKernelArg(kernel, 3, sizeof(int),    &N);

    cl_event event;
    clock_t t_start = clock();
    CHECK(clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_size, &local_size, 0, NULL, &event));
    CHECK(clFinish(queue));
    clock_t t_end = clock();

    double cpu_time_ms = (double)(t_end - t_start) * 1000.0 / CLOCKS_PER_SEC;
    printf("Время выполнения ядра + finish: %.1f мс\n", cpu_time_ms);

    // 6. Получение результата
    CHECK(clEnqueueReadBuffer(queue, d_C, CL_TRUE, 0, bytes, h_C, 0, NULL, NULL));

    // Проверка первых 5 элементов
    printf("Проверка: C[0..4] = ");
    for(int i = 0; i < 5; i++) {
        printf("%.2f ", h_C[i]);
    }
    printf("\n");

    // 7. Освобождение ресурсов
    clReleaseMemObject(d_A);
    clReleaseMemObject(d_B);
    clReleaseMemObject(d_C);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    free(h_A); free(h_B); free(h_C);

    return 0;
}
