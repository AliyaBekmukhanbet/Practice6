#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#define CHECK(err) do { if (err != CL_SUCCESS) { \
    fprintf(stderr, "OpenCL error %d at %s:%d\n", err, __FILE__, __LINE__); \
    exit(1); }} while(0)

int main()
{
    const int N = 1024;   // A: N×M
    const int M = 1024;   // B: M×K
    const int K = 1024;   // C: N×K

    // ─── 1. Подготовка данных ───────────────────────────────────────────
    size_t sizeA = N * M * sizeof(float);
    size_t sizeB = M * K * sizeof(float);
    size_t sizeC = N * K * sizeof(float);

    float *h_A = (float*)malloc(sizeA);
    float *h_B = (float*)malloc(sizeB);
    float *h_C_cpu = (float*)malloc(sizeC);
    float *h_C_opencl = (float*)malloc(sizeC);

    // Заполняем матрицы (можно использовать rand() или фиксированные значения)
    srand(time(NULL));
    for (int i = 0; i < N*M; i++) h_A[i] = (float)(rand() % 100) / 10.0f;
    for (int i = 0; i < M*K; i++) h_B[i] = (float)(rand() % 100) / 10.0f;

    // ─── 2. Инициализация OpenCL ────────────────────────────────────────
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel kernel;

    CHECK(clGetPlatformIDs(1, &platform, NULL));

    // Предпочитаем GPU, если нет — CPU
    cl_uint num_devices;
    CHECK(clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, &num_devices));
    if (num_devices == 0) {
        printf("GPU не найдено → используем CPU\n");
        CHECK(clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &device, NULL));
    }

    char dev_name[128];
    clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(dev_name), dev_name, NULL);
    printf("Устройство: %s\n", dev_name);

    context = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);
    queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, NULL);

    // ─── 3. Загрузка и компиляция ядра ──────────────────────────────────
    FILE *f = fopen("kernel.cl", "rb");
    if (!f) { perror("kernel.cl"); exit(1); }
    fseek(f, 0, SEEK_END);
    size_t src_size = ftell(f);
    rewind(f);
    char *src = malloc(src_size + 1);
    fread(src, 1, src_size, f);
    src[src_size] = '\0';
    fclose(f);

    program = clCreateProgramWithSource(context, 1, (const char**)&src, &src_size, NULL);
    free(src);

    cl_int build_err = clBuildProgram(program, 1, &device, "", NULL, NULL);
    if (build_err != CL_SUCCESS) {
        size_t log_size;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        char *log = malloc(log_size);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
        fprintf(stderr, "Build error:\n%s\n", log);
        free(log);
        exit(1);
    }

    kernel = clCreateKernel(program, "matmul", NULL);

    // ─── 4. Буферы на устройстве ────────────────────────────────────────
    cl_mem d_A = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeA, h_A, NULL);
    cl_mem d_B = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeB, h_B, NULL);
    cl_mem d_C = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeC, NULL, NULL);

    // ─── 5. Запуск ядра ─────────────────────────────────────────────────
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_A);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_B);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_C);
    clSetKernelArg(kernel, 3, sizeof(int), &N);
    clSetKernelArg(kernel, 4, sizeof(int), &M);
    clSetKernelArg(kernel, 5, sizeof(int), &K);

    size_t global_work_size[2] = {N, K};
    size_t local_work_size[2]  = {16, 16};   // типичный размер для начала

    cl_event event;
    clock_t t_start = clock();
    clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global_work_size, local_work_size, 0, NULL, &event);
    clFinish(queue);
    clock_t t_end = clock();

    double time_opencl_ms = (double)(t_end - t_start) * 1000.0 / CLOCKS_PER_SEC;
    printf("OpenCL время: %.1f мс\n", time_opencl_ms);

    // Читаем результат
    clEnqueueReadBuffer(queue, d_C, CL_TRUE, 0, sizeC, h_C_opencl, 0, NULL, NULL);

    // ─── 6. Проверка на CPU (последовательная версия) ───────────────────
    t_start = clock();
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < K; j++) {
            float sum = 0.0f;
            for (int m = 0; m < M; m++) {
                sum += h_A[i*M + m] * h_B[m*K + j];
            }
            h_C_cpu[i*K + j] = sum;
        }
    }
    t_end = clock();
    double time_cpu_ms = (double)(t_end - t_start) * 1000.0 / CLOCKS_PER_SEC;
    printf("CPU (последовательный) время: %.1f мс\n", time_cpu_ms);

    // ─── 7. Сравнение результатов ───────────────────────────────────────
    int errors = 0;
    float eps = 1e-4f;
    for (int i = 0; i < N*K; i++) {
        if (fabs(h_C_opencl[i] - h_C_cpu[i]) > eps) {
            errors++;
            if (errors < 5) {
                printf("Ошибка в элементе %d: CPU=%.4f  OpenCL=%.4f\n", i, h_C_cpu[i], h_C_opencl[i]);
            }
        }
    }
    printf("Количество ошибок: %d (из %d элементов)\n", errors, N*K);
    if (errors == 0) printf("Результаты совпадают (в пределах погрешности)\n");

    // ─── Освобождение ────────────────────────────────────────────────────
    clReleaseMemObject(d_A);
    clReleaseMemObject(d_B);
    clReleaseMemObject(d_C);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    free(h_A); free(h_B); free(h_C_cpu); free(h_C_opencl);

    return 0;
}
