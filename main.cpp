#include <stdio.h>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

static const int NumElemets = 10000;
static const int MaxDevices = 10;
static const int MaxLogSize = 5000;

static float In1[NumElemets];
static float In2[NumElemets];
static float Out[NumElemets];

static void printBuildLog(const cl_program program, const cl_device_id device);
static void printError(const cl_int err);

int main() {
	cl_int status;

	// 1. Create Context
	cl_context context;
	context = clCreateContextFromType(NULL, CL_DEVICE_TYPE_CPU, NULL, NULL, &status);
	if (status != CL_SUCCESS) {
		fprintf(stderr, "clCreateContextFromType failed.\n");
		printError(status);
		return 1;
	}
	//2. Get Device in Context
	cl_device_id devices[MaxDevices];
	size_t size_return;

	status = clGetContextInfo(context, CL_CONTEXT_DEVICES, sizeof(devices), devices, &size_return);
	if (status != CL_SUCCESS) {
		fprintf(stderr, "clGetContextInfo failed.\n");
		printError(status);
		return 2;
	}

	// 3. Create Command Queue
	cl_command_queue queue;
	queue = clCreateCommandQueue(context, devices[0], 0, &status);
	if (status != CL_SUCCESS) {
		fprintf(stderr, "clCreateCommandQueue failed.\n");
		printError(status);
		return 3;
	}

	// 4. Create Program Object
	static const char* sources[] = {
		"__kernel void\n\
		addVector(__global const float *in1, __global const float *in2, __global float *out)\n\
		{\n\
			int index = get_global_id(0);\n\
			out[index] = in1[index] + in2[index];\n\
		}\n" };
	cl_program program;
	program = clCreateProgramWithSource(context, 1, (const char**)&sources, NULL, &status);
	if (status != CL_SUCCESS) {
		fprintf(stderr, "clCreateProgramWithSource failed.\n");
		printError(status);
		return 4;
	}
	
	// 5. Build the Program
	status = clBuildProgram(program, 1, devices, NULL, NULL, NULL);
	if (status != CL_SUCCESS) {
		fprintf(stderr, "clBuildProgram failed.\n");
		printError(status);
		printBuildLog(program, devices[0]);
		return 5;
	}
	clUnloadCompiler();

	// 6. Create Kernel
	cl_kernel kernel;
	kernel = clCreateKernel(program, "addVector", &status);
	if (status != CL_SUCCESS) {
		fprintf(stderr, "clCreateKernel failed.\n");
		printError(status);
		return 6;
	}

	// 7. Create Memory Object
	for (int i = 0; i < NumElemets; i++) {
		In1[i] = (float)i * 100.0f;
		In2[i] = (float)i / 100.0f;
		Out[i] = 0.0f;
	}

	cl_mem memIn1;
	memIn1 = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(cl_float) * NumElemets, In1, &status);
	if (status != CL_SUCCESS) {
		fprintf(stderr, "clCreateBuffer for memIn1 failed.\n");
		printError(status);
		return 7;
	}

	cl_mem memIn2;
	memIn2 = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(cl_float) * NumElemets, In2, &status);
	if (status != CL_SUCCESS) {
		fprintf(stderr, "clCreateBuffer for memIn2 failed.\n");
		printError(status);
		return 7;
	}

	cl_mem memOut;
	memOut = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(cl_float) * NumElemets, NULL, &status);
	if (status != CL_SUCCESS) {
		fprintf(stderr, "clCreateBuffer for memOut failed.\n");
		printError(status);
		return 7;
	}

	// 8. Set Kernel Arg
	status = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&memIn1);
	if (status != CL_SUCCESS) {
		fprintf(stderr, "clSetKernelArg for memIn1 failed.\n");
		printError(status);
		return 8;
	}

	status = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&memIn2);
	if (status != CL_SUCCESS) {
		fprintf(stderr, "clSetKernelArg for memIn2 failed.\n");
		printError(status);
		return 8;
	}

	status = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&memOut);
	if (status != CL_SUCCESS) {
		fprintf(stderr, "clSetKernelArg for memOut failed.\n");
		printError(status);
		return 8;
	}

	// 9. Implement Kernel
	size_t globalSize[] = { NumElemets };
	status = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, globalSize, 0, 0, NULL, NULL);
	if (status != CL_SUCCESS) {
		fprintf(stderr, "clEnqueueNDRangeKernel.\n");
		printError(status);
		return 9;
	}

	// 10. Get Result
	status = clEnqueueReadBuffer(queue, memOut, CL_TRUE, 0, sizeof(cl_float) * NumElemets, Out, 0, NULL, NULL);
	if (status != CL_SUCCESS) {
		fprintf(stderr, "clEnqueueReadBuffer.\n");
		printError(status);
		return 10;
	}

	printf("(In1, In2, Out)\n");
	for (int i = 0; i < 100; i++) {
		printf("%f, %f, %f (%f) \n", In1[i], In2[i], Out[i], In1[i] + In2[i]);
	}

	// 11. Release Resources
	clReleaseMemObject(memOut);
	clReleaseMemObject(memIn2);
	clReleaseMemObject(memIn1);
	clReleaseKernel(kernel);
	clReleaseProgram(program);
	clReleaseCommandQueue(queue);
	clReleaseContext(context);

	return 0;
}

static void printBuildLog(const cl_program program, const cl_device_id device) {
	cl_int status;
	size_t size_ret;

	char buffer[MaxLogSize + 1];
	status = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, MaxLogSize, buffer, &size_ret);

	if (status == CL_SUCCESS) {
		buffer[size_ret] = '\0';
		printf(">>>build log<<<\n");
		printf("%s\n", buffer);
		printf(">>> end of build log <<<\n");
	}
	else {
		printf("clGetProgramInfo failed.\n");
		printError(status);
	}
}

static void printError(const cl_int err) {
	switch (err) {
	case CL_BUILD_PROGRAM_FAILURE :
		fprintf(stderr, "Program Build failed\n");
		break;
	case CL_DEVICE_NOT_FOUND:
		fprintf(stderr, "Device not found\n");
		break;
	case CL_INVALID_CONTEXT :
		fprintf(stderr, "Invaild context\n");
		break;
	case CL_INVALID_DEVICE :
		fprintf(stderr, "Invaild device\n");
		break;
	case CL_INVALID_DEVICE_TYPE:
		fprintf(stderr, "Invaild device type\n");
		break;
	case CL_INVALID_PLATFORM:
		fprintf(stderr, "Invalid platform\n");
		break;
	default:
		fprintf(stderr, "Unknown error code : %d\n", err);
		break;
	}
}