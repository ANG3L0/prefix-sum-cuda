// MP Scan
// Given a list (lst) of length n
// Output its prefix sum = {lst[0], lst[0] + lst[1], lst[0] + lst[1] + ... + lst[n-1]}

#include    <wb.h>

#define BLOCK_SIZE 256 //@@ You can change this

#define wbCheck(stmt) do {                                                    \
  cudaError_t err = stmt;                                               \
  if (err != cudaSuccess) {                                             \
    wbLog(ERROR, "Failed to run stmt ", #stmt);                       \
    wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));    \
    return -1;                                                        \
  }                                                                     \
} while(0)

__global__ void scan(float * input, float * output, int len) {

  __shared__ float subArray[BLOCK_SIZE*2];
  unsigned int t = threadIdx.x;
  unsigned int start = 2*blockIdx.x*blockDim.x;

  //load subarray into the shared subarray vector
  int firstHalfIdx = start + t;
  int secondHalfIdx = start + blockDim.x + t;
  subArray[t] = firstHalfIdx < len ? input[firstHalfIdx] : 0.0f;
  subArray[t + blockDim.x] = secondHalfIdx < len ? input[secondHalfIdx] : 0.0f;
  //reduction phase to finish off all powers of 2 indices.
  __syncthreads(); //wait for load to finish.
  for (int stride = 1; stride <= BLOCK_SIZE; stride *= 2) {
    //produce indices of the following form:
    //stride = 1 : 1, 3, 5, 7, ...
    //stride = 2 : 3, 7, 11, 15, ...
    //stride = 3 : 7, 15, 23, 31, ...
    int index = (threadIdx.x + 1)*stride*2 - 1;
    if (index < 2*BLOCK_SIZE)
      subArray[index] += subArray[index - stride];
    __syncthreads(); //wait for writes to finish before moving on to the next iter.
  }
  //reverse reduction phase to finish off all non-power-of-2 indices.
  for (int stride = BLOCK_SIZE/2; stride > 0; stride/=2){
    int index = (threadIdx.x + 1)*stride*2 - 1; //same indices
    if (index + stride < 2*BLOCK_SIZE) {
      subArray[index + stride] += subArray[index]; //but add in other direction.
    }
    __syncthreads();
  }
  if (firstHalfIdx < len)
    output[firstHalfIdx] = subArray[t];
  if (secondHalfIdx < len)
    output[secondHalfIdx] = subArray[t + blockDim.x];
}

__global__ void saveToList(float * input, float * sumList, int len) {
  //given a list of subarrays that are correct, generate another list
  //that holds the last value of each subarray
  //e.g. [3,6,8,11 | 12,13,14,15 | 27,39,40,42 | 55 ]
  //output [11,15,42]
  unsigned int t = threadIdx.x;
  unsigned int idx = (t + 1)*BLOCK_SIZE*2 - 1;
  if (idx < len) {
    sumList[t] = input[idx];
  }
}

__global__ void addToOutput(float * output, float * sumList, int len) {
  //given output [11,15,42]
  //Add output[idx] to [blockDim.x*(idx+1), blockDim.x*(idx+2)-1]
  unsigned int t = threadIdx.x;
  unsigned int idx = t + blockDim.x*blockIdx.x;
  unsigned int fillIdx = idx + blockDim.x; //offset this by 1 block.

  if (fillIdx < len) {
    output[fillIdx] += sumList[blockIdx.x]; //boundchecking for sumlist not necessary
    //as fillIdx < len should make it so that we don't access outside of sumList
  }

}
int main(int argc, char ** argv) {
  wbArg_t args;
  float * hostInput; // The input 1D list
  float * hostOutput; // The output list
  float * deviceInput;
  float * deviceOutput;
  float * sumList;
  float * sumListOutput;
  int numElements; // number of elements in the list

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostInput = (float *) wbImport(wbArg_getInputFile(args, 0), &numElements);
  hostOutput = (float*) malloc(numElements * sizeof(float));
  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The number of input elements in the input is ", numElements);

  wbTime_start(GPU, "Allocating GPU memory.");
  wbCheck(cudaMalloc((void**)&deviceInput, numElements*sizeof(float)));
  wbCheck(cudaMalloc((void**)&deviceOutput, numElements*sizeof(float)));
  wbCheck(cudaMalloc((void**)&sumList, ((numElements-1)/BLOCK_SIZE + 1) * sizeof(float)));
  wbCheck(cudaMalloc((void**)&sumListOutput, ((numElements-1)/BLOCK_SIZE + 1) * sizeof(float)));
  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Clearing output memory.");
  wbCheck(cudaMemset(deviceOutput, 0, numElements*sizeof(float)));
  wbTime_stop(GPU, "Clearing output memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  wbCheck(cudaMemcpy(deviceInput, hostInput, numElements*sizeof(float), cudaMemcpyHostToDevice));
  wbTime_stop(GPU, "Copying input memory to the GPU.");

  //@@ Initialize the grid and block dimensions here
  dim3 DimGrid((numElements-1)/(BLOCK_SIZE*2) + 1,1,1);
  dim3 DimBlock(BLOCK_SIZE,1,1);
  dim3 SaveGrid(1,1,1);
  dim3 SaveBlock((numElements-1)/(BLOCK_SIZE*2)+1,1,1);
  //divide by extra factor of 2 because each subarray is BLOCK_SIZE*2 large.
  dim3 AccumGrid(1,1,1);
  dim3 AccumBlock((numElements-1)/(BLOCK_SIZE*2)+1,1,1);
  dim3 AddBlock(2*BLOCK_SIZE,1,1);

  wbTime_start(Compute, "Performing CUDA computation");
  //@@ Modify this to complete the functionality of the scan
  //@@ on the deivce
  //initial scan on subarrays
  scan<<<DimGrid,DimBlock>>>(deviceInput, deviceOutput, numElements);
  //save deviceOutput[idx % BLOCK_SIZE == 0 OR if BLOCK_SIZE*idx out of bounds, last value] -> sumList[idx]
  saveToList<<<SaveGrid,SaveBlock>>>(deviceOutput,sumList, numElements);
  //scan on the sumlist
  scan<<<AccumGrid,AccumBlock>>>(sumList, sumListOutput, (numElements-1)/(BLOCK_SIZE*2)+1);
  //add sumlist to deviceOutput e.g. deviceOutput[idx] += sumList[idx/BLOCK_SIZE];
  addToOutput<<<DimGrid,AddBlock>>> (deviceOutput,sumListOutput, numElements);


  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  wbCheck(cudaMemcpy(hostOutput, deviceOutput, numElements*sizeof(float), cudaMemcpyDeviceToHost));
  wbTime_stop(Copy, "Copying output memory to the CPU");

  wbTime_start(GPU, "Freeing GPU Memory");
  cudaFree(deviceInput);
  cudaFree(deviceOutput);
  wbTime_stop(GPU, "Freeing GPU Memory");

  wbSolution(args, hostOutput, numElements);

  free(hostInput);
  free(hostOutput);

  return 0;
}


