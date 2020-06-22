//NOTE : RGB2GRAY, ZNCC and OCCLUSION FILLIN COMPLETELY IMPLEMENTED BUT IMAGE RESIZING WHICH IS A SIMPLE STEP IS NOT DONE YET, RESIZED IMAGES ARE READ
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "lodepng.h"
#include <iostream>
#include <fstream>
#include <stdio.h>

using namespace std;


void encodeOneStep(const char* filename, std::vector<unsigned char>& image, unsigned width, unsigned height);
int SaveDepthMap(const char* filename, std::vector<unsigned char>& Depth, unsigned width, unsigned height);

__global__ void RGB2Gray(const unsigned char *OriginalimageL, const unsigned char *OriginalimageR, unsigned char *imageL, unsigned char *imageR, int Width, int Height)
{
	int i = blockIdx.x ;
	int j = threadIdx.x ;
	int M = Width;
	int N = Height;

	imageL[i*N + j] = (char)(0.2126*OriginalimageL[(i*N + j) * 4] + 0.7152*OriginalimageL[(i*N + j) * 4 + 1] + 0.0722*OriginalimageL[(i*N + j) * 4 + 2]);
	imageR[i*N + j] = (char)(0.2126*OriginalimageR[(i*N + j) * 4] + 0.7152*OriginalimageR[(i*N + j) * 4 + 1] + 0.0722*OriginalimageR[(i*N + j) * 4 + 2]);

}


__global__ void ZNNC_Kernel(const unsigned char *imageL, const unsigned char *imageR, unsigned char *DisparityMap, int width, int height)
{
    int i = blockIdx.x;
	int j = threadIdx.x;
	int M = width;
	int N = height;
	//if (i+j<3) printf("blockIdx.x: %d, threadIdx.x: %d, M: %d, Ny: %d \n", blockIdx.x,  threadIdx.x, M, N);
	int W = 9;
	int DM = 55;

	if (i >= M-DM-W)
		i = M-DM-W;
	if (i <= DM+W)
		i = DM+W;
	if (j >= N-W)
		j = N-W;
	if (j <= W)
		j = W;


	int BestDisparityValue = 0;
	float CurrentMaximum = (float)-1;
	for (char d = -DM; d <DM; d++) {
		float Nomi = 0, Denomi1 = 0, Denomi2 = 0;
		int counterTest = 0;
		for (int x = 0; x < W; x++)
			for (int y = 0; y< W; y++)
			{
				Nomi = Nomi + (imageL[(i + x - d) + (j + y)*M])*(imageR[(i + x) + (j + y)*M]);
				Denomi1 = Denomi1 + (float)(pow((float)imageL[(i + x - d) + (j + y)*M], (float)2));
				Denomi2 = Denomi2 + (float)(pow((float)imageR[(i + x) + (j + y)*M], (float)2));
				counterTest++;
			}
		 float ZNCCValue = fabs((float)Nomi / (float)sqrt((float)Denomi1*Denomi2));
		if (ZNCCValue >= CurrentMaximum) {
			CurrentMaximum = ZNCCValue;
			BestDisparityValue = abs(d) * 255 / DM;
			//					BestDisparityValue = 233;
		}
	}
	DisparityMap[i + j*M] = BestDisparityValue;// (BestDisparityValue);
											   //		DisparityMap[i + j*M] = 111;
	
};

__global__ void Post_Processing(const unsigned char *RightDisparity, const unsigned char *LeftDisparity, unsigned char *Porcessed_Disparity, int width, int height)
{

	int i = blockIdx.x;
	int j = threadIdx.x;
	int M = width;
	int N = height;
	int Threshold = 18;

	int Diff = abs(LeftDisparity[i + j* M] - RightDisparity[i + j* M]);
	if (Diff < Threshold)
		Porcessed_Disparity[i + j*M] = LeftDisparity[i + j* M];
	else
		Porcessed_Disparity[i + j*M] = 0;
}
//
int main()
{

	/*Reading the data from disk to memory and preparing for deeding to GPU*/
	char *filenameL = "sim0.png";
	char *filenameR = "sim1.png";
	unsigned int width, height, h, w;
	std::vector<unsigned char> imageOriginalL, imageOriginalR;
	unsigned error1 = lodepng::decode(imageOriginalL, width, height, filenameL);
	unsigned error2 = lodepng::decode(imageOriginalR, h, w, filenameR);
	std::vector<unsigned char> DisparityMapL2R(width* height), DisparityMapR2L(width* height),Processed(width*height); //Vector of Disparity map


	//the pixels are now in the vector "image", 4 bytes per pixel, ordered RGBARGBA..., use it as texture, draw it, ...
	std::vector<unsigned char> imageL(width * height), imageR(width * height);




	/* CUDA SECTION*/
	cudaError_t cudaStatus;
	cudaEvent_t start_Kernel1, stop_Kernel1;
	cudaEventCreate(&start_Kernel1);
	cudaEventCreate(&stop_Kernel1);
	cudaEventRecord(start_Kernel1, 0);
	cudaEventRecord(stop_Kernel1, 0);

	//Stage 1 : Choosing the device
	// Choose the GPU to run on
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		printf("cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
	}

	//Stage 2 : Allocating memory
	// Allocate GPU buffers for three right and left Image and disparity map
	unsigned char *Colourimageleft, *Colourimageright,*imageleft, *imageright, *disparityL2R, *disparityR2L,*ProcessedDispary;
	cudaStatus = cudaMalloc((void**)&Colourimageleft, 4*width* height * sizeof(char));
	if (cudaStatus != cudaSuccess) {
		printf("1.cudaMalloc failed!");
	}
	cudaStatus = cudaMalloc((void**)&Colourimageright, 4*width* height * sizeof(char));
	if (cudaStatus != cudaSuccess) {
		printf("1.cudaMalloc failed!");
	}
	cudaStatus = cudaMalloc((void**)&imageleft, width* height * sizeof(char));
	if (cudaStatus != cudaSuccess) {
		printf("1.cudaMalloc failed!");
	}
	cudaStatus = cudaMalloc((void**)&imageright, width* height * sizeof(char));
	if (cudaStatus != cudaSuccess) {
		printf("2.cudaMalloc failed!");
	}
	cudaStatus = cudaMalloc((void**)&disparityL2R, width* height * sizeof(char));
	if (cudaStatus != cudaSuccess) {
		printf("3.cudaMalloc failed!");
	}
	cudaStatus = cudaMalloc((void**)&disparityR2L, width* height * sizeof(char));
	if (cudaStatus != cudaSuccess) {
		printf("4.cudaMalloc failed!");
	}
	cudaStatus = cudaMalloc((void**)&ProcessedDispary, width* height * sizeof(char));
	if (cudaStatus != cudaSuccess) {
		printf("5.cudaMalloc failed!");
	}

	//Stage 3 : Loading the memory from host
	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(Colourimageleft, imageOriginalL.data(), 4 * width * height * sizeof(char), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		printf("cudaMemcpy failed!");
	}
	cudaStatus = cudaMemcpy(Colourimageright, imageOriginalR.data(), 4 * width * height * sizeof(char), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		printf("cudaMemcpy failed!");
	}


	//Stage 4 : Launching the Kernel
	// Launch a kernel on the GPU with one thread for each element.
	//dim3 D(width,height); Too many blocks causes crash due to TDR
	//First we need to make the images from RBG to GRAY (Of course, this step can be integrated with resizing step but I do not have time to debig!)
	RGB2Gray <<< width, height >>> (Colourimageleft, Colourimageright, imageleft, imageright, width, height);
	cudaEventSynchronize(stop_Kernel1);
	float timeK1;
	cudaEventElapsedTime(&timeK1, start_Kernel1, stop_Kernel1);
	printf("Elapsed Time for Performing RGB2Gray: %f\n", timeK1);
	cudaEventDestroy(start_Kernel1);
	cudaEventDestroy(stop_Kernel1);



	//width number of blocks ,height number of thread in each block

	cudaEvent_t start_Kernel2, stop_Kernel2;
	cudaEventCreate(&start_Kernel2);
	cudaEventCreate(&stop_Kernel2);
	cudaEventRecord(start_Kernel2, 0);
	cudaEventRecord(stop_Kernel2, 0);

	ZNNC_Kernel <<< width, height >>>(imageleft, imageright, disparityL2R,width,height); // Of course this is not the best and most optimum option
	//Stage 5 : Checking for possible errors and timing
	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.

	cudaStatus=cudaEventSynchronize(stop_Kernel2);
	if (cudaStatus != cudaSuccess) {
		printf( "cudaDeviceSynchronize returned error code %d after launching Left_to_Rights_Kernel!\n", cudaStatus); 
		printf(cudaGetErrorString(cudaStatus)); printf("!\n");
	}
	float timeK2;
	cudaEventElapsedTime(&timeK2, start_Kernel2, stop_Kernel2);
	printf("Elapsed Time for Performing ZNCC1 Kernel: %f\n", timeK2);
	cudaEventDestroy(start_Kernel2);
	cudaEventDestroy(stop_Kernel2);
	//Stage 5 : Getting Back the results
	// Copy output vector from GPU buffer to host memory.
	unsigned char *data;
	data = (unsigned char *)malloc(sizeof( char)*width*height);
	*(data + 10000) = 121;
	cudaStatus = cudaMemcpy(data, disparityL2R, width*height * sizeof(char), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		printf("cudaMemcpy failed!");
	}
	*(data + 1) = 121;

	for (int i = 0; i < width; i++) {  // Normalization of image to gray scale value
		for (int j = 0; j < height; j++) {
			DisparityMapL2R[i*height + j] = *(data+i*height+j);
		}
	}

	
	/*Left to Right Image*/

	cudaEvent_t start_Kernel3, stop_Kernel3;
	cudaEventCreate(&start_Kernel3);
	cudaEventCreate(&stop_Kernel3);
	cudaEventRecord(start_Kernel3, 0);
	cudaEventRecord(stop_Kernel3, 0);
	ZNNC_Kernel << < width, height >> >(imageright, imageleft, disparityR2L, width, height); // Of course this is not the best and most optimum option
	cudaStatus = cudaEventSynchronize(stop_Kernel3);
	if (cudaStatus != cudaSuccess) {
		printf("cudaDeviceSynchronize returned error code %d after launching Left_to_Rights_Kernel!\n", cudaStatus);
		printf(cudaGetErrorString(cudaStatus)); printf("!\n");
	}
	float timeK3;
	cudaEventElapsedTime(&timeK3, start_Kernel3, stop_Kernel3);
	printf("Elapsed Time for Performing ZNCC2 Kernel: %f\n", timeK3);
	cudaEventDestroy(start_Kernel3);
	cudaEventDestroy(stop_Kernel3);
 // Repitiiton of Stage 5 for right to left matching image  : Getting Back the results
 // Copy output vector from GPU buffer to host memory.

	cudaStatus = cudaMemcpy(data, disparityR2L, width*height * sizeof(char), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		printf("cudaMemcpy failed!");
	}

	for (int i = 0; i < width; i++) {  // Normalization of image to gray scale value
		for (int j = 0; j < height; j++) {
			DisparityMapR2L[i*height + j] = *(data + i*height + j);
		}
	}



	/*Saving two images*/
	SaveDepthMap("D_Right_to_Left.png", DisparityMapR2L, width, height); 
	SaveDepthMap("D_Left_to_Right.png", DisparityMapL2R, width, height);

	cudaEvent_t start_Kernel4, stop_Kernel4;
	cudaEventCreate(&start_Kernel4);
	cudaEventCreate(&stop_Kernel4);
	cudaEventRecord(start_Kernel4, 0);
	cudaEventRecord(stop_Kernel4, 0);

	Post_Processing << < width, height >> >(disparityL2R, disparityR2L, ProcessedDispary, width, height);
	cudaStatus = cudaEventSynchronize(stop_Kernel4);
	float timeK4;
	cudaEventElapsedTime(&timeK4, start_Kernel4, stop_Kernel4);
	printf("Elapsed Time for Performing PostProcessing Kernel: %f\n", timeK4);
	cudaEventDestroy(start_Kernel4);
	cudaEventDestroy(stop_Kernel4);

	cudaStatus = cudaMemcpy(data, ProcessedDispary, width*height * sizeof(char), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		printf("cudaMemcpy failed!");
	}

	for (int i = 0; i < width; i++) {  // Normalization of image to gray scale value
		for (int j = 0; j < height; j++) {
			Processed[i*height + j] = *(data + i*height + j);
		}
	}

	//Color_of_Nearest_Neighbor = 0; a simple mask form of neigherest neigbour filling
	for (int i = 0; i < width; i++) { // finding paired pixels
		char Color_of_Nearest_Neighbor = 0;
		for (int j = 0; j < height; j++) {
			if (Processed[i + j*width] == 0)
				Processed[i + j*width] = Color_of_Nearest_Neighbor;
			else
				Color_of_Nearest_Neighbor = Processed[i + j*width];
		}
	}
	SaveDepthMap("D_ProcessedImage.png", Processed, width, height);

	getchar();
    return 0;
}

void encodeOneStep(const char* filename, std::vector<unsigned char>& image, unsigned width, unsigned height)
{
	//Encode the image
	unsigned error = lodepng::encode(filename, image, width, height);

	//if there's an error, display it
	if (error) std::cout << "encoder error " << error << ": " << lodepng_error_text(error) << std::endl;
	//int A;
	//std::cin >> A;
}

int SaveDepthMap(const char* filename, std::vector<unsigned char>& Depth, unsigned width, unsigned height)
{
	std::vector<unsigned char> imageSave(width * height * 4);
	int i;
	for (i = 0; i <width; i++)  // Normalization of image to gray scale value
		for (unsigned int j = 0; j < height; j++) {
			imageSave[(i*height + j) * 4] = (Depth[i*height + j]);
			imageSave[(i*height + j) * 4 + 1] = (Depth[i*height + j]);
			imageSave[(i*height + j) * 4 + 2] = (Depth[i*height + j]);
			imageSave[(i*height + j) * 4 + 3] = (255);
		}
	encodeOneStep(filename, imageSave, width, height);
	return (-0);
}