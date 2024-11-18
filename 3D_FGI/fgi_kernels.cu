#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/functional.h>
#include <thrust/device_malloc.h>
#include <thrust/device_free.h>
#include <thrust/device_vector.h>
#include <chrono>
#include "tmwtypes.h"
#include "cxxopts.hpp"
#include "fgi_device_functions.h"

__global__ void computeImpulsivity(const double* imageIn, double* impulsivityDegrees, uint32_T rows, uint32_T cols,  uint32_T cuts, uint32_T roadElements, uint32_T windowSize){
	
	float distances[15]; //60 Bytes
	uint32_T windowIndexes[15]; //60 Bytes
	uint32_T x =  blockIdx.x * blockDim.x + threadIdx.x; 
	uint32_T y = blockIdx.y * blockDim.y + threadIdx.y; 
	uint32_T index = x+y*cols*rows;
	
	if(x < cols*rows && y < cuts){
		float road_m = 0;
		//uint32_T offset = y*cols*rows;
		uint8_T maxRange = (2*windowSize+1)*(2*windowSize+1)*(2*windowSize+1)-12;
		uint8_T ws = maxRange;
		calcWindowIndexes3dfast_CS(windowIndexes, ws, rows, cols, cuts, windowSize, (2*windowSize+1), x, y);
		ws = min(ws, maxRange);
		
		uint8_T elements = min(ws, roadElements);
		road_m = calculateRoad3dSP(windowIndexes, ws, imageIn, index, elements, distances);

		if(road_m <= P1) {
			impulsivityDegrees[index] = 0;
			//return 0;
		} else if(road_m >= P2) {
			impulsivityDegrees[index] = 1;
			//return 1;
		} else {
			impulsivityDegrees[index] = (road_m - P1) / (P2 - P1);
		}
	}
}


__global__ void fuzzyFilter(double* imageIn, uint32_T windowSize, const double* impulsivityDegrees, uint32_T rows, uint32_T cols, uint32_T cuts, uint32_T q) {
	float dist[15]; //60 Bytes
	uint32_T windowIndexes[15]; //60 Bytes
    uint32_T x =  blockIdx.x * blockDim.x + threadIdx.x; 
	uint32_T y = blockIdx.y * blockDim.y + threadIdx.y; 
	uint32_T index = x+y*cols*rows;
	
	if(x < cols*rows && y < cuts){
		float oldPx = imageIn[index], newPx;
		//uint32_T offset = y*cols*rows;
		uint8_T maxRange = (2*windowSize+1)*(2*windowSize+1)*(2*windowSize+1)-12;
		
		uint8_T ws = maxRange;
		calcWindowIndexes3dfast_CS(windowIndexes, ws, rows, cols, cuts, windowSize, (2*windowSize+1), x, y);
		for(uint i = 0; i < ws; i++){
			dist[i]=fabs(oldPx-imageIn[windowIndexes[i]]);
		}
		bubbleSortSP(dist, windowIndexes, ws);

		uint8_T localq = min(ws,q);
		float numerator = 0;
		float totalWeights = 0;

		for(uint32_T i = 0; i < localq; i++) {
			float pixel = imageIn[windowIndexes[i]];
			float weight = calculateWeight(index, oldPx, windowIndexes[i], pixel, impulsivityDegrees);
			numerator += pixel * weight;
			totalWeights += weight;
		}
		
		newPx =  numerator / totalWeights;
		imageIn[index] = newPx;
	}
}
