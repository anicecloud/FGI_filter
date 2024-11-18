#define P1 0.035
#define P2 0.06
#define P3 1.96/255
#define P4 0.9


__host__ __device__ double HighSimilarityEstimation(double distance) {
	if(distance <= P3)
		return 1;
	if(distance >= 4 * P3)
		return 0;

	return 4.0/3.0 - distance / (3 * P3);
}
__host__ __device__ double ModerateSimilarityEstimation(double distance) {
	if(distance > P3 && distance < 2 * P3)
		return (distance - P3) / P3;
	if(distance >= 2 * P3 && distance <= 3 * P3)
		return 1;
	if(distance > 3 * P3 && distance < 4 * P3)
		return 4.0 - distance / P3;

	return 0;
}


/*
 * Section 2.5: Fuzzy rules
 */

__host__ __device__ double S_norm(double x) {
	return x;
}

/**
 * Snorm used in the article to compute the fuzzy rules
 */
template<typename... Args>
__host__ __device__ inline double S_norm(double x, Args... args) {
	double y = S_norm(std::forward<Args>(args)...);
	return x + y - x * y;
}

__host__ __device__ inline double ModerateFuzzyRule(double idegree_i, double idegree_j, double MSE) {
	return (1 - idegree_j) * idegree_i * MSE;
}

__host__ __device__ inline double HighFuzzyRule(double idegree_i, double idegree_j, double HSE, double LSE) {

	double x = (1 - idegree_j);
	double y = idegree_i;

	return S_norm(x * y * LSE,
		      x * (1 - y) * HSE);
}

__host__ __device__ inline double LowFuzzyRule(double idegree_i, double idegree_j, double HSE, double MSE, double LSE) {
	double x = idegree_i;
	double y = (1 - idegree_j);
	
	return S_norm((1 - y),
		      y * x * HSE,
		      y * (1 - x) * MSE,
		      y * (1 - x) * LSE);
}


__host__ __device__ double calculateWeight(const uint32_T central, const double x_i, const uint32_T adjacent, 
                        const double x_j, const double* ImpulsivityDegree) {


	//Impulsivity degree for each pixel
	double impulsivityDegree_i = ImpulsivityDegree[central];
	double impulsivityDegree_j = ImpulsivityDegree[adjacent];

	double pixelsDistance = fabs(x_i-x_j);
	//High, medium and low similarity estimations
	double HSE = HighSimilarityEstimation(pixelsDistance);
	double MSE = ModerateSimilarityEstimation(pixelsDistance);
	double LSE = 1-HSE;
	
	
	// Fuzzy rules Low, Moderate and High.
	double LowFR = LowFuzzyRule(impulsivityDegree_i, impulsivityDegree_j,
				HSE, MSE, LSE);	
	double ModerateFR = ModerateFuzzyRule(impulsivityDegree_i, impulsivityDegree_j, MSE);	
	double HighFR = HighFuzzyRule(impulsivityDegree_i, impulsivityDegree_j, HSE, LSE);	

	/* Areas from the defuzzification diagram. 
	 * 'Low' part: Area1 + Area2
	 * 'Moderate' part: Area3 + Area4 + Area5 = AreaM
	 * 'High' part: Area 6 + Area7
	 */
	double Area1 = LowFR * (1 - LowFR) * (1 - P4);
	double Area2 = LowFR * LowFR * (1 - P4) / 2;
	double AreaM = ModerateFR * (2 - ModerateFR) * (2 * P4 - 1) / 2;
	double Area6 = (1 - P4) * HighFR * HighFR / 2;
	double Area7 = HighFR * (1 - HighFR) * (1 - P4);

	/*
	 * X-axis projection of the centers of gravity
	 * corresponding to the areas mentioned above.
	 */
	double CG1 = (1 - LowFR) * (1 - P4) / 2;
	double CG2 = (3 - 2 * LowFR) * (1 - P4) / 3;
	double CGM = 0.5;
	double CG6 = (3 * P4 + 2 * HighFR - 2 * P4 * HighFR) / 3;
	double CG7 = (1 - HighFR) * (1 - P4) / 2;

	double numerator = Area1 * CG1 + Area2 * CG2 + 
		AreaM * CGM + Area6 * CG6 + Area7 * CG7;
	double denominator = Area1 + Area2 + AreaM + Area6 + Area7;
	

	return numerator / denominator;
}

__host__ __device__ void distancesSorted(uint32_T* window,uint32_T ws, double* imageIn, uint32_T index, double* dist){
    
    for(uint i = 0; i < ws; i++){
        dist[i]=fabs(imageIn[index]-imageIn[window[i]]);
    }
    thrust::sort(thrust::seq,dist,dist+ws);
    /*for(uint i = 0; i < ws; i++){
		distances[i] = dist[i];
	}*/
}

__host__ __device__ void swap(double* xp, double* yp){
    double temp = *xp;
    *xp = *yp;
    *yp = temp;
}
__host__ __device__ void swap(float* xp, float* yp){
    float temp = *xp;
    *xp = *yp;
    *yp = temp;
}
__host__ __device__ void swap(uint32_T* xp, uint32_T* yp){
    uint32_T temp = *xp;
    *xp = *yp;
    *yp = temp;
}
__host__ __device__ void bubbleSort(double* key, uint32_T* val, int n){
    int i, j;
    for (i = 0; i < n - 1; i++)
	// Last i elements are already in place
        for (j = 0; j < n - i - 1; j++)
            if (key[j] > key[j + 1]){
                swap(&key[j], &key[j + 1]);
				swap(&val[j], &val[j + 1]);
			}
}
__host__ __device__ void bubbleSortSP(float* key, uint32_T* val, int n){
    int i, j;
    for (i = 0; i < n - 1; i++)
	// Last i elements are already in place
        for (j = 0; j < n - i - 1; j++)
            if (key[j] > key[j + 1]){
                swap(&key[j], &key[j + 1]);
				swap(&val[j], &val[j + 1]);
			}
}

__host__ __device__ float calculateRoad3dSP(uint32_T* window,uint32_T ws, const double* imageIn, uint32_T index, uint32_T elements, float* dist){
    float road_m = 0;
    for(uint i = 0; i < ws; i++){
		dist[i]=fabs(imageIn[index]-imageIn[window[i]]);
    }
	//Sort both arrays by dist 
	bubbleSortSP(dist, window, ws);
	road_m = thrust::reduce(thrust::device, dist, dist+elements, 0.0);
	return road_m;
}

__host__ __device__  void calculateLostRows(int64_T imageRows, int64_T x, uint32_T size, 
		uint32_T &lostUpperRows, uint32_T &lostLowerRows) {
	int64_T startingX = x - size;
	lostUpperRows = startingX < 0 ? -startingX : 0;

	int64_T biggestX = x + size;
	lostLowerRows = biggestX > (imageRows - 1)? biggestX - imageRows + 1: 0;
}

__host__ __device__  void calculateLostCols(int64_T imageCols, int64_T y, uint32_T size,
		uint32_T &lostLeftCols, uint32_T &lostRightCols) {
	int64_T startingY = y - size;
	lostLeftCols = startingY < 0 ? -startingY : 0;

	int64_T biggestY = y + size;
	lostRightCols = biggestY > (imageCols - 1)? biggestY - imageCols + 1 : 0;
}


__host__ __device__  void calcWindowIndexes3dfast_CS(uint32_T* windowIndexes, uint8_T &windowRange, uint32_T rows, uint32_T cols, uint32_T cuts, uint32_T windowSize , uint32_T range, uint32_T index, uint32_T cut ) {
	int r = 2 * windowSize + 1;
	int c = 2 * windowSize + 1;
	uint32_T y = index%cols;
	uint32_T x = index/cols;
	uint32_T lostUpperRows, lostLowerRows, lostLeftCols, lostRightCols;
	calculateLostRows(rows, x, windowSize, lostUpperRows, lostLowerRows);
	calculateLostCols(cols, y, windowSize, lostLeftCols, lostRightCols);

	r = r - lostUpperRows - lostLowerRows;
	c = c - lostLeftCols - lostRightCols;
	uint32_T startingRow = x + lostUpperRows - windowSize;
	uint32_T startingCol = y + lostLeftCols - windowSize;
	uint32_T offset = cut*cols*rows;
	uint32_T pos = 0;

	int k = 0;
	for(uint32_T i = startingRow; i < startingRow+range; i++)
		for(uint32_T j = startingCol; j < startingCol+range; j++) {
			if(i<startingRow+r && j<startingCol+c){
				pos = cols*i+j+offset;
				if(pos != index+offset-cols-1 && pos != index+offset-cols+1 && pos != index+offset+cols-1 && pos != index+offset+cols+1 ){
					windowIndexes[k]=pos;
					k+=1;
				}
			}
		}
	if(cut>0){
		for(uint32_T i = startingRow; i < startingRow+range; i++)
			for(uint32_T j = startingCol; j < startingCol+range; j++) {
				if(i<startingRow+r && j<startingCol+c){
					pos = cols*i+j+offset-cols*rows;
					if(pos != index+offset-cols-cols*rows && pos != index+offset+cols-cols*rows && pos != index+offset-1-cols*rows && pos != index+offset+1-cols*rows ){
						windowIndexes[k]=pos;
						k+=1;
					}
				}
			}
	}
	if(cut<cuts-1){
		for(uint32_T i = startingRow; i < startingRow+range; i++)
			for(uint32_T j = startingCol; j < startingCol+range; j++) {
				if(i<startingRow+r && j<startingCol+c){
					pos = cols*i+j+offset+cols*rows;
					if(pos != index+offset-cols+cols*rows && pos != index+offset+cols+cols*rows && pos != index+offset-1+cols*rows && pos != index+offset+1+cols*rows ){
						windowIndexes[k]=pos;
						k+=1;
					}
				}
			}
	}
	windowRange = k;
}

