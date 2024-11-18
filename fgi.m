function [V_FILTERED_CPU,V_IMPULSE_CPU] = fgi(V,r,q,i,integrated)
% Volumetric FGI filter with 2D windows
%   f_vfgi(V,r,q,i,[integrated])
%   V - doubles : CT volume to filter in linear attenuation coefficients
%   r, q - integers : filter parameters
%   i - integer : number of iterations
%   integrated (Optional) - boolean : avoid GPU clear. Do NOT set TRUE unless the GPU
%                          is cleared elsewhere.

if max(V,[],"all") > 4
    disp("The volume must be in linear attenuation coefficients")
    V_FILTERED_CPU = V;
    return
end

 if ~exist('integrated','var')
     % Default false
      integrated = 0;
 end

gpu = gpuDevice;
k_imp = parallel.gpu.CUDAKernel("fgi_kernels.ptx","FGI/fgi_kernels.cu","computeImpulsivity");
k_fuzzy = parallel.gpu.CUDAKernel("fgi_kernels.ptx","FGI/fgi_kernels.cu","fuzzyFilter");


[dim1,dim2,cuts] = size(V);
imageSize =dim1*dim2;
volumeSize =imageSize*cuts;
V_FILTERED = gpuArray(V);
V_IMPULSE = zeros(size(V),"gpuArray");
windowSize = 1;
range = 2 * windowSize + 1;

BLOCKSIZE = single(16);
blocks_index = ceil( single(imageSize) / BLOCKSIZE );
blocks_depth = ceil( single(cuts) / BLOCKSIZE );
dimGrid = [blocks_index blocks_depth 1];
k_imp.ThreadBlockSize = [BLOCKSIZE BLOCKSIZE];
k_fuzzy.ThreadBlockSize = [BLOCKSIZE BLOCKSIZE];
k_imp.GridSize = dimGrid;
k_fuzzy.GridSize = dimGrid;

for iter=1:i
    V_IMPULSE = feval(k_imp,V_FILTERED,V_IMPULSE,dim2,dim1,cuts,r,windowSize);
    V_FILTERED = feval(k_fuzzy,V_FILTERED,windowSize,V_IMPULSE,dim2,dim1,cuts,q);
end

V_FILTERED_CPU = gather(V_FILTERED);
V_IMPULSE_CPU = gather(V_IMPULSE);

%clear gpu objects
if ~integrated
    reset(gpu)
end

end

