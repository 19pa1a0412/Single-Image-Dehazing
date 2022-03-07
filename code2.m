clc;
clear all;
close all;
image='C:\Users\adire\Downloads\c1.JPEG';
inp = imread(image);
%figure;
%imshow(im);
%title('input image');
%normalize
inp=imresize(inp,[500 500]);
inp = double(inp);
inp = inp./255;
figure;
imshow(inp);
title('input image');
[j,k,l] = size(inp);

%Specify the new image dimensions we want for our smaller output image
%In this case we will downsample the image by a fixed ratio
%Since the ratios are different, the image will appear distored
%We can also set x_new and y_new to arbitrary values, but it will not work
%if they are larger than j and k. That would be upsampling/interpolation,
%and will be covered in a future tutorial.
x_new = j/4;
y_new = k/4;
%x_new=uint8(x_new);
%y_new=uint8(y_new);
%Determine the ratio of the old dimensions compared to the new dimensions
x_scale = j/x_new;
y_scale = k/y_new;

%Declare and initialize an output image buffer
im = zeros(x_new,y_new,l);

%Generate the output image
for p=1:l
for count1 = 1:x_new
 for count2 = 1:y_new
 im(count1,count2,p) = inp(count1*x_scale,count2*y_scale,p);
 end
end
end

figure;
imshow(im);

%function JDark = darkChannel(im1)
[height, width, ~] = size(im);
patchSize = 15; %the patch size is set to be 15 x 15
padSize = 7; % half the patch size to pad the image with for the array to 
%work (be centered at 1,1 as well as at height,1 and 1,width and height,width etc)
JDark = zeros(height, width); % the dark channel
%figure;
%imshow(JDark);
imJ = padarray(im, [padSize padSize], Inf); % the new image
% imagesc(imJ); colormap gray; axis off image
for j = 1:height
    for i = 1:width
        % the patch has top left corner at (jj, ii)
        patch = imJ(j:(j+patchSize-1), i:(i+patchSize-1),:);
        % the dark channel for a patch is the minimum value for all
        % channels for that patch
        JDark(j,i) = min(patch(:));
     end
end
figure;
imshow(JDark);
title('Dark pixels');
%function A = atmLight(im, JDark)
% the color of the atmospheric light is very close to the color of the sky
% so just pick the first few pixels that are closest to 1 in JDark
% and take the average
% pick top 0.1% brightest pixels in the dark channel
% get the image size
[height, width, ~] = size(im);
imsize = width * height;
numpx = floor(imsize/1000); % accomodate for small images
JDarkVec = reshape(JDark,imsize,1); % a vector of pixels in JDark
ImVec = reshape(im,imsize,3);  % a vector of pixels in my image
[JDarkVec, indices] = sort(JDarkVec); %sort
indices = indices(imsize-numpx+1:end); % need the last few pixels because those are closest to 1
atmSum = zeros(1,3);
atmSum=double(atmSum);
ImVec=double(ImVec);
indices=double(indices);
for ind = 1:numpx
    atmSum = atmSum + ImVec(indices(ind),:);
end
A = atmSum / numpx;
disp(A);
%function transmission = transmissionEstimate(im, A)
omega = 0.45; % the amount of haze we're keeping
im3 = zeros(size(im));
for ind = 1:3 
    im3(:,:,ind) = im(:,:,ind)./A(ind);
end
% imagesc(im3./(max(max(max(im3))))); colormap gray; axis off image
[height, width, ~] = size(im3);
patchSize = 15; %the patch size is set to be 15 x 15
padSize = 7; % half the patch size to pad the image with for the array to 
%work (be centered at 1,1 as well as at height,1 and 1,width and height,width etc)
JDark = zeros(height, width); % the dark channel
%figure;
%imshow(JDark);
imJ = padarray(im3, [padSize padSize], Inf); % the new image
% imagesc(imJ); colormap gray; axis off image
for j = 1:height
    for i = 1:width
        % the patch has top left corner at (jj, ii)
        patch = imJ(j:(j+patchSize-1), i:(i+patchSize-1),:);
        % the dark channel for a patch is the minimum value for all
        % channels for that patch
        JDark(j,i) = min(patch(:));
     end
end
%figure;
%imshow(JDark);
transmission = 1-omega*JDark;
figure;
imshow(transmission);
title('transmission');
I=transmission;
sigma=4;
%function I=imgaussian(I,sigma,siz)
% IMGAUSSIAN filters an 1D, 2D color/greyscale or 3D image with an 
% Gaussian filter. This function uses for filtering IMFILTER or if 
% compiled the fast  mex code imgaussian.c . Instead of using a 
% multidimensional gaussian kernel, it uses the fact that a Gaussian 
% filter can be separated in 1D gaussian kernels.
%
% J=IMGAUSSIAN(I,SIGMA,SIZE)
%
% inputs,
%   I: The 1D, 2D greyscale/color, or 3D input image with 
%           data type Single or Double
%   SIGMA: The sigma used for the Gaussian kernel
%   SIZE: Kernel size (single value) (default: sigma*6)
% 
% outputs,
%   J: The gaussian filtered image
%
% note, compile the code with: mex imgaussian.c -v
%
% example,
%   I = im2double(imread('peppers.png'));
%   figure, imshow(imgaussian(I,10));
% 
% Function is written by D.Kroon University of Twente (September 2009)
if(~exist('siz','var')), siz=sigma*6; end
if(sigma>0)
    % Make 1D Gaussian kernel
    x=-ceil(siz/2):ceil(siz/2);
    H = exp(-(x.^2/(2*sigma^2)));
    H = H/sum(H(:));
    % Filter each dimension with the 1D Gaussian kernels\
    if(ndims(I)==1)
        I=imfilter(I,H, 'same' ,'replicate');
    elseif(ndims(I)==2)
        Hx=reshape(H,[length(H) 1]);
        Hy=reshape(H,[1 length(H)]);
        I=imfilter(imfilter(I,Hx, 'same' ,'replicate'),Hy, 'same' ,'replicate');
    elseif(ndims(I)==3)
        if(size(I,3)<4) % Detect if 3D or color image
            Hx=reshape(H,[length(H) 1]);
            Hy=reshape(H,[1 length(H)]);
            for k=1:size(I,3)
                I(:,:,k)=imfilter(imfilter(I(:,:,k),Hx, 'same' ,'replicate'),Hy, 'same' ,'replicate');
            end
        else
            Hx=reshape(H,[length(H) 1 1]);
            Hy=reshape(H,[1 length(H) 1]);
            Hz=reshape(H,[1 1 length(H)]);
            I=imfilter(imfilter(imfilter(I,Hx, 'same' ,'replicate'),Hy, 'same' ,'replicate'),Hz, 'same' ,'replicate');
        end
    else
        error('imgaussian:input','unsupported input dimension');
    end
end
figure;
imshow(I);
title('Filtered Image');

%function J = getRadiance(A,im,tMat)
tMat=I;
t0 = 0.1;
J = zeros(size(im));
for ind = 1:3
   J(:,:,ind) = A(ind) + (im(:,:,ind) - A(ind))./max(tMat,t0); 
end
J = J./(max(max(max(J))));
figure;
imshow(J);
title('Dehazed image');
% % %psnr
  f=inp;
 F=J;
[im,in]=size(J);
sum=double(0);
for i=1:im
    for j=1:in
        if F(i,j)>f(i,j)
          s=double((F(i,j)-f(i,j)));
        else
            s=double((f(i,j)-F(i,j)));
        end
        p=double(power(s,2));
        sum=sum+p;
    end
end
error=sum/(im*in);
rms=sqrt(error);
psn=20*log10(255/rms)