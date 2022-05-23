clc;
clear;
close all;

% % ========================== input 1 ==================================
% filename = 'input/input1.bmp' ;
% filename_ori = 'ori/input1_ori.bmp' ;
% in_image = imread(filename) ;
% in_image_ori = imread(filename_ori) ;
% in_blur =  double(in_image) ;
% in_ori  =  double(in_image_ori) ;
% PSNR_b = psnr(uint8(in_blur),uint8(in_ori));
% 
% % ----- Get TA kernel -----
% in_blur_f = fftshift(fft2(in_blur));
% in_ori_f = fftshift(fft2(in_ori));
% H1_TA = in_blur_f ./ in_ori_f;
% H1_TA = real(H1_TA);
% H1_TA1 = H1_TA(:,:,1);
% figure; 
% imshow(H1_TA);
% 
% % % ----- Guess kernel (input1) -----
% PSF1 = fspecial('gaussian', 41, 9.86) ;  % find the higest parameter.
% wnr1 = wiener(in_blur, PSF1, 0.001);
% figure()
% imshow( uint8( wnr1 ) ) ;
% PSNR_r1 = psnr(uint8(wnr1),uint8(in_ori));
% 
% PSF2 = fspecial('gaussian', 25, 2.17) ;  % find the higest parameter.
% wnr2 = CLS(wnr1, PSF2, 0.001);
% figure()
% imshow( uint8( wnr2 ) ) ;
% PSNR_r2 = psnr(uint8(wnr2),uint8(in_ori));
% imwrite(uint8( wnr1 ), 'restored/restored_input1_1.bmp');
% 
% [s1, s2, s3] = size(in_blur);
% H1 = psf2otf(PSF1,[s1 s2]);
% H1 = fftshift(H1);
% figure
% imshow(H1)
% figure;
% mesh(H1);
% 
% H1 = psf2otf(PSF2,[s1 s2]);
% H1 = fftshift(H1);
% figure
% imshow(H1)
% figure;
% mesh(H1);
% 
% PSF = conv2(PSF1, PSF2);
% H1 = psf2otf(PSF,[s1 s2]);
% H1 = fftshift(H1);
% figure
% imshow(H1)
% figure;
% mesh(H1);
% % % ---------- Fine-tune parameter ------------
% % for i = 3
% %     i
% %     for j = 1:10
% %         j
% %         hsize = 19+2*i;
% %         sigma = 2.1+0.01*j; 
% %         PSF2 = fspecial('gaussian', hsize, sigma);
% %         % H = psf2otf(PSF,[ s1 s2]);
% %         % H = fftshift(H);
% %         % figure
% %         % imshow(H)
% %         % figure;
% %         % mesh(H);
% %         wnr2 = wiener(wnr1, PSF2, 0.001);
% %     %     figure()
% %     %     imshow( uint8( wnr ) ) ;
% %         PSNR_r(i,j) = psnr(uint8(wnr2),uint8(in_ori));
% %     end
% % end

% % ========================== input 2 ==================================
% filename = 'input/input2.bmp' ;
% filename_ori = 'ori/input2_ori.bmp' ;
% in_image = imread(filename) ;
% in_image_ori = imread(filename_ori) ;
% in_blur =  double(in_image) ;
% in_ori  =  double(in_image_ori) ;
% PSNR_b = psnr(uint8(in_blur),uint8(in_ori));
% 
% % % ----- Get TA kernel -----
% in_blur_f = fftshift(fft2(in_blur));
% in_ori_f = fftshift(fft2(in_ori));
% H2_TA = in_blur_f ./ in_ori_f;
% H2_TA = real(H2_TA);
% H2_TA1 = H2_TA(:,:,1);
% figure; 
% imshow(H2_TA);
% 
% % ----- Guess kernels (input2) -----
% PSF1 = fspecial('motion', 28, 95) ;  
% wnr1 = wiener(in_blur, PSF1, 0.01);
% cls = CLS(in_blur, PSF1, 0.01);
% figure()
% imshow( uint8( wnr1 ) ) ;
% PSNR_r1 = psnr(uint8(wnr1),uint8(in_ori));
% 
% PSF2 = fspecial('motion', 8, 10) ;  
% wnr2 = wiener(wnr1, PSF2, 0.1);
% cls = CLS(wnr1, PSF2, 0.01);
% figure()
% imshow( uint8( wnr2 ) ) ;
% PSNR_r2 = psnr(uint8(wnr2),uint8(in_ori));
% 
% [s1, s2, s3] = size(in_blur);
% PSF = conv2(PSF1, PSF2);
% H2 = psf2otf(PSF,[s1 s2]);
% H2 = fftshift(H2);
% figure
% imshow(H2)
% figure;
% mesh(H2);
% 
% % Find the best parameter
% % for l = 1:5
% %     for th = 1:36
% %         len = l;
% %         theta = 10*th;
% %         PSF1 = fspecial('motion', len, theta) ;
% % %         H1 = psf2otf(PSF1,[ s1 s2]);
% % %         H1 = fftshift(H1);
% %         % figure;
% %         % imshow(H1);
% %         % figure;
% %         % mesh(H1);
% %         
% % %         PSF2 = fspecial('gaussian', 5, 4) ;
% % %         H2 = psf2otf(PSF2,[ s1 s2]);
% % %         H2 = fftshift(H2);
% % %         H = H1 .* H2;
% %         % figure;
% %         % imshow(H);
% %         % figure;
% %         % mesh(H);
% %         
% % %         PSF = conv2(PSF1, PSF2);
% %         wnr1 = wiener(in_blur, PSF1, 0.001);
% %         
% % %         figure()
% % %         imshow( uint8( wnr1 ) ) ;
% %         PSNR_r(l, th) = psnr(uint8(wnr1),uint8(in_ori));
% %     end
% % end
% ========================================================================

% % ========================== input 3 =================================
filename = 'input/input3.bmp' ;
filename_ori = 'ori/input3_ori.bmp' ;
in_image = imread(filename) ;
in_image_ori = imread(filename_ori) ;
in_blur =  double(in_image) ;
in_ori  =  double(in_image_ori) ;
PSNR_b = psnr(uint8(in_blur),uint8(in_ori));

% % ----- Get TA kernel -----
in_blur_f = fftshift(fft2(in_blur));
in_ori_f = fftshift(fft2(in_ori));
H2_TA = in_blur_f ./ in_ori_f;
H2_TA = real(H2_TA);
H2_TA1 = H2_TA(:,:,1);
figure; 
imshow(H2_TA);

% ----- Guess kernels (input2) -----
PSF = fspecial('gaussian', 31, 4.5) ;  
wnr1 = wiener(in_blur, PSF, 0.1);
cls = CLS(in_blur, PSF, 0.1);
figure()
imshow( uint8( wnr1 ) ) ;
PSNR_r1 = psnr(uint8(wnr1),uint8(in_ori));

[s1, s2, s3] = size(in_blur);
H3 = psf2otf(PSF,[s1 s2]);
H3 = fftshift(H3);
figure
imshow(H3)
% % ========================================================================

% ------ Delete fft pick ------
function H_out = delPick(H_in)
H_out = H_in;
for i = 1:s1
    for j = 1:s2
        if H_in(i,j) > 1
            H_out(i,j) = 0;
        elseif H_in(i,j) < -1
            H_out(i,j) = 0;
        end
    end
end
end

% ----- CLS -----
function cls = CLS(in_blur, PSF, gamma)
[s1, s2, s3] = size(in_blur);
in_blur_pad(:,:,1) = padarray( in_blur(:,:,1), [s1, s2], 'symmetric' );
in_blur_pad(:,:,2) = padarray( in_blur(:,:,2), [s1, s2], 'symmetric' );
in_blur_pad(:,:,3) = padarray( in_blur(:,:,3), [s1, s2], 'symmetric' );
[height width channel] =  size(in_blur_pad);

H = psf2otf(PSF,[ height width]);
G(:,:,1) = fft2( in_blur_pad(:,:,1) ) ;
G(:,:,2) = fft2( in_blur_pad(:,:,2) ) ;
G(:,:,3) = fft2( in_blur_pad(:,:,3) ) ;

p = [  0, -1,  0;
      -1 , 4, -1;
       0, -1,  0;];
p_f = psf2otf(p,[ height width]);
F(:,:, 1) = ( conj(H) ./ ( abs(H.*conj(H)) +  gamma*abs(p_f.*conj(p_f))) ) .* G(:,:,1);
F(:,:, 2) = ( conj(H) ./ ( abs(H.*conj(H)) +  gamma*abs(p_f.*conj(p_f))) ) .* G(:,:,2);
F(:,:, 3) = ( conj(H) ./ ( abs(H.*conj(H)) +  gamma*abs(p_f.*conj(p_f))) ) .* G(:,:,3);

R(:,:,1) = G(:,:,1) - H .* F(:,:,1);
R(:,:,2) = G(:,:,2) - H .* F(:,:,2);
R(:,:,3) = G(:,:,3) - H .* F(:,:,3);
r2(1) = sum(R(:,:,1).^2,'all');
r2(2) = sum(R(:,:,2).^2,'all');
r2(3) = sum(R(:,:,3).^2,'all');

im_n=abs(ifft2(F)); %iFFT
cls  = im_n(s1+1:end-s1, s2+1:end-s2, :) ; 
end

function wnr = wiener(in_blur, PSF, SNR)
[s1, s2, s3] = size(in_blur);
in_blur_pad(:,:,1) = padarray( in_blur(:,:,1), [s1, s2], 'symmetric' );
in_blur_pad(:,:,2) = padarray( in_blur(:,:,2), [s1, s2], 'symmetric' );
in_blur_pad(:,:,3) = padarray( in_blur(:,:,3), [s1, s2], 'symmetric' );
[height width channel] =  size(in_blur_pad);

H = psf2otf(PSF,[ height width]);
G(:,:,1) = fft2( in_blur_pad(:,:,1) ) ;
G(:,:,2) = fft2( in_blur_pad(:,:,2) ) ;
G(:,:,3) = fft2( in_blur_pad(:,:,3) ) ;

% ---- Weiner deblur -----
F(:,:, 1) = ( conj(H) ./ ( abs(H.*conj(H)) + SNR ) ) .* G(:,:,1);
F(:,:, 2) = ( conj(H) ./ ( abs(H.*conj(H)) + SNR ) ) .* G(:,:,2);
F(:,:, 3) = ( conj(H) ./ ( abs(H.*conj(H)) + SNR ) ) .* G(:,:,3);

R(:,:,1) = G(:,:,1) - H .* F(:,:,1);
R(:,:,2) = G(:,:,2) - H .* F(:,:,2);
R(:,:,3) = G(:,:,3) - H .* F(:,:,3);
r2(1) = sum(R(:,:,1).^2,'all');
r2(2) = sum(R(:,:,2).^2,'all');
r2(3) = sum(R(:,:,3).^2,'all');

im_n=abs(ifft2(F)); %iFFT
wnr  = im_n(s1+1:end-s1, s2+1:end-s2, :) ; 
end

function [bl, psfr] = blind(in_blur, PSF)
[s1, s2, s3] = size(in_blur);
in_blur_pad(:,:,1) = padarray( in_blur(:,:,1), [s1, s2], 'symmetric' );
in_blur_pad(:,:,2) = padarray( in_blur(:,:,2), [s1, s2], 'symmetric' );
in_blur_pad(:,:,3) = padarray( in_blur(:,:,3), [s1, s2], 'symmetric' );
figure();
imshow(uint8(in_blur_pad));
[height, width, channel] =  size(in_blur_pad);

PSF = psf2otf(PSF,[height width]);
[bl, psfr] = deconvblind(in_blur_pad, PSF, 10, 5);
bl  = bl(s1+1:end-s1, s2+1:end-s2, :);
end