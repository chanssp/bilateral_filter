A = imread('univ_studio.jpg');
A = im2double(A);
[x,y,channel] = size(A);

%%%%%%%% parameters %%%%%%%%
k = 4;  % kernel : size = 2k+1
sigma_s = k+0.5;
sigma_r = 0.4;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% calculate Euclidean distance from pixel
[X,Y] = meshgrid(-k:k, -k:k);
G = exp(-(X.^2 + Y.^2)/(sigma_s^2));    % Gaussian convolution filter

R = zeros(x,y);
R = cat(3,R,R,R);

for i=1:x
    for j=1:y
        % extracting local kernel to apply Gaussian filter
        iMin = max(i-k,1);
        iMax = min(i+k,x);
        jMin = max(j-k,1);
        jMax = min(j+k,y);
        M = A(iMin:iMax,jMin:jMax,:);
        
        % compute Gaussian filter : range
        dR = M(:,:,1) - A(i,j,1);
        dG = M(:,:,2) - A(i,j,2);
        dB = M(:,:,3) - A(i,j,3);
        D = cat(3,dR,dG,dB);    % |Ip - Iq|
        G_range = exp(-(sum(D(:,:,:).^2,3))/(sigma_r^2));    % G(|Ip - Iq|)
        M_range = cat(3,G_range,G_range,G_range);
        
        % compute Gaussian filter : space
        G_space = G((k+1)-(i-iMin):(k+1)+(iMax-i),(k+1)-(j-jMin):(k+1)+(jMax-j));   % G(|p - q|)
        M_space = cat(3,G_space,G_space,G_space);
        
        M_final = M_space.* M_range.* M;      % C = G(|p - q|) * G(|Ip - Iq|) * Iq
        C = [sum(sum(M_final(:,:,1)));sum(sum(M_final(:,:,2)));sum(sum(M_final(:,:,3)))];   % Sigma (C)
        norm = sum(sum(G_range(:,:).* G_space(:,:)));      % Sigma (G(|p - q|) * G(|Ip - Iq|))
        
        R(i,j,:) = C/norm;
        
    end
end

%imshow(R);
imwrite(R, 'univ_bilateral.jpg');