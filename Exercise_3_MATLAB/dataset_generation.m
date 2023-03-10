%@Dustin Hanusch 
%% Generation of dataset.
% pseudo random mode combination 
clear all
close all
%%  set parameters 
number_of_modes = 5;    %option: 3 or 5
number_of_data = 10000;
image_size = 32;    % resolution 32x32
 
%% generation of complex mode weights and label vector - step 1








for n=1 : number_of_data

    rho = rand(1,number_of_modes);                              % 1. create random amplitude weights. The weights of amplitude should be normalized.
    rho_n = rho/norm(rho);

    phase = rand(1,number_of_modes-1)*2*pi;                      % 2. create random phase amplitude. (Using realtive phase difference)
    phase = phase-pi;
    phaes_rel = [0 phase];
    phase_norm = cos(phaes_rel);    
    phase_scale = rescale(phase_norm,InputMin=-1,InputMax=1);   % 4. normalize cos(phase) to (0,1)
    k_i = rho_n.*exp(1i*phaes_rel);                             % 3. complex mode weights vector
                                                            
                                                            

    label = [rho_n phase_scale(:,2:size(phase_scale,2))];
    
    label_vec(:,n) = label; 
    complexMode_vec(:,n) = k_i; 
    
end
label_vec = label_vec';
complexMode_vec = complexMode_vec';

% 6. split complex mode weights vector and label vector into Training,
%dividerand
% validation and test set. 

[trainInd,testInd,validInd] = dividerand(number_of_data,0.7,0.1,0.2);
YTest = label_vec(testInd,:);
cplM_Test = complexMode_vec(testInd,:);
YValid = label_vec(validInd,:);
cplM_Valid = complexMode_vec(validInd,:);
YTrain = label_vec(trainInd,:);
cplM_Train = complexMode_vec(trainInd,:);


%% create image data - step 2
% use function mmf_build_image()


XTest = mmf_build_image(number_of_modes,image_size,size(cplM_Test,1),cplM_Test);
XValid = mmf_build_image(number_of_modes,image_size,size(cplM_Valid,1),cplM_Valid);
XTrain = mmf_build_image(number_of_modes,image_size,size(cplM_Train,1),cplM_Train);

figure
k=0;
for i=1:30
    k = k+1;
    subplot(10,3,k), imshow(XTest(:,:,1,i),[0 1])
    

end
%% save dataset

save('mmf_Traingsdata_5modes.mat',"XTrain","YTrain","XValid","YValid",'XTest','YTest');
