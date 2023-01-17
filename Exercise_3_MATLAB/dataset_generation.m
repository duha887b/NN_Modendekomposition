%% Generation of dataset.
% pseudo random mode combination 

%%  set parameters 
number_of_modes = 3;    %option: 3 or 5
number_of_data = 10000;
image_size = 32;    % resolution 32x32
 
%% generation of complex mode weights and label vector - step 1








for n=1 : number_of_data

    rho = rand(1,number_of_modes);                              % 1. create random amplitude weights. The weights of amplitude should be normalized.
    rho_n = rho/norm(rho);

    phase = rand(1,number_of_modes)*2*pi;                       % 2. create random phase amplitude. (Using realtive phase difference)
    phaes_rel = phase - phase(1);
    phase_norm = cos(phaes_rel);    
    phase_scale = rescale(phase_norm,InputMin=-1,InputMax=1);   % 4. normalize cos(phase) to (0,1)
    k_i = rho_n.*exp(1i*phase_scale);                           % 3. complex mode weights vector
                                                            
                                                            

    label = [rho_n phase_scale(:,2:size(phase_scale,2))];
    
    lebel_vec(:,n) = label; 
    complexMode_vec(:,n) = k_i; 
end




%dividerand
% 6. split complex mode weights vector and label vector into Training,
% validation and test set. 

%% create image data - step 2
% use function mmf_build_image()


%% save dataset


