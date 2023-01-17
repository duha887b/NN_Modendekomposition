%% Generation of dataset.
% pseudo random mode combination 

%%  set parameters 
number_of_modes = 3;    %option: 3 or 5
number_of_data = 10000;
image_size = 32;    % resolution 32x32
 
%% generation of complex mode weights and label vector - step 1

% 1. create random amplitude weights. The weights of amplitude should be normalized.

% 2. create random phase amplitude. (Using realtive phase difference)

% 3. complex mode weights vector

% 4. normalize cos(phase) to (0,1)

% 5. combine amplitude and phase into a label vector (1,2N-1)

% 6. split complex mode weights vector and label vector into Training,
% validation and test set. 

%% create image data - step 2
% use function mmf_build_image()


%% save dataset


