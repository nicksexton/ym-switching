# not a function file
1;


function task_activation = calc_activation (input, noise_mean, noise_sd, c)
  % Implementation of Yeung & Monsell Equations 1 & 2, calculates an activation function 

  task_activation = 1 - exp(-c * (input + (randn(1) * noise_sd) + noise_mean))
end

  

function generation_rate = calc_generation_rate (activation)
  % activation is a column vector

  generation_rate = zeros(rows(activation),1);
  total_activation = sum(activation);

  for i = 1:rows(activation)
	    generation_rate(i,1) = activation(i,1) / total_activation;
  end
end


function generation_time = calc_generation_time (activation, threshold)
% threshold is parameter

  generation_time = zeros (rows(activation),1);
  generation_rate = calc_generation_rate (activation);

  for i = 1:rows(generation_rate)
	    generation_time(i,1) = threshold/generation_rate(i,1);
  end
end

function f = calc_f (x, gradient)
    % gradient needs to be fitted? for congruent stimuli, =0, for incongruent =0.5
    if (x > 0)
      f = x * gradient
    else
      f = 0
    endif
end


function resolution_time = calc_resolution_time (generation_time, f_gradient, gauss_mean, gauss_sd, exp_lambda)
% column vector generation time

    %generate random number from ex-gaussian distribution 
    r = ((randn(1) * gauss_sd) + gauss_mean) + exprnd(exp_lambda)
    
	% gradient needs to be fitted? for congruent stimuli, =0, for incongruent =0.5
    resolution_time = r + calc_f (r - generation_time(1,1) - generation_time(2,1), f_gradient)
end



function rt = calc_rt (activation, f_gradient, gauss_mean, gauss_sd, exp_lambda, threshold, constant)

   generation_time = calc_generation_time (activation, threshold)
   rt =  constant + generation_time + ...
   calc_resolution_time (generation_time, f_gradient, gauss_mean, gauss_sd, exp_lambda)

end


% task parameters

    INPUT_C = 1.5
    NOISE_MEAN = 0.0
    NOISE_SD = 0.1
    THRESHOLD = 100
    F_GRADIENT = 1
    EXG_GAUSS_MEAN = 140
    EXG_GAUSS_SD = 10
    EXG_EXP_LAMBDA = 40
    RT_CONST = 150

    taskstrength         = [0.1; 0.5] % colour, word

    control = [0.00, 0.00, 0.97, 0.38;
	       0.20, 0.15, 0.00, 0.00]

    priming = [0.3, 0.0, 0.0, 0.3; 
	       0.0, 0.3, 0.3, 0.0 ] 

    % columns = word switch, word nonswitch, colour switch, colour nonswitch





%%%%%%%%%
% Calculate total input for simulation
    input = zeros (2, 4);
    act = zeros (2, 4);
    rt = zeros (2, 4);
    for i = 1:columns(input)
	      input(:,i) = taskstrength(:,1) + control(:,i) + priming(:,i);
act(:,i) = calc_activation (input(:,i), NOISE_MEAN, NOISE_SD, INPUT_C);
rt(:,i) = calc_rt (act(:,i), F_GRADIENT, EXG_GAUSS_MEAN, EXG_GAUSS_SD, ...
		   EXG_EXP_LAMBDA, THRESHOLD, RT_CONST);

end


% Calculate RTs
% nb need to train to obtain value for F_GRADIENT

input
act
rt

