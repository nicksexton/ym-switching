# not a function file
1;


function task_activation = calc_activation (input, noise_mean, noise_sd, c)
  % Implementation of Yeung & Monsell Equations 1 & 2, calculates an activation function 

  # task_activation = 1 - exp(-1 * c * (input + (randn(1) * noise_sd) + noise_mean));
				# Undocumented assumption 1: Activations are lower-bounded at 0
#  task_activation = max (1 - exp(-1 * c * (input + (randn(1) * noise_sd) + noise_mean)), .0001);
  input_noise = randn(2,4) * noise_sd + noise_mean;
  task_activation = max (1 - exp(-1 * c * (input + input_noise)), .0001);
end

  

function generation_rate = calc_generation_rate (activation)
  total_activation = sum(activation); # is this summing the column (correct) or a 2x4 matrix? (incorrect)
				     # checked the calculations and it sums by column
  generation_rate = activation ./ repmat(total_activation, 2, 1);
end


function generation_time = calc_generation_time (activation, threshold)
  generation_rate = calc_generation_rate (activation);
  generation_time = repmat(threshold, 2, 4)./generation_rate;
end


function f = calc_f (r_minus_gen_time_diff, gradient)
    % gradient needs to be fitted? for congruent stimuli, =0, for incongruent =0.5
  f = max ([0, r_minus_gen_time_diff * gradient]);
    # if (r_minus_gen_time_diff > 0)
    #   f = r_minus_gen_time_diff * gradient;
    # else
    #   f = 0;
    # endif
end


function resolution_time = calc_resolution_time (generation_time, f_gradient, gauss_mean, gauss_sd, 
						 exp_lambda, irrelevant_stimulus_onset)

  generation_time_plus_onset = generation_time + irrelevant_stimulus_onset;
  generation_time_difference = [generation_time_plus_onset(2,:) - generation_time_plus_onset(1,:);
   				generation_time_plus_onset(1,:) - generation_time_plus_onset(2,:)];

				# Where is r recalculated?? 
				# insert r here if it is drawn once for all trials
				# at present, model does not work for delayed onset if r calc for all trials
#  r = ((randn(1) * gauss_sd) + gauss_mean) + exprnd(exp_lambda) # Calculating for each trial
      resolution_time = zeros(2, 4);
      for j = 1:columns(resolution_time)
	for i = 1:rows(resolution_time)

	                        # Insert here if it is drawn individually for both tasks  
	  r = ((randn(1) * gauss_sd) + gauss_mean) + exprnd(exp_lambda); # Calculating for each trial
	  resolution_time(i,j) = r + calc_f (r - generation_time_difference(i,j), f_gradient);
	end
      end
      resolution_time;
end



function rt = calc_rt (activation, f_gradient, gauss_mean, gauss_sd, 
		       exp_lambda, threshold, constant, irrelevant_stimulus_onset, response_gating)

  generation_time = calc_generation_time (activation, threshold);
  resolution_time = calc_resolution_time (generation_time, f_gradient, gauss_mean, gauss_sd, 
					  exp_lambda, irrelevant_stimulus_onset);

  rt =  constant + generation_time + resolution_time;
  

end




function rt = run_trial ( params )

  input = zeros (2, 4);
  act = zeros (2, 4);
  rt = zeros (2, 4);

#  printf ("Beginning trial!\n");
    input = (repmat(params.TASKSTRENGTH, 1, 4) + params.PRIMING) * params.UNCONTROLLED_SCALING + params.CONTROL;
    act = calc_activation (input, params.NOISE_MEAN, params.NOISE_SD, params.INPUT_C);
    rt = calc_rt (act, params.F_GRADIENT, params.EXG_GAUSS_MEAN, params.EXG_GAUSS_SD, ...
		  params.EXG_EXP_LAMBDA, params.THRESHOLD, params.RT_CONST, params.IRRELEVANT_STIM_ONSET, 
		  params.RESPONSE_GATING);


end


function rt_row = make_rt_row_vector (rt) 
  rt_row = [rt(2,1), rt(2,2), rt(1,3), rt(1,4)]; % W-NS, W-S, C-NS, C-S
end


function correct = make_correct_rt_row_vector (rt)
  correct = [rt(2,1), rt(2,2), rt(1,3), rt(1,4)]; % W-NS, W-S, C-NS, C-S
  alternative = [rt(1,1), rt(1,2), rt(2,3), rt(2,4)];

  for i = 1:4
    if correct(i) > alternative(i)
      correct(i) = NaN;
    endif
  end
end

function plotdata = make_correct_plot_data (correct)
  plotdata = [correct(1,2), correct(1,4); correct(1,1), correct(1,3)];
				% [ W-S,  C-S  ]
				% [ W-NS, C-NS ]
end


function [new_control_strengths] =  train_model (params, iterations, 
				   n_increment, step_increment, 
				   n_decrement, step_decrement)
# Trains the model for given number of iterations, to achieve the desired error rate
# For n incorrect/correct responses, it increments/decrements the control setting by step size. 

# first make a step size mask for increment and decrement

  params_copy = params;
  
  mask_increment = [0, 0, step_increment, step_increment; step_increment, step_increment, 0, 0 ]; 
  mask_decrement = [0, 0, step_decrement, step_decrement; step_decrement, step_decrement, 0, 0 ];
	
  count_correct = zeros(1, 4);
  count_error = zeros(1,4);
  new_control_strengths = zeros(iterations, 4);

  for i = 1:iterations
    rt_row = make_correct_rt_row_vector (run_trial (params_copy));
    for task = 1:columns(rt_row)
      
      if isnan (rt_row(task))
	count_error(task) ++;
      elseif
	count_correct(task) ++;
      endif

      if count_error(task) == n_increment;
	params_copy.CONTROL(:,task) = params_copy.CONTROL(:,task) + mask_increment(:,task);
	params_copy.CONTROL;
	count_error(task) = 0;
      elseif count_correct(task) == n_decrement;
	params_copy.CONTROL(:,task) = params_copy.CONTROL(:,task) + mask_decrement(:,task);
	params_copy.CONTROL;
	count_correct(task) = 0;
      endif
    end
    new_control_strengths(i,:) = [params_copy.CONTROL(2,1), params_copy.CONTROL(2,2), ...
				  params_copy.CONTROL(1,3), params_copy.CONTROL(1,4)];  
  end

  plot (new_control_strengths)
end



function block = run_block (n, params)
  block = zeros(n, 4);
  for i = 1:n
				%		block(i,:) = make_rt_row_vector (run_trial (taskstrength,
    block(i,:) = make_correct_rt_row_vector (run_trial (params));
  end
end


function return_bit = classify_switchcost (pvalue, difference_in_means)
  if (pvalue < .05) # is p < .05?
    if (difference_in_means > 0) 
      return_bit = 2; # switch cost is positive and p < .05
    else
      return_bit = -2; # switch cost is negative and p < .05
    endif
  else # test is not significant
    if (difference_in_means > 0) 
      return_bit = 1; # if switch cost is positive but p > .05
    else
      return_bit = -1; # switch cost is negative but p > .05
    endif
  endif

  
end

function pattern = classify_behaviour (block)
  errors = sum(isnan(block))/rows(block);
  mean_rts = nanmean(block); % (exclude NaNs)
  std_rts = nanstd(block);

  word_sc = zeros (1,4); % store results of t-test for word switch cost
  word_sc(1) = mean_rts(1) - mean_rts(2);
  [word_sc(2), word_sc(3), word_sc(4)] = ...
      t_test_2 (block(isfinite(block(:,1)),1), block(isfinite(block(:,2)),2));

  # t_test_2 returns [pval, t, df]. isfinite construction used to filter NaN values
  # if (word_sc(2) < .05) # is p < .05?
  #   if (word_sc(1) > 0) 
  #     word_sc_bit = 2; # switch cost is positive and p < .05
  #   else
  #     word_sc_bit = -2; # switch cost is negative and p < .05
  #   endif
  # else # test is not significant
  #   if (word_sc(1) > 0) 
  #     word_sc_bit = 1; # if switch cost is positive but p > .05
  #   else
  #     word_sc_bit = -1; # switch cost is negative but p > .05
  #   endif
  # endif


  colour_sc = zeros (1,4); % store results of t-test for word switch cost
  colour_sc(1) = mean_rts(3) - mean_rts(4);
  [colour_sc(2), colour_sc(3), colour_sc(4)] = ...
      t_test_2 (block(isfinite(block(:,3)),3), block(isfinite(block(:,4)),4));


  pattern = struct ('Error_w_S', errors(1) < .1, # Error rate less than 10%
		    'Error_w_NS', errors(2) < .1, # 1 if errors < 10%, 0 otherwise
		    'Error_c_S', errors(3) < .1, # 1 if errors < 10%, 0 otherwise
		    'Error_c_NS', errors(4) < .1, # 1 if errors < 10%, 0 otherwise
		    'SC_w', classify_switchcost (word_sc(2), word_sc(1)), # 2 = sig, 1 = n.s., +/- indicates dir
		    'SC_c', classify_switchcost (colour_sc(2), colour_sc(1)), # 2 = sig, 1 = n.s., +/- indicates dir
		    'SC_a', word_sc(1) > colour_sc(1)); # 1 = SC asymmetry, 0 = no SC asymmetry,

  

end


function p = plot_single_trial (params)
# Plots results of a single trial
rt = run_trial (params );

correct = make_rt_row_vector (rt);
plotdata = make_correct_plot_data (correct);


plot (plotdata);
xlim ([1, 2]);
xlabel ('Repeat vs. Switch');
ylabel ('Simulated RT (ms)');
legend ('Word Reading', 'Colour Naming');

end

function block = plot_block (params)

n = 600
block = run_block (n, params);
mean_rts = nanmean(block) % (exclude NaNs)
std_rts = nanstd(block)
errors = sum(isnan(block))/n;

plotdata = make_correct_plot_data (mean_rts);
stddata = make_correct_plot_data (std_rts);

close all;

figure(1)
plot (plotdata);
xlim ([0, 3]);
xlabel ('Repeat vs. Switch');
ylabel ('Simulated RT (ms)');
legend ('Word Reading', 'Colour Naming');
%hold on;
%errorbar ([1,2], plotdata(:,1), stddata(:,1) '.');
%hold on;
%errorbar ([1,2], plotdata(:,2), stddata(:,2) '.');

figure(2)
ploterrors = make_correct_plot_data (errors)
plot (ploterrors);
xlim ([0, 3]);
xlabel ('Repeat vs. Switch');
ylabel ('Error Rate (%)');
legend ('Word Reading', 'Colour Naming');

end

function [pattern] = ym_model_wrapper (params) # params must be column vector
# First version of PSP in YM model.
# Explore 4-dimensional parameter space defined by 4 control parameters
# Accepts input [word switch; word nonswitch; colour switch; colour nonswitch]

trials = 600;

model_params  = struct('INPUT_C', 1.5, ...
		'NOISE_MEAN', 0.0, ...
		'NOISE_SD', 0.1, ...
		'THRESHOLD', 100, ...
		'F_GRADIENT', 0.5, ...
		'EXG_GAUSS_MEAN', 140, ...
		'EXG_GAUSS_SD', 10, ...
		'EXG_EXP_LAMBDA', 40, ...
		'RT_CONST', 150, ...
		'TASKSTRENGTH', [0.1; 0.5],
		'CONTROL', [0.00, 0.00, params(3), params(4);
			    params(1), params(2), 0.00, 0.00],
		'PRIMING', [0.3, 0.0, 0.0, 0.3; 
			    0.0, 0.3, 0.3, 0.0 ],
		'UNCONTROLLED_SCALING', 1.0, # scaling for strength + priming (0.6 in in neutral condition)
		'IRRELEVANT_STIM_ONSET', [0, 0, 0, 0;
					  0, 0, 0, 0],
		'RESPONSE_GATING', [1, 1, 1, 1;  # should be 1.0 everywhere to reproduce original paper
				    1, 1, 1 1]);



results = run_block (trials, model_params);
p = classify_behaviour (results); # returns a struct
pattern = [p.Error_w_S, p.Error_w_NS, p.Error_c_S, p.Error_c_NS, p.SC_w, p.SC_c, p.SC_a]; # return p as row vector


end


% =============================================================== Load task parameters =====================

    % task parameters

control_default = [0.00, 0.00, 0.97, 0.38;
		   0.20, 0.15, 0.00, 0.00];

stim_onset = 160
stim_onset_asynchronous = [stim_onset, stim_onset, 0, 0;
			   0, 0, stim_onset, stim_onset]

stim_onset_synchronous = [0, 0, 0, 0;
			  0, 0, 0, 0]


params_default = struct('INPUT_C', 1.5, ...
		'NOISE_MEAN', 0.0, ...
		'NOISE_SD', 0.1, ...
		'THRESHOLD', 100, ...
		'F_GRADIENT', 0.5, ...
		'EXG_GAUSS_MEAN', 140, ...
		'EXG_GAUSS_SD', 10, ...
		'EXG_EXP_LAMBDA', 40, ...
		'RT_CONST', 150, ...
		'TASKSTRENGTH', [0.1; 0.5],
		'CONTROL', control_default,
		'PRIMING', [0.3, 0.0, 0.0, 0.3; 
			    0.0, 0.3, 0.3, 0.0 ],
		'UNCONTROLLED_SCALING', 1.0, # scaling for strength + priming (0.6 in in neutral condition)
		'IRRELEVANT_STIM_ONSET', stim_onset_synchronous,
		'RESPONSE_GATING', [1, 1, 1, 1;  # should be 1.0 everywhere to reproduce original paper
				    1, 1, 1 1])


function params_neutral = make_neutral (params)
		     params_neutral = params
		     params_neutral.F_GRADIENT = 0.0
		     params_neutral.UNCONTROLLED_SCALING = 0.6		     
end

params_untrained = params_default;
params_untrained.CONTROL = [0.00, 0.00, 0.15, 0.15;
			    0.15, 0.15, 0.00, 0.00]


params_delayedonset = params_default;
params_delayedonset.CONTROL = [0.00, 0.00, 0.15, 0.15;
			       0.15, 0.15, 0.00, 0.00]
params_delayedonset.IRRELEVANT_STIM_ONSET = stim_onset_asynchronous; # Not used in the original paper
params_delayedonset

params_delayedonset_1 = params_delayedonset;
params_delayedonset_1.CONTROL = [0.0, 0.0, 0.56, 0.15;
				 0.10, 0.10, 0.0, 0.0]


params_responsegating = params_default;
params_responsegating.F_GRADIENT = 0.0;
#params_responsegating.RESPONSE_GATING = [0, 0, 1, 1; # Not used in the original paper
#					 1, 1, 0 0]
params_responsegating
