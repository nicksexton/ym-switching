# not a function file
1;


function task_activation = calc_activation (input, noise_mean, noise_sd, c)
  % Implementation of Yeung & Monsell Equations 1 & 2, calculates an activation function 

  task_activation = 1 - exp(-1 * c * (input + (randn(1) * noise_sd) + noise_mean));
end

  

function generation_rate = calc_generation_rate (activation)

  total_activation = sum(activation);
  generation_rate = activation ./ repmat(total_activation, 2, 1);

end


function generation_time = calc_generation_time (activation, threshold)

  generation_rate = calc_generation_rate (activation);
  generation_time = repmat(threshold, 2, 4)./generation_rate;

end


function f = calc_f (r_minus_gen_time_diff, gradient)
    % gradient needs to be fitted? for congruent stimuli, =0, for incongruent =0.5
    if (r_minus_gen_time_diff > 0)
      f = r_minus_gen_time_diff * gradient;
    else
      f = 0;
    endif
end


function resolution_time = calc_resolution_time (generation_time, f_gradient, gauss_mean, gauss_sd, 
						 exp_lambda, irrelevant_stimulus_onset)

				%r generates random number from ex-gaussian distribution 
  # Is r generated independently for each task? Each trial?
#     generation_time_difference = [1, 1, -1, -1; 1, 1, -1, -1] .* [generation_time(2,:) - generation_time(1,:);
#   								   generation_time(1,:) - generation_time(2,:)]  
     # flip the signs for colour naming, 

  generation_time_plus_onset = generation_time + irrelevant_stimulus_onset

  generation_time_difference = [generation_time_plus_onset(2,:) - generation_time_plus_onset(1,:);
   				generation_time_plus_onset(1,:) - generation_time_plus_onset(2,:)]
  
				# insert r here if it is drawn once for all trials
  r = ((randn(1) * gauss_sd) + gauss_mean) + exprnd(exp_lambda) # Calculating for each trial

      resolution_time = zeros(2, 4);
      for j = 1:columns(resolution_time)

	for i = 1:rows(resolution_time)

				# Where is r recalculated?? 
	                        # Insert here if it is drawn individually for both tasks
	  resolution_time(i,j) = r + calc_f (r - generation_time_difference(i,j), f_gradient);
	end
      end
      resolution_time
end



function rt = calc_rt (activation, f_gradient, gauss_mean, gauss_sd, 
		       exp_lambda, threshold, constant, irrelevant_stimulus_onset)

  constant
  generation_time = calc_generation_time (activation, threshold)
  resolution_time = calc_resolution_time (generation_time, f_gradient, gauss_mean, gauss_sd, 
					  exp_lambda, irrelevant_stimulus_onset)

  rt =  constant + generation_time + resolution_time
  

end



# function rt = run_trial (taskstrength, control, priming, noise_mean, noise_sd, input_c, f_gradient, ...
# 			 exg_gauss_mean, exg_gauss_sd, exg_exp_lambda, threshold, rt_const)
function rt = run_trial ( params )

  input = zeros (2, 4);
  act = zeros (2, 4);
  rt = zeros (2, 4);
#  for i = 1:columns(input)
#    input = repmat(params.TASKSTRENGTH, 1, 4) + params.CONTROL(:,i) + params.PRIMING(:,i)
#    act = calc_activation (input(:,i), params.NOISE_MEAN, params.NOISE_SD, params.INPUT_C)
#    rt = calc_rt (act(:,i), params.F_GRADIENT, params.EXG_GAUSS_MEAN, params.EXG_GAUSS_SD, ...
#		  params.EXG_EXP_LAMBDA, params.THRESHOLD, params.RT_CONST, params.IRRELEVANT_STIM_ONSET)
#  end
    input = repmat(params.TASKSTRENGTH, 1, 4) + params.CONTROL + params.PRIMING;
    act = calc_activation (input, params.NOISE_MEAN, params.NOISE_SD, params.INPUT_C);
    rt = calc_rt (act, params.F_GRADIENT, params.EXG_GAUSS_MEAN, params.EXG_GAUSS_SD, ...
		  params.EXG_EXP_LAMBDA, params.THRESHOLD, params.RT_CONST, params.IRRELEVANT_STIM_ONSET);


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



function block = run_block (n, params)
  
  block = zeros(n, 4);
  for i = 1:n
				%		block(i,:) = make_rt_row_vector (run_trial (taskstrength,
    block(i,:) = make_correct_rt_row_vector (run_trial (params));
  end

  block

end




function p = plot_single_trial (params)

    % task parameters
  # params = struct('INPUT_C', 1.5, ...
  # 		  'NOISE_MEAN', 0.0, ...
  # 		  'NOISE_SD', 0.1, ...
  # 		  'THRESHOLD', 100, ...
  # 		  'F_GRADIENT', 0.5, ...
  # 		  'EXG_GAUSS_MEAN', 140, ...
  # 		  'EXG_GAUSS_SD', 10, ...
  # 		  'EXG_EXP_LAMBDA', 40, ...
  # 		  'RT_CONST', 150, ...
  # 		  'TASKSTRENGTH', [0.1; 0.5],
  # 		  'CONTROL', [0.00, 0.00, 0.97, 0.38;
  # 			      0.20, 0.15, 0.00, 0.00],
  # 		  'PRIMING', [0.3, 0.0, 0.0, 0.3; 
  # 			      0.0, 0.3, 0.3, 0.0 ]);


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

n = 100
# params = struct('INPUT_C', 1.5, ...
# 		'NOISE_MEAN', 0.0, ...
# 		'NOISE_SD', 0.1, ...
# 		'THRESHOLD', 100, ...
# 		'F_GRADIENT', 0.5, ...
# 		'EXG_GAUSS_MEAN', 140, ...
# 		'EXG_GAUSS_SD', 10, ...
# 		'EXG_EXP_LAMBDA', 40, ...
# 		'RT_CONST', 150, ...
# 		'TASKSTRENGTH', [0.1; 0.5],
# 		'CONTROL', [0.00, 0.00, 0.97, 0.38;
# 			    0.20, 0.15, 0.00, 0.00],
# 		'PRIMING', [0.3, 0.0, 0.0, 0.3; 
# 			    0.0, 0.3, 0.3, 0.0 ]);

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

% =============================================================== Load task parameters =====================

    % task parameters

control_default = [0.00, 0.00, 0.97, 0.38;
		   0.20, 0.15, 0.00, 0.00];

control_delayedonset = [0.00, 0.00, 0.15, 0.15;
			0.15, 0.15, 0.00, 0.00]

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
		'IRRELEVANT_STIM_ONSET', stim_onset_synchronous)


params_delayedonset = struct('INPUT_C', 1.5, ...
		'NOISE_MEAN', 0.0, ...
		'NOISE_SD', 0.1, ...
		'THRESHOLD', 100, ...
		'F_GRADIENT', 0.5, ...
		'EXG_GAUSS_MEAN', 140, ...
		'EXG_GAUSS_SD', 10, ...
		'EXG_EXP_LAMBDA', 40, ...
		'RT_CONST', 150, ...
		'TASKSTRENGTH', [0.1; 0.5],
		'CONTROL', control_delayedonset,
		'PRIMING', [0.3, 0.0, 0.0, 0.3; 
			    0.0, 0.3, 0.3, 0.0 ],
		'IRRELEVANT_STIM_ONSET', stim_onset_asynchronous)

# for debugging
params_nonoise = struct('INPUT_C', 1.5, ...
		'NOISE_MEAN', 0.0, ...
		'NOISE_SD', 0.0, ...
		'THRESHOLD', 100, ...
		'F_GRADIENT', 0.5, ...
		'EXG_GAUSS_MEAN', 140, ...
		'EXG_GAUSS_SD', 0, ...
		'EXG_EXP_LAMBDA', 40, ...
		'RT_CONST', 150, ...
		'TASKSTRENGTH', [0.1; 0.5],
		'CONTROL', control_default,
		'PRIMING', [0.3, 0.0, 0.0, 0.3; 
			    0.0, 0.3, 0.3, 0.0 ],
		'IRRELEVANT_STIM_ONSET', stim_onset_synchronous)
