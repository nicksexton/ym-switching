# not a function file
1;


function task_activation = calc_activation (input, noise_mean, noise_sd, c)
  % Implementation of Yeung & Monsell Equations 1 & 2, calculates an activation function 

  task_activation = 1 - exp(-1 * c * (input + (randn(1) * noise_sd) + noise_mean));
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
      f = x * gradient;
    else
      f = 0;
    endif
end


function resolution_time = calc_resolution_time (generation_time, f_gradient, gauss_mean, gauss_sd, exp_lambda)
% column vector generation time

    %generate random number from ex-gaussian distribution 
      r = ((randn(1) * gauss_sd) + gauss_mean) + exprnd(exp_lambda);
    
	% gradient needs to be fitted? for congruent stimuli, =0, for incongruent =0.5
      resolution_time = r + calc_f (r - generation_time(1,1) - generation_time(2,1), f_gradient);
end



function rt = calc_rt (activation, f_gradient, gauss_mean, gauss_sd, exp_lambda, threshold, constant)

  generation_time = calc_generation_time (activation, threshold);
   rt =  constant + generation_time + ...
     calc_resolution_time (generation_time, f_gradient, gauss_mean, gauss_sd, exp_lambda);

end


    % task parameters
    params = struct('INPUT_C', 1.5, ...
		    'NOISE_MEAN', 0.0, ...
		    'NOISE_SD', 0.1, ...
		    'THRESHOLD', 100, ...
		    'F_GRADIENT', 0.5, ...
		    'EXG_GAUSS_MEAN', 140, ...
		    'EXG_GAUSS_SD', 10, ...
		    'EXG_EXP_LAMBDA', 40, ...
		    'RT_CONST', 150, ...
		    'TASKSTRENGTH', [0.1; 0.5],
		    'CONTROL', [0.00, 0.00, 0.97, 0.38;
				 0.20, 0.15, 0.00, 0.00],
		    'PRIMING', [0.3, 0.0, 0.0, 0.3; 
				 0.0, 0.3, 0.3, 0.0 ])


function rt = run_trial (taskstrength, control, priming, noise_mean, noise_sd, input_c, f_gradient, ...
			 exg_gauss_mean, exg_gauss_sd, exg_exp_lambda, threshold, rt_const)

    input = zeros (2, 4);
    act = zeros (2, 4);
    rt = zeros (2, 4);
    for i = 1:columns(input)
	      input(:,i) = taskstrength(:,1) + control(:,i) + priming(:,i);
              act(:,i) = calc_activation (input(:,i), noise_mean, noise_sd, input_c);
              rt(:,i) = calc_rt (act(:,i), f_gradient, exg_gauss_mean, exg_gauss_sd, ...
		   exg_exp_lambda, threshold, rt_const);
    end
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



function block = run_block (n, taskstrength, control, priming, noise_mean, noise_sd, input_c, f_gradient, ...
			 exg_gauss_mean, exg_gauss_sd, exg_exp_lambda, threshold, rt_const)

      block = zeros(n, 4);
      for i = 1:n
%		block(i,:) = make_rt_row_vector (run_trial (taskstrength,
		block(i,:) = make_correct_rt_row_vector (run_trial (taskstrength,
								     control,
								     priming,
								     noise_mean,
								     noise_sd,
								     input_c,
								     f_gradient,
								     exg_gauss_mean,
								     exg_gauss_sd,
								    exg_exp_lambda,
								    threshold,
								    rt_const));
      end


end




function p = plot_single_trial ()

    % task parameters
    params = struct('INPUT_C', 1.5, ...
		    'NOISE_MEAN', 0.0, ...
		    'NOISE_SD', 0.1, ...
		    'THRESHOLD', 100, ...
		    'F_GRADIENT', 0.5, ...
		    'EXG_GAUSS_MEAN', 140, ...
		    'EXG_GAUSS_SD', 10, ...
		    'EXG_EXP_LAMBDA', 40, ...
		    'RT_CONST', 150, ...
		    'TASKSTRENGTH', [0.1; 0.5],
		    'CONTROL', [0.00, 0.00, 0.97, 0.38;
				 0.20, 0.15, 0.00, 0.00],
		    'PRIMING', [0.3, 0.0, 0.0, 0.3; 
				 0.0, 0.3, 0.3, 0.0 ]);


rt = run_trial (params.TASKSTRENGTH,
		params.CONTROL,
		params.PRIMING,
		params.NOISE_MEAN,
		params.NOISE_SD,
		params.INPUT_C,
		params.F_GRADIENT,
		params.EXG_GAUSS_MEAN,
		params.EXG_GAUSS_SD,
		params.EXG_EXP_LAMBDA,
		params.THRESHOLD,
		params.RT_CONST);


correct = make_rt_row_vector (rt);
plotdata = make_correct_plot_data (correct);



plot (plotdata);
xlim ([1, 2]);
xlabel ('Repeat vs. Switch');
ylabel ('Simulated RT (ms)');
legend ('Word Reading', 'Colour Naming');


end


function block = plot_block     ()

n = 100
		params = struct('INPUT_C', 1.5, ...
		    'NOISE_MEAN', 0.0, ...
		    'NOISE_SD', 0.1, ...
		    'THRESHOLD', 100, ...
		    'F_GRADIENT', 0.5, ...
		    'EXG_GAUSS_MEAN', 140, ...
		    'EXG_GAUSS_SD', 10, ...
		    'EXG_EXP_LAMBDA', 40, ...
		    'RT_CONST', 150, ...
		    'TASKSTRENGTH', [0.1; 0.5],
		    'CONTROL', [0.00, 0.00, 0.97, 0.38;
				 0.20, 0.15, 0.00, 0.00],
		    'PRIMING', [0.3, 0.0, 0.0, 0.3; 
				 0.0, 0.3, 0.3, 0.0 ]);


		block = run_block (n,
				params.TASKSTRENGTH,
				params.CONTROL,
				params.PRIMING,
				params.NOISE_MEAN,
				params.NOISE_SD,
				params.INPUT_C,
				params.F_GRADIENT,
				params.EXG_GAUSS_MEAN,
				params.EXG_GAUSS_SD,
				params.EXG_EXP_LAMBDA,
				params.THRESHOLD,
				   params.RT_CONST);
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

