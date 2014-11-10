function x = exg_rnd_create (T, burnin, mu, sigma, lambda, proposal_sigma)
# Creates x, a vector of length T-burnin of ex-gaussian distributed random variates, using the Metropolis sampler



# Initialise the Metropolis sampler
#T = 5000;     # max iterations
#burnin = 500; # samples to exclude
#mu = 140;
#sigma = 10;
#lambda = 40;

#proposal_sigma = 20 # SD for proposal distribuition

thetamin  = 0; thetamax = 400; # define a range for starting values
theta = zeros(1, T); # init storage space for our samples
#seed=1; rand('state', seed); randn('state',seed); # set the random seed
theta(1) = unifrnd (thetamin, thetamax);
accepted = [0 0]; # accept : reject

# start sampling
t = 1;
while t < T # iterate until we have T samples
  t=t+1;

  # propose a new value for theta using proposal density
#  theta_star = gamrnd (theta(t-1)*tau, 1/tau);
 theta_star = normrnd (theta(t-1), proposal_sigma);

  # calculate the acceptance ratio


  alpha = min ([ 1 (exgauss_pdf ( theta_star, [mu, sigma, lambda]) / ...
		    exgauss_pdf (theta(t-1), [mu, sigma, lambda] )) ...
		]);
  # draw a uniform deviate from [0. 1]
  u = rand;
  # do we accept this proposal?
  if u < alpha
     theta(t) = theta_star; # if so, proposal becomes a new state
     accepted(1) ++;
  else
    theta(t) = theta(t-1); # if not, copy old state
    accepted(2) ++;
  end
end

# # Display histogram of our samples
# figure (1); clf;
# subplot (3,1,1);
# nbins = 200;
# thetabins = linspace (thetamin, thetamax, nbins);
# counts = hist (theta(burnin:T), thetabins);
# bar (thetabins, counts/sum(counts), 'k');
# xlim ([thetamin thetamax]);
# xlabel ('\theta'); ylabel ('p(\theta)');

# # # # overlay the theoretical density
# y = exgauss_pdf (thetabins, [mu, sigma, lambda]);
# hold on;
# plot (thetabins, y/sum(y), 'r--', 'LineWidth', 3);
# set (gca, 'YTick', []);

# # display history of our samples
# subplot (3,1,2:3);
# stairs (theta, 1:T, 'k-');
# ylabel ('t'); xlabel ('\theta');
# set (gca, 'YDir', 'reverse');
# xlim ([thetamin thetamax]);

x = theta(burnin:T);

proportion_accepted = accepted(1) / sum(accepted)
expected_value = sum(theta(burnin:T)/(T-burnin))
variance = sum((theta(burnin:T) - expected_value).^2)/(T-burnin)
