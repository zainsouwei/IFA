function [O1,O2] = fpica2(X1,X2, ...
        approach, numOfIC, g, a1, a2, epsilon, maxNumIterations, ...
        initState, guess, displayMode, displayInterval, s_verbose);



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Default values

if nargin < 3, error('Not enought arguments!'); end
[vectorSize1, numSamples1] = size(X1);
[vectorSize2, numSamples2] = size(X2);

if nargin < 15, s_verbose = 'on'; end
if nargin < 14, displayInterval = 1; end
if nargin < 13, displayMode = 'on'; end
if nargin < 12, guess = 1; end
if nargin < 11, initState = 'rand'; end
if nargin < 10, maxNumIterations = 1000; end
if nargin < 9, epsilon = 0.0001; end
if nargin < 8, a2 = 1; end
if nargin < 7, a1 = 1; end
if nargin < 6, g = 'pow3'; end
if nargin < 5, numOfIC = vectorSize1; end     % vectorSize = Dim

b_verbose=1;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Checking the value for nonlinearity.

if strcmp(lower(g), 'pow3'),
  usedNlinearity = 1;
elseif strcmp(lower(g), 'tanh'),
  usedNlinearity = 2;
elseif strcmp(lower(g), 'gaus'),
  usedNlinearity = 3;
elseif strcmp(lower(g), 'gauss'),
  usedNlinearity = 3;
else
  error(sprintf('Illegal value [ %s ] for parameter: ''g''\n', g));
end
if b_verbose,
  fprintf('Used nonlinearity [ %s ].\n', g);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Checking the value for initial state.

if strcmp(lower(initState), 'rand'),
  initialStateMode = 0;
elseif strcmp(lower(initState), 'guess'),
  if approachMode == 7 %%% == 2 original CBchange
    initialStateMode = 0;
    if b_verbose, fprintf('Warning: Deflation approach - Ignoring initial guess.\n'); end
  else
    % Check the size of the initial guess. If it's not right then
    % use random initial guess
    if (size(whiteningMatrix,1)>=size(guess,2)) & ...
       (size(whiteningMatrix,2)==size(guess,1))
      initialStateMode = 1;
      if b_verbose, fprintf('Using initial guess.\n'); end
    else
      initialStateMode = 0;
      if b_verbose
        fprintf('Warning: size of initial guess is incorrect. Using random initial guess.\n');
      end
    end
  end
else
  error(sprintf('Illegal value [ %s ] for parameter: ''initState''\n', initState));
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% How many times do we try for convergence until we give up.
failureLimit = 5;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if b_verbose, fprintf('Starting ICA calculation...\n'); end
 

approachMode = 1; 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% SYMMETRIC APPROACH
if approachMode == 1,
  fprintf('.');

  if initialStateMode == 0
    % Take random orthonormal initial vectors.
    B = (rand(vectorSize1) - .5);
  end

  
  % This is the actual fixed-point iteration loop.
  for round = 1 : maxNumIterations + 1,

    if round == maxNumIterations + 1,
      if b_verbose 
        fprintf('No convergence after %d steps\n', maxNumIterations);
      end
      break;
    end

    [B1,S,B2]=svd(B);
    B1=B1*sqrt(S);
    B2=B2*sqrt(S);
    
    % Symmetric orthogonalization.
    %B = B*real(sqrtm(inv(B'*B)
B1Old = zeros(size(B1));

    % Test for termination condition. Note that we consider opposite
    % directions here as well.
    minAbsCos = min(abs(diag(B1'*B1Old)));
    if (1 - minAbsCos < epsilon),
      if b_verbose, fprintf('Convergence after %d steps\n', round);
      end
      O1=B1'*X1;
      O2=B2'*X2;
      break;
    end

 

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Show the progress...
    if b_verbose
      if round == 1
        fprintf('Step no. %d\n', round);
      else
        fprintf('Step no. %d, change in value of estimate: %.3f \n', round, 1 - minAbsCos);
      end
    end

    % First calculate the independent components (u_i's).
    % u_i = b_i' x = x' b_i. For all x:s simultaneously this is
    U1 = X1' * B1;
    U2 = X2' * B2;

    if usedNlinearity == 2,
      % tanh
      hypTan1 = tanh(a1 * U1);
      B1 = X1 * hypTan1 / numSamples1 - ...
          ones(size(B1,1),1) * sum(ones(size(U1)) .* (1 - hypTan1 .^ 2)) ...
          .* B1 / numSamples1 * a1;     
      hypTan2 = tanh(a1 * U2);
      B2 = X2 * hypTan2 / numSamples2 - ...
          ones(size(B2,1),1) * sum(ones(size(U2)) .* (1 - hypTan2 .^ 2)) ...
          .* B2 / numSamples2 * a1;
    end
  B=B1*B2';
  end
end
