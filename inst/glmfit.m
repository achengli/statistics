## Copyright (C) 2024 Ruchika Sonagote <ruchikasonagote2003@gmail.com>
## Copyright (C) 2024 Andreas Bertsatos <abertsatos@biol.uoa.gr>
## Copyright (C) 2024 Yassin Achengli <0619883460@uma.es>
##
## This file is part of the statistics package for GNU Octave.
##
## Octave is free software; you can redistribute it and/or modify it
## under the terms of the GNU General Public License as published by
## the Free Software Foundation; either version 3 of the License, or
## (at your option) any later version.
##
## Octave is distributed in the hope that it will be useful, but
## WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.
##
## You should have received a copy of the GNU General Public License
## along with Octave; see the file COPYING.  If not,
## see <http://www.gnu.org/licenses/>.

## -*- texinfo -*-
## @deftypefn  {statistics} {@var{b} =} glmfit (@var{X}, @var{y}, @var{distribution})
## @deftypefnx {statistics} {@var{b} =} glmfit (@var{X}, @var{y}, @var{distribution},@var{Name}, @var{Value})
## @deftypefnx {statistics} {[@var{b}, @var{dev}] =} glmfit (@dots{})
##
## Perform generalized linear model fitting.
##
## @code{@var{b} = glmfit (@var{X}, @var{y}, @var{distribution})} returns a
## coefficient estimates vector, @var{b} for a
## generalized linear regression model of responses in @var{y} and
## predictors in @var{X}, using the @var{distribution}.
##
## @itemize
## @item @var{X} is an @math{nxp} numeric matrix of predictor variables with
## @math{n} observations and @math{p} predictors.
## @item @var{y} is a @math{n x 1} numeric vector of responses for all supported
## distributions, except for the 'binomial' distribution which can also have
## @var{y} as a @math{n x 2} matrix, where the first column contains the number
## of successes and the second column contains the number of trials.
## @item @var{distribution} specifies the distribution of the response variable
## (e.g., 'poisson').
## @end itemize
##
## @code{@var{b} = glmfit (@dots{}, @var{Name}, @var{Value})}
## specifies additional options using @qcode{Name-Value} pair arguments.
##
## @multitable @columnfractions 0.18 0.02 0.8
## @headitem @var{Name} @tab @tab @var{Value}
##
## @item @qcode{"link"} @tab @tab A character vector specifying a link
## function.
##
## @item @qcode{"constant"} @tab @tab Specifies whether to
## include a constant term in the model. Options are
## @var{"on"} (default) or @var{"off"}.
## @end multitable
##
## @code{[@var{b}, @var{dev}] = glmfit (@dots{})}
## returns the estimated coefficient vector, @var{b}, as well as
## the deviance, @var{dev}, of the fit.
##
## Supported distributions include 'poisson', 'binomial', and 'normal'.
## Supported link functions include 'identity', 'log', 'logit', 'probit',
## 'loglog', 'comploglog', 'reciprocal' and a custom link.
## Custom link function provided as a structure with three fields:
## Link Function, Derivative Function, Inverse Function.
## @end deftypefn

function [b, varargout] = glmfit (X, y, distribution, varargin)

  ## Check input
  if (nargin < 3)
    if (nargin == 2)
      distribution = 'normal';
    else
      error ("glmfit: too few input arguments.");
    endif
  elseif (mod(nargin - 3, 2) != 0)
    error ("glmfit: Name-Value arguments must be in pairs.");
  elseif (! isnumeric (X))
    error ("glmfit: X must be a numeric.");
  elseif (! isnumeric (y))
    error ("glmfit: Y must be a numeric.");
  elseif (size (X, 1) != size (y, 1))
    error ("glmfit: X and Y must have the same number of observations.");
  elseif (! ischar (distribution))
    error ("glmfit: DISTRIBUTION must be a character vector.");
  endif

  ## Missing values set to 0
  X(isnan(X)) = 0;
  y(isnan(y)) = 0;

  [my, ny] = size (y);
  [mx, nx] = size (X);

  ## Check dimensions based on distribution
  if (strncmp (distribution, 'binomial', 5) && ny > 2)
    error (["glmfit: for a binomial distribution,", ...
      " Y must be an n-by-1 or n-by-2 matrix."]);
  elseif (n != 1)
    error (["glmfit: for distributions other than the binomial,", ...
      " Y must be an n-by-1 column vector."]);
  endif
   
  IDENTITY_LINK = struct('link', @(t) t, 'derivative', @(t) 1, 'inverse', @(t) t);
  LOG_LINK = struct('link', @(t) log(t), 'derivative', @(t) 1/abs(t), ...
  'inverse', @(t) exp(t));
  LOGIT_LINK = struct('link', @(t) log(t./(1-t)), 'derivative', @(t) -1./(t^2-t),...
  'inverse', @(t) exp(t)./(1 + exp(t)));
  PROBIT_LINK = struct('link', @(t) probit(t), 'derivative', @(t), -sqrt(2^5)...
  *exp(4*t.^2)/sqrt(pi), 'inverse', @(t) erfc(-t/sqrt(2))/2);
  LOGLOG_LINK = struct('link', @(t) log(-log(t)), 'derivative', @(t) 1./(t.*log(t)),...
  'inverse', @(t) exp(-exp(t)));
  COMPLOGLOG_LINK = struct('link', @(t) log(-log(1-t)), 'derivative', -1/(log(1-t).*(1-t)),
  'inverse', 1 - exp(-exp(t)));
  RECIPROCAL_LINK = struct('link', @(t) 1./t, 'derivative', @(t) -1./(t.^2), 'inverse',...
  @(t) 1./t);
  P_LINK = struct('link', @(t, p) t.^p, 'derivative', @(t, p) p*t.^(p-1), 'inverse', ...
  @(t, p) t.^(1/p));

  DEFAULT_OPTIONS = struct('display', 'off', 'maxiter', 100, 'tolx', 1e-6);

  if (strcmp(distribution, 'poisson'))
    DEFAULT_LINK = LOG_LINK;
  elseif (strcmp(distribution, 'binomial'))
    DEFAULT_LINK = LOGIT_LINK;
    y = min(y(:,1) ./ y(:,2), y(:,2) ./ y(:,1));
  elseif (strcmp(distribution, 'gamma'))
    DEFAULT_LINK = RECIPROCAL_LINK;
  elseif (strcmp(distribution, 'inverse gaussian'))
    DEFAULT_LINK = P_LINK;
  else
    DEFAULT_LINK = IDENTITY_LINK;
  endif

  ## parsing named arguments
  parser = inputParser();
  parser.addParameter('link', DEFAULT_LINK, @isstruct);
  parser.addParameter('constant', 'on', @(x) sum(strcmp(x,{'on','off'})));

  if (strncmp(distribution, 'binomial') || strcmp(distribution, 'poisson'))
    parser.addParameter('estimdisp', 'off', @(x) sum(strcmp(x,{'on','off'})));
  else
    parser.addParameter('estimdisp', 'on', @(x) sum(strcmp(x,{'on','off'})));
  endif

  parser.addParameter('likelihoodpenalty', 'none', @ischar);
  parser.addParameter('offset', [], @(x) (isnumeric(x)));
  parser.addParameter('options', DEFAULT_OPTIONS, @isstruct);
  parser.addParameter('weights', ones(mx, 1), @isnumeric);
  parser.addParameter('B0', zeros(mx, 1), @isnumeric);

  parser.parse(varargin{:});
  results = parser.Results;

  if (strcmp(results.constant, 'off'))
    b = results.B0;
  else
    b = [1; results.B0]; % adding constant term
    X = [ones(mx,1), X];
  endif

  w = results.weights;

  N = 100;
  if (results.options.maxiter > 0)
    N = results.options.maxiter;
  endif

  if (ischar(link))
    if (strcmp(link, 'identity'))
      link = IDENTITY_LINK;
    elseif (strcmp(link, 'log'))
      link = LOG_LINK;
    elseif (strcmp(link, 'logit'))
      link = LOGIT_LINK;
    elseif (strcmp(link, 'probit'))
      link = PROBIT_LINK;
    elseif (strcmp(link, 'loglog'))
      link = LOGLOG_LINK;
    elseif (strcmp(link, 'comploglog'))
      link = COMPLOGLOG_LINK;
    elseif (strcmp(link, 'reciprocal'))
      link = RECIPROCAL_LINK;
    elseif (strncmp(link, 'p_function'))
      link = P_LINK;
    else
      error(strcat('<',link,'> link function not yet supported')
    endif
  elseif (isstruct(link))
    if !(isfield(link, 'link') && isfield(link, 'derivative') && isfield(link, 'inverse'))
      error('link function as struct must have "link", "derivative" and "inverse" ...
      fields each one containing the correspondant function')
    endif
  else
    error('link function must be a struct or a string, type help glmfit');
  endif
  
  % adding the offset to the model
  if (results.offset && size(results.offset) == [mx, 1])
    X = X + results.offset' * ones(mx,ny);
  endif

  % creating
  % estimation of the coefficients using IRLS (iteratively reweighted least squares)
  [b, tolerance] = irls(X, results.link.link(y), N, 'tolerance', abs(results.tolx), ...
  'weights', w, 'B0', results.B0);
  
  stats = struct;
  stats.covb = cov(X * b, X * b);
  stats.coeffcorr = corrcoef(X, y'*ones(size(X)));
  stats.dfe = length(b);
  # stats.sfit = ...;
  stats.estdisp = [0, 1](strcmp(results.estimdisp,'on') + 1);
  [M, V] = tstat(b);
  stats.t = {M,V};

  % residuals: basic residual, p-value residual, deviance residual, anscombe residual
  if (strcmp(results.constant, 'on'))
    stats.resid = abs(y - link.inverse(X(:,2:end)*b(2:end) + b(1)));
    deviance = y - mean(X(:,2:end)*b(2:end) + b(1));
  else
    stats.resid = abs(y - X*b);
    deviance = y - mean(X*b)
  endif

  stats.residp = stats.resid ./ sqrt(y);
  stats.residd = (y - link.inverse(X(:,2:end)*b(2:end) + b(1))) ./ sqrt(deviance);

  if (strcmp(results.constant, 'on'))
    mu = X(:,2:end) * b(2:end) + b(1);
  else
    mu = X*b;
  endif

  if (strncmp(distribution, 'binomial', 5))
    stats.resida = sqrt(y(:,1)).*(link.link(y(:,2)) - link.link(mu)) ./ ...
    (link.derivative(mu).*sqrt(var(mu)));
  else
    stats.resida = (link.link(y) - link.link(mu))./(link.derivative(mu).*sqrt(var(mu)));
  endif
endfunction

%!demo
%! rand ("seed", 1);
%! X = rand (100, 1);
%! b_true = [0.5; -1.2];
%! mu = exp (b_true(1) + b_true(2) * X);
%! randp ("seed", 1);
%! y = poissrnd (mu);
%! ## Fit a GLM model using the poisson distribution
%! [b,dev] = glmfit (X, y, 'poisson');

%!demo
%! x = [2100 2300 2500 2700 2900 3100 3300 3500 3700 3900 4100 4300]';
%! n = [48 42 31 34 31 21 23 23 21 16 17 21]';
%! y = [1 2 0 3 8 8 14 17 19 15 17 21]';
%! [b,dev] = glmfit (x,[y n],'binomial','Link','probit');

## Test output
%!test
%! rand ("seed", 1);
%! X = rand (50, 1);
%! b_true = [0.4; 1.5];
%! mu_true = exp (b_true(1) + b_true(2) * X);
%! randp ("seed", 1);
%! y = poissrnd (mu_true);
%! b = glmfit (X, y, "poisson", "link", "log");
%! assert (b(1), b_true(1), 0.5);
%! assert (b(2), b_true(2), 0.5);
%!test
%! rand ("seed", 1);
%! X1 = rand (50, 1);
%! X2 = rand (50, 1) * 0.5;
%! b_true = [0.4; 1.5; -0.7];
%! mu_true = exp (b_true(1) + b_true(2) * X1 + b_true(3) * X2);
%! randp ("seed", 1);
%! y = poissrnd(mu_true);
%! [b, dev] = glmfit ([X1, X2], y, "poisson", "link", "log");
%! assert (b(1), b_true(1), 1);
%! assert (b(2), b_true(2), 1);
%! assert (b(3), b_true(3), 1);
%! assert (dev < 60, true);
