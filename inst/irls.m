## Copyright (C) 2024 Yassin Achengli <0619883460@uma.es>
##
## This file is part of the statistics package for GNU Octave.
##
## This program is free software; you can redistribute it and/or modify it under
## the terms of the GNU General Public License as published by the Free Software
## Foundation; either version 3 of the License, or (at your option) any later
## version.
##
## This program is distributed in the hope that it will be useful, but WITHOUT
## ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
## FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
## details.
##
## You should have received a copy of the GNU General Public License along with
## this program; if not, see <http://www.gnu.org/licenses/>.

## -*- texinfo -*-
## @deftypefn {statistics} {[@var{beta}, @var{tolerance}] =} irls (@var{X}, @var{y}, @var{N}, @dots{})
## 
## Iteratively Reweighted Least Squares method.
##
## Used to solve optimization problems with an iterative method in which each step 
## involves solving a weighted least squares problem of the form:
##
## @tex
## \begin{equation}
## \argmin_{\beta} \sum_{i=1}^n{w_i\left(\beta^{(t)}\right)\mid y_i - f_i(\beta)\mid^2
## \end{equation}
## @end tex
## 
## The algorithm optimizes 
## 
## @example
## @group
## | yi - fi(beta) |^p |
## @end group
## @end example
##
## The p-norm of the error of yi estimated using a function of beta weights. This function
## takes linear functions of beta provided by a matrix of input variables @var{X}.
## 
## PARAMETERS:
## 
## @itemize
## @item
## var{X}
## categorical variables samples
## @item 
## @var{y}
## estimations
## @item 
## @var{N}
## number of iterations. Must be 1 or greater
## @end itemize
##
## OPTIONAL PARAMETERS:
##
## @itemize
## @item 
## @var{beta_0}
## seed values for beta output (first iteration) {nx1 matrix} (zeros by default)
## @item 
## @var{w_0}
## seed values for weight inter-iteration vector {nx1 matrix} (ones by default)
## @item 
## @var{normscale}
## level of normalization (2 by default)
## @item 
## @var{gamma}
## in order to avoid dividing by zero if @var{normscale} < 2 (default 1e-6)
## @end itemize
## 
## OUTPUT:
##
## @itemize
## @item 
## @var{beta}
## optimized weights
## @item 
## @{tolerance}
## mean of absolute value of the divergence between the last beta and 
## the previous one
## @end itemize
## 
## @seealso{glmfit,fsolve}
## @end deftypefn

function [beta, tolerance, Niter] = irls(X, y, N, varargin)
  if (nargin < 3)
    error("You need to explicitly give almost 3 arguments: X, y and N\n")
  endif

  if (size(X,1) ~= size(y,1))
    error("X and y must have the same row dimension")
  endif

  if (N <= 0)
    warning("N must be an integer greater to 0, taking 1 by default");
    N = 1;
  endif

  params = inputParser();

  params.addParameter('B0', zeros(size(X,2),1), @isvector);

  params.addParameter('weights', ones(size(y)), @(x) size(x) == size(y));

  params.addParameter('normscale', 2, @(x) isinteger(x) && x >= 0);
  params.addParameter('gamma', 1e-6, @isfloat);
  params.addParameter('tolerance', 1e-6, @isfloat);
  params.parse(varargin{:});
  
  beta = params.Results.B0;
  w = diag(params.Results.weights);
  p = params.Results.normscale;
  gamma = params.Results.gamma;
  beta_next = params.Results.B0;
  Niter = N;

  for iter = 1:N
    beta_next = inv(X'*w*X)*X'*w*y;
    tolerance = mean(abs(beta_next - beta));
    beta = beta_next;

    if (p < 2) % in order to avoid dividing by zero
      w = diag(1./max(abs(y - X*beta).^(-(p-2)), gamma));
    else
      w = diag(abs(y - X*beta).^(p-2));
    endif

    if (tolerance <= params.Results.tolerance)
      Niter = iter;
      break
    endif
  endfor
endfunction
## demonstrations
%!demo
%! X = randn(10,9);
%! y = rand(10,1);
%! [beta, tolerance] = irls(X, y, 100);
%! format long
%! disp("Teoric tolerance:")
%! tolerance_teo = mean(abs(y - X*beta))
%! disp("tolerance obtained with the algorithm:")
%! tolerance
## test
%!test
%! X = rand(10,9);
%! y = rand(10,1);
%! [beta, tol] = irls(X,y);
%! assert(size(beta), size(y));
