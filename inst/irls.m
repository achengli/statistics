## Copyright (C) 2024 Yassin Achengli <0619883460@uma.es>
##
## This file is part of algorithm package for GNU Octave.
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
## @deftypefn {algorithm} {[@var{beta}, @var{tolerance}] =} irls(@var{X}, @var{y}, @var{N}, ...)
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
## X
function [beta, tolerance] = irls(X, y, N, varargin)
  if (nargin < 3)
    fprintf("You need to explicitly give almost 3 arguments: X, y and N\n")
    print_usage
  endif

  if (size(X,1) ~= size(y,1))
    error("X and y must have the same row dimension")
  endif

  if (N <= 0)
    warning("N must be an integer greater to 0, taking 1 by default");
    N = 1;
  endif

  params = inputParser();
  params.addOptional('beta_0', zeros(size(y)),
  @(x) isvector(x) && size(x) == size(y));
  params.addOptional('w_0', diag(ones(size(y))),
  @(x) isdiag(x) && size(diag(x)) == size(y));
  params.addOptional('normscale', 2, @(x) isinteger(x) && x >= 0);
  params.addOptional('gamma', 1e-6, @isfloat);
  params.parse(varargin{:});
  
  beta = params.Results.beta_0;
  w = params.Results.w_0;
  p = params.Results.normscale;
  gamma = params.Results.gamma;
  beta_next = params.Results.beta_0;

  for iter = 1:N
    beta_next = inv(X'*w*X)*X'*w*y;
    if (iter == N)
      tolerance = mean(abs(beta_next - beta));
    endif
    beta = beta_next;
    if (p < 2) % in order to avoid dividing by zero
      w = diag(1./max(abs(y - X*beta).^(-(p-2)), gamma));
    else
      w = diag(abs(y - X*beta).^(p-2));
    endif
  endfor
endfunction
%!demo
%! X = randn(10,9);
%! y = rand(10,1);
%! [beta, tolerance] = irls(X, y, 100);
%! format long
%! disp("Teoric tolerance:")
%! tolerance_teo = mean(abs(y - X*beta))
%! disp("tolerance obtained with the algorithm:")
%! tolerance
