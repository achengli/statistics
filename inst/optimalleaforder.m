## Copyright (C) 2021 Stefano Guidoni
##
## This file is part of the statistics package for GNU Octave.
##
## This program is free software: you can redistribute it and/or
## modify it under the terms of the GNU General Public License as
## published by the Free Software Foundation, either version 3 of the
## License, or (at your option) any later version.
##
## This program is distributed in the hope that it will be useful, but
## WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
## General Public License for more details.
##
## You should have received a copy of the GNU General Public License
## along with this program; see the file COPYING.  If not, see
## <http://www.gnu.org/licenses/>.

## -*- texinfo -*-
## @deftypefn  {statistics} {@var{leafOrder} =} optimalleaforder (@var{tree}, @var{D})
## @deftypefnx {statistics} {@var{leafOrder} =} optimalleaforder (@dots{}, @var{Name}, @var{Value})
##
## Compute the optimal leaf ordering of a hierarchical binary cluster tree.
##
## The optimal leaf ordering of a tree is the ordering which minimizes the sum
## of the distances between each leaf and its adjacent leaves, without altering
## the structure of the tree, that is without redefining the clusters of the
## tree.
##
## Required inputs:
## @itemize
## @item
## @var{tree}: a hierarchical cluster tree @var{tree} generated by the
## @code{linkage} function.
##
## @item
## @var{D}: a matrix of distances as computed by @code{pdist}.
## @end itemize
##
## Optional inputs can be the following property/value pairs:
## @itemize
## @item
## property 'Criteria' at the moment can only have the value 'adjacent',
## for minimizing the distances between leaves.
##
## @item
## property 'Transformation' can have one of the values 'linear', 'inverse'
## or a handle to a custom function which computes @var{S} the similarity
## matrix.
## @end itemize
##
## optimalleaforder's output @var{leafOrder} is the optimal leaf ordering.
##
## @strong{Reference}
## Bar-Joseph, Z., Gifford, D.K., and Jaakkola, T.S. Fast optimal leaf ordering
## for hierarchical clustering. Bioinformatics vol. 17 suppl. 1, 2001.
## @end deftypefn
##
## @seealso{dendrogram,linkage,pdist}

function leafOrder = optimalleaforder ( varargin )

  ## check the input
  if ( nargin < 2 )
    print_usage ();
  endif

  tree = varargin{1};
  D = varargin{2};
  criterion = "adjacent";               # default and only value at the moment
  transformation = "linear";

  if ((columns (tree) != 3) || (! isnumeric (tree)) || ...
      (! (max (tree(end, 1:2)) == rows (tree) * 2)))
    error (["optimalleaforder: tree must be a matrix as generated by the " ...
      "linkage function"]);
  endif

  ## read the paired arguments
  if (! all (cellfun ("ischar", varargin(3:end))))
    error ("optimalleaforder: character inputs expected for arguments 3 and up");
  else
    varargin(3:end) = lower (varargin(3:end));
  endif
  pair_index = 3;
  while (pair_index <= (nargin - 1))
    switch (varargin{pair_index})
      case "criteria"
        criterion = varargin{pair_index + 1};
        if (strcmp (criterion, "group"))
          ## MATLAB compatibility:
          ## the 'group' criterion is not implemented
          error ("optimalleaforder: unavailable criterion 'group'");
        elseif (! strcmp (criterion, "adjacent"))
          error ("optimalleaforder: invalid criterion %s", criterion);
        endif
      case "transformation"
        transformation = varargin{pair_index + 1};
      otherwise
        error ("optimalleaforder: unknown property %s", varargin{pair_index});
    endswitch

    pair_index += 2;
  endwhile

  ## D can be either a vector or a matrix,
  ## but it is easier to work with a matrix
  if (isvector (D))
    D = squareform (D);
  endif

  n = rows (D);
  m = rows (tree);

  if (n != (m + 1))
    error (["optimalleaforder: D must be a matrix or vector generated by " ...
      "the pdist function"]);
  endif


  ## the similarity matrix, basically an inverted distance matrix
  S = zeros (n);

  if (strcmpi (transformation, "linear"))
    ## linear similarity
    maxD = max (max (D));
    S = maxD - D;
  elseif (strcmpi (transformation, "inverse"))
    ## similarity as inverted distance
    S = 1 ./ D;
  elseif (is_function_handle (transformation))
    ## custom similarity
    S = feval (transformation, D);
  else
    error ("optimalleaforder: invalid transformation %s", transformation);
  endif


  ## main body

  ## for each node v we compute the maximum similarity of the subtree M(w,u,v),
  ## where the leftmost leaf is w and the rightmost is u; remember that
  ## M(w,u,v) = M(u,w,v)
  M = zeros (n, n, n + m);

  ## O is a utility matrix: for each node of the tree we store the left and
  ## right leaves of the optimal subtree
  O = [1:( n + m ); 1:( n + m ); (zeros (1, (n + m)))]';

  ## compute M for every node v
  for iter = 1 : m
    v = iter + n;                                     # current node
    l = optimalleaforder_getLeafList (tree(iter, 1)); # the left subtree
    r = optimalleaforder_getLeafList (tree(iter, 2)); # the right subtree

    if (tree(iter,1) > n)
      l_l = optimalleaforder_getLeafList (tree(tree(iter, 1) - n, 1));
      l_r = optimalleaforder_getLeafList (tree(tree(iter, 1) - n, 2));
    else
      l_l = l_r = l;
    endif

    if (tree(iter,2) > n)
      r_l = optimalleaforder_getLeafList (tree(tree(iter, 2) - n, 1));
      r_r = optimalleaforder_getLeafList (tree(tree(iter, 2) - n, 2));
    else
      r_l = r_r = r;
    endif

    ## let's find the maximum value of M(w,u,v) when: w is a leaf of the left
    ## subtree of v and u is a leaf of the right subtree of v
    for i = 1 : length (l)
      if (isempty (find (l(i) == l_l)))
        x = l_l;
      else
        x = l_r;
      endif
      for j = 1 : length (r)
        if (isempty (find (r(j) == r_l)))
          y = r_l;
        else
          y = r_r;
        endif

        ## max(M(w,u,v)) = max(M(w,k,v_l)) + max(M(h,u,v_r)) + S(k,h)
        ## where: v_l is the left child of v and v_r the right child of v
        M_tmp = repmat (M(l(i), x(:), tree(iter, 1)), length (y), 1) + ...
                repmat (M(y(:), r(j), tree(iter, 2)), 1, length (x)) + ...
                S(y(:), x(:));
        M_max = max (max (M_tmp));      # this is M(l(i), r(j), v)
        [h, k] = find (M_tmp == M_max);

        M(l(i), r(j), v) = M_max;
        M(r(j), l(i), v) = M(l(i), r(j), v);

        if (M_max > O(v,3))
          O(v, 1) = l(i);               # this is w
          O(v, 2) = r(j);               # this is u
          O(v, 3) = M_max;              # this is M(w, u, v)
        endif
      endfor
    endfor
  endfor

  ## reordering:
  ## we found the M(w,u,v) corresponding to the optimal leaf order, now we can
  ## compute the optimal leaf order given our M(w,u,v)

  ## the return value
  leafOrder = zeros ( 1, n );
  leafOrder(1) = O(end, 1);
  leafOrder(n) = O(end, 2);

  ## the inverse operation, only easier, to get the leaf order: now we know the
  ## leftmost and rightmost leaves of the best subtree, we may have to flip it
  ## though
  for iter = m : -1 : 1
    v = iter + n;

    extremes = O(v, [1, 2]);

    l_node = tree(iter, 1);
    r_node = tree(iter, 2);

    l = optimalleaforder_getLeafList (l_node);
    r = optimalleaforder_getLeafList (r_node);

    if (l_node > n)
      l_l = optimalleaforder_getLeafList (tree(l_node - n, 1));
      l_r = optimalleaforder_getLeafList (tree(l_node - n, 2));
    else
      l_l = l_r = l;
    endif

    if (r_node > n)
      r_l = optimalleaforder_getLeafList (tree(r_node - n, 1));
      r_r = optimalleaforder_getLeafList (tree(r_node - n, 2));
    else
      r_l = r_r = r;
    endif

    ## this means that we need to flip the subtree
    if (isempty (find (extremes(1) == l)))
      l_tmp = l;
      l_l_tmp = l_l;
      l_r_tmp = l_r;

      l = r;
      l_l = r_l;
      l_r = r_r;

      r = l_tmp;
      r_l = l_l_tmp;
      r_r = l_r_tmp;

      node_tmp = l_node;
      l_node = r_node;
      r_node = node_tmp;
    endif

    if (isempty (find (extremes(1) == l_l)))
      x = l_l;
    else
      x = l_r;
    endif

    if (isempty (find (extremes(2) == r_l)))
      y = r_l;
    else
      y = r_r;
    endif

    M_tmp = repmat (M(extremes(1), x(:), l_node), length (y), 1) + ...
            repmat (M(y(:), extremes(2), r_node), 1, length (x)) + ...
            S(y(:), x(:));
    M_max = max (max (M_tmp));
    [h, k] = find (M_tmp == M_max);

    O(l_node, 1) = extremes(1);
    O(l_node, 2) = x(k);
    O(r_node, 1) = y(h);
    O(r_node, 2) = extremes(2);

    p_1 = find (leafOrder == extremes(1));
    p_2 = find (leafOrder == extremes(2));

    leafOrder (p_1 + (length (l)) - 1) = x(k);
    leafOrder (p_1 + (length (l))) = y(h);
  endfor

  ## function: optimalleaforder_getLeafList
  ## get the list of leaves under a given node
  function vector = optimalleaforder_getLeafList (nodes_to_visit)
    vector = [];
    while (! isempty (nodes_to_visit))
      currentnode = nodes_to_visit(1);
      nodes_to_visit(1) = [];
      if (currentnode > n)
        node = currentnode - n;
        nodes_to_visit = [tree(node, [2 1]) nodes_to_visit];
      endif

      if (currentnode <= n)
        vector = [vector currentnode];
      endif
    endwhile
  endfunction

endfunction

%!demo
%! randn ("seed", 5)  # for reproducibility
%! X = randn (10, 2);
%! D = pdist (X);
%! tree = linkage(D, 'average');
%! optimalleaforder (tree, D, 'Transformation', 'linear')

## Test input validation
%!error optimalleaforder ()
%!error optimalleaforder (1)
%!error <tree must be .*> optimalleaforder (ones (2, 2), 1)
%!error <character inputs expected> optimalleaforder ([1 2 3], [1 2; 3 4], "criteria", 5)
%!error <D must be .*> optimalleaforder ([1 2 1], [1 2 3])
%!error <unknown property .*> optimalleaforder ([1 2 1], 1, "xxx", "xxx")
%!error optimalleaforder ([1 2 1], 1, "Transformation", "xxx")
