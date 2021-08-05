#!/usr/bin/env python3
# Author: Rafael Polli Carneiro
# Date  : First semester of 2021
#
#
# * IMPORTANT
# -----------
#   Here the field acting over the vector space is Z/2Z.
#
# * HOW TO USE THIS SCRIPT
# -------------------------
#   Given a network N = (X, A) where X represents the
#   set of the points being studied at the graphs and
#   A a matrix of the weights.
#
#   X is stored as a numpy array with shape satisfying
#     + X.shape[0] ---> size of our data
#     + X.shape[1] ---> dimension of the R^n space
#
#   A is stored as a numpy array with shape
#      A.shape == (X.shape[0], X.shape[0]),
#   and A must satisfy:
#      + A[x,y] >= 0            for all x,y in {0, 1, 2, 3, ..., X.shape[0] - 1};
#      + A[x,y] == 0 iff x = y, for all x,y in {0, 1, 2, 3, ..., X.shape[0] - 1}.
#
#
#   Then, if one wants to calculate the persistent path homology of
#   dimensions 0, 1, 2, ..., p (with p >= 0), it has to do the following
#   in a python interpreter (or in a scriptm it is up to you)
#
#   >>> from persistent_path_homology import *
#   >>> pph_X = PPH(X, A)
#   >>> pph_X.ComputePPH()
#   >>> print(pph.Pers)
#
#   Above pph_X is an object of the class PPH and the method
#   ComputePPPH() will calculate the persistent path homology
#   and will store at the atribute of pph_X, pph_X.Pers, a list
#   containing all persistent path intervals, where
#     pph_X.Pers[0] is a list containing persistent path features of dimension 0
#     pph_X.Pers[1] is a list containing persistent path features of dimension 1
#     pph_X.Pers[2] is a list containing persistent path features of dimension 2
#                              .
#                              .
#                              .
#
#  In case you want to see step by step of the algorithm working you
#  must call the method ComputePPH_printing_step_by_step(). For
#  instance:
#
#   >>> from persistent_path_homology import *
#   >>> pph_X = PPH(X, A)
#   >>> pph_X.ComputePPH_printing_step_by_step()
#
# * TECHNICAL INFORMATION.
# ------------------------
#
# 1) The algorithm proposed by the authors above do not
#    work properly if we do not set the regular paths of
#    dimension 0 to be marked. That is, it is mandatory to
#    mark this regular paths otherwise the persistent features
#    of dimension 0 won't be detected by the algorithm and also
#    it will lead to incorrect persistent diagrams.
#    This can be noticed by checking the algorithm proposed
#    by Afra Zomorodian and Gunnar Carlsson, namely
#    Computing Persistent Homology, to calculate persistent
#    homology. This paper (the pdf) can be found in the link:
#    ---> https://geometry.stanford.edu/papers/zc-cph-05/
#
# 2) The algorithm implemented here can be found
#    here:
#    ---> https://epubs.siam.org/doi/10.1137/1.9781611975031.75
#    by the authors: Samir Chowdhury and Facundo Mémoli.
#
# 3) The software used to write this python script
#    was the magnificent Doom-Emacs, which is an
#    Emacs embedded with all functionalities of
#    VIM. I really recommend using emacs with
#    this framework. Down bellow I leave its github
#    repository:
#    ---> https://github.com/hlissner/doom-emacs
#
# 4) This script has been tested using python3 on
#    a Ubuntu machine.


import numpy as np
from math import *

#############################################
######## Auxiliary Functions
#############################################

def generating_all_regular_paths_dim_p( B_i, network_set_size ):
    """
    Given a numpy array B_i of shape (a,b), representing
    all regular paths of dimension b, this function
    will return a numpy array B_{i+1} of shape (a * (network_set_size - 1), b+1),
    representing all regular paths of dimension b+1.

    This auxiliary function will be used at the method:
       PPH.Basis_of_the_vector_spaces_spanned_by_regular_paths().
    """

    # Notation ->  B_next := B_{i+1}
    size_B_next = B_i.shape[0] * (network_set_size - 1 )
    B_next = np.zeros( (size_B_next, B_i.shape[1] + 1) )

    l = 0
    for i in range( B_i.shape[0] ):
        for j in range( network_set_size ):
            if B_i[ i, -1 ] != j:
                B_next[l, 0:B_i.shape[1] ] = B_i[i,:]
                B_next[l, -1 ] = j
                l += 1

    return B_next


                                             
#############################################
######## Main Class: PPH
#############################################
class PPH:
    """
    This class will calculate the persitent path homology
    of a network (network_set, network_weight) which satisfies
      (i)  network_set is a finite set represented by a
           numpy array;

      (ii) network_weight is a function definied mathematically
           as:
           network_weight: network_set x network_set -> R_+

           The machine will store this function as a numpy array
           of dimension (network_set, network_set)


    After storing the network the class PPH will calculate its
    persistent path homology of a given dimension d. The method
    responsible for such calculation is: diagram

    All the algorithm deployed here is based at the theory developed
    at the following paper:

       + PERSISTENT PATH HOMOLOGY OF DIRECTED NETWORKS, from the
         authors Samir Chowdhury and Facundo Mémoli

    This paper can be easily found at the website:
       -> https://arxiv.org/abs/1701.00565

    The same authors have a paper with an algorithm implemented
    to calculate the persistent path homology. The algorithm
    resembles the one to calculate the persitent diagram when
    considering the field Z/2Z over the vector spaces from
    the homology groups.

    The paper with the algorithm can be found here:
       -> https://epubs.siam.org/doi/10.1137/1.9781611975031.75
    exectly at the end of the file.
    """

    def __init__(self, network_set, network_weight, pph_dim):
        """
        atributes:
          + network_set: a numpy array storing the sets of
                         the network

          + network_weight: a nupy array storing the weight
                            function which do not need to be symmetric

          + network_set_size: integer representing the size of our
                              data

          + pph_dim: dimension of the persistent path diagram

          + basis: a list looking like

                 basis := [ B0, B_1, B_2, ..., B_(self.dim + 1) ].

            where each element is a numpy array:
                B0: stores regular paths of dimension 0;
                B1: stores regular paths of dimension 1;
                B2: stores regular paths of dimension 2;
                                 .
                                 .
                                 .
          + basis_dim: a list containing as elements the values
                       of the dimensions of each vector space
                       spaned by the basis B0, B1, B2, above.
                       In other words:
                       basis_dim := [ basis[0].shape[0],
                                      basis[1].shape[0], ... ]

          + T_p : a list that looks like
                  T_p := [ T_p_0, T_p_1, ..., T_p_{self.pph_dim} ]
                  where T_p_0, T_p_1, ..., T_p_{self.pph_dim} are
                  lists so, lets say for the index i, we will
                  have
                      T_p_i = [ [v_0, et( v_0 ), at( v_0 ), empty_or_not],
                                [v_1, et( v_1 ), at( v_1 ), empty_or_not],
                                [v_2, et( v_2 ), at( v_2 ), empty_or_not],
                                                 .
                                                 .
                                                 .
                              ],
                  where v_0, v_1, v_2, ... are vectors of the
                  vector space of dimension dim( basis[i] ). Also,
                  et(v_0), et(v_1), et(v_2) stands for the entry time
                  of such vectors. Finnaly, the variable named
                  empty_or_not registers wheter T_p_i[j]  is empty or not,
                  for j in 0, 1, 2, ..., self.basis_dim[i].

          + Marked: a list
                       Marked := [Marked0, Marked1, Marked_{self.pph_dim}]
                    where each element Marked0, Marked1, Marked_{self.pph_dim} is
                    a numpy.array that stores wheter a basis vector is marked or not.

          + Pers : a list consisting of the persistent diagrams of dimensions 0, 1, ..., self.pph_dim.
                   That is:
                            Pers = [ Pers0, Pers1, ..., Pers_{self.pph_dim} ],
                   where Pers0, Pers1, ..., Pers_{self.pph_dim} are lists whose elements are the intervals of
                   the persistent diagrams.
        """

        self.pph_dim   = pph_dim
        self.basis     = []
        self.basis_dim = []
        self.T_p       = []
        self.Marked    = []
        self.Pers      = []

        if network_set.size != 0:
            self.network_set = network_set
            self.network_set_size = network_set.shape[0]

        else:
            print( "Please the network_set cannot be the empty set\n." )
            return 0


        if network_weight.shape[0] == network_weight.shape[1] and \
           network_weight.shape[0] > 0:

            self.network_weight = network_weight

        else:
            print( "Please the network_weight must be a square matrix and must not be empty\n." )
            return 0

        #### constraints for the T_p structure indexes
        self.ARRAY_INDEX = 0
        self.ENTRY_INDEX = 1
        self.EMPTY_INDEX = 2


    def Basis_of_the_vector_spaces_spanned_by_regular_paths(self):
        """
        Here we will be storing the regular paths of
        dimension 0, 1, ..., self.pph_dim + 1 into a list
        called basis:

             basis := [ B0, B_1, B_2, ..., B_(self.pph_dim + 1) ].

        The elements of the list basis are numpy arrays such
        that:
            B0: stores regular paths of dimension 0;
            B1: stores regular paths of dimension 1;
            B2: stores regular paths of dimension 2;
                             .
                             .
                             .

        The list Basis is an atribute of our class PPH.

        Important, the paths of dimension p will be represented
        by a sequence a_0, a_1, a_2, ..., a_p satisfying

             * a_0, a_1, a_2, ..., a_p in {0, 1, 2, ..., p}
             * and a_i != a_{i+1}, for all i in {0, 1, 2, ..., p-1}.

        Therefore, if a_i = 7, then a_i is merely an index pointer
        of the element 7 of the numpy.array network_set.

        With that in mind, suppose that we have
        network_set_size = 3, then

        B0 = numpy.array( [ [0], [1], [2] ] )
        B1 = numpy.array( [ [0,1], [0,2], [1,2], [1,0], [2,0], [2,1]] )
        B2 = numpy.array( [ [0,1,2], [1,2,0], [2,0,1], ... )

        This is merely a combinatorial task!
        """

        (self.basis).append( np.zeros( (self.network_set_size, 1) ) )

        for i in range( self.network_set_size ):
            self.basis[0][i,0] = i

        i = 0
        while i < self.pph_dim + 1:
            self.basis.append( generating_all_regular_paths_dim_p( self.basis[i], self.network_set_size ) )
            i += 1


    def dimensions_of_each_vector_space_spanned_by_regular_paths(self):
        """
        Given the list self.basis of numpy arrays this method will
        store at the list self.basis_dim the dimensions of the
        vector spaces spanned by regular paths of dimension 0,1,2,...,
        self.dim + 1.
        """
        i = 0
        while i <= self.pph_dim + 1:
           self.basis_dim.append( self.basis[i].shape[0] )
           i += 1


    def allow_time (self, path_vector, path_dim):
        """
        Given the vector path_vector, an element of the vector
        space spanned by the regular paths of dimension path_dim.
        That is
            path_vector := sum_{i=0}^{self.basis_dim} alpha_i * basis[path_dim][i]
        with alpha_i in Z/2Z.

        Then this function will calculate the time of birth of
        such vector. In another words, this method
        returns the value
            min { k in A_k; path_vector in A_k, k >= = },
        where A_k is the set of the allowed regular paths of dimension
        k.

        Note tha path_vector is a numpy array that will eventually
        look like this, for instance
           path_vector = np.array([ 0, 0, 1, 1, 1, 0, 0, 1]),
        meaning that
           path_vector = self.basis[path_dim][2] + self.basis[path_dim][3] +
                         self.basis[path_dim][4] + self.basis[path_dim][7]
        """

        if path_dim == 0:
            return 0

        distance = []

        # Considering that
        # path_vector := sum_{i=0}^{self.basis_dim[path_dim]} alpha_i * self.basis[path_dim][i]
        find_indexes = np.arange( path_vector.size )[ path_vector == 1 ]

        # Let's find the elements of the basis that generate path_vector.
        # We will write vector_path = sum_i alpha_i * sigma_i,
        # with alpha_i in Z/2Z and sigma_i in self.basis[path.dim]

        for i in find_indexes:
            j = 0
            sigma_i = self.basis[path_dim][i] # sigma_i will look like: np.array( [ 0,2, 5,1, 3 ] ), for  instance

            # now we will run through the the vertices of the path sigma_i and we will
            # store the time needed for the edges to appear.
            while j < path_dim:
                distance.append( self.network_weight[ int(sigma_i[j]) , int(sigma_i[j+1]) ] )
                j += 1

        return max( distance )


    def entry_time(self, path_vector, path_dim ):

        if path_dim == 0:
            return 0

        elif path_dim == 1:
            return self.allow_time( path_vector, path_dim )

        else:
            distance = [ self.allow_time( path_vector, path_dim ) ]

            # Finding the basis vectors that generate
            # the vector path_vector.
            basis_that_generate_path_vector = (self.basis[ path_dim ])[ path_vector == 1 ]

            # Now we will apply the boundary transformation
            # d = sum_{j=0}^{path_dim} (-1)^j [a_0, a_1, ..., â_j, ...]

            # Taking
            #   path_vector = sum_i sigma_i
            # we will take the allow times of each d(sigma_i),
            # wth d() the boundary function

            for sigma_i in basis_that_generate_path_vector:
                i = 0
                while i <= path_dim:
                    aux_index = [ x != i for x in range(path_dim + 1) ]

                    # Now we will write sigma_i[ aux_index ] as
                    # a linear combination of the basis vectors of
                    # dimensionpath_dim - 1
                    # NOTATION:
                    #      aux_path := sigma_i[ aux_index ]
                    aux_path = np.zeros( self.basis_dim[ path_dim - 1 ] )
                    for j in range( self.basis_dim[ path_dim - 1 ] ):
                        if np.all( self.basis[path_dim-1][j] == sigma_i[ aux_index ]):
                            aux_path[j] = 1

                    if np.any( aux_path != 0 ):
                        distance.append( self.allow_time( aux_path, path_dim -1 ) )
                    i += 1

            return max( distance )


    def sorting_the_basis_by_their_allow_times(self):
        """
        Sorting the structures T_p and basis in agreement
        with the allow times.
        """
        for i in range( self.pph_dim + 2):
            basis_i_aux = []
            for j in range( self.basis_dim[i] ):
                basis_i_aux.append( [np.zeros( self.basis_dim[i] ), j] )
                basis_i_aux[j][0][j] = 1

            
            basis_i_aux.sort( key = lambda x: self.allow_time( x[0], i ) )

            basis_i_copy = self.basis[i].copy()

            for j in range( self.basis_dim[ i ] ):
                self.basis[i][j] = basis_i_copy[ basis_i_aux[j][1] ]


    def initialize_Marking_basis_vectors(self):
        i = 0
        while i <= self.pph_dim + 1:
            self.Marked.append( np.zeros( self.basis_dim[i]  ) )
            i += 1

        i = 0
        while i < self.basis_dim[ 0 ]:
            self.Marked[0][i] = 1
            i+=1


    def marking_vector_basis(self, vector_dim, vector_index):
        self.Marked[vector_dim][vector_index] = 1


    def generating_T_p(self):
        i = 0

        while i <= self.pph_dim + 1:
            j   = 0

            # T_i = [ v_i, et(v_i), at(v_i), mark ]
            T_i = []

            while j < self.basis_dim[ i ]:
                # writing the vector self.basis[i][j] in terms of the
                # basis
                aux = np.zeros( self.basis_dim[i] )
                #aux[j] = 1

                T_i.append( [ aux,  0, True ] )

                j += 1

            i += 1
            self.T_p.append( T_i )


    def is_T_p_dim_i_vector_j_empty(self, dim, index):
        """
        Return wheter self.T_p[dim][index] is empty or not
        """
        return self.T_p[dim][index][self.EMPTY_INDEX]


    def fill_T_p_dim_i_vecto_j(self, dim, index, u, et):
        self.T_p[dim][index][ self.ARRAY_INDEX ] = u
        self.T_p[dim][index][ self.ENTRY_INDEX ] = et
        self.T_p[dim][index][ self.EMPTY_INDEX ] = False


    def BasisChange(self, path_vector, path_dim ):
        """
        Implementing the function BasisChange as in the paper
        """

        # Calculating the vector
        #    u  = d( path_vector ),
        # where d() is the boundary transformation.
        # Here u is going to have dimension path_dim - 1.
        #
        # Here path_vector_indexes is a numpy array
        # storing the basis vectors that generate path_vector
        # and are not marked
        aux_basis = self.basis[ path_dim ][ path_vector == 1  ]
        u         = np.zeros( self.basis_dim[ path_dim - 1 ] )

        for basis in aux_basis:
            # we will calculate the boundary transformation of
            # each basis.

            i = 0
            while i <= path_dim:
                boundary_indexes =  [ x != i  for x in range( path_dim + 1 ) ]
                l = 0
                for sigma in self.basis[path_dim - 1]:
                    if np.all( sigma == basis[ boundary_indexes ] ):
                        u[ l ] = (u[l] +  1) % 2

                    l += 1

                i += 1

        # Removing unmarked terms from u (pivots)
        i = 0
        while i < self.basis_dim[ path_dim - 1 ]:
            if self.Marked[path_dim - 1 ][i] == 0:
                u[i] = 0
            i += 1


        et               = 0
        sigma_max_index  = 0 #np.arange( self.basis_dim[ path_dim - 1 ] )

        while np.any( u != 0 ):
            sigma_max_index_aux  = np.arange( self.basis_dim[ path_dim - 1 ] )
            sigma_arg_max        = np.zeros( self.basis_dim[ path_dim - 1 ] )

            sigma_max_index  = sigma_max_index_aux[ u == 1 ][-1]
            sigma_arg_max[sigma_max_index] = 1

            et = max( [self.allow_time( path_vector, path_dim ), self.allow_time( sigma_arg_max, path_dim - 1 ) ] )

            if  self.is_T_p_dim_i_vector_j_empty( path_dim - 1, sigma_max_index ) == True:
                break
            #if  self.T_p[path_dim - 1][sigma_max_index][self.ARRAY_INDEX][sigma_max_index] == 0 : break


            u = (u + self.T_p[ path_dim - 1 ][ sigma_max_index ][ self.ARRAY_INDEX ] ) % 2

        return u, sigma_max_index, et


    def ComputePPH(self ):

        # ----> Start by initializing all the enviroment needed for the calculations
        self.Basis_of_the_vector_spaces_spanned_by_regular_paths()
        self.dimensions_of_each_vector_space_spanned_by_regular_paths()
        self.sorting_the_basis_by_their_allow_times()
        self.initialize_Marking_basis_vectors()
        self.generating_T_p()

        # ----> Now start with the algorithm proposed by the paper referenced
        #       at the beginning of this file.
                 
        for i in range( self.pph_dim + 1 ):
            self.Pers.append( [] )

        for p in range( self.pph_dim + 1): # max_dimension_studied + 1
                                           # because range returns
                                           # a interval like [a,b)

            j = 0
            while j < self.basis_dim[ p + 1 ]:
                path_vector_of_basis = np.zeros( self.basis_dim[ p+1 ] )
                path_vector_of_basis[j] = 1

                u, i, et = self.BasisChange( path_vector_of_basis, p + 1 ) # following the paper's notation for these variables


                if np.all( u == 0 ):
                    self.marking_vector_basis( p + 1, j )

                else:


                    self.T_p[p][i][ self.ARRAY_INDEX ] = u
                    self.T_p[p][i][ self.ENTRY_INDEX ] = et
                    self.T_p[p][i][ self.EMPTY_INDEX ] = False

                    basis_p_i = np.zeros( self.basis_dim[p] )
                    basis_p_i[i] = 1

                    #print('checking')
                    #print( self.allow_time( path_vector_of_basis, p+1 ) == self.allow_time( basis_p_i, p ))

                    self.Pers[p].append( [self.entry_time( basis_p_i, p ), et ] )

                j += 1

            j = 0
            while j < self.basis_dim[ p ]:
                #if self.T_p[ p ][j][ self.MARK_INDEX ] == True and np.all( self.T_p[ p ][j][ self.ARRAY_INDEX ] == 0):

                #if self.T_p[ p ][j][ self.MARK_INDEX ] == True and  self.is_empty( self.T_p[ p ][j][ self.ARRAY_INDEX ], p ) == True:
                if self.Marked[ p ][j] == 1 and  self.is_T_p_dim_i_vector_j_empty( p, j ) == True:
                    basis_p_j = np.zeros( self.basis_dim[p])
                    basis_p_j[j] = 1

                    self.Pers[p].append( [self.entry_time(basis_p_j, p), np.inf] )

                j += 1


    def BasisChange_printing_step_by_step(self, path_vector, path_dim ):

        # Calculating the vector
        #    u  = d( path_vector ),
        # where d() is the boundary transformation.
        # Here u is going to have dimension path_dim - 1.

        # Here path_vector_indexes is a numpy array
        # storing the basis vectors that generate path_vector
        # and are not marked
        aux_basis = self.basis[ path_dim ][ path_vector == 1  ]
        u         = np.zeros( self.basis_dim[ path_dim - 1 ] )

        for basis in aux_basis:
            # we will calculate the boundary transformation of
            # each basis.

            i = 0
            while i <= path_dim:
                boundary_indexes =  [ x != i  for x in range( path_dim + 1 ) ]
                l = 0
                for sigma in self.basis[path_dim - 1]:
                    if np.all( sigma == basis[ boundary_indexes ] ):
                        u[ l ] = (u[l] +  1) % 2

                    l += 1

                i += 1

        self.print_vector(u, path_dim - 1, 'u = d(path_vector) ')

        # Removing unmarked terms from u (pivots)
        i = 0
        while i < self.basis_dim[ path_dim - 1 ]:
            if self.Marked[path_dim - 1 ][i] == 0:
                u[i] = 0
            i += 1

        self.print_vector(u, path_dim - 1, 'u (without unmarked terms) ')
        print('')

        et               = 0
        sigma_max_index  = 0 #np.arange( self.basis_dim[ path_dim - 1 ] )

        while np.any( u != 0 ):
            sigma_max_index_aux  = np.arange( self.basis_dim[ path_dim - 1 ] )
            sigma_arg_max        = np.zeros( self.basis_dim[ path_dim - 1 ] )

            sigma_max_index  = sigma_max_index_aux[ u == 1 ][-1]
            sigma_arg_max[sigma_max_index] = 1

            et = max( [self.allow_time( path_vector, path_dim ), self.allow_time( sigma_arg_max, path_dim - 1 ) ] )

            self.print_vector(sigma_arg_max, path_dim - 1, 'sigma_arg_max ')
            print( 'sigma_max_index = {}, et ={}'.format(sigma_max_index, et) )
            print('')

            if  self.is_T_p_dim_i_vector_j_empty( path_dim - 1, sigma_max_index ) == True:
                print('T_p[{}][{}] is empty'.format(path_dim - 1, sigma_max_index))
                break
            #if  self.T_p[path_dim - 1][sigma_max_index][self.ARRAY_INDEX][sigma_max_index] == 0 : break


            u = (u + self.T_p[ path_dim - 1 ][ sigma_max_index ][ self.ARRAY_INDEX ] ) % 2

            self.print_vector(u, path_dim - 1, 'u ')

            print('Stop with the column reduction\n')
        return u, sigma_max_index, et


    def ComputePPH_printing_step_by_step(self):

        # ----> Start by initializing all the enviroment needed for the calculations
        self.Basis_of_the_vector_spaces_spanned_by_regular_paths()
        self.dimensions_of_each_vector_space_spanned_by_regular_paths()
        self.sorting_the_basis_by_their_allow_times()
        self.initialize_Marking_basis_vectors()
        self.generating_T_p()

        print('Data info:')
        print('Network size = {}'.format(self.network_set_size))
        print('Network weight:')
        for i in range( self.network_set_size ):
            print( (self.network_set_size * '{:3.6f}   ').format( *self.network_weight.tolist()[i]) )

        print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
        print('\n')

        print('dimension of the vector spaces spanned by the regular paths:')
        for i in range( self.pph_dim + 2 ):
            str = 'regular path of dim = {}, dim vector space = {}\n'
            print( str.format(i, self.basis_dim[i]) )
        print('')
        print('')

        print('Basis')
        for i in range( self.pph_dim + 2 ):
            print( self.basis[i].tolist() )
            print('')


        # ----> Now start with the algorithm proposed by the paper referenced
        #       at the beginning of this file.

        for i in range( self.pph_dim + 1 ):
            self.Pers.append( [] )

        for p in range( self.pph_dim + 1): # max_dimension_studied + 1
                                           # because range returns
                                           # a interval like [a,b)


            j = 0
            while j < self.basis_dim[ p + 1 ]:
                str_algorithm_step = 'Algorithm step: dim = {}, index = {}\n-----------------------------------------\n'
                print( str_algorithm_step.format( p, j) )

                self.print_Marked()
                self.print_T_p()

                path_vector_of_basis = np.zeros( self.basis_dim[ p+1 ] )
                path_vector_of_basis[j] = 1

                self.print_vector(path_vector_of_basis, p + 1, 'For the basis vector: ' )
                print('This basis vetor ir referent to regular paths of dimension {}'.format(p+1))
                print('')

                print("Calculating the column reduction!")
                print('')

                # down below the variable's names (u,i,et) are following the paper's notation
                u, i, et = self.BasisChange_printing_step_by_step( path_vector_of_basis, p + 1 )

                if np.all( u == 0 ):
                    self.marking_vector_basis( p + 1, j )
                    print('Vector u is 0 thus nothing is left to be done at this iteration')
                    print('')

                else:


                    self.T_p[p][i][ self.ARRAY_INDEX ] = u
                    self.T_p[p][i][ self.ENTRY_INDEX ] = et
                    self.T_p[p][i][ self.EMPTY_INDEX ] = False

                    basis_p_i = np.zeros( self.basis_dim[p] )
                    basis_p_i[i] = 1


                    self.Pers[p].append( [self.entry_time( basis_p_i, p ), et ] )

                    print('Vector u is not 0 thus we have the interval: [{}, {}]'.format(self.entry_time( basis_p_i, p ), et))
                    print('')

                j += 1

            print('\nChecking for topological features that last up to infinity\n')
            self.print_Marked()
            self.print_T_p()

            j = 0
            while j < self.basis_dim[ p ]:
                #if self.T_p[ p ][j][ self.MARK_INDEX ] == True and np.all( self.T_p[ p ][j][ self.ARRAY_INDEX ] == 0):

                #if self.T_p[ p ][j][ self.MARK_INDEX ] == True and  self.is_empty( self.T_p[ p ][j][ self.ARRAY_INDEX ], p ) == True:
                if self.Marked[ p ][j] == 1 and  self.is_T_p_dim_i_vector_j_empty( p, j ) == True:
                    basis_p_j = np.zeros( self.basis_dim[p])
                    basis_p_j[j] = 1

                    print('Temos que Marked[{}][{}] esta marcado e T_p[{}][{}] esta vazio'.format(p,j,p,j))
                    print('Adicionar  o intervalo: [{}, {}]'.format(self.entry_time(basis_p_j, p), np.inf))
                    self.Pers[p].append( [self.entry_time(basis_p_j, p), np.inf] )

                j += 1


    def print_Marked(self):

        print('Marked elements')
        str = [ 'dim= {:<3} ->  ' + x.size * '{:<3.0f}' for x in self.Marked ]

        for i in range( self.pph_dim + 2 ):
            print( str[i].format(i, *self.Marked[i]) )

        print('\n')


    def print_T_p(self):

        print('T_p elements')
        for i in range( self.pph_dim + 2 ):
            print('dim = {}'.format(i))

            for j in range( self.basis_dim[i] ):
                if self.is_T_p_dim_i_vector_j_empty( i, j ) == False:
                    basis_elements = np.arange( self.basis_dim[i] )[ self.T_p[i][j][self.ARRAY_INDEX] == 1 ]
                    str            = (basis_elements.size - 1) * 'v[{}] + ' + 'v[{}]'

                    print( ('element {}: ' + str + ', et = {}').format( j, *basis_elements.tolist(), self.T_p[i][j][self.ENTRY_INDEX] )  )
            print('')

        print('')


    def print_vector(self, path_vector, path_dim, label_for_the_vector = 'u'):

        vector_indexes = np.arange( self.basis_dim[ path_dim ] )
        vector_indexes = vector_indexes[ path_vector == 1 ]

        if vector_indexes.size == 0:
            str = label_for_the_vector + ' = 0'
            print(str)

        else:
            str = label_for_the_vector + ' = '+ (vector_indexes.size - 1) * 'v[{}] + ' + 'v[{}]'
            print( str.format(*vector_indexes) )
