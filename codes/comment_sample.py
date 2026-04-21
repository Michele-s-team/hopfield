'''
this module contains methods for mesh generation
'''

import colorama as col
import command as cmd
from fenics import *
import gmsh
import math
import meshio
import numpy as np
import os
import pygmsh
import shutil
import sys
import ufl

'''
set field defined on a DG space equal to a profile in a mesh region 
Input values: 
    * Mandatory:
        - 'f': the field defined on a DG space
        - 'g': the profile to which 'f' will be set
        - 'sf': the mesh function that tags mesh surfaces
    * Optional:
        - 'region_id': 'None' by default, the id of the mesh region (surface) on which 'f' will be set equal to 'g'. If 'id' is 'None' then this method will run through all cells in the mesh, 'f' to 'g' on the cell DOFs
'''
def interpolate_dg(f, g, sf, region_id=None):

    Q = f.function_space()
    element = Q.ufl_element()

    if (element.family() != 'Discontinuous Lagrange'):
        # the method has been called on a field defined on a continuous function space -> throw an error and exit

        print(f'{col.Fore.RED}Error: interpolate_dg has been called on a field with a continuous function space!! Stopping now.{col.Fore.RESET}')

        sys.exit(1)

    if (element.value_shape() != g.value_shape()):
        # the value shape of Q and that of g differ -> check whether this is due to a 'convention' issue where scalars have been given a value shape of (1,) vs. ()

        if ((((element.value_shape() == ()) and (g.value_shape() == (1,))) or ((element.value_shape() == (1,)) and  (g.value_shape() == ()))) == False):
            # the discrepancy was not due to a convention issue -> throw an error and exit

            print(f'{col.Fore.RED}Error: value shapes are different!!\n\telement value shape = {element.value_shape()}\n\tg value shape= {g.value_shape()}\nStopping now.{col.Fore.RESET}')

            sys.exit(1)
   
    value_size  = int(np.prod(element.value_shape())) if element.value_shape() else 1

    mesh = Q.mesh()

    '''
    dof_coordinates stores the coordinates of the points where DOFs sit. Because the field 'f' defined on each DOF has value_size components, dof_coordinates is composed of blocks, where each block has 'value_size' entries, and blocks are all identical
    For example, dof_coordinates is of the form ->
        row 0:  [x0, y0]   ← this corresponds to f[0] at DOF point 0
        row 1:  [x0, y0]   ← this corresponds to f[1] at DOF point 0
        ...
        row value_size  [x1, y1]   ← this corresponds to f[0] at DOF point 1
        row value_size+1 [x1, y1]   ← this corresponds to f[1] at DOF point 1
        ...
        
    '''
    dof_coordinates = Q.tabulate_dof_coordinates()


    '''
    f_values contains the value of 'f' on the DOFs, and it has the same structure as 'dof_coordinates'
    f_values = 
            entry 0: f[0] at DOF point 0
            entry 1: f[1] at DOF point 0
            ...
            entry value_size  f[0] at DOF point 1
            entry value_size + 1 f[1] at DOF point 1
            ...
        
    '''
    f_values = f.vector().get_local()   # get a copy of field values
    
    for cell in cells(mesh):
        # run on all mesh cells

        # compute 'sf' on the cell; under consideration
        cell_tag = sf[cell]

        '''
        cell_dofs contains the IDs of the DOFs that are contained into 'cell', it has the structure
        [
            id_f_0_on_DOF_0, 
            id_f_0_on_DOF_1,
            ...,
            id_f_0_on_DOF_{n_nodes-1},

            id_f_1_on_DOF_0, 
            id_f_1_on_DOF_1,
            ...,
            id_f_1_on_DOF_{n_nodes-1},

            ...
        ]
        where the pattern is repeated value_size times, i.e., one for each component of 'f', and n_nodes = [number of DOFs in the cell] / [value_size]. In other words

        cell_dofs[j * n_nodes + i] = [index in f.values().get_local() corresponding to the j-th component of the field 'f' sitting on ith DOF in the cell 'cell']
        '''
        cell_dofs = Q.dofmap().cell_dofs(cell.index())

        n_nodes = len(cell_dofs) // value_size


        '''
        remove the redundancy in cell_dofs and store the result in 
        cell_dofs_unique = 
            [
                id_DOF_0, 
                id_DOF_1,
                ...,
                id_DOF_{n_nodes-1}
            ]
        
        '''
        cell_dofs_unique = cell_dofs[:n_nodes]


        if (cell_tag == region_id) or (region_id == None):
            # if 'cell_tag' == 'id', then the cell under consideration belongs to the surface tagged with 'id' -> set the DOFs of 'f' relative to this cell according to 'g'

            for i in range(len(cell_dofs_unique)):
                # run over physical DOFs contained to 'cell' and print out the value of 'f' by specifying that those DOFs belong to region tagged with 'cell_tag' in a separate column of the csv output file. Note that, because the space of 'f' is discontinuous, here DOFs in 'cell' may belong to different mesh regions, and thus have different tags

                # pad 'x' to three dimensions
                dof_coordinate = dof_coordinates[cell_dofs_unique[i]]

                for j in range(value_size):
                    f_values[cell_dofs[j * n_nodes + i]] = np.atleast_1d(g(dof_coordinate[:2]))[j]


    f.vector().set_local(f_values) 
    f.vector().apply("insert")
   
