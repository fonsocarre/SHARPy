import sharpy.utils.cout_utils as cout
from sharpy.presharpy.utils.settings import str2bool
from sharpy.utils.solver_interface import solver, BaseSolver
import sharpy.utils.algebra as algebra

from tvtk.api import tvtk, write_data
import numpy as np
import os


@solver
class AeroGridPlot(BaseSolver):
    solver_id = 'AeroGridPlot'
    solver_type = 'postproc'
    solver_unsteady = False

    def __init__(self):
        self.ts = 0  # steady solver
        pass

    def initialise(self, data):
        self.data = data
        self.it_max = data.grid.ts
        self.settings = data.settings[self.solver_id]
        self.convert_settings()

    def run(self):
        # create folder for containing files if necessary
        if not os.path.exists(self.settings['route']):
            os.makedirs(self.settings['route'])
        for self.ts in range(self.it_max):
            self.plot_grid()
            self.plot_wake()
        cout.cout_wrap('...Finished', 1)
        return self.data

    def convert_settings(self):
        try:
            self.settings['route'] = ((self.settings['route']))
        except KeyError:
            cout.cout_wrap('AeroGridPlot: no location for figures defined, defaulting to ./output', 3)
            self.settings['route'] = './output'
        try:
            self.settings['include_rbm'] = str2bool(self.settings['include_rbm'])
        except KeyError:
            self.settings['include_rbm'] = False

    def plot_grid(self):
        folder = self.settings['route'] + '/' + self.data.settings['SHARPy']['case'] + '/aero/'
        if not os.path.exists(folder):
            os.makedirs(folder)
        for i_surf in range(self.data.grid.timestep_info[self.ts].n_surf):
            filename = (folder +
                           'body_' +
                           self.data.settings['SHARPy']['case'] +
                           '_' +
                           '%02u_' % i_surf +
                           '%06u' % self.ts)

            dims = self.data.grid.timestep_info[self.ts].dimensions[i_surf, :]
            # dims_star = self.data.grid.timestep_info[self.ts].dimensions_star[i_surf, :]
            point_data_dim = (dims[0]+1)*(dims[1]+1)  # + (dims_star[0]+1)*(dims_star[1]+1)
            panel_data_dim = (dims[0])*(dims[1])  # + (dims_star[0])*(dims_star[1])

            coords = np.zeros((point_data_dim, 3))
            conn = []
            panel_id = np.zeros((panel_data_dim,), dtype=int)
            panel_surf_id = np.zeros((panel_data_dim,), dtype=int)
            panel_gamma = np.zeros((panel_data_dim,))
            normal = np.zeros((panel_data_dim, 3))
            point_struct_id = np.zeros((point_data_dim,), dtype=int)
            point_cf = np.zeros((point_data_dim, 3))
            point_unsteady_cf = np.zeros((point_data_dim, 3))
            counter = -1

            rotation_mat = algebra.euler2rot(self.data.beam.timestep_info[self.ts].for_pos[3:6])
            # coordinates of corners
            for i_n in range(dims[1]+1):
                for i_m in range(dims[0]+1):
                    counter += 1
                    coords[counter, :] = self.data.grid.timestep_info[self.ts].zeta[i_surf][:, i_m, i_n]
                    if self.settings['include_rbm']:
                        coords[counter, :] = np.dot(rotation_mat, self.data.grid.timestep_info[self.ts].zeta[i_surf][:, i_m, i_n])
                        coords[counter, :] += self.data.beam.timestep_info[self.ts].for_pos[0:3]
            # for i_n in range(dims_star[1]+1):
            #     for i_m in range(dims_star[0]+1):
            #         counter += 1
            #         coords[counter, :] = self.data.grid.timestep_info[self.ts].zeta_star[i_surf][:, i_m, i_n]

            counter = -1
            node_counter = -1
            for i_n in range(dims[1] + 1):
                global_counter = self.data.grid.aero2struct_mapping[i_surf][i_n]
                for i_m in range(dims[0] + 1):
                    node_counter += 1
                    # point data
                    point_struct_id[node_counter] = global_counter
                    point_cf[node_counter, :] = self.data.grid.timestep_info[self.ts].forces[i_surf][0:3, i_m, i_n]
                    try:
                        point_unsteady_cf[node_counter, :] = self.data.grid.timestep_info[self.ts].dynamic_forces[i_surf][0:3, i_m, i_n]
                    except AttributeError:
                        pass
                    if i_n < dims[1] and i_m < dims[0]:
                        counter += 1
                    else:
                        continue

                    conn.append([node_counter + 0,
                                 node_counter + 1,
                                 node_counter + dims[0]+2,
                                 node_counter + dims[0]+1])
                    # cell data
                    normal[counter, :] = self.data.grid.timestep_info[self.ts].normals[i_surf][:, i_m, i_n]
                    panel_id[counter] = counter
                    panel_surf_id[counter] = i_surf
                    panel_gamma[counter] = self.data.grid.timestep_info[self.ts].gamma[i_surf][i_m, i_n]

            ug = tvtk.UnstructuredGrid(points=coords)
            ug.set_cells(tvtk.Quad().cell_type, conn)
            ug.cell_data.scalars = panel_id
            ug.cell_data.scalars.name = 'panel_n_id'
            ug.cell_data.add_array(panel_surf_id)
            ug.cell_data.get_array(1).name = 'panel_surface_id'
            ug.cell_data.add_array(panel_gamma)
            ug.cell_data.get_array(2).name = 'panel_gamma'
            ug.cell_data.vectors = normal
            ug.cell_data.vectors.name = 'panel_normal'
            ug.point_data.scalars = np.arange(0, coords.shape[0])
            ug.point_data.scalars.name = 'n_id'
            ug.point_data.add_array(point_struct_id)
            ug.point_data.get_array(1).name = 'point_struct_id'
            ug.point_data.add_array(point_cf)
            ug.point_data.get_array(2).name = 'point_steady_force'
            ug.point_data.add_array(point_unsteady_cf)
            ug.point_data.get_array(3).name = 'point_unsteady_force'
            write_data(ug, filename)
        pass

    def plot_wake(self):
        folder = self.settings['route'] + '/' + self.data.settings['SHARPy']['case'] + '/aero/'
        if not os.path.exists(folder):
            os.makedirs(folder)
        for i_surf in range(self.data.grid.timestep_info[self.ts].n_surf):
            filename = (folder +
                        'wake_' +
                        self.data.settings['SHARPy']['case'] +
                        '_' +
                        '%02u_' % i_surf +
                        '%06u' % self.ts)

            dims_star = self.data.grid.timestep_info[self.ts].dimensions_star[i_surf, :]
            point_data_dim = (dims_star[0]+1)*(dims_star[1]+1)
            panel_data_dim = (dims_star[0])*(dims_star[1])

            coords = np.zeros((point_data_dim, 3))
            conn = []
            panel_id = np.zeros((panel_data_dim,), dtype=int)
            panel_surf_id = np.zeros((panel_data_dim,), dtype=int)
            panel_gamma = np.zeros((panel_data_dim,))
            counter = -1
            rotation_mat = algebra.euler2rot(self.data.beam.timestep_info[self.ts].for_pos[3:6])
            # coordinates of corners
            for i_n in range(dims_star[1]+1):
                for i_m in range(dims_star[0]+1):
                    counter += 1
                    coords[counter, :] = self.data.grid.timestep_info[self.ts].zeta_star[i_surf][:, i_m, i_n]
                    if self.settings['include_rbm']:
                        coords[counter, :] = np.dot(rotation_mat, coords[counter, :])
                        coords[counter, :] += self.data.beam.timestep_info[self.ts].for_pos[0:3]

            counter = -1
            node_counter = -1
            # wake
            for i_n in range(dims_star[1]+1):
                for i_m in range(dims_star[0]+1):
                    node_counter += 1
                    # cell data
                    if i_n < dims_star[1] and i_m < dims_star[0]:
                        counter += 1
                    else:
                        continue

                    conn.append([node_counter + 0,
                                 node_counter + 1,
                                 node_counter + dims_star[0]+2,
                                 node_counter + dims_star[0]+1])
                    panel_id[counter] = counter
                    panel_surf_id[counter] = i_surf
                    panel_gamma[counter] = self.data.grid.timestep_info[self.ts].gamma_star[i_surf][i_m, i_n]

            ug = tvtk.UnstructuredGrid(points=coords)
            ug.set_cells(tvtk.Quad().cell_type, conn)
            ug.cell_data.scalars = panel_id
            ug.cell_data.scalars.name = 'panel_n_id'
            ug.cell_data.add_array(panel_surf_id)
            ug.cell_data.get_array(1).name = 'panel_surface_id'
            ug.cell_data.add_array(panel_gamma)
            ug.cell_data.get_array(2).name = 'panel_gamma'
            ug.point_data.scalars = np.arange(0, coords.shape[0])
            ug.point_data.scalars.name = 'n_id'
            write_data(ug, filename)
