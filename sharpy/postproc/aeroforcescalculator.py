import numpy as np
import os

import sharpy.utils.cout_utils as cout
from sharpy.utils.solver_interface import solver, BaseSolver
import sharpy.utils.settings as settings
import sharpy.utils.algebra as algebra
import sharpy.aero.utils.mapping as mapping


@solver
class AeroForcesCalculator(BaseSolver):
    """AeroForcesCalculator

    Calculates the total aerodynamic forces and moments on the frame of reference ``A``.

    """
    solver_id = 'AeroForcesCalculator'
    solver_classification = 'post-processor'

    settings_types = dict()
    settings_default = dict()
    settings_description = dict()

    settings_types['write_text_file'] = 'bool'
    settings_default['write_text_file'] = False
    settings_description['write_text_file'] = 'Write ``txt`` file with results'

    settings_types['text_file_name'] = 'str'
    settings_default['text_file_name'] = 'aeroforces.txt'
    settings_description['text_file_name'] = 'Text file name'

    settings_types['screen_output'] = 'bool'
    settings_default['screen_output'] = True
    settings_description['screen_output'] = 'Show results on screen'

    settings_types['unsteady'] = 'bool'
    settings_default['unsteady'] = False
    settings_description['unsteady'] = 'Include unsteady contributions'

    settings_default['coefficients'] = False
    settings_types['coefficients'] = 'bool'
    settings_description['coefficients'] = 'Calculate aerodynamic coefficients'

    settings_types['q_ref'] = 'float'
    settings_default['q_ref'] = 1
    settings_description['q_ref'] = 'Reference dynamic pressure'

    settings_types['S_ref'] = 'float'
    settings_default['S_ref'] = 1
    settings_description['S_ref'] = 'Reference area'

    settings_types['b_ref'] = 'float'
    settings_default['b_ref'] = 1
    settings_description['b_ref'] = 'Reference span'

    settings_types['c_ref'] = 'float'
    settings_default['c_ref'] = 1
    settings_description['c_ref'] = 'Reference chord'

    settings_table = settings.SettingsTable()
    __doc__ += settings_table.generate(settings_types, settings_default, settings_description)

    def __init__(self):

        self.settings = None
        self.data = None
        self.ts_max = 0
        self.ts = 0

        self.folder = None
        self.caller = None

    def initialise(self, data, custom_settings=None, caller=None):
        self.data = data
        self.settings = data.settings[self.solver_id]
        if self.data.structure.settings['unsteady']:
            self.ts_max = self.data.ts + 1
        else:
            self.ts_max = 1
            self.ts_max = len(self.data.structure.timestep_info)
        settings.to_custom_types(self.settings, self.settings_types, self.settings_default)
        self.caller = caller

        self.folder = data.output_folder + '/forces/'
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)

    def run(self, online=False):
        self.ts = 0

        self.calculate_forces()
        if self.settings['write_text_file']:
            self.file_output(self.settings['text_file_name'])
        if self.settings['screen_output']:
            self.screen_output()
        cout.cout_wrap('...Finished', 1)
        return self.data

    def calculate_forces(self):
        for self.ts in range(self.ts_max):
            rot = algebra.quat2rotation(self.data.structure.timestep_info[self.ts].quat)

            # Forces per surface in G frame
            force = self.data.aero.timestep_info[self.ts].forces
            unsteady_force = self.data.aero.timestep_info[self.ts].dynamic_forces
            n_surf = len(force)
            for i_surf in range(n_surf):
                total_steady_force = np.zeros((3,))
                total_unsteady_force = np.zeros((3,))
                _, n_rows, n_cols = force[i_surf].shape
                for i_m in range(n_rows):
                    for i_n in range(n_cols):
                        total_steady_force += force[i_surf][0:3, i_m, i_n]
                        total_unsteady_force += unsteady_force[i_surf][0:3, i_m, i_n]
                self.data.aero.timestep_info[self.ts].inertial_steady_forces[i_surf, 0:3] = total_steady_force
                self.data.aero.timestep_info[self.ts].inertial_unsteady_forces[i_surf, 0:3] = total_unsteady_force
                self.data.aero.timestep_info[self.ts].body_steady_forces[i_surf, 0:3] = np.dot(rot.T, total_steady_force)
                self.data.aero.timestep_info[self.ts].body_unsteady_forces[i_surf, 0:3] = np.dot(rot.T, total_unsteady_force)

            # Forces expressed in the beam degrees of freedom
            try:
                steady_forces_b = self.data.aero.timestep_info[self.ts].aero_steady_forces_beam_dof
            except AttributeError:
                steady_forces_b = self.map_forces_beam_dof(self.ts, force)

            try:
                unsteady_forces_b = self.data.aero.timestep_info[self.ts].aero_unsteady_forces_beam_dof
            except AttributeError:
                unsteady_forces_b = self.map_forces_beam_dof(self.ts, unsteady_force)

            steady_forces_a = self.data.structure.timestep_info[self.ts].nodal_b_for_2_a_for(steady_forces_b,
                                                                                             self.data.structure)

            unsteady_forces_a = self.data.structure.timestep_info[self.ts].nodal_b_for_2_a_for(unsteady_forces_b,
                                                                                               self.data.structure)

            # Express total forces in A frame
            self.data.aero.timestep_info[self.ts].total_steady_body_forces = np.sum(steady_forces_a, axis=0)
            self.data.aero.timestep_info[self.ts].total_unsteady_body_forces = np.sum(unsteady_forces_a, axis=0)

            # Express total forces in G frame
            self.data.aero.timestep_info[self.ts].total_steady_inertial_forces = \
                np.block([[rot, np.zeros((3, 3))],
                          [np.zeros((3, 3)), rot]]).dot(
                    self.data.aero.timestep_info[self.ts].total_steady_body_forces)

            self.data.aero.timestep_info[self.ts].total_unsteady_inertial_forces = \
                np.block([[rot, np.zeros((3, 3))],
                          [np.zeros((3, 3)), rot]]).dot(
                    self.data.aero.timestep_info[self.ts].total_unsteady_body_forces)

    def map_forces_beam_dof(self, ts, force):
        aero_tstep = self.data.aero.timestep_info[ts]
        struct_tstep = self.data.structure.timestep_info[ts]
        aero_forces_beam_dof = mapping.aero2struct_force_mapping(force,
                                                                 self.data.aero.struct2aero_mapping,
                                                                 aero_tstep.zeta,
                                                                 struct_tstep.pos,
                                                                 struct_tstep.psi,
                                                                 None,
                                                                 self.data.structure.connectivities,
                                                                 struct_tstep.cag())
        return aero_forces_beam_dof

    def calculate_coefficients(self, fx, fy, fz, mx, my, mz):
        qS = self.settings['q_ref'] * self.settings['S_ref']
        return fx/qS, fy/qS, fz/qS, mx/qS/self.settings['b_ref'], my/qS/self.settings['c_ref'], \
               mz/qS/self.settings['b_ref']

    def screen_output(self):
        line = ''
        cout.cout_wrap.print_separator()
        # output header
        if self.settings['coefficients']:
            line = "{0:5s} | {1:10s} | {2:10s} | {3:10s} | {4:10s} | {5:10s} | {6:10s}".format(
                'tstep', '  Cfx_g', '  Cfy_g', '  Cfz_g', '  Cmx_g', '  Cmy_g', '  Cmz_g')
            cout.cout_wrap(line, 1)
        else:
            line = "{0:5s} | {1:10s} | {2:10s} | {3:10s}| {4:10s} | {5:10s} | {6:10s}".format(
                'tstep', '  fx_g', '  fy_g', '  fz_g',  '  mx_g', '  my_g', '  mz_g')
            cout.cout_wrap(line, 1)

        # print time step total aero forces
        for self.ts in range(self.ts_max):
            aero_tstep = self.data.aero.timestep_info[self.ts]
            fx, fy, fz = aero_tstep.total_steady_inertial_forces[:3] + aero_tstep.total_unsteady_inertial_forces[:3]
            mx, my, mz = aero_tstep.total_steady_inertial_forces[3:] + aero_tstep.total_unsteady_inertial_forces[3:]

            if self.settings['coefficients']:
                Cfx, Cfy, Cfz, Cmx, Cmy, Cmz = self.calculate_coefficients(fx, fy, fz, mx, my, mz)
                line = "{0:5d} | {1: 8.3e} | {2: 8.3e} | {3: 8.3e} | {4: 8.3e} | {5: 8.3e} | {6: 8.3e}".format(
                    self.ts, Cfx, Cfy, Cfz, Cmx, Cmy, Cmz)
            else:
                line = "{0:5d} | {1: 8.3e} | {2: 8.3e} | {3: 8.3e}| {1: 8.3e} | {2: 8.3e} | {3: 8.3e}".format(
                    self.ts, fx, fy, fz, mx, my, mz)

            cout.cout_wrap(line, 1)

    def file_output(self, filename):
        # assemble forces/moments matrix
        # (1 timestep) + (3+3 inertial steady+unsteady) + (3+3 body steady+unsteady)
        force_matrix = np.zeros((self.ts_max, 1 + 3 + 3 + 3 + 3 + 3 + 3))
        moment_matrix = np.zeros((self.ts_max, 1 + 3 + 3 + 3 + 3 + 3 + 3))
        for self.ts in range(self.ts_max):
            aero_tstep = self.data.aero.timestep_info[self.ts]
            i = 0
            force_matrix[self.ts, i] = self.ts
            moment_matrix[self.ts, i] = self.ts
            i += 1

            # Steady forces/moments G
            force_matrix[self.ts, i:i+3] = aero_tstep.total_steady_inertial_forces[:3]
            moment_matrix[self.ts, i:i+3] = aero_tstep.total_steady_inertial_forces[3:]
            i += 3

            # Unsteady forces/moments G
            force_matrix[self.ts, i:i+3] = aero_tstep.total_unsteady_inertial_forces[:3]
            moment_matrix[self.ts, i:i+3] = aero_tstep.total_unsteady_inertial_forces[3:]
            i += 3

            # Steady forces/moments A
            force_matrix[self.ts, i:i+3] = aero_tstep.total_steady_body_forces[:3]
            moment_matrix[self.ts, i:i+3] = aero_tstep.total_steady_body_forces[3:]
            i += 3

            # Unsteady forces/moments A
            force_matrix[self.ts, i:i+3] = aero_tstep.total_unsteady_body_forces[:3]
            moment_matrix[self.ts, i:i+3] = aero_tstep.total_unsteady_body_forces[3:]


        header = ''
        header += 'tstep, '
        header += 'fx_steady_G, fy_steady_G, fz_steady_G, '
        header += 'fx_unsteady_G, fy_unsteady_G, fz_unsteady_G, '
        header += 'fx_steady_a, fy_steady_a, fz_steady_a, '
        header += 'fx_unsteady_a, fy_unsteady_a, fz_unsteady_a'
        header += 'mx_total_G, my_total_G, mz_total_G'
        header += 'mx_total_a, my_total_a, mz_total_a'

        np.savetxt(self.folder + 'forces_' + filename,
                   force_matrix,
                   fmt='%i' + ', %10e'*18,
                   delimiter=',',
                   header=header,
                   comments='#')

        header = ''
        header += 'tstep, '
        header += 'mx_steady_G, my_steady_G, mz_steady_G, '
        header += 'mx_unsteady_G, my_unsteady_G, mz_unsteady_G, '
        header += 'mx_steady_a, my_steady_a, mz_steady_a, '
        header += 'mx_unsteady_a, my_unsteady_a, mz_unsteady_a'

        np.savetxt(self.folder + 'moments_' + filename,
                   moment_matrix,
                   fmt='%i' + ', %10e'*18,
                   delimiter=',',
                   header=header,
                   comments='#')
