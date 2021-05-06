import ctypes as ct
import numpy as np

import sharpy.utils.algebra as algebra
import sharpy.aero.utils.uvlmlib as uvlmlib
import sharpy.utils.cout_utils as cout
import sharpy.utils.settings as settings
from sharpy.utils.solver_interface import solver, BaseSolver
import sharpy.utils.generator_interface as gen_interface
from sharpy.utils.constants import vortex_radius_def


@solver
class StaticUvlm(BaseSolver):
    """
    ``StaticUvlm`` solver class, inherited from ``BaseSolver``

    Aerodynamic solver that runs a UVLM routine to solve the steady or unsteady aerodynamic problem.
    The aerodynamic problem is posed in the form of an ``Aerogrid`` object.

    Args:
        data (PreSharpy): object with problem data
        custom_settings (dict): custom settings that override the settings in the solver ``.txt`` file. None by default

    Attributes:
        settings (dict): Name-value pair of settings employed by solver. See Notes for valid combinations
        settings_types (dict): Acceptable data types for entries in ``settings``
        settings_default (dict): Default values for the available ``settings``
        data (PreSharpy): object containing the information of the problem
        velocity_generator(object): object containing the flow conditions information


    """
    solver_id = 'StaticUvlm'
    solver_classification = 'aero'

    settings_types = dict()
    settings_default = dict()
    settings_description = dict()

    settings_types['print_info'] = 'bool'
    settings_default['print_info'] = True
    settings_description['print_info'] = 'Print info to screen'

    settings_types['horseshoe'] = 'bool'
    settings_default['horseshoe'] = False
    settings_description['horseshoe'] = 'Horseshoe wake modelling for steady simulations.'

    settings_types['nonlifting_body_interactions'] = 'bool'
    settings_default['nonlifting_body_interactions'] = False
    settings_description['nonlifting_body_interactions'] = 'Consider nonlifting body interactions'

    settings_types['only_nonlifting'] = 'bool'
    settings_default['only_nonlifting'] = False
    settings_description['only_nonlifting'] = 'Consider only nonlifting bodies'
    settings_types['num_cores'] = 'int'
    settings_default['num_cores'] = 0
    settings_description['num_cores'] = 'Number of cores to use in the VLM lib'

    settings_types['n_rollup'] = 'int'
    settings_default['n_rollup'] = 1
    settings_description['n_rollup'] = 'Number of rollup iterations for free wake. Use at least ``n_rollup > 1.1*m_star``'

    settings_types['rollup_dt'] = 'float'
    settings_default['rollup_dt'] = 0.1
    settings_description['rollup_dt'] = 'Pseudo time step for wake convection. Chose it so that it is similar to the unsteady time step'

    settings_types['rollup_aic_refresh'] = 'int'
    settings_default['rollup_aic_refresh'] = 1
    settings_description['rollup_dt'] = 'Controls when the AIC matrix is refreshed during the wake rollup'

    settings_types['rollup_tolerance'] = 'float'
    settings_default['rollup_tolerance'] = 1e-4
    settings_description['rollup_tolerance'] = 'Convergence criterium for rollup wake'

    settings_types['iterative_solver'] = 'bool'
    settings_default['iterative_solver'] = False
    settings_description['iterative_solver'] = 'Not in use'

    settings_types['iterative_tol'] = 'float'
    settings_default['iterative_tol'] = 1e-4
    settings_description['iterative_tol'] = 'Not in use'

    settings_types['iterative_precond'] = 'bool'
    settings_default['iterative_precond'] = False
    settings_description['iterative_precond'] = 'Not in use'

    settings_types['velocity_field_generator'] = 'str'
    settings_default['velocity_field_generator'] = 'SteadyVelocityField'
    settings_description['velocity_field_generator'] = 'Name of the velocity field generator to be used in the simulation'

    settings_types['velocity_field_input'] = 'dict'
    settings_default['velocity_field_input'] = {}
    settings_description['velocity_field_input'] = 'Dictionary of settings for the velocity field generator'

    settings_types['rho'] = 'float'
    settings_default['rho'] = 1.225
    settings_description['rho'] = 'Air density'

    settings_types['cfl1'] = 'bool'
    settings_default['cfl1'] = True
    settings_description['cfl1'] = 'If it is ``True``, it assumes that the discretisation complies with CFL=1'

    settings_types['vortex_radius'] = 'float'
    settings_default['vortex_radius'] = vortex_radius_def
    settings_description['vortex_radius'] = 'Distance between points below which induction is not computed'

    settings_types['vortex_radius_wake_ind'] = 'float'
    settings_default['vortex_radius_wake_ind'] = vortex_radius_def
    settings_description['vortex_radius_wake_ind'] = 'Distance between points below which induction is not computed in the wake convection'

    settings_types['rbm_vel_g'] = 'list(float)'
    settings_default['rbm_vel_g'] = [0., 0., 0., 0., 0., 0.]
    settings_description['rbm_vel_g'] = 'Rigid body velocity in G FoR'

    settings_types['centre_rot_g'] = 'list(float)'
    settings_default['centre_rot_g'] = [0., 0., 0.]
    settings_description['centre_rot_g'] = 'Centre of rotation in G FoR around which ``rbm_vel_g`` is applied'

    settings_table = settings.SettingsTable()
    __doc__ += settings_table.generate(settings_types, settings_default, settings_description)

    def __init__(self):
        # settings list
        self.data = None
        self.settings = None
        self.velocity_generator = None

    def initialise(self, data, custom_settings=None):
        self.data = data
        if custom_settings is None:
            self.settings = data.settings[self.solver_id]
        else:
            self.settings = custom_settings
        settings.to_custom_types(self.settings, self.settings_types, self.settings_default)

        self.update_step()

        # init velocity generator
        velocity_generator_type = gen_interface.generator_from_string(
            self.settings['velocity_field_generator'])
        self.velocity_generator = velocity_generator_type()
        self.velocity_generator.initialise(self.settings['velocity_field_input'])

    def run(self):
        if not self.data.aero.timestep_info[self.data.ts].zeta:
            return self.data

        # generate the wake because the solid shape might change
        aero_tstep = self.data.aero.timestep_info[self.data.ts]
        self.data.aero.wake_shape_generator.generate({'zeta': aero_tstep.zeta,
                                            'zeta_star': aero_tstep.zeta_star,
                                            'gamma': aero_tstep.gamma,
                                            'gamma_star': aero_tstep.gamma_star,
                                            'dist_to_orig': aero_tstep.dist_to_orig})

        # check if nonlifting body interactions have to be considered
        if not self.settings['nonlifting_body_interactions']:
            self.velocity_generator.generate({'zeta': self.data.nonlifting_body.timestep_info[self.data.ts].zeta,
                                            'override': True,
                                            'for_pos': self.data.structure.timestep_info[self.data.ts].for_pos[0:3]},
                                            self.data.nonlifting_body.timestep_info[self.data.ts].u_ext)
            uvlmlib.vlm_solver_nonlifting_body(self.data.nonlifting_body.timestep_info[self.data.ts],
                                            self.settings)
        elif self.settings['nonlifting_body_interactions']:
            # generate uext
            self.velocity_generator.generate({'zeta': self.data.nonlifting_body.timestep_info[self.data.ts].zeta,
                                            'override': True,
                                            'for_pos': self.data.structure.timestep_info[self.data.ts].for_pos[0:3]},
                                            self.data.nonlifting_body.timestep_info[self.data.ts].u_ext)
            # generate uext
            self.velocity_generator.generate({'zeta': self.data.aero.timestep_info[self.data.ts].zeta,
                                            'override': True,
                                            'for_pos': self.data.structure.timestep_info[self.data.ts].for_pos[0:3]},  
                                            self.data.aero.timestep_info[self.data.ts].u_ext)
            # grid orientation
            uvlmlib.vlm_solver_lifting_and_nonlifting_bodies(self.data.aero.timestep_info[self.data.ts],
                                                            self.data.nonlifting_body.timestep_info[self.data.ts],
                                                            self.settings)
        else:        
            # generate uext
            self.velocity_generator.generate({'zeta': self.data.aero.timestep_info[self.data.ts].zeta,
                                              'override': True,
                                              'for_pos': self.data.structure.timestep_info[self.data.ts].for_pos[0:3]},
                                             self.data.aero.timestep_info[self.data.ts].u_ext)

            # grid orientation
            uvlmlib.vlm_solver(self.data.aero.timestep_info[self.data.ts],
                               self.settings)

        return self.data

    def next_step(self):
        """ Updates de aerogrid based on the info of the step, and increases
        the self.ts counter """
        self.data.aero.add_timestep()
        if self.settings['nonlifting_body_interactions']:
            self.data.nonlifting_body.add_timestep()
        self.update_step()

    def update_step(self):
        self.data.aero.generate_zeta(self.data.structure,
                                     self.data.aero.aero_settings,
                                     self.data.ts)
        # for i_surf in range(self.data.aero.timestep_info[self.data.ts].n_surf):
        #     self.data.aero.timestep_info[self.data.ts].forces[i_surf].fill(0.0)
        #     self.data.aero.timestep_info[self.data.ts].dynamic_forces[i_surf].fill(0.0)
        if self.settings['nonlifting_body_interactions']:
            self.data.nonlifting_body.generate_zeta(self.data.structure,
                                                    self.data.aero.aero_settings,
                                                    self.data.ts)
