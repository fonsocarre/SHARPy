import numpy as np
import os

import sharpy.utils.generate_cases as gc
import cases.templates.template_wt as template_wt
from sharpy.utils.constants import deg2rad

######################################################################
###########################  PARAMETERS  #############################
######################################################################
# Case
case = 'rotor'
route = os.path.dirname(os.path.realpath(__file__)) + '/'

# Geometry discretization
chord_panels = np.array([8], dtype=int)
revs_in_wake = 1

# Operation
rotation_velocity = 1.366190
pitch_deg = 0. #degrees

# Wind
WSP = 11.4
air_density = 1.225

# Simulation
dphi = 4.*deg2rad
revs_to_simulate = 5

######################################################################
##########################  GENERATE WT  #############################
######################################################################
dt = dphi/rotation_velocity
# time_steps = int(revs_to_simulate*2.*np.pi/dphi)
time_steps = 2 # For the test cases

mstar = int(revs_in_wake*2.*np.pi/dphi)

rotor = template_wt.rotor_from_excel_type02(
                                  chord_panels,
                                  rotation_velocity,
                                  pitch_deg,
                                  excel_file_name = route + '../../../docs/source/content/example_notebooks/source/type02_db_NREL5MW_v02.xlsx',
                                  excel_sheet_parameters = 'parameters',
                                  excel_sheet_structural_blade = 'structural_blade',
                                  excel_sheet_discretization_blade = 'discretization_blade',
                                  excel_sheet_aero_blade = 'aero_blade',
                                  excel_sheet_airfoil_info = 'airfoil_info',
                                  excel_sheet_airfoil_coord = 'airfoil_coord',
                                  m_distribution = 'uniform',
                                  n_points_camber = 100,
                                  tol_remove_points = 1e-8)

######################################################################
######################  DEFINE SIMULATION  ###########################
######################################################################
SimInfo = gc.SimulationInformation()
SimInfo.set_default_values()

SimInfo.solvers['SHARPy']['flow'] = ['BeamLoader',
                        'AerogridLoader',
                        'StaticCoupledRBM',
                        'BeamPlot',
                        'AerogridPlot',
                        'DynamicCoupled',
                        'SaveData']
SimInfo.solvers['SHARPy']['case'] = case
SimInfo.solvers['SHARPy']['route'] = route
SimInfo.solvers['SHARPy']['write_log'] = True
SimInfo.set_variable_all_dicts('dt', dt)
SimInfo.set_variable_all_dicts('rho', air_density)

SimInfo.solvers['SteadyVelocityField']['u_inf'] = WSP
SimInfo.solvers['SteadyVelocityField']['u_inf_direction'] = np.array([0., 0., 1.])
SimInfo.set_variable_all_dicts('velocity_field_input', SimInfo.solvers['SteadyVelocityField'])

SimInfo.solvers['BeamLoader']['unsteady'] = 'on'

SimInfo.solvers['AerogridLoader']['unsteady'] = 'on'
SimInfo.solvers['AerogridLoader']['mstar'] = mstar
SimInfo.solvers['AerogridLoader']['freestream_dir'] = np.array([0.,0.,0.])

SimInfo.solvers['NonLinearStatic']['max_iterations'] = 200
SimInfo.solvers['NonLinearStatic']['num_load_steps'] = 1
SimInfo.solvers['NonLinearStatic']['min_delta'] = 1e-5

SimInfo.solvers['StaticCoupledRBM']['structural_solver'] = 'RigidDynamicPrescribedStep'
SimInfo.solvers['StaticCoupledRBM']['structural_solver_settings'] = SimInfo.solvers['RigidDynamicPrescribedStep']
SimInfo.solvers['StaticCoupledRBM']['structural_solver'] = 'NonLinearDynamicPrescribedStep'
SimInfo.solvers['StaticCoupledRBM']['structural_solver_settings'] = SimInfo.solvers['NonLinearDynamicPrescribedStep']
SimInfo.solvers['StaticCoupledRBM']['structural_solver'] = 'NonLinearStatic'
SimInfo.solvers['StaticCoupledRBM']['structural_solver_settings'] = SimInfo.solvers['NonLinearStatic']
SimInfo.solvers['StaticCoupledRBM']['aero_solver'] = 'SHWUvlm'
SimInfo.solvers['StaticCoupledRBM']['aero_solver_settings'] = SimInfo.solvers['SHWUvlm']

SimInfo.solvers['StaticCoupledRBM']['tolerance'] = 1e-6
SimInfo.solvers['StaticCoupledRBM']['n_load_steps'] = 0
SimInfo.solvers['StaticCoupledRBM']['relaxation_factor'] = 0.

SimInfo.solvers['SHWUvlm']['convection_scheme'] = 2
SimInfo.solvers['SHWUvlm']['num_cores'] = 15
SimInfo.solvers['SHWUvlm']['rot_vel'] = rotation_velocity
SimInfo.solvers['SHWUvlm']['rot_axis'] = np.array([0.,0.,1.])
SimInfo.solvers['SHWUvlm']['rot_center'] = np.zeros((3),)

# SimInfo.solvers['StaticCoupledRBM']['newmark_damp'] = 0.1
SimInfo.solvers['StepUvlm']['convection_scheme'] = 2
SimInfo.solvers['StepUvlm']['num_cores'] = 15

SimInfo.solvers['WriteVariablesTime']['FoR_variables'] = ['total_forces',]
SimInfo.solvers['WriteVariablesTime']['FoR_number'] = [0,]

# SimInfo.solvers['DynamicCoupled']['structural_solver'] = 'NonLinearDynamicMultibody'
# SimInfo.solvers['DynamicCoupled']['structural_solver_settings'] = SimInfo.solvers['NonLinearDynamicMultibody']
SimInfo.solvers['DynamicCoupled']['structural_solver'] = 'RigidDynamicPrescribedStep'
SimInfo.solvers['DynamicCoupled']['structural_solver_settings'] = SimInfo.solvers['RigidDynamicPrescribedStep']
SimInfo.solvers['DynamicCoupled']['aero_solver'] = 'StepUvlm'
SimInfo.solvers['DynamicCoupled']['aero_solver_settings'] = SimInfo.solvers['StepUvlm']
SimInfo.solvers['DynamicCoupled']['postprocessors'] = ['BeamPlot', 'AerogridPlot', 'WriteVariablesTime', 'Cleanup']
SimInfo.solvers['DynamicCoupled']['postprocessors_settings'] = {'BeamPlot': SimInfo.solvers['BeamPlot'],
                                                             'AerogridPlot': SimInfo.solvers['AerogridPlot'],
                                                             'WriteVariablesTime': SimInfo.solvers['WriteVariablesTime'],
                                                             'Cleanup': SimInfo.solvers['Cleanup']}
SimInfo.solvers['DynamicCoupled']['minimum_steps'] = 0

SimInfo.define_num_steps(time_steps)

# Define dynamic simulation
SimInfo.with_forced_vel = True
SimInfo.for_vel = np.zeros((time_steps,6), dtype=float)
SimInfo.for_vel[:,5] = rotation_velocity
SimInfo.for_acc = np.zeros((time_steps,6), dtype=float)
SimInfo.with_dynamic_forces = True
SimInfo.dynamic_forces = np.zeros((time_steps,rotor.StructuralInformation.num_node,6), dtype=float)


######################################################################
#######################  GENERATE FILES  #############################
######################################################################
gc.clean_test_files(SimInfo.solvers['SHARPy']['route'], SimInfo.solvers['SHARPy']['case'])
rotor.generate_h5_files(SimInfo.solvers['SHARPy']['route'], SimInfo.solvers['SHARPy']['case'])
SimInfo.generate_solver_file()
SimInfo.generate_dyn_file(time_steps)
