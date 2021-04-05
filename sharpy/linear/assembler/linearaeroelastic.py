import numpy as np
import scipy.linalg as sclalg
import warnings

import sharpy.linear.utils.ss_interface as ss_interface
import sharpy.linear.src.libss as libss
import sharpy.utils.settings as settings
import sharpy.utils.cout_utils as cout
import sharpy.utils.algebra as algebra
import sharpy.utils.generator_interface as gi
import sharpy.aero.utils.utils as aero_utils


@ss_interface.linear_system
class LinearAeroelastic(ss_interface.BaseElement):
    r"""
    Assemble a linearised aeroelastic system

    The aeroelastic system can be seen as the coupling between a linearised aerodynamic system (System 1) and
    a linearised beam system (System 2).

    The coupled system retains inputs and outputs from both systems such that

    .. math:: \mathbf{u} = [\mathbf{u}_1;\, \mathbf{u}_2]

    and the outputs are also ordered in a similar fashion

    .. math:: \mathbf{y} = [\mathbf{y}_1;\, \mathbf{y}_2]

    Reference the individual systems for the particular ordering of the respective input and output variables.
    """
    sys_id = 'LinearAeroelastic'

    settings_default = dict()
    settings_types = dict()
    settings_description = dict()

    settings_types['aero_settings'] = 'dict'
    settings_default['aero_settings'] = None
    settings_description['aero_settings'] = 'Linear UVLM settings'

    settings_types['beam_settings'] = 'dict'
    settings_default['beam_settings'] = None
    settings_description['beam_settings'] = 'Linear Beam settings'

    settings_types['uvlm_filename'] = 'str'
    settings_default['uvlm_filename'] = ''
    settings_description['uvlm_filename'] = 'Path to .data.h5 file containing UVLM/ROM state space to load'

    settings_types['track_body'] = 'bool'
    settings_default['track_body'] = True
    settings_description['track_body'] = 'UVLM inputs and outputs projected to coincide with lattice at linearisation'

    settings_types['use_euler'] = 'bool'
    settings_default['use_euler'] = False
    settings_description['use_euler'] = 'Parametrise orientations in terms of Euler angles'

    settings_table = settings.SettingsTable()
    __doc__ += settings_table.generate(settings_types, settings_default, settings_description)

    def __init__(self):

        self.ss = None  # The state space object
        self.lsys = dict()  # Contains the underlying objects
        self.uvlm = None
        self.beam = None

        self.load_uvlm_from_file = False

        self.settings = dict()
        self.state_variables = None
        self.couplings = dict()
        self.linearisation_vectors = dict()

        # Aeroelastic coupling gains
        # transfer
        self.Kdisp = None
        self.Kvel_disp = None
        self.Kdisp_vel = None
        self.Kvel_vel = None
        self.Kforces = None

        # stiffening factors
        self.Kss = None
        self.Krs = None
        self.Csr = None
        self.Crs = None
        self.Crr = None

    def initialise(self, data):

        try:
            self.settings = data.settings['LinearAssembler']['linear_system_settings']
        except KeyError:
            self.settings = None
        settings.to_custom_types(self.settings, self.settings_types, self.settings_default, no_ctype=True)

        if self.settings['use_euler']:
            self.settings['beam_settings']['use_euler'] = True

        # Create Linear UVLM
        self.uvlm = ss_interface.initialise_system('LinearUVLM')
        self.uvlm.initialise(data, custom_settings=self.settings['aero_settings'])

        # Get the minimum parameters needed to define the wake
        vel_gen_name, vel_gen_settings = aero_utils.find_velocity_generator(data.settings)
        vel_gen_type = gi.generator_from_string(vel_gen_name)
        vel_gen = vel_gen_type()
        vel_gen.initialise(vel_gen_settings) 

        wake_prop_settings = {'dt': self.settings['aero_settings']['dt'],
                              'ts': data.ts,
                              't': data.ts*self.settings['aero_settings']['dt'],
                              'for_pos': data.structure.timestep_info[-1].for_pos,
                              'cfl1': self.settings['aero_settings']['cfl1'],
                              'vel_gen': vel_gen}

        if self.settings['uvlm_filename'] == '':
            self.uvlm.assemble(track_body=self.settings['track_body'], wake_prop_settings=wake_prop_settings)
        else:
            self.load_uvlm_from_file = True

        # Create beam
        self.beam = ss_interface.initialise_system('LinearBeam')
        self.beam.initialise(data, custom_settings=self.settings['beam_settings'])

        for k, v in self.uvlm.linearisation_vectors.items():
            self.linearisation_vectors[k] = v
        for k, v in self.beam.linearisation_vectors.items():
            self.linearisation_vectors[k] = v

        self.get_gebm2uvlm_gains(data)

    def assemble(self):
        r"""
        Assembly of the linearised aeroelastic system.

        The UVLM state-space system has already been assembled. Prior to assembling the beam's first order state-space,
        the damping and stiffness matrices have to be modified to include the damping and stiffenning terms that arise
        from the linearisation of the aeordynamic forces with respect to the A frame of reference. See
        :func:`sharpy.linear.src.lin_aeroela.get_gebm2uvlm_gains()` for details on the linearisation.

        Then the beam is assembled as per the given settings in normalised time if the aerodynamic system has been
        scaled. The discrete time systems of the UVLM and the beam must have the same time step.

        The UVLM inputs and outputs are then projected onto the structural degrees of freedom (obviously with the
        exception of external gusts and control surfaces). Hence, the gains :math:`\mathbf{K}_{sa}` and
        :math:`\mathbf{K}_{as}` are added to the output and input of the UVLM system, respectively. These gains perform
        the following relation:

        .. math:: \begin{bmatrix}\zeta \\ \zeta' \\ u_g \\ \delta \end{bmatrix} = \mathbf{K}_{as}
            \begin{bmatrix} \eta \\ \eta' \\ u_g \\ \delta \end{bmatrix} =

        .. math:: \mathbf{N}_{nodes} = \mathbf{K}_{sa} \mathbf{f}_{vertices}

        If the beam is expressed in modal form, the UVLM is further projected onto the beam's modes to have the
        following input/output structure:


        Returns:

        """
        uvlm = self.uvlm
        beam = self.beam

        # Linearisation of the aerodynamic forces introduces stiffenning and damping terms into the beam matrices
        flex_nodes = self.beam.sys.num_dof_flex

        rigid_dof = beam.sys.Kstr.shape[0] - flex_nodes
        total_dof = flex_nodes + rigid_dof

        if uvlm.scaled:
            beam.assemble(t_ref=uvlm.sys.ScalingFacts['time'])
        else:
            beam.assemble()

        if not self.load_uvlm_from_file:
            # Projecting the UVLM inputs and outputs onto the structural degrees of freedom
            Ksa = self.Kforces[:beam.sys.num_dof, :]  # maps aerodynamic grid forces to nodal forces

            # Map the nodal displacement and velocities onto the grid displacements and velocities
            Kas = np.zeros((uvlm.ss.inputs, 2*beam.sys.num_dof + (uvlm.ss.inputs - 2*self.Kdisp.shape[0])))
            Kas[:2*self.Kdisp.shape[0], :2*beam.sys.num_dof] = \
                np.block([[self.Kdisp[:, :beam.sys.num_dof], self.Kdisp_vel[:, :beam.sys.num_dof]],
                [self.Kvel_disp[:, :beam.sys.num_dof], self.Kvel_vel[:, :beam.sys.num_dof]]])

            # Retain other inputs
            Kas[2*self.Kdisp.shape[0]:, 2*beam.sys.num_dof:] = np.eye(uvlm.ss.inputs - 2 * self.Kdisp.shape[0])

            # Scaling
            if uvlm.scaled:
                Kas /= uvlm.sys.ScalingFacts['length']

            uvlm.ss.addGain(Ksa, where='out')
            uvlm.ss.addGain(Kas, where='in')

            # Stiffenning and damping terms within the uvlm
            Dmod = np.zeros_like(uvlm.ss.D)
            Dmod[:flex_nodes, :flex_nodes] -= self.Kss
            if rigid_dof > 0:
                Dmod[flex_nodes:, :flex_nodes] -= self.Krs
                Dmod[flex_nodes:, total_dof: total_dof + flex_nodes] -= self.Crs
                Dmod[:flex_nodes, total_dof + flex_nodes: 2 * total_dof] -= self.Csr
                Dmod[flex_nodes:, total_dof + flex_nodes: 2 * total_dof] -= self.Crr
                if uvlm.scaled:
                    Dmod /= uvlm.sys.ScalingFacts['force']
                uvlm.ss.D += Dmod

            self.couplings['Ksa'] = Ksa
            self.couplings['Kas'] = Kas

            if self.settings['beam_settings']['modal_projection'] is True and \
                    self.settings['beam_settings']['inout_coords'] == 'modes':
                # Project UVLM onto modal space
                phi = beam.sys.U
                in_mode_matrix = np.zeros((uvlm.ss.inputs, beam.ss.outputs + (uvlm.ss.inputs - 2*beam.sys.num_dof)))
                in_mode_matrix[:2*beam.sys.num_dof, :2*beam.sys.num_modes] = sclalg.block_diag(phi, phi)
                in_mode_matrix[2*beam.sys.num_dof:, 2*beam.sys.num_modes:] = np.eye(uvlm.ss.inputs - 2*beam.sys.num_dof)
                out_mode_matrix = phi.T

                uvlm.ss.addGain(in_mode_matrix, where='in')
                uvlm.ss.addGain(out_mode_matrix, where='out')

            # Reduce uvlm projected onto structural coordinates
            if uvlm.rom:
                if rigid_dof != 0:
                    self.runrom_rbm(uvlm)
                else:
                    for k, rom in uvlm.rom.items():
                        uvlm.ss = rom.run(uvlm.ss)

        else:
            uvlm.ss = self.load_uvlm(self.settings['uvlm_filename'])

        # Coupling matrices
        Tas = np.eye(uvlm.ss.inputs, beam.ss.outputs)
        Tsa = np.eye(beam.ss.inputs, uvlm.ss.outputs)

        # Scale coupling matrices
        if uvlm.scaled:
            Tsa *= uvlm.sys.ScalingFacts['force'] * uvlm.sys.ScalingFacts['time'] ** 2
            if rigid_dof > 0:
                Tas[:flex_nodes + 6, :flex_nodes + 6] /= uvlm.sys.ScalingFacts['length']
                Tas[total_dof: total_dof + flex_nodes + 6] /= uvlm.sys.ScalingFacts['length']
            else:
                if not self.settings['beam_settings']['modal_projection']:
                    Tas /= uvlm.sys.ScalingFacts['length']

        ss = libss.couple(ss01=uvlm.ss, ss02=beam.ss, K12=Tas, K21=Tsa)

        self.couplings['Tas'] = Tas
        self.couplings['Tsa'] = Tsa
        self.state_variables = {'aero': uvlm.ss.states,
                                'beam': beam.ss.states}

        # Save zero force reference
        self.linearisation_vectors['forces_aero_beam_dof'] = Ksa.dot(self.linearisation_vectors['forces_aero'])
        if self.settings['beam_settings']['modal_projection']:
            self.linearisation_vectors['mode_shapes'] = beam.sys.U

        if self.settings['beam_settings']['modal_projection'] is True and \
                self.settings['beam_settings']['inout_coords'] == 'modes':
            self.linearisation_vectors['forces_aero_beam_dof'] = out_mode_matrix.dot(self.linearisation_vectors['forces_aero_beam_dof'])

        cout.cout_wrap('Aeroelastic system assembled:')
        cout.cout_wrap('\tAerodynamic states: %g' % uvlm.ss.states, 1)
        cout.cout_wrap('\tStructural states: %g' % beam.ss.states, 1)
        cout.cout_wrap('\tTotal states: %g' % ss.states, 1)
        cout.cout_wrap('\tInputs: %g' % ss.inputs, 1)
        cout.cout_wrap('\tOutputs: %g' % ss.outputs, 1)

        self.ss = ss
        return self.ss

    def update(self, u_infty):
        """
        Updates the aeroelastic scaled system with the new reference velocity.

        Only the beam equations need updating since the only dependency in the forward flight velocity resides there.

        Args:
              u_infty (float): New reference velocity

        Returns:
            sharpy.linear.src.libss.ss: Updated aeroelastic state-space system

        """
        t_ref = self.uvlm.sys.ScalingFacts['length'] / u_infty

        self.beam.sys.update_matrices_time_scale(t_ref)
        self.beam.sys.assemble()
        self.beam.ss = self.beam.sys.SSdisc

        self.ss = libss.couple(ss01=self.uvlm.ss, ss02=self.beam.ss,
                               K12=self.couplings['Tas'], K21=self.couplings['Tsa'])

        return self.ss

    def runrom_rbm(self, uvlm):
        ss = uvlm.ss
        rig_nodes = self.beam.sys.num_dof_rig
        if rig_nodes == 9:
            orient_dof = 3
        else:
            orient_dof = 4
        # Input side
        if self.settings['beam_settings']['modal_projection'] == True and \
                self.settings['beam_settings']['inout_coords'] == 'modes':
            rem_int_modes = np.zeros((ss.inputs, ss.inputs - rig_nodes))
            rem_int_modes[rig_nodes:, :] = np.eye(ss.inputs - rig_nodes)

            # Output side - remove quaternion equations output
            rem_quat_out = np.zeros((ss.outputs-orient_dof, ss.outputs))
            # find quaternion indices
            U = self.beam.sys.U
            indices = np.where(U[-orient_dof:, :] == 1.)[1]
            j = 0
            for i in range(ss.outputs):
                if i in indices:
                    continue
                rem_quat_out[j, i] = 1
                j += 1

        else:
            rem_int_modes = np.zeros((ss.inputs, ss.inputs - rig_nodes))
            rem_int_modes[:self.beam.sys.num_dof_flex, :self.beam.sys.num_dof_flex] = \
                np.eye(self.beam.sys.num_dof_flex)
            rem_int_modes[self.beam.sys.num_dof_flex+rig_nodes:, self.beam.sys.num_dof_flex:] = \
                np.eye(ss.inputs - self.beam.sys.num_dof_flex - rig_nodes)

            rem_quat_out = np.zeros((ss.outputs-orient_dof, ss.outputs))
            rem_quat_out[:, :-orient_dof] = np.eye(ss.outputs-orient_dof)

        ss.addGain(rem_int_modes, where='in')
        ss.addGain(rem_quat_out, where='out')
        for k, rom in uvlm.rom.items():
            uvlm.ss = rom.run(uvlm.ss)

        uvlm.ss.addGain(rem_int_modes.T, where='in')
        uvlm.ss.addGain(rem_quat_out.T, where='out')

    def get_gebm2uvlm_gains(self, data):
        r"""
        Provides:

            - the gain matrices required to connect the linearised GEBM and UVLM
             inputs/outputs

            - the stiffening and damping factors to be added to the linearised
              GEBM equations in order to account for non-zero aerodynamic loads at
              the linearisation point.

        The function produces the gain matrices:

            - ``Kdisp``: gains from GEBM to UVLM grid displacements
            - ``Kvel_disp``: influence of GEBM dofs displacements to UVLM grid
              velocities.
            - ``Kvel_vel``: influence of GEBM dofs displacements to UVLM grid
              displacements.
            - ``Kforces`` (UVLM->GEBM) dimensions are the transpose than the
               Kdisp and Kvel* matrices. Hence, when allocation this term, ``ii``
               and ``jj`` indices will unintuitively refer to columns and rows,
              respectively.


        And the stiffening/damping terms accounting for non-zero aerodynamic
        forces at the linearisation point:

            - ``Kss``: stiffness factor (flexible dof -> flexible dof) accounting
              for non-zero forces at the linearisation point.
            - ``Csr``: damping factor  (rigid dof -> flexible dof)
            - ``Crs``: damping factor (flexible dof -> rigid dof)
            - ``Crr``: damping factor (rigid dof -> rigid dof)


        Stiffening and damping related terms due to the non-zero aerodynamic forces at the linearisation point:

        .. math::
            \mathbf{F}_{A,n} = C^{AG}(\mathbf{\chi})\sum_j \mathbf{f}_{G,j} \rightarrow
            \delta\mathbf{F}_{A,n} = C^{AG}_0 \sum_j \delta\mathbf{f}_{G,j} + \frac{\partial}{\partial\chi}(C^{AG}\sum_j
            \mathbf{f}_{G,j}^0)\delta\chi

        The term multiplied by the variation in the quaternion, :math:`\delta\chi`, couples the forces with the rigid
        body equations and becomes part of :math:`\mathbf{C}_{sr}`.

        Similarly, the linearisation of the moments results in expression that contribute to the stiffness and
        damping matrices.

        .. math::
            \mathbf{M}_{B,n} = \sum_j \tilde{X}_B C^{BA}(\Psi)C^{AG}(\chi)\mathbf{f}_{G,j}

        .. math::
            \delta\mathbf{M}_{B,n} = \sum_j \tilde{X}_B\left(C_0^{BG}\delta\mathbf{f}_{G,j}
            + \frac{\partial}{\partial\Psi}(C^{BA}\delta\mathbf{f}^0_{A,j})\delta\Psi
            + \frac{\partial}{\partial\chi}(C^{BA}_0 C^{AG} \mathbf{f}_{G,j})\delta\chi\right)

        The linearised equations of motion for the geometrically exact beam model take the input term :math:`\delta
        \mathbf{Q}_n = \{\delta\mathbf{F}_{A,n},\, T_0^T\delta\mathbf{M}_{B,n}\}`, which means that the moments
        should be provided as :math:`T^T(\Psi)\mathbf{M}_B` instead of :math:`\mathbf{M}_A = C^{AB}\mathbf{M}_B`,
        where :math:`T(\Psi)` is the tangential operator.

        .. math::
            \delta(T^T\mathbf{M}_B) = T^T_0\delta\mathbf{M}_B
            + \frac{\partial}{\partial\Psi}(T^T\delta\mathbf{M}_B^0)\delta\Psi

        is the linearised expression for the moments, where the first term would correspond to the input terms to the
        beam equations and the second arises due to the non-zero aerodynamic moment at the linearisation point and
        must be subtracted (since it comes from the forces) to form part of :math:`\mathbf{K}_{ss}`. In addition, the
        :math:`\delta\mathbf{M}_B` term depends on both :math:`\delta\Psi` and :math:`\delta\chi`, therefore those
        terms would also contribute to :math:`\mathbf{K}_{ss}` and :math:`\mathbf{C}_{sr}`, respectively.

        The contribution from the total forces and moments will be accounted for in :math:`\mathbf{C}_{rr}` and
        :math:`\mathbf{C}_{rs}`.

        .. math::
            \delta\mathbf{F}_{tot,A} = \sum_n\left(C^{GA}_0 \sum_j \delta\mathbf{f}_{G,j}
            + \frac{\partial}{\partial\chi}(C^{AG}\sum_j
            \mathbf{f}_{G,j}^0)\delta\chi\right)

        Therefore, after running this method, the beam matrices will be updated as:

        >>> K_beam[:flex_dof, :flex_dof] += Kss
        >>> C_beam[:flex_dof, -rigid_dof:] += Csr
        >>> C_beam[-rigid_dof:, :flex_dof] += Crs
        >>> C_beam[-rigid_dof:, -rigid_dof:] += Crr

        Track body option

        The ``track_body`` setting restricts the UVLM grid to linear translation motions and therefore should be used to
        ensure that the forces are computed using the reference linearisation frame.

        The UVLM and beam are linearised about a reference equilibrium condition. The UVLM is defined in the inertial
        reference frame while the beam employs the body attached frame and therefore a projection from one frame onto
        another is required during the coupling process.

        However, the inputs to the UVLM (i.e. the lattice grid coordinates) are obtained from the beam deformation which
        is expressed in A frame and therefore the grid coordinates need to be projected onto the inertial frame ``G``.
        As the beam rotates, the projection onto the ``G`` frame of the lattice grid coordinates will result in a grid
        that is not coincident with that at the linearisation reference and therefore the grid coordinates must be
        projected onto the original frame, which will be referred to as ``U``. The transformation between the inertial
        frame ``G`` and the ``U`` frame is a function of the rotation of the ``A`` frame and the original position:

        .. math:: C^{UG}(\chi) = C^{GA}(\chi_0)C^{AG}(\chi)

        Therefore, the grid coordinates obtained in ``A`` frame and projected onto the ``G`` frame can be transformed
        to the ``U`` frame using

        .. math:: \zeta_U = C^{UG}(\chi) \zeta_G

        which allows the grid lattice coordinates to be projected onto the original linearisation frame.

        In a similar fashion, the output lattice vertex forces of the UVLM are defined in the original linearisation
        frame ``U`` and need to be transformed onto the inertial frame ``G`` prior to projecting them onto the ``A``
        frame to use them as the input forces to the beam system.

        .. math:: \boldsymbol{f}_G = C^{GU}(\chi)\boldsymbol{f}_U

        The linearisation of the above relations lead to the following expressions that have to be added to the
        coupling matrices:

            * ``Kdisp_vel`` terms:

                .. math::
                    \delta\boldsymbol{\zeta}_U= C^{GA}_0 \frac{\partial}{\partial \boldsymbol{\chi}}
                    \left(C^{AG}\boldsymbol{\zeta}_{G,0}\right)\delta\boldsymbol{\chi} + \delta\boldsymbol{\zeta}_G

            * ``Kvel_vel`` terms:

                .. math::
                    \delta\dot{\boldsymbol{\zeta}}_U= C^{GA}_0 \frac{\partial}{\partial \boldsymbol{\chi}}
                    \left(C^{AG}\dot{\boldsymbol{\zeta}}_{G,0}\right)\delta\boldsymbol{\chi}
                    + \delta\dot{\boldsymbol{\zeta}}_G

        The transformation of the forces and moments introduces terms that are functions of the orientation and
        are included as stiffening and damping terms in the beam's matrices:

            * ``Csr`` damping terms relating to translation forces:

                .. math::
                    C_{sr}^{tra} -= \frac{\partial}{\partial\boldsymbol{\chi}}
                    \left(C^{GA} C^{AG}_0 \boldsymbol{f}_{G,0}\right)\delta\boldsymbol{\chi}

            * ``Csr`` damping terms related to moments:

                .. math::
                    C_{sr}^{rot} -= T^\top\widetilde{\mathbf{X}}_B C^{BG}
                    \frac{\partial}{\partial\boldsymbol{\chi}}
                    \left(C^{GA} C^{AG}_0 \boldsymbol{f}_{G,0}\right)\delta\boldsymbol{\chi}


        The ``track_body`` setting.

        When ``track_body`` is enabled, the UVLM grid is no longer coincident with the inertial reference frame
        throughout the simulation but rather it is able to rotate as the ``A`` frame rotates. This is to simulate a free
        flying vehicle, where, for instance, the orientation does not affect the aerodynamics. The UVLM defined in this
        frame of reference, named ``U``, satisfies the following convention:

            * The ``U`` frame is coincident with the ``G`` frame at the time of linearisation.

            * The ``U`` frame rotates as the ``A`` frame rotates.

        Transformations related to the ``U`` frame of reference:

            * The angle between the ``U`` frame and the ``A`` frame is always constant and equal
              to :math:`\boldsymbol{\Theta}_0`.

            * The angle between the ``A`` frame and the ``G`` frame is :math:`\boldsymbol{\Theta}=\boldsymbol{\Theta}_0
              + \delta\boldsymbol{\Theta}`

            * The projection of a vector expressed in the ``G`` frame onto the ``U`` frame is expressed by:

                .. math:: \boldsymbol{v}^U = C^{GA}_0 C^{AG} \boldsymbol{v}^G

            * The reverse, a projection of a vector expressed in the ``U`` frame onto the ``G`` frame, is expressed by

                .. math:: \boldsymbol{v}^U = C^{GA} C^{AG}_0 \boldsymbol{v}^U

        The effect this has on the aeroelastic coupling between the UVLM and the structural dynamics is that the
        orientation and change of orientation of the vehicle has no effect on the aerodynamics. The aerodynamics are
        solely affected by the contribution of the 6-rigid body velocities (as well as the flexible DOFs velocities).

        """

        aero = data.aero
        structure = data.structure
        tsaero = self.uvlm.tsaero0
        tsstr = self.beam.tsstruct0

        Kzeta = self.uvlm.sys.Kzeta
        num_dof_str = self.beam.sys.num_dof_str
        num_dof_rig = self.beam.sys.num_dof_rig
        num_dof_flex = self.beam.sys.num_dof_flex
        use_euler = self.beam.sys.use_euler

        # allocate output
        Kdisp = np.zeros((3 * Kzeta, num_dof_str))
        Kdisp_vel = np.zeros((3 * Kzeta, num_dof_str))  # Orientation is in velocity DOFs
        Kvel_disp = np.zeros((3 * Kzeta, num_dof_str))
        Kvel_vel = np.zeros((3 * Kzeta, num_dof_str))
        Kforces = np.zeros((num_dof_str, 3 * Kzeta))

        Kss = np.zeros((num_dof_flex, num_dof_flex))
        Csr = np.zeros((num_dof_flex, num_dof_rig))
        Crs = np.zeros((num_dof_rig, num_dof_flex))
        Crr = np.zeros((num_dof_rig, num_dof_rig))
        Krs = np.zeros((num_dof_rig, num_dof_flex))

        # get projection matrix A->G
        # (and other quantities indep. from nodal position)
        Cga = algebra.quat2rotation(tsstr.quat)  # NG 6-8-19 removing .T
        Cag = Cga.T

        # for_pos=tsstr.for_pos
        for_vel = tsstr.for_vel[:3]
        for_rot = tsstr.for_vel[3:]
        skew_for_rot = algebra.skew(for_rot)
        Der_vel_Ra = np.dot(Cga, skew_for_rot)

        Faero = np.zeros(3)
        FaeroA = np.zeros(3)

        # GEBM degrees of freedom
        jj_for_tra = range(num_dof_str - num_dof_rig,
                           num_dof_str - num_dof_rig + 3)
        jj_for_rot = range(num_dof_str - num_dof_rig + 3,
                           num_dof_str - num_dof_rig + 6)

        if use_euler:
            jj_euler = range(num_dof_str - 3, num_dof_str)
            euler = algebra.quat2euler(tsstr.quat)
            tsstr.euler = euler
        else:
            jj_quat = range(num_dof_str - 4, num_dof_str)

        jj = 0  # nodal dof index
        for node_glob in range(structure.num_node):

            ### detect bc at node (and no. of dofs)
            bc_here = structure.boundary_conditions[node_glob]

            if bc_here == 1:  # clamp (only rigid-body)
                dofs_here = 0
                jj_tra, jj_rot = [], []
            # continue

            elif bc_here == -1 or bc_here == 0:  # (rigid+flex body)
                dofs_here = 6
                jj_tra = 6 * structure.vdof[node_glob] + np.array([0, 1, 2], dtype=int)
                jj_rot = 6 * structure.vdof[node_glob] + np.array([3, 4, 5], dtype=int)
            else:
                raise NameError('Invalid boundary condition (%d) at node %d!' \
                                % (bc_here, node_glob))

            jj += dofs_here

            # retrieve element and local index
            ee, node_loc = structure.node_master_elem[node_glob, :]

            # get position, crv and rotation matrix
            Ra = tsstr.pos[node_glob, :]  # in A FoR, w.r.t. origin A-G
            Rg = np.dot(Cag.T, Ra)  # in G FoR, w.r.t. origin A-G
            psi = tsstr.psi[ee, node_loc, :]
            psi_dot = tsstr.psi_dot[ee, node_loc, :]
            Cab = algebra.crv2rotation(psi)
            Cba = Cab.T
            Cbg = np.dot(Cab.T, Cag)
            Tan = algebra.crv2tan(psi)

            track_body = self.settings['track_body']

            ### str -> aero mapping
            # some nodes may be linked to multiple surfaces...
            for str2aero_here in aero.struct2aero_mapping[node_glob]:

                # detect surface/span-wise coordinate (ss,nn)
                nn, ss = str2aero_here['i_n'], str2aero_here['i_surf']
                # print('%.2d,%.2d'%(nn,ss))

                # surface panelling
                M = aero.aero_dimensions[ss][0]
                N = aero.aero_dimensions[ss][1]

                Kzeta_start = 3 * sum(self.uvlm.sys.MS.KKzeta[:ss])
                shape_zeta = (3, M + 1, N + 1)

                for mm in range(M + 1):
                    # get bound vertex index
                    ii_vert = [Kzeta_start + np.ravel_multi_index(
                        (cc, mm, nn), shape_zeta) for cc in range(3)]

                    # get position vectors
                    zetag = tsaero.zeta[ss][:, mm, nn]  # in G FoR, w.r.t. origin A-G
                    zetaa = np.dot(Cag, zetag)  # in A FoR, w.r.t. origin A-G
                    Xg = zetag - Rg  # in G FoR, w.r.t. origin B
                    Xb = np.dot(Cbg, Xg)  # in B FoR, w.r.t. origin B

                    # get rotation terms
                    Xbskew = algebra.skew(Xb)
                    XbskewTan = np.dot(Xbskew, Tan)

                    # get velocity terms
                    zetag_dot = tsaero.zeta_dot[ss][:, mm, nn] - Cga.dot(for_vel)  # in G FoR, w.r.t. origin A-G
                    zetaa_dot = np.dot(Cag, zetag_dot)  # in A FoR, w.r.t. origin A-G

                    # get aero force
                    faero = tsaero.forces[ss][:3, mm, nn]
                    Faero += faero
                    faero_a = np.dot(Cag, faero)
                    FaeroA += faero_a
                    maero_g = np.cross(Xg, faero)
                    maero_b = np.dot(Cbg, maero_g)

                    ### ---------------------------------------- allocate Kdisp

                    if bc_here != 1:
                        # wrt pos - Eq 25 second term
                        Kdisp[np.ix_(ii_vert, jj_tra)] += Cga

                        # wrt psi - Eq 26
                        Kdisp[np.ix_(ii_vert, jj_rot)] -= np.dot(Cbg.T, XbskewTan)

                    # w.r.t. position of FoR A (w.r.t. origin G)
                    # null as A and G have always same origin in SHARPy

                    # # ### w.r.t. quaternion (attitude changes)
                    if use_euler:
                        Kdisp_vel[np.ix_(ii_vert, jj_euler)] += \
                            algebra.der_Ceuler_by_v(tsstr.euler, zetaa)

                        # Track body - project inputs as for A not moving
                        if track_body:
                            Kdisp_vel[np.ix_(ii_vert, jj_euler)] += \
                                Cga.dot(algebra.der_Peuler_by_v(tsstr.euler, zetag))
                    else:
                        # Equation 25
                        # Kdisp[np.ix_(ii_vert, jj_quat)] += \
                        #     algebra.der_Cquat_by_v(tsstr.quat, zetaa)
                        Kdisp_vel[np.ix_(ii_vert, jj_quat)] += \
                            algebra.der_Cquat_by_v(tsstr.quat, zetaa)

                        # Track body - project inputs as for A not moving
                        if track_body:
                            Kdisp_vel[np.ix_(ii_vert, jj_quat)] += \
                                Cga.dot(algebra.der_CquatT_by_v(tsstr.quat, zetag))

                    ### ------------------------------------ allocate Kvel_disp

                    if bc_here != 1:
                        # # wrt pos
                        Kvel_disp[np.ix_(ii_vert, jj_tra)] += Der_vel_Ra

                        # wrt psi (at zero psi_dot)
                        Kvel_disp[np.ix_(ii_vert, jj_rot)] -= \
                            np.dot(Cga,
                                   np.dot(skew_for_rot,
                                          np.dot(Cab, XbskewTan)))

                        # # wrt psi (psi_dot contributions - verified)
                        Kvel_disp[np.ix_(ii_vert, jj_rot)] += np.dot(Cbg.T, np.dot(
                            algebra.skew(np.dot(XbskewTan, psi_dot)), Tan))

                        if np.linalg.norm(psi) >= 1e-6:
                            Kvel_disp[np.ix_(ii_vert, jj_rot)] -= \
                                np.dot(Cbg.T,
                                       np.dot(Xbskew,
                                              algebra.der_Tan_by_xv(psi, psi_dot)))

                    # # w.r.t. position of FoR A (w.r.t. origin G)
                    # # null as A and G have always same origin in SHARPy

                    # # ### w.r.t. quaternion (attitude changes) - Eq 30
                    if use_euler:
                        Kvel_vel[np.ix_(ii_vert, jj_euler)] += \
                            algebra.der_Ceuler_by_v(tsstr.euler, zetaa_dot)

                        # Track body if ForA is rotating
                        if track_body:
                            Kvel_vel[np.ix_(ii_vert, jj_euler)] += \
                                Cga.dot(algebra.der_Peuler_by_v(tsstr.euler, zetag_dot))
                    else:
                        Kvel_vel[np.ix_(ii_vert, jj_quat)] += \
                            algebra.der_Cquat_by_v(tsstr.quat, zetaa_dot)

                        # Track body if ForA is rotating
                        if track_body:
                            Kvel_vel[np.ix_(ii_vert, jj_quat)] += \
                                Cga.dot(algebra.der_CquatT_by_v(tsstr.quat, zetag_dot))

                    ### ------------------------------------- allocate Kvel_vel

                    if bc_here != 1:
                        # wrt pos_dot
                        Kvel_vel[np.ix_(ii_vert, jj_tra)] += Cga

                        # # wrt crv_dot
                        Kvel_vel[np.ix_(ii_vert, jj_rot)] -= np.dot(Cbg.T, XbskewTan)

                    # # wrt velocity of FoR A
                    Kvel_vel[np.ix_(ii_vert, jj_for_tra)] += Cga
                    Kvel_vel[np.ix_(ii_vert, jj_for_rot)] -= \
                        np.dot(Cga, algebra.skew(zetaa))

                    # wrt rate of change of quaternion: not implemented!

                    ### -------------------------------------- allocate Kforces

                    if bc_here != 1:
                        # nodal forces
                        Kforces[np.ix_(jj_tra, ii_vert)] += Cag

                        # nodal moments
                        Kforces[np.ix_(jj_rot, ii_vert)] += \
                            np.dot(Tan.T, np.dot(Cbg, algebra.skew(Xg)))
                    # or, equivalently, np.dot( algebra.skew(Xb),Cbg)

                    # total forces
                    Kforces[np.ix_(jj_for_tra, ii_vert)] += Cag

                    # total moments
                    Kforces[np.ix_(jj_for_rot, ii_vert)] += \
                        np.dot(Cag, algebra.skew(zetag))

                    # quaternion equation
                    # null, as not dep. on external forces

                    ### --------------------------------------- allocate Kstiff

                    ### flexible dof equations (Kss and Csr)
                    if bc_here != 1:
                        # nodal forces
                        if use_euler:
                            if not track_body:
                                Csr[jj_tra, -3:] -= algebra.der_Peuler_by_v(tsstr.euler, faero)
                                # Csr[jj_tra, -3:] -= algebra.der_Ceuler_by_v(tsstr.euler, Cga.T.dot(faero))

                        else:
                            if not track_body:
                                Csr[jj_tra, -4:] -= algebra.der_CquatT_by_v(tsstr.quat, faero)

                            # Track body
                            # if track_body:
                            #     Csr[jj_tra, -4:] -= algebra.der_Cquat_by_v(tsstr.quat, Cga.T.dot(faero))

                        ### moments
                        TanTXbskew = np.dot(Tan.T, Xbskew)
                        # contrib. of TanT (dpsi) - Eq 37 - Integration of UVLM and GEBM
                        Kss[np.ix_(jj_rot, jj_rot)] -= algebra.der_TanT_by_xv(psi, maero_b)
                        # contrib of delta aero moment (dpsi) - Eq 36
                        Kss[np.ix_(jj_rot, jj_rot)] -= \
                            np.dot(TanTXbskew, algebra.der_CcrvT_by_v(psi, np.dot(Cag, faero)))
                        # contribution of delta aero moment (dquat)
                        if use_euler:
                            if not track_body:
                                Csr[jj_rot, -3:] -= \
                                    np.dot(TanTXbskew,
                                           np.dot(Cba,
                                                  algebra.der_Peuler_by_v(tsstr.euler, faero)))

                            # if track_body:
                            #     Csr[jj_rot, -3:] -= \
                            #         np.dot(TanTXbskew,
                            #                np.dot(Cbg,
                            #                       algebra.der_Peuler_by_v(tsstr.euler, Cga.T.dot(faero))))
                        else:
                            if not track_body:
                                Csr[jj_rot, -4:] -= \
                                    np.dot(TanTXbskew,
                                           np.dot(Cba,
                                                  algebra.der_CquatT_by_v(tsstr.quat, faero)))

                            # Track body
                            # if track_body:
                            #     Csr[jj_rot, -4:] -= \
                            #         np.dot(TanTXbskew,
                            #                np.dot(Cbg,
                            #                       algebra.der_CquatT_by_v(tsstr.quat, Cga.T.dot(faero))))

                    ### rigid body eqs (Crs and Crr)

                    if bc_here != 1:
                        # Changed Crs to Krs - NG 14/5/19
                        # moments contribution due to delta_Ra (+ sign intentional)
                        Krs[3:6, jj_tra] += algebra.skew(faero_a)
                        # moment contribution due to delta_psi (+ sign intentional)
                        Krs[3:6, jj_rot] += np.dot(algebra.skew(faero_a),
                                                   algebra.der_Ccrv_by_v(psi, Xb))

                    if use_euler:
                        if not track_body:
                            # total force
                            Crr[:3, -3:] -= algebra.der_Peuler_by_v(tsstr.euler, faero)

                            # total moment contribution due to change in euler angles
                            Crr[3:6, -3:] -= algebra.der_Peuler_by_v(tsstr.euler, np.cross(zetag, faero))
                            Crr[3:6, -3:] += np.dot(
                                np.dot(Cag, algebra.skew(faero)),
                                algebra.der_Peuler_by_v(tsstr.euler, np.dot(Cab, Xb)))

                    else:
                        if not track_body:
                            # total force
                            Crr[:3, -4:] -= algebra.der_CquatT_by_v(tsstr.quat, faero)

                            # total moment contribution due to quaternion
                            Crr[3:6, -4:] -= algebra.der_CquatT_by_v(tsstr.quat, np.cross(zetag, faero))
                            Crr[3:6, -4:] += np.dot(
                                np.dot(Cag, algebra.skew(faero)),
                                algebra.der_CquatT_by_v(tsstr.quat, np.dot(Cab, Xb)))

                        # # Track body
                        # if track_body:
                        #     # NG 20/8/19 - is the Cag needed here? Verify
                        #     Crr[:3, -4:] -= Cag.dot(algebra.der_Cquat_by_v(tsstr.quat, Cga.T.dot(faero)))
                        #
                        #     Crr[3:6, -4:] -= Cag.dot(algebra.skew(zetag).dot(algebra.der_Cquat_by_v(tsstr.quat, Cga.T.dot(faero))))
                        #     Crr[3:6, -4:] += Cag.dot(algebra.skew(faero)).dot(algebra.der_Cquat_by_v(tsstr.quat, Cga.T.dot(zetag)))


        # transfer
        self.Kdisp = Kdisp
        self.Kvel_disp = Kvel_disp
        self.Kdisp_vel = Kdisp_vel
        self.Kvel_vel = Kvel_vel
        self.Kforces = Kforces

        # stiffening factors
        self.Kss = Kss
        self.Krs = Krs
        self.Csr = Csr
        self.Crs = Crs
        self.Crr = Crr

    def to_nodal_coordinates(self):

        is_modal = self.beam.sys.modal

        if is_modal:
            phi = self.beam.sys.U

            # these would include control surface deflections, thrust etc.
            non_structural_inputs = self.uvlm.ss.inputs - self.beam.ss.outputs

            input_gain = sclalg.block_diag(phi.T, phi.T, np.eye(non_structural_inputs), phi.T)
            output_gain = sclalg.block_diag(phi, phi, phi)

            self.ss.addGain(input_gain, where='in')
            self.ss.addGain(output_gain, where='out')

    @staticmethod
    def load_uvlm(filename):
        import sharpy.utils.h5utils as h5
        cout.cout_wrap('Loading UVLM state space system projected onto structural DOFs from file')
        read_data = h5.readh5(filename).ss
        # uvlm_ss_read = read_data.linear.linear_system.uvlm.ss
        uvlm_ss_read = read_data
        return libss.ss(uvlm_ss_read.A, uvlm_ss_read.B, uvlm_ss_read.C, uvlm_ss_read.D, dt=uvlm_ss_read.dt)
